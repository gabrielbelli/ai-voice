#!/usr/bin/env python3
"""Measure what decode-time boosting costs and what it buys, on both axes.

    python boost_bench.py run --cache ../../../../stt-stack/bench/cache
    python boost_bench.py report          # the sweep, one line per config
    python boost_bench.py ci              # paired bootstrap on the deltas

WHAT IT FOUND, so a reader of this file does not have to run it
---------------------------------------------------------------
145 clips, 942 s, four corpora, Parakeet TDT 0.6B v3 ONNX int8 on CPU. Pooled
relative WER against no boosting, 95% paired bootstrap over clips:

    shipped tech+dictation profile, terms absent   byte-identical, 0 false fires
    200 unrelated phrases (MAX_PHRASES exactly)    +0.4%  [-0.8, +1.7]
    the words that ARE in the audio, weight 3      -5.2%  [-9.2, -1.3]
    the same, padded to the 200-phrase ceiling    -5.2%  [-9.5, -1.4]

The absent axis is free -- the thing this was built to find out. The last two
lines being equal is why: at START_WEIGHT=0 a phrase must be entered on
acoustics, so a word that is not in the audio is never entered and costs
nothing. The win is real (its interval excludes zero) and small: FOURTEEN words
out of 2,378, term recall 0.908 -> 0.922. Weight 3.0, chosen before this ran, is
the curve's peak and the largest weight whose interval still excludes zero.
START_WEIGHT is the one setting that ruins things: 1.5 on a realistic list is
+81% WER [+49, +122]. Throughput is unaffected (17.8x plain, 18.0x at 200
phrases, +/-8% spread). See bench/README.md and ADR 0005.

Not measured, and it is the case the feature exists for: there is no jargon
corpus on this machine, so terms-present is a proxy built from rare corpus
words rather than Catallaxy and Theoria in the voice that says them. CORAA
returned a null result at every weight -- 3.4 s clips, 14 of 40 with no word
distinctive enough to boost.

WHY THIS IS A SECOND HARNESS AND NOT A FLAG ON bench.py
-------------------------------------------------------
bench.py measures ONE configuration against a fixed condition matrix and talks
to a running service over HTTP. Boosting is not a configuration, it is a
per-request argument whose effect depends on a term list that has to be built
PER CLIP from that clip's own reference — there is no public corpus containing
Catallaxy or Theoria, so the terms-present axis cannot come from a dataset the
way a noise condition can. It also has to sweep a weight, which means twenty-odd
decodes of the same audio, which is the shape bench.py's HTTP client is worst
at. So this drives `asr.Parakeet` in process, and imports bench.normalise so
both harnesses score identically and the numbers stay comparable to the
25-cell table this project quotes.

It does NOT start a server. An earlier round left four orphaned uvicorn
processes on this laptop, one at 489% CPU for thirteen minutes.

THE TWO AXES, AND WHY THE SECOND ONE DECIDES
--------------------------------------------
ADR 0002 removed an always-on glossary because terms that do NOT occur in the
audio raised WER by 28% on Whisper, and nothing measurable on Parakeet across 25 cells.
Any biasing feature therefore has two numbers, not one:

    present   the clip's own distinctive words are boosted   does WER improve
    absent    only terms that occur nowhere are boosted      what does it cost

`mixed` is the one that matters most in practice, because it is the shape a
saved glossary profile actually has: the right terms are in the list, and so
are ninety wrong ones. A feature that wins on `present` and loses on `mixed`
does not help a real deployment, it helps an oracle.

CASE IS PART OF THE MEASUREMENT, NOT AN ARTEFACT
------------------------------------------------
boosting.compile_automaton matches character-exactly against " " + phrase.
Corpus references are lowercase and unpunctuated; Parakeet emits cased,
punctuated text, so boosting "bairro" cannot fire on a sentence-initial
"Bairro". Every term is therefore entered in both casings, which is what a
careful caller would do and what the phrase counts below already account for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import bench  # noqa: E402  - for normalise(), so both harnesses score alike
import degrade  # noqa: E402

from app import asr, boosting, profiles  # noqa: E402

SAMPLE_RATE = 16_000
RESULTS = HERE / "results" / "boost.jsonl"
# The sweep and the significance run are separate files because they answer
# separate questions: boost.jsonl is the whole curve at aggregate resolution,
# boost-paired.jsonl is the handful of configs that decide, re-decoded with
# per-clip error counts kept so `ci` can put an interval on them. Decoding is
# deterministic, so every WER in the second file reproduces the first exactly
# -- which is also a second, independent check that the sweep is repeatable.
PAIRED = HERE / "results" / "boost-paired.jsonl"

# The four sources this runs over, and why these four. Two locales, and read
# against spontaneous in each, because bench/README.md's own warning applies
# here too: a model can be excellent on read speech and fall apart on
# spontaneous speech at the same WER, and a biasing feature can do the same.
# CORAA carries the most weight — it is unscripted Brazilian speech, the
# closest public corpus to what this deployment is actually used for.
SOURCES = [
    ("pt-BR", "coraa", 40),               # spontaneous, the one that counts
    ("pt-BR", "fleurs-pt", 25),           # read; long clips, so rtf is honest
    ("en-US", "librispeech-clean", 40),   # read, dense in proper nouns
    ("en-US", "earnings22", 40),          # spontaneous, real meeting jargon
]

# Terms that occur in NONE of the audio above: this project's own vocabulary,
# in the shape a real glossary has. This is the headline cost number, and it is
# small on purpose — a user's jargon list is a dozen terms, not two hundred.
ABSENT_TERMS = [
    "Anthropic", "Claude Code", "Catallaxy", "Theoria", "Ghost Pepper",
    "Parakeet", "CTranslate2", "TrueNAS", "FreeBSD", "ZFS Pool",
    "ElevenLabs", "Kokoro",
]

def shipped_profile_terms() -> tuple[str, ...]:
    """The vocabulary the SHIPPED profiles actually send to the decoder.

    ABSENT_TERMS above is a hand-written stand-in for a user's jargon list. This
    is not a stand-in: it is `glossaries/tech.txt` and `glossaries/dictation.txt`
    put through `profiles.Registry.select`, the same call the request path
    makes, so what the automaton sees here is character-for-character what it
    sees when someone sends `-F glossary=tech,dictation -F boost=true`.

    It is here because it is the question an operator actually asks. "What does
    a 200-phrase synthetic list cost" is a bound; "what does selecting the
    profile that ships in the image cost, on audio that is about something
    else" is the decision. Those are 79 terms, not 200, and the answer differs.
    """
    reg = profiles.Registry(HERE.parent / "glossaries")
    reg.reload()
    return reg.select(reg.names).terms


# A word has to be at least this long to be a candidate, matching
# boosting.MIN_PHRASE_CHARS — shorter phrases are refused by the automaton, so
# including them would measure the refusal rather than the boost.
MIN_CHARS = boosting.MIN_PHRASE_CHARS

# Words per clip in the `present` list, and total words in `mixed`/`absent-large`.
# 100 words in two casings is 200 phrases, which is boosting.MAX_PHRASES exactly
# — deliberately at the ceiling, because the ceiling is what bounds the
# collateral-damage surface and an operator who fills it should know the cost.
PRESENT_WORDS = 12
LARGE_WORDS = 100

_WORD = re.compile(r"[^\W\d_]{%d,}" % MIN_CHARS, re.UNICODE)


# ------------------------------------------------------------- term lists ---

def _candidates(refs: list[str], locale: str, common: set[str]) -> list[list[str]]:
    """Per clip, the distinctive words of its own reference.

    Distinctive means low document frequency across this source's own clips AND
    not in the locale's hundred commonest words. Both filters are needed and the
    second one is not fussiness: on a five-word earnings22 clip, df alone left
    "been", "here" and "happy" as the clip's "vocabulary", which would have
    measured the boost's collateral on ordinary English and called it the
    terms-present axis. A caller who knows what they are about to dictate knows
    proper nouns and jargon, not function words — that is the whole premise of
    a vocabulary field, and the list has to have that shape or it measures
    something else.
    """
    tokens = [_WORD.findall(bench.normalise(r, locale)) for r in refs]
    df = Counter(w for t in tokens for w in set(t))
    out = []
    for t in tokens:
        rare = sorted({w for w in t if df[w] <= 2 and w not in common},
                      key=lambda w: (-len(w), w))
        out.append(rare[:PRESENT_WORDS])
    return out


def _cased(words: list[str]) -> tuple[str, ...]:
    """Each word in both casings. See the module docstring: the automaton is
    character-exact, so a lowercase corpus word cannot match a cased emission."""
    out: list[str] = []
    for w in words:
        out.append(w)
        if w.capitalize() != w:
            out.append(w.capitalize())
    return tuple(out)


def build_lists(clips: list[dict], locale: str, filler: list[str],
                common: set[str]) -> None:
    """Attach the four boost lists to every clip, in place."""
    per_clip = _candidates([c["ref"] for c in clips], locale, common)
    for clip, own in zip(clips, per_clip):
        # `mixed`: the clip's own terms plus foreign filler up to LARGE_WORDS.
        # The filler comes from ANOTHER locale's corpus, so it is guaranteed
        # absent from this audio without being adversarially chosen.
        pad = [w for w in filler if w not in own][:LARGE_WORDS - len(own)]
        clip["lists"] = {
            "present": _cased(own),
            "mixed": _cased(own + pad),
            "absent-small": tuple(ABSENT_TERMS),
            "absent-large": _cased(filler[:LARGE_WORDS]),
            # Not _cased(): the shipped profiles are measured exactly as they
            # are sent, casing included. Re-casing them here would measure a
            # list no request ever produces.
            "profile": shipped_profile_terms(),
        }


# ----------------------------------------------------------------- corpus ---

def load_corpus(cache: Path, condition: str) -> dict[str, list[dict]]:
    rng = np.random.default_rng(0)
    corpus: dict[str, list[dict]] = {}
    spec = dict(next(s for name, s in degrade.MATRIX if name == condition))
    for locale, source, limit in SOURCES:
        sdir = cache / locale / source
        man_path = sdir / "manifest.json"
        if not man_path.is_file():
            print(f"  MISSING {sdir}", file=sys.stderr)
            continue
        rows = json.load(open(man_path))["rows"][:limit]
        pool = [sf.read(sdir / r["file"], dtype="float32")[0] for r in rows[:8]]
        clips = []
        for r in rows:
            x, _ = sf.read(sdir / r["file"], dtype="float32")
            if spec:
                x = degrade.apply(x, spec, pool, rng)
            clips.append({"locale": locale, "source": source,
                          "audio": x.astype(np.float32), "ref": r["text"],
                          "seconds": len(x) / SAMPLE_RATE})
        corpus[f"{locale}/{source}"] = clips
    # Filler for `mixed` and `absent-large` comes from the OTHER locale, so a
    # pt-BR clip is padded with English words and vice versa. Absent by
    # construction rather than by inspection.
    counts: dict[str, Counter] = {}
    for clips in corpus.values():
        locale = clips[0]["locale"]
        c = counts.setdefault(locale, Counter())
        for clip in clips:
            c.update(_WORD.findall(bench.normalise(clip["ref"], locale)))
    ranked = {loc: [w for w, _ in c.most_common()] for loc, c in counts.items()}
    for clips in corpus.values():
        locale = clips[0]["locale"]
        other = next(l for l in ranked if l != locale)
        build_lists(clips, locale, ranked[other], set(ranked[locale][:100]))
    return corpus


# -------------------------------------------------------------------- run ---

def _tuned(weight: float, start_weight: float, gate: float):
    """Replace boosting.Booster with one that carries the swept weights.

    asr.Parakeet._booster constructs `boosting.Booster(automaton=...)` by
    attribute lookup at call time, so swapping the module attribute is enough
    and no code in app/ is modified to run this. The alternative — one
    subprocess per weight with STT_BOOST_WEIGHT set, because the dataclass
    defaults bind at import — would reload the model twenty-one times.

    `check_shim` below asserts this reproduces the untouched path exactly at
    the shipped defaults, because a sweep whose weight knob is subtly not the
    real one measures nothing.
    """
    real = boosting.Booster

    # Two callers construct a Booster and BOTH go through this name:
    # asr.Parakeet._booster builds the request's one with defaults only, and
    # boosting._decoding rebuilds a fresh one per utterance in the batch,
    # forwarding the outer booster's weights explicitly. So the swept values
    # are defaults here and an explicit keyword still wins — patching this
    # without forwarding raised TypeError inside the decode loop, which is how
    # the second caller was found.
    def make(*, automaton, weight=weight, start_weight=start_weight, gate=gate):  # noqa: ANN001, ANN202
        return real(automaton=automaton, weight=weight,
                    start_weight=start_weight, gate=gate)
    return make


def check_shim(engine, clips: list[dict]) -> None:
    # `mixed` rather than `present`: it is never empty, and an empty list makes
    # _booster return None, which would make this assertion pass by comparing
    # the unboosted path against itself.
    sample = [c for c in clips if c["lists"]["mixed"]][:3]
    opts_for = lambda c: asr.Options(vocabulary=c["lists"]["mixed"], boost=True)  # noqa: E731
    plain = [engine.transcribe(c["audio"], opts_for(c)).text for c in sample]
    real = boosting.Booster
    try:
        boosting.Booster = _tuned(boosting.WEIGHT, boosting.START_WEIGHT,
                                  boosting.GATE)
        shimmed = [engine.transcribe(c["audio"], opts_for(c)).text for c in sample]
    finally:
        boosting.Booster = real
    assert plain == shimmed, "the weight shim is not the shipped path"
    print("  shim check: identical to the untouched path at shipped defaults")


def _score(clips: list[dict], hyps: list[str]) -> dict:
    from jiwer import cer, process_words, wer

    pairs = [(bench.normalise(c["ref"], c["locale"]), bench.normalise(h, c["locale"]))
             for c, h in zip(clips, hyps) if c["ref"].strip()]
    return {
        "wer": round(wer([r for r, _ in pairs], [h for _, h in pairs]), 4),
        "cer": round(cer([r for r, _ in pairs], [h for _, h in pairs]), 4),
        # Per clip: reference length and edit distance, in the SAME clip order
        # for every config, which is what makes a PAIRED bootstrap possible in
        # report --ci. Without it the only honest thing this harness could say
        # about "-8.5%" is that it is a point estimate, and the effect it is
        # estimating turned out to be fourteen words in two thousand — an
        # effect that size cannot be read off a corpus WER without an interval.
        "per_clip": [_edits(r, h) for r, h in pairs],
    }


def _edits(ref: str, hyp: str) -> list[int]:
    """[reference words, word edit distance] for one clip."""
    from jiwer import process_words

    o = process_words([ref], [hyp])
    return [len(ref.split()), o.substitutions + o.deletions + o.insertions]


def _term_stats(clips: list[dict], hyps: list[str], list_name: str) -> dict:
    """Recall of terms that ARE in the audio, and false fires of terms that are not.

    WER over a whole corpus is a blunt instrument for a feature that touches a
    handful of words per clip: recovering one proper noun in a twenty-word
    utterance moves WER by 0.05 on that clip and by 0.001 overall. These two
    counts are what the feature actually claims to do, so they are measured
    directly rather than inferred from a WER that may not move.
    """
    hit = miss = fired = 0
    for clip, hyp in zip(clips, hyps):
        ref_words = set(_WORD.findall(bench.normalise(clip["ref"], clip["locale"])))
        hyp_words = set(_WORD.findall(bench.normalise(hyp, clip["locale"])))
        for phrase in clip["lists"][list_name]:
            words = _WORD.findall(bench.normalise(phrase, clip["locale"]))
            if not words:
                continue
            present = all(w in ref_words for w in words)
            in_hyp = all(w in hyp_words for w in words)
            if present:
                hit += in_hyp
                miss += not in_hyp
            elif in_hyp:
                fired += 1
    total = hit + miss
    return {"term_recall": round(hit / total, 4) if total else None,
            "terms_in_audio": total, "false_fires": fired}


def run(cache: Path, condition: str, configs: list[dict], out: Path) -> None:
    print(f"loading corpus ({condition}) ...", flush=True)
    corpus = load_corpus(cache, condition)
    if not corpus:
        print("no corpus found; pass --cache", file=sys.stderr)
        raise SystemExit(2)
    for key, clips in corpus.items():
        secs = sum(c["seconds"] for c in clips)
        sizes = {n: len(clips[0]["lists"][n]) for n in clips[0]["lists"]}
        print(f"  {key:26} {len(clips):3d} clips  {secs:6.0f}s  phrases={sizes}")

    engine = asr.Parakeet(os.getenv("STT_MODEL_ID", asr.PARAKEET_DEFAULT),
                          os.getenv("STT_QUANTISATION", "int8"))
    if not engine.accepts_vocabulary:
        print(f"biasing unavailable: {engine.vocabulary_unavailable}", file=sys.stderr)
        raise SystemExit(2)
    check_shim(engine, next(iter(corpus.values())))

    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["config"], r["source"]))

    real = boosting.Booster
    with open(out, "a") as fh:
        for cfg in configs:
            name = cfg["name"]
            for key, clips in corpus.items():
                if (condition, name, key) in done:
                    print(f"  {condition}/{name}/{key}: cached")
                    continue
                if cfg["list"] is None:
                    opts_for = lambda c: asr.Options()  # noqa: E731
                else:
                    opts_for = lambda c: asr.Options(  # noqa: E731
                        vocabulary=c["lists"][cfg["list"]], boost=True)
                boosting.Booster = _tuned(cfg["weight"], cfg["start_weight"],
                                          cfg["gate"])
                hyps, elapsed = [], 0.0
                try:
                    for c in clips:
                        t0 = time.perf_counter()
                        hyps.append(engine.transcribe(c["audio"], opts_for(c)).text)
                        elapsed += time.perf_counter() - t0
                finally:
                    boosting.Booster = real
                secs = sum(c["seconds"] for c in clips)
                row = {"condition": condition, "config": name, "source": key,
                       "list": cfg["list"], "weight": cfg["weight"],
                       "start_weight": cfg["start_weight"], "gate": cfg["gate"],
                       "clips": len(clips),
                       "rtf": round(secs / elapsed, 2),
                       **_score(clips, hyps)}
                # The control is scored against the `present` list too. Term
                # recall is the number this feature actually claims to move,
                # and without an unboosted recall to compare against, "0.922
                # recall at weight 3" is a figure with no denominator — the
                # first pass recorded exactly that and had to be re-run.
                row.update(_term_stats(clips, hyps, cfg["list"] or "present"))
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {condition:12} {name:22} {key:26} "
                      f"WER={row['wer']:.4f} rtf={row['rtf']:.1f} "
                      f"recall={row.get('term_recall')} fires={row.get('false_fires')}",
                      flush=True)


# ----------------------------------------------------------------- config ---

def sweep(condition: str) -> list[dict]:
    """Every configuration this measures, and why each one is here.

    The weight ladder brackets the shipped 3.0 on both sides and ends at
    boosting.MAX_WEIGHT, so the curve covers the whole range an operator can
    reach through STT_BOOST_WEIGHT — above 6.0 the Booster clamps, which is
    itself worth confirming rather than assuming.
    """
    cfgs = [{"name": "plain", "list": None, "weight": 0.0,
             "start_weight": 0.0, "gate": boosting.GATE}]
    lists = ["present", "mixed", "absent-small", "absent-large", "profile"]
    # A degraded condition gets a coarser ladder, not because the curve is less
    # interesting there but because it is the same curve: the gate only lets a
    # bonus through on a near-tie, and noise makes near-ties, so the question a
    # degraded run answers is whether the whole curve shifts — three points
    # settle that, and the fourth hour of decoding would not add to it.
    weights = [1.0, 2.0, 3.0, 4.5, 6.0] if condition == "clean" else [1.0, 3.0, 6.0]
    for name in lists:
        for w in weights:
            cfgs.append({"name": f"{name}@w{w:g}", "list": name, "weight": w,
                         "start_weight": 0.0, "gate": boosting.GATE})
    if condition == "clean":
        # START_WEIGHT is the recall knob and ships at 0, which means a phrase
        # must be ENTERED on acoustics and is only helped to finish. Raising it
        # is the single change most likely to move the present axis, and the
        # one ADR 0005 warns costs the most on the absent axis. Measured rather
        # than argued.
        for name in lists:
            for sw in [1.5, 3.0]:
                cfgs.append({"name": f"{name}@w3+s{sw:g}", "list": name,
                             "weight": 3.0, "start_weight": sw,
                             "gate": boosting.GATE})
    return cfgs


def report(out: Path) -> None:
    if not out.is_file():
        print("no results; run first")
        return
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    base = {(r["condition"], r["source"]): r for r in rows if r["config"] == "plain"}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        print(f"\n=== {condition} ===")
        sources = dict.fromkeys(r["source"] for r in rows if r["condition"] == condition)
        head = f"{'config':24}" + "".join(f"{s.split('/')[1][:12]:>14}" for s in sources)
        print(head + f"{'mean d%':>9}{'recall':>8}{'fires':>7}{'rtf':>7}")
        for cfg in dict.fromkeys(r["config"] for r in rows if r["condition"] == condition):
            got = [r for r in rows if r["condition"] == condition and r["config"] == cfg]
            line, deltas = f"{cfg:24}", []
            for s in sources:
                r = next((x for x in got if x["source"] == s), None)
                if r is None:
                    line += f"{'-':>14}"
                    continue
                b = base.get((condition, s))
                d = (r["wer"] - b["wer"]) / b["wer"] * 100 if b and b["wer"] else 0.0
                deltas.append(d)
                line += f"{r['wer']:>9.4f}{d:>+5.0f}%"
            rec = [x["term_recall"] for x in got if x.get("term_recall") is not None]
            fires = sum(x.get("false_fires", 0) for x in got)
            rtf = np.mean([x["rtf"] for x in got])
            line += f"{np.mean(deltas) if deltas else 0:>+8.1f}%"
            line += f"{np.mean(rec):>8.3f}" if rec else f"{'-':>8}"
            line += f"{fires:>7d}{rtf:>7.1f}"
            print(line)


# ---------------------------------------------------------- significance ---

def ci(out: Path, iters: int = 4000, seed: int = 0) -> None:
    """Paired bootstrap over clips for every config that carries per-clip data.

    WHY THIS EXISTS. The sweep's headline was "present@w3 is -8.5% WER". Turned
    into absolute errors that is FOURTEEN words out of 2,378 across four
    corpora, and the absent-axis cost is one or two. Numbers that small are not
    obviously distinguishable from which clips happened to be in the manifest,
    and this project has already deleted a feature (denoising) on a measured
    result — so the measurement that justifies keeping one has to be able to
    say whether its effect is separable from zero at all.

    Paired, and resampling CLIPS rather than words: both configs decoded the
    same clips in the same order, so resampling an index resamples both sides
    together and the per-clip correlation (a hard clip is hard for both) is
    kept instead of being counted as noise. Unpaired intervals on data this
    correlated are far too wide.

    WER is recomputed corpus-style on each resample -- sum(errors)/sum(words),
    not a mean of per-clip WERs, because that is how every other WER in this
    repository is computed and a bootstrap that scores differently from the
    point estimate is measuring a different quantity.
    """
    if not out.is_file():
        print("no results; run first")
        return
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    rows = [r for r in rows if r.get("per_clip")]
    if not rows:
        print("no per-clip data in these results; re-run to record it")
        return

    rng = np.random.default_rng(seed)
    base = {(r["condition"], r["source"]): r for r in rows if r["config"] == "plain"}

    def delta(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> float:
        """Relative WER change of b against a over the resampled clips."""
        wa = a[idx, 1].sum() / a[idx, 0].sum()
        wb = b[idx, 1].sum() / b[idx, 0].sum()
        return (wb - wa) / wa * 100 if wa else 0.0

    for condition in dict.fromkeys(r["condition"] for r in rows):
        print(f"\n=== {condition}: relative WER change vs plain, 95% paired bootstrap ===")
        print(f"{'config':22}{'source':20}{'plain':>8}{'boosted':>9}"
              f"{'d%':>8}{'95% CI':>18}{'errors':>9}")
        for cfg in dict.fromkeys(r["config"] for r in rows if r["condition"] == condition):
            if cfg == "plain":
                continue
            got = [r for r in rows if r["condition"] == condition and r["config"] == cfg]
            pool_a, pool_b = [], []
            for r in sorted(got, key=lambda x: x["source"]):
                b0 = base.get((condition, r["source"]))
                if b0 is None or len(b0["per_clip"]) != len(r["per_clip"]):
                    continue
                a = np.array(b0["per_clip"], dtype=float)
                b = np.array(r["per_clip"], dtype=float)
                pool_a.append(a)
                pool_b.append(b)
                n = len(a)
                draws = np.array([delta(a, b, rng.integers(0, n, n)) for _ in range(iters)])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                de = int(b[:, 1].sum() - a[:, 1].sum())
                print(f"{cfg:22}{r['source']:20}{b0['wer']:>8.4f}{r['wer']:>9.4f}"
                      f"{delta(a, b, np.arange(n)):>+7.1f}%"
                      f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}{de:>+9d}")
            if pool_a:
                a, b = np.vstack(pool_a), np.vstack(pool_b)
                n = len(a)
                draws = np.array([delta(a, b, rng.integers(0, n, n)) for _ in range(iters)])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                sig = "" if lo <= 0 <= hi else "   <- excludes 0"
                print(f"{cfg:22}{'POOLED (all clips)':20}"
                      f"{a[:, 1].sum() / a[:, 0].sum():>8.4f}"
                      f"{b[:, 1].sum() / b[:, 0].sum():>9.4f}"
                      f"{delta(a, b, np.arange(n)):>+7.1f}%"
                      f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}"
                      f"{int(b[:, 1].sum() - a[:, 1].sum()):>+9d}{sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", type=Path, required=True,
                   help="a bench.py cache/ directory (fetch it with bench.py fetch)")
    r.add_argument("--condition", default="clean",
                   help="a degrade.MATRIX condition; default clean")
    r.add_argument("--out", type=Path, default=RESULTS)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, default=RESULTS)
    c = sub.add_parser("ci", help="paired bootstrap CI on the WER deltas")
    c.add_argument("--out", type=Path, default=PAIRED)
    c.add_argument("--iters", type=int, default=4000)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cache, a.condition, sweep(a.condition), a.out)
    elif a.cmd == "ci":
        ci(a.out, a.iters)
    else:
        report(a.out)
