"""Decode-time biasing for Parakeet. Every test is named for what it prevents.

THREE LAYERS, DELIBERATELY, because they fail for different reasons.

  compile   the automaton, against a made-up piece inventory. No model, no
            onnxruntime, microseconds. This is where the shape of the thing is
            pinned.
  loop      the real `_decoding` override, driven by a SCRIPTED decoder that
            returns logits this file chose. That is what makes the branches
            nobody's audio reaches — step == 0, ten tokens in one frame, a
            phrase whose interior spans a blank — testable at all, instead of
            being asserted about in a comment.
  model     the real Parakeet, on real corpus audio, skipped when the weights
            are not cached. The only place a claim about a TRANSCRIPT can be
            made honestly.

The measurements quoted throughout were taken on Parakeet TDT 0.6B v3 at int8
on CPU. They are not portable to another model or quantisation, and
boosting.verify_seam is what refuses a shape they were not measured on.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest

from app import boosting

# ── a made-up piece inventory ────────────────────────────────────────────────
#
# Single-character pieces for everything the phrases below are spelled with,
# plus a handful of multi-character pieces so that a phrase has more than one
# segmentation — which is the whole reason this is an automaton over character
# offsets rather than a trie over one chosen token path.
_MULTI = (" An", "th", "rop", "ic", " Cla", "ude", " Co", "de", " the", "ing")
_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-<>"

VOCAB: dict[int, str] = {i: c for i, c in enumerate(_CHARS)}
for piece in _MULTI:
    VOCAB[len(VOCAB)] = piece
# Control tokens, which must never be reachable as an automaton edge.
SPECIALS = {len(VOCAB): "<unk>", len(VOCAB) + 1: "<blk>"}
VOCAB.update(SPECIALS)
BLANK = max(SPECIALS)
PIECE = {piece: token for token, piece in VOCAB.items()}


def completed(live: boosting.Booster) -> bool:
    """Has every live match reached the end of its phrase?

    A completed phrase does not vanish, it sits on its terminal state — which
    has no outgoing edges, so it contributes no bonus and drops out on the next
    token. Checking for an empty live set instead would be checking the wrong
    thing one token too early.
    """
    return bool(live.live) and all(
        not live.automaton.edges[state] for state in live.live)


# ══ compile ═══════════════════════════════════════════════════════════════════


def test_a_phrase_is_matched_by_every_segmentation_not_one() -> None:
    """The failure: picking a tokenisation the model does not use, and missing.

    The sentencepiece model is not shipped, only the piece list with no unigram
    scores, so the model's canonical segmentation of a phrase cannot be
    reproduced. Measured on the real vocabulary: " Anthropic" has 75 distinct
    segmentations, greedy longest-match gives [' An','th','rop','ic'], and the
    model itself emitted [' An','t','rop','ic'] on real audio. A trie built
    from one path would have matched neither reliably.
    """
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])
    live = boosting.Booster(automaton=automaton, start_weight=1.0)

    # The greedy path.
    for piece in (" An", "th", "rop", "ic"):
        assert PIECE[piece] in live.bonuses(), f"{piece} was not a live edge"
        live.advance(PIECE[piece])
    assert completed(live), "the greedy segmentation did not reach the end"

    # The path the model actually took, through the SAME automaton.
    live = boosting.Booster(automaton=automaton, start_weight=1.0)
    for piece in (" An", "t", "h", "rop", "ic"):
        assert PIECE[piece] in live.bonuses(), f"{piece} was not a live edge"
        live.advance(PIECE[piece])
    assert completed(live), "the model's own segmentation did not reach the end"


def test_the_bonus_is_weighted_by_characters_not_by_tokens() -> None:
    """The failure: a flat per-token bonus that cannot break its own tie.

    At offset 3 of " Anthropic" both 'th' and 't' are live edges. A flat bonus
    raises the two equally and changes nothing about which one wins — the
    automaton would be doing work for no effect. Weighting by the characters an
    edge consumes gives 'th' twice as much, and that is what actually turned
    "Antropic" into "Anthropic" on real audio.
    """
    live = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]),
                            weight=3.0, start_weight=3.0)
    live.advance(PIECE[" An"])
    bonuses = live.bonuses()
    assert bonuses[PIECE["th"]] == pytest.approx(6.0)
    assert bonuses[PIECE["t"]] == pytest.approx(3.0)


def test_a_completed_phrase_earns_the_same_total_whichever_route_it_took() -> None:
    """Character weighting is what makes accumulated boost well defined.

    Two segmentations of one phrase must not be worth different amounts, or the
    decoder would be steered towards a tokenisation rather than towards a word.
    """
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])

    def earned(path: tuple[str, ...]) -> float:
        live = boosting.Booster(automaton=automaton, weight=2.0, start_weight=2.0)
        total = 0.0
        for piece in path:
            total += live.bonuses()[PIECE[piece]]
            live.advance(PIECE[piece])
        return total

    assert earned((" An", "th", "rop", "ic")) == pytest.approx(
        earned((" An", "t", "h", "rop", "ic")))
    assert earned((" An", "th", "rop", "ic")) == pytest.approx(2.0 * len(" Anthropic"))


def test_untokenisable_phrase_is_named_not_dropped() -> None:
    """The failure: a caller believing their vocabulary was applied.

    A phrase fails only when it contains a character with no single-character
    piece. Verified against the shipped model: '日本語' fails at '日' and
    'café ☕' at '☕', while 'São Paulo', 'conteúdo', 'ação' and 'naïve' build.
    Silently dropping it is the same silence profiles.UnknownProfile exists to
    prevent, so it comes back named — with the offending character, because
    "somewhere in this phrase" is not actionable.
    """
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic", "café ☕"])
    assert automaton.phrases == ("Anthropic",)
    assert automaton.untokenisable == (("café ☕", "é"),)


def test_untokenisability_is_reported_ahead_of_the_length_floor() -> None:
    """The failure: telling someone to lengthen a phrase this model cannot spell.

    '日本語' is three characters AND has no pieces. Reporting the length would
    send the caller off to pad a term that would still never work.
    """
    automaton = boosting.compile_automaton(VOCAB, ["日本語"],
                                           min_chars=4)
    assert automaton.untokenisable == (("日本語", "日"),)
    assert automaton.refused == ()


def test_a_control_token_is_never_an_automaton_edge() -> None:
    """The failure: a bonus landing on blank, which would silence a transcript.

    <blk> sits INSIDE the boosted token slice, so nothing about the array shape
    keeps a bonus off it. What keeps a bonus off it is that specials are
    excluded when the automaton is built.

    The phrase below is what makes this test bite rather than pass by luck.
    Specials are excluded by SURFACE FORM, so on ordinary vocabulary they are
    unreachable anyway — no real term contains "<blk>" — and a test over
    ordinary terms would go on passing with the exclusion deleted. A term that
    literally spells the blank token is the one input that tells the two apart,
    and a glossary is a file a human types: "<blk>" is a strange line to write
    and not an impossible one.

    With the exclusion, a bonus can only suppress blank RELATIVELY, whose
    failure mode is an insertion. There is no path from this design to a silent
    transcript, and this is what holds that open.
    """
    automaton = boosting.compile_automaton(VOCAB, ["say <blk> now", "Anthropic"])
    assert "say <blk> now" in automaton.phrases, "the probe phrase was not built"
    reachable = {token for edges in automaton.edges for token in edges}
    assert reachable.isdisjoint(SPECIALS), "a control token is a boostable edge"

    live = boosting.Booster(automaton=automaton, weight=5.0, start_weight=5.0)
    assert BLANK not in live.bonuses()
    for piece in ("s", "a", "y", " ", "<", "b", "l", "k"):
        live.advance(PIECE[piece])
        assert BLANK not in live.bonuses(), f"blank became boostable after {piece!r}"


def test_two_live_matches_sharing_a_token_do_not_stack_their_bonuses() -> None:
    """The failure: a long boost list quietly biasing harder than a short one.

    A token that continues two phrases is not twice as likely to be right.
    Summing would make the effective weight a function of how many terms happen
    to overlap, which is exactly the wrong dependency for the terms-absent
    axis: adding an irrelevant near-duplicate would turn up the aggression on
    the term that was already there.

    "Coding" and "Codings" share every interior state, so after " Co" there are
    two live states both proposing 'd'.
    """
    automaton = boosting.compile_automaton(VOCAB, ["Coding", "Codings"])
    live = boosting.Booster(automaton=automaton, weight=2.0, start_weight=2.0)
    live.advance(PIECE[" Co"])
    assert len(live.live) == 2, "the two phrases did not both stay live"
    assert live.bonuses()[PIECE["d"]] == pytest.approx(2.0)


def test_a_short_phrase_is_refused_by_name() -> None:
    """The failure: 'US' rewriting 'he told us'.

    A two-character phrase is a start edge that matches constantly. NVIDIA's
    own context-biasing paper says terms under three characters "lead to an
    excessive number of false accepts and, therefore, should not be included",
    and profiles._could_occur_innocently already reasons this way about the
    repair half. Refused with a reason attached, not dropped.
    """
    automaton = boosting.compile_automaton(VOCAB, ["US", "Anthropic"], min_chars=4)
    assert automaton.phrases == ("Anthropic",)
    assert automaton.refused == (("US", "shorter than 4 characters"),)


def test_the_phrase_ceiling_refuses_by_name_rather_than_truncating() -> None:
    """The failure: a pasted term list silently becoming a different one.

    The ceiling is about the COLLATERAL-DAMAGE SURFACE rather than speed —
    every phrase in the list is a phrase that can fire on audio it does not
    occur in — so which phrases were dropped is exactly what the caller needs.
    """
    automaton = boosting.compile_automaton(
        VOCAB, [f"Phrase{n:03d}" for n in range(6)], max_phrases=4)
    assert len(automaton.phrases) == 4
    assert [phrase for phrase, _ in automaton.refused] == ["Phrase004", "Phrase005"]
    assert all("ceiling" in why for _, why in automaton.refused)


def test_two_phrases_sharing_a_token_do_not_stack_their_bonuses() -> None:
    """The failure: a long boost list being quietly more aggressive than a short one.

    A token that continues two phrases is not twice as likely to be right.
    Summing would make the bonus a function of how many terms happen to share a
    prefix, which is precisely the wrong thing for the terms-absent axis.
    """
    one = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]),
                           start_weight=2.0).bonuses()
    two = boosting.Booster(
        automaton=boosting.compile_automaton(VOCAB, ["Anthropic", "Anthropology"]),
        start_weight=2.0).bonuses()
    assert two[PIECE[" An"]] == one[PIECE[" An"]]


def test_boosting_cannot_overtake_a_confident_decision() -> None:
    """The gate, which is the property that makes the whole feature safe.

    A boosted token can overtake the winner ONLY IF it was already within
    `gate` logits of it. That is what stops a bonus converting a confident
    blank frame into a hallucinated term: measured, blank wins by a median of
    5.21 logits and a p90 of 8.43 when it wins.
    """
    live = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]),
                            weight=6.0, start_weight=6.0, gate=6.0)
    logits = np.full(len(VOCAB), -30.0, dtype=np.float32)
    logits[PIECE["x"]] = 0.0        # a confident winner
    # The candidate is 10 logits behind and the bonus is 18 (6.0 x the three
    # characters of " An"), so WITHOUT the gate it would overtake comfortably.
    # That margin is the point of the numbers: a test where the bonus loses on
    # arithmetic alone would pass with the gate deleted.
    logits[PIECE[" An"]] = -10.0
    assert int(live.apply(logits).argmax()) == PIECE["x"], (
        "a bonus overrode a decision the model was not close on")

    # …and the same bonus DOES win once the model was already close.
    logits[PIECE[" An"]] = -2.0
    assert int(live.apply(logits).argmax()) == PIECE[" An"]


def test_boost_weight_above_the_ceiling_does_not_run_away() -> None:
    """The failure: weight 15 turning a transcript into a stream of fragments.

    Measured on the real model: at weight 15 with no gate the decoder emitted
    "Anthropic C C A A The A A C A Gh Clau The C Cl The Ant ..." — it did not
    hang and did not go silent, it destroyed the transcript while looking like
    output. THE WEIGHT CEILING ALONE IS NOT ENOUGH, which is the finding this
    test exists to hold: clamping weight 15 to 6 and leaving the gate open
    still ran away. The gate is clamped too, and that is what bounds the reach.
    """
    live = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]),
                            weight=999.0, start_weight=999.0, gate=999.0)
    assert live.weight == boosting.MAX_WEIGHT
    assert live.start_weight == boosting.MAX_WEIGHT
    assert live.gate == boosting.MAX_WEIGHT

    logits = np.full(len(VOCAB), -30.0, dtype=np.float32)
    logits[BLANK] = 0.0
    logits[PIECE[" An"]] = -30.0
    assert int(live.apply(logits).argmax()) == BLANK, (
        "a bonus reached past the gate and turned a blank frame into a token")


def test_a_negative_weight_is_floored_rather_than_inverting_the_bonus() -> None:
    """The failure: STT_BOOST_WEIGHT=-3 quietly biasing AWAY from the vocabulary."""
    live = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]),
                            weight=-3.0, start_weight=-3.0)
    assert live.weight == 0.0
    assert live.start_weight == 0.0


def test_phrase_start_is_not_boosted_by_default() -> None:
    """The failure: 'dictation' becoming 'Dictation', and an invented 'The'.

    With start_weight at its default of 0 a phrase must be ENTERED on
    acoustics and is only helped to FINISH. Measured on the real model at
    start_weight == weight == 2.0, the same probe additionally recovered
    "Claude Code" and "Theoria" AND inserted a spurious "The" before "Theoria"
    (" The" is a start edge of it) and capitalised "dashboard" and "dictation"
    for no reason. That is the +12% finding reproducing itself inside the
    decoder, where the post-decode glossary cannot undo it.
    """
    assert boosting.START_WEIGHT == 0.0
    live = boosting.Booster(automaton=boosting.compile_automaton(VOCAB, ["Anthropic"]))
    assert live.bonuses() == {}, "a phrase start was boosted with nothing live"

    # Entering on acoustics still works — that is the whole design.
    live.advance(PIECE[" An"])
    assert live.bonuses(), "a phrase entered acoustically was not helped to finish"


# ══ loop ══════════════════════════════════════════════════════════════════════


MAX_TOKENS_PER_STEP = 10


def scripted(script):  # noqa: ANN001, ANN201
    """A real BoostedParakeetTdt whose `_decode` returns logits we chose.

    It has to BE one — not merely quack like one — because the override's
    no-boost branch calls `super()._decoding`, which is the very delegation
    that makes an unboosted decode upstream's code. A duck-typed stand-in
    would pass every other test in this file and fail exactly that one.

    Constructed without ONNX_ASR's __init__, which wants model files. Every
    attribute the loop reads is set here instead, which is also a list of the
    loop's dependencies — if upstream's loop starts reading something else,
    this raises AttributeError rather than quietly testing the wrong thing.

    The point of scripting at all is the branches real audio does not reach.
    Measured over a 6.6 s probe the duration head only ever produced step in
    {1, 2, 3}, but it is five wide — so step 0 and step 4 are possible and
    untested, and step 0 is the one that emits several tokens for one frame.
    """
    pytest.importorskip("onnx_asr")

    class ScriptedTdt(boosting.make_subclass()):  # type: ignore[misc]
        use_low_precision = False
        _vocab_size = len(VOCAB)
        _blank_idx = BLANK
        _max_tokens_per_step = MAX_TOKENS_PER_STEP

        def __init__(self, script) -> None:  # noqa: ANN001
            self.script = list(script)
            self.calls = 0
            self.seen: list[np.ndarray] = []

        def _create_state(self):  # noqa: ANN202
            return ("state",)

        def _decode(self, prev_tokens, prev_state, encoder_out):  # noqa: ANN001, ANN202
            del prev_tokens, prev_state, encoder_out
            wins, step = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            logits = np.full(len(VOCAB), -8.0, dtype=np.float32)
            for token, value in wins.items():
                logits[token] = value
            self.seen.append(logits)
            return logits, step, ("state",)

    return ScriptedTdt(script)


def _run_loop(script, boost=None, frames: int = 40, **kwargs):  # noqa: ANN001, ANN202
    decoder = scripted(script)
    encoder_out = np.zeros((1, frames, 4), dtype=np.float32)
    lens = np.array([frames], dtype=np.int64)
    if boost is not None:
        kwargs["boost"] = boost
    tokens, timestamps, logprobs = next(
        iter(decoder._decoding(encoder_out, lens, **kwargs)))
    return list(tokens), list(timestamps), logprobs, decoder


def test_blank_does_not_kill_a_live_match() -> None:
    """The failure: every phrase whose interior spans a frame, broken.

    In a transducer, blank means "nothing more at this frame", NOT "word
    boundary" — 16 of the probe's 51 decode calls were blank, interleaved
    through the middle of words. A design that killed a live match on blank
    would work on the probe and fail on any phrase the model happens to spread
    over two frames.

    Here " An" is emitted, then a blank frame, then the interior. The interior
    is scored BELOW 'x' and only wins if the match survived the blank.
    """
    script = [
        ({PIECE[" An"]: 0.0}, 1),
        ({BLANK: 0.0}, 1),
        ({PIECE["x"]: 0.0, PIECE["th"]: -3.0}, 1),
        ({BLANK: 0.0}, 1),
    ]
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])
    tokens, _, _, _ = _run_loop(script, boosting.Booster(automaton=automaton,
                                                        weight=3.0))
    assert PIECE["th"] in tokens, "the blank frame killed the live match"

    # Control: with no boost, 'x' wins the third frame outright.
    tokens, _, _, _ = _run_loop(script)
    assert PIECE["x"] in tokens and PIECE["th"] not in tokens


def test_step_zero_emits_multiple_tokens_per_frame() -> None:
    """The failure: the automaton losing its place when the frame does not advance.

    step == 0 means the decoder emits again for the SAME frame. The automaton
    must advance per emitted token and not per frame, or a phrase decoded
    inside one frame falls apart. This is the branch a 6.6 s probe never
    reached, and the reason for scripting the decoder at all.
    """
    script = [
        ({PIECE[" An"]: 0.0}, 0),
        ({PIECE["x"]: 0.0, PIECE["th"]: -3.0}, 0),
        ({PIECE["x"]: 0.0, PIECE["rop"]: -3.0}, 0),
        ({BLANK: 0.0}, 1),
    ]
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])
    tokens, timestamps, _, _ = _run_loop(
        script, boosting.Booster(automaton=automaton, weight=3.0))
    assert tokens[:3] == [PIECE[" An"], PIECE["th"], PIECE["rop"]]
    assert timestamps[:3] == [0, 0, 0], "three emissions were not on one frame"


def test_a_runaway_still_terminates_and_is_bounded_per_frame() -> None:
    """The failure that 'runaway' sounds like but is not: a wedged decoder.

    Worth pinning because the word suggests a hang. It cannot hang: step > 0
    always advances the frame, and when step == 0 the loop is bounded by
    _max_tokens_per_step. So the worst a bonus can do is ten spurious tokens
    per frame — a destroyed transcript, which is bad enough, and a decode that
    finishes.
    """
    script = [({PIECE[" An"]: 0.0}, 0)]
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])
    tokens, _, _, decoder = _run_loop(
        script, boosting.Booster(automaton=automaton, weight=6.0, start_weight=6.0),
        frames=3)
    assert len(tokens) == 3 * MAX_TOKENS_PER_STEP
    assert decoder.calls == len(tokens)


def test_logprobs_report_the_model_not_the_bonus() -> None:
    """The failure: include[]=logprobs quietly reporting our own bonus as confidence.

    `logits` is a slice VIEW of the ONNX Runtime output buffer, so boosting in
    place would corrupt the array log_softmax is taken over. asr._parakeet_words
    averages those into every Word.logprob and every synthesised avg_logprob, so
    the fiction would reach verbose_json and look like the model's own number.
    The boosted array must be a copy, and the logprob must come from the
    original.
    """
    script = [
        ({PIECE["x"]: 0.0, PIECE[" An"]: -3.0}, 1),
        ({BLANK: 0.0}, 1),
    ]
    automaton = boosting.compile_automaton(VOCAB, ["Anthropic"])
    booster = boosting.Booster(automaton=automaton, weight=6.0, start_weight=6.0,
                               gate=6.0)
    tokens, _, logprobs, decoder = _run_loop(script, booster,
                                             need_logprobs="yes")
    assert tokens[0] == PIECE[" An"], "the bonus did not change the decision"

    raw = decoder.seen[0]
    expected = float(np.log(np.exp(raw - raw.max()).sum()) * -1 + raw[PIECE[" An"]]
                     - raw.max())
    assert logprobs[0] == pytest.approx(expected, abs=1e-5)
    # And the buffer we were handed is unmodified, which is the other half.
    assert raw[PIECE[" An"]] == pytest.approx(-3.0)


def test_no_boost_list_runs_upstreams_own_loop() -> None:
    """The failure this whole subclass could have introduced: changing the default.

    With no `boost` keyword the override delegates to onnx-asr's `_decoding`
    rather than running a copy of it, so an unboosted decode is upstream's code
    executing upstream's arithmetic. Structural, not a promise to re-verify
    after every edit — and pinned here because "it currently happens to match"
    and "it cannot differ" are not the same guarantee.
    """
    pytest.importorskip("onnx_asr")
    from onnx_asr.asr import _AsrWithTransducerDecoding  # noqa: PLC0415

    script = [({PIECE["x"]: 0.0, PIECE[" An"]: -0.5}, 1), ({BLANK: 0.0}, 1)]
    frames = 6
    encoder_out = np.zeros((1, frames, 4), dtype=np.float32)
    lens = np.array([frames], dtype=np.int64)

    ours = next(iter(scripted(script)._decoding(
        encoder_out, lens, need_logprobs="yes")))
    stock = next(iter(_AsrWithTransducerDecoding._decoding(
        scripted(script), encoder_out, lens, need_logprobs="yes")))
    assert list(ours[0]) == list(stock[0])
    assert list(ours[1]) == list(stock[1])
    assert list(ours[2]) == list(stock[2])


# ══ model ═════════════════════════════════════════════════════════════════════

MODEL_ID = os.getenv("STT_MODEL_ID", "istupakov/parakeet-tdt-0.6b-v3-onnx")
_CACHE = Path.home() / ".cache/huggingface/hub"
_CACHED = list(_CACHE.glob("models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/*"))

# The corpora the benchmark uses, which are gitignored and fetched separately.
# Absent on a fresh checkout, which is why every test below skips rather than
# failing: these are the only tests here that can make a claim about a real
# transcript, and a claim that cannot be made is not a test that should fail.
_CORPUS = sorted(
    glob.glob(str(Path.home() / "Projects/tts/stt-stack/bench/cache/*/*/0*.wav")))

needs_model = pytest.mark.skipif(
    not _CACHED or pytest.importorskip is None,
    reason="Parakeet ONNX weights are not in the Hugging Face cache")


@pytest.fixture(scope="module")
def parakeet():  # noqa: ANN201
    pytest.importorskip("onnx_asr")
    if not _CACHED:
        pytest.skip("Parakeet ONNX weights are not in the Hugging Face cache")
    adapter = boosting.load(MODEL_ID, os.getenv("STT_QUANTISATION", "int8"))
    boosting.verify_seam(adapter)
    return adapter


def test_the_decoding_seam_is_the_one_this_module_was_written_for(parakeet) -> None:  # noqa: ANN001
    """The failure: a version bump silently reverting the feature.

    THE HEADLINE RISK, and the only one with no symptom. Upstream's `_decoding`
    reads its options with `kwargs.get` and ignores unknown keys, so a renamed
    method or an adapter that stopped forwarding kwargs would send `boost=...`
    down a stock path where it is discarded — no error, no warning, no visible
    difference except worse transcripts. A pin in requirements.txt delays that;
    it does not detect it. verify_seam is the detection, and this is the test
    that stops someone deleting it as ceremony.
    """
    from importlib.metadata import version  # noqa: PLC0415

    assert version("onnx-asr") in boosting.SUPPORTED_ONNX_ASR
    assert type(parakeet.asr).__name__ == "BoostedParakeetTdt"

    probe: list[str] = []
    parakeet.recognize(np.zeros(3200, dtype=np.float32), sample_rate=16_000,
                       _boost_probe=probe)
    assert probe, "a keyword argument never reached the boosted _decoding"


@pytest.mark.skipif(len(_CORPUS) < 4, reason="bench corpora are not fetched here")
def test_an_unboosted_decode_is_byte_identical_to_the_stock_model(parakeet) -> None:  # noqa: ANN001
    """The non-negotiable: adding this feature must not move the default path.

    Proved on real corpus audio rather than by inspection, and on all four
    things the pipeline reads — text, tokens, timestamps and logprobs — because
    only the first of those is visible in a transcript and the other three feed
    verbose_json's word timings and confidences.

    Both shapes are checked: no `boost` keyword at all (which delegates to
    upstream), and an EMPTY automaton (which runs our copy of the loop with
    every bonus empty). The second is the one that proves the copy is faithful
    rather than merely unused.
    """
    onnx_asr = pytest.importorskip("onnx_asr")
    stock = onnx_asr.load_model(MODEL_ID,
                                quantization=os.getenv("STT_QUANTISATION", "int8"))
    empty = boosting.Booster(automaton=boosting.compile_automaton(
        parakeet.asr._vocab, []))

    for path in _CORPUS[:4]:
        assert stock.recognize(path) == parakeet.recognize(path), path
        assert stock.recognize(path) == parakeet.recognize(path, boost=empty), path

        want = stock.with_timestamps().recognize(path)
        got = parakeet.with_timestamps().recognize(path)
        assert (want.text, want.tokens, want.timestamps, want.logprobs) == (
            got.text, got.tokens, got.timestamps, got.logprobs), path


@pytest.mark.skipif(len(_CORPUS) < 1, reason="bench corpora are not fetched here")
def test_an_absent_vocabulary_does_not_change_the_transcript(parakeet) -> None:  # noqa: ANN001
    """The terms-ABSENT axis, which is the one that decides the defaults.

    A glossary whose terms do not occur in the audio raised WER by 12% on this
    engine across 25 cells, so a biasing feature has to be judged on what
    it costs when the vocabulary is irrelevant — not only on what it recovers
    when the vocabulary is right. At the shipped defaults the transcript of
    real corpus audio is unchanged by a list of terms that are not in it.

    This is a REGRESSION GUARD ON THE DEFAULTS, not a WER measurement. It says
    the defaults are inert on these clips; it does not say they are inert in
    general, and bench/bench.py on both axes is still what has to decide that
    before boosting is turned on for a deployment.
    """
    absent = boosting.compile_automaton(parakeet.asr._vocab, [
        "Kubernetes", "PostgreSQL", "Wireguard", "Cloudflare",
        "Terraform", "Prometheus", "Grafana", "Elasticsearch"])
    assert len(absent.phrases) == 8
    booster = boosting.Booster(automaton=absent)

    for path in _CORPUS[:3]:
        assert parakeet.recognize(path) == parakeet.recognize(path, boost=booster), (
            f"an irrelevant boost list changed the transcript of {path}")
