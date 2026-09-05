"""Decode-time vocabulary biasing for Parakeet: shallow fusion in the TDT loop.

WHAT THIS FILE OVERTURNS
------------------------
Until this module, every docstring in this service said Parakeet's decoder
takes no vocabulary. That was true of the ARGUMENT and false about what is
reachable, and the difference cost this project a feature: `prompt` and
`keywords[]` were a 4xx on the default engine for months, and a profile's
hotword-only lines did nothing at all.

onnx-asr's transducer decoding loop is PLAIN PYTHON with the per-frame joint
logits in hand (`onnx_asr/asr.py`, `_AsrWithTransducerDecoding._decoding`):

    logits, step, state = self._decode(tokens, prev_state, encodings[t])
    token = logits.argmax()              # <-- the hook

`NemoConformerTdt._decode` has ALREADY split the joint's 8198-wide output into
`output[:8193]` (token logits, returned) and `int(output[8193:].argmax())` (the
duration, returned as an int). So a bonus added at this seam is structurally
incapable of perturbing the duration head: by the time we see the logits, how
many frames to skip is a decided integer. That is why the hook is `_decoding`
and not `_decode`.

The technique is shallow fusion against a boosting trie — the same idea as
NVIDIA's CTC Word Spotter (arXiv:2406.07096), which FluidAudio implements for
this exact model in Swift. We do it in the transducer loop instead, which needs
no second CTC encoder and therefore works in every language the TDT model
works in, including the Brazilian Portuguese this deployment is used in.

THE MEASUREMENT THAT SHAPES EVERY DEFAULT BELOW
-----------------------------------------------
Across 25 cells, a glossary whose terms do NOT occur in the audio raised
WER by 28% on Whisper, and nothing measurable on Parakeet. Irrelevant vocabulary is not inert;
it actively costs accuracy. So this feature is judged on two axes — what it
recovers when the terms are there, and what it costs when they are not — and
every default here is chosen against the second one.

That is also why the whole mechanism is OFF unless a request asks for it. See
`ENABLED_BY_DEFAULT` and openai_api's `boost` field. A deployment-wide always-on
boost list is exactly the shape the 12% measurement rules out, and it is the
shape ADR 0002 removed from the repair half for the same reason.

LOGITS ARE RAW, WHICH IS WHY THE BONUS IS ADDITIVE AND WHY THE SCALE IS THIS
---------------------------------------------------------------------------
The joint's ONNX output node is an Add — the final linear bias — with no
LogSoftmax in the graph, and logsumexp over the 8193-wide token slice measures
-1.26 for a zeros encoder frame and -1439 for a random one. Not 0. onnx-asr
applies `log_softmax` itself when it wants log-probabilities. So these logits
are unnormalised and a bonus is simply added to them.

Measured logit statistics over 51 decode calls of a 6.6 s probe, Parakeet TDT
0.6B v3 at int8 on CPU — read this as the calibration table:

    top1                                       mean -4.91, max -0.42
    top1 - top2 gap                            median 3.61, p10 0.75, p90 8.24
    best token minus blank, when a token wins  median 6.97, p10 3.82
    blank minus best token, when blank wins    median 5.21, p90 8.43

A per-character bonus of 2-3 sits inside the near-tie band: it flips decisions
the model was already unsure about. Above roughly 8.4 it starts converting
blank frames into emissions, which is the runaway — at weight 15 with no gate
the probe produced "Anthropic C C A A The A A C A Gh Clau The C Cl The Ant ..."
rather than a transcript. THESE NUMBERS ARE A PROPERTY OF THIS MODEL AT THIS
QUANTISATION. Changing STT_MODEL_ID or STT_QUANTISATION invalidates them, which
is why `verify_seam` reads the joint's output width back and refuses a model
whose shape it does not recognise.

WHY A CHARACTER-OFFSET AUTOMATON RATHER THAN A TOKEN TRIE
---------------------------------------------------------
The sentencepiece model is not shipped — only the piece list, with no unigram
scores — so the model's canonical segmentation of a phrase is not reproducible.
It is genuinely ambiguous: " Anthropic" has 75 distinct segmentations and
" Ghost Pepper" 448, greedy longest-match gives [' An','th','rop','ic'], and the
model itself emitted [' An','t','rop','ic'] on real audio. A trie keyed by one
chosen token path would simply have missed.

So the states here are (phrase, character offset) over `" " + phrase`, and the
edges out of a state are every vocabulary piece that matches the text there.
That collapses every segmentation into O(total phrase length) states with no
enumeration: 10 phrases is 43 states, 200 phrases is 1800.

An edge's bonus is proportional to the CHARACTERS it consumes, and that is
load-bearing rather than cosmetic. At offset 3 of " Anthropic" both 'th' and 't'
are live edges; a flat per-token bonus raises both equally and changes nothing.
The character-weighted bonus gives 'th' twice as much, and that is what actually
turned "Antropic" into "Anthropic" on the probe. It also makes the total boost
for a completed phrase `weight * len(phrase)` whichever route the model took.

WHAT IS DELIBERATELY NOT HERE: SNAPSHOT ROLLBACK
------------------------------------------------
Greedy decoding commits each emission, so a match that dies leaves debris — the
probe produced "Anthropique" at start_weight 2. The ground work proposed a
snapshot/replay claw-back: remember (t, state, len(tokens)) when a match begins
and rewind if it dies. It is implementable and it fired correctly in a probe.

It is not here, and the omission is deliberate rather than an oversight. It is
the most intricate part of the design, its failure mode is an off-by-one in the
parallel truncation of tokens/timestamps/logprobs, and that failure produces a
subtly WRONG transcript rather than a crash. It only earns its keep at
`start_weight > 0`, which is off by default and which no WER measurement
justifies yet. At the shipped default the probe recorded zero match deaths and
zero rollbacks, so it would be dead code guarding a setting nobody should turn
on until the two-axis benchmark has run. `GATE` below is the cheap half of the
same idea and it does ship.

If start_weight is ever raised in anger, this is the thing to build first.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("stt-stack.boosting")

# onnx-asr versions whose `_decoding` seam this module has been read against.
#
# THIS IS THE TRIPWIRE FOR THE FEATURE'S WORST FAILURE MODE, and it is worth
# being blunt about why it exists. Upstream's `_decoding` reads its options with
# `kwargs.get(...)` and IGNORES every key it does not know. So if a future
# onnx-asr renames the method, changes its signature, or stops routing kwargs
# through the adapter, a `boost=[...]` argument would flow down a stock code
# path and be discarded with no error anywhere: biasing would quietly stop
# working and every transcript would look plausible. A version pin in
# requirements.txt does not prevent that, it only delays it until someone bumps
# the pin. Only an assertion catches it, which is what verify_seam is.
SUPPORTED_ONNX_ASR = frozenset({"0.12.0"})

# Off unless a request asks. A deployment can flip the default with
# STT_BOOST=1 and pay the 12% cost knowingly, exactly as STT_GLOSSARY_DEFAULT
# lets it opt back into an always-on repair glossary.
ENABLED_BY_DEFAULT = os.getenv("STT_BOOST", "0") not in {"0", "false", "no", ""}


def _number(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


# The hard ceiling on any weight, and the reason there is one: at weight 15 the
# decoder did not hang and did not go silent, it emitted a stream of phrase
# prefixes. Termination is guaranteed — `step > 0` always advances the frame and
# `step == 0` is bounded by `_max_tokens_per_step` — so the runaway is a
# destroyed transcript rather than a wedged process, which is arguably worse
# because it looks like output. Configurable because the scale is a property of
# the model and quantisation, not of this code.
MAX_WEIGHT = _number("STT_BOOST_MAX_WEIGHT", 6.0)

# Bonus per character for a token that CONTINUES a live match. 2.0-5.0 all
# recovered "Antropic" -> "Anthropic" on the probe with every other word
# byte-identical to the unboosted transcript.
WEIGHT = _number("STT_BOOST_WEIGHT", 3.0)

# Bonus per character for a phrase's FIRST token. This is the recall knob and
# the dangerous one, and it defaults to zero.
#
# At 0 a phrase must be ENTERED on acoustics and is only helped to FINISH,
# which is precisely the failure this user has: the model heard " An" correctly
# and then lost the interior. Measured collateral at 0: none.
#
# At start_weight == weight == 2.0 the same probe additionally recovered
# "Claude Code" and "Theoria" — and inserted a spurious "The" before "Theoria"
# (" The" is a start edge of "Theoria"), and capitalised "dashboard" and
# "dictation" for no reason. That is the 25-cell finding reproducing
# itself inside the decoder, where the post-decode glossary cannot undo it.
START_WEIGHT = _number("STT_BOOST_START_WEIGHT", 0.0)

# Apply the bonus only where `top1 - logits[token] <= GATE`, so boosting can
# only break a decision the model was already close on and can never override a
# confident one. 6.0 sits between the p10 (0.75) and p90 (8.24) of the measured
# top1-top2 gap. Removing this gate is what let weight 15 run away.
GATE = _number("STT_BOOST_GATE", 6.0)

# A ceiling in the spirit of profiles.MAX_ENTRIES, and for the same kind of
# reason rather than for speed: 200 phrases is 1800 automaton states and cost a
# measured +1.1% wall clock on a 6.6 s clip. What it bounds is the
# COLLATERAL-DAMAGE SURFACE — every phrase in the list is a phrase that can fire
# on audio it does not occur in.
MAX_PHRASES = int(_number("STT_BOOST_MAX_PHRASES", 200))

# Refuse to boost very short phrases. A two-character phrase is a start edge
# that matches constantly, and short terms are the shape most likely to
# reproduce the "dictation" -> "Dictation" collateral; NVIDIA's own paper says
# terms under three characters "lead to an excessive number of false accepts
# and, therefore, should not be included in the context-biasing list".
# Mirrors the reasoning already in profiles._could_occur_innocently.
MIN_PHRASE_CHARS = int(_number("STT_BOOST_MIN_PHRASE_CHARS", 4))

# The longest piece in this model's vocabulary is 21 characters. Read back from
# the real vocabulary at build time; this is only the cap on the substring
# probe when a caller hands us a vocabulary we have not measured.
_MAX_PIECE = 32

# Control tokens: <unk>, <pad>, <blk>, <|nospeech|>, <|startoftranscript|> and
# friends. Excluded from the automaton at build time, which is what guarantees
# a bonus can never land on blank — see `bonuses` for why that matters.
_SPECIAL = re.compile(r"^<.*>$")


class UnsupportedSeam(RuntimeError):
    """The onnx-asr decoding seam is not the one this module was written for.

    Raised by verify_seam and turned into a disabled capability flag rather than
    a crash, because a deployment that never asks for boosting should not be
    taken down by it. What must never happen is the third option: accepting
    `boost` and silently not boosting.
    """


@dataclass(frozen=True)
class Automaton:
    """Phrases compiled against the model's own vocabulary.

    States are (phrase, character offset) over `" " + phrase`, flattened to
    integer indices. `edges[s]` maps a token id to (next state, characters
    consumed). A state whose offset is the end of its phrase has no outgoing
    edges, so a completed match simply drops out of the live set.
    """

    edges: tuple[dict[int, tuple[int, int]], ...]
    roots: tuple[int, ...]
    # Merged over every root: token id -> the most characters any phrase's first
    # token consumes. Precomputed because it is the same on every frame.
    start_edges: dict[int, int]
    phrases: tuple[str, ...]
    # (phrase, the character that has no piece). Reported, never dropped.
    untokenisable: tuple[tuple[str, str], ...] = ()
    # (phrase, why) for phrases refused by a ceiling rather than by the vocabulary.
    refused: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.roots)

    @property
    def states(self) -> int:
        return len(self.edges)


def compile_automaton(
    vocab: Mapping[int, str],
    phrases: Sequence[str],
    *,
    min_chars: int = MIN_PHRASE_CHARS,
    max_phrases: int = MAX_PHRASES,
) -> Automaton:
    """Compile boost phrases against the model's own piece inventory.

    `vocab` is onnx-asr's {id: piece} map, which already has sentencepiece's
    U+2581 rewritten to a leading space — so a word-initial piece is literally
    a piece that starts with " ", which is why every phrase is matched as
    `" " + phrase`. There is no `sp.encode()` to call: onnx-asr ships no
    tokeniser, only the piece list, and the whole reason this is an automaton
    over character offsets rather than a token trie is that the model's own
    segmentation cannot be reproduced from what is shipped.

    A PHRASE THAT CANNOT BE TOKENISED IS RETURNED, NEVER DROPPED. A phrase
    fails only when it contains a character with no single-character piece —
    verified: "日本語" fails at '日' and "café ☕" at '☕', while "São Paulo",
    "conteúdo", "ação", "naïve", "ZFS" and "FreeBSD" all build cleanly. The
    caller turns that into a 400 naming the phrase and the character, for the
    same reason profiles.UnknownProfile exists: a caller who believes their
    vocabulary was applied when it was not is the silence the /v1 surface is
    built to prevent.
    """
    pieces: dict[str, int] = {}
    longest = 1
    for token_id, piece in vocab.items():
        if not piece or _SPECIAL.match(piece):
            continue
        # A duplicate surface form would make the choice of id arbitrary. This
        # model has none (verified: 8193 entries, zero duplicates); keeping the
        # first is a defined answer rather than a silent last-wins.
        pieces.setdefault(piece, int(token_id))
        longest = max(longest, min(len(piece), _MAX_PIECE))

    singles = {piece for piece in pieces if len(piece) == 1}

    edges: list[dict[int, tuple[int, int]]] = []
    roots: list[int] = []
    kept: list[str] = []
    untokenisable: list[tuple[str, str]] = []
    refused: list[tuple[str, str]] = []
    seen: set[str] = set()

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)

        # Tokenisability is checked FIRST, ahead of both ceilings, because it is
        # the only one of the three a caller cannot fix by editing their list:
        # "日本語" is three characters AND has no pieces, and telling them it is
        # too short would send them to lengthen a phrase this model can never
        # spell. The ceilings are policy; this is a fact about the vocabulary.
        text = " " + phrase
        missing = next((c for c in text if c not in singles), None)
        if missing is not None:
            untokenisable.append((phrase, missing))
            continue

        if len(phrase) < min_chars:
            refused.append((phrase, f"shorter than {min_chars} characters"))
            continue
        if len(kept) >= max_phrases:
            refused.append((phrase, f"over the {max_phrases}-phrase ceiling"))
            continue

        # One state per character offset, plus the terminal state at the end.
        base = len(edges)
        edges.extend({} for _ in range(len(text) + 1))
        for offset in range(len(text)):
            span = min(longest, len(text) - offset)
            for length in range(1, span + 1):
                token_id = pieces.get(text[offset:offset + length])
                if token_id is not None:
                    edges[base + offset][token_id] = (base + offset + length, length)
        roots.append(base)
        kept.append(phrase)

    start_edges: dict[int, int] = {}
    for root in roots:
        for token_id, (_, chars) in edges[root].items():
            if chars > start_edges.get(token_id, 0):
                start_edges[token_id] = chars

    return Automaton(
        edges=tuple(edges),
        roots=tuple(roots),
        start_edges=start_edges,
        phrases=tuple(kept),
        untokenisable=tuple(untokenisable),
        refused=tuple(refused),
    )


@dataclass
class Booster:
    """One utterance's live matches. NEVER shared between requests.

    pipeline.py guards the model with a BoundedSemaphore and main.py hands
    decodes to a threadpool, so two requests can be inside the same model object
    at once. Anything mutable stored on the model instance would be one caller's
    vocabulary leaking into another caller's transcript. This rides the call
    stack: `Parakeet.transcribe` builds it, passes it down as a keyword
    argument, and drops it.
    """

    automaton: Automaton
    weight: float = WEIGHT
    start_weight: float = START_WEIGHT
    gate: float = GATE
    live: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        # THE GATE IS CLAMPED TOO, AND THAT IS NOT SYMMETRY FOR ITS OWN SAKE —
        # it is the clamp that actually holds the invariant. Measured on the
        # probe: weight 15 clamped to 6 STILL ran away into "The The The The …"
        # once the gate was removed, because the ceiling on the bonus is not the
        # ceiling on its reach. The property that makes this feature safe is
        #
        #     a boosted token can overtake the winner ONLY IF it was already
        #     within `gate` logits of it
        #
        # so bounding `gate` by the same number bounds the reach at 6.0, under
        # the p90 of 8.43 by which blank wins when blank wins. That is what
        # keeps a bonus from converting silence into hallucinated vocabulary.
        # Clamped and logged rather than rejected: the ceiling exists to make a
        # destroyed transcript unreachable, not to argue with an operator about
        # a number in an environment variable.
        for name in ("weight", "start_weight", "gate"):
            value = getattr(self, name)
            if value > MAX_WEIGHT:
                log.warning("boost %s %.1f is over the %.1f ceiling; clamping",
                            name, value, MAX_WEIGHT)
                setattr(self, name, MAX_WEIGHT)
            elif value < 0.0:
                setattr(self, name, 0.0)

    def bonuses(self) -> dict[int, float]:
        """Token id -> bonus, for the frame about to be decoded.

        BLANK IS NEVER IN HERE. `<blk>` is index 8192, inside the boosted
        8193-wide slice, but specials are excluded when the automaton is built
        so blank is never an edge. A bonus can therefore only suppress blank
        RELATIVELY, whose failure mode is an insertion — there is no path from
        this design to a silent transcript. The measured budget before a blank
        flips is a median of 5.21 logits, p90 8.43.

        Bonuses are combined with max(), not sum(). A token that continues two
        different phrases is not twice as likely to be right, and summing would
        make a long boost list quietly more aggressive than a short one — the
        opposite of what the collateral-damage measurement asks for.
        """
        out: dict[int, float] = {}
        # Skipped entirely at the default start_weight of 0, which is what keeps
        # the hot path proportional to the number of matches actually in flight
        # rather than to the size of the boost list.
        if self.start_weight > 0.0:
            for token_id, chars in self.automaton.start_edges.items():
                out[token_id] = self.start_weight * chars
        if self.live:
            edges = self.automaton.edges
            for state in self.live:
                for token_id, (_, chars) in edges[state].items():
                    value = self.weight * chars
                    if value > out.get(token_id, 0.0):
                        out[token_id] = value
        return out

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """The boosted logits. Returns a COPY; the caller keeps the original.

        Two reasons the copy is not optional. `logits` is a slice view of the
        ONNX Runtime output buffer, so writing through it corrupts a buffer we
        do not own. And the log-probabilities onnx-asr reports must be computed
        from the UNBOOSTED array: a logprob is a claim about the model's
        confidence, and asr._parakeet_words averages those into every
        Word.logprob and every synthesised avg_logprob, so a boosted one would
        propagate the bonus into verbose_json as if the model had said it.
        """
        bonuses = self.bonuses()
        if not bonuses:
            return logits
        boosted = logits.copy()
        ceiling = float(logits.max()) - self.gate
        for token_id, bonus in bonuses.items():
            if logits[token_id] >= ceiling:
                boosted[token_id] += bonus
        return boosted

    def advance(self, token: int) -> None:
        """Fold one EMITTED token into the live set.

        Called exactly where the decoder appends a token, which is what makes
        this correct under TDT's frame skipping without ever looking at `t`.
        A skip is a decision about WHEN the next token is predicted, not about
        WHAT was emitted, and the live set is a function of the emitted token
        sequence alone.

        A BLANK DOES NOT KILL A LIVE MATCH, and must not. In a transducer,
        blank means "nothing more at this frame", not "word boundary" — 16 of
        the probe's 51 decode calls were blank, interleaved through the middle
        of words. Killing on blank would break every phrase whose interior
        happens to span a frame. Blank never reaches here because the caller
        only calls this on an emission.

        Roots are tried on every token regardless of start_weight: entering a
        phrase on acoustics alone is exactly what start_weight 0 means.
        """
        edges = self.automaton.edges
        nxt: set[int] = set()
        for state in self.live:
            edge = edges[state].get(token)
            if edge is not None:
                nxt.add(edge[0])
        for root in self.automaton.roots:
            edge = edges[root].get(token)
            if edge is not None:
                nxt.add(edge[0])
        self.live = nxt


def make_subclass():  # noqa: ANN201 - the class is built lazily, see below
    """Build BoostedParakeetTdt. Imports onnx-asr, so it is called lazily.

    A SUBCLASS, NOT A MONKEY-PATCH, and constructed through onnx-asr's own
    resolver so nothing about the pinned dependency is mutated: no assignment
    onto an upstream class, no sys.modules trick, no vendored fork. If the seam
    ever moves, this raises at startup instead of silently reverting — which is
    the entire argument for doing it this way. See verify_seam.
    """
    from onnx_asr.asr import log_softmax  # noqa: PLC0415
    from onnx_asr.models.nemo import NemoConformerTdt  # noqa: PLC0415

    class BoostedParakeetTdt(NemoConformerTdt):
        """NemoConformerTdt with a boost hook between the joint and the argmax.

        `_decoding` is upstream's greedy loop. With no `boost` keyword this
        class does not run its own copy of that loop at all — it delegates to
        `super()._decoding`, so an unboosted decode is upstream's code executing
        upstream's arithmetic. That is what makes "no boost list means
        byte-identical output" a structural property rather than a claim to be
        re-verified after every edit. (It is verified anyway, on real audio;
        see tests/test_boosting.py.)
        """

        def _decoding(self, encoder_out, encoder_out_lens, /, **kwargs):  # noqa: ANN001, ANN202
            # The seam probe: proof, at startup, that a keyword argument handed
            # to adapter.recognize() actually arrives in THIS method. Upstream's
            # loop would ignore it, which is precisely the silent revert this
            # exists to catch, so the probe must be popped here and nowhere else.
            probe = kwargs.pop("_boost_probe", None)
            if probe is not None:
                probe.append(type(self).__name__)

            booster = kwargs.pop("boost", None)
            if booster is None:
                yield from super()._decoding(encoder_out, encoder_out_lens, **kwargs)
                return

            # Below this line is upstream's loop with three lines added. It is a
            # copy on purpose: the alternative is re-entering the stock loop
            # per frame, which the generator shape does not allow. Keep it
            # aligned with onnx_asr/asr.py:_AsrWithTransducerDecoding._decoding
            # — SUPPORTED_ONNX_ASR is what stops it drifting unnoticed.
            need_logprobs = kwargs.get("need_logprobs")
            if self.use_low_precision:
                encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])

            for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
                assert encodings_len <= encodings.shape[0]
                prev_state = self._create_state()
                tokens: list[int] = []
                timestamps: list[int] = []
                logprobs: list[float] = []
                # One Booster per utterance in the batch. Sharing it across a
                # batch would carry one clip's half-finished phrase into the
                # next clip's first frame.
                live = Booster(automaton=booster.automaton, weight=booster.weight,
                               start_weight=booster.start_weight, gate=booster.gate)

                t = 0
                emitted_tokens = 0
                while t < encodings_len:
                    logits, step, state = self._decode(tokens, prev_state, encodings[t])
                    assert logits.shape[-1] <= self._vocab_size

                    # Two arrays from here on, and keeping them apart is the
                    # whole discipline of this block: `boosted` decides, and
                    # `logits` reports. They are the same object when nothing
                    # is live, because apply() returns its argument unchanged
                    # rather than copying for nothing.
                    boosted = live.apply(logits)
                    token = boosted.argmax()

                    if token != self._blank_idx:
                        prev_state = state
                        tokens.append(int(token))
                        timestamps.append(t)
                        emitted_tokens += 1
                        if need_logprobs:
                            # From `logits`, NOT from `boosted`. The bonus is
                            # ours; the confidence is the model's, and
                            # asr._parakeet_words averages these into every
                            # Word.logprob and every synthesised avg_logprob.
                            logprobs.append(log_softmax(logits)[token])
                        live.advance(int(token))

                    if step > 0:
                        t += step
                        emitted_tokens = 0
                    elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                        t += 1
                        emitted_tokens = 0

                yield tokens, timestamps, logprobs if need_logprobs else None

    return BoostedParakeetTdt


def load(model_id: str, quantisation: str | None):  # noqa: ANN201
    """Load Parakeet with the boosted decoder, through onnx-asr's own resolver.

    This mirrors `onnx_asr.loader.Manager.create_asr` with one line changed —
    the model type is our subclass instead of the one the registry would map
    `nemo-conformer-tdt` to. `Resolver.__init__` accepts a bare type as well as
    the registry dict, so passing the class bypasses the registry while still
    using stock Hugging Face resolution: the same four files are fetched,
    because `_get_model_files` is inherited unchanged.

    It touches two underscore-prefixed helpers, `_create_preprocessor` and
    `_create_asr_adapter`. That is a deliberate trade and the better one: if a
    version bump moves them this raises AttributeError at startup, where the
    alternative — a public path that quietly ignores our class — would revert
    the feature in silence.
    """
    from onnx_asr.loader import Manager  # noqa: PLC0415
    from onnx_asr.onnx import update_onnx_providers  # noqa: PLC0415
    from onnx_asr.resolver import Resolver  # noqa: PLC0415

    boosted = make_subclass()
    manager = Manager()
    resolver = Resolver(boosted, model_id, None, offline=None)
    config = update_onnx_providers(
        manager.default_onnx_config,
        excluded_providers=boosted._get_excluded_providers(),
    )
    asr = boosted(resolver.resolve_model(quantization=quantisation),
                  manager._create_preprocessor, config)
    return manager._create_asr_adapter(asr)


def verify_seam(adapter) -> None:  # noqa: ANN001
    """Prove, before serving, that biasing will actually reach the decoder.

    THE FAILURE THIS PREVENTS IS THE ONLY ONE THAT MATTERS. Upstream's
    `_decoding` reads its options with `kwargs.get` and ignores unknown keys, so
    a renamed method, a changed signature, or an adapter that stops forwarding
    kwargs would send `boost=...` down a stock code path where it is discarded
    with no error, no warning and no visible difference except worse
    transcripts. A pinned requirement delays that; it does not detect it.

    Every check below is a thing that has to be true for a bonus to land on the
    right number, and each raises UnsupportedSeam naming what changed. The
    caller turns that into `accepts_vocabulary = False` and a refusal by name,
    which is this service's standing answer to a capability it has not got.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        installed = version("onnx-asr")
    except PackageNotFoundError:  # pragma: no cover - the package is a hard dep
        installed = "unknown"
    if installed not in SUPPORTED_ONNX_ASR:
        raise UnsupportedSeam(
            f"onnx-asr {installed} is installed; decode-time biasing has only "
            f"been read against {', '.join(sorted(SUPPORTED_ONNX_ASR))}. The "
            "greedy TDT loop is a private method and its kwargs are read with "
            "kwargs.get(), so an unreviewed version could accept the boost "
            "argument and discard it. Re-read "
            "_AsrWithTransducerDecoding._decoding and add the version to "
            "boosting.SUPPORTED_ONNX_ASR.")

    asr = getattr(adapter, "asr", None)
    expected = make_subclass().__name__
    if asr is None or type(asr).__name__ != expected:
        raise UnsupportedSeam(
            f"the loaded model is {type(asr).__name__}, not {expected}: "
            "onnx-asr's resolver did not construct the boosted subclass.")

    vocab = getattr(asr, "_vocab", None)
    if not vocab or asr._vocab_size != len(vocab):
        raise UnsupportedSeam(
            f"vocabulary is {len(vocab or ())} pieces but _vocab_size is "
            f"{getattr(asr, '_vocab_size', None)}; the boost automaton is built "
            "from _vocab and indexed against _vocab_size.")
    if not 0 <= asr._blank_idx < asr._vocab_size:
        raise UnsupportedSeam(
            f"_blank_idx {asr._blank_idx} is outside the {asr._vocab_size}-wide "
            "token slice, so blank cannot be identified.")

    # The joint's output is token logits followed by the duration head. Read the
    # split back rather than trusting it: a model swap that changes the number
    # of durations would mis-slice silently, and the calibrated weights in this
    # module are a property of THIS model's logit scale anyway.
    joint_width = None
    for output in asr._decoder_joint.get_outputs():
        if output.name == "outputs":
            last = output.shape[-1]
            joint_width = last if isinstance(last, int) else None
    if joint_width is not None and joint_width <= asr._vocab_size:
        raise UnsupportedSeam(
            f"the joint emits {joint_width} values but the vocabulary is "
            f"{asr._vocab_size} wide; there is no duration head to slice off, "
            "so this is not the TDT joint these weights were calibrated on.")

    # The end-to-end proof: a keyword argument handed to the PUBLIC adapter
    # arrives in OUR _decoding. Two tenths of a second of silence through the
    # real encoder, once, at startup. Both call sites in Parakeet.transcribe are
    # covered because the plain and timestamped adapters have different
    # _recognize_batch implementations and only one of them adds need_logprobs.
    silence = np.zeros(3200, dtype=np.float32)
    for name, target in (("recognize", adapter),
                         ("with_timestamps", adapter.with_timestamps())):
        probe: list[str] = []
        target.recognize(silence, sample_rate=16_000, _boost_probe=probe)
        if not probe:
            raise UnsupportedSeam(
                f"a keyword argument passed to {name}() never reached the "
                "boosted _decoding. onnx-asr's adapter has stopped forwarding "
                "unknown options, so a boost list would be silently discarded.")

    log.info("boost seam verified: onnx-asr %s, %d pieces, blank %d, joint %s",
             installed, asr._vocab_size, asr._blank_idx, joint_width or "dynamic")
