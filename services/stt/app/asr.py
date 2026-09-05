"""The recogniser, chosen at startup.

Parakeet is the default. Measured across 25 conditions and five Brazilian
Portuguese corpora on identical audio, it beat Whisper large-v3 on 21 of them
at roughly seventy times the speed, in a third of the memory:

    Parakeet TDT 0.6B v3   WER 0.144 pt-BR / 0.121 en   47-63x realtime
    Whisper large-v3       WER 0.250 pt-BR / 0.131 en   0.5-0.9x realtime

It also degrades far more gracefully. Band-limiting the audio to 4 kHz — a
cheap or distant microphone — cost Whisper +206% WER on CORAA and Parakeet
+41%. That collapse was a property of Whisper's autoregressive decoder, not a
physical limit.

Whisper remains available, because it genuinely wins on clean read speech and
because it translates. It used to be here for a third reason — that it was the
only one of the two that accepted a vocabulary at DECODE time — and that reason
is gone: see boosting.py. Both engines now bias their decoder, by different
mechanisms, from the same list of terms.

There is deliberately no second recogniser. A consensus pass was tried and
removed: across every disagreement observed, the second model was the wrong
one, so its dissent carried no information and cost roughly 40% of throughput
— worst on the short clips that dictation actually consists of.

WHAT EACH ENGINE CAN ACTUALLY DO
--------------------------------
The two are not interchangeable, and the compatibility layer answers for the
difference rather than papering over it — every capability below is a flag the
route reads to decide between honouring a field and refusing it by name. The
flags exist so that a refusal can never drift out of step with the code that
would have done the work.

  translate       Whisper has a translate task (transcribe.py takes
                  task="transcribe"|"translate"). Parakeet has no translate
                  task and no target-language conditioning; there is nothing to
                  route the request to.
  language        Whisper takes a language hint and reports what it detected.
                  Parakeet v3 takes none — onnx-asr's RecognizeOptions
                  documents `language` as "only for Whisper and Canary models"
                  — and surfaces no detected language to report back.
  vocabulary      BOTH ENGINES, by different mechanisms, from one list of
                  terms. Whisper takes hotwords at decode time. Parakeet takes
                  a boosting automaton through its TDT decoding loop — shallow
                  fusion added to the joint's logits one frame before the
                  argmax; boosting.py is the whole argument and the measured
                  numbers.
                  This file said for a long time that Parakeet "has no
                  vocabulary argument at all". That was true of the ARGUMENT
                  and false about what is reachable, and believing it cost this
                  service a feature and cost `prompt` a 4xx on the default
                  engine. Corrected here rather than left as a footnote,
                  because it is the sentence that misled everyone who read it.
                  `accepts_vocabulary` is still a claim about the DECODER and
                  only about the decoder. It is now true on Parakeet, and it is
                  set per INSTANCE rather than per class, because a version of
                  onnx-asr whose decoding seam this service has not been read
                  against turns it back off — see boosting.verify_seam. It does
                  not decide whether `prompt` and `keywords[]` are accepted;
                  those are accepted either way, because the terms also compile
                  into that request's post-decode repair.
  temperature     Whisper has one, though pinning it disables the fallback
                  ladder. A TDT decoder has no sampling temperature.
  streaming       faster-whisper yields each segment as its 30 s window is
                  decoded. Measured here on a 297 s clip: the first segment
                  arrived 7.8 s into a 68.6 s transcription, 11% of the way.
                  Parakeet encodes the whole waveform before its decode loop
                  starts and emits nothing until that loop ends — measured 5.07
                  s to first and only output on a 14.2 s clip.
  token logprobs  onnx-asr returns a logprob per token. faster-whisper exposes
                  an average per segment and a probability per word, but no
                  per-token logprob, so include[]=logprobs is refused there
                  rather than answered with a differently-shaped number.
  token ids       Whisper's segments carry them. onnx-asr maps ids to strings
                  inside its decoder and returns only the strings, so
                  `segments[].tokens` is empty under Parakeet.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

import numpy as np

from . import boosting

log = logging.getLogger("stt-stack.asr")

SAMPLE_RATE = 16_000

PARAKEET_DEFAULT = "istupakov/parakeet-tdt-0.6b-v3-onnx"
WHISPER_DEFAULT = "large-v3"

# Whisper's own default ladder, kept for every request that does not pin a
# temperature: it is what retries a low-confidence decode.
TEMPERATURE_LADDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass(frozen=True)
class Word:
    """One word and where it was heard. Times are in the audio the ASR saw."""

    word: str
    start: float
    end: float
    # Mean log probability of the tokens that formed the word. Not part of the
    # wire shape — it is what a segment's avg_logprob is averaged from on the
    # engine that reports no segments of its own.
    logprob: float = 0.0


@dataclass(frozen=True)
class Segment:
    """The specification's TranscriptionSegment, all ten required fields."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class TokenLogprob:
    """One entry of the `logprobs` array include[]=logprobs asks for."""

    token: str
    logprob: float
    bytes: tuple[int, ...]


@dataclass(frozen=True)
class Options:
    """What one request asked the recogniser for."""

    language: str | None = None
    # The same vocabulary in the two shapes the two decoders take. Both are
    # filled from one list — openai_api._decode_vocabulary — so they cannot
    # disagree about what this request asked for, and each engine reads the one
    # it can use. Re-splitting the joined string would corrupt a term with a
    # comma in it, which is why the tuple exists rather than being derived.
    hotwords: str | None = None          # faster-whisper's shape
    vocabulary: tuple[str, ...] = ()     # boosting.compile_automaton's shape
    # Decode-time biasing on Parakeet is OPT-IN PER REQUEST and off by default.
    # A glossary whose terms do not occur in the audio raised WER by 12% on
    # this engine across 25 cells, so a boost list nobody asked for is a
    # measured cost rather than a neutral. Whisper's hotwords are unaffected by
    # this flag: they predate it, they are what `prompt` has always done there,
    # and moving them would change a shipped engine under a change about the
    # other one. See openai_api._boost.
    boost: bool = False
    temperature: float | None = None
    task: str = "transcribe"
    want_words: bool = False
    want_segments: bool = False
    want_logprobs: bool = False

    @property
    def want_detail(self) -> bool:
        return self.want_words or self.want_segments or self.want_logprobs


@dataclass(frozen=True)
class Recognition:
    """What one run produced, before the glossary and before the timeline.

    `segments` is None on an engine that reports none of its own; the pipeline
    then cuts the words into segments at the VAD's own boundaries, which are
    the only real ones available.
    """

    text: str
    language: str | None = None
    segments: tuple[Segment, ...] | None = None
    words: tuple[Word, ...] = ()
    logprobs: tuple[TokenLogprob, ...] | None = None
    # The phrases that actually reached the decoder as a boost automaton.
    # Empty when nothing was boosted, which is the default.
    #
    # This exists so that "honoured" and "did something" can be told apart from
    # outside, which is the same reason X-Glossary-Repaired exists for the
    # repair half. A term can be dropped from the boost list for three
    # different reasons — no pieces for one of its characters, under the
    # minimum length, over the phrase ceiling — and a caller who cannot see
    # which of their terms survived is back to believing their vocabulary was
    # applied when it was not.
    boosted: tuple[str, ...] = ()


class Parakeet:
    """Parakeet TDT via ONNX Runtime, with decode-time biasing.

    A GLOSSARY HAS TWO HALVES HERE NOW, AND IT USED TO HAVE ONE. Terms are
    still repaired after decoding (glossary.py), and a request that opts in
    with `boost` additionally gets them compiled into a boosting automaton and
    fused into the TDT decoding loop before the argmax. boosting.py holds the
    mechanism, the measured logit scale and every default.

    What this docstring said until that landed — "CTC/TDT, so no decode-time
    vocabulary", and that FluidAudio's CTC boosting for the same model was
    CoreML-only and therefore out of reach — was wrong in the way that matters:
    it described onnx-asr's public argument list as though it were the model's
    capability, and the next reader concluded the feature was impossible. It
    was reachable in about forty lines.

    Biasing is off unless a request asks, because a boost list whose terms are
    absent from the audio is a measured accuracy cost, not a neutral.
    """

    name = "parakeet"
    # Set per instance in __init__ as well, and deliberately: a version of
    # onnx-asr whose decoding seam has not been read turns this back off rather
    # than accepting a boost list and quietly discarding it.
    accepts_vocabulary = True
    # Whether `boost` is a field this engine has an answer for. Separate from
    # accepts_vocabulary on purpose: Whisper takes a vocabulary at decode time
    # and has no switch for it, so it refuses the field rather than pretending
    # the request changed something. One flag per capability, as the header
    # above argues, so that a refusal cannot drift from the code behind it.
    accepts_boost = True
    reports_segments = False
    accepts_language = False
    accepts_temperature = False
    can_translate = False
    can_stream = False
    reports_language = False
    reports_token_logprobs = True
    reports_token_ids = False

    def __init__(self, model_id: str, quantisation: str) -> None:
        # Compiled automata, keyed by the exact term tuple that produced them.
        # This is a CACHE OF DERIVED IMMUTABLE DATA and nothing else — no
        # request's state lives on this object, because pipeline.py guards the
        # model with a BoundedSemaphore and main.py decodes in a threadpool, so
        # anything mutable and per-request here would be one caller's
        # vocabulary appearing in another caller's transcript. The Booster that
        # holds the live matches is built per call and rides the stack.
        self._automata: dict[tuple[str, ...], boosting.Automaton] = {}
        self._automata_lock = threading.Lock()
        self.vocabulary_unavailable: str | None = None

        try:
            self._model = boosting.load(model_id, quantisation)
        except (AttributeError, TypeError) as exc:
            # boosting.load reaches two underscore-prefixed helpers on
            # onnx-asr's Manager. If a version bump moves them this is where it
            # surfaces, and the right answer is to serve without biasing rather
            # than to refuse to start: transcription is the service, biasing is
            # an opt-in extra. It is loud in the log, false in /health and
            # refused by name on the route — the one thing it must never be is
            # accepted and silently skipped.
            import onnx_asr  # noqa: PLC0415

            self._model = onnx_asr.load_model(model_id, quantization=quantisation)
            self._disable_boosting(
                f"onnx-asr's loader would not construct the boosted decoder "
                f"({type(exc).__name__}: {exc})")
            return

        try:
            boosting.verify_seam(self._model)
        except boosting.UnsupportedSeam as exc:
            # The adapter still works — BoostedParakeetTdt with no boost
            # argument delegates to upstream's own loop — so there is nothing
            # to reload, only a capability to withdraw.
            self._disable_boosting(str(exc))

    def _disable_boosting(self, reason: str) -> None:
        self.accepts_vocabulary = False
        self.accepts_boost = False
        self.vocabulary_unavailable = reason
        log.error("decode-time biasing is UNAVAILABLE: %s", reason)

    def vocabulary_problems(self, terms: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        """Phrases this model cannot spell, as (phrase, offending character).

        A phrase fails only when it contains a character with no
        single-character piece in the model's vocabulary — "café ☕" at '☕',
        "日本語" at '日' — while "São Paulo", "conteúdo" and "ação" all build.
        The route turns this into a 400 naming both, for the same reason
        profiles.UnknownProfile exists: a caller who believes their vocabulary
        was applied when it was not is the silence this surface is built to
        prevent.
        """
        return self._automaton(terms).untokenisable

    def _automaton(self, terms: tuple[str, ...]) -> boosting.Automaton:
        with self._automata_lock:
            cached = self._automata.get(terms)
            if cached is not None:
                return cached
        compiled = boosting.compile_automaton(self._model.asr._vocab, terms)
        with self._automata_lock:
            # Bounded for profiles.py's reason rather than a fresh one: a
            # client sending a different one-off term list every request should
            # not grow this without limit. Cleared rather than evicted by age
            # because the cost of a miss is a few hundred microseconds.
            if len(self._automata) >= 64:
                self._automata.clear()
            self._automata[terms] = compiled
        return compiled

    def _booster(self, opts: Options) -> tuple[object | None, tuple[str, ...]]:
        """This request's boost automaton, and the phrases that survived it."""
        if not (opts.boost and opts.vocabulary and self.accepts_vocabulary):
            return None, ()
        automaton = self._automaton(opts.vocabulary)
        if not automaton:
            return None, ()
        return boosting.Booster(automaton=automaton), automaton.phrases

    def transcribe(self, samples: np.ndarray, opts: Options) -> Recognition:
        # Parakeet v3 detects language itself and takes no hint, and its TDT
        # decoder has no temperature. The route refuses both by name before
        # reaching here, so either still set would be a routing bug rather than
        # something to swallow quietly.
        #
        # opts.hotwords is neither honoured nor a bug here: it is the same
        # vocabulary in faster-whisper's shape, filled by the same function
        # that filled opts.vocabulary. This engine reads the tuple.
        booster, boosted = self._booster(opts)
        # `boost=None` is not merely equivalent to the unboosted path, it IS
        # the unboosted path: BoostedParakeetTdt delegates straight to
        # onnx-asr's own _decoding, so a request that does not opt in runs
        # upstream's code. Verified on sixteen real corpus clips —
        # tests/test_boosting.py, text, tokens, timestamps and logprobs all
        # identical.
        extra = {"boost": booster} if booster is not None else {}

        if not opts.want_detail:
            # The plain adapter, for the default response_format. Asking for
            # timestamps costs about 5% on a 14.2 s clip here (5.07 s against
            # 5.34 s), which is not worth paying on every dictation request for
            # numbers no one reads.
            text = self._model.recognize(
                samples, sample_rate=SAMPLE_RATE, **extra).strip()  # type: ignore[arg-type]
            return Recognition(text=text, boosted=boosted)

        result = self._model.with_timestamps().recognize(
            samples, sample_rate=SAMPLE_RATE, **extra)  # type: ignore[arg-type]
        words, logprobs = _parakeet_words(result)
        return Recognition(
            text=result.text.strip(),
            words=words,
            logprobs=logprobs if opts.want_logprobs else None,
            boosted=boosted,
        )

    def stream(self, samples: np.ndarray, opts: Options):
        raise NotImplementedError(self.name)


def _parakeet_words(result) -> tuple[tuple[Word, ...], tuple[TokenLogprob, ...]]:  # noqa: ANN001
    """Group onnx-asr's token stream into words, with times and logprobs.

    Two properties of the TDT decoder shape this, both verified against a
    14.2 s reading of Dumas at 16 kHz:

    * A token that starts a word carries a leading space (onnx-asr rewrites
      sentencepiece's U+2581 to one), and punctuation does not, so a word
      boundary is exactly a leading space. Joining the words back with single
      spaces reproduces the transcript character for character.
    * A timestamp is the frame at which the token was EMITTED, quantised to
      0.08 s (0.01 s window x 8 subsampling). "English" came back as
      En=0.32 gl=0.48 ish=0.64 for a word audibly beginning around 0.2 s —
      the emission time trails the token. So a word's `end` is its last
      token's timestamp, and its `start` is the timestamp of the token BEFORE
      it, which is the last moment the previous word was still being emitted.
      Words are therefore contiguous, with no invented gaps.
    """
    tokens = list(result.tokens or [])
    times = list(result.timestamps or [])
    probs = list(result.logprobs or [])
    if not tokens or len(times) != len(tokens):
        return (), ()

    logprobs = tuple(
        TokenLogprob(token=token, logprob=float(probs[i]) if i < len(probs) else 0.0,
                     bytes=tuple(token.encode("utf-8")))
        for i, token in enumerate(tokens)
    )

    words: list[Word] = []
    start_index = 0
    for index in range(len(tokens) + 1):
        starts_word = index == len(tokens) or tokens[index].startswith(" ")
        if not starts_word or index == 0:
            continue
        piece = "".join(tokens[start_index:index]).strip()
        if piece:
            first, last = start_index, index - 1
            start = times[first - 1] if first > 0 else 0.0
            end = max(times[last], start)
            window = probs[start_index:index]
            words.append(Word(word=piece, start=float(start), end=float(end),
                              logprob=float(np.mean(window)) if window else 0.0))
        start_index = index

    return tuple(words), logprobs


class Whisper:
    """Whisper via CTranslate2. Takes its vocabulary as decoder hotwords."""

    name = "whisper"
    accepts_vocabulary = True
    # No `boost` switch: hotwords here are unconditional and predate the field.
    # Adding one would change what `prompt` does on a shipped engine as a side
    # effect of a change about the other engine.
    accepts_boost = False
    vocabulary_unavailable: str | None = None
    reports_segments = True
    accepts_language = True
    accepts_temperature = True
    can_translate = True
    can_stream = True
    reports_language = True
    reports_token_logprobs = False
    reports_token_ids = True

    def __init__(self, model_id: str, compute_type: str, threads: int,
                 language: str | None, hotwords: str | None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.language = language
        self.hotwords = hotwords
        self._model = WhisperModel(
            model_id, device="cpu", compute_type=compute_type, cpu_threads=threads
        )

    def vocabulary_problems(self, terms: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        """None ever. Whisper's tokeniser has a byte fallback, so every string
        it is handed is expressible; there is no phrase it cannot spell."""
        return ()

    def _vocabulary(self, extra: str | None) -> str | None:
        """The configured glossary, plus whatever this request added.

        A request prompt extends the deployment's vocabulary rather than
        replacing it. The glossary is the list of terms this box is known to
        mishear, measured; dropping it because a client named one extra proper
        noun would trade a measured win for a guess.
        """
        return ", ".join(part for part in (self.hotwords, extra) if part) or None

    def _run(self, samples: np.ndarray, opts: Options):
        return self._model.transcribe(
            samples,
            # None means autodetect, which is right for a speaker who
            # code-switches. Pinning the wrong language does not degrade the
            # transcript, it TRANSLATES it: English speech under language="pt"
            # returns fluent Portuguese that reads like a working transcript.
            language=opts.language or self.language,
            task=opts.task,
            beam_size=5,
            # The ladder is the default because pinning a single temperature
            # disables Whisper's retry on low-confidence output. A request that
            # names one has asked for that trade explicitly.
            temperature=(TEMPERATURE_LADDER if opts.temperature is None
                         else [opts.temperature]),
            condition_on_previous_text=False,
            hotwords=self._vocabulary(opts.hotwords),
            # Word timings cost a second decoder pass per segment, so they are
            # computed only when timestamp_granularities asked for them.
            word_timestamps=opts.want_words,
            vad_filter=False,  # done once upstream, for whichever model runs
        )

    def transcribe(self, samples: np.ndarray, opts: Options) -> Recognition:
        segments, info = self._run(samples, opts)
        collected = tuple(_whisper_segment(s) for s in segments)
        return Recognition(
            text=" ".join(s.text.strip() for s in collected).strip(),
            # On a translation the output language is English by definition;
            # info.language is what was DETECTED in the input, which is a
            # different claim and not the one verbose_json makes.
            language="en" if opts.task == "translate" else info.language,
            segments=collected,
            words=tuple(word for s in collected for word in s.words),
        )

    def stream(self, samples: np.ndarray, opts: Options):
        """Yield each segment as CTranslate2 finishes the window it is in.

        Genuinely incremental, and only as incremental as the model is: a
        30 s window is decoded as a unit, so nothing at all can be emitted for
        a clip shorter than one window until it is done. Measured with
        `tiny`/int8 on 4 threads — 297 s clip, first segment at 7.8 s of 68.6 s
        total; 14.2 s clip, first and last together at 2.6 s.
        """
        segments, _ = self._run(samples, opts)
        for segment in segments:
            yield _whisper_segment(segment)


def _whisper_segment(segment) -> Segment:  # noqa: ANN001 - faster_whisper.Segment
    """faster-whisper's segment, in the specification's shape.

    Every one of the ten required fields is a field faster-whisper already
    computed and this service used to discard in ' '.join(s.text ...).
    """
    return Segment(
        id=segment.id,
        # faster-whisper's own window offset, in centiseconds of the audio the
        # DECODER saw. With the VAD on that is the compacted timeline, not the
        # client's clip; start and end are mapped back, seek is left alone
        # because it indexes a window rather than naming a moment.
        seek=segment.seek,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        tokens=tuple(segment.tokens or ()),
        temperature=float(segment.temperature if segment.temperature is not None else 0.0),
        avg_logprob=segment.avg_logprob,
        compression_ratio=segment.compression_ratio,
        no_speech_prob=segment.no_speech_prob,
        words=tuple(
            Word(word=w.word.strip(), start=w.start, end=w.end,
                 # faster-whisper reports a linear probability per word; the
                 # log of it is what the rest of this file speaks in.
                 logprob=math.log(w.probability) if w.probability > 0 else -20.0)
            for w in (segment.words or ())
        ),
    )


def build(threads: int, hotwords: str | None) -> Parakeet | Whisper:
    """Load the model named by STT_MODEL. Parakeet unless asked otherwise."""
    choice = os.getenv("STT_MODEL", "parakeet").strip().lower()

    if choice in {"parakeet", "parakeet-v3"}:
        model_id = os.getenv("STT_MODEL_ID", PARAKEET_DEFAULT)
        model = Parakeet(model_id, os.getenv("STT_QUANTISATION", "int8"))
        log.info("parakeet ready: %s (decode-time biasing %s)", model_id,
                 "available, opt in per request with boost=true"
                 if model.accepts_vocabulary else
                 f"UNAVAILABLE: {model.vocabulary_unavailable}")
        return model

    if choice == "whisper":
        model_id = os.getenv("STT_MODEL_ID", WHISPER_DEFAULT)
        model = Whisper(
            model_id=model_id,
            compute_type=os.getenv("STT_QUANTISATION", "int8"),
            threads=threads,
            language=os.getenv("STT_LANGUAGE") or None,
            hotwords=hotwords,
        )
        log.info("whisper ready: %s (hotwords %s)", model_id,
                 "on" if hotwords else "off")
        return model

    raise ValueError(
        f"STT_MODEL={choice!r} is not recognised; expected 'parakeet' or 'whisper'"
    )
