"""The transcription pipeline, separate from the shape it is returned in.

    audio -> VAD -> recogniser -> glossary repair -> text

Three routes share this: the native /transcribe, which returns everything the
run measured, and /v1/audio/transcriptions and /v1/audio/translations, which
return the subset OpenAI's specification has fields for. The work lives here so
the compatibility layer cannot slowly become a second pipeline that drifts from
the first — which is the usual way these shims rot.

Two things this file owns that the recogniser cannot:

  the timeline   The ASR sees speech runs, concatenated. Every timestamp it
                 reports is in that compacted timeline, and a subtitle needs
                 them in the client's. vad.Speech carries the offsets; the
                 mapping is applied here, once, for both engines.

  the segments   Whisper reports its own. Parakeet reports words only, so its
                 segments are cut at the VAD's boundaries — real pauses in the
                 recording, which is the only honest place to cut. Two of the
                 ten required fields have no Parakeet equivalent and are
                 synthesised; see _segments_from_words.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np
from fastapi import HTTPException

from . import asr, audio, glossary, profiles, vad

SAMPLE_RATE = 16_000

MODEL = os.getenv("STT_MODEL", "parakeet")
# Profiles applied when a request selects none. UNSET BY DEFAULT, and that is a
# deliberate behaviour change: this service used to compile one file at boot and
# apply it to every transcript, which is both a public image carrying one
# person's project names and a measured accuracy cost — a glossary whose terms
# do NOT occur in the audio raised WER by 12% on Parakeet and 28% on Whisper
# across 250 conditions. Absent a selection, behaviour is now the
# specification's: no glossary, no biasing.
#
# A deployment that wants the old always-on shape asks for it by name —
# STT_GLOSSARY_DEFAULT=dictation,tech — and pays the cost knowingly.
DEFAULT_PROFILES = os.getenv("STT_GLOSSARY_DEFAULT", "")
# The one knob that matters on a shared host. ONNX Runtime and CTranslate2
# both size their pools from the host core count, not the cgroup, so a
# container CPU limit without this leaves threads fighting for their own
# slice. See the README.
THREADS = int(os.getenv("STT_THREADS", "4"))
VAD_ENABLED = os.getenv("STT_VAD", "1") not in {"0", "false", "no"}
# Off switch for decode-time biasing, so a benchmark can separate what the
# vocabulary contributes from what the model does. BOTH ENGINES: it used to say
# "Whisper only — Parakeet has no such mechanism", which stopped being true
# when boosting.py landed, and an off switch that covered one of the two
# engines would have made this variable a lie on the default deployment.
# Text repair is unaffected either way, and that includes the repair a
# request's own `prompt` compiles into: this switch is about what the DECODER
# is told, which is the only place a vocabulary changes what the model
# produces. A benchmark measuring the model is not contaminated by a rewrite
# applied to the model's finished output, and would be lied to by an off switch
# that quietly dropped it.
HOTWORDS_ENABLED = os.getenv("STT_HOTWORDS", "1") not in {"0", "false", "no"}
# Requests allowed inside the recogniser at once, or 0 for no limit.
#
# 0 is the default because this service is already deployed and a ceiling
# picked here rather than measured on the host it runs on would start refusing
# work it is doing happily today. Set it and /v1 answers 429 with Retry-After
# instead of queueing — which is what the specification enumerates for this
# path, and what an OpenAI client's own backoff is written against.
MAX_CONCURRENT = int(os.getenv("STT_MAX_CONCURRENT", "0"))

log = logging.getLogger("stt-stack")

state: dict[str, object] = {}
_slots = threading.BoundedSemaphore(MAX_CONCURRENT) if MAX_CONCURRENT > 0 else None


class Busy(Exception):
    """Every recogniser slot is taken. The caller answers 429."""


@dataclass(frozen=True)
class Tuning:
    """Per-request VAD settings, from chunking_strategy. None means the default."""

    threshold: float | None = None
    min_silence_ms: int | None = None
    speech_pad_ms: int | None = None
    enabled: bool = True


@dataclass(frozen=True)
class Result:
    """Everything one run measured. Rounded here, so every route agrees."""

    text: str
    raw: str
    repaired: list[str]
    model: str
    audio_seconds: float
    speech_seconds: float
    compute_seconds: float
    realtime_factor: float
    # Everything below is for /v1 only; the native route's body is unchanged.
    language: str | None = None
    segments: tuple[asr.Segment, ...] = ()
    words: tuple[asr.Word, ...] = ()
    logprobs: tuple[asr.TokenLogprob, ...] | None = None
    task: str = "transcribe"
    # Phrases that reached the decoder as a boost automaton, for
    # x-boost-applied. Empty on every request that did not opt in, which is
    # every request by default.
    boosted: tuple[str, ...] = ()


def start() -> None:
    """Load the glossary, the recogniser and, if enabled, the VAD."""
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

    registry = profiles.load_registry()
    state["glossaries"] = registry
    log.info("glossary profiles: %s", ", ".join(
        f"{name} ({registry.profiles[name].source})" for name in registry.names)
        or "none")

    # The default selection, which is EMPTY unless STT_GLOSSARY_DEFAULT names
    # profiles. state["rules"] is what a request that selects nothing gets;
    # everything else is compiled per selection and cached in the registry.
    default = registry.select(profiles.split_selection(DEFAULT_PROFILES))
    state["rules"] = default.rules

    # Whisper takes the glossary at decode time, which beats repairing the text
    # afterwards, and only the DEFAULT profiles can be baked in here because
    # faster-whisper takes its hotwords when the model is constructed. A
    # profile a request names reaches the decoder through asr.Options instead,
    # on the same argument, so per-request selection is not second-class — see
    # openai_api._glossary.
    #
    # Nothing is baked in for Parakeet, and that is deliberate rather than a
    # gap. Its biasing is compiled per request from asr.Options.vocabulary and
    # is off unless the request asked, because a deployment-wide always-on
    # boost list is the shape the +12% measurement above rules out. A
    # deployment that wants it anyway sets STT_BOOST=1, which changes the
    # DEFAULT the route applies rather than bypassing the route.
    hotwords = default.hotwords if HOTWORDS_ENABLED else None

    state["asr"] = asr.build(THREADS, hotwords)

    if VAD_ENABLED:
        state["vad"] = vad.Vad()
        log.info("vad ready")


def stop() -> None:
    state.clear()


def loaded() -> object | None:
    """The recogniser, or None while it is still loading. For /health."""
    return state.get("asr")


def registry() -> profiles.Registry:
    """The glossary profiles this process serves.

    Built at startup, but never frozen there: the four /glossaries routes and
    every request that names a profile call refresh() first, which rescans only
    when a stat() says a file changed. "Load once at boot" was the shape that
    made per-request selection meaningless — changing one term needed a new
    container.
    """
    existing = state.get("glossaries")
    if existing is None:
        # Only reachable before the lifespan has run — the /glossaries routes
        # are as useful with no model loaded as with one, and refusing them
        # with a 503 about the RECOGNISER would be answering a question nobody
        # asked. A registry built here is replaced by start()'s.
        existing = profiles.load_registry()
        state["glossaries"] = existing
    return existing  # type: ignore[return-value]


def default_rules() -> list[tuple[re.Pattern[str], str]]:
    """Rules for a request that selected no profile. Usually empty."""
    return state.get("rules") or []  # type: ignore[return-value]


def engine() -> asr.Parakeet | asr.Whisper:
    """The loaded recogniser. The /v1 route reads its capability flags to
    decide between honouring a field and refusing it by name."""
    if "asr" not in state:
        raise HTTPException(503, "model still loading")
    return state["asr"]  # type: ignore[return-value]


def acquire() -> None:
    """Take one recogniser slot, or raise Busy. A no-op unless configured.

    Split from the context manager because a stream outlives the function that
    started it: the slot is taken before the 200 goes out, so a refusal is
    still a 429 with a status line, and released in the generator's finally.
    """
    if _slots is not None and not _slots.acquire(blocking=False):
        raise Busy


def release() -> None:
    if _slots is not None:
        _slots.release()


@contextlib.contextmanager
def slot() -> Iterator[None]:
    """Hold one recogniser slot for the duration of a buffered request."""
    acquire()
    try:
        yield
    finally:
        release()


def decode(data: bytes, *, allow_resample: bool) -> np.ndarray:
    """Bytes to 16 kHz mono float32.

    `allow_resample` is False for the native route and True for /v1, and the
    difference is deliberate. /transcribe has refused anything but 16 kHz since
    it existed, on the grounds that a client sending 44.1 kHz should find out
    rather than be quietly resampled; that is a documented contract with
    clients this service already has. No OpenAI client anticipates it — a
    44.1 kHz mp3 is the ordinary case there — so the compatibility route
    resamples, which is what the specification's nine input formats require.
    """
    try:
        decoded = audio.decode(data)
    except audio.AudioError as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc

    if not allow_resample and decoded.source_rate != SAMPLE_RATE:
        raise HTTPException(
            400,
            f"expected {SAMPLE_RATE} Hz, got {decoded.source_rate} Hz — "
            "resample before sending",
        )
    return decoded.samples


def _speech(samples: np.ndarray, tuning: Tuning) -> vad.Speech:
    detector = state.get("vad")
    if detector is None or not tuning.enabled:
        return vad.whole(samples)
    return detector.speech_only(  # type: ignore[attr-defined]
        samples,
        threshold=tuning.threshold,
        min_silence_ms=tuning.min_silence_ms,
        speech_pad_ms=tuning.speech_pad_ms,
    )


def _compression_ratio(text: str) -> float:
    """Whisper's own measure, computed the same way for the other engine.

    Text over its zlib length. Above about 2.4 the decoder is repeating itself,
    which is what the field is for.
    """
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def _map_word(word: asr.Word, speech: vad.Speech,
              bounds: tuple[float, float] | None = None) -> asr.Word:
    start = speech.original(word.start)
    end = speech.original(word.end)
    if bounds is not None:
        low, high = bounds
        start = min(max(start, low), high)
        end = min(max(end, start), high)
    return asr.Word(word=word.word, start=start, end=max(end, start),
                    logprob=word.logprob)


def _segments_from_words(words: tuple[asr.Word, ...], speech: vad.Speech,
                         rules) -> tuple[tuple[asr.Segment, ...], tuple[asr.Word, ...]]:  # noqa: ANN001
    """Cut a word stream into segments at the VAD's own boundaries.

    The engine that needs this reports no segments of its own, so the choice is
    between the pauses the VAD already found and an arbitrary length in
    seconds. The pauses are real, so they win.

    Two of the ten required fields have no equivalent on this engine and are
    written down here rather than left to a reader to discover:

      seek            a Whisper 30 s window offset. There are no windows here;
                      the encoder runs over the whole waveform. Always 0.
      no_speech_prob  a Whisper decoder output. A TDT decoder produces none.
                      Always 0.0 — and note the VAD has already removed what
                      it believed was silence, so a segment reaching this
                      function is speech by construction.

    `tokens` is empty for the same class of reason: onnx-asr maps token ids to
    strings inside its decoder and returns only the strings, and an array of
    invented integers would be worse than an empty one.
    """
    # Boundaries of each span in the compacted timeline the words are timed in.
    edges: list[float] = [0.0]
    for span_start, span_end in speech.spans:
        edges.append(edges[-1] + (span_end - span_start) / SAMPLE_RATE)

    buckets: list[list[asr.Word]] = [[] for _ in speech.spans]
    for word in words:
        # Assigned on the END of the word: that is the emission time, the one
        # the decoder actually reported. A word's start is borrowed from the
        # token before it and can sit a hair inside the previous run.
        index = 0
        while index + 1 < len(speech.spans) and word.end > edges[index + 1]:
            index += 1
        buckets[index].append(word)

    segments: list[asr.Segment] = []
    mapped: list[asr.Word] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        low = speech.spans[index][0] / SAMPLE_RATE
        high = speech.spans[index][1] / SAMPLE_RATE
        placed = [_map_word(word, speech, (low, high)) for word in bucket]
        repaired = tuple(
            asr.Word(word=glossary.apply(word.word, rules)[0], start=word.start,
                     end=word.end, logprob=word.logprob)
            for word in placed
        )
        mapped.extend(repaired)
        # Joining with single spaces reproduces the decoder's own text: a token
        # that starts a word carries the space, punctuation does not.
        text = " ".join(word.word for word in repaired)
        logprobs = [word.logprob for word in bucket]
        segments.append(asr.Segment(
            id=len(segments),
            seek=0,
            start=repaired[0].start,
            end=repaired[-1].end,
            text=text,
            tokens=(),
            temperature=0.0,
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            compression_ratio=_compression_ratio(text),
            no_speech_prob=0.0,
            words=repaired,
        ))
    return tuple(segments), tuple(mapped)


def _place_segments(segments: tuple[asr.Segment, ...], speech: vad.Speech,
                    rules) -> tuple[tuple[asr.Segment, ...], tuple[asr.Word, ...]]:  # noqa: ANN001
    """Move an engine's own segments onto the client's timeline, and repair.

    The glossary is applied to each segment and each word as well as to the
    whole transcript. A rule that spans a boundary — "cloud code" split across
    two segments, or across two words — fires in `text` and cannot fire in the
    smaller unit; that is a property of doing the repair on strings, and it is
    named in the README rather than hidden.
    """
    placed: list[asr.Segment] = []
    words: list[asr.Word] = []
    for index, segment in enumerate(segments):
        segment_words = tuple(
            asr.Word(word=glossary.apply(word.word, rules)[0],
                     start=speech.original(word.start),
                     end=speech.original(word.end),
                     logprob=word.logprob)
            for word in segment.words
        )
        words.extend(segment_words)
        text = glossary.apply(segment.text, rules)[0]
        placed.append(asr.Segment(
            # Renumbered from 0, not passed through. faster-whisper increments
            # before it yields (transcribe.py:1352), so its first segment is
            # id 1, while the specification's own verbose_json example starts
            # at 0 and so does the engine that has its segments cut here. One
            # service answering the same field 0-based or 1-based depending on
            # a startup env var is exactly the kind of difference a client
            # indexes by and only discovers on the other deployment.
            id=index,
            seek=segment.seek,
            start=speech.original(segment.start),
            end=speech.original(segment.end),
            text=text,
            tokens=segment.tokens,
            temperature=segment.temperature,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
            no_speech_prob=segment.no_speech_prob,
            words=segment_words,
        ))
    return tuple(placed), tuple(words)


def run(data: bytes, opts: asr.Options | None = None, *,
        allow_resample: bool = False, tuning: Tuning | None = None,
        rules: list[tuple[re.Pattern[str], str]] | None = None) -> Result:
    """Transcribe one clip. Blocking CPU work — never call this on the loop.

    `rules` is this request's compiled glossary, from the profiles it selected.
    None means the deployment default, which is empty unless
    STT_GLOSSARY_DEFAULT names profiles — an unselected request gets no
    repair, which is the specification's behaviour and the measured one.

    `opts.hotwords` and `opts.vocabulary` are one vocabulary in the two shapes
    the two decoders take, and both are extended for this request. Whisper
    reads the joined string as hotwords. Parakeet compiles the tuple into a
    boosting automaton and fuses it into its TDT decoding loop, but only when
    `opts.boost` is set — see boosting.py. This docstring used to say Parakeet
    "has no mechanism for a vocabulary"; that was true of onnx-asr's argument
    list and false about the decoder.

    STT_HOTWORDS=0 outranks the request: with biasing switched off, a request
    carrying a vocabulary is transcribed without it.
    """
    model = engine()
    opts = opts or asr.Options()
    tuning = tuning or Tuning()

    # The off switch is absolute, not a default a request can talk its way
    # past. Joining a request's prompt into the decoder's vocabulary here would
    # hand decode-time biasing back to a run configured to measure the model
    # without it — silently, and only on the requests that carried a prompt,
    # which is the worst way for a benchmark to be wrong. The route refuses the
    # field when biasing is off; this is the second lock on the same door.
    #
    # ALL THREE FIELDS, not just hotwords. Clearing the Whisper half and
    # leaving the Parakeet half would make STT_HOTWORDS=0 a half-open door on
    # the DEFAULT engine, which is worse than not having the switch: a
    # benchmark would believe it had measured the model.
    if not HOTWORDS_ENABLED and (opts.hotwords or opts.vocabulary or opts.boost):
        opts = replace(opts, hotwords=None, vocabulary=(), boost=False)

    samples = decode(data, allow_resample=allow_resample)
    if samples.size == 0:
        raise HTTPException(400, "audio contains no samples")
    audio_seconds = samples.size / SAMPLE_RATE

    started = time.monotonic()
    speech = _speech(samples, tuning)
    speech_seconds = speech.samples.size / SAMPLE_RATE

    recognition = model.transcribe(speech.samples, opts)
    if rules is None:
        rules = default_rules()
    text, repaired = glossary.apply(recognition.text, rules)  # type: ignore[arg-type]

    if recognition.segments is not None:
        segments, words = _place_segments(recognition.segments, speech, rules)
    else:
        segments, words = _segments_from_words(recognition.words, speech, rules)
    compute = time.monotonic() - started

    log.info("%.1fs audio, %.1fs speech, %.2fs compute (%.1fx), repaired=%s, "
             "boosted=%s", audio_seconds, speech_seconds, compute,
             audio_seconds / compute if compute else 0.0, repaired or "none",
             ", ".join(recognition.boosted) or "none")

    return Result(
        text=text,
        raw=recognition.text,
        repaired=repaired,
        model=MODEL,
        audio_seconds=round(audio_seconds, 2),
        speech_seconds=round(speech_seconds, 2),
        compute_seconds=round(compute, 2),
        realtime_factor=round(audio_seconds / compute, 1) if compute else 0.0,
        language=recognition.language,
        segments=segments,
        words=words,
        logprobs=recognition.logprobs,
        task=opts.task,
        boosted=recognition.boosted,
    )


@dataclass
class Stream:
    """A transcription in progress. `deltas` yields text as it is decoded."""

    audio_seconds: float
    deltas: Iterator[str]
    text: str = ""
    # Filled in as the run proceeds, for the log line at the end.
    stats: dict[str, float] = field(default_factory=dict)


def open_stream(data: bytes, opts: asr.Options, *, allow_resample: bool = True,
                tuning: Tuning | None = None,
                rules: list[tuple[re.Pattern[str], str]] | None = None) -> Stream:
    """Start a streaming transcription. Decoding and VAD happen up front.

    Only the engine that can genuinely emit before it finishes reaches here —
    the route refuses stream=true on the other one by name rather than
    delivering one delta at the end and calling it a stream.
    """
    model = engine()
    tuning = tuning or Tuning()
    if not HOTWORDS_ENABLED and (opts.hotwords or opts.vocabulary or opts.boost):
        opts = replace(opts, hotwords=None, vocabulary=(), boost=False)

    samples = decode(data, allow_resample=allow_resample)
    if samples.size == 0:
        raise HTTPException(400, "audio contains no samples")
    audio_seconds = samples.size / SAMPLE_RATE

    started = time.monotonic()
    speech = _speech(samples, tuning)
    if rules is None:
        rules = default_rules()
    stream = Stream(audio_seconds=audio_seconds, deltas=iter(()))

    def deltas() -> Iterator[str]:
        first = True
        for segment in model.stream(speech.samples, opts):
            # The repair runs per segment, so that concatenating every delta
            # reproduces the terminal event's text exactly. A rule spanning a
            # segment boundary cannot fire — the alternative is a `done` text
            # that differs from the deltas the client already rendered.
            piece, _ = glossary.apply(segment.text, rules)  # type: ignore[arg-type]
            if first:
                piece = piece.lstrip()
                stream.stats["first_delta"] = time.monotonic() - started
                first = False
            if not piece:
                continue
            stream.text += piece
            yield piece
        stream.stats["compute"] = time.monotonic() - started
        log.info("%.1fs audio streamed in %.2fs (first delta %.2fs)",
                 audio_seconds, stream.stats["compute"],
                 stream.stats.get("first_delta", stream.stats["compute"]))

    stream.deltas = deltas()
    return stream
