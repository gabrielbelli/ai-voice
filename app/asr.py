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
it is the only one of the two that accepts a vocabulary at decode time.

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
  vocabulary      Whisper accepts hotwords at decode time. Parakeet's TDT
                  decoder has no vocabulary argument at all; the glossary is
                  post-decode repair there, which cannot recover a word the
                  acoustic model never approached.
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
from dataclasses import dataclass

import numpy as np

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
    hotwords: str | None = None
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


class Parakeet:
    """Parakeet TDT via ONNX Runtime. CTC/TDT, so no decode-time vocabulary.

    Terms are repaired after decoding instead (see glossary.py). That is
    weaker than biasing — it cannot recover a word the acoustic model never
    approached — but it is what this runtime offers. FluidAudio implements
    real CTC boosting for the same model, and is CoreML-only.
    """

    name = "parakeet"
    accepts_vocabulary = False
    reports_segments = False
    accepts_language = False
    accepts_temperature = False
    can_translate = False
    can_stream = False
    reports_language = False
    reports_token_logprobs = True
    reports_token_ids = False

    def __init__(self, model_id: str, quantisation: str) -> None:
        import onnx_asr  # noqa: PLC0415

        self._model = onnx_asr.load_model(model_id, quantization=quantisation)

    def transcribe(self, samples: np.ndarray, opts: Options) -> Recognition:
        # Parakeet v3 detects language itself and takes no hint, and its TDT
        # decoder has no vocabulary and no temperature. The route refuses those
        # fields by name before reaching here, so anything still set would be a
        # routing bug rather than something to swallow quietly.
        if not opts.want_detail:
            # The plain adapter, for the default response_format. Asking for
            # timestamps costs about 5% on a 14.2 s clip here (5.07 s against
            # 5.34 s), which is not worth paying on every dictation request for
            # numbers no one reads.
            text = self._model.recognize(samples, sample_rate=SAMPLE_RATE).strip()
            return Recognition(text=text)

        result = self._model.with_timestamps().recognize(
            samples, sample_rate=SAMPLE_RATE)
        words, logprobs = _parakeet_words(result)
        return Recognition(
            text=result.text.strip(),
            words=words,
            logprobs=logprobs if opts.want_logprobs else None,
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
    """Whisper via CTranslate2. Accepts hotwords at decode time."""

    name = "whisper"
    accepts_vocabulary = True
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
        log.info("parakeet ready: %s (no decode-time vocabulary)", model_id)
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
