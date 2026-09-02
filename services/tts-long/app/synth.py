"""Chatterbox synthesis, CPU.

Slow by nature. Measured on an M2 Max at 4, 8 and 16 threads it held around
0.21x realtime with under 5% spread — it is not thread-bound, because
autoregressive token generation is sequential and cores cannot parallelise it.
Peak RSS was 6.5-6.8 GB. The deployed instance re-measured at 0.217x on
2026-09-01, which is the same number a year later.

So this is a batch service, not an interactive one. A ten-minute recording
takes roughly three quarters of an hour to produce. It exists because Kokoro,
which is twenty times faster and twenty times lighter, is a small model and
sounds like one on long-form material.

The model is loaded lazily and unloaded after an idle timeout, because 6.5 GB
resident is not something to leave sitting on a shared host between jobs.

**Nothing here is handed more than one chunk of text.** generate() stops after
1000 speech tokens, which is 40 seconds of audio, and says nothing when it
does — see app/chunking.py for the measurement. Splitting is the caller's job;
this module's job is one piece at a time, plus the silence between them.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from voice_common.audio import SAMPLE_RATE, check_rate, splice

log = logging.getLogger("tts-long.synth")

# Re-exported so app.main keeps importing it from here. The constant itself is
# voice_common.audio's: 24 kHz is OpenAI's headerless `pcm` rate and the native
# rate of both Chatterbox and Kokoro, so it is a wire fact rather than a
# property of this model file.
__all__ = ["SAMPLE_RATE", "SPEECH_TOKEN_RATE", "SUPPORTED_LANGUAGES", "Spoken",
           "Synth", "speech_tokens"]

# chatterbox/models/s3tokenizer/s3tokenizer.py:18 — S3_TOKEN_RATE = 25. One
# speech token is 40 ms of audio, which is what makes `output_tokens` in the
# SSE `speech.audio.done` event a count rather than a guess, and what puts
# generate()'s max_new_tokens=1000 exactly 40 seconds from the start.
SPEECH_TOKEN_RATE = 25

# chatterbox/mtl_tts.py:24, SUPPORTED_LANGUAGES. Copied rather than imported
# because importing it drags in torch, and this list is needed to answer a
# request BEFORE anything is loaded: generate() raises ValueError on an
# unsupported language_id, and finding that out in the worker means the caller
# waited in a queue to be told about a typo. Cross-checked against the model
# when it loads, so a chatterbox upgrade that adds a language is a log line
# rather than a mystery 400.
SUPPORTED_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")


def speech_tokens(samples: int) -> int:
    """Speech tokens behind `samples` of audio, at the model's own token rate.

    s3gen turns one 25 Hz speech token into 40 ms of 24 kHz audio, so the
    length of what came back is the count of what was generated.
    """
    return int(round(samples / SAMPLE_RATE * SPEECH_TOKEN_RATE))


@dataclass
class Spoken:
    """Audio, and what it cost in the model's own units."""

    audio: np.ndarray
    input_tokens: int


def _stub_watermarker() -> None:
    """Neutralise resemble-perth.

    It imports pkg_resources, removed in Python 3.14, and it only stamps an
    inaudible watermark. Stubbing keeps the dependency from deciding which
    interpreter this image runs.
    """
    import perth

    class _NoWatermark:
        def apply_watermark(self, wav, sample_rate=None, **kw):  # noqa: ANN001
            return wav

    perth.PerthImplicitWatermarker = _NoWatermark


class Synth:
    """Loads on first use, unloads after `idle_timeout` seconds of quiet."""

    def __init__(self, idle_timeout: float = 600.0, threads: int = 8) -> None:
        self.idle_timeout = idle_timeout
        self.threads = threads
        self._model = None
        self._last_used = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._reaper, daemon=True).start()

    # ---------------------------------------------------------------- load --

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _stub_watermarker()
        os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
        import torch

        torch.set_num_threads(self.threads)
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as MODEL_LANGUAGES
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t = time.monotonic()
        log.info("loading chatterbox on cpu, %d threads", self.threads)
        self._model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        log.info("loaded in %.0fs", time.monotonic() - t)

        # The list this service validates requests against is a copy, so say
        # so the moment the real one disagrees rather than answering 400 to a
        # language the model would have accepted.
        drift = set(MODEL_LANGUAGES) ^ set(SUPPORTED_LANGUAGES)
        if drift:
            log.warning("chatterbox language list has moved: %s. Requests are "
                        "validated against app/synth.py's copy, which needs "
                        "updating.", ", ".join(sorted(drift)))
        # 24 kHz is asserted in every wav header this service writes and in the
        # `pcm` contract. A model update that changed it would ship every file
        # at the wrong pitch, playable and wrong, with nothing reporting an
        # error. voice_common.audio.check_rate exists for exactly this and had
        # never been called from here.
        check_rate(int(getattr(self._model, "sr", SAMPLE_RATE)))

    def _reaper(self) -> None:
        while not self._stop.wait(30.0):
            with self._lock:
                idle = time.monotonic() - self._last_used
                if self._model is not None and self._last_used and idle > self.idle_timeout:
                    log.info("unloading after %.0fs idle", idle)
                    self._model = None
                    import gc

                    gc.collect()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- generate --

    def _count_tokens(self, text: str, language: str) -> int:
        """Text tokens for `text`, from the model's own tokeniser.

        Called with the lock held and the model loaded. `usage.input_tokens` in
        the SSE done event is required to be an integer, and the only honest
        integer is the one the model actually tokenised — generate() runs
        punc_norm first, so this does too.

        The fallback is four characters to a token, the ratio every OpenAI
        tokeniser lands near on English prose. It is an approximation and it is
        logged as one; it is reached only if a chatterbox upgrade moves the
        tokeniser, and reporting a slightly wrong count beats failing a
        synthesis that already succeeded.
        """
        try:
            from chatterbox.mtl_tts import punc_norm

            tokens = self._model.tokenizer.text_to_tokens(  # type: ignore[union-attr]
                punc_norm(text),
                language_id=language.lower() if language else None)
            return int(tokens.shape[-1])
        except Exception:  # noqa: BLE001 - usage must not fail a synthesis
            log.warning("tokeniser unavailable; input_tokens is approximated "
                        "from character count", exc_info=True)
            return max(1, math.ceil(len(text) / 4))

    def speak(self, text: str, language: str, exaggeration: float,
              cfg_weight: float, temperature: float,
              reference: str | None = None) -> np.ndarray:
        """One chunk. Anything over 40 seconds of speech is truncated: chunk it."""
        return self._speak(text, language, exaggeration, cfg_weight,
                           temperature, reference).audio

    def _speak(self, text: str, language: str, exaggeration: float,
               cfg_weight: float, temperature: float,
               reference: str | None) -> Spoken:
        with self._lock:
            self._ensure_loaded()
            tokens = self._count_tokens(text, language)
            wav = self._model.generate(  # type: ignore[union-attr]
                text,
                language_id=language,
                audio_prompt_path=reference,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            self._last_used = time.monotonic()
        audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return Spoken(audio=audio, input_tokens=tokens)

    def speak_segments(self, segments: list[tuple[str, float]], language: str,
                       exaggeration: float, cfg_weight: float,
                       temperature: float,
                       reference: str | None = None,
                       on_chunk: Callable[[np.ndarray], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Spoken:
        """Synthesise each segment, inserting real silence between them.

        Pauses are generated here rather than asked of the model. No TTS model
        reliably produces a beat you can act inside — punctuation buys a
        breath, an instruction needs a gap.

        The splicing itself is voice_common.audio.splice, which tts-stack also
        calls: a segment whose text is empty contributes its pause and nothing
        else, and a request of nothing but pauses returns silence rather than
        raising. It is applied per segment here so the piece handed to
        `on_chunk` is exactly the piece that lands in the finished array —
        concatenating what a stream emitted and splicing the whole list are the
        same bytes.

        `on_chunk` is called from this thread, in order, as each segment
        finishes: that call is the only reason a stream can start before the
        whole job does. `cancelled` is polled between segments, which is as
        fine-grained as cancellation gets — generate() has no interruption
        point inside it.
        """
        parts: list[np.ndarray] = []
        total_tokens = 0
        for text, pause_after in segments:
            if cancelled is not None and cancelled():
                log.info("cancelled after %d of %d segments",
                         len(parts), len(segments))
                break
            if text.strip():
                spoken = self._speak(text, language, exaggeration, cfg_weight,
                                     temperature, reference)
                total_tokens += spoken.input_tokens
                piece = splice([(spoken.audio, pause_after)])
            else:
                piece = splice([(None, pause_after)])
            if piece.size:
                parts.append(piece)
                if on_chunk is not None:
                    on_chunk(piece)
        audio = (np.concatenate(parts) if parts
                 else np.zeros(0, dtype=np.float32))
        return Spoken(audio=audio, input_tokens=total_tokens)
