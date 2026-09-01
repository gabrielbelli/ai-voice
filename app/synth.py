"""Chatterbox synthesis, CPU.

Slow by nature. Measured on an M2 Max at 4, 8 and 16 threads it held around
0.21x realtime with under 5% spread — it is not thread-bound, because
autoregressive token generation is sequential and cores cannot parallelise it.
Peak RSS was 6.5-6.8 GB.

So this is a batch service, not an interactive one. A ten-minute recording
takes roughly three quarters of an hour to produce. It exists because Kokoro,
which is twenty times faster and twenty times lighter, is a small model and
sounds like one on long-form material.

The model is loaded lazily and unloaded after an idle timeout, because 6.5 GB
resident is not something to leave sitting on a shared host between jobs.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
from voice_common.audio import SAMPLE_RATE, splice

log = logging.getLogger("tts-long.synth")

# Re-exported so app.main keeps importing it from here. The constant itself is
# voice_common.audio's: 24 kHz is OpenAI's headerless `pcm` rate and the native
# rate of both Chatterbox and Kokoro, so it is a wire fact rather than a
# property of this model file.
__all__ = ["SAMPLE_RATE", "Synth"]


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
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t = time.monotonic()
        log.info("loading chatterbox on cpu, %d threads", self.threads)
        self._model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        log.info("loaded in %.0fs", time.monotonic() - t)

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

    def speak(self, text: str, language: str, exaggeration: float,
              cfg_weight: float, temperature: float,
              reference: str | None = None) -> np.ndarray:
        with self._lock:
            self._ensure_loaded()
            wav = self._model.generate(  # type: ignore[union-attr]
                text,
                language_id=language,
                audio_prompt_path=reference,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            self._last_used = time.monotonic()
        return wav.squeeze().detach().cpu().numpy().astype(np.float32)

    def speak_segments(self, segments: list[tuple[str, float]], language: str,
                       exaggeration: float, cfg_weight: float,
                       temperature: float,
                       reference: str | None = None) -> np.ndarray:
        """Synthesise each segment, inserting real silence between them.

        Pauses are generated here rather than asked of the model. No TTS model
        reliably produces a beat you can act inside — punctuation buys a
        breath, an instruction needs a gap.

        The splicing itself is voice_common.audio.splice, which tts-stack also
        calls: a segment whose text is empty contributes its pause and nothing
        else, and a request of nothing but pauses returns silence rather than
        raising.
        """
        return splice([
            (self.speak(text, language, exaggeration, cfg_weight, temperature,
                        reference) if text.strip() else None,
             pause_after)
            for text, pause_after in segments])
