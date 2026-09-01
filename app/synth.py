"""Kokoro speech synthesis.

One model, 82M parameters, 310 MB of shared weights. A voice is a separate
510 KB embedding tensor — so switching voices costs nothing after load, and a
request may use a different voice per segment if it wants. Only swapping the
model itself is expensive, and there is only one model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("tts-stack.synth")

SAMPLE_RATE = 24_000


def _wire_espeak() -> None:
    """Point the phonemiser at the system espeak-ng.

    The espeakng-loader wheel hardcodes a path from its own build machine
    (/Users/runner/work/...), which of course does not exist anywhere else.
    Left alone it fails at first synthesis, not at import, so the service
    starts healthy and then breaks on the first request.
    """
    data = os.getenv("ESPEAK_DATA_PATH", "/usr/lib/x86_64-linux-gnu/espeak-ng-data")
    lib = os.getenv("ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
    if not Path(data).is_dir():
        for candidate in ("/usr/share/espeak-ng-data",
                          "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
                          "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"):
            if Path(candidate).is_dir():
                data = candidate
                break
    if not Path(lib).is_file():
        for candidate in ("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
                          "/usr/lib/libespeak-ng.so.1",
                          "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"):
            if Path(candidate).is_file():
                lib = candidate
                break

    import espeakng_loader
    espeakng_loader.get_data_path = lambda: data
    espeakng_loader.get_library_path = lambda: lib
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_data_path(data)
    EspeakWrapper.set_library(lib)
    log.info("espeak-ng: %s", lib)


class Synth:
    def __init__(self, model_path: str, voices_path: str) -> None:
        _wire_espeak()
        from kokoro_onnx import Kokoro
        self._k = Kokoro(model_path, voices_path)
        self.voices = sorted(self._k.get_voices())
        log.info("kokoro ready, %d voices", len(self.voices))

    def speak(self, text: str, voice: str, language: str,
              speed: float) -> np.ndarray:
        audio, rate = self._k.create(text, voice=voice, speed=speed, lang=language)
        if rate != SAMPLE_RATE:  # kokoro is 24 kHz; guard against a change
            raise RuntimeError(f"unexpected sample rate {rate}")
        return audio.astype(np.float32)

    def speak_segments(self, segments: list[tuple[str, float]], voice: str,
                       language: str, speed: float) -> np.ndarray:
        """Synthesise each segment and insert real silence between them.

        Pauses are generated here rather than asked of the model, because no
        TTS model reliably produces a beat you can act inside. Punctuation
        buys a breath; an instruction needs a gap. Measured by ear on the same
        voice and words, inserted silence is what separates audio that sounds
        like instructions from audio that sounds like narration.
        """
        parts: list[np.ndarray] = []
        for text, pause_after in segments:
            if text.strip():
                parts.append(self.speak(text, voice, language, speed))
            if pause_after > 0:
                parts.append(np.zeros(int(SAMPLE_RATE * pause_after), dtype=np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)
