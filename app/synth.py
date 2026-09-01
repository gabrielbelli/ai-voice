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
from voice_common.audio import SAMPLE_RATE, check_rate, splice

log = logging.getLogger("tts-stack.synth")

# Re-exported so app.main keeps importing the rate from the module that
# produces the audio, rather than reaching past it into the shared package.
__all__ = ["SAMPLE_RATE", "Synth"]


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
        # kokoro is 24 kHz; a model that changed it would otherwise ship every
        # file with the wrong rate in its header and play at the wrong pitch.
        check_rate(rate)
        return audio.astype(np.float32)

    def speak_segments(self, segments: list[tuple[str, float, str]],
                       language: str, speed: float) -> np.ndarray:
        """Synthesise each (text, pause_after, voice) and splice in silence.

        Pauses are generated here rather than asked of the model, because no
        TTS model reliably produces a beat you can act inside. Punctuation
        buys a breath; an instruction needs a gap. Measured by ear on the same
        voice and words, inserted silence is what separates audio that sounds
        like instructions from audio that sounds like narration.

        The voice arrives per segment, already resolved by the caller. It used
        to be one voice for the whole call, which quietly made a documented
        per-segment `voice` a lie; there was never a cost to honouring it, a
        voice being a 510 KB embedding over weights that are already resident.

        The splicing itself is voice_common.audio.splice: an empty segment
        contributes its pause and no audio, and a request of nothing but
        pauses returns zeros(0) rather than raising, exactly as this did.
        """
        return splice([(self.speak(text, voice, language, speed)
                        if text.strip() else None, pause_after)
                       for text, pause_after, voice in segments])
