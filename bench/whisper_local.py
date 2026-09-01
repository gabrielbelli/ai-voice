"""Local Whisper backend, so the Mac can carry part of the matrix.

Same model and settings as stt-stack runs on orko (large-v3, int8,
beam 5, VAD off because the harness degrades audio itself). Threads are
capped below the core count so a benchmark does not saturate the machine it
is measuring — contention would show up as a worse rtf and be mistaken for
the engine being slow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


class WhisperLocal:
    def __init__(self, model: str = "large-v3", threads: int = 6,
                 language: str | None = None, hotwords: str | None = None) -> None:
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device="cpu", compute_type="int8",
                              cpu_threads=threads)
        self.language = language
        self.hotwords = hotwords

    def transcribe(self, samples: np.ndarray) -> dict:
        import time
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            sf.write(wav, samples, SAMPLE_RATE)
            t = time.monotonic()
            segs, _ = self.m.transcribe(
                str(wav), language=self.language, beam_size=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                condition_on_previous_text=False, vad_filter=False,
                hotwords=self.hotwords)
            text = " ".join(s.text.strip() for s in segs).strip()
            el = time.monotonic() - t
        dur = samples.size / SAMPLE_RATE
        return {"text": text, "realtime_factor": round(dur / el, 2) if el else 0.0}
