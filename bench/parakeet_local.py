"""Local Parakeet backend for the benchmark, via FluidAudio's CLI.

The HTTP path in bench.py talks to stt-stack. This one shells out to
fluidaudiocli on the Mac, so Parakeet is measured on the Neural Engine it
actually runs on — and with the CTC vocabulary boosting Ghost Pepper links but
does not expose.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


class ParakeetLocal:
    def __init__(self, cli: str, vocab: str | None, language: str = "pt") -> None:
        self.cli = cli
        self.vocab = vocab
        self.language = language

    def transcribe(self, samples: np.ndarray) -> dict:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            out = Path(td) / "r.json"
            sf.write(wav, samples, SAMPLE_RATE)
            cmd = [self.cli, "transcribe", str(wav), "--model-version", "v3",
                   "--language", self.language, "--output-json", str(out)]
            if self.vocab:
                cmd += ["--custom-vocab", self.vocab]
            subprocess.run(cmd, capture_output=True, check=False)
            if not out.is_file():
                return {"text": "", "realtime_factor": 0.0}
            d = json.load(open(out))
        return {"text": d.get("text", ""), "realtime_factor": d.get("rtfx", 0.0)}
