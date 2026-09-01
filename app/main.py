"""Self-hosted text-to-speech. Kokoro, CPU only.

    text or segments -> phonemise -> Kokoro -> wav/opus

There is one model and it stays resident: at 330 MB and 4x realtime on CPU it
is too cheap to unload. Chatterbox, the long-form alternative, is deliberately
not here — it needs 5.3 GB and runs below realtime, so it belongs behind a
separate service that can be started on demand.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .synth import SAMPLE_RATE, Synth

MODEL_DIR = Path(os.getenv("TTS_MODEL_DIR", "/models"))
MODEL_URL = os.getenv(
    "TTS_MODEL_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
VOICES_URL = os.getenv(
    "TTS_VOICES_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
DEFAULT_VOICE = os.getenv("TTS_VOICE", "bm_george")
DEFAULT_LANG = os.getenv("TTS_LANGUAGE", "en-us")
THREADS = int(os.getenv("TTS_THREADS", "4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts-stack")

state: dict[str, object] = {}


def _fetch(url: str, dest: Path) -> None:
    if dest.is_file():
        return
    log.info("downloading %s", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSfL", "-o", str(tmp), url], check=True)
    tmp.rename(dest)  # rename last, so a killed download never looks complete


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    started = time.monotonic()
    model = MODEL_DIR / "kokoro.onnx"
    voices = MODEL_DIR / "voices.bin"
    _fetch(MODEL_URL, model)
    _fetch(VOICES_URL, voices)
    state["synth"] = Synth(str(model), str(voices))
    log.info("ready in %.1fs, %d threads", time.monotonic() - started, THREADS)
    yield
    state.clear()


app = FastAPI(title="tts-stack",
              description="Kokoro text-to-speech, CPU only.",
              lifespan=lifespan)


class Segment(BaseModel):
    text: str
    pause_after: float = Field(default=0.0, ge=0.0, le=10.0)


class SpeakRequest(BaseModel):
    text: str | None = None
    segments: list[Segment] | None = None
    voice: str | None = None
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="wav", pattern="^(wav|opus)$")


@app.get("/health")
def health() -> dict[str, object]:
    s = state.get("synth")
    return {"status": "ok" if s else "loading",
            "voices": len(getattr(s, "voices", [])),
            "default_voice": DEFAULT_VOICE,
            "threads": THREADS}


@app.get("/voices")
def voices() -> dict[str, object]:
    s = state.get("synth")
    if not s:
        raise HTTPException(503, "model still loading")
    all_voices = s.voices  # type: ignore[attr-defined]
    # Kokoro encodes locale in the prefix: p = Portuguese, a/b = US/UK English.
    return {"voices": all_voices,
            "pt_br": [v for v in all_voices if v.startswith(("pf_", "pm_"))],
            "en_us": [v for v in all_voices if v.startswith(("af_", "am_"))],
            "en_gb": [v for v in all_voices if v.startswith(("bf_", "bm_"))]}


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        return buf.getvalue(), "audio/wav"
    with tempfile.TemporaryDirectory() as td:
        wav, opus = Path(td) / "a.wav", Path(td) / "a.opus"
        sf.write(wav, audio, SAMPLE_RATE)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-b:a", "32k", str(opus)], check=True)
        return opus.read_bytes(), "audio/ogg"


# Deliberately `def`, not `async def`: synthesis is blocking CPU work, and on
# the event loop it would starve /health during any sustained load.
@app.post("/speak")
def speak(req: SpeakRequest) -> Response:
    synth = state.get("synth")
    if not synth:
        raise HTTPException(503, "model still loading")
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")

    voice = req.voice or DEFAULT_VOICE
    if voice not in synth.voices:  # type: ignore[attr-defined]
        raise HTTPException(400, f"unknown voice {voice!r}; see GET /voices")
    language = req.language or DEFAULT_LANG

    started = time.monotonic()
    try:
        if req.segments:
            audio = synth.speak_segments(  # type: ignore[attr-defined]
                [(s.text, s.pause_after) for s in req.segments],
                voice, language, req.speed)
        else:
            audio = synth.speak(req.text or "", voice, language, req.speed)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(500, f"synthesis failed: {exc}") from exc

    compute = time.monotonic() - started
    duration = audio.size / SAMPLE_RATE
    data, mime = _encode(audio, req.format)
    log.info("%.1fs audio in %.2fs (%.1fx) voice=%s",
             duration, compute, duration / compute if compute else 0.0, voice)

    return Response(content=data, media_type=mime, headers={
        "X-Audio-Seconds": f"{duration:.2f}",
        "X-Compute-Seconds": f"{compute:.2f}",
        "X-Realtime-Factor": f"{duration / compute:.1f}" if compute else "0",
    })
