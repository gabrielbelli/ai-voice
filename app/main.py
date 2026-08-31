"""CPU-only speech-to-text over HTTP.

One endpoint, one model, no cleanup stage. The transcript is returned raw
apart from glossary repair — deliberately. A local LLM small enough to sit
beside this on a CPU box is not reliable enough to rewrite a technical
transcript: it summarises, inverts meaning, and leaks its own reasoning. The
consumer of this text (an editor, an agent, a person) is better placed to
resolve ambiguity than a 4B model is.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import onnx_asr
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import glossary

# Parakeet expects 16 kHz mono. Anything else is resampled by the caller or
# rejected here — silently resampling in-process would hide a client bug that
# degrades every transcript.
SAMPLE_RATE = 16_000

MODEL_ID = os.getenv("STT_MODEL", "istupakov/parakeet-tdt-0.6b-v3-onnx")
QUANT = os.getenv("STT_QUANTISATION", "int8")
GLOSSARY_PATH = os.getenv("STT_GLOSSARY", "/etc/parakeet-stt/glossary.txt")
# Threads are the one knob worth exposing. ONNX Runtime scales sub-linearly
# past about 8; the default of 4 is chosen to leave a small host usable.
THREADS = int(os.getenv("STT_THREADS", "4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parakeet-stt")

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    log.info("loading %s (%s), %d threads", MODEL_ID, QUANT, THREADS)
    started = time.monotonic()
    state["model"] = onnx_asr.load_model(MODEL_ID, quantization=QUANT)
    state["rules"] = glossary.compile_rules(glossary.load(GLOSSARY_PATH))
    log.info("ready in %.1fs", time.monotonic() - started)
    yield
    state.clear()


app = FastAPI(
    title="parakeet-stt",
    description="CPU-only speech-to-text. Raw transcript plus glossary repair.",
    lifespan=lifespan,
)


class Transcript(BaseModel):
    text: str
    audio_seconds: float
    compute_seconds: float
    realtime_factor: float
    repaired: list[str]


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if "model" in state else "loading",
        "model": MODEL_ID,
        "quantisation": QUANT,
        "threads": THREADS,
    }


def _decode(raw: bytes) -> np.ndarray:
    """Read any container soundfile understands, downmix, and require 16 kHz."""
    try:
        audio, rate = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(400, f"could not decode audio: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise HTTPException(
            400, f"expected {SAMPLE_RATE} Hz, got {rate} Hz — resample before sending"
        )
    return audio.mean(axis=1)


@app.post("/transcribe", response_model=Transcript)
async def transcribe(file: UploadFile = File(...)) -> Transcript:
    if "model" not in state:
        raise HTTPException(503, "model still loading")

    samples = _decode(await file.read())
    if samples.size == 0:
        raise HTTPException(400, "audio contains no samples")

    started = time.monotonic()
    text = state["model"].recognize(samples, sample_rate=SAMPLE_RATE)  # type: ignore[attr-defined]
    compute = time.monotonic() - started

    text, repaired = glossary.apply(text.strip(), state["rules"])  # type: ignore[arg-type]
    audio_seconds = samples.size / SAMPLE_RATE

    log.info(
        "%.1fs audio in %.2fs (%.1fx realtime), repaired=%s",
        audio_seconds,
        compute,
        audio_seconds / compute if compute else 0.0,
        repaired or "none",
    )

    return Transcript(
        text=text,
        audio_seconds=round(audio_seconds, 2),
        compute_seconds=round(compute, 2),
        realtime_factor=round(audio_seconds / compute, 1) if compute else 0.0,
        repaired=repaired,
    )
