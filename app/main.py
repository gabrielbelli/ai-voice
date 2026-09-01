"""Self-hosted speech-to-text. One container, one model.

    audio -> VAD -> recogniser -> glossary repair -> text

Parakeet by default, Whisper by request. There is no LLM cleanup stage and no
second recogniser; both were tried and measured, and both made the transcript
worse. The reasoning is in asr.py and in the README.

Two routes reach the same pipeline. /transcribe is native and returns
everything the run measured; /v1/audio/transcriptions is OpenAI-compatible and
returns the subset that specification has fields for, so existing clients work
unchanged. Prefer the native one where you control the client — see
openai_api.py for what the compatible shape has to drop.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from . import auth, openai_api, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt-stack")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.monotonic()
    # Before the model loads, so the warning about unauthenticated access is
    # on screen even if loading then fails and the service never serves.
    auth.announce()

    pipeline.start()

    log.info("ready in %.1fs, model=%s, %d threads",
             time.monotonic() - started, pipeline.MODEL, pipeline.THREADS)
    yield
    pipeline.stop()


app = FastAPI(
    title="stt-stack",
    description="VAD, one recogniser, glossary repair. Parakeet by default.",
    lifespan=lifespan,
)
auth.install(app)
app.include_router(openai_api.router)


class Transcript(BaseModel):
    text: str
    raw: str
    repaired: list[str]
    model: str
    audio_seconds: float
    speech_seconds: float
    compute_seconds: float
    realtime_factor: float


# Deliberately unauthenticated, even when STT_API_KEYS is set. Container
# healthchecks call this and have no key; requiring one turns a working
# service into a restart loop.
@app.get("/health")
def health() -> dict[str, object]:
    loaded = pipeline.loaded()
    return {
        "status": "ok" if loaded else "loading",
        "model": pipeline.MODEL,
        "model_id": os.getenv("STT_MODEL_ID", ""),
        "accepts_vocabulary": bool(getattr(loaded, "accepts_vocabulary", False)),
        "hotwords": pipeline.HOTWORDS_ENABLED,
        "vad": pipeline.VAD_ENABLED,
        "threads": pipeline.THREADS,
    }


# Deliberately `def`, not `async def`. pipeline.run is blocking CPU work;
# declared async it would run ON the event loop and starve every other
# request, /health included, so a container healthcheck fails under load and
# the orchestrator restarts a service that is working correctly.
@app.post("/transcribe", response_model=Transcript,
          dependencies=[Depends(auth.require_key)])
def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> Transcript:
    return Transcript(**asdict(pipeline.run(file.file.read(), language=language)))
