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

The wire contract around those two routes — the API keys, the 401, OpenAI's
error envelope, the health route and the log configuration — comes from
voice_common, which this service shares with tts-stack and tts-long. Three
hand-vendored copies of that code had drifted into three different defects;
the package docstrings carry the detail. Everything below this line is what is
genuinely particular to speech-to-text.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from voice_common import auth, errors, health
from voice_common import logging as voice_logging

from . import openai_api, pipeline

# STT_LOG_LEVEL comes with this: until now the only way to get DEBUG out of a
# running container was to edit the source and rebuild the image.
log = voice_logging.setup("stt-stack", "STT")


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    started = time.monotonic()
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

# ApiError and the /v1 validation handler, both rendered in OpenAI's envelope.
errors.install_errors(app)


def _health_details() -> dict[str, object]:
    """The per-service half of /health. Must not block: it runs on the loop.

    Every value here is a module constant or a dict lookup, so it does not.
    `status` is overridden while the model is still loading — that is the one
    field voice_common fills in itself, and the one this service has to
    contradict.
    """
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


# Registers GET /health AND exempts exactly that path from the key check, so
# the route and the exemption can never come to name different strings.
# Container healthchecks call it and have no key; requiring one would turn a
# working service into a restart loop.
health.install_health(app, details=_health_details)

# Everything else needs the key, /openapi.json, /docs and /redoc included: a
# schema dump is a free map of the service. This is middleware rather than a
# per-route dependency, which is what lets FastAPI's own three pages be used
# again — they are plain Starlette routes that no router dependency reaches,
# so this file used to re-register all three by hand just to get the check in
# front of them. Middleware also cannot be forgotten on a route added later.
auth.install(app, "STT_API_KEYS")

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


# Deliberately `def`, not `async def`. pipeline.run is blocking CPU work;
# declared async it would run ON the event loop and starve every other
# request, /health included, so a container healthcheck fails under load and
# the orchestrator restarts a service that is working correctly.
@app.post("/transcribe", response_model=Transcript)
def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> Transcript:
    return Transcript(**asdict(pipeline.run(file.file.read(), language=language)))
