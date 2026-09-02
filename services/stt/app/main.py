"""Self-hosted speech-to-text. One container, one model.

    audio -> VAD -> recogniser -> glossary repair -> text

Parakeet by default, Whisper by request. There is no LLM cleanup stage and no
second recogniser; both were tried and measured, and both made the transcript
worse. The reasoning is in asr.py and in the README.

Three routes reach the same pipeline. /transcribe is native and returns
everything the run measured; /v1/audio/transcriptions and
/v1/audio/translations are OpenAI-compatible and return the subset that
specification has fields for, so existing clients work unchanged. Prefer the
native one where you control the client — see openai_api.py for what the
compatible shape has to drop, and for the one rule that surface is built
around: every field is honoured or refused by name, never accepted and
dropped.

The wire contract around those routes — the API keys, the 401, OpenAI's error
envelope, the health route and the log configuration — comes from voice_common,
which this service shares with tts-stack and tts-long. Three hand-vendored
copies of that code had drifted into three different defects; the package
docstrings carry the detail. app/errors.py completes that envelope with the two
things the specification requires and the shared package does not emit yet,
over the shared code rather than instead of it. Everything below this line is
what is genuinely particular to speech-to-text.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from voice_common import auth, errors, health
from voice_common import logging as voice_logging

from . import asr, openai_api, pipeline
from . import errors as v1_errors

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
        # What the compatibility surface will and will not do on this
        # deployment, so a client can find out without spending a request on a
        # refusal. Both are properties of the engine, not of the service.
        "translations": bool(getattr(loaded, "can_translate", False)),
        "streaming": bool(getattr(loaded, "can_stream", False)),
        "hotwords": pipeline.HOTWORDS_ENABLED,
        "vad": pipeline.VAD_ENABLED,
        "threads": pipeline.THREADS,
        "max_concurrent": pipeline.MAX_CONCURRENT,
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

# The two pieces of the error envelope voice_common does not yet emit: `param`,
# which the specification requires on every error, and a handler for 404 and
# 405, which leaked FastAPI's {"detail": ...} to a client that reads none of it.
# They live here rather than upstream because requirements.txt pins voice-common
# by an immutable tarball SHA — see app/errors.py.
#
# AFTER auth.install, and that order is load-bearing. The 401 body is built by
# the key middleware and returned from outside every exception handler, so the
# backfill has to be the outer of the two to reach it — and Starlette makes the
# most recently added middleware the outermost.
v1_errors.install(app)

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
    # Built field by field rather than from asdict(): the pipeline's Result now
    # also carries segments, words and logprobs for the /v1 shapes, and this
    # body is a contract that already has clients. Widening it because another
    # route needed the data would be the same mistake as narrowing it.
    result = pipeline.run(
        file.file.read(),
        asr.Options(language=language),
        # 16 kHz only, still. See pipeline.decode: /v1 resamples because no
        # OpenAI client expects otherwise, and this route does not because
        # telling a client its audio is the wrong rate is the documented
        # behaviour it was built with.
        allow_resample=False,
    )
    return Transcript(
        text=result.text,
        raw=result.raw,
        repaired=result.repaired,
        model=result.model,
        audio_seconds=result.audio_seconds,
        speech_seconds=result.speech_seconds,
        compute_seconds=result.compute_seconds,
        realtime_factor=result.realtime_factor,
    )
