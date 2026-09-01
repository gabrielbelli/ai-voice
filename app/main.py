"""Long-form text-to-speech. Chatterbox on CPU, as a job queue.

Kokoro (tts-stack) answers requests. This does not: at roughly 0.21x realtime
a ten-minute recording takes three quarters of an hour, so an HTTP request
that waits for the audio would time out long before it arrived.

So /jobs accepts work and returns immediately with an id. One job runs at a
time — the model is 6.5 GB and concurrency would only make both jobs slower
while doubling the memory.

/v1/audio/speech sits on top of the same queue for OpenAI clients. It cannot
be honestly synchronous for long input, so it is synchronous only where the
arithmetic allows: short text is waited on and returned as audio, anything
longer gets 202 and a Location header pointing at the native job. The native
routes stay the ones to prefer — realtime_factor, per-segment pauses and the
queue position have no field in the OpenAI shape and are dropped there.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import auth
from .synth import SAMPLE_RATE, Synth

OUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/output"))
THREADS = int(os.getenv("TTS_THREADS", "8"))
IDLE_TIMEOUT = float(os.getenv("TTS_IDLE_TIMEOUT", "600"))
DEFAULT_LANG = os.getenv("TTS_LANGUAGE", "en")
# Stock defaults (0.5 / 0.5 / 0.8) read as animated and over-cheerful, which
# is the wrong register for an explanation. These are calmer.
DEF_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.3"))
DEF_CFG_WEIGHT = float(os.getenv("TTS_CFG_WEIGHT", "0.3"))
DEF_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.6"))

# How much text /v1/audio/speech will block for. Roughly 14 characters of text
# become a second of speech at 150 wpm, and a second of speech costs about
# 4.8 seconds of CPU at 0.21x realtime — so a third of a second per character.
# 300 characters is therefore around 100 seconds of work: inside
# openai-python's 600 s default timeout, and survivable behind a proxy that
# gives up at 120. Set to 0 to always answer 202.
SYNC_MAX_CHARS = int(os.getenv("TTS_OPENAI_SYNC_MAX_CHARS", "300"))
# Longer than the estimate above because the queue may be busy and the first
# job of the day also loads 6.5 GB of weights. When it runs out the job is not
# cancelled — the caller gets 202 and can collect the audio from /jobs.
SYNC_TIMEOUT = float(os.getenv("TTS_OPENAI_SYNC_TIMEOUT", "180"))

# Everything libsndfile writes without an external encoder. mp3, opus and aac
# are in OpenAI's list and are not here: they would mean putting ffmpeg in an
# image that already carries torch, to serve a format nobody asked for yet.
MEDIA_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "pcm": "audio/pcm"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts-long")

jobs: dict[str, dict] = {}
queue: Queue = Queue()
state: dict[str, object] = {}
# Only the OpenAI route registers here, and only while it is waiting on a job.
# Kept out of the job dict so nothing unserialisable can reach /jobs.
#
# An asyncio.Event rather than a threading one, paired with the loop that owns
# it: the waiter is a coroutine on the event loop and the setter is the worker
# thread, and asyncio primitives are not thread-safe to touch from outside.
events: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}


def _worker() -> None:
    synth: Synth = state["synth"]  # type: ignore[assignment]
    while True:
        job_id = queue.get()
        if job_id is None:
            return
        job = jobs[job_id]
        job.update(status="running", started_at=time.time())
        try:
            t = time.monotonic()
            if job["segments"]:
                audio = synth.speak_segments(
                    job["segments"], job["language"], job["exaggeration"],
                    job["cfg_weight"], job["temperature"])
            else:
                audio = synth.speak(
                    job["text"], job["language"], job["exaggeration"],
                    job["cfg_weight"], job["temperature"])
            compute = time.monotonic() - t
            path = OUT_DIR / f"{job_id}.{job['format']}"
            if job["format"] == "pcm":
                # OpenAI's "pcm" is 24 kHz 16-bit mono little-endian, which is
                # exactly what Chatterbox emits — no resampling, just a cast.
                path.write_bytes(
                    (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
            else:
                sf.write(path, audio, SAMPLE_RATE)
            duration = audio.size / SAMPLE_RATE
            job.update(status="done", path=str(path),
                       audio_seconds=round(duration, 1),
                       compute_seconds=round(compute, 1),
                       realtime_factor=round(duration / compute, 3) if compute else 0.0,
                       finished_at=time.time())
            log.info("%s done: %.1fs audio in %.0fs (%.2fx)",
                     job_id[:8], duration, compute, duration / compute if compute else 0)
        except Exception as exc:  # noqa: BLE001 - surfaced on the job, not raised
            job.update(status="failed", error=str(exc), finished_at=time.time())
            log.exception("%s failed", job_id[:8])
        finally:
            # In `finally`, so a failed job wakes its caller with an error
            # rather than leaving it to sit out the whole timeout.
            waiter = events.pop(job_id, None)
            if waiter is not None:
                loop, event = waiter
                try:
                    loop.call_soon_threadsafe(event.set)
                except RuntimeError:
                    # The loop shut down while this job ran. Nothing is waiting
                    # on the other end any more, and the audio is on disk.
                    pass
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["synth"] = Synth(idle_timeout=IDLE_TIMEOUT, threads=THREADS)
    # One worker on purpose. The model is 6.5 GB and generation is sequential,
    # so a second job would double memory and slow both.
    threading.Thread(target=_worker, daemon=True).start()
    auth.log_state(log)
    log.info("ready, %d threads, idle timeout %.0fs (model loads on first job)",
             THREADS, IDLE_TIMEOUT)
    yield
    queue.put(None)
    state["synth"].close()  # type: ignore[attr-defined]
    state.clear()


app = FastAPI(
    title="tts-long",
    description="Chatterbox long-form speech, CPU, as a job queue.",
    lifespan=lifespan,
)
auth.install(app)


class Segment(BaseModel):
    text: str
    pause_after: float = Field(default=0.0, ge=0.0, le=10.0)


class JobRequest(BaseModel):
    text: str | None = None
    segments: list[Segment] | None = None
    language: str = DEFAULT_LANG
    exaggeration: float = Field(default=DEF_EXAGGERATION, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=DEF_CFG_WEIGHT, ge=0.0, le=1.0)
    temperature: float = Field(default=DEF_TEMPERATURE, ge=0.1, le=1.5)


@app.get("/health")
async def health() -> dict[str, object]:
    # `async def`, so this is answered on the event loop and never queues for
    # an AnyIO worker thread. It used to be a sync route sharing that pool
    # (40 threads) with /v1/audio/speech, which held a thread for up to
    # TTS_OPENAI_SYNC_TIMEOUT seconds each: 40 concurrent speech requests
    # starved the pool, /health stopped answering, and an orchestrator
    # restarted a service that was merely busy. Nothing below blocks.
    synth: Synth = state["synth"]  # type: ignore[assignment]
    return {
        "status": "ok",
        "model_loaded": synth.loaded,
        "threads": THREADS,
        "queued": queue.qsize(),
        "running": sum(1 for j in jobs.values() if j["status"] == "running"),
    }


def _estimate(words: int) -> int:
    # ~150 wpm speech at ~0.21x realtime. Rough, but the caller deserves to
    # know this is minutes rather than seconds before it waits on anything.
    return round(words / 150 * 60 / 0.21)


def _enqueue(*, text: str | None, segments: list[tuple[str, float]] | None,
             language: str, exaggeration: float, cfg_weight: float,
             temperature: float, fmt: str = "wav",
             waiter: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        "text": text, "language": language, "segments": segments,
        "exaggeration": exaggeration, "cfg_weight": cfg_weight,
        "temperature": temperature, "format": fmt,
    }
    # Registered before the job is visible to the worker, or a fast job could
    # finish and find nothing to wake.
    if waiter is not None:
        events[job_id] = waiter
    queue.put(job_id)
    return job_id


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest) -> dict[str, object]:
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")
    words = len((req.text or " ".join(s.text for s in (req.segments or []))).split())
    job_id = _enqueue(
        text=req.text,
        segments=[(s.text, s.pause_after) for s in req.segments] if req.segments else None,
        language=req.language, exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight, temperature=req.temperature)
    return {"id": job_id, "status": "queued", "queued_ahead": queue.qsize() - 1,
            "estimated_seconds": _estimate(words)}


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in {"segments", "text"}}


@app.get("/jobs")
def list_jobs() -> dict[str, object]:
    return {"jobs": [_public(j) for j in
                     sorted(jobs.values(), key=lambda j: -j["created_at"])[:50]]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _public(job)


@app.get("/jobs/{job_id}/audio")
def get_audio(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    fmt = job["format"]
    return FileResponse(job["path"], media_type=MEDIA_TYPES[fmt],
                        filename=f"{job_id}.{fmt}")


# ------------------------------------------------------- OpenAI compatible --
#
# Added alongside the native routes, not in place of them. /jobs stays the one
# to prefer, for the reasons in the module docstring.


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    # Accepted and ignored. Chatterbox clones from a reference clip and has no
    # named voices, so mapping 'alloy' onto anything would be an invention. The
    # field stays because OpenAI clients require it and would refuse to send
    # the request without one.
    voice: str = "alloy"
    # OpenAI defaults this to mp3; there is no mp3 encoder in this image, so
    # the default here is wav. Clients that never send the field get audio
    # they can play either way.
    response_format: str = "wav"
    speed: float = 1.0
    # Not OpenAI fields. Accepted because the OpenAI shape has no room for the
    # knobs that decide how this reads, and extra_body is how openai-python
    # passes vendor options through.
    language: str = DEFAULT_LANG
    exaggeration: float = Field(default=DEF_EXAGGERATION, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=DEF_CFG_WEIGHT, ge=0.0, le=1.0)
    temperature: float = Field(default=DEF_TEMPERATURE, ge=0.1, le=1.5)


def _error(status: int, message: str, code: str,
           type_: str = "invalid_request_error") -> JSONResponse:
    """The OpenAI error envelope. openai-python reads the message off this."""
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": type_,
                                           "code": code}})


# Codes openai-python surfaces as `.code` on the exception it raises. Kept
# distinct so a caller can tell "you left out `input`" from "9 is not a valid
# exaggeration" without parsing the message.
_VALIDATION_CODES = {"missing": "missing_required_parameter"}


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
    """Put a rejected body into the OpenAI envelope, but only under /v1.

    A body pydantic rejects never reaches openai_speech, so every envelope
    that route is careful to build was bypassed and the client got FastAPI's
    `{"detail": [...]}` with a 422. openai-python reads no message off that
    shape and raises a bare APIStatusError, which is the same silence a
    missing `input` deserved a sentence for.

    The native routes keep FastAPI's own handler untouched: /jobs is a
    compatibility boundary, and something out there already parses `detail`.
    """
    if not request.url.path.startswith("/v1/"):
        return await request_validation_exception_handler(request, exc)
    errors = exc.errors()
    if not errors:  # Nothing pydantic could locate. Say so rather than guess.
        return _error(400, "request body is not valid", "invalid_value")
    first = errors[0]
    if first["type"] == "json_invalid":
        # loc is ("body", <character offset>) here, and naming a field "12" is
        # worse than not naming one.
        return _error(400, "request body is not valid JSON", "invalid_value")
    # loc[0] is always "body"; the rest names the field, with ints for list
    # indices — segments.0.text.
    field = ".".join(str(part) for part in first["loc"][1:]) or "body"
    return _error(400, f"{field}: {first['msg']}",
                  _VALIDATION_CODES.get(first["type"], "invalid_value"))


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest) -> Response:
    """OpenAI's speech endpoint, synchronous only where it can honestly be.

    OpenAI's contract is request/response. This service runs at 0.21x
    realtime, so for anything but short text a synchronous answer is not slow,
    it is impossible — the socket dies long before the audio exists. Both
    halves of the obvious compromise are here rather than one:

    - input at or under TTS_OPENAI_SYNC_MAX_CHARS is waited on and returned as
      audio, which is what an unmodified OpenAI client needs;
    - anything longer, or a wait that runs out, returns 202 with the job id
      and a Location header for the native route.

    The alternative — always 202 — would hand openai-python a JSON body it
    would happily write into a .wav file. The other alternative, always
    blocking, would be a lie the network catches. A caller that cannot handle
    202 should set TTS_OPENAI_SYNC_MAX_CHARS to something it can wait for and
    keep its input under it.
    """
    text = req.input.strip()
    if not text:
        return _error(400, "input must not be empty", "invalid_input")
    if req.response_format not in MEDIA_TYPES:
        return _error(400, f"response_format '{req.response_format}' is not "
                           f"available: this image has no encoder beyond "
                           f"libsndfile, so only "
                           f"{', '.join(sorted(MEDIA_TYPES))} are produced.",
                      "unsupported_value")
    if abs(req.speed - 1.0) > 1e-6:
        # Silently ignoring it would return audio of the wrong length, which
        # is worse than refusing: Chatterbox has no rate control, and
        # resampling to fake one shifts the pitch with it.
        return _error(400, "speed is not supported: Chatterbox has no rate "
                           "control and resampling would shift pitch.",
                      "unsupported_value")

    # `async def` and an asyncio wait, not a sync route blocking on a
    # threading.Event. A sync route holds one of AnyIO's 40 worker threads for
    # the whole wait — up to TTS_OPENAI_SYNC_TIMEOUT seconds — and this
    # service is slow by design, so 40 concurrent callers took every thread
    # and /health, itself a sync route on that pool, stopped answering. An
    # orchestrator then restarted a service that was only busy. Waiting on the
    # event loop costs a coroutine instead, and nothing else queues behind it.
    done = asyncio.Event() if len(text) <= SYNC_MAX_CHARS else None
    job_id = _enqueue(text=text, segments=None, language=req.language,
                      exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
                      temperature=req.temperature, fmt=req.response_format,
                      waiter=(asyncio.get_running_loop(), done) if done else None)

    if done is not None:
        try:
            await asyncio.wait_for(done.wait(), SYNC_TIMEOUT)
        except TimeoutError:
            # Deregister before falling through to the 202, so the worker does
            # not later wake an event nobody is holding. `pop` is the whole
            # handshake: if the worker got there first this is a no-op.
            events.pop(job_id, None)
        else:
            job = jobs[job_id]
            if job["status"] == "done":
                return FileResponse(job["path"],
                                    media_type=MEDIA_TYPES[req.response_format],
                                    filename=f"{job_id}.{req.response_format}")
            return _error(500, job.get("error", "synthesis failed"),
                          "synthesis_failed", type_="server_error")

    # Either too long to wait for, or the wait ran out. The job is untouched
    # and still queued, so nothing has been wasted — Location points at where
    # it will appear.
    return JSONResponse(
        status_code=202,
        headers={"Location": f"/jobs/{job_id}"},
        content={"id": job_id, "status": "queued",
                 "queued_ahead": queue.qsize() - 1,
                 "estimated_seconds": _estimate(len(text.split())),
                 "audio_url": f"/jobs/{job_id}/audio"},
    )
