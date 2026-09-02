"""Long-form text-to-speech. Chatterbox on CPU, as a job queue.

Kokoro (tts-stack) answers requests. This does not: at roughly 0.21x realtime
a ten-minute recording takes three quarters of an hour, so an HTTP request
that waits for the audio would time out long before it arrived.

So /jobs accepts work and returns immediately with an id. One job runs at a
time — the model is 6.5 GB and concurrency would only make both jobs slower
while doubling the memory.

/v1/audio/speech sits on top of the same queue for OpenAI clients, and it now
answers in all three of the ways that endpoint can honestly be answered here:

  * `stream_format: "sse"` streams the audio as it is generated. This is the
    route worth using for anything long. The first `speech.audio.delta` leaves
    when the FIRST SENTENCE finishes rather than when the whole input does, so
    a caller hears sound in tens of seconds instead of waiting out a silence
    measured in minutes. It does not make the service faster — the compute is
    unchanged — it removes the dead air and the client-side timeout.
  * input short enough that the arithmetic allows it is waited on and returned
    as one buffered body, which is what an unmodified OpenAI client needs.
  * anything longer gets 202 and a Location header pointing at the native job.
    That is a deliberate deviation from OpenAI's contract, documented in the
    README with the measurement that forces it.

Every input is split into sentence-sized chunks before it reaches the model,
and that is a bug fix rather than a streaming detail: generate() stops after
1000 speech tokens, which is 40 seconds of audio, and 1690 characters measured
on the deployed instance came back as exactly 40.0 seconds with no error. See
app/chunking.py.

The native routes stay the ones to prefer for batch work — realtime_factor,
per-segment pauses and the queue position have no field in the OpenAI shape
and are dropped there.

Authentication, the health contract, the error envelope, `Segment` and the
logging setup are voice_common's. app/envelope.py is gone: it was this
service's private copy of the envelope, written while voice-common was pinned
by tarball SHA, and every line of it now lives in voice_common.errors — it was
the best of the three vendored copies and the one the shared code was built
from, so this service's wire output did not change. app/encoders.py is the one
encoder both the buffered and the streamed paths use. What is left in this file
is the queue, which is the whole reason this service exists separately from
tts-stack.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from voice_common import auth, logging as voice_logging
from voice_common.errors import error_response, install_errors
from voice_common.health import install_health
from voice_common.models import OpenAISpeechRequest, Segment

from . import voices as voice_registry
from .chunking import chunk_text, speech_seconds
from .encoders import MEDIA_TYPES, available_formats, encode, make_encoder
from .synth import (SAMPLE_RATE, SUPPORTED_LANGUAGES, Synth, speech_tokens)

OUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/output"))
THREADS = int(os.getenv("TTS_THREADS", "8"))
IDLE_TIMEOUT = float(os.getenv("TTS_IDLE_TIMEOUT", "600"))
DEFAULT_LANG = os.getenv("TTS_LANGUAGE", "en")
# Stock defaults (0.5 / 0.5 / 0.8) read as animated and over-cheerful, which
# is the wrong register for an explanation. These are calmer.
DEF_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.3"))
DEF_CFG_WEIGHT = float(os.getenv("TTS_CFG_WEIGHT", "0.3"))
DEF_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.6"))

# OpenAI's schema caps `input` at 4096 characters. It was not enforced: a
# 5000-character body was accepted and queued.
MAX_INPUT_CHARS = 4096

# The rate this host is achieving, seeded from the measurement and then
# corrected by every job that finishes. Seeded rather than hardcoded because
# the same code measured 0.217x on the deployed instance and 0.138x on a
# loaded laptop, and an estimate built on the wrong one of those is wrong by
# 60% — which is how a request UNDER the documented synchronous threshold used
# to turn into a 202 with no explanation.
RTF_SEED = float(os.getenv("TTS_REALTIME_FACTOR", "0.21"))
# What a cold start costs the caller who triggers it: 6.5 GB off disk, or a
# ~3 GB download on a truly cold image. Counted against the synchronous
# budget rather than ignored, because it lands inside whatever the caller is
# waiting on.
COLD_LOAD_SECONDS = float(os.getenv("TTS_COLD_LOAD_SECONDS", "60"))

# How much text /v1/audio/speech will block for, as a hard ceiling. The
# arithmetic that decides whether a given request under this ceiling can
# actually be finished in time is _sync_budget(): fifteen characters is a
# second of speech (measured — see app/chunking.py) and a second of speech
# costs 1/rtf seconds of compute, so 300 characters is around 95 s at 0.21x.
# Set to 0 to always answer 202.
SYNC_MAX_CHARS = int(os.getenv("TTS_OPENAI_SYNC_MAX_CHARS", "300"))
# When it runs out the job is not cancelled — the caller gets 202 and can
# collect the audio from /jobs.
SYNC_TIMEOUT = float(os.getenv("TTS_OPENAI_SYNC_TIMEOUT", "180"))

# Queue depth past which /v1 answers 429 and /jobs answers 429. It is the only
# error response OpenAI's schema declares for this path, and there was no
# backpressure of any kind here: an unbounded number of multi-minute jobs
# could be queued, and the queue is the memory and the disk of one process.
MAX_QUEUE = int(os.getenv("TTS_MAX_QUEUE", "32"))
# Finished jobs and their audio are swept this many seconds after they end.
# Until now nothing was ever removed and both the dict and /output grew for
# the lifetime of the process. 0 disables the sweep and restores that.
JOB_TTL = float(os.getenv("TTS_JOB_TTL", "86400"))

# Seconds between `:` comment lines on an idle SSE stream. Comments are
# ignored by openai-python's SSEDecoder (verified) and by every other SSE
# client; they exist so a stream that is waiting — for the model to load, for
# a job ahead of it in the queue, or for a long sentence — does not look dead
# to a proxy or trip a read timeout.
SSE_KEEPALIVE = float(os.getenv("TTS_SSE_KEEPALIVE", "10"))

# The same one line of configuration this always had, plus the TTS_LOG_LEVEL
# switch it never had: getting DEBUG out of a running container used to mean
# editing the source and rebuilding a 6.5 GB image, which is exactly the moment
# that is impossible. Unset still means INFO, so nothing changes for anyone who
# has not asked for it.
log = voice_logging.setup("tts-long", "TTS")

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

# A module-level singleton, and the CMD is a bare `uvicorn app.main:app` with
# no --workers, so there is exactly ONE process holding exactly one of these.
# That is what makes Registry.refresh worth having: a clip written into the
# shared volume by services/ui becomes resolvable on the next request rather
# than on the next restart of a container that carries 6.5 GB of Chatterbox.
VOICES = voice_registry.load_registry()
FORMATS = available_formats()


@dataclass
class Stream:
    """The pipe from the worker thread to one SSE response.

    Same reasoning as `events` above: the producer is the worker thread and
    the consumer is a coroutine, so every hand-off goes through
    call_soon_threadsafe onto the loop that owns the queue.
    """

    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def put(self, kind: str, payload: object) -> None:
        with suppress(RuntimeError):
            # The loop is gone: the client disconnected and the response was
            # torn down. The job carries on to disk, where /jobs can collect
            # it, and nothing here needs to hear about it.
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, payload))


class _Rate:
    """The realtime factor this host is actually achieving.

    An exponential moving average over finished jobs, because the alternative
    — a constant — was wrong by 60% on the machine the audit ran on, and every
    estimate and every synchronous/202 decision is built on it. Weighted
    towards recent jobs so a host that gets busier is noticed within a couple
    of jobs rather than averaged away.
    """

    def __init__(self, seed: float) -> None:
        self.value = seed
        self._lock = threading.Lock()

    def observe(self, audio_seconds: float, compute_seconds: float) -> None:
        if audio_seconds <= 0 or compute_seconds <= 0:
            return
        with self._lock:
            self.value += 0.3 * (audio_seconds / compute_seconds - self.value)


rate = _Rate(RTF_SEED)


def _compute_seconds(chars: int) -> float:
    """Estimated CPU seconds to speak `chars` characters, at the observed rate."""
    return speech_seconds(chars) / max(rate.value, 1e-3)


def _job_chars(job: dict) -> int:
    return sum(len(text) for text, _ in job["segments"])


def _backlog_seconds() -> float:
    """Estimated compute already accepted and not yet finished.

    Over a `list()` snapshot, and so is every other walk of `jobs` below. The
    dict is read from the event loop and from AnyIO's thread pool — every sync
    route runs there — while DELETE pops from it and the sweeper pops from it,
    and a plain `.values()` iteration that is interrupted by a pop raises
    RuntimeError: dictionary changed size during iteration. That would be a 500
    on /v1/audio/speech caused by an unrelated DELETE landing in the same
    millisecond. Building the list is one C-level call and cannot be
    interrupted part way.
    """
    return sum(_compute_seconds(_job_chars(job)) for job in list(jobs.values())
               if job["status"] in {"queued", "running"})


# ------------------------------------------------------------------ worker --


def _worker() -> None:
    synth: Synth = state["synth"]  # type: ignore[assignment]
    while True:
        job_id = queue.get()
        if job_id is None:
            return
        job = jobs[job_id]
        try:
            _run(synth, job)
        finally:
            # In `finally`, so a failed job wakes its caller with an error
            # rather than leaving it to sit out the whole timeout.
            waiter = events.pop(job["id"], None)
            if waiter is not None:
                loop, event = waiter
                with suppress(RuntimeError):
                    # The loop shut down while this job ran. Nothing is waiting
                    # on the other end any more, and the audio is on disk.
                    loop.call_soon_threadsafe(event.set)
            queue.task_done()


def _run(synth: Synth, job: dict) -> None:
    stream: Stream | None = job.get("stream")
    if job["cancelled"]:
        # Cancelled while it sat in the queue. Nothing was generated, so there
        # is nothing to write and nothing to stream.
        job.update(status="cancelled", finished_at=time.time())
        if stream is not None:
            stream.put("error", "the job was cancelled before it started")
        return

    job.update(status="running", started_at=time.time())
    encoder = None
    try:
        started = time.monotonic()
        if stream is not None:
            # The streaming encoder is created here rather than in the route so
            # that an ffmpeg process is never started for a request that then
            # waits ten minutes in the queue.
            encoder = make_encoder(job["format"])

        def on_chunk(piece) -> None:  # noqa: ANN001 - numpy array
            if encoder is None or stream is None:
                return
            data = encoder.write(piece)
            if data:
                stream.put("delta", data)

        spoken = synth.speak_segments(
            job["segments"], job["language"], job["exaggeration"],
            job["cfg_weight"], job["temperature"], job["reference"],
            on_chunk=on_chunk if stream is not None else None,
            cancelled=lambda: bool(job["cancelled"]))
        compute = time.monotonic() - started

        if encoder is not None and stream is not None:
            tail = encoder.close()
            if tail:
                stream.put("delta", tail)
            encoder = None

        # The file on disk is always the buffered encoding, headers and all,
        # even for a streamed request: /jobs/<id>/audio has to hand over a
        # complete file, and a client whose stream dropped half way should
        # still find the whole thing there.
        path = OUT_DIR / f"{job['id']}.{job['format']}"
        data, _ = encode(spoken.audio, job["format"])
        path.write_bytes(data)

        duration = spoken.audio.size / SAMPLE_RATE
        rate.observe(duration, compute)
        usage = {
            "input_tokens": spoken.input_tokens,
            "output_tokens": speech_tokens(spoken.audio.size),
            "total_tokens": spoken.input_tokens + speech_tokens(spoken.audio.size),
        }
        job.update(status="cancelled" if job["cancelled"] else "done",
                   path=str(path),
                   audio_seconds=round(duration, 1),
                   compute_seconds=round(compute, 1),
                   realtime_factor=round(duration / compute, 3) if compute else 0.0,
                   usage=usage,
                   finished_at=time.time())
        log.info("%s %s: %.1fs audio in %.0fs (%.2fx)", job["id"][:8],
                 job["status"], duration, compute,
                 duration / compute if compute else 0)
        if stream is not None:
            stream.put("done", usage)
    except Exception as exc:  # noqa: BLE001 - surfaced on the job, not raised
        job.update(status="failed", error=str(exc), finished_at=time.time())
        log.exception("%s failed", job["id"][:8])
        if stream is not None:
            stream.put("error", str(exc))
    finally:
        if encoder is not None:
            # Only reached on the failure path: close() on a half-fed ffmpeg
            # would wait on a process nobody is going to read.
            kill = getattr(encoder, "kill", None)
            if kill is not None:
                kill()


async def _sweeper() -> None:
    """Remove finished jobs and their audio once TTS_JOB_TTL has passed.

    Nothing was ever removed before: the dict and /output grew for the lifetime
    of the process, which on a service whose files are minutes of 24 kHz wav is
    a disk that fills quietly. A day is far longer than any sane collection
    window and short enough to bound the growth; 0 restores the old behaviour.
    """
    while JOB_TTL > 0:
        await asyncio.sleep(min(JOB_TTL, 300))
        cutoff = time.time() - JOB_TTL
        for job_id, job in list(jobs.items()):
            if job.get("finished_at", time.time()) < cutoff:
                _discard(job)
                jobs.pop(job_id, None)
                log.info("%s swept after %.0fs", job_id[:8], JOB_TTL)


def _discard(job: dict) -> None:
    path = job.get("path")
    if path:
        with suppress(OSError):
            Path(path).unlink()


@asynccontextmanager
async def lifespan(app: FastAPI):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["synth"] = Synth(idle_timeout=IDLE_TIMEOUT, threads=THREADS)
    # One worker on purpose. The model is 6.5 GB and generation is sequential,
    # so a second job would double memory and slow both.
    threading.Thread(target=_worker, daemon=True).start()
    sweeper = asyncio.create_task(_sweeper())
    log.info("ready, %d threads, idle timeout %.0fs, formats %s, voices %s "
             "(model loads on first job)", THREADS, IDLE_TIMEOUT,
             ", ".join(FORMATS), ", ".join(VOICES.names))
    if VOICES.aliased:
        log.warning("no reference clip for %s: those names answer with the "
                    "built-in voice and every response says so in X-Voice. "
                    "Drop <name>.wav into TTS_VOICE_DIR to give them their own "
                    "voice, or set TTS_VOICE_STRICT=1 to refuse them.",
                    ", ".join(VOICES.aliased))
    if len(FORMATS) < len(MEDIA_TYPES):
        # Including the schema's DEFAULT, which is mp3 — so on a checkout
        # without ffmpeg every request that omits response_format is refused.
        # Refused rather than quietly answered in wav: handing a caller who
        # believes it asked for mp3 a wav file with no error is the exact
        # defect this release removed. The image installs ffmpeg and the build
        # checks for it, so this is a warning for a hand-rolled environment.
        log.warning("ffmpeg is not on PATH: %s are unavailable and requests "
                    "for them are refused with a 400 — mp3 included, which is "
                    "the default, so a request that omits response_format will "
                    "be refused too",
                    ", ".join(f for f in MEDIA_TYPES if f not in FORMATS))
    yield
    sweeper.cancel()
    queue.put(None)
    state["synth"].close()  # type: ignore[attr-defined]
    state.clear()


app = FastAPI(
    title="tts-long",
    description="Chatterbox long-form speech, CPU, as a job queue.",
    lifespan=lifespan,
)

# Installed on the app rather than route by route, so a route added later is
# covered without anyone having to remember to ask for it. TTS_API_KEYS is a
# parameter of the shared middleware precisely so that no operator-visible
# variable had to be renamed for this service to stop keeping its own copy —
# a copy that, among other things, could never authenticate a key with an
# accent in it and answered 401 to a probe written `/health/`.
auth.install(app, "TTS_API_KEYS")

# The /v1 error envelope, in the four-field shape the schema requires, plus
# the 404, 405 and 500 handlers that used to escape it. Order against
# auth.install no longer matters: the middleware's 401 is built by the same
# error_response as everything else, so it carries `param` without this
# service rebinding a name to put it there. The native routes keep FastAPI's
# own `{"detail": ...}` and its 422: /jobs is the older contract and something
# out there already parses it.
install_errors(app)


class JobRequest(BaseModel):
    text: str | None = None
    # voice_common.models.Segment: the same `text` and the same 0.0–10.0
    # second `pause_after`, which is a published part of both this API and
    # tts-stack's and no longer free to drift in one of them.
    segments: list[Segment] | None = None
    language: str = DEFAULT_LANG
    exaggeration: float = Field(default=DEF_EXAGGERATION, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=DEF_CFG_WEIGHT, ge=0.0, le=1.0)
    temperature: float = Field(default=DEF_TEMPERATURE, ge=0.1, le=1.5)
    voice: str | None = None


def _health() -> dict[str, object]:
    """The body, unchanged apart from the queue's new ceiling.

    install_health registers `/health` AND exempts exactly that string from
    authentication, so a rename can no longer lock the container healthcheck
    out — the two used to be independent literals in two modules.

    It also makes the route `async def`, which this service had already paid to
    learn: a sync route runs on AnyIO's forty-thread pool, shared with
    /v1/audio/speech, which held a thread for up to TTS_OPENAI_SYNC_TIMEOUT
    seconds each. Forty concurrent speech requests starved the pool, /health
    stopped answering, and an orchestrator restarted a service that was merely
    busy. Nothing below blocks.
    """
    synth: Synth = state["synth"]  # type: ignore[assignment]
    return {
        "status": "ok",
        "model_loaded": synth.loaded,
        "threads": THREADS,
        "queued": queue.qsize(),
        "queue_capacity": MAX_QUEUE,
        "running": sum(1 for j in list(jobs.values()) if j["status"] == "running"),
        "realtime_factor": round(rate.value, 3),
    }


install_health(app, _health)


def _estimate(chars: int) -> int:
    """Seconds of compute for `chars`, at the rate this host is achieving.

    Two things were wrong with the number this replaced. It used a fixed
    0.21x, which under-predicted by 1.5x on the machine the audit measured;
    and it counted whitespace-separated words, so a 5000-character string with
    no spaces in it was estimated at two seconds.
    """
    return round(_compute_seconds(chars))


def _enqueue(*, segments: list[tuple[str, float]], language: str,
             exaggeration: float, cfg_weight: float, temperature: float,
             voice: str, reference: str | None, fmt: str = "wav",
             text: str | None = None,
             waiter: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None,
             stream: Stream | None = None) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        "text": text, "language": language, "segments": segments,
        "exaggeration": exaggeration, "cfg_weight": cfg_weight,
        "temperature": temperature, "format": fmt, "voice": voice,
        "reference": reference, "cancelled": False, "stream": stream,
        "chunks": len(segments),
    }
    # Registered before the job is visible to the worker, or a fast job could
    # finish and find nothing to wake.
    if waiter is not None:
        events[job_id] = waiter
    queue.put(job_id)
    return job_id


def _full() -> bool:
    return MAX_QUEUE > 0 and queue.qsize() >= MAX_QUEUE


def _retry_after() -> int:
    """Seconds a rejected caller should wait: how long the backlog will take."""
    return max(1, round(_backlog_seconds()))


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest) -> dict[str, object]:
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"unsupported language {req.language!r}; "
                                 f"chatterbox has {', '.join(SUPPORTED_LANGUAGES)}")
    # One stat() before resolving, so a voice that arrived since this process
    # started is found. See Registry.refresh for why it is not a full listing.
    VOICES.refresh()
    resolved = VOICES.resolve(req.voice)
    if resolved is None:
        raise HTTPException(400, f"unknown voice {req.voice!r}; this service "
                                 f"has {', '.join(VOICES.names)}")
    if _full():
        # The queue is one process's memory and disk. Nothing bounded it
        # before, so a client in a loop could accept an hour of work in a
        # second and then wait an hour for the first of it.
        retry = _retry_after()
        raise HTTPException(429, f"queue is full ({MAX_QUEUE} jobs); retry in "
                                 f"about {retry}s",
                            headers={"Retry-After": str(retry)})
    name, reference = resolved
    segments = _segments(req.text, req.segments)
    job_id = _enqueue(segments=segments, text=req.text, language=req.language,
                      exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
                      temperature=req.temperature, voice=name,
                      reference=reference)
    return {"id": job_id, "status": "queued", "queued_ahead": queue.qsize() - 1,
            "chunks": len(segments),
            "estimated_seconds": _estimate(sum(len(t) for t, _ in segments))}


def _segments(text: str | None,
              segments: list[Segment] | None) -> list[tuple[str, float]]:
    """Everything the worker will speak, as (text, pause_after) pairs.

    Every piece goes through the chunker, segments included: a `segments` entry
    of 1200 characters hits generate()'s 40-second ceiling exactly as a flat
    `text` of 1200 characters does. The pause belongs to the LAST piece of a
    segment, so splitting a segment does not insert a gap that was never asked
    for.
    """
    if segments:
        out: list[tuple[str, float]] = []
        for segment in segments:
            pieces = chunk_text(segment.text)
            if not pieces:
                # A segment with no text is a pause and nothing else, which
                # voice_common.audio.splice already handles.
                out.append(("", segment.pause_after))
                continue
            for index, piece in enumerate(pieces):
                last = index == len(pieces) - 1
                out.append((piece, segment.pause_after if last else 0.0))
        return out
    return [(piece, 0.0) for piece in chunk_text(text or "")]


def _public(job: dict) -> dict:
    """The job as /jobs reports it, over a snapshot for the reason above.

    `dict(job)` rather than `job.items()`: the worker thread ADDS keys to a
    running job — started_at, then path and the timings — so a comprehension
    over the live dict can be interrupted mid-walk by the job it is reporting
    on finishing, which is a 500 on GET /jobs/<id> at the exact moment a caller
    is most likely to be polling it.
    """
    return {k: v for k, v in dict(job).items()
            if k not in {"segments", "text", "stream", "reference"}}


@app.get("/jobs")
def list_jobs() -> dict[str, object]:
    recent = sorted(list(jobs.values()), key=lambda j: -j["created_at"])[:50]
    return {"jobs": [_public(j) for j in recent]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _public(job)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Cancel a queued job, or discard a finished one.

    There was no way to do either: a 202 handed out an id and the worker ground
    through it whatever happened at the other end. A job that has already
    started stops at its next chunk boundary — generate() has no interruption
    point inside it, so a sentence already in flight is finished and kept.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job["cancelled"] = True
    if job["status"] in {"done", "failed", "cancelled"}:
        _discard(job)
        jobs.pop(job_id, None)
        return {"id": job_id, "status": "deleted"}
    return {"id": job_id, "status": "cancelling"}


def _file_stream(path: str) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        while True:
            block = handle.read(65536)
            if not block:
                return
            yield block


@app.get("/jobs/{job_id}/audio")
def get_audio(job_id: str) -> Response:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] not in {"done", "cancelled"} or not job.get("path"):
        raise HTTPException(409, f"job is {job['status']}")
    fmt = job["format"]
    # Content-Disposition stays on the NATIVE route: this one hands over a
    # finished artefact and a filename is useful. It is gone from /v1, where
    # OpenAI's schema has no such header and it forced a download instead of
    # inline playback.
    return StreamingResponse(
        _file_stream(job["path"]), media_type=MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{job_id}.{fmt}"'})


@app.get("/voices")
def list_voices() -> dict[str, object]:
    """What `voice` may name. Unknown names are a 400, so they are listed.

    Refreshed first, because this is the call a UI makes to build a voice
    picker and a picker that cannot see a clip someone just added is the whole
    reason the registry stopped being immutable.
    """
    VOICES.refresh()
    return {"voices": VOICES.names,
            "openai_aliases": {name: voice_registry.BUILTIN
                               for name in VOICES.aliased},
            "strict": VOICES.strict}


# ------------------------------------------------------- OpenAI compatible --
#
# Added alongside the native routes, not in place of them. /jobs stays the one
# to prefer for batch work, for the reasons in the module docstring.


class CustomVoice(BaseModel):
    """OpenAI's custom-voice object: `{"id": "voice_1234"}`.

    The schema's VoiceIdsOrCustomVoice is anyOf[string, {id: string}] and
    openai-python's `Voice` alias includes it. It was rejected here with
    "voice: Input should be a valid string" — including for the minimal SDK
    call, which sends exactly this form.

    Declared as a model rather than unwrapped by a validator so that this
    service's own /openapi.json keeps the anyOf, which is what a generated
    client reads to learn the object form is accepted. The name is visible on
    the wire — pydantic tags a failed union branch with the class name, and
    voice_common.errors puts that tag in the message — so it is OpenAI's word
    for the shape rather than an internal one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str


class SpeechRequest(OpenAISpeechRequest):
    """OpenAI's /v1/audio/speech body, with the parts that are Chatterbox's.

    `model`, `input` and `speed` come from voice_common.models.OpenAISpeechRequest.

    **`extra="forbid"`, overriding the base.** The base allows unknown fields
    so that a client speaking a newer dialect of OpenAI's API is not rejected
    for it, and that reasoning is sound for a field OpenAI adds. It was not
    sound for what it actually did here: `{"stream": true}` and
    `{"totally_unknown_field": 123}` both returned 200 with audio, and
    `stream` is the TRANSCRIPTION-side switch, so a client that sent it
    expecting a stream got a buffered file and no way to tell. OpenAI's own
    schema sets additionalProperties: false and its API answers "Unrecognized
    request argument supplied"; matching that is what makes the reply
    trustworthy. The cost is that a genuinely new OpenAI field is a 400 until
    it is added here, which is the trade the README states.

    `response_format` and the speed range are deliberately NOT in the base.
    Both are properties of this image.
    """

    model_config = ConfigDict(extra="forbid")

    # "tts-1" rather than the base's "default", because that is what this
    # service has always answered with and something may be reading it back.
    model: str = "tts-1"
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    # str or {"id": ...}; resolved against app/voices.py, which is also what
    # decides whether an unknown name is a 400.
    voice: str | CustomVoice | None = None
    # OpenAI's default, and now this service's, because the image carries an
    # encoder for it. It used to default to wav while an EXPLICIT mp3 was
    # refused with a 400 — so a caller who omitted the field believing it had
    # asked for mp3 was handed wav with no error at all.
    response_format: str = "mp3"
    # "audio" (one buffered body) or "sse". Validated in the route so the
    # message can explain, rather than as a Literal that produces pydantic's.
    stream_format: str = "audio"
    # Refused rather than ignored. See the route.
    instructions: str | None = None
    # Not OpenAI fields. Accepted because the OpenAI shape has no room for the
    # knobs that decide how this reads, and extra_body is how openai-python
    # passes vendor options through. Declared rather than left to extra="allow"
    # now that unknown fields are refused.
    language: str = DEFAULT_LANG
    exaggeration: float = Field(default=DEF_EXAGGERATION, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=DEF_CFG_WEIGHT, ge=0.0, le=1.0)
    temperature: float = Field(default=DEF_TEMPERATURE, ge=0.1, le=1.5)


def _validate(req: SpeechRequest) -> tuple[str, str | None] | JSONResponse:
    """Everything that can be refused before a single token is generated.

    Every parameter OpenAI's schema declares is either honoured or refused here
    by name. Nothing is accepted and dropped.
    """
    if not req.input.strip():
        return error_response(400, "input must not be empty",
                              code="invalid_value", param="input")
    if req.instructions is not None:
        # Chatterbox has no instruction conditioning of any kind. Accepting
        # "speak cheerfully" and returning the same flat delivery is the
        # failure the audit found: no error, no warning header, nothing.
        return error_response(
            400, "instructions is not supported: Chatterbox has no "
                 "instruction conditioning. Delivery is controlled with the "
                 "exaggeration, cfg_weight and temperature vendor fields, "
                 "which openai-python sends through extra_body.",
            code="unsupported_value", param="instructions")
    if req.response_format not in FORMATS:
        known = req.response_format in MEDIA_TYPES
        return error_response(
            400,
            f"response_format '{req.response_format}' is not available here: "
            + ("ffmpeg is not on PATH in this container."
               if known else
               f"OpenAI's formats are {', '.join(MEDIA_TYPES)}.")
            + f" This image produces {', '.join(FORMATS)}.",
            code="unsupported_value", param="response_format")
    if abs(req.speed - 1.0) > 1e-6:
        # Silently ignoring it would return audio of the wrong length, which
        # is worse than refusing: Chatterbox has no rate control, and
        # resampling to fake one shifts the pitch with it.
        return error_response(400, "speed is not supported: Chatterbox has no "
                                   "rate control and resampling would shift "
                                   "pitch.",
                              code="unsupported_value", param="speed")
    if req.stream_format not in {"audio", "sse"}:
        return error_response(
            400, f"stream_format '{req.stream_format}' is not one of 'audio' "
                 f"or 'sse'.", code="invalid_value", param="stream_format")
    if req.language not in SUPPORTED_LANGUAGES:
        # generate() raises ValueError on an unsupported language_id, and
        # finding that out in the worker means the caller queued to be told
        # about a typo.
        return error_response(
            400, f"language '{req.language}' is not one chatterbox speaks: "
                 f"{', '.join(SUPPORTED_LANGUAGES)}.",
            code="unsupported_value", param="language")

    requested = req.voice.id if isinstance(req.voice, CustomVoice) else req.voice
    VOICES.refresh()
    resolved = VOICES.resolve(requested)
    if resolved is None:
        return error_response(
            400, f"unknown voice '{requested}': this service has "
                 f"{', '.join(VOICES.names)}. Chatterbox clones from a "
                 f"reference clip, so a voice is a file in TTS_VOICE_DIR.",
            code="unsupported_value", param="voice")
    return resolved


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest) -> Response:
    """OpenAI's speech endpoint: streamed, synchronous, or a job id.

    OpenAI's contract is request/response. This service runs at roughly 0.21x
    realtime, so for anything but short text a buffered synchronous answer is
    not slow, it is impossible — the socket dies long before the audio exists.
    Three answers, in the order a caller should want them:

    - `stream_format: "sse"` streams deltas as sentences finish. Nothing is
      buffered and nothing is faked: the first frame leaves when the first
      sentence is generated.
    - input the arithmetic says can be finished inside TTS_OPENAI_SYNC_TIMEOUT
      is waited on and returned as one body.
    - anything longer, or a wait that runs out, returns 202 with the job id
      and a Location header for the native route. openai-python treats a 202
      as success and will write that JSON into the caller's file; the README
      says so, and Retry-After plus Location are there for the clients that
      can act on them.
    """
    resolved = _validate(req)
    if isinstance(resolved, JSONResponse):
        return resolved
    voice_name, reference = resolved

    if _full():
        # The only error response OpenAI's schema declares for this path, and
        # there was none of any kind here. Retry-After is the half that makes
        # it actionable: openai-python honours it when it retries a 429.
        retry = _retry_after()
        response = error_response(
            429, f"the queue is full ({MAX_QUEUE} jobs ahead). This service "
                 f"generates one job at a time on CPU; retry in about "
                 f"{retry}s.",
            type_="rate_limit_error", code="rate_limit_exceeded")
        response.headers["Retry-After"] = str(retry)
        return response

    text = req.input.strip()
    segments = _segments(text, None)
    fmt = req.response_format

    if req.stream_format == "sse":
        return _sse_response(req, segments, text, voice_name, reference, fmt)

    chars = len(text)
    budget = _sync_budget()
    done = (asyncio.Event()
            if 0 < chars <= SYNC_MAX_CHARS and _compute_seconds(chars) <= budget
            else None)
    job_id = _enqueue(
        segments=segments, text=text, language=req.language,
        exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
        temperature=req.temperature, voice=voice_name, reference=reference,
        fmt=fmt,
        waiter=(asyncio.get_running_loop(), done) if done else None)

    # `async def` and an asyncio wait, not a sync route blocking on a
    # threading.Event. A sync route holds one of AnyIO's 40 worker threads for
    # the whole wait — up to TTS_OPENAI_SYNC_TIMEOUT seconds — and this
    # service is slow by design, so 40 concurrent callers took every thread
    # and /health, itself a sync route on that pool, stopped answering. An
    # orchestrator then restarted a service that was only busy. Waiting on the
    # event loop costs a coroutine instead, and nothing else queues behind it.
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
                return StreamingResponse(
                    _file_stream(job["path"]), media_type=MEDIA_TYPES[fmt],
                    headers={"X-Voice": voice_name})
            return error_response(500, job.get("error", "synthesis failed"),
                                  type_="server_error",
                                  code="synthesis_failed")

    # Either too long to wait for, or the wait ran out. The job is untouched
    # and still queued, so nothing has been wasted — Location points at where
    # it will appear.
    estimate = _estimate(chars)
    return JSONResponse(
        status_code=202,
        headers={"Location": f"/jobs/{job_id}",
                 "Retry-After": str(max(1, estimate)),
                 "X-Voice": voice_name},
        content={"id": job_id, "status": "queued",
                 "queued_ahead": queue.qsize() - 1,
                 "estimated_seconds": estimate,
                 "audio_url": f"/jobs/{job_id}/audio",
                 "message": "This input is too long to answer inside one "
                            "request on this hardware. The audio is being "
                            "generated; collect it from audio_url, or send "
                            "stream_format='sse' to receive it as it is made. "
                            "This 202 is a documented deviation from OpenAI's "
                            "contract — see the README."},
    )


def _sync_budget() -> float:
    """Seconds of compute a synchronous request may still spend.

    The timeout, less what is already queued, less the model load if it is not
    resident. The old code compared a character count against a constant and
    ignored both, so a request UNDER the documented threshold turned into a
    202 whenever the queue was busy or the model was cold — and the README
    table promised otherwise.
    """
    synth: Synth | None = state.get("synth")  # type: ignore[assignment]
    cold = 0.0 if (synth is not None and synth.loaded) else COLD_LOAD_SECONDS
    return SYNC_TIMEOUT - _backlog_seconds() - cold


# ------------------------------------------------------------------- SSE --
#
# Framing rules, verified against openai-python 3.6.0's SSEDecoder:
#
#   * every event ends with a BLANK LINE, the last one included. A final event
#     terminated by a single \n is silently DROPPED, with no error;
#   * two data: lines with no blank line between them are joined into one
#     event and then fail to parse;
#   * a line starting with `:` is a comment and is ignored — which is what the
#     keepalives below are;
#   * a top-level `error` key makes the client raise APIError and stop, which
#     is the only way to report a failure after the 200 headers have gone.
#
# Bare `data:` frames, with no `event:` name line. The schema models the JSON
# payload only and gives no event field, and the one verbatim OpenAI audio SSE
# transcript in the same spec — the transcription example — uses bare data
# lines. openai-python dispatches on the JSON `type` either way. No `[DONE]`
# sentinel is sent: nothing authoritative says OpenAI emits one for this
# endpoint, and inventing a frame is worse than omitting one both SDKs treat
# as optional.


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_response(req: SpeechRequest, segments: list[tuple[str, float]],
                  text: str, voice_name: str, reference: str | None,
                  fmt: str) -> StreamingResponse:
    stream = Stream(loop=asyncio.get_running_loop())
    job_id = _enqueue(
        segments=segments, text=text, language=req.language,
        exaggeration=req.exaggeration, cfg_weight=req.cfg_weight,
        temperature=req.temperature, voice=voice_name, reference=reference,
        fmt=fmt, stream=stream)
    log.info("%s streaming %d chunks as %s", job_id[:8], len(segments), fmt)
    return StreamingResponse(
        _sse_events(job_id, stream),
        media_type="text/event-stream",
        headers={
            # no-store as well as no-cache: an intermediary that cached a
            # partial stream would replay someone else's audio.
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would hold
            # every delta until the response ended and turn a stream back into
            # the silence it exists to remove.
            "X-Accel-Buffering": "no",
            "X-Voice": voice_name,
            "X-Job-Id": job_id,
        })


async def _sse_events(job_id: str, stream: Stream) -> AsyncIterator[str]:
    """The frames themselves. One delta per encoded chunk, then one done.

    The audio in each delta is a slice of a single encode, so concatenating
    every decoded delta reproduces the buffered body byte for byte — for wav
    and flac, everything except the length fields in the header, which are not
    knowable until the last sentence is generated. app/encoders.py has the
    measured diff.
    """
    # Before anything is generated, so the headers reach the client now rather
    # than when the first sentence is ready. A comment line is legal SSE and
    # ignored by every decoder.
    yield ": tts-long stream open\n\n"
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(stream.queue.get(),
                                                       SSE_KEEPALIVE)
            except TimeoutError:
                # Waiting on the model to load, on a job ahead in the queue, or
                # on a long sentence. Concurrent callers serialise: this
                # service generates one job at a time, so a second stream sits
                # here producing keepalives until the first finishes.
                yield ": keepalive\n\n"
                continue
            if kind == "delta":
                yield _frame({"type": "speech.audio.delta",
                              "audio": base64.b64encode(payload).decode("ascii")})
            elif kind == "done":
                yield _frame({"type": "speech.audio.done", "usage": payload})
                return
            else:
                # The only in-band error channel there is once 200 has gone
                # out. openai-python raises APIError(message=...) off this.
                yield _frame({"error": {"message": str(payload),
                                        "type": "server_error",
                                        "param": None,
                                        "code": "synthesis_failed"}})
                return
    finally:
        job = jobs.get(job_id)
        if job is not None and job["status"] in {"queued", "running"}:
            # The client hung up. Cancel rather than spend ten more minutes of
            # one CPU on audio nobody is holding a socket for; what has already
            # been generated is still written to disk and still collectable
            # from /jobs.
            job["cancelled"] = True
            log.info("%s client disconnected, cancelling", job_id[:8])
