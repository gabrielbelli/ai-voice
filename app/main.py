"""Long-form text-to-speech. Chatterbox on CPU, as a job queue.

Kokoro (tts-stack) answers requests. This does not: at roughly 0.21x realtime
a ten-minute recording takes three quarters of an hour, so an HTTP request
that waits for the audio would time out long before it arrived.

So /jobs accepts work and returns immediately with an id. One job runs at a
time — the model is 6.5 GB and concurrency would only make both jobs slower
while doubling the memory.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts-long")

jobs: dict[str, dict] = {}
queue: Queue = Queue()
state: dict[str, object] = {}


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
            path = OUT_DIR / f"{job_id}.wav"
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
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["synth"] = Synth(idle_timeout=IDLE_TIMEOUT, threads=THREADS)
    # One worker on purpose. The model is 6.5 GB and generation is sequential,
    # so a second job would double memory and slow both.
    threading.Thread(target=_worker, daemon=True).start()
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
def health() -> dict[str, object]:
    synth: Synth = state["synth"]  # type: ignore[assignment]
    return {
        "status": "ok",
        "model_loaded": synth.loaded,
        "threads": THREADS,
        "queued": queue.qsize(),
        "running": sum(1 for j in jobs.values() if j["status"] == "running"),
    }


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest) -> dict[str, object]:
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")
    job_id = str(uuid.uuid4())
    words = len((req.text or " ".join(s.text for s in (req.segments or []))).split())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        "text": req.text, "language": req.language,
        "segments": [(s.text, s.pause_after) for s in req.segments] if req.segments else None,
        "exaggeration": req.exaggeration, "cfg_weight": req.cfg_weight,
        "temperature": req.temperature,
    }
    queue.put(job_id)
    # ~150 wpm speech at ~0.21x realtime. Rough, but the caller deserves to
    # know this is minutes rather than seconds before it waits on anything.
    estimate = round(words / 150 * 60 / 0.21)
    return {"id": job_id, "status": "queued", "queued_ahead": queue.qsize() - 1,
            "estimated_seconds": estimate}


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
    return FileResponse(job["path"], media_type="audio/wav",
                        filename=f"{job_id}.wav")
