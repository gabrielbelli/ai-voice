"""Self-hosted speech-to-text. One container, the whole pipeline.

    audio -> VAD -> primary ASR -> secondary ASR -> consensus -> glossary -> text

There is no LLM cleanup stage, deliberately. A model small enough to sit
beside two recognisers on a CPU box is not reliable enough to rewrite a
technical transcript: tested at 4B with an explicit prompt forbidding it, the
cleanup stage still inverted meaning, reversed pronouns, deleted content and
leaked its own reasoning. The consensus pass replaces it — flagging the words
worth doubting instead of inventing confidence about them.

The stages are separable on purpose. Splitting this into services later is a
matter of moving each module behind a socket, not restructuring the pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import asr, consensus, glossary

SAMPLE_RATE = 16_000

GLOSSARY_PATH = os.getenv("STT_GLOSSARY", "/etc/stt-stack/glossary.txt")
# The one knob that matters on a shared host. ONNX Runtime and CTranslate2
# both size their pools from the host core count, not the cgroup, so a
# container CPU limit without this leaves threads fighting for their own
# slice. See the README.
THREADS = int(os.getenv("STT_THREADS", "4"))
VAD_ENABLED = os.getenv("STT_VAD", "1") not in {"0", "false", "no"}
MARKER = os.getenv("STT_MARKER", "<{a}|{b}>")
# Off switch for the glossary's decode-time biasing, so a benchmark can
# isolate what the vocabulary contributes from what the model does. Text
# repair is unaffected — only the hotwords passed to Whisper.
HOTWORDS_ENABLED = os.getenv("STT_HOTWORDS", "1") not in {"0", "false", "no"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt-stack")

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    started = time.monotonic()

    terms, hotword_list = glossary.load(GLOSSARY_PATH)
    state["rules"] = glossary.compile_rules(terms)
    # Whisper takes the glossary at decode time, which beats repairing the
    # text afterwards: biasing the decoder can recover a word that string
    # replacement never sees, because the wrong word was never in the list.
    # Measured on real recordings, hotwords fixed every technical term and the
    # post-decode replacement never had to fire.
    hotwords = (", ".join(hotword_list) or None) if HOTWORDS_ENABLED else None
    if not HOTWORDS_ENABLED:
        log.info("hotwords DISABLED by STT_HOTWORDS")

    primary, secondary = asr.build(THREADS, hotwords)
    state["primary"] = primary
    state["secondary"] = secondary

    if VAD_ENABLED:
        from .vad import Vad  # noqa: PLC0415

        state["vad"] = Vad()
        log.info("vad ready")

    log.info("ready in %.1fs, %d threads", time.monotonic() - started, THREADS)
    yield
    state.clear()


app = FastAPI(
    title="stt-stack",
    description="VAD, two ASR models, disagreement marking and glossary repair.",
    lifespan=lifespan,
)


class Transcript(BaseModel):
    text: str
    primary: str
    secondary: str | None
    disagreements: list[dict[str, str]]
    agreement: float | None
    repaired: list[str]
    audio_seconds: float
    speech_seconds: float
    compute_seconds: float
    realtime_factor: float


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if "primary" in state else "loading",
        "primary": os.getenv("STT_PRIMARY", "large-v3"),
        "secondary": os.getenv("STT_SECONDARY", "istupakov/parakeet-tdt-0.6b-v3-onnx"),
        "vad": VAD_ENABLED,
        "hotwords": HOTWORDS_ENABLED,
        "threads": THREADS,
    }


def _decode(raw: bytes) -> np.ndarray:
    try:
        audio, rate = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(400, f"could not decode audio: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise HTTPException(
            400, f"expected {SAMPLE_RATE} Hz, got {rate} Hz — resample before sending"
        )
    return audio.mean(axis=1)


# Deliberately `def`, not `async def`. The body is blocking CPU work lasting
# tens of seconds; declared async it would run ON the event loop and starve
# every other request, /health included. A container healthcheck then fails
# during any sustained load and the orchestrator restarts a perfectly healthy
# service mid-request. As a plain `def`, FastAPI runs it in a threadpool and
# the loop stays free to answer.
@app.post("/transcribe", response_model=Transcript)
def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> Transcript:
    if "primary" not in state:
        raise HTTPException(503, "models still loading")

    samples = _decode(file.file.read())
    if samples.size == 0:
        raise HTTPException(400, "audio contains no samples")
    audio_seconds = samples.size / SAMPLE_RATE

    started = time.monotonic()

    if "vad" in state:
        samples, _kept = state["vad"].speech_only(samples)  # type: ignore[attr-defined]
    speech_seconds = samples.size / SAMPLE_RATE

    primary_text = state["primary"].transcribe(samples, language)  # type: ignore[attr-defined]

    secondary_text: str | None = None
    disagreements: list[dict[str, str]] = []
    score: float | None = None
    text = primary_text

    if state.get("secondary") is not None:
        secondary_text = state["secondary"].transcribe(samples, language)  # type: ignore[attr-defined]
        text, disagreements = consensus.merge(primary_text, secondary_text, MARKER)
        score = round(consensus.agreement(primary_text, secondary_text), 3)

    text, repaired = glossary.apply(text, state["rules"])  # type: ignore[arg-type]
    compute = time.monotonic() - started

    log.info(
        "%.1fs audio, %.1fs speech, %.2fs compute (%.1fx), agreement=%s, disagreements=%d",
        audio_seconds, speech_seconds, compute,
        audio_seconds / compute if compute else 0.0, score, len(disagreements),
    )

    return Transcript(
        text=text,
        primary=primary_text,
        secondary=secondary_text,
        disagreements=disagreements,
        agreement=score,
        repaired=repaired,
        audio_seconds=round(audio_seconds, 2),
        speech_seconds=round(speech_seconds, 2),
        compute_seconds=round(compute, 2),
        realtime_factor=round(audio_seconds / compute, 1) if compute else 0.0,
    )
