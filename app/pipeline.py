"""The transcription pipeline, separate from the shape it is returned in.

    audio -> VAD -> recogniser -> glossary repair -> text

Two routes share this: the native /transcribe, which returns everything the
run measured, and /v1/audio/transcriptions, which returns the subset OpenAI's
specification has fields for. The work lives here so the compatibility layer
cannot slowly become a second pipeline that drifts from the first — which is
the usual way these shims rot.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from . import asr, glossary

SAMPLE_RATE = 16_000

MODEL = os.getenv("STT_MODEL", "parakeet")
GLOSSARY_PATH = os.getenv("STT_GLOSSARY", "/etc/stt-stack/glossary.txt")
# The one knob that matters on a shared host. ONNX Runtime and CTranslate2
# both size their pools from the host core count, not the cgroup, so a
# container CPU limit without this leaves threads fighting for their own
# slice. See the README.
THREADS = int(os.getenv("STT_THREADS", "4"))
VAD_ENABLED = os.getenv("STT_VAD", "1") not in {"0", "false", "no"}
# Off switch for decode-time biasing, so a benchmark can separate what the
# vocabulary contributes from what the model does. Whisper only — Parakeet has
# no such mechanism. Text repair is unaffected either way.
HOTWORDS_ENABLED = os.getenv("STT_HOTWORDS", "1") not in {"0", "false", "no"}

log = logging.getLogger("stt-stack")

state: dict[str, object] = {}


@dataclass(frozen=True)
class Result:
    """Everything one run measured. Rounded here, so both routes agree."""

    text: str
    raw: str
    repaired: list[str]
    model: str
    audio_seconds: float
    speech_seconds: float
    compute_seconds: float
    realtime_factor: float


def start() -> None:
    """Load the glossary, the recogniser and, if enabled, the VAD."""
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

    terms, hotword_list = glossary.load(GLOSSARY_PATH)
    state["rules"] = glossary.compile_rules(terms)
    # Whisper takes the glossary at decode time, which beats repairing the
    # text afterwards. Parakeet cannot, so for it the glossary is repair only.
    #
    # Note this is not free: measured on audio containing NONE of its terms, a
    # glossary raised WER by 12% on Parakeet and 28% on Whisper. Keep the list
    # relevant to what is actually being said.
    hotwords = (", ".join(hotword_list) or None) if HOTWORDS_ENABLED else None

    state["asr"] = asr.build(THREADS, hotwords)

    if VAD_ENABLED:
        from .vad import Vad  # noqa: PLC0415

        state["vad"] = Vad()
        log.info("vad ready")


def stop() -> None:
    state.clear()


def loaded() -> object | None:
    """The recogniser, or None while it is still loading. For /health."""
    return state.get("asr")


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


def run(data: bytes, language: str | None = None, prompt: str | None = None) -> Result:
    """Transcribe one clip. Blocking CPU work — never call this on the loop.

    `prompt` extends the glossary's decode-time vocabulary for this request.
    Whisper honours it; Parakeet has no mechanism for a vocabulary and
    silently cannot. See asr.py.
    """
    if "asr" not in state:
        raise HTTPException(503, "model still loading")

    samples = _decode(data)
    if samples.size == 0:
        raise HTTPException(400, "audio contains no samples")
    audio_seconds = samples.size / SAMPLE_RATE

    started = time.monotonic()
    if "vad" in state:
        samples, _kept = state["vad"].speech_only(samples)  # type: ignore[attr-defined]
    speech_seconds = samples.size / SAMPLE_RATE

    raw = state["asr"].transcribe(samples, language, prompt)  # type: ignore[attr-defined]
    text, repaired = glossary.apply(raw, state["rules"])  # type: ignore[arg-type]
    compute = time.monotonic() - started

    log.info("%.1fs audio, %.1fs speech, %.2fs compute (%.1fx), repaired=%s",
             audio_seconds, speech_seconds, compute,
             audio_seconds / compute if compute else 0.0, repaired or "none")

    return Result(
        text=text,
        raw=raw,
        repaired=repaired,
        model=MODEL,
        audio_seconds=round(audio_seconds, 2),
        speech_seconds=round(speech_seconds, 2),
        compute_seconds=round(compute, 2),
        realtime_factor=round(audio_seconds / compute, 1) if compute else 0.0,
    )
