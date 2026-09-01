"""Self-hosted text-to-speech. Kokoro, CPU only.

    text or segments -> phonemise -> Kokoro -> wav, opus, mp3, aac, flac, pcm

There is one model and it stays resident: at 330 MB and 4x realtime on CPU it
is too cheap to unload. Chatterbox, the long-form alternative, is deliberately
not here — it needs 5.3 GB and runs below realtime, so it belongs behind a
separate service that can be started on demand.

Two request shapes reach the same synthesiser. `/speak` is the native one and
the one to prefer: it takes segments with explicit pauses and a voice each,
and returns the realtime factor in headers. `/v1/audio/speech` is OpenAI's
shape, kept alongside so existing clients need only a base URL change; it can
express none of those things.
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from . import auth
from .openai_api import (VOICE_ALIASES, error_response, language_for_voice,
                         resolve_voice, unmapped_aliases)
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
    synth = Synth(str(model), str(voices))
    state["synth"] = synth

    # The alias table names Kokoro voices from outside the model file, so a
    # different voices.bin could leave one of the six OpenAI names pointing at
    # nothing. Complain at load rather than at the first request that uses it —
    # the same failure the espeak wiring exists to avoid.
    missing = unmapped_aliases(synth.voices)
    if missing:
        log.warning("OpenAI voice names absent from this voices file: %s; "
                    "requests using them will be rejected", ", ".join(missing))

    log.info("ready in %.1fs, %d threads", time.monotonic() - started, THREADS)
    yield
    state.clear()


app = FastAPI(title="tts-stack",
              description="Kokoro text-to-speech, CPU only.",
              lifespan=lifespan)

# Installed on the app rather than route by route, so a route added later is
# covered without anyone having to remember to ask for it.
auth.install(app)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # A malformed body on /v1 has to come back in OpenAI's envelope, because
    # openai-python reads error.message and shows FastAPI's {"detail": [...]}
    # as a bare status code. The native routes keep FastAPI's shape: clients
    # are already written against it.
    if request.url.path.startswith("/v1/"):
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ())[1:])
        reason = first.get("msg", "invalid value")
        return error_response(400, f"{field}: {reason}" if field else reason,
                              "invalid_request_error", "invalid_value")
    return await request_validation_exception_handler(request, exc)


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    pause_after: float = Field(default=0.0, ge=0.0, le=10.0)
    # A voice is a 510 KB embedding against 310 MB of shared weights, so
    # changing it between segments costs nothing once the model is loaded.
    # Absent means the request's voice. The phonemiser language is not
    # inferred from it: `language` stays a property of the request, as it has
    # always been on this route.
    voice: str | None = None


class SpeakRequest(BaseModel):
    # Unknown fields are rejected rather than dropped. A per-segment `voice`
    # was documented for as long as it was silently ignored, and the caller
    # got the default voice back with nothing to read that said why. A typo
    # belongs in a 422, not in the audio.
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    segments: list[Segment] | None = None
    voice: str | None = None
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="wav", pattern="^(wav|opus|mp3|aac|flac|pcm)$")


class SpeechRequest(BaseModel):
    """OpenAI's /v1/audio/speech body.

    `model` is accepted and ignored: there is one model here, and rejecting
    "tts-1" would break every client that sends it while claiming to honour it
    would be a lie. `voice` is optional, unlike upstream, because a service
    with a configured default voice has an answer when the field is absent.

    Unknown fields are accepted and ignored here, unlike on /speak: OpenAI
    keeps adding them — `instructions`, `stream_format` — and a service whose
    whole purpose is to answer OpenAI's clients must not reject one for
    speaking a newer version of the dialect it claims to speak.
    """

    model: str = "kokoro"
    input: str
    voice: str | None = None
    response_format: str = Field(default="mp3",
                                 pattern="^(mp3|opus|aac|flac|wav|pcm)$")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


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
            "en_gb": [v for v in all_voices if v.startswith(("bf_", "bm_"))],
            "openai_aliases": VOICE_ALIASES}


# Lossy formats go through ffmpeg; wav, flac and pcm are libsndfile or numpy
# and stay in-process. Bitrates are set for one voice at 24 kHz, not for music.
# 32k is what opus shipped with here; mp3 and aac get double because their
# psychoacoustic models predate Opus by fifteen years and more, and neither is
# clean at 32k on speech.
_FFMPEG = {
    "opus": ("a.opus", ["-b:a", "32k"], "audio/ogg"),
    "mp3": ("a.mp3", ["-b:a", "64k"], "audio/mpeg"),
    "aac": ("a.aac", ["-b:a", "64k"], "audio/aac"),
}


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "pcm":
        # OpenAI's `pcm` is headerless 24 kHz 16-bit little-endian mono, which
        # is Kokoro's own sample rate — nothing is resampled on the way out.
        # Clipped before scaling because a sample above 1.0 wraps to the
        # opposite extreme in int16: one overshoot becomes a click, not a
        # loud sample. libsndfile does this for wav and flac already.
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(), "audio/pcm"

    name, args, mime = _FFMPEG[fmt]
    with tempfile.TemporaryDirectory() as td:
        wav, out = Path(td) / "a.wav", Path(td) / name
        sf.write(wav, audio, SAMPLE_RATE)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        *args, str(out)], check=True)
        return out.read_bytes(), mime


def _headers(duration: float, compute: float) -> dict[str, str]:
    return {"X-Audio-Seconds": f"{duration:.2f}",
            "X-Compute-Seconds": f"{compute:.2f}",
            "X-Realtime-Factor": f"{duration / compute:.1f}" if compute else "0"}


# Deliberately `def`, not `async def`: synthesis is blocking CPU work, and on
# the event loop it would starve /health during any sustained load.
@app.post("/speak")
def speak(req: SpeakRequest) -> Response:
    synth = state.get("synth")
    if not synth:
        raise HTTPException(503, "model still loading")
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")

    # Accepts the six OpenAI names as well, so a caller that learned them from
    # /v1/audio/speech does not have to unlearn them to use segments.
    voice = resolve_voice(req.voice, synth.voices, DEFAULT_VOICE)  # type: ignore[attr-defined]
    if voice is None:
        raise HTTPException(
            400, f"unknown voice {(req.voice or DEFAULT_VOICE)!r}; see GET /voices")
    language = req.language or DEFAULT_LANG

    # Each segment may name its own voice, resolved the same way and falling
    # back to the request's. Resolved here rather than in the synthesiser so
    # that an unknown name is a 400 naming the segment, not a failure part way
    # through a long synthesis.
    segments: list[tuple[str, float, str]] = []
    for index, segment in enumerate(req.segments or ()):
        seg_voice = resolve_voice(segment.voice, synth.voices, voice)  # type: ignore[attr-defined]
        if seg_voice is None:
            raise HTTPException(
                400,
                f"unknown voice {segment.voice!r} in segment {index}; "
                "see GET /voices")
        segments.append((segment.text, segment.pause_after, seg_voice))

    started = time.monotonic()
    try:
        if segments:
            audio = synth.speak_segments(segments, language, req.speed)  # type: ignore[attr-defined]
        else:
            audio = synth.speak(req.text or "", voice, language, req.speed)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(500, f"synthesis failed: {exc}") from exc

    compute = time.monotonic() - started
    duration = audio.size / SAMPLE_RATE
    # Handled for the same reason synthesis is: ffmpeg can be missing or can
    # fail, and outside the handler that reached the caller as a 500 with
    # nothing in it to act on.
    try:
        data, mime = _encode(audio, req.format)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(500, f"encoding to {req.format} failed: {exc}") from exc
    log.info("%.1fs audio in %.2fs (%.1fx) voice=%s",
             duration, compute, duration / compute if compute else 0.0, voice)

    return Response(content=data, media_type=mime, headers=_headers(duration, compute))


# Same reasoning as /speak: blocking work, so a worker thread rather than the
# event loop.
@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest) -> Response:
    synth = state.get("synth")
    if not synth:
        return error_response(503, "model still loading", "server_error",
                              "model_loading")

    voice = resolve_voice(req.voice, synth.voices, DEFAULT_VOICE)  # type: ignore[attr-defined]
    if voice is None:
        return error_response(
            400,
            f"Unknown voice {(req.voice or DEFAULT_VOICE)!r}. Accepted: "
            f"{', '.join(sorted(VOICE_ALIASES))}, or any Kokoro voice from "
            "GET /voices.",
            "invalid_request_error", "invalid_value")

    # OpenAI's schema allows 0.25 to 4.0, wider than the 0.5 to 2.0 /speak has
    # always accepted. Clamped to that range rather than rejected: a client
    # sending 4.0 wants fast speech, not a 400 it has no way to act on, and an
    # OpenAI client cannot be told to send something else.
    speed = min(max(req.speed, 0.5), 2.0)

    language = language_for_voice(voice, DEFAULT_LANG)

    started = time.monotonic()
    try:
        audio = synth.speak(req.input, voice, language, speed)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        return error_response(500, f"synthesis failed: {exc}",
                              "server_error", "synthesis_failed")

    compute = time.monotonic() - started
    duration = audio.size / SAMPLE_RATE
    # Handled, not left to FastAPI: an ffmpeg that is missing or fails used to
    # return the plain "Internal Server Error" body, which openai-python shows
    # as a bare status code. Every other error on this route is an envelope
    # and this one has to be as well.
    try:
        data, mime = _encode(audio, req.response_format)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        return error_response(500, f"encoding to {req.response_format} failed: {exc}",
                              "server_error", "encoding_failed")
    log.info("%.1fs audio in %.2fs (%.1fx) voice=%s openai",
             duration, compute, duration / compute if compute else 0.0, voice)

    # The realtime factor still goes out in headers. OpenAI's response has no
    # field for it and its clients ignore what they do not know, but curl and
    # the logs can still see whether the box is keeping up.
    return Response(content=data, media_type=mime, headers=_headers(duration, compute))
