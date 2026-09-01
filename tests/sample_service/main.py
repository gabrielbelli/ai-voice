"""A minimal consumer, built the way the three real services will be.

Its job is to be the app voice_common.conformance runs against, so that the
suite the package ships is proven against a real FastAPI app here rather than
only in the consumers' CI.

Note the install order: health BEFORE auth, and auth given no public paths of
its own. That is the point of the shared exemption set — the two calls may
come in either order and the route still cannot be locked out. Real services
can keep the default and get /health exempted either way.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import Field

from voice_common import auth, errors, health, logging as voice_logging
from voice_common.models import OpenAISpeechRequest, Segment

log = voice_logging.setup("sample-service", "TTS")

app = FastAPI(title="sample-service")
errors.install_errors(app)
health.install_health(app, details=lambda: {"threads": 4})
auth.install(app, "TTS_API_KEYS", extra_public_paths=())


class SpeechRequest(OpenAISpeechRequest):
    response_format: str = Field(default="wav", pattern="^(wav|pcm)$")


class SpeakRequest(OpenAISpeechRequest):
    segments: list[Segment] | None = None


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> dict[str, object]:
    return {"format": req.response_format, "input": req.input}


@app.post("/speak")
async def speak(req: SpeakRequest) -> dict[str, object]:
    return {"input": req.input}
