"""OpenAI-compatible transcription, alongside the native route, not instead of it.

Anything that already speaks to OpenAI's /v1/audio/transcriptions — the
openai-python client, Open WebUI, a dictation app with a base-URL field —
works against this service unchanged. That is the whole reason this module
exists.

What such a client cannot see is everything /transcribe returns and this
specification has no field for: `realtime_factor`, the `raw` transcript before
glossary repair, and `repaired`, the list of terms that were rewritten. That
last one matters most — a silent substitution is worse than no substitution.
Prefer /transcribe wherever you control the client.

Three fields cannot be honoured as written, and are documented rather than
faked:

  model        Names a model this server does not choose. The engine is fixed
               at startup by STT_MODEL; loading a second one per request does
               not fit the memory the service is deployed under.
  prompt       Maps onto the glossary's decode-time hotwords. Whisper accepts
               those and measurably benefits; Parakeet, the default, has no
               mechanism for a vocabulary at all, so under it the field does
               nothing. This is a real difference between the engines, not a
               shortcut taken here. Under STT_HOTWORDS=0 it does nothing on
               either engine: the off switch governs the decoder, not just
               the glossary, or an A/B run would silently get biasing back
               through this field.
  temperature  Meaningless for Parakeet, and for Whisper the default fallback
               ladder is a deliberate reliability choice — pinning a single
               temperature disables the retry on low-confidence output.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from . import pipeline
from .auth import ApiError, require_key

FORMATS = ("json", "text", "verbose_json", "srt", "vtt")

router = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])


def install(app: FastAPI) -> None:
    """Render this route's validation failures in OpenAI's envelope too.

    A request missing `file` never reaches the handler below, so the ApiError
    path cannot shape it: FastAPI answers first, with {"detail": [...]}.
    openai-python cannot read that and reports "unknown error", which tells
    the caller nothing about the field it forgot.

    Native routes keep FastAPI's body — same rule as everywhere else here,
    that contract already has clients.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> Response:
        if not request.url.path.startswith("/v1"):
            return await request_validation_exception_handler(request, exc)

        # loc is ("body", "file"); the first element only ever repeats where
        # the failure was, which the message already says.
        detail = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:]) or 'request'}: "
            f"{err['msg']}"
            for err in exc.errors()
        ) or "invalid request"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": detail,
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        )


def _clock(seconds: float, decimal: str) -> str:
    """HH:MM:SS with the fraction separator each subtitle format insists on.

    Computed in whole milliseconds because 4.1 seconds is not 4.1 in binary:
    formatting the fractional part directly renders it as 04,099.
    """
    total = int(max(seconds, 0.0) * 1000 + 0.5)
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


# Deliberately `def`, not `async def`, for the same reason as /transcribe: the
# body is blocking CPU work, and declared async it would run ON the event loop
# and starve /health until the transcription finished.
@router.post("/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> Response:
    # Accepted and discarded, both on purpose. See the module docstring: the
    # engine is fixed at startup, and Whisper's temperature ladder is worth
    # more than obedience to a field.
    del model, temperature

    if response_format not in FORMATS:
        raise ApiError(
            400,
            f"response_format must be one of {', '.join(FORMATS)}",
            code="invalid_value",
        )

    try:
        result = pipeline.run(file.file.read(), language=language, prompt=prompt)
    except HTTPException as exc:
        # Only this side re-shapes errors. The native route's bodies are part
        # of a contract that already has clients, so they stay exactly as
        # FastAPI renders them.
        raise ApiError(
            exc.status_code,
            str(exc.detail),
            kind="invalid_request_error" if exc.status_code < 500 else "server_error",
        ) from exc

    if response_format == "text":
        return PlainTextResponse(result.text + "\n")

    # Both subtitle formats carry one cue spanning the clip, because there
    # are no segment timings to divide it by. Parakeet's decoder does not
    # expose them through onnx-asr, and on the Whisper path the segments are
    # joined before the glossary rewrites the text, after which the original
    # boundaries no longer describe what would be displayed. One true cue
    # beats several invented ones.
    if response_format == "srt":
        span = f"{_clock(0.0, ',')} --> {_clock(result.audio_seconds, ',')}"
        return PlainTextResponse(f"1\n{span}\n{result.text}\n")

    if response_format == "vtt":
        span = f"{_clock(0.0, '.')} --> {_clock(result.audio_seconds, '.')}"
        return PlainTextResponse(f"WEBVTT\n\n{span}\n{result.text}\n")

    if response_format == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            # Neither engine reports a detected language through this
            # interface: Parakeet v3 takes no language argument and returns
            # none, and the Whisper path discards faster-whisper's detection
            # when it joins the segments. What the client asked for is the
            # only answer available, and "" is the honest one when it asked
            # for nothing.
            "language": language or "",
            "duration": result.audio_seconds,
            "text": result.text,
            # No `segments` key, for the same reason the subtitle formats
            # carry one cue. An empty list would read as "no speech found",
            # which is a different claim and a false one.
        })

    return JSONResponse({"text": result.text})
