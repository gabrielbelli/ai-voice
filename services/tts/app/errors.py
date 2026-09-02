"""The /v1 error envelope, with the `param` key the schema puts in REQUIRED.

    {"error": {"message": …, "type": …, "param": …, "code": …}}

`voice_common.errors` builds three of those four. `param` appears nowhere in
the module, so every error this service emitted was schema-invalid — and the
information was available and thrown away: the unknown-voice branch knows the
field is `voice`, and the validation handler already computes a field name from
`loc[1:]` to put in the message string.

`param` is required-but-nullable. It has to be present as an explicit JSON null
when no single field is at fault, which is why every helper here takes it and
none of them omits the key.

**Why this is in tts-stack and not in voice-common, where it belongs.** The
shared package is pinned to a commit tarball in requirements.txt, so a change
there reaches this image only on the next bump, and this service is live now.
The three handlers below are written to be lifted into `voice_common.errors`
unchanged when the estate moves together; the module is small and the seam is
one import.

Two things beyond `param` are fixed here as well:

  * **404 and 405 under /v1 escaped the envelope.** `GET /v1/audio/speech`
    returned `{"detail":"Method Not Allowed"}` and `/v1/audio/transcriptions`
    returned `{"detail":"Not Found"}` — FastAPI's shape, which openai-python
    reads no message off. `voice_common.errors` already ships `v1_path()` for
    exactly this guard; `install_errors` never registered a handler that used
    it for StarletteHTTPException.
  * **The 401 the shared auth middleware builds** is the one /v1 error body
    this repo does not construct, and it has no `param` either. It is fixed by
    the middleware at the bottom, which adds the missing key on the way out
    and deletes itself the day voice-common grows one.

The native routes are untouched throughout. `{"detail": …}` is a contract that
already has clients, and /v1 is the only compatibility boundary here.
"""

from __future__ import annotations

import json
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import (http_exception_handler,
                                        request_validation_exception_handler)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from voice_common.errors import v1_path

__all__ = ["speech_error", "install_speech_errors"]

# Codes openai-python surfaces as `.code` on the exception it raises, so a
# caller can tell "you left out `input`" from "9 is not a valid speed" without
# parsing the message. Taken from voice_common.errors, which has the same map.
_VALIDATION_CODES = {"missing": "missing_required_parameter"}


def speech_error(status: int, message: str, *,
                 type_: str = "invalid_request_error",
                 code: str | None = None,
                 param: str | None = None) -> JSONResponse:
    """An error in OpenAI's envelope, all four keys always present.

    `type_`, `code` and `param` are keyword-only, for the reason
    voice_common.errors gives at length: two sibling repos once passed `code`
    and `type_` positionally in opposite orders, both produced a valid-looking
    body, and nothing caught it. A third string in the same position would be
    a third chance to make that mistake.
    """
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": type_,
                                           "param": param,
                                           "code": code}})


def _param_of(loc: tuple[object, ...]) -> str | None:
    """The parameter a pydantic error is about, or None if it is about the body.

    loc[0] is always "body"; the rest names the field, with ints for list
    indices. `voice` is normalised before validation rather than typed as a
    union precisely so this stays a flat name — see openai_api.custom_voice_id.
    """
    parts = [str(part) for part in loc[1:]]
    return ".".join(parts) or None


class _ParamKeyMiddleware:
    """Add `param: null` to a /v1 error body that was built without it.

    Pure ASGI, and installed last so that it wraps the auth middleware and can
    see the 401 that runs before routing. It buffers only bodies that are
    already known to be small: a JSON response with a 4xx or 5xx status under
    /v1. An audio response never matches, so nothing on the streaming path is
    held up by it.

    This exists solely to cover `voice_common.auth`, the one /v1 error body
    this repo does not construct. Delete it when the shared package's
    error_response takes a `param`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not v1_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        start: Message | None = None
        buffered = bytearray()

        async def _send(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                content_type = dict(message["headers"]).get(b"content-type", b"")
                if (message["status"] >= 400
                        and content_type.startswith(b"application/json")):
                    # Held back: rewriting the body changes its length, so the
                    # Content-Length header cannot go out before the body is.
                    start = message
                    return
            elif message["type"] == "http.response.body" and start is not None:
                buffered.extend(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = _with_param(bytes(buffered))
                header = dict(start)
                header["headers"] = [
                    (key, value) for key, value in header["headers"]
                    if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(body)).encode())]
                start = None
                await send(header)
                await send({"type": "http.response.body", "body": body,
                            "more_body": False})
                return
            await send(message)

        await self.app(scope, receive, _send)


def _with_param(body: bytes) -> bytes:
    """Insert `param: null` into an error envelope that lacks it."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or "param" in error:
        return body
    # Rebuilt rather than mutated in place so the key lands in the order
    # OpenAI's own bodies use: message, type, param, code.
    payload["error"] = {"message": error.get("message"),
                        "type": error.get("type"),
                        "param": None,
                        "code": error.get("code")}
    return json.dumps(payload).encode()


def install_speech_errors(app: FastAPI) -> None:
    """Register the /v1 handlers. Call once, after voice_common's install_errors.

    Later registration wins — FastAPI keeps handlers in a dict — so this
    replaces the shared RequestValidationError handler with one that fills in
    `param`, and leaves the ApiError handler alone.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> Response:
        if not v1_path(request.url.path):
            return await request_validation_exception_handler(request, exc)

        errors = exc.errors()
        if not errors:
            return speech_error(400, "request body is not valid",
                                code="invalid_value")

        first = errors[0]
        if first["type"] == "json_invalid":
            # loc is ("body", <character offset>) here, and naming a parameter
            # "12" is worse than naming none at all.
            return speech_error(400, "request body is not valid JSON",
                                code="invalid_value")

        param = _param_of(tuple(first["loc"]))
        return speech_error(
            400, f"{param or 'body'}: {first['msg']}",
            code=_VALIDATION_CODES.get(first["type"], "invalid_value"),
            param=param)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request,
                    exc: StarletteHTTPException) -> Response:
        """404 and 405 under /v1, in the envelope rather than FastAPI's shape.

        The message mirrors what api.openai.com answers an unrouted request
        with — "Invalid URL (POST /v1/nope)" — because that is the sentence
        openai-python will show, and a client comparing this service against
        the real one should not have to learn a second wording.
        """
        if not v1_path(request.url.path):
            return await http_exception_handler(request, exc)
        if exc.status_code in (404, 405):
            return speech_error(
                exc.status_code,
                f"Invalid URL ({request.method} {request.url.path})")
        return speech_error(exc.status_code, str(exc.detail))

    # Added last, so it is outermost and sees the auth middleware's 401.
    app.add_middleware(_ParamKeyMiddleware)
