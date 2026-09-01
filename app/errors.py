"""The /v1 error envelope, completed.

voice_common owns the shape three services share, and this module does not
replace it: it raises voice_common's ApiError, renders through its
error_response, and reuses its v1_path. What it adds is the two things the
specification requires that the shared package does not yet emit — and the
reason they are added here rather than there is the pin: requirements.txt names
voice-common by an immutable tarball SHA, so a change in that repository is not
a change in this image until the pin moves.

  param   `Error` requires ALL FOUR of type, message, param and code. `param`
          and `code` are required-but-NULLABLE — present as JSON null, not
          absent. voice_common builds three keys, so `param` was missing from
          every error this service has ever produced, and a client reading
          `err.param` to find out which field it got wrong read None whether or
          not the server knew the answer.

  404/405 install_errors registers no StarletteHTTPException handler, so an
          unknown /v1 path and a wrong method on a known one both leaked
          FastAPI's `{"detail": "Not Found"}`. openai-python reads no message
          off that shape and reports a bare "unknown error", which is the same
          silence a missing route deserved a sentence for.

Both fixes stop at /v1. The native routes' `{"detail": ...}` bodies are a
contract that already has clients, and reshaping them to tidy up a
compatibility layer those clients never touch is how a working deployment gets
broken by a refactor.

The backfill middleware is the belt to that pair of braces. voice_common's API
key middleware builds its own 401 body and returns it from *outside* the
exception handlers, where nothing below can reach it, so the one error this
service emits most often would otherwise still be missing `param`. Patching
error bodies on the way out is the only place that covers every path, including
one added later by someone who never reads this file.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from voice_common.errors import ApiError as _BaseApiError
from voice_common.errors import error_response, v1_path

log = logging.getLogger("stt-stack.errors")

# Codes openai-python surfaces as `.code`. Every one used here is a code the
# real API sends, so a client can branch on the same strings against both.
CODE_MISSING = "missing_required_parameter"
CODE_INVALID = "invalid_value"
CODE_UNKNOWN_PARAM = "unknown_parameter"
CODE_UNSUPPORTED_PARAM = "unsupported_parameter"
CODE_UNSUPPORTED_VALUE = "unsupported_value"
CODE_UNKNOWN_URL = "unknown_url"

# 64 KB. An error body is a few hundred bytes; anything larger under /v1 with a
# 4xx and a JSON content type is not an envelope, and buffering it to look would
# be the wrong trade.
_MAX_PATCHED_BODY = 64 * 1024


class ApiError(_BaseApiError):
    """voice_common's ApiError, plus the `param` the specification requires.

    A subclass rather than a fork: the handler registered below renders both,
    so code that raises either shape produces the same four keys.
    """

    def __init__(self, status: int, message: str, *,
                 type_: str = "invalid_request_error",
                 code: str | None = None,
                 param: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(status, message, type_=type_, code=code)
        self.param = param
        # Retry-After on a 429 is the one header the specification enumerates
        # for this path, and the one an OpenAI client's backoff reads.
        self.headers = headers or {}


def _envelope(status: int, message: str, *, type_: str, code: str | None,
              param: str | None) -> Response:
    """error_response, with `param` inserted into the body it builds.

    Built by editing voice_common's body rather than by writing a fourth copy
    of the envelope, so the three keys it owns cannot drift from the one added
    here.
    """
    response = error_response(status, message, type_=type_, code=code)
    body = json.loads(response.body)
    body["error"]["param"] = param
    return _rewrite(response, body)


def _rewrite(response: Response, body: dict[str, object]) -> Response:
    """Put `body` back on a response, Content-Length included."""
    raw = json.dumps(body).encode("utf-8")
    response.body = raw
    response.headers["content-length"] = str(len(raw))
    return response


def install(app: FastAPI) -> None:
    """Register the /v1 handlers and the backfill. Call after install_errors."""

    @app.exception_handler(_BaseApiError)
    async def _render(request: Request, exc: _BaseApiError) -> Response:
        del request
        response = _envelope(exc.status, exc.message, type_=exc.type_,
                             code=exc.code, param=getattr(exc, "param", None))
        for header, value in getattr(exc, "headers", {}).items():
            response.headers[header] = value
        if exc.status == 401:
            # RFC 9110 wants a challenge on a 401. Kept from voice_common's
            # handler, which this one replaces.
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> Response:
        """voice_common's handler, with the rejected field named in `param`.

        Reached only for what this service does not parse by hand — `file`,
        in practice. Every other field is validated in openai_api.py, which
        can say something more useful than pydantic can.
        """
        if not v1_path(request.url.path):
            return await request_validation_exception_handler(request, exc)

        errors = exc.errors()
        if not errors:
            return _envelope(400, "request body is not valid",
                             type_="invalid_request_error",
                             code=CODE_INVALID, param=None)

        first = errors[0]
        if first["type"] == "json_invalid":
            return _envelope(400, "request body is not valid JSON",
                             type_="invalid_request_error",
                             code=CODE_INVALID, param=None)

        # loc[0] is always "body"; the rest names the field.
        field = ".".join(str(part) for part in first["loc"][1:]) or None
        code = CODE_MISSING if first["type"] == "missing" else CODE_INVALID
        return _envelope(400, f"{field or 'body'}: {first['msg']}",
                         type_="invalid_request_error", code=code, param=field)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Response:
        """404, 405 and every other bare HTTPException raised under /v1.

        Native routes keep FastAPI's own renderer: `{"detail": ...}` there is
        load-bearing for clients that predate this module.
        """
        if not v1_path(request.url.path):
            return await http_exception_handler(request, exc)

        if exc.status_code == 404:
            # The real API's wording, so a client's log line reads the same
            # against either server.
            message = f"Invalid URL ({request.method} {request.url.path})"
            code: str | None = CODE_UNKNOWN_URL
        elif exc.status_code == 405:
            message = f"Invalid method ({request.method} {request.url.path})"
            code = None
        else:
            message = str(exc.detail)
            code = None

        response = _envelope(
            exc.status_code, message,
            type_="server_error" if exc.status_code >= 500 else "invalid_request_error",
            code=code, param=None)
        for header, value in (exc.headers or {}).items():
            response.headers[header] = value
        return response

    app.add_middleware(EnvelopeBackfill)


class EnvelopeBackfill:
    """Give every /v1 error body all four keys, whoever built it.

    Raw ASGI rather than BaseHTTPMiddleware on purpose: BaseHTTPMiddleware
    consumes the response as a stream and buffers it, which would turn the SSE
    transcript stream into a single delivery at the end — the exact defect the
    streaming work here exists to avoid. This one holds nothing back unless the
    status is 4xx/5xx and the body is JSON.
    """

    def __init__(self, app) -> None:  # noqa: ANN001 - ASGI app, no useful type
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http" or not v1_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        state: dict[str, object] = {"patching": False, "start": None,
                                    "body": bytearray()}

        async def send_wrapper(message) -> None:  # noqa: ANN001
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = message.get("headers", [])
                json_body = any(
                    key.lower() == b"content-type"
                    and value.lower().startswith(b"application/json")
                    for key, value in headers
                )
                if status >= 400 and json_body:
                    # Hold the head: patching the body changes Content-Length,
                    # and the head cannot be recalled once it has gone.
                    state["patching"] = True
                    state["start"] = message
                    return
                await send(message)
                return

            if message["type"] != "http.response.body" or not state["patching"]:
                await send(message)
                return

            body: bytearray = state["body"]  # type: ignore[assignment]
            body += message.get("body", b"")
            if message.get("more_body", False):
                if len(body) <= _MAX_PATCHED_BODY:
                    return
                # Too large to be an envelope. Give up and let it through
                # unchanged rather than buffer an unbounded response.
                state["patching"] = False
                await send(state["start"])  # type: ignore[arg-type]
                await send({"type": "http.response.body", "body": bytes(body),
                            "more_body": True})
                return

            await _flush(send, state["start"], bytes(body))  # type: ignore[arg-type]

        await self.app(scope, receive, send_wrapper)


async def _flush(send, start, body: bytes) -> None:  # noqa: ANN001
    """Send a held error response, with `param` added if it was missing."""
    patched = body
    try:
        parsed = json.loads(body)
        error = parsed["error"]
        if "param" not in error:
            error["param"] = None
            patched = json.dumps(parsed).encode("utf-8")
    except (ValueError, KeyError, TypeError):
        # Not an envelope. Nothing to complete, and inventing one would hide
        # whatever really went wrong.
        pass

    headers = [
        (key, value) for key, value in start.get("headers", [])
        if key.lower() != b"content-length"
    ]
    headers.append((b"content-length", str(len(patched)).encode("ascii")))
    await send({**start, "headers": headers})
    await send({"type": "http.response.body", "body": patched,
                "more_body": False})
