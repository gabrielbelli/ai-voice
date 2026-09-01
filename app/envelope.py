"""OpenAI's error envelope, with all four fields, on every /v1 response.

`CreateSpeechRequest`'s error schema requires four properties — `type`,
`message`, `param` and `code` — of which `param` and `code` are
required-but-nullable. voice_common.errors.error_response builds three: it
omits `param` entirely. Six error responses were checked on the deployed
instance and none carried the field, including the ones that knew perfectly
well which parameter was at fault ("voice: Input should be a valid string"
could have said `param: "voice"` and did not).

So this module is voice_common.errors plus `param`, and it is deliberately a
SUPERSET rather than a fork:

  * `error_response` here takes the same keyword-only `type_` and `code`, for
    the same reason they are keyword-only there — two sibling repos once passed
    those two strings positionally in opposite orders and produced a
    valid-looking envelope with the wrong field in each;
  * `ApiError` here subclasses voice_common's, so a `raise ApiError` caught by
    either handler renders the same way;
  * the RequestValidationError handler keeps voice_common's behaviour and adds
    the field name to `param`.

**It belongs upstream.** It is here rather than in voice-common because
requirements.txt pins voice-common by commit SHA to a GitHub tarball, and a
pin cannot name a commit that has not been published; moving it up is a
one-line diff at the next bump, and the tests below move with it.

Two things voice-common does not have at all, and both are wire defects rather
than tidiness:

  * **404 and 405 under /v1 escaped the envelope entirely.** install_errors
    registers handlers for ApiError and RequestValidationError, not for
    StarletteHTTPException, so `GET /v1/audio/speech` answered
    `{"detail":"Method Not Allowed"}` and `POST /v1/audio/transcriptions`
    answered `{"detail":"Not Found"}`. openai-python reads no message off that
    shape and reports a bare "unknown error".
  * **An unhandled exception under /v1 was a plain-text 500.** Same silence,
    at the moment the caller most needs a sentence.

The native routes keep FastAPI's `{"detail": ...}` and its 422 throughout.
/v1 is a compatibility boundary; /jobs is an older contract with clients that
already parse it.
"""

from __future__ import annotations

import logging

import voice_common.auth
import voice_common.errors
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import (http_exception_handler,
                                        request_validation_exception_handler)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from voice_common.errors import v1_path

__all__ = ["ApiError", "error_response", "install_openai_errors"]

log = logging.getLogger("tts-long.envelope")

# Codes openai-python surfaces as `.code`. voice_common's table plus the one
# it had no reason to carry: an unknown property is a 400 in OpenAI's own API
# ("Unrecognized request argument supplied: stream"), and telling a client
# which field it invented is the whole point of `param`.
_VALIDATION_CODES = {
    "missing": "missing_required_parameter",
    "extra_forbidden": "unknown_parameter",
}


def error_response(status: int, message: str, *,
                   type_: str = "invalid_request_error",
                   code: str | None = None,
                   param: str | None = None) -> JSONResponse:
    """An error in OpenAI's envelope. All four fields, `param` included.

    `param` and `code` are serialised even when null, because the schema marks
    them required-but-nullable: a client generated from it may read the key
    rather than test for its presence.
    """
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": type_,
                                           "param": param,
                                           "code": code}})


class ApiError(voice_common.errors.ApiError):
    """voice_common's ApiError with the parameter name it could not carry."""

    def __init__(self, status: int, message: str, *,
                 type_: str = "invalid_request_error",
                 code: str | None = None,
                 param: str | None = None) -> None:
        super().__init__(status, message, type_=type_, code=code)
        self.param = param


def _render(exc: voice_common.errors.ApiError) -> Response:
    response = error_response(exc.status, exc.message, type_=exc.type_,
                              code=exc.code, param=getattr(exc, "param", None))
    if exc.status == 401:
        # RFC 9110 wants a challenge on a 401. OpenAI's own API omits it;
        # sending it costs nothing and keeps generic HTTP clients honest.
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def install_openai_errors(app: FastAPI) -> None:
    """Register the four handlers, and give the shared 401 the same envelope.

    Call INSTEAD of voice_common.errors.install_errors, and after auth.install:
    the 401 is emitted by the shared authentication middleware, which has
    nothing above it to catch a raise and therefore calls `error_response`
    directly. Rebinding the name that middleware imported is what stops one
    /v1 error out of the set from being a three-field envelope while the rest
    carry four. It is one name, rebound once, at import — and the conformance
    test asserts the 401 body has all four keys, so if a future voice-common
    carries `param` itself and this shim is deleted, nothing silently regresses.
    """
    voice_common.errors.error_response = error_response  # type: ignore[assignment]
    voice_common.auth.error_response = error_response  # type: ignore[assignment]

    @app.exception_handler(voice_common.errors.ApiError)
    async def _api_error(request: Request,
                         exc: voice_common.errors.ApiError) -> Response:
        del request
        return _render(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> Response:
        """A rejected body in the envelope, naming the field, but only under /v1.

        A body pydantic rejects never reaches the route, so every envelope the
        route is careful to build was bypassed and the client got FastAPI's
        `{"detail": [...]}` with a 422 — a shape openai-python reads no message
        off at all.
        """
        if not v1_path(request.url.path):
            return await request_validation_exception_handler(request, exc)

        errors = exc.errors()
        if not errors:
            # Nothing pydantic could locate. Say so rather than guess at it.
            return error_response(400, "request body is not valid",
                                  code="invalid_value")

        first = errors[0]
        if first["type"] == "json_invalid":
            # loc is ("body", <character offset>) here, and naming a field
            # "12" is worse than not naming one at all.
            return error_response(400, "request body is not valid JSON",
                                  code="invalid_value")

        # loc[0] is always "body"; the rest names the field, with ints for
        # list indices — segments.0.text.
        field = ".".join(str(part) for part in first["loc"][1:]) or None
        if first["type"] == "extra_forbidden":
            # Worded as OpenAI words it, because a client that greps its own
            # logs for this sentence is a client that has met the real API.
            return error_response(
                400, f"Unrecognized request argument supplied: {field}",
                code="unknown_parameter", param=field)
        return error_response(
            400, f"{field or 'body'}: {first['msg']}",
            code=_VALIDATION_CODES.get(first["type"], "invalid_value"),
            param=field)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request,
                          exc: StarletteHTTPException) -> Response:
        """404 and 405 under /v1, in the envelope OpenAI answers them in.

        The message is OpenAI's own wording for an unrouted request — `Invalid
        URL (POST /v1/audio/transcriptions)` — so a client comparing strings
        against the real API sees the same sentence.
        """
        if not v1_path(request.url.path):
            return await http_exception_handler(request, exc)
        if exc.status_code in (404, 405):
            message = f"Invalid URL ({request.method} {request.url.path})"
            code = "unknown_url" if exc.status_code == 404 else "method_not_allowed"
        else:
            message = str(exc.detail)
            code = None
        response = error_response(exc.status_code, message, code=code)
        # .items(), not the dict: a 405 carries {"Allow": "POST"} and
        # iterating the mapping itself unpacks the KEY, which is a 500.
        for name, value in (getattr(exc, "headers", None) or {}).items():
            response.headers[name] = value
        return response

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Response:
        """An unhandled failure under /v1 as an envelope, not as bare text.

        Logged at exception level either way: this handler replaces Starlette's
        default, which is where the traceback used to come from.
        """
        log.exception("unhandled error on %s", request.url.path)
        if not v1_path(request.url.path):
            return Response("Internal Server Error", status_code=500,
                            media_type="text/plain")
        return error_response(500, f"internal error: {exc}",
                              type_="server_error", code="internal_error")
