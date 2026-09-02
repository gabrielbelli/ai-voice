"""The OpenAI error envelope, with one argument order.

openai-python reads `error.message` and shows it to the caller. Handed
anything else — FastAPI's own `{"detail": ...}` included — it reports a bare
"unknown error", which tells the caller nothing about the field it forgot.
Every service here claims to answer OpenAI's clients, so every /v1 error it
emits has to be in this shape.

There are four incompatible copies of that shape in the estate today:

  stt-stack/app/auth.py        ApiError + its exception handler
  tts-stack/app/openai_api.py  error_response(status, message, type_, code)
  tts-long/app/main.py         _error(status, message, code, type_)
  tts-long/app/auth.py         unauthorised(), a hardcoded 401 body

Note the middle two: `code` and `type_` are swapped between sibling repos that
are otherwise copy-pasted from one another. Nothing catches that — both are
strings, both produce a valid-looking envelope, and the only symptom is a
client reading "invalid_api_key" out of the field that should name a category.
So here the two are keyword-only. A positional call cannot compile the swap in,
and a copied line from either repo fails loudly at the first test rather than
quietly on the wire.

Both shapes are provided because both call sites are real: `raise ApiError`
from a route handler, `return error_response(...)` from middleware, where
there is nothing above to catch a raise. The ApiError handler renders through
error_response, so the two cannot drift apart.

This module also owns the RequestValidationError handler, taken from
tts-long/app/main.py, which is the best of the three: it guards on the /v1
prefix, special-cases json_invalid, handles the empty-errors case, drops
loc[0] from the field path, and maps `missing` to a code a client can branch
on. It answers **400**, which fixes a genuine wire-contract drift: stt-stack
answers 422 to a bad /v1 body where the other two — and OpenAI itself — answer
400, so a client written against the real API mishandles the one service in
the estate that claims to imitate it most closely.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

__all__ = ["ApiError", "error_response", "install_errors", "v1_path"]

# Codes openai-python surfaces as `.code` on the exception it raises. Kept
# distinct so a caller can tell "you left out `input`" from "9 is not a valid
# exaggeration" without parsing the message.
_VALIDATION_CODES = {"missing": "missing_required_parameter"}


def error_response(status: int, message: str, *,
                   type_: str = "invalid_request_error",
                   code: str | None = None) -> JSONResponse:
    """An error in OpenAI's envelope. `type_` and `code` are keyword-only.

    Keyword-only on purpose: see the module docstring. Two sibling repos pass
    these two strings positionally in opposite orders, and the resulting body
    is valid JSON either way, so the mistake reaches production intact.

    Only /v1 routes should use this. The native routes' `{"detail": ...}`
    bodies are part of a contract that already has clients, and reshaping them
    would break callers to tidy up a compatibility layer they never touch.
    """
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": type_,
                                           "code": code}})


class ApiError(Exception):
    """An error rendered in OpenAI's envelope instead of FastAPI's default.

    For route handlers, where raising is the natural control flow. Middleware
    has nothing above it to catch a raise and returns error_response instead;
    the handler installed below renders this THROUGH error_response so the two
    paths cannot produce different bodies.
    """

    def __init__(self, status: int, message: str, *,
                 type_: str = "invalid_request_error",
                 code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type_ = type_
        self.code = code


def v1_path(path: str) -> bool:
    """Is this path part of the OpenAI compatibility surface?

    `/v1` exactly, or anything below it. Written out rather than a bare
    startswith("/v1") because that also matches a future `/v1beta` or
    `/v1x`, and reshaping an unrelated route's errors is the sort of thing
    nobody notices until a client does.
    """
    return path == "/v1" or path.startswith("/v1/")


def install_errors(app: FastAPI) -> None:
    """Register both handlers on the app. Call once, before serving."""

    @app.exception_handler(ApiError)
    async def _render(request: Request, exc: ApiError) -> Response:
        del request
        response = error_response(exc.status, exc.message,
                                  type_=exc.type_, code=exc.code)
        if exc.status == 401:
            # RFC 9110 wants a challenge on a 401. OpenAI's own API omits it;
            # sending it costs nothing and keeps generic HTTP clients honest.
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> Response:
        """Put a rejected body into the OpenAI envelope, but only under /v1.

        A body pydantic rejects never reaches the handler, so every envelope
        that handler is careful to build was bypassed and the client got
        FastAPI's `{"detail": [...]}` with a 422. openai-python reads no
        message off that shape and raises a bare APIStatusError, which is the
        same silence a missing `input` deserved a sentence for.

        The native routes keep FastAPI's own handler untouched: /v1 is a
        compatibility boundary, and something out there already parses
        `detail` on the routes that are not one.
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
        field = ".".join(str(part) for part in first["loc"][1:]) or "body"
        return error_response(
            400, f"{field}: {first['msg']}",
            code=_VALIDATION_CODES.get(first["type"], "invalid_value"))
