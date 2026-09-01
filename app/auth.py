"""API-key authentication for everything except /health.

Everything means everything: /openapi.json, /docs and /redoc are re-registered
in main.py behind this check, because FastAPI's own copies are Starlette
routes that no router dependency reaches, and a schema dump is a free map of
the service.

Off unless STT_API_KEYS is set, and loudly off: this service already runs on a
LAN, and an upgrade that suddenly refused every unauthenticated request would
break the deployment it was meant to protect. A WARNING at startup is the
compromise — open is allowed, quietly open is not.

Keys are compared with hmac.compare_digest, never ==. A LAN is not a trust
boundary; anything that can reach port 8000 can also time it.

The 401 body is OpenAI's error envelope rather than FastAPI's {"detail": ...},
because openai-python reads {"error": {"message": ...}} and reports an
otherwise useless "unknown error" when it finds anything else.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("stt-stack.auth")

# Read once, at import. Rotating a key is therefore a restart, which is also
# the only moment the startup line below can be trusted to describe reality.
#
# The raw value is kept because the parsed list cannot tell "unset" from
# STT_API_KEYS=" , , ": both leave KEYS empty, and announcing the second as
# the first served a typo as though it were a decision.
RAW: str | None = os.getenv("STT_API_KEYS")
KEYS: list[str] = [key.strip() for key in (RAW or "").split(",") if key.strip()]


class ApiError(Exception):
    """An error rendered in OpenAI's envelope instead of FastAPI's default.

    Raised by the auth check and by the compatibility layer. The native routes
    keep FastAPI's {"detail": ...} bodies untouched — those are part of a
    contract that already has clients.
    """

    def __init__(self, status: int, message: str, *,
                 kind: str = "invalid_request_error", code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.code = code


def install(app: FastAPI) -> None:
    """Register the renderer for ApiError. Call once, on the app."""

    @app.exception_handler(ApiError)
    async def _render(request: Request, exc: ApiError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {"message": exc.message, "type": exc.kind, "code": exc.code}
            },
            headers={"WWW-Authenticate": "Bearer"} if exc.status == 401 else None,
        )


def announce() -> None:
    """Say, at startup, whether anything is being checked.

    Refuses to serve a key list that was set and parsed to nothing. Open is
    allowed because someone chose it; a value of separators and whitespace is
    nobody's choice, and starting anyway would leave a deployment that asked
    for authentication running without it and told it was fine.
    """
    if KEYS:
        log.info("api key auth enabled, %d key(s) accepted", len(KEYS))
        return

    if RAW is not None and RAW.strip():
        raise ValueError(
            f"STT_API_KEYS={RAW!r} contains separators and whitespace but no "
            "key. Set it to a comma-separated list of keys, or unset it to "
            "serve every request unauthenticated on purpose."
        )

    log.warning(
        "STT_API_KEYS is %s: every request is accepted. Set it to a "
        "comma-separated list of keys to require Authorization: Bearer",
        "unset" if RAW is None else "empty",
    )


def require_key(authorization: str | None = Header(default=None)) -> None:
    """Reject anything not presenting a configured key. A no-op when unset."""
    if not KEYS:
        return

    scheme, _, presented = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise ApiError(401, "missing bearer token", code="invalid_api_key")

    # Encoded, because compare_digest refuses str containing non-ASCII and the
    # header is whatever the client chose to send — a stray accented character
    # would otherwise be a 500 rather than a 401.
    #
    # Accumulated rather than any(), which short-circuits on the first match
    # and so would leak which key was presented through the response time.
    candidate = presented.encode("utf-8")
    matched = False
    for key in KEYS:
        matched |= hmac.compare_digest(candidate, key.encode("utf-8"))

    if not matched:
        raise ApiError(401, "invalid API key", code="invalid_api_key")
