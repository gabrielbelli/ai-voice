"""API key authentication, and the decision to default to open.

This service is already running on a LAN with callers that have no key. An
upgrade that started refusing them would turn a feature into an outage, so an
unset `TTS_API_KEYS` leaves authentication **disabled** and says so at WARNING
level on every start. Refusing to boot is the tidier position and the worse
one: it breaks working deployments to protect a network that was already open,
and a warning in the log is what actually gets read.

Keys are compared with `hmac.compare_digest`. A LAN is not a threat-free
network — anything that can reach the port can time the response — and a
constant-time comparison of a short string costs nothing worth measuring.

`/health` stays open because the image's HEALTHCHECK calls it and has no key.
Nothing else is exempt: `/docs` and `/openapi.json` need a key too when one is
configured, which is inconvenient in a browser and correct.

An unset `TTS_API_KEYS` disables authentication; a *set* one that parses to no
keys at all — `','` — refuses to start rather than quietly doing the same.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def _load_keys() -> list[str]:
    """Parse TTS_API_KEYS, refusing a value that asks for auth and gives none.

    Unset or blank means disabled, for the reason in the module docstring.
    But `TTS_API_KEYS=','` is not blank: separators with nothing between them
    survived the filter as an empty list, which disabled authentication and
    then logged that the variable was unset — so an operator who fat-fingered
    the value read a warning that told them their own configuration was not
    there, on a service that was now open. A value that was set and yielded no
    key is a mistake, and the only safe reading of it is to stop.
    """
    raw = os.getenv("TTS_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if raw.strip() and not keys:
        raise RuntimeError(
            f"TTS_API_KEYS is set to {raw!r}, which contains no key. Set it to "
            "a comma-separated list of keys, or unset it to run without "
            "authentication.")
    return keys


KEYS = _load_keys()

# The HEALTHCHECK in the Containerfile has no credentials to offer.
OPEN_PATHS = {"/health"}


def enabled() -> bool:
    return bool(KEYS)


def authorised(header: str | None) -> bool:
    if not KEYS:
        return True
    if not header or not header.lower().startswith("bearer "):
        return False
    presented = header[7:].strip().encode()
    # The list is built before `any` sees it, so every configured key is
    # compared even once one has matched. Short-circuiting would make the
    # response time depend on which key was presented, which is exactly the
    # leak compare_digest is here to close.
    return any([hmac.compare_digest(k.encode(), presented) for k in KEYS])


def unauthorised() -> JSONResponse:
    """The OpenAI error envelope, because OpenAI clients parse it.

    openai-python raises AuthenticationError with a usable message off this
    shape and a bare `{"detail": ...}` off nothing.
    """
    return JSONResponse(
        status_code=401,
        content={"error": {
            "message": "Incorrect API key provided. Send it as "
                       "'Authorization: Bearer <key>'.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }},
    )


def install(app: FastAPI) -> None:
    """Attach the check as middleware, not as a per-route dependency.

    A dependency has to be remembered on every route added from here on;
    middleware cannot be forgotten. The exemption is a single named path
    rather than a decorator someone has to spot.
    """

    @app.middleware("http")
    async def _api_key(request: Request, call_next):  # noqa: ANN001, ANN202
        if request.url.path in OPEN_PATHS or authorised(request.headers.get("authorization")):
            return await call_next(request)
        return unauthorised()


def log_state(log: logging.Logger) -> None:
    if KEYS:
        log.info("api key auth enabled, %d key(s) accepted", len(KEYS))
    else:
        log.warning("TTS_API_KEYS is unset or empty: every request is accepted "
                    "without a key, including /v1/audio/speech")
