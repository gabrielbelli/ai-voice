"""API key authentication, off unless keys are configured.

    TTS_API_KEYS=key1,key2   ->  Authorization: Bearer key1

Unset or empty means every request is accepted, and the service says so loudly
at startup rather than either refusing to boot or running open in silence.
That asymmetry is deliberate: these services already run on a LAN with no keys
anywhere, and an upgrade that starts rejecting every existing caller is a worse
outage than a warning nobody reads.

Enforcement is middleware rather than a per-route dependency so that a route
added later is protected by default. The exception list is short and explicit,
and /health is on it because container healthchecks have no key and no way to
be given one.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, Request

from .openai_api import error_response

log = logging.getLogger("tts-stack.auth")

PUBLIC_PATHS = frozenset({"/health"})


def load_keys() -> tuple[str, ...]:
    raw = os.getenv("TTS_API_KEYS", "")
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def _authorised(header: str | None, keys: tuple[str, ...]) -> bool:
    if not header:
        return False
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False

    # Bytes, not str: compare_digest raises TypeError on a non-ASCII str, and
    # the presented value is whatever the network sent.
    offered = presented.encode("utf-8", "surrogateescape")

    ok = False
    for key in keys:
        # compare_digest rather than ==, which returns at the first differing
        # byte and so leaks the length of the matching prefix. And no early
        # break: stopping at the match would time how far down the list a
        # valid key sits, which is a slower version of the same leak.
        ok |= hmac.compare_digest(offered, key.encode("utf-8"))
    return ok


def install(app: FastAPI) -> None:
    keys = load_keys()

    if not keys:
        log.warning("TTS_API_KEYS is unset or empty: authentication is DISABLED "
                    "and every request is accepted. Set TTS_API_KEYS to a "
                    "comma-separated list of keys to require one.")
        return

    log.info("authentication enabled, %d key(s); unauthenticated: %s",
             len(keys), ", ".join(sorted(PUBLIC_PATHS)))

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        if not _authorised(request.headers.get("authorization"), keys):
            response = error_response(
                401,
                "Incorrect API key provided. Send it as "
                "'Authorization: Bearer <key>'.",
                "invalid_request_error",
                "invalid_api_key")
            # RFC 9110 wants a challenge on a 401. OpenAI's own API omits it;
            # sending it costs nothing and keeps generic HTTP clients honest.
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        return await call_next(request)
