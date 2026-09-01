"""API key authentication, off unless keys are configured.

    TTS_API_KEYS=key1,key2   ->  Authorization: Bearer key1

Unset means every request is accepted, and the service says so loudly at
startup rather than either refusing to boot or running open in silence. That
asymmetry is deliberate: these services already run on a LAN with no keys
anywhere, and an upgrade that starts rejecting every existing caller is a worse
outage than a warning nobody reads.

Set but naming no key is a different thing and refuses to start. See load_keys.

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


def public_path(path: str) -> bool:
    """Is this request path exempt from authentication?

    The trailing slash is stripped first. This middleware runs before routing,
    so a request for /health/ never reaches the redirect that would have taken
    it to /health: matched as an exact string it missed the exemption and came
    back 401, where with authentication off it had always been a 307. A probe
    written that way then goes permanently unhealthy the moment keys are set —
    the exact failure the exemption exists to prevent.
    """
    return (path.rstrip("/") or "/") in PUBLIC_PATHS


def load_keys() -> tuple[str, ...]:
    """The configured keys, or an empty tuple when the variable is unset.

    Each key is stripped, so `k1, k2` works as written. A key is therefore
    never surrounded by whitespace, which is just as well: HTTP strips a field
    value's trailing whitespace, so a key with a space on the end could not be
    presented even if it were configured.
    """
    raw = os.getenv("TTS_API_KEYS")
    if raw is None:
        return ()

    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if not keys:
        # Set but degenerate: '', ',', '  ', ',,  ,'. Reached by ordinary
        # accident — `-e TTS_API_KEYS=$SECRET` with SECRET unset hands the
        # container an empty value — and treating that as "off" turned an
        # operator's intent to require keys into a service open to anyone,
        # under a warning that claimed the variable was unset. Unset means
        # off; set means keys, or nothing starts.
        raise SystemExit(
            "TTS_API_KEYS is set but names no key: it is empty, or only "
            "commas and spaces. Set it to a comma-separated list of keys, or "
            "unset it entirely to run without authentication.")
    return keys


def _authorised(header: str | None, keys: tuple[str, ...]) -> bool:
    if not header:
        return False
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False

    # Back to the bytes the client actually put on the wire. Starlette decodes
    # header bytes as latin-1, which round-trips all 256 of them; re-encoding
    # that decoded string as UTF-8 instead produced different bytes for every
    # non-ASCII key, so the CORRECT key was rejected as "Incorrect API key"
    # and no key containing an accent could ever authenticate. The configured
    # key is encoded as UTF-8 because that is what a client sends.
    try:
        offered = presented.encode("latin-1")
    except UnicodeEncodeError:
        # Not reachable from a starlette header; nothing on the wire could
        # have carried it, so nothing can match it.
        return False

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
        log.warning("TTS_API_KEYS is unset: authentication is DISABLED and "
                    "every request is accepted. Set TTS_API_KEYS to a "
                    "comma-separated list of keys to require one.")
        return

    # Complained about at load rather than at the request that fails, for the
    # same reason the voice table is. A non-ASCII key now authenticates, but
    # only because the comparison is done on wire bytes and the client sends
    # UTF-8; HTTP does not require it to, and not every client obliges.
    exotic = sum(1 for key in keys if not key.isascii())
    if exotic:
        log.warning("%d configured key(s) contain non-ASCII characters; these "
                    "work only with clients that send the header as UTF-8, "
                    "which HTTP does not guarantee. ASCII keys avoid the "
                    "question entirely.", exotic)

    log.info("authentication enabled, %d key(s); unauthenticated: %s",
             len(keys), ", ".join(sorted(PUBLIC_PATHS)))

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if public_path(request.url.path):
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
