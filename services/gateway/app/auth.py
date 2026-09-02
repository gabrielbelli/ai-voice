"""The one place a bearer token is checked, for the whole stack.

    GATEWAY_API_KEYS=key1,key2   ->  Authorization: Bearer key1

This is the component's main claim on its own existence. stt-stack, tts-stack
and tts-long each carry a copy of this file, and the three copies have already
diverged in observable ways: tts-stack enforces via ASGI middleware with a
PUBLIC_PATHS set, stt-stack via per-route Depends *plus* a re-registration of
/openapi.json, /docs and /redoc behind the key — FastAPI's own registrations
bypassed the router dependency and were handing out a free map of the service
— and tts-long carries a third. Three enforcement points is three places for
the fourth bug.

So the backends run with STT_API_KEYS / TTS_API_KEYS unset, and are reachable
only through this process because only :8080 is published. What that trades
away is written down in the README, honestly: anything already on the
container network can call them unauthenticated.

This file is deliberately the FIXED version of tts-stack/app/auth.py rather
than a fresh one. Two bugs were found and paid for there, and both are
inherited below with the reasoning attached: /health with a trailing slash
missing the exemption, and header bytes decoded latin-1 then re-encoded UTF-8
so that a correct non-ASCII key was rejected as "Incorrect API key". Copying
the fix is cheaper than rediscovering the bug.

Enforcement is middleware over the whole app, not a per-route dependency, so a
route added later is protected by default. /health is the only exemption, for
the same reason all three backends exempt theirs: the TrueNAS healthcheck has
no key and no way to be given one.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, Request

from .openai_api import error_response

log = logging.getLogger("voice-gateway.auth")

PUBLIC_PATHS = frozenset({"/health"})


def public_path(path: str) -> bool:
    """Is this request path exempt from authentication?

    The trailing slash is stripped first. This middleware runs before routing,
    so a request for /health/ never reaches the redirect that would have taken
    it to /health: matched as an exact string it missed the exemption and came
    back 401, where with authentication off it had always been a 307. A probe
    written that way then goes permanently unhealthy the moment keys are set —
    the exact failure the exemption exists to prevent. (Found in tts-stack.)
    """
    return (path.rstrip("/") or "/") in PUBLIC_PATHS


def load_keys() -> tuple[str, ...]:
    """The configured keys, or an empty tuple when the variable is unset.

    Each key is stripped, so `k1, k2` works as written. A key is therefore
    never surrounded by whitespace, which is just as well: HTTP strips a field
    value's trailing whitespace, so a key with a space on the end could not be
    presented even if it were configured.
    """
    raw = os.getenv("GATEWAY_API_KEYS")
    if raw is None:
        return ()

    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if not keys:
        # Set but degenerate: '', ',', '  ', ',,  ,'. Reached by ordinary
        # accident — `-e GATEWAY_API_KEYS=$SECRET` with SECRET unset hands the
        # container an empty value — and treating that as "off" turns an
        # operator's intent to require keys into a service open to anyone,
        # under a warning that claims the variable was unset. Unset means off;
        # set means keys, or nothing starts. It matters more here than in the
        # backends: this process is the only thing checking a token for three
        # services, so the accident opens all of them at once.
        raise SystemExit(
            "GATEWAY_API_KEYS is set but names no key: it is empty, or only "
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
        # Unset means open, loudly. The asymmetry is deliberate and is the
        # backends' own: this stack runs on a LAN with no keys anywhere today,
        # and an upgrade that starts rejecting every existing caller is a
        # worse outage than a warning nobody reads.
        log.warning("GATEWAY_API_KEYS is unset: authentication is DISABLED "
                    "and every request is accepted — for all three backends, "
                    "since this gateway is the only thing checking a token. "
                    "Set GATEWAY_API_KEYS to a comma-separated list of keys.")
        return

    # Complained about at load rather than at the request that fails. A
    # non-ASCII key now authenticates, but only because the comparison is done
    # on wire bytes and the client sends UTF-8; HTTP does not require it to,
    # and not every client obliges.
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
            # Before any backend is contacted: a rejected request costs
            # nothing downstream, and nothing in the log of a backend that
            # never saw it.
            return error_response(
                401,
                "Incorrect API key provided. Send it as "
                "'Authorization: Bearer <key>'.",
                "invalid_request_error",
                "invalid_api_key",
                # RFC 9110 wants a challenge on a 401. OpenAI's own API omits
                # it; sending it costs nothing and keeps generic HTTP clients
                # honest.
                headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)
