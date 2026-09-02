"""API key authentication: the whole key story, in one place.

    TTS_API_KEYS=key1,key2   ->  Authorization: Bearer key1

**Why this module exists.** `app/auth.py` exists in all three service repos.
Diffed pairwise, the copies differ by 197 lines (stt-stack vs tts-stack), 187
(tts-stack vs tts-long) and 170 (stt-stack vs tts-long). They implement the
same idea and drifted, and the drift *was* the defects. One adversarial review
round found three DIFFERENT bugs, because each copy had drifted separately:

  * a non-ASCII API key could never authenticate (tts-stack only). Starlette
    decodes header bytes as latin-1; the code re-encoded that string as UTF-8
    before comparing, producing different bytes for every non-ASCII key, so
    the CORRECT key was rejected with a message saying it was wrong.
  * `GET /health/` with a trailing slash returned 401 once keys were on
    (tts-stack only). The middleware runs before routing, so FastAPI's 307 to
    `/health` never happens; matched as an exact string the request missed the
    exemption. Any probe written that way goes permanently unhealthy the
    moment keys are set.
  * `TTS_API_KEYS=','` silently disabled authentication entirely (tts-stack
    and tts-long, not stt-stack) and logged that the variable was UNSET — so
    an operator who fat-fingered the value read a warning telling them their
    own configuration was not there, on a service now open to anyone.

Two of those three are still live today in the copies that never got the fix:
stt-stack/app/auth.py:115 and tts-long/app/auth.py:68 both encode UTF-8, and
tts-long/app/auth.py:103 still matches OPEN_PATHS as an exact string. This is
not a predicted benefit of sharing code. It is a measured one.

**Off unless keys are configured, and loudly off.** These services already run
on a LAN with callers that have no key. An upgrade that started refusing them
would turn a feature into an outage, so an unset variable leaves authentication
disabled and says so at WARNING on every start. Open is allowed because someone
chose it; quietly open is not.

**Set but naming no key refuses to start.** `''`, `','`, `'  '`, `',,'` are
reached by ordinary accident — `-e TTS_API_KEYS=$SECRET` with SECRET unset
hands the container an empty value. Treating that as "off" turns an operator's
intent to require keys into a service open to anyone. Unset means off; set
means keys, or nothing starts.

**Installed as ASGI middleware, not as per-route dependencies.** A dependency
has to be remembered on every route added from here on; middleware cannot be
forgotten. That also puts the check ahead of routing, which is why the path
exemption normalises its own trailing slash rather than relying on a redirect
that will not happen. `/docs`, `/redoc` and `/openapi.json` are covered by
default and deliberately: a schema dump is a free map of the service.

The env var name is a parameter (`STT_API_KEYS` / `TTS_API_KEYS`) so that no
operator-visible variable has to be renamed for a service to adopt this. The
name is the only thing that varies; the behaviour must not.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Iterable

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import error_response

__all__ = ["ConfigurationError", "install", "exempt", "load_keys",
           "normalise_path", "public_paths", "authorised",
           "DEFAULT_PUBLIC_PATHS"]

log = logging.getLogger("voice_common.auth")

# The container HEALTHCHECK calls this and has no key, and no way to be given
# one. Requiring one turns a working service into a restart loop. Nothing else
# is exempt by default; voice_common.health adds the real health path through
# exempt() so the route and the exemption can never name different strings.
DEFAULT_PUBLIC_PATHS = ("/health",)

_STATE = "voice_common_public_paths"


class ConfigurationError(SystemExit):
    """A configuration that cannot be served, raised at install time.

    A SystemExit subclass on purpose. install() is called at import, so this
    stops the process — and uvicorn prints the message alone rather than a
    traceback that buries it. It derives from BaseException, so a broad
    `except Exception` somewhere in a startup path cannot swallow a refusal to
    enforce authentication. It is still a named class, so a test and the
    conformance suite can assert on it rather than on a string.
    """


def normalise_path(path: str) -> str:
    """The form a path is compared in: no trailing slash, except for "/".

    This middleware runs before routing, so a request for `/health/` never
    reaches the redirect that would have taken it to `/health`. Matched as an
    exact string it missed the exemption and came back 401, where with
    authentication off it had always been a 307 — see the module docstring.
    """
    return path.rstrip("/") or "/"


def load_keys(env_var: str) -> tuple[str, ...]:
    """The configured keys, or an empty tuple when the variable is unset.

    Each key is stripped, so `k1, k2` works as written. A key is therefore
    never surrounded by whitespace, which is just as well: HTTP strips a field
    value's trailing whitespace, so a key with a space on the end could not be
    presented even if it were configured.

    A value that was SET and yields no key raises ConfigurationError. The
    distinction between that and unset is the whole point, and the reason the
    raw value is inspected rather than the parsed list: both leave the list
    empty, and announcing the second as the first served a typo as though it
    were a decision.
    """
    raw = os.getenv(env_var)
    if raw is None:
        return ()

    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if not keys:
        raise ConfigurationError(
            f"{env_var}={raw!r} is set but names no key: it is empty, or only "
            f"commas and whitespace. Set it to a comma-separated list of "
            f"keys, or unset it entirely to run without authentication.")
    return keys


def public_paths(app: FastAPI) -> set[str]:
    """The mutable exemption set for this app, created on first use.

    A single mutable set shared by install() and exempt(), rather than a value
    copied into the middleware, so install() and install_health() may be
    called in either order. Two independent literals in two modules is exactly
    how tts-long ended up with the exemption at app/auth.py:56 and the route
    at app/main.py:175, free to disagree.
    """
    existing = getattr(app.state, _STATE, None)
    if existing is None:
        existing = set()
        setattr(app.state, _STATE, existing)
    return existing


def exempt(app: FastAPI, path: str) -> None:
    """Exempt one path from authentication, normalised the same way requests are."""
    public_paths(app).add(normalise_path(path))


def authorised(header: str | None, keys: Iterable[str]) -> bool:
    """Does this Authorization header present one of the configured keys?"""
    if not header:
        return False
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented.strip():
        return False
    presented = presented.strip()

    # Back to the bytes the client actually put on the wire. Starlette decodes
    # header bytes as latin-1, which round-trips all 256 of them; re-encoding
    # that decoded string as UTF-8 instead produced different bytes for every
    # non-ASCII key, so the CORRECT key was rejected as "Incorrect API key"
    # and no key containing an accent could ever authenticate. The configured
    # key is encoded as UTF-8 because that is what a client sends.
    try:
        offered = presented.encode("latin-1")
    except UnicodeEncodeError:
        # Not reachable from a starlette header — nothing on the wire could
        # have carried it, so nothing can match it. Reachable from a direct
        # call to this function, which is why it is caught rather than left
        # to become a 500.
        return False

    ok = False
    for key in keys:
        # compare_digest rather than ==, which returns at the first differing
        # byte and so leaks the length of the matching prefix. A LAN is not a
        # trust boundary: anything that can reach the port can time it.
        #
        # And no early break. Stopping at the match would time how far down
        # the list a valid key sits, which is a slower version of the same
        # leak. `ok |=` visits every key on every request, always.
        ok |= hmac.compare_digest(offered, key.encode("utf-8"))
    return ok


class ApiKeyMiddleware:
    """Pure ASGI middleware, so it sees the raw scope and runs before routing.

    Not BaseHTTPMiddleware: this needs nothing from a Request object beyond
    the path and one header, and a plain ASGI callable avoids the extra task
    group BaseHTTPMiddleware wraps every request in.
    """

    def __init__(self, app: ASGIApp, *, keys: tuple[str, ...],
                 exemptions: set[str]) -> None:
        self.app = app
        self.keys = keys
        # The set, not a copy: see public_paths().
        self.exemptions = exemptions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # websocket and lifespan scopes pass through untouched. None of these
        # services has a websocket route today; if one appears, it needs a
        # deliberate decision rather than a 401 shaped like a JSON body.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if normalise_path(scope["path"]) in self.exemptions:
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization")
        if authorised(header, self.keys):
            await self.app(scope, receive, send)
            return

        response = error_response(
            401,
            "Incorrect API key provided. Send it as "
            "'Authorization: Bearer <key>'.",
            type_="invalid_request_error",
            code="invalid_api_key")
        # RFC 9110 wants a challenge on a 401. OpenAI's own API omits it;
        # sending it costs nothing and keeps generic HTTP clients honest.
        response.headers["WWW-Authenticate"] = "Bearer"
        await response(scope, receive, send)


def install(app: FastAPI, env_var: str,
            extra_public_paths: Iterable[str] = DEFAULT_PUBLIC_PATHS) -> tuple[str, ...]:
    """Read the keys, announce the state, and enforce it. Call once, at import.

    Returns the keys, mostly so a caller can log its own line about them.
    Raises ConfigurationError if the variable is set but names no key, which
    stops the process before it can serve a single unauthenticated request.
    """
    keys = load_keys(env_var)
    exemptions = public_paths(app)
    for path in extra_public_paths:
        exemptions.add(normalise_path(path))

    if not keys:
        # Distinct from the ConfigurationError above, and deliberately so: the
        # old code logged "unset" for a value that was set to ',', which told
        # an operator their configuration was missing when in fact it was
        # present and being ignored. Reaching this line means the variable
        # really is absent, because the other case did not get here.
        log.warning(
            "%s is unset: authentication is DISABLED and every request is "
            "accepted, including /v1. Set %s to a comma-separated list of "
            "keys to require Authorization: Bearer.", env_var, env_var)
        return keys

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

    log.info("api key auth enabled, %d key(s); unauthenticated: %s",
             len(keys), ", ".join(sorted(exemptions)))

    app.add_middleware(ApiKeyMiddleware, keys=keys, exemptions=exemptions)
    return keys
