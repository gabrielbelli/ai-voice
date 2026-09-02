"""The health CONTRACT, not the payload.

Two decisions live here and nothing else does. The payloads themselves stay
per-service — they share only `status` and `threads`, and the rest is a
callable the caller supplies.

**The route is `async def`, always.** tts-long/app/main.py:176 documents why,
having paid for it: a sync route runs on AnyIO's worker pool, forty threads
shared with every other sync route in the app. tts-long's synthesis routes
hold a thread for up to TTS_OPENAI_SYNC_TIMEOUT seconds each, so forty
concurrent callers took the whole pool, `/health` stopped answering, and the
orchestrator restarted a service that was merely busy. tts-stack and stt-stack
still declare `def health()` and still have that exposure. Registering the
route here removes the choice. The consequence is that `details` must not
block — it runs on the event loop.

**The route and its authentication exemption are the same string.** That is
the real job of this module. Today they are independent literals in every
repo — tts-long/app/auth.py:56 says `{"/health"}` and app/main.py:175 says
`@app.get("/health")`, with the matching pairs in the other two — free to
disagree the moment someone renames one. install_health registers the route
AND calls auth.exempt with the same value, so a container healthcheck cannot
be locked out by a rename that looked local.

Deliberately about thirty lines. This is the weakest of the core modules and
it earns its place on the second point alone; if it starts growing a payload
schema, that is the signal it was the wrong boundary.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI

from .auth import exempt

__all__ = ["install_health"]

Details = Callable[[], "dict[str, Any] | Awaitable[dict[str, Any]]"]


def install_health(app: FastAPI, details: Details | None = None,
                   path: str = "/health") -> None:
    """Register the health route and exempt exactly that path from auth.

    `details` returns the per-service body — model_loaded, threads, queued.
    It is merged under a fixed `{"status": ...}` envelope and may override
    `status` itself, which is how a service still loading its model says so.
    It may be a coroutine function, but it must not block either way: see the
    module docstring.
    """
    exempt(app, path)

    @app.get(path)
    async def health() -> dict[str, Any]:  # noqa: D401 - the docstring is above
        payload: dict[str, Any] = {"status": "ok"}
        if details is not None:
            extra = details()
            if inspect.isawaitable(extra):
                extra = await extra
            payload.update(extra)
        return payload
