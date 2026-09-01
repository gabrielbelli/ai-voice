"""The health contract: an async route, and one string for route and exemption."""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from voice_common import auth
from voice_common.health import install_health


def test_the_route_is_a_coroutine_function_not_a_thread_pool_route() -> None:
    """A sync /health shares AnyIO's 40-thread pool with the blocking routes.

    tts-long paid for this: 40 concurrent synthesis requests each held a
    thread for up to TTS_OPENAI_SYNC_TIMEOUT seconds, /health stopped
    answering, and the orchestrator restarted a service that was merely busy.
    tts-stack and stt-stack still declare `def health()`.
    """
    app = FastAPI()
    install_health(app)
    route, = [r for r in app.routes if getattr(r, "path", None) == "/health"]
    assert inspect.iscoroutinefunction(route.endpoint)


def test_the_route_and_its_exemption_are_the_same_string(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The real job of this module.

    Today the two are independent literals in every repo —
    tts-long/app/auth.py:56 against tts-long/app/main.py:175, and the matching
    pairs in the other two — so renaming one locks the healthcheck out and
    nothing says so until a container restarts in a loop.
    """
    monkeypatch.setenv("TTS_API_KEYS", "k1")
    app = FastAPI()
    auth.install(app, "TTS_API_KEYS", extra_public_paths=())
    install_health(app, path="/healthz")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz/").status_code == 200


def test_health_may_be_installed_before_auth(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Either order works, because the exemption set is shared, not copied."""
    monkeypatch.setenv("TTS_API_KEYS", "k1")
    app = FastAPI()
    install_health(app, path="/healthz")
    auth.install(app, "TTS_API_KEYS", extra_public_paths=())
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/healthz").status_code == 200


def test_the_status_envelope_is_fixed_and_the_details_are_the_callers() -> None:
    app = FastAPI()
    install_health(app, details=lambda: {"threads": 4, "queued": 0})
    body = TestClient(app).get("/health").json()
    assert body == {"status": "ok", "threads": 4, "queued": 0}


def test_a_service_still_loading_can_say_so() -> None:
    """tts-stack and stt-stack both report "loading" before the model is up."""
    app = FastAPI()
    install_health(app, details=lambda: {"status": "loading"})
    assert TestClient(app).get("/health").json() == {"status": "loading"}


def test_an_async_details_callable_is_awaited() -> None:
    async def details() -> dict[str, int]:
        return {"threads": 8}

    app = FastAPI()
    install_health(app, details=details)
    assert TestClient(app).get("/health").json() == {"status": "ok", "threads": 8}


def test_no_details_callable_still_answers() -> None:
    app = FastAPI()
    install_health(app)
    assert TestClient(app).get("/health").json() == {"status": "ok"}
