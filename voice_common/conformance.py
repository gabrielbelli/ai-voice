"""A pytest suite the package ships and every consumer's CI runs against its own app.

This is the highest-value thing in voice-common and the one a code library
cannot deliver on its own: the invariants that are not functions.

Sharing `auth.py` stops the three copies of auth.py from drifting. It does
nothing about the parts each service still writes itself — its own routes, its
own error paths, its own health payload — and those are where the same class of
defect reappears. So the package ships the assertions too, and each service's
CI runs them against the app object it actually builds. A bad voice-common bump
then fails at the consumer's build rather than in production, which is the
honest answer to "a shared package has to be validated against three consumers".

Every assertion here is a bug that was really found, or a contract that really
drifted:

  * a non-ASCII configured key authenticates            (tts-stack, fixed)
  * GET /health/ with a trailing slash is not a 401     (tts-stack, fixed)
  * a set-but-keyless variable refuses to start         (tts-stack + tts-long)
  * /docs and /openapi.json need a key when keys are on
  * a bad /v1 body is 400 with a readable error.message (stt-stack answers 422)
  * /health is a coroutine function, not a thread-pool route (tts-stack, stt-stack)

Use it from a consumer by creating one test module:

    # tests/test_conformance.py
    import pytest
    from voice_common.conformance import *          # noqa: F401,F403
    from voice_common.conformance import Service, module_app

    @pytest.fixture
    def voice_service():
        return Service(env_var="TTS_API_KEYS",
                       build=module_app("app.main"),
                       v1_path="/v1/audio/speech")

The star import is deliberate: it puts the test functions in a module inside
the consumer's own tree, so its conftest, its fixtures and its rootdir all
apply normally. `pytest --pyargs` would collect them out of site-packages,
where the consumer's conftest is not visible.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

__all__ = [
    "Service", "module_app", "voice_service",
    "test_a_non_ascii_key_authenticates",
    "test_health_with_a_trailing_slash_is_not_rejected",
    "test_a_set_but_keyless_variable_refuses_to_start",
    "test_docs_and_openapi_require_a_key",
    "test_a_bad_v1_body_is_400_with_a_readable_message",
    "test_health_is_a_coroutine_function",
    "test_a_wrong_key_is_rejected_with_a_challenge",
]

# Non-ASCII on purpose. This exact shape is what tts-stack rejected as
# "Incorrect API key provided" while it was the configured key.
EXOTIC_KEY = "clé-très-secrète"


@dataclass(frozen=True)
class Service:
    """What the suite needs to know about the app under test.

    env_var      the service's key variable, STT_API_KEYS or TTS_API_KEYS
    build        returns a fresh app. Called AFTER the environment is set, so
                 it must re-read it — see module_app.
    v1_path      any POST route on the OpenAI compatibility surface, used to
                 check the error envelope and the 401
    health_path  the unauthenticated probe path
    docs_paths   paths that must NOT be public
    """

    env_var: str
    build: Callable[[], FastAPI]
    v1_path: str = "/v1/audio/speech"
    health_path: str = "/health"
    docs_paths: tuple[str, ...] = ("/docs", "/openapi.json")


def module_app(module: str, attr: str = "app") -> Callable[[], FastAPI]:
    """A build callable that re-imports `module` from scratch each time.

    Keys are read once, at import, in all three services — rotating one is a
    restart, which is also the only moment the startup announcement can be
    trusted to describe reality. That means a plain `import app.main` returns
    the app built under whatever environment the first import saw, and every
    assertion below about a different key list would be tested against the
    wrong app. So the module and its package are dropped from sys.modules
    first.
    """
    root = module.split(".")[0]

    def build() -> FastAPI:
        for name in [n for n in sys.modules
                     if n == root or n.startswith(f"{root}.")]:
            del sys.modules[name]
        return getattr(importlib.import_module(module), attr)

    return build


@pytest.fixture
def voice_service() -> Service:
    """Overridden by the consumer. Defined here only to fail readably."""
    pytest.fail(
        "voice_common.conformance needs a `voice_service` fixture returning a "
        "Service(env_var=..., build=..., v1_path=...). See the module "
        "docstring for the four-line version.")


def _client(service: Service, monkeypatch: pytest.MonkeyPatch,
            keys: str | None) -> TestClient:
    """A client on a freshly built app, with the key variable set as given.

    Deliberately NOT used as a context manager, so the app's lifespan never
    runs. Nothing asserted below needs a loaded model, and tts-stack's startup
    downloads 340 MB while tts-long's allocates 6.5 GB — a conformance suite
    that pulled either into every consumer's CI would be abandoned within a
    week, which is a worse outcome than any of the defects it guards.

    raise_server_exceptions=False so a 500 is asserted on rather than raised
    through the test, which would report the service's traceback instead of
    the contract that failed.
    """
    if keys is None:
        monkeypatch.delenv(service.env_var, raising=False)
    else:
        monkeypatch.setenv(service.env_var, keys)
    return TestClient(service.build(), raise_server_exceptions=False)


def test_a_non_ascii_key_authenticates(voice_service: Service,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """The correct key was rejected as the wrong one for every non-ASCII key.

    Starlette decodes header bytes latin-1; tts-stack re-encoded that string
    as UTF-8 before comparing, so the bytes never matched and no key with an
    accent in it could ever authenticate. Anything but 401 passes here: the
    request may well be rejected for its body, which is a different layer.
    """
    client = _client(voice_service, monkeypatch, EXOTIC_KEY)
    # Sent as bytes: that is what reaches the socket, and httpx refuses to
    # guess an encoding for a non-ASCII header value. Starlette decodes those
    # bytes latin-1, which is the gap the defect lived in.
    response = client.post(
        voice_service.v1_path, json={},
        headers={"Authorization": f"Bearer {EXOTIC_KEY}".encode("utf-8")})
    assert response.status_code != 401, response.text


def test_health_with_a_trailing_slash_is_not_rejected(
        voice_service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health/ returned 401 once keys were on, and stayed unhealthy.

    The check runs before routing, so FastAPI's 307 to /health never happens.
    A probe written with the trailing slash worked until the day keys were
    configured, then failed permanently and the orchestrator restarted a
    healthy service.
    """
    client = _client(voice_service, monkeypatch, "k1")
    response = client.get(voice_service.health_path + "/")
    assert response.status_code != 401, response.text
    # The two spellings must be indistinguishable. Asserted against the
    # unslashed path rather than against a literal 200 so the invariant holds
    # for a service whose health route reports "loading" or fails for its own
    # reasons — the defect being guarded is the trailing slash, nothing else.
    assert response.status_code == client.get(voice_service.health_path).status_code


def test_a_set_but_keyless_variable_refuses_to_start(
        voice_service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    """TTS_API_KEYS=',' silently disabled auth and logged the variable as unset.

    Reached by ordinary accident: `-e TTS_API_KEYS=$SECRET` with SECRET unset.
    The operator asked for authentication, got none, and read a warning saying
    their configuration was not there.
    """
    for degenerate in (",", "  ", ",,", ""):
        monkeypatch.setenv(voice_service.env_var, degenerate)
        # SystemExit, because voice_common.auth.ConfigurationError is one:
        # uvicorn then prints the sentence rather than a traceback, and no
        # broad `except Exception` in a startup path can swallow a refusal to
        # enforce authentication.
        with pytest.raises(SystemExit):
            voice_service.build()


def test_docs_and_openapi_require_a_key(voice_service: Service,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema dump is a free map of the service, so it is not public.

    FastAPI's own /docs and /openapi.json are Starlette routes that no router
    dependency reaches, which is exactly why the check is middleware.
    """
    client = _client(voice_service, monkeypatch, "k1")
    for path in voice_service.docs_paths:
        assert client.get(path).status_code == 401, path


def test_a_bad_v1_body_is_400_with_a_readable_message(
        voice_service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    """stt-stack answers 422 where the other two, and OpenAI, answer 400.

    And the body must carry error.message: openai-python reads that field and
    reports a useless "unknown error" off anything else, so a client is told
    nothing about the field it left out.
    """
    client = _client(voice_service, monkeypatch, None)
    response = client.post(voice_service.v1_path, json={})
    assert response.status_code == 400, response.text
    message = response.json().get("error", {}).get("message")
    assert isinstance(message, str) and message.strip(), response.text


def test_health_is_a_coroutine_function(voice_service: Service,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A sync /health shares AnyIO's 40-thread pool with the blocking routes.

    Forty concurrent synthesis requests took the pool, /health stopped
    answering, and the orchestrator restarted a service that was merely busy.
    """
    monkeypatch.delenv(voice_service.env_var, raising=False)
    app = voice_service.build()
    matches = [route for route in app.routes
               if getattr(route, "path", None) == voice_service.health_path]
    assert matches, f"no route at {voice_service.health_path}"
    for route in matches:
        assert inspect.iscoroutinefunction(route.endpoint), (
            f"{voice_service.health_path} is a sync route and will queue for "
            f"an AnyIO worker thread")


def test_a_wrong_key_is_rejected_with_a_challenge(
        voice_service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    """401, in OpenAI's envelope, with the WWW-Authenticate RFC 9110 asks for."""
    client = _client(voice_service, monkeypatch, "k1")
    response = client.post(voice_service.v1_path, json={},
                           headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401, response.text
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"
