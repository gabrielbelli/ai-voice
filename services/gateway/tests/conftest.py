"""Mock backends, wired in through httpx's own transport layer.

The gateway is exercised as the real ASGI app, through a real httpx client, so
every test below runs the actual proxy code — header filtering, streaming,
timeout mapping and all. Only the socket is replaced: `Router` dispatches by
hostname to a mock backend, and a mock backend is a plain ASGI callable that
records what reached it.

That is why httpx was worth the dependency. A hand-rolled fake client would
have tested the gateway's idea of httpx rather than httpx.

The app is reloaded per test because its configuration — backend URLs,
timeouts, GATEWAY_API_KEYS — is read at import, exactly as it is in the
siblings and exactly as it is in the container, where the process is the unit
of configuration.
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager

import httpx
import pytest

# The three hostnames the reloaded app is pointed at. They resolve nowhere;
# every request for them is answered by Router below.
STT_URL = "http://stt.test"
TTS_URL = "http://tts.test"
LONG_URL = "http://long.test"


class MockBackend:
    """An ASGI app that records what it was sent and answers what it was told.

    Deliberately raw ASGI rather than a FastAPI app: half the assertions here
    are about bytes and framing — was the upload forwarded in chunks, did the
    Authorization header survive, was content-length preserved — and a
    framework in the way would answer those questions for us.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[dict] = []
        self.reply = self.default_reply

    def default_reply(self, record: dict) -> tuple[int, dict[str, str], bytes]:
        return (200, {"content-type": "application/json"},
                f'{{"backend":"{self.name}","path":"{record["path"]}"}}'.encode())

    async def __call__(self, scope, receive, send) -> None:
        assert scope["type"] == "http"
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            if message.get("body"):
                chunks.append(message["body"])
            if not message.get("more_body", False):
                break

        record = {
            "method": scope["method"],
            "path": scope["path"],
            "query": scope["query_string"].decode(),
            "headers": {k.decode().lower(): v.decode() for k, v in scope["headers"]},
            # The same headers undeduplicated. The dict above cannot show a
            # field name arriving twice, which is exactly what the duplicate-
            # header tests have to assert on.
            "raw_headers": [(k.decode().lower(), v.decode())
                            for k, v in scope["headers"]],
            "body": b"".join(chunks),
            # One ASGI message per chunk the gateway forwarded, which is how
            # the streaming tests tell a pass-through from a buffer.
            "chunks": len(chunks),
        }
        self.seen.append(record)

        status, headers, body = self.reply(record)
        # A list of pairs is accepted as well as a dict, because a dict cannot
        # express the case the proxy has to get right: the same field name
        # twice. Real HTTP allows it (Set-Cookie above all) and a dict silently
        # loses one of them, so a dict-only mock could not have caught the
        # header collapsing that this fixture now tests for.
        pairs = headers.items() if isinstance(headers, dict) else headers
        await send({"type": "http.response.start", "status": status,
                    "headers": [(k.encode(), v.encode()) for k, v in pairs]})
        await send({"type": "http.response.body", "body": body})

    @property
    def last(self) -> dict:
        return self.seen[-1]


class Unreachable(httpx.AsyncBaseTransport):
    """A container that is down, restarting, or has no DNS entry yet."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)


class Slow(httpx.AsyncBaseTransport):
    """A container that accepted the connection and then never answered."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)


class Router(httpx.AsyncBaseTransport):
    """Dispatch by hostname to one of the three mock backends."""

    def __init__(self, mapping: dict[str, object]) -> None:
        self.mapping = mapping

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        target = self.mapping[request.url.host]
        if not isinstance(target, httpx.AsyncBaseTransport):
            target = httpx.ASGITransport(app=target)
        return await target.handle_async_request(request)


def reload_gateway(monkeypatch, *, api_keys: str | None = None):
    """Import a fresh copy of the app under the given environment."""
    monkeypatch.setenv("GATEWAY_STT_URL", STT_URL)
    monkeypatch.setenv("GATEWAY_TTS_URL", TTS_URL)
    monkeypatch.setenv("GATEWAY_TTS_LONG_URL", LONG_URL)
    if api_keys is None:
        monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_API_KEYS", api_keys)
    return importlib.reload(importlib.import_module("app.main"))


@asynccontextmanager
async def gateway(monkeypatch, *, stt=None, tts=None, long=None,
                  api_keys: str | None = None):
    """A client speaking to the real app, which speaks to the mock backends."""
    main = reload_gateway(monkeypatch, api_keys=api_keys)
    router = Router({"stt.test": stt or MockBackend("stt-stack"),
                     "tts.test": tts or MockBackend("tts-stack"),
                     "long.test": long or MockBackend("tts-long")})
    monkeypatch.setattr(main, "new_client",
                        lambda: httpx.AsyncClient(transport=router,
                                                  follow_redirects=False))
    # The lifespan is what opens the client, so it is run rather than skipped:
    # a test against an app whose startup never ran would not be testing this
    # app. ASGITransport does not run it, so it is entered by hand.
    async with main.app.router.lifespan_context(main.app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://gateway.test") as client:
            yield client, main


@pytest.fixture
def backends():
    """The three mock backends, fresh for each test."""
    return (MockBackend("stt-stack"), MockBackend("tts-stack"),
            MockBackend("tts-long"))
