"""Mock gateway and mock MeTube, wired in through httpx's own transport layer.

NO SERVER IS STARTED BY ANY TEST HERE, and none may be: the app is exercised
as the real ASGI app through fastapi.testclient, and the two things it talks to
are replaced at the httpx transport rather than at a socket. So the forwarding
code, the header filtering, the streaming and the error mapping are the real
ones — only the wire is fake.

That is the same shape services/gateway/tests uses, and it is why httpx was
worth the dependency in both: a hand-rolled fake client would have tested this
service's idea of httpx rather than httpx.

The app is reloaded per test because its configuration is read at import,
exactly as it is in the four siblings and exactly as it is in the container,
where the process is the unit of configuration.
"""

from __future__ import annotations

import contextlib
import importlib
import ipaddress
import json
import socket

import httpx
from urllib.parse import unquote
import pytest
from fastapi.testclient import TestClient

GATEWAY = "http://gateway.test"
METUBE = "http://metube.test"


class FakeGateway:
    """Just enough of services/gateway to exercise this one."""

    def __init__(self) -> None:
        self.keys: tuple[str, ...] = ()
        self.seen: list[httpx.Request] = []
        self.reply: dict[str, tuple[int, dict[str, str], bytes]] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "gateway": "ok",
                                             "backends": {}})
        if self.keys:
            header = request.headers.get("authorization", "")
            if header.removeprefix("Bearer ").strip() not in self.keys:
                return httpx.Response(401, json={"error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error", "param": None,
                    "code": "invalid_api_key"}})
        if path in self.reply:
            status, headers, body = self.reply[path]
            return httpx.Response(status, headers=headers, content=body)
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(200, json={"gateway": path,
                                         "method": request.method})


class FakeMeTube:
    """MeTube's three lists and the four routes this service uses.

    Modelled on the real thing's observed behaviour rather than its intent, so
    the tests can assert the two traps: an auto_start:false item lands in
    `pending` and not `queue`, and /delete answers {"status":"ok"} whether or
    not it deleted anything.
    """

    def __init__(self) -> None:
        self.pending: dict[str, dict] = {}
        self.queue: dict[str, dict] = {}
        self.done: dict[str, dict] = {}
        self.refuse: str | None = None
        # What finish() last wrote, so the static route can serve exactly it.
        self.filename: str = ""
        self.folder: str = ""
        # What the static route hands back, so a test can serve a real WebVTT
        # body rather than the audio placeholder.
        self.content: bytes = b"RIFFfake-audio-bytes"
        # WHICH OF MeTube'S TWO STATIC ROUTES HOLDS THE FILE. Both resolve to
        # one directory on this deployment -- AUDIO_DOWNLOAD_DIR defaults to
        # "%%DOWNLOAD_DIR" and is unset -- so the default is "both", which is
        # the deployed shape. "video" stands in for a deployment that does set
        # them apart, where a captions download is written beside the video and
        # /audio_download/ 404s for it.
        self.served_from: str = "both"
        # An OUTAGE rather than a refusal: MeTube unreachable, which must stay
        # a 502 now that a refusal is a 400. See
        # test_an_unreachable_metube_is_still_a_502.
        self.down: bool = False
        self.calls: list[tuple[str, dict]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if request.method == "GET":
            if path == "/history":
                return httpx.Response(200, json={
                    "pending": list(self.pending.values()),
                    "queue": list(self.queue.values()),
                    "done": list(self.done.values())})
            for route, offered in (("audio_download", ("both", "audio")),
                                   ("download", ("both", "video"))):
                if not path.startswith(f"/{route}/"):
                    continue
                # SERVES ONE EXACT PATH, and 404s on anything else, because the
                # real one does. This used to answer 200 to any path under
                # /audio_download/, which meant the suite could not tell a
                # correct URL from one missing its folder segment -- and that
                # is exactly the bug that shipped: every real download 404'd
                # while 82 tests passed. A fixture more permissive than the
                # thing it stands in for tests nothing.
                want = f"/{route}/" + "/".join(
                    part for part in (self.folder, self.filename) if part)
                if unquote(path) == want and self.served_from in offered:
                    return httpx.Response(200, content=self.content)
                return httpx.Response(404, text=f"not here; the file is at {want}")
            return httpx.Response(404)

        body = json.loads(request.content or b"{}")
        self.calls.append((path, body))
        if path == "/add":
            if self.refuse:
                return httpx.Response(200, json={"status": "error",
                                                 "msg": self.refuse})
            url = body["url"]
            record = {"id": "short-id", "url": url, "title": "A Title",
                      "status": "pending", "size": None, "percent": None,
                      "speed": None, "eta": None, "live_status": "not_live",
                      "filename": None}
            # auto_start defaults TRUE when the field is None, which is why the
            # client under test always sends it explicitly.
            if body.get("auto_start", True):
                self.queue[url] = dict(record, status="downloading", percent=10)
            else:
                self.pending[url] = record
            return httpx.Response(200, json={"status": "ok"})
        if path == "/start":
            for url in body["ids"]:
                record = self.pending.pop(url, None)
                if record:
                    self.queue[url] = dict(record, status="downloading")
            return httpx.Response(200, json={"status": "ok"})
        if path == "/delete":
            where = body["where"]
            target = self.queue if where == "queue" else self.done
            for url in body["ids"]:
                # Both queues are consulted for a "queue" delete because
                # MeTube's cancel() handles pending explicitly.
                target.pop(url, None)
                if where == "queue":
                    self.pending.pop(url, None)
            # ALWAYS ok, even when nothing matched. This is the real
            # behaviour and it is why abandon() verifies afterwards.
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    def finish(self, url: str, filename: str = "A Title.opus",
               folder: str = "stt-ingest") -> None:
        """Finish a download.

        `folder` defaults to the value compose.yaml actually deploys rather
        than to "", because the default should be the deployed shape: MeTube
        writes into that subdirectory and records it, and a fixture that
        defaulted to no folder is why the missing path segment went unnoticed.
        """
        self.queue.pop(url, None)
        self.pending.pop(url, None)
        self.filename = filename
        self.folder = folder
        self.done[url] = {"id": "short-id", "url": url, "title": "A Title",
                          "status": "finished", "filename": filename,
                          "folder": folder, "size": 1234, "percent": 100}


class Bytes(httpx.AsyncByteStream):
    """One chunk, as a stream that has NOT been consumed yet.

    httpx.Response(json=...) arrives with its content already loaded, and
    aiter_raw() on such a response raises StreamConsumed. The service under
    test streams every forwarded response — that is the whole point of it — so
    a mock that hands back a pre-read body tests the wrong thing and fails for
    the wrong reason.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        yield self.data


class Router(httpx.AsyncBaseTransport):
    def __init__(self, gateway: FakeGateway, tube: FakeMeTube) -> None:
        self.gateway, self.tube = gateway, tube

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Read the body here, once, so the fakes below can be plain synchronous
        # functions and can still assert on it. Uploads through this service
        # are STREAMED — that is deliberate, a 131 MB ingest must not be
        # buffered — so `request.content` raises RequestNotRead until it is.
        if request.method in {"POST", "PUT", "PATCH"}:
            await request.aread()
        host = request.url.host
        if host == "gateway.test":
            answer = self.gateway.handle(request)
        elif host == "metube.test":
            answer = self.tube.handle(request)
        else:
            raise httpx.ConnectError(f"nothing is listening on {host}")
        return httpx.Response(answer.status_code, headers=answer.headers,
                              stream=Bytes(answer.content))


def fake_getaddrinfo(host, port, **kwargs):
    """DNS, without DNS.

    The guard resolves before it decides, which is the point of it — so a test
    suite that let it use the real resolver would depend on the network, and
    `media.example` does not resolve anywhere. A literal address answers as
    itself, exactly as getaddrinfo does, so the private-range rules are still
    exercised by app/guard.py's own tests; anything else answers with a public
    address so the ingestion tests get past the guard and on to what they are
    about.
    """
    try:
        ipaddress.ip_address(host)
        address = host
    except ValueError:
        address = "93.184.216.34"
    family = (socket.AF_INET6 if ":" in address else socket.AF_INET)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]


@pytest.fixture
def build(monkeypatch, tmp_path):
    """Reload the app with an environment, and hand back its pieces."""
    def make(**environment):
        environment.setdefault("UI_GATEWAY_URL", GATEWAY)
        environment.setdefault("UI_METUBE_URL", METUBE)
        environment.setdefault("UI_VOICE_DIR", str(tmp_path / "voices"))
        environment.setdefault("UI_PROBE", "0")
        for name, value in environment.items():
            monkeypatch.setenv(name, value)

        from app import config, guard, main, metube, probe, clips, ingest
        for module in (config, guard, probe, metube, clips, ingest, main):
            importlib.reload(module)

        monkeypatch.setattr(guard.socket, "getaddrinfo", fake_getaddrinfo)

        gateway, tube = FakeGateway(), FakeMeTube()
        transport = Router(gateway, tube)
        # The one thing replaced: the factory, not httpx.AsyncClient itself.
        # Patching the class means the replacement's own call to it recurses,
        # which is a stack overflow inside the lifespan and a confusing one.
        monkeypatch.setattr(main, "new_client",
                            lambda: httpx.AsyncClient(transport=transport))
        return main, gateway, tube
    return make


@pytest.fixture
def client(build):
    """A TestClient with the LIFESPAN RUN, which is not the default.

    TestClient only runs startup and shutdown when it is used as a context
    manager, and this app builds its one httpx client in the lifespan. Without
    this, every test sees `'State' object has no attribute 'client'` — which is
    also exactly what a production process would do if the lifespan were
    skipped, so it is worth failing loudly rather than lazily constructing one.
    """
    stack = contextlib.ExitStack()

    def make(**environment):
        main, gateway, tube = build(**environment)
        api = stack.enter_context(TestClient(main.app))
        return api, gateway, tube

    yield make
    stack.close()
