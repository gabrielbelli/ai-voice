"""Whether a streamed answer crosses the gateway as a stream.

tts-stack answers `stream_format: "sse"` with one event per chunk of audio, so
a caller can play the first sentence while the rest is still being made, and
every byte of that crosses this process. A proxy that read the upstream body to
the end before answering would deliver identical bytes, identical headers and
an identical log line: the only symptom is a stack that looks slow, and there
is nothing in a response to see it in.

NOT THROUGH httpx.ASGITransport, which is what the rest of this suite uses.
That transport collects the response body before it returns, so time to first
byte through it is always time to last byte and a proxy that buffers is
indistinguishable from one that does not. The app is driven as raw ASGI here
and the `send` callable is the witness.
"""

from __future__ import annotations

import json

import anyio
import httpx
from conftest import MockBackend, Router, reload_gateway

SPEECH = "/v1/audio/speech"

FRAMES = [b'data: {"type":"speech.audio.delta","audio":"AAAA"}\n\n',
          b'data: {"type":"speech.audio.delta","audio":"BBBB"}\n\n',
          b'data: {"type":"speech.audio.done","usage":{}}\n\n']


class Paced(httpx.AsyncByteStream):
    """A backend answer that is still being made, and a record of what has left.

    `produced` gets one entry per piece as it is yielded, so a test can read
    how much of the answer existed at the moment the gateway sent its own Nth
    message. The bytes are the same either way; only the timing differs, so the
    timing is what has to be asserted.

    A byte stream rather than the MockBackend the rest of this suite uses,
    because Router reaches a mock backend through httpx.ASGITransport and that
    transport collects the whole response body before it returns. A backend
    that cannot stream cannot show whether the gateway streams.
    """

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = pieces
        self.produced: list[bytes] = []

    async def __aiter__(self):
        for piece in self.pieces:
            self.produced.append(piece)
            yield piece


class Streaming(httpx.AsyncBaseTransport):
    """The tts-stack an SSE request actually meets: headers now, body later."""

    def __init__(self, paced: Paced) -> None:
        self.paced = paced

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            "x-accel-buffering": "no"}, stream=self.paced)


async def _drive(app, payload: bytes, watch=None) -> list[bytes]:
    """One POST through the real ASGI app, recording each body message."""
    delivered: list[bytes] = []
    sent_body = False

    async def receive():
        nonlocal sent_body
        if sent_body:
            # A caller that is still there and says nothing. Answering
            # http.request twice makes starlette raise, and answering
            # http.disconnect makes it abandon the response after the first
            # event, which would leave this file asserting that a stream is
            # cut short rather than that it streams.
            await anyio.Event().wait()
        sent_body = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            delivered.append(message["body"])
            if watch is not None:
                watch()

    await app({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": SPEECH,
        "raw_path": SPEECH.encode(), "query_string": b"", "root_path": "",
        "client": ("127.0.0.1", 12345), "server": ("gateway.test", 80),
        "headers": [(b"host", b"gateway.test"),
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode())],
    }, receive, send)
    return delivered



async def _app(monkeypatch, backend):
    """The real gateway, pointed at a tts-stack that answers in pieces."""
    main = reload_gateway(monkeypatch)
    router = Router({"stt.test": MockBackend("stt-stack"),
                     "tts.test": Streaming(backend),
                     "long.test": MockBackend("tts-long")})
    monkeypatch.setattr(main, "new_client",
                        lambda: httpx.AsyncClient(transport=router,
                                                  follow_redirects=False))
    return main


async def test_each_event_is_forwarded_before_the_next_one_is_made(monkeypatch):
    """The claim the whole feature rests on, and the one no response can show.

    Recorded at each body message, `produced` is how many pieces the backend
    had sent at that moment. The first entry is the claim: the first event must
    leave the gateway while one piece exists. Reading the upstream body to the
    end first makes every entry 3, because everything exists before anything is
    sent, and `await upstream.aread()` in place of the aiter_raw loop in _body
    is exactly that failure -- one that breaks nothing else in this suite.

    Only the FIRST entry is pinned to a value. The later ones can read ahead by
    a message, because starlette pumps a streaming body through a memory
    stream. That moves nothing on the wire, and time to first byte is what this
    feature is.
    """
    backend = Paced(FRAMES)
    main = await _app(monkeypatch, backend)
    at_send: list[int] = []

    async with main.app.router.lifespan_context(main.app):
        delivered = await _drive(
            main.app, json.dumps({"model": "kokoro", "input": "One. Two.",
                                  "stream_format": "sse"}).encode(),
            watch=lambda: at_send.append(len(backend.produced)))

    assert at_send[0] == 1, (
        f"the first event left after {at_send[0]} of {len(FRAMES)} existed; "
        "3 of 3 is the whole body read before anything was sent")
    assert at_send == sorted(at_send)
    assert delivered == FRAMES, "and the bytes are untouched as well as prompt"


async def test_the_instruction_not_to_buffer_survives_the_proxy(monkeypatch):
    """X-Accel-Buffering: no is addressed to whatever proxy is in front, and
    both producers set it. Dropped here, the stream is reassembled a hop later
    and nothing in this service can tell.

    Cache-Control is set for the reason tts-long gives: an intermediary that
    cached part of a stream would replay someone else's audio.
    """
    backend = Paced(FRAMES)
    main = await _app(monkeypatch, backend)
    async with main.app.router.lifespan_context(main.app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://gateway.test") as client:
            response = await client.post(
                SPEECH, content=json.dumps({"model": "kokoro", "input": "One.",
                                            "stream_format": "sse"}))
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["content-type"] == "text/event-stream"
    assert "content-length" not in response.headers, \
        "a length this response cannot keep makes a client stop reading early"
