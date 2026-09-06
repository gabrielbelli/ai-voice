"""Whether a streamed answer crosses this service as a stream.

The Speak tab asks tts-stack for `stream_format: "sse"` so the first sentence
can be played while the rest is still being made, and every byte of that
crosses this process. A relay that read the upstream body to the end before
answering would deliver exactly the same bytes, the same headers and the same
log line, and the only symptom would be a service that looks slow. There is
nothing in a response to see it in, so it is asserted here.

NOT THROUGH TestClient, and that is why this file exists rather than a few more
tests in test_ui.py. starlette's TestClient writes every response body message
into a BytesIO and hands back the finished bytes, so time to first byte through
it is always time to last byte: a streamed response and a collected one are
indistinguishable, which is how a proxy that quietly buffers passes a suite.
The app is driven as raw ASGI instead, and the `send` callable is the witness.
"""

from __future__ import annotations

import asyncio

import httpx

PATH = "/v1/audio/speech"


class Paced(httpx.AsyncByteStream):
    """A body that is still being made, and a record of what has left so far.

    `produced` gets one entry per piece as it is yielded, so a test can read
    how much of the upstream body existed at the moment the app sent its own
    Nth message. That comparison is the only thing that separates a relay from
    a buffer, because the bytes are identical either way.
    """

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = pieces
        self.produced: list[bytes] = []

    async def __aiter__(self):
        for piece in self.pieces:
            self.produced.append(piece)
            yield piece


def _drive(app, watch=None) -> list[bytes]:
    """One POST through the real ASGI app, recording each body message.

    `watch` is called as each body message is sent, before the next piece of
    the upstream body is asked for.
    """
    delivered: list[bytes] = []

    # The body once, and then a client that is still there and says nothing.
    # BaseHTTPMiddleware keeps reading after the request body is complete, to
    # learn whether the caller went away: answering http.request twice makes it
    # raise, and answering http.disconnect makes it abandon the response after
    # the first event -- which would leave this file asserting that a stream
    # gets cut short rather than that it streams.
    sent_body = False

    async def receive():
        nonlocal sent_body
        if sent_body:
            await asyncio.Event().wait()      # cancelled when the response ends
        sent_body = True
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            delivered.append(message["body"])
            if watch is not None:
                watch()

    async def run():
        # The lifespan is what opens the httpx client this service proxies
        # with, so it is entered rather than skipped.
        async with app.router.lifespan_context(app):
            await app({
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": PATH, "raw_path": PATH.encode(), "query_string": b"",
                "root_path": "", "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
                "headers": [(b"host", b"testserver"),
                            (b"content-type", b"application/json"),
                            (b"content-length", b"2")],
            }, receive, send)

    asyncio.run(run())
    return delivered


FRAMES = [b'data: {"type":"speech.audio.delta","audio":"AAAA"}\n\n',
          b'data: {"type":"speech.audio.delta","audio":"BBBB"}\n\n',
          b'data: {"type":"speech.audio.done","usage":{}}\n\n']


def test_each_event_is_forwarded_before_the_next_one_is_made(build):
    """The claim the whole feature rests on, and the one no response can show.

    Recorded at each body message, `produced` is how many pieces the upstream
    had made at that moment. The first entry is the whole claim: the first
    event must leave while one piece exists. Collecting the body first makes
    every entry 3, because everything exists before anything is sent, and
    `await upstream.aread()` in place of the aiter_raw loop in _relay is
    exactly that failure -- one it breaks nothing else in this suite to make.

    Only the FIRST entry is pinned to a value. The later ones read ahead:
    starlette's BaseHTTPMiddleware pumps the response body into an anyio memory
    stream, so the app can hold a message it has not sent yet. That moves no
    byte earlier or later on the wire, and time to first byte is what this
    feature is.
    """
    main, gateway, _ = build()
    paced = Paced(FRAMES)
    gateway.streams[PATH] = ({"content-type": "text/event-stream"}, paced)

    at_send: list[int] = []
    delivered = _drive(main.app, watch=lambda: at_send.append(len(paced.produced)))

    assert at_send[0] == 1, (
        f"the first event left after {at_send[0]} of {len(FRAMES)} existed; "
        "3 of 3 is the whole body collected before anything was sent")
    assert at_send == sorted(at_send)
    assert delivered == FRAMES, "and the bytes are untouched as well as prompt"


def test_the_frames_are_not_regrouped_on_the_way_through(build):
    """One SSE event per message, not two events in one write or one event
    split across two. Nothing downstream would notice -- a client parses on
    the blank line, not on the message boundary -- but a relay that regroups
    is a relay that is holding bytes, and this is the cheapest way to see it.
    """
    main, gateway, _ = build()
    gateway.streams[PATH] = ({"content-type": "text/event-stream"},
                             Paced(FRAMES))
    assert _drive(main.app) == FRAMES


def test_the_instruction_not_to_buffer_survives_the_relay(client):
    """X-Accel-Buffering: no is addressed to whatever proxy is in front of this
    one, and both producers set it. Dropped here, the stream is reassembled a
    hop later and nothing in this service can tell.

    Cache-Control matters for the same reason it is set: an intermediary that
    cached part of a stream would replay someone else's audio.
    """
    api, gateway, _ = client()
    gateway.streams[PATH] = (
        {"content-type": "text/event-stream", "x-accel-buffering": "no",
         "cache-control": "no-cache, no-store"}, Paced([b"data: x\n\n"]))

    response = api.post(PATH, json={})
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache, no-store"
    assert response.headers["content-type"] == "text/event-stream"


def test_a_stream_is_not_given_a_length_it_does_not_have(client):
    """A Content-Length on a response whose body is still being generated is
    a promise this service cannot keep, and a client that trusts it stops
    reading early. The upstream sends none; none may be invented here."""
    api, gateway, _ = client()
    gateway.streams[PATH] = ({"content-type": "text/event-stream"},
                             Paced(FRAMES))
    response = api.post(PATH, json={})
    assert "content-length" not in response.headers
    assert response.content == b"".join(FRAMES)
