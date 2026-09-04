"""The OpenAI surface: the envelope, the parameters, and the event stream.

Every assertion here is a defect that was really found on a running instance,
and the comment on each says which. Nothing loads Kokoro — a fake synthesiser
stands in for it, for the reason tests/test_conformance.py gives: a suite that
pulled 340 MB of weights into CI would be switched off within a week, which is
worse than any defect it guards. The chunking and the encoders are real, so
what is faked is the model and nothing else.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import shutil
import subprocess
import time

import numpy as np
import pytest
from starlette.testclient import TestClient
from voice_common.conformance import module_app

import app.main as main
from app.audio_out import FORMATS, encode, encode_stream
from app.openai_api import VOICE_ALIASES, custom_voice_id, resolve_voice
from app.synth import MAX_CHUNK_PHONEMES, chunk_phonemes

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg is not on PATH")

# A realistic slice of the 54 Kokoro ships, including every alias target.
VOICES = sorted({*VOICE_ALIASES.values(), "bm_george", "af_nova", "pf_dora"})

# Long enough to plan into several chunks at the default 509-phoneme target,
# which is what makes a stream observably a stream: a one-chunk request sends
# one delta and proves nothing about when it left.
MULTI_CHUNK = "One. Two. Three. Four. Five. Six. Seven. Eight. " * 40


class FakeSynth:
    """The real chunking and the real encoders; a sine wave for the model.

    speak_chunk is deterministic in the phonemes it is given, so the buffered
    body and the concatenated deltas of one request are comparable — which is
    the property this file exists to hold onto.
    """

    voices = VOICES

    def __init__(self) -> None:
        # Every call, in order, so a test can see whether a delta left before
        # the next chunk was synthesised — the one property that separates a
        # real stream from a buffer sliced up afterwards.
        self.calls: list[str] = []
        # Raise on the nth call, to reach the in-band error channel.
        self.fail_on: int | None = None

    def plan(self, text: str, language: str,
             target: int = MAX_CHUNK_PHONEMES) -> list[str]:
        if not text.strip():
            return []
        return chunk_phonemes(text, target=target)

    def token_count(self, chunks: list[str]) -> int:
        return sum(len(chunk) for chunk in chunks)

    def speak(self, text: str, voice: str, language: str, speed: float):
        """The unsegmented path. The fake had only speak_chunk, so a plain
        `text` request 500'd with "no attribute 'speak'" -- invisible until a
        test asked for one."""
        return self.speak_chunk(text, voice, language, speed)

    def speak_segments(self, segments, language: str, speed: float):
        """Mirrors the real one's shape: audio, and where each segment starts.

        The route reads the second value into the X-Segment-Offsets header, so
        a fake that returned only audio would make that header untestable."""
        from app.synth import _offsets

        pieces = []
        for text, pause_after, voice in segments:
            audio = (self.speak_chunk(text, voice, language, speed)
                     if text.strip() else np.zeros(0, dtype=np.float32))
            pad = np.zeros(int(pause_after * 24_000), dtype=np.float32)
            pieces.append(np.concatenate([audio, pad]))
        joined = (np.concatenate(pieces) if pieces
                  else np.zeros(0, dtype=np.float32))
        return joined, _offsets(pieces)

    def speak_chunk(self, phonemes: str, voice: str, language: str,
                    speed: float) -> np.ndarray:
        self.calls.append(phonemes)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("index 510 is out of bounds for axis 0")
        samples = 2400 * len(phonemes)
        t = np.arange(samples, dtype=np.float32) / 24_000.0
        return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def client() -> TestClient:
    app = module_app("app.main")()
    import app.main as main
    synth = FakeSynth()
    main.state["synth"] = synth
    test_client = TestClient(app)
    # Hung off the client so a test can read back what the model was asked for
    # and when, without a second fixture threaded through every signature.
    test_client.synth = synth  # type: ignore[attr-defined]
    return test_client


def speech(client: TestClient, **body: object):
    body.setdefault("model", "tts-1")
    body.setdefault("input", "One. Two. Three.")
    body.setdefault("voice", "fable")
    return client.post("/v1/audio/speech", json=body)


def envelope(response) -> dict[str, object]:
    return response.json()["error"]


# --- the voice table -------------------------------------------------------

def test_all_thirteen_published_voice_names_are_accepted() -> None:
    """Seven of the thirteen came back 400: ash, ballad, coral, sage, verse,
    marin, cedar. With 54 voices loaded that was a table nobody had extended,
    and a client written against the published enum had no way to know which
    half of it this service would take."""
    published = ("alloy ash ballad coral echo fable onyx nova sage shimmer "
                 "verse marin cedar").split()
    assert sorted(VOICE_ALIASES) == sorted(published)
    for name in published:
        assert resolve_voice(name, VOICES, "bm_george") in VOICES


def test_a_native_kokoro_name_still_wins_over_the_table() -> None:
    assert resolve_voice("af_nova", VOICES, "bm_george") == "af_nova"


def test_the_custom_voice_object_is_unwrapped() -> None:
    """`{"id": "voice_1234"}` is the schema's other form and the reference
    client sends it on its minimal call; it used to be `400 voice: Input should
    be a valid string`."""
    assert custom_voice_id({"id": "voice_1234"}) == "voice_1234"
    assert custom_voice_id("fable") == "fable"
    assert custom_voice_id({"name": "fable"}) == {"name": "fable"}


# --- the chunker -----------------------------------------------------------

def test_no_chunk_can_index_the_voice_tensor_out_of_bounds() -> None:
    """400 characters of unpunctuated English phonemise to 518 symbols, reach
    _create_audio as one oversized batch, truncate to exactly 510 tokens and
    index a 510-row voice tensor at row 510: HTTP 500, "index 510 is out of
    bounds for axis 0 with size 510", from prose well inside the 4096
    characters the schema allows."""
    for text in ("word " * 400, "x" * 5000, "a, " * 900, "no punctuation here " * 90):
        chunks = chunk_phonemes(text)
        assert chunks
        assert max(len(chunk) for chunk in chunks) <= MAX_CHUNK_PHONEMES


def test_no_chunk_is_empty() -> None:
    """Upstream's splitter emits a leading '' whenever the first piece is over
    the limit, and an empty batch is a model call with nothing in it."""
    assert all(chunk.strip() for chunk in chunk_phonemes("x" * 2000))


def test_the_default_chunk_size_reproduces_upstreams_own_batching() -> None:
    """The default is the model's context window less the one row that cannot
    be indexed, and the greedy fill is upstream's line for line, so /speak and
    a buffered /v1/audio/speech return the samples they always returned.

    Checked against upstream's own splitter rather than against a list of
    numbers, because the claim is about upstream and not about a fixture:
    identical batches on 400 randomly generated texts, and separately, audio
    bit-identical to what `create()` returned for the same text.
    """
    kokoro = pytest.importorskip("kokoro_onnx")
    words = "the quick brown fox jumps over a lazy dog while nine reviewers argue"
    rng = random.Random(11)
    for _ in range(400):
        parts = []
        for _ in range(rng.randint(3, 400)):
            parts.append(rng.choice(words.split()))
            if rng.random() < 0.12:
                parts[-1] += rng.choice(".,!?;")
        phonemes = " ".join(parts)
        # An instance method that touches no instance state, so no model loads.
        upstream = kokoro.Kokoro._split_phonemes(None, phonemes)
        if any(len(b) > MAX_CHUNK_PHONEMES or not b for b in upstream):
            # The inputs upstream mishandles are the ones this exists to fix.
            assert all(0 < len(b) <= MAX_CHUNK_PHONEMES
                       for b in chunk_phonemes(phonemes))
        else:
            assert chunk_phonemes(phonemes) == upstream


def test_chunking_loses_no_phonemes() -> None:
    """A splitter that drops a symbol drops a sound, and nothing downstream
    would notice."""
    phonemes = "wʌn tuː, θɹˈiː. fɔːɹ faɪv " * 90
    assert "".join(chunk_phonemes(phonemes)).replace(" ", "") == \
        phonemes.replace(" ", "")


# --- the encoders ----------------------------------------------------------

def _pieces() -> list[np.ndarray]:
    t = np.arange(48_000, dtype=np.float32) / 24_000.0
    return [(0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)
            for f in (220.0, 330.0, 440.0)]


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f not in ("wav", "opus")])
@needs_ffmpeg
def test_streamed_bytes_concatenate_to_the_buffered_body(fmt: str) -> None:
    """The property the whole of app/audio_out.py exists for: a client that
    streams and a client that does not must end up with the same file."""
    pieces = _pieces()
    assert b"".join(encode_stream(pieces, fmt)) == encode(pieces, fmt)


@needs_ffmpeg
def test_wav_differs_from_its_stream_in_exactly_the_two_length_fields() -> None:
    """A length cannot be written before the last sample exists. The buffered
    body knows it and writes it; the stream leaves 0xFFFFFFFF."""
    pieces = _pieces()
    streamed = b"".join(encode_stream(pieces, "wav"))
    buffered = encode(pieces, "wav")
    assert len(streamed) == len(buffered)
    assert streamed[44:] == buffered[44:]
    assert streamed[4:8] == b"\xff\xff\xff\xff"
    assert streamed[40:44] == b"\xff\xff\xff\xff"
    assert [i for i in range(44) if streamed[i] != buffered[i]] == [4, 5, 6, 7,
                                                                   40, 41, 42, 43]


@needs_ffmpeg
def test_opus_differs_only_in_the_ogg_serial_and_its_crc() -> None:
    """The Ogg muxer randomises the page serial per stream and -serial_offset
    only offsets that random base — measured, two runs still differ. So two
    identical buffered requests already returned different bytes before any of
    this, and 40 bytes of 8980 is the honest bound rather than zero."""
    pieces = _pieces()
    streamed = b"".join(encode_stream(pieces, "opus"))
    buffered = encode(pieces, "opus")
    assert len(streamed) == len(buffered)
    pages = [i for i in range(len(buffered) - 4) if buffered[i:i + 4] == b"OggS"]
    volatile = {i for page in pages
                for i in (*range(page + 14, page + 18), *range(page + 22, page + 26))}
    differing = {i for i in range(len(buffered)) if streamed[i] != buffered[i]}
    assert differing <= volatile


def test_wav_carries_exactly_the_pcm_body_after_its_header() -> None:
    """wav went through libsndfile and pcm through pcm_bytes, and the two
    rounded differently: 54.6% of samples differed, by up to 2 LSB. Both
    claimed to carry the same samples."""
    pieces = _pieces()
    assert encode(pieces, "wav")[44:] == encode(pieces, "pcm")


# --- the error envelope ----------------------------------------------------

def test_every_error_carries_param(client: TestClient) -> None:
    """`param` is in the schema's REQUIRED list and appeared nowhere in the
    envelope, so every error this service emitted was schema-invalid."""
    for response in (speech(client, voice="nope"),
                     speech(client, response_format="ogg"),
                     speech(client, input=None),
                     client.post("/v1/audio/speech", content=b"{not json",
                                 headers={"content-type": "application/json"}),
                     client.get("/v1/audio/speech"),
                     client.post("/v1/audio/transcriptions", json={})):
        assert response.status_code >= 400
        assert set(envelope(response)) == {"message", "type", "param", "code"}


@pytest.mark.parametrize("body,param", [
    ({"voice": "nope"}, "voice"),
    ({"voice": {"id": "voice_1234"}}, "voice"),
    ({"response_format": "ogg"}, "response_format"),
    ({"stream_format": "nonsense_value"}, "stream_format"),
    ({"speed": 9}, "speed"),
    ({"input": "x" * 4097}, "input"),
])
def test_param_names_the_field_at_fault(client: TestClient, body: dict,
                                        param: str) -> None:
    response = speech(client, **body)
    assert response.status_code == 400
    assert envelope(response)["param"] == param


def test_a_missing_input_is_named(client: TestClient) -> None:
    response = client.post("/v1/audio/speech", json={"model": "tts-1",
                                                     "voice": "fable"})
    assert response.status_code == 400
    assert envelope(response)["param"] == "input"
    assert envelope(response)["code"] == "missing_required_parameter"


def test_404_and_405_under_v1_are_in_the_envelope(client: TestClient) -> None:
    """`GET /v1/audio/speech` returned {"detail":"Method Not Allowed"} and
    /v1/audio/transcriptions returned {"detail":"Not Found"} — FastAPI's shape,
    which openai-python reads no message off."""
    for response, status in ((client.get("/v1/audio/speech"), 405),
                             (client.post("/v1/audio/transcriptions", json={}), 404)):
        assert response.status_code == status
        assert envelope(response)["message"].startswith("Invalid URL (")


def test_the_native_routes_keep_fastapis_shape(client: TestClient) -> None:
    """/v1 is the only compatibility boundary here; something out there
    already parses `detail` on the routes that are not one."""
    response = client.post("/nope", json={})
    assert response.status_code == 404
    assert "detail" in response.json()


# --- nothing is dropped in silence -----------------------------------------

def test_the_speed_clamp_is_announced(client: TestClient) -> None:
    """speed 0.25 and 0.5 returned byte-identical audio, and so did 4, 3 and 2,
    with nothing on the wire to say so. kokoro_onnx asserts 0.5 to 2.0 before it
    will run, so the clamp stays; the silence does not."""
    assert speech(client, speed=4).headers["X-Speed-Clamped"] == "4 to 2"
    assert speech(client, speed=0.25).headers["X-Speed-Clamped"] == "0.25 to 0.5"
    assert "X-Speed-Clamped" not in speech(client, speed=1.5).headers


def ignored(response) -> str | None:
    return response.headers.get("X-Ignored-Parameters")


def test_ignored_parameters_are_named(client: TestClient) -> None:
    """instructions cannot be honoured — Kokoro has no style conditioning — and
    `stream: true` is not a field of this schema at all. Both used to return
    200 with an ordinary mp3 and no signal of any kind."""
    assert ignored(speech(client, model="kokoro",
                          instructions="Shout.")) == "instructions"
    assert ignored(speech(client, model="kokoro", stream=True)) == "stream"
    assert ignored(speech(client, model="kokoro")) is None


def test_the_ignored_list_is_sorted_as_a_whole(client: TestClient) -> None:
    """It used to sort the unknown fields and then append the known ones, so
    the order depended on which kind of field each name was."""
    assert ignored(speech(client, model="kokoro", stream=True, alpha=1,
                          instructions="Shout.")) == "alpha, instructions, stream"


def test_a_model_that_did_not_synthesise_is_named(client: TestClient) -> None:
    """`tts-1` cannot be rejected — every OpenAI client sends a model name and
    refusing them refuses the compatibility this route is for — but a request
    answered by an 82M-parameter model it did not ask for was told nothing."""
    assert speech(client, model="tts-1").status_code == 200
    assert ignored(speech(client, model="tts-1")) == "model"
    assert ignored(speech(client, model="gpt-4o-mini-tts",
                          instructions="Shout.")) == "instructions, model"
    # The name that did synthesise is not a deviation and is not named.
    assert ignored(speech(client, model="kokoro")) is None


def test_the_stream_carries_the_deviation_headers_too(client: TestClient) -> None:
    """A caller that asked for a stream is owed the same admission as one that
    did not: the headers go out before the first delta, not instead of it."""
    response = speech(client, model="tts-1", speed=4, instructions="Shout.",
                      stream_format="sse")
    assert response.status_code == 200
    assert ignored(response) == "instructions, model"
    assert response.headers["X-Speed-Clamped"] == "4 to 2"


# --- the event stream ------------------------------------------------------

def parse(raw: bytes) -> list[dict]:
    assert raw.endswith(b"\n\n"), (
        "every event must end in a blank line, the last one included: fed a "
        "stream that ended in a single newline, openai-python's SSEDecoder "
        "dropped the final event with no error and no warning")
    frames = [frame for frame in raw.split(b"\n\n") if frame]
    for frame in frames:
        assert frame.startswith(b"data: ") and b"\n" not in frame
    return [json.loads(frame[len(b"data: "):]) for frame in frames]


@pytest.mark.parametrize("fmt", FORMATS)
@needs_ffmpeg
def test_the_stream_says_what_the_schema_says(client: TestClient, fmt: str) -> None:
    response = speech(client, response_format=fmt, stream_format="sse")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse(response.content)

    deltas = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert deltas, "a stream with no delta carries no audio"
    for delta in deltas:
        # Two properties, and no index, id, sequence_number or content_type.
        assert set(delta) == {"type", "audio"}
        assert isinstance(delta["audio"], str)

    assert events[-1]["type"] == "speech.audio.done"
    assert set(events[-1]) == {"type", "usage"}
    usage = events[-1]["usage"]
    assert set(usage) == {"input_tokens", "output_tokens", "total_tokens"}
    assert all(isinstance(value, int) for value in usage.values())
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert [e for e in events if e.get("type") == "speech.audio.done"] == [events[-1]]


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f not in ("wav", "opus")])
@needs_ffmpeg
def test_the_deltas_concatenate_to_the_buffered_body(client: TestClient,
                                                    fmt: str) -> None:
    """The one property a client can check for itself, and the reason both
    stream formats come out of one encoder."""
    text = "One. Two. Three. Four. Five."
    streamed = parse(speech(client, input=text, response_format=fmt,
                            stream_format="sse").content)
    joined = b"".join(base64.b64decode(e["audio"]) for e in streamed
                      if e["type"] == "speech.audio.delta")
    assert joined == speech(client, input=text, response_format=fmt).content


def test_stream_format_is_validated_against_its_enum(client: TestClient) -> None:
    """`stream_format: "nonsense_value"` returned 200 with the same mp3 as
    `"audio"`, so a client that misspelled "sse" got exactly the same silence
    as one that spelled it right."""
    assert speech(client, stream_format="nonsense_value").status_code == 400
    assert speech(client, stream_format="audio").status_code == 200


def test_an_empty_input_is_valid_and_returns_a_valid_empty_body(
        client: TestClient) -> None:
    """The schema sets no minLength, so "" is a legal request. It reached numpy
    as an empty concatenate and came back 500, "need at least one array to
    concatenate"."""
    for text in ("", "   "):
        response = speech(client, input=text, response_format="wav")
        assert response.status_code == 200
        # A wav header and no samples, rather than a zero-byte body.
        assert len(response.content) == 44
        events = parse(speech(client, input=text, response_format="wav",
                              stream_format="sse").content)
        assert events[-1]["type"] == "speech.audio.done"
        assert events[-1]["usage"]["output_tokens"] == 0


def test_the_first_delta_leaves_before_the_last_chunk_is_synthesised() -> None:
    """The property that separates a stream from a buffer sliced up afterwards.

    Nothing else in this file would catch that regression: the deltas of a
    faked stream concatenate to the buffered body just as well as a real one's
    do, and every shape assertion passes either way. So this pulls one frame
    out of the generator and asks the synthesiser how much work it had been
    given by the time that frame existed.

    Driven at the generator rather than over HTTP because starlette's
    TestClient runs the whole ASGI app to completion and hands back a
    BytesIO — it buffers, so it cannot show when anything left. The
    HTTP-level proof is a measurement on the running service instead, on the
    schema's own 4096-character maximum: 12 frames, first byte at 4.32 s of a
    47.66 s stream, against 50.20 s before the buffered response sends
    anything at all. 11.6x sooner, first byte at 9% of total.

    `pcm` because it is the one format with no encoder between the model and
    the wire — a codec's lookahead window is a legitimate reason for a first
    delta to arrive one chunk late, and this test is not about the codec.
    """
    synth = FakeSynth()
    chunks = synth.plan(MULTI_CHUNK, "en-us")
    assert len(chunks) > 2, "a one-chunk stream cannot show incrementality"

    stream = main._sse_body(synth, chunks, "bm_fable", "en-gb", 1.0, "pcm", 0)
    first = next(stream)
    assert first.startswith(b'data: {"type":"speech.audio.delta"')
    assert len(synth.calls) == 1, (
        f"the first delta existed only after {len(synth.calls)} of "
        f"{len(chunks)} chunks had been synthesised; a genuinely incremental "
        "stream emits the first piece before it asks the model for the second")

    # …and the rest still arrive, so this is a stream that finishes rather
    # than one frame followed by nothing.
    rest = list(stream)
    assert len(synth.calls) == len(chunks)
    assert json.loads(rest[-1][len(b"data: "):])["type"] == "speech.audio.done"


def test_a_failure_after_the_headers_becomes_an_in_band_error_frame(
        client: TestClient) -> None:
    """The only error channel a stream has left once 200 has gone out.

    openai-python raises `APIError(message=data["error"]["message"])` on any
    frame whose JSON has a top-level `error` key and stops reading, so this is
    what a mid-stream synthesis failure has to look like. Before it existed the
    generator simply raised, the connection dropped mid-body, and the client
    saw a truncated file rather than a reason.

    The failure injected is the real one: `index 510 is out of bounds for axis
    0`, which ordinary unpunctuated prose reached before app/synth.py bounded
    its own chunks.
    """
    client.synth.fail_on = 2  # type: ignore[attr-defined]
    events = parse(speech(client, input=MULTI_CHUNK, response_format="pcm",
                          stream_format="sse").content)

    # The chunks synthesised before the failure still went out as audio.
    assert events[0]["type"] == "speech.audio.delta"
    error = events[-1]["error"]
    assert "index 510 is out of bounds" in error["message"]
    # The same four keys as every buffered error on this route, `param`
    # included — it is required-but-nullable, not optional.
    assert set(error) == {"message", "type", "param", "code"}
    assert error["param"] is None and error["code"] == "synthesis_failed"
    # No done event: the usage it is required to carry would be a lie about an
    # utterance that was never finished.
    assert not [e for e in events if e.get("type") == "speech.audio.done"]


def _ffmpeg_children() -> list[int]:
    """The pids of this process's ffmpeg children. No psutil in the image."""
    found = subprocess.run(["pgrep", "-P", str(os.getpid()), "ffmpeg"],
                           capture_output=True, text=True).stdout.split()
    return [int(pid) for pid in found]


def _gone(pid: int, timeout: float = 5.0) -> bool:
    """True once the pid is neither running nor an unreaped zombie."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid not in _ffmpeg_children():
            return True
        time.sleep(0.05)
    return False


@needs_ffmpeg
def test_closing_a_half_read_stream_kills_the_encoder() -> None:
    """An abandoned encode_stream must not leave an ffmpeg blocked on a pipe.

    `close()` is what a caller that gives up has to reach, because ffmpeg is
    waiting on a stdin nobody will ever close and would otherwise sit there for
    as long as this process lives.

    Guards the kill. It does not guard the reap: a zombie was not observable
    on this platform either way, so asserting one would be asserting something
    that is not true here.
    """
    chunk = np.zeros(24_000, dtype=np.float32)
    stream = encode_stream((chunk for _ in range(200)), "mp3")
    next(stream)  # an encoder is now running

    running = _ffmpeg_children()
    assert running, "expected a running encoder to abandon"
    stream.close()
    assert all(_gone(pid) for pid in running)


def test_the_response_closes_its_generator_when_the_client_hangs_up() -> None:
    """The fix for a leak that had no upper bound on it.

    Starlette stops iterating a generator on disconnect but never closes it, so
    `encode_stream`'s GeneratorExit branch — the one that kills ffmpeg — waited
    on the cyclic collector. Measured before the fix: a stream abandoned after
    its first delta still had a live encoder 200 s later, on a stream that
    would have finished in 29 s. After it, on the same request: gone in 9.3,
    9.6 and 12.4 s over three runs, which is the chunk already in flight
    finishing and nothing more.

    Built through the route rather than by hand, so it also fails if the route
    goes back to a plain StreamingResponse. Driven at the response rather than
    over HTTP because starlette's TestClient buffers the whole body and never
    disconnects.
    """
    import app.main as main_module
    main_module.state["synth"] = FakeSynth()
    request = main_module.SpeechRequest(model="tts-1", input=MULTI_CHUNK,
                                        voice="fable", response_format="pcm",
                                        stream_format="sse")
    response = main_module.openai_speech(request)
    assert isinstance(response, main_module.ClosingStreamingResponse)

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        # What uvicorn does to a send on a socket the client has closed, and
        # the exception starlette turns into ClientDisconnect.
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = {"type": "http", "method": "POST", "path": "/v1/audio/speech",
             "headers": [], "asgi": {"spec_version": "2.4", "version": "3.0"}}
    with pytest.raises(Exception):
        asyncio.run(response(scope, receive, send))

    # A generator with no frame has been closed or has run to the end; this one
    # was suspended at its first yield a moment ago, so it was closed.
    assert response._source.gi_frame is None, (
        "the generator was left suspended, so the encoder it owns lives until "
        "the cyclic collector runs — which on an idle service can be minutes, "
        "or never")


# ---------------------------------------------- following the text ------


def test_segment_offsets_are_exact_not_estimated():
    """Each segment is synthesised and spliced on its own, so its length is
    known at the moment it is made and the running total is a real boundary.

    The alternative -- duration x (characters so far / characters total) -- is
    wrong from the first sentence: the pause after a segment is a fixed number
    of seconds regardless of its length, and speech rate moves with
    punctuation. A highlight built on that drifts away from the audio.
    """
    import numpy as np

    from app.synth import _offsets
    from voice_common.audio import SAMPLE_RATE

    pieces = [np.zeros(2 * SAMPLE_RATE), np.zeros(SAMPLE_RATE // 2),
              np.zeros(3 * SAMPLE_RATE)]
    assert _offsets(pieces) == [0.0, 2.0, 2.5]


def test_a_pause_only_segment_still_advances_the_clock():
    """An empty segment contributes its silence and no speech. The next
    segment starts after that silence, not at the same moment."""
    import numpy as np

    from app.synth import _offsets
    from voice_common.audio import SAMPLE_RATE

    assert _offsets([np.zeros(SAMPLE_RATE), np.zeros(3 * SAMPLE_RATE // 4)]) \
        == [0.0, 1.0]


def test_speak_returns_the_offsets_in_a_header(client):
    """The body is audio, so there is nowhere else to put them without
    inventing a second response shape for a route that has clients."""
    response = client.post("/speak", json={
        "segments": [{"text": "One."}, {"text": "Two."}], "format": "wav"})
    assert response.status_code == 200, response.text
    header = response.headers.get("x-segment-offsets")
    assert header, "no offsets returned"
    offsets = [float(x) for x in header.split(",")]
    assert len(offsets) == 2
    assert offsets[0] == 0.0
    assert offsets[1] > 0, "the second segment cannot start at zero"


def test_a_plain_text_request_gets_no_offsets(client):
    """One segment is one highlight covering everything, which is worse than
    none: it says the page knows where the words are when it does not."""
    response = client.post("/speak", json={"text": "Just a sentence.",
                                           "format": "wav"})
    assert response.status_code == 200, response.text
    assert "x-segment-offsets" not in response.headers
