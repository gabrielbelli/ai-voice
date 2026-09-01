"""The OpenAI surface: the envelope, the parameters, and the event stream.

Every assertion here is a defect that was really found on a running instance,
and the comment on each says which. Nothing loads Kokoro — a fake synthesiser
stands in for it, for the reason tests/test_conformance.py gives: a suite that
pulled 340 MB of weights into CI would be switched off within a week, which is
worse than any defect it guards. The chunking and the encoders are real, so
what is faked is the model and nothing else.
"""

from __future__ import annotations

import base64
import json
import random
import shutil

import numpy as np
import pytest
from starlette.testclient import TestClient
from voice_common.conformance import module_app

from app.audio_out import FORMATS, encode, encode_stream
from app.openai_api import VOICE_ALIASES, custom_voice_id, resolve_voice
from app.synth import MAX_CHUNK_PHONEMES, chunk_phonemes

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg is not on PATH")

# A realistic slice of the 54 Kokoro ships, including every alias target.
VOICES = sorted({*VOICE_ALIASES.values(), "bm_george", "af_nova", "pf_dora"})


class FakeSynth:
    """The real chunking and the real encoders; a sine wave for the model.

    speak_chunk is deterministic in the phonemes it is given, so the buffered
    body and the concatenated deltas of one request are comparable — which is
    the property this file exists to hold onto.
    """

    voices = VOICES

    def plan(self, text: str, language: str,
             target: int = MAX_CHUNK_PHONEMES) -> list[str]:
        if not text.strip():
            return []
        return chunk_phonemes(text, target=target)

    def token_count(self, chunks: list[str]) -> int:
        return sum(len(chunk) for chunk in chunks)

    def speak_chunk(self, phonemes: str, voice: str, language: str,
                    speed: float) -> np.ndarray:
        samples = 2400 * len(phonemes)
        t = np.arange(samples, dtype=np.float32) / 24_000.0
        return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def client() -> TestClient:
    app = module_app("app.main")()
    import app.main as main
    main.state["synth"] = FakeSynth()
    return TestClient(app)


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


def test_ignored_parameters_are_named(client: TestClient) -> None:
    """instructions cannot be honoured — Kokoro has no style conditioning — and
    `stream: true` is not a field of this schema at all. Both used to return
    200 with an ordinary mp3 and no signal of any kind."""
    assert speech(client, instructions="Shout.").headers[
        "X-Ignored-Parameters"] == "instructions"
    assert speech(client, stream=True).headers[
        "X-Ignored-Parameters"] == "stream"
    assert speech(client, stream=True, instructions="Shout.").headers[
        "X-Ignored-Parameters"] == "stream, instructions"
    assert "X-Ignored-Parameters" not in speech(client).headers


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
