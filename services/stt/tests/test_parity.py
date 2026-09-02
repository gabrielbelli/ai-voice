"""The OpenAI surface, asserted against a recogniser that is not a model.

Every assertion here is a gap that was really found on this service by driving
it with openai-python 3.6.0, and the point of the fake engine is that CI can
run all of them: the real ones are 460 MB and 2.9 GB, and a parity suite that
only runs where a model is already downloaded is a parity suite that stops
running.

The fake is deliberately a *capability profile* rather than a mock of one
engine. What the route does with a field is decided by the flags in asr.py, so
the two profiles here are the two deployments: one that can translate, stream
and take a language hint, and one that can do none of those and reports
per-token logprobs instead. That is Whisper and Parakeet as far as this module
is concerned, and a third engine would only have to declare its flags.
"""

from __future__ import annotations

import io
import json
import struct
import wave

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import asr, glossary, pipeline
from app.main import app


def wav(seconds: float = 2.0, rate: int = 16_000, channels: int = 1) -> bytes:
    """A real WAV, built with the stdlib so the tests need no audio library."""
    frames = int(seconds * rate)
    tone = [int(8000 * np.sin(2 * np.pi * 220 * n / rate)) for n in range(frames)]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(struct.pack("<h", s) for s in tone
                                 for _ in range(channels)))
    return buffer.getvalue()


# Numbered from 1, as faster-whisper numbers them (transcribe.py:1352
# increments before it yields). The wire has to be 0-based anyway — see
# test_segment_ids_start_at_zero.
SEGMENTS = (
    asr.Segment(id=1, seek=0, start=0.0, end=1.0, text=" First piece.",
                tokens=(1, 2), temperature=0.0, avg_logprob=-0.2,
                compression_ratio=1.1, no_speech_prob=0.01,
                words=(asr.Word("First", 0.0, 0.5, -0.1),
                       asr.Word("piece.", 0.5, 1.0, -0.3))),
    asr.Segment(id=2, seek=100, start=1.0, end=2.0, text=" Second piece.",
                tokens=(3, 4), temperature=0.0, avg_logprob=-0.3,
                compression_ratio=1.2, no_speech_prob=0.02,
                words=(asr.Word("Second", 1.0, 1.5, -0.2),
                       asr.Word("piece.", 1.5, 2.0, -0.4))),
)


class FakeWhisper:
    """The profile of an engine that can do everything this service offers."""

    name = "whisper"
    accepts_vocabulary = True
    accepts_language = True
    accepts_temperature = True
    can_translate = True
    can_stream = True
    reports_language = True
    reports_segments = True
    reports_token_logprobs = False
    reports_token_ids = True

    def __init__(self) -> None:
        self.seen: asr.Options | None = None

    def transcribe(self, samples, opts):  # noqa: ANN001, ANN201
        del samples
        self.seen = opts
        return asr.Recognition(
            text="First piece. Second piece.",
            language="en",
            segments=SEGMENTS,
            words=tuple(w for s in SEGMENTS for w in s.words),
        )

    def stream(self, samples, opts):  # noqa: ANN001, ANN201
        del samples
        self.seen = opts
        yield from SEGMENTS


class FakeParakeet:
    """The profile of the default engine: batch, no hints, token logprobs."""

    name = "parakeet"
    accepts_vocabulary = False
    accepts_language = False
    accepts_temperature = False
    can_translate = False
    can_stream = False
    reports_language = False
    reports_segments = False
    reports_token_logprobs = True
    reports_token_ids = False

    def transcribe(self, samples, opts):  # noqa: ANN001, ANN201
        del samples, opts
        return asr.Recognition(
            text="First piece. Second piece.",
            words=(asr.Word("First", 0.0, 0.5, -0.1),
                   asr.Word("piece.", 0.5, 1.0, -0.3),
                   asr.Word("Second", 1.0, 1.5, -0.2),
                   asr.Word("piece.", 1.5, 2.0, -0.4)),
            logprobs=(asr.TokenLogprob(" First", -0.1, (32, 70)),
                      asr.TokenLogprob(" piece.", -0.3, (32, 112))),
        )

    def stream(self, samples, opts):  # noqa: ANN001, ANN201
        raise NotImplementedError(self.name)


@pytest.fixture
def whisper() -> TestClient:
    yield from _serve(FakeWhisper())


@pytest.fixture
def parakeet() -> TestClient:
    yield from _serve(FakeParakeet())


def _serve(engine):  # noqa: ANN001, ANN201
    pipeline.state.clear()
    pipeline.state["asr"] = engine
    pipeline.state["rules"] = glossary.compile_rules({"pece": "piece"})
    # TestClient WITHOUT its context manager, which is what keeps the lifespan
    # from running: `with TestClient(app)` starts it, and the lifespan loads a
    # real 460 MB model and then overwrites the fake installed above. Nothing
    # here needs a model, and a parity suite that downloads one does not run in
    # CI.
    client = TestClient(app)
    client.engine = engine  # type: ignore[attr-defined]
    yield client
    pipeline.state.clear()


def post(client: TestClient, **fields):  # noqa: ANN003, ANN201
    files = {"file": ("clip.wav", wav(), "audio/wav")}
    data = {"model": "whisper-1", **fields}
    return client.post("/v1/audio/transcriptions", files=files, data=data)


# ── the envelope ──────────────────────────────────────────────────────────────

def test_every_error_carries_all_four_keys(whisper: TestClient) -> None:
    """param and code are required-but-nullable, and param was never emitted."""
    response = post(whisper, response_format="xml")
    assert response.status_code == 400
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["param"] == "response_format"


def test_an_unknown_v1_path_is_an_envelope_not_a_detail(whisper: TestClient) -> None:
    response = whisper.post("/v1/nonexistent")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "unknown_url"
    assert set(error) == {"message", "type", "param", "code"}


def test_a_wrong_method_is_an_envelope(whisper: TestClient) -> None:
    response = whisper.get("/v1/audio/transcriptions")
    assert response.status_code == 405
    assert set(response.json()["error"]) == {"message", "type", "param", "code"}


def test_a_native_error_keeps_its_detail_body(whisper: TestClient) -> None:
    """The native contract is not reshaped to tidy up the compatible one."""
    response = whisper.post("/transcribe",
                            files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400
    assert "detail" in response.json()


# ── fields that used to be swallowed ─────────────────────────────────────────

def test_an_unknown_field_is_refused_by_name(whisper: TestClient) -> None:
    response = post(whisper, wibble="1")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "wibble"


def test_model_is_required(whisper: TestClient) -> None:
    response = whisper.post("/v1/audio/transcriptions",
                            files={"file": ("clip.wav", wav(), "audio/wav")})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "model"
    assert error["code"] == "missing_required_parameter"


def test_the_engine_that_ran_is_named_on_every_response(parakeet: TestClient) -> None:
    response = post(parakeet)
    assert response.status_code == 200
    assert response.headers["x-stt-engine"] == "parakeet"


@pytest.mark.parametrize("field,value", [
    ("language", "pt"),
    ("prompt", "Theoria"),
    ("temperature", "0.9"),
])
def test_parakeet_refuses_what_it_cannot_honour(parakeet: TestClient,
                                                field: str, value: str) -> None:
    """Accepted-and-dropped is the defect; a refusal naming the field is the fix."""
    response = post(parakeet, **{field: value})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == field
    assert "parakeet" in error["message"]


def test_whisper_honours_language_and_temperature(whisper: TestClient) -> None:
    assert post(whisper, language="pt", temperature="0.4").status_code == 200
    assert whisper.engine.seen.language == "pt"  # type: ignore[attr-defined]
    assert whisper.engine.seen.temperature == 0.4  # type: ignore[attr-defined]


def test_an_absent_temperature_leaves_the_fallback_ladder_alone(
        whisper: TestClient) -> None:
    assert post(whisper).status_code == 200
    assert whisper.engine.seen.temperature is None  # type: ignore[attr-defined]


def test_prompt_and_keywords_reach_the_decoder(whisper: TestClient) -> None:
    assert post(whisper, prompt="Theoria").status_code == 200
    assert whisper.engine.seen.hotwords == "Theoria"  # type: ignore[attr-defined]


def test_diarisation_is_refused_rather_than_ignored(whisper: TestClient) -> None:
    for field in ("known_speaker_names", "known_speaker_references"):
        response = post(whisper, **{f"{field}[]": "agent"})
        assert response.status_code == 400
        assert response.json()["error"]["param"] == field
    response = post(whisper, response_format="diarized_json")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_value"


def test_chunking_strategy_reaches_the_vad(whisper: TestClient) -> None:
    """The service does run a VAD, and this used to look like it had landed."""
    files = {"file": ("clip.wav", wav(), "audio/wav")}
    response = whisper.post("/v1/audio/transcriptions", files=files, data={
        "model": "whisper-1",
        "chunking_strategy[type]": "server_vad",
        "chunking_strategy[threshold]": "0.7",
        "chunking_strategy[silence_duration_ms]": "500",
    })
    assert response.status_code == 200
    response = whisper.post("/v1/audio/transcriptions", files=files, data={
        "model": "whisper-1",
        "chunking_strategy[type]": "server_vad",
        "chunking_strategy[threshold]": "7",
    })
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "chunking_strategy[threshold]"


def test_logprobs_are_honoured_where_they_exist(parakeet: TestClient) -> None:
    response = post(parakeet, **{"include[]": "logprobs"})
    assert response.status_code == 200
    entries = response.json()["logprobs"]
    assert entries[0]["token"] == " First"
    assert entries[0]["bytes"] == [32, 70]


def test_logprobs_are_refused_where_they_do_not(whisper: TestClient) -> None:
    response = post(whisper, **{"include[]": "logprobs"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "include"


# ── the shapes ────────────────────────────────────────────────────────────────

def test_text_has_no_trailing_newline(whisper: TestClient) -> None:
    response = post(whisper, response_format="text")
    assert response.text == "First piece. Second piece."


def test_verbose_json_carries_segments_by_default(whisper: TestClient) -> None:
    body = post(whisper, response_format="verbose_json").json()
    assert body["task"] == "transcribe"
    assert body["language"] == "english"  # the name, as the specification does
    assert body["usage"] == {"type": "duration", "seconds": 2}
    assert "words" not in body
    required = {"id", "seek", "start", "end", "text", "tokens", "temperature",
                "avg_logprob", "compression_ratio", "no_speech_prob"}
    assert required <= set(body["segments"][0])
    assert len(body["segments"]) == 2


def test_segment_ids_start_at_zero(whisper: TestClient) -> None:
    """Both engines number from 0, whatever the recogniser numbered from.

    faster-whisper's first segment is id 1; the specification's own
    verbose_json example is id 0, and so is the engine whose segments are cut
    from word timings here. One service answering 0-based or 1-based depending
    on a startup env var is a difference a client indexes by.
    """
    body = post(whisper, response_format="verbose_json").json()
    assert [s["id"] for s in body["segments"]] == [0, 1]


def test_parakeet_segment_ids_start_at_zero(parakeet: TestClient) -> None:
    body = post(parakeet, response_format="verbose_json").json()
    assert [s["id"] for s in body["segments"]] == list(
        range(len(body["segments"])))


def test_verbose_json_carries_words_when_asked(whisper: TestClient) -> None:
    """The bracketed spelling is what openai-python actually sends."""
    files = {"file": ("clip.wav", wav(), "audio/wav")}
    response = whisper.post("/v1/audio/transcriptions", files=files, data={
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert [w["word"] for w in body["words"]][:2] == ["First", "piece."]
    assert set(body["words"][0]) == {"word", "start", "end"}


def test_granularities_need_verbose_json(whisper: TestClient) -> None:
    response = post(whisper, **{"timestamp_granularities[]": "word"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "timestamp_granularities"


def test_parakeet_segments_are_synthesised_and_say_so(parakeet: TestClient) -> None:
    body = post(parakeet, response_format="verbose_json").json()
    assert body["language"] == "unknown"  # no LID on this engine; not a guess
    segment = body["segments"][0]
    assert segment["seek"] == 0
    assert segment["no_speech_prob"] == 0.0
    assert segment["tokens"] == []
    assert segment["text"] == "First piece. Second piece."


def test_srt_is_a_cue_per_utterance_terminated_by_a_blank_line(
        whisper: TestClient) -> None:
    body = post(whisper, response_format="srt").text
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,000\nFirst piece.\n\n")
    assert "2\n00:00:01,000 --> 00:00:02,000\nSecond piece.\n\n" in body
    # The blank line after the LAST block is what SubRip requires and what the
    # single-cue body used to leave off.
    assert body.endswith("\n\n")


def test_vtt_is_served_as_vtt(whisper: TestClient) -> None:
    response = post(whisper, response_format="vtt")
    assert response.headers["content-type"] == "text/vtt; charset=utf-8"
    assert response.text.startswith("WEBVTT\n\n")
    assert response.text.endswith("\n\n")


def test_the_native_body_did_not_grow(whisper: TestClient) -> None:
    """Segments and words are for /v1. /transcribe's shape has clients."""
    response = whisper.post("/transcribe",
                            files={"file": ("clip.wav", wav(), "audio/wav")})
    assert response.status_code == 200
    assert set(response.json()) == {
        "text", "raw", "repaired", "model", "audio_seconds", "speech_seconds",
        "compute_seconds", "realtime_factor"}


# ── streaming ────────────────────────────────────────────────────────────────

def test_stream_is_refused_on_a_batch_engine(parakeet: TestClient) -> None:
    response = post(parakeet, stream="true")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "stream"
    assert "parakeet" in error["message"]


def test_stream_framing(whisper: TestClient) -> None:
    """Framing verified against openai-python's SSEDecoder.

    A final event terminated by a single newline is dropped silently by that
    decoder — no error, no warning — so the trailing blank line is the single
    most damaging thing that could be missing here.
    """
    response = post(whisper, stream="true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-cache"

    body = response.text
    assert body.endswith("\n\n")
    frames = [f for f in body.split("\n\n") if f]
    assert all(f.startswith("data: ") for f in frames)
    events = [json.loads(f[len("data: "):]) for f in frames]

    assert [e["type"] for e in events[:-1]] == ["transcript.text.delta"] * 2
    assert events[-1]["type"] == "transcript.text.done"
    # The invariant a client renders against: the deltas ARE the transcript.
    assert "".join(e["delta"] for e in events[:-1]) == events[-1]["text"]
    assert events[0]["delta"] == "First piece."


def test_stream_needs_json(whisper: TestClient) -> None:
    response = post(whisper, stream="true", response_format="verbose_json")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream"


# ── translations ─────────────────────────────────────────────────────────────

def test_translations_exist_on_the_engine_that_has_the_task(
        whisper: TestClient) -> None:
    response = whisper.post("/v1/audio/translations",
                            files={"file": ("clip.wav", wav(), "audio/wav")},
                            data={"model": "whisper-1",
                                  "response_format": "verbose_json"})
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "translate"
    assert body["language"] == "english"
    assert whisper.engine.seen.task == "translate"  # type: ignore[attr-defined]


def test_translations_are_refused_on_the_engine_that_does_not(
        parakeet: TestClient) -> None:
    """Silently transcribing a translate request is the outcome to avoid."""
    response = parakeet.post("/v1/audio/translations",
                             files={"file": ("clip.wav", wav(), "audio/wav")},
                             data={"model": "whisper-1"})
    assert response.status_code == 400
    assert "parakeet" in response.json()["error"]["message"]


def test_translations_refuse_transcription_only_fields(whisper: TestClient) -> None:
    response = whisper.post("/v1/audio/translations",
                            files={"file": ("clip.wav", wav(), "audio/wav")},
                            data={"model": "whisper-1", "stream": "true"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream"


# ── decoding and the timeline ────────────────────────────────────────────────

def test_v1_accepts_a_rate_the_native_route_refuses(whisper: TestClient) -> None:
    """No OpenAI client anticipates a 16 kHz rule; every native client has one."""
    clip = wav(rate=44_100)
    assert post(whisper).status_code == 200
    compatible = whisper.post("/v1/audio/transcriptions",
                              files={"file": ("clip.wav", clip, "audio/wav")},
                              data={"model": "whisper-1"})
    assert compatible.status_code == 200
    native = whisper.post("/transcribe",
                          files={"file": ("clip.wav", clip, "audio/wav")})
    assert native.status_code == 400
    assert "44100 Hz" in native.json()["detail"]


def test_stereo_is_downmixed(whisper: TestClient) -> None:
    response = whisper.post("/v1/audio/transcriptions",
                            files={"file": ("clip.wav", wav(channels=2),
                                            "audio/wav")},
                            data={"model": "whisper-1"})
    assert response.status_code == 200


def test_the_vad_timeline_maps_back_to_the_original_clip() -> None:
    """The mapping subtitles depend on, asserted without a model.

    Two one-second runs with a second of silence between them: a recogniser
    time of 1.5 s is half way through the second run, which is 2.5 s in the
    clip the client sent.
    """
    speech = pipeline.vad.Speech(
        samples=np.zeros(32_000, dtype=np.float32),
        spans=((0, 16_000), (32_000, 48_000)),
        kept=2 / 3)
    assert speech.original(0.0) == 0.0
    assert speech.original(0.5) == 0.5
    assert speech.original(1.5) == 2.5
    # Past the end of the speech reports the end of it, not a time the audio
    # does not reach.
    assert speech.original(99.0) == 3.0
