"""Long audio, asserted against a recogniser that is not a model.

THE DEFECT THIS FILE EXISTS FOR. A 13m38s recording came back

    Transcription failed: stt-stack could not be reached: RemoteProtocolError

and the 500 underneath it was

    [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while
    running Add node. Name:'/layers.0/self_attn/Add_2'

Parakeet's Conformer encoder attends over everything it is handed in one pass,
O(n^2) in the length, and pipeline.run handed it the whole VAD-filtered
waveform. Bisected against the deployed service: 6.6 min answered 200 in
71.2 s, 7.7 min answered 500 in 7.6 s. Not a timeout, not a body-size limit --
the failure arrives faster than the successes because the encoder gives up
inside the first layer.

WHAT IS ACTUALLY HARD HERE IS THE OFFSETS, NOT THE SPLIT. A shift bug is
invisible in the first window and wrong in every window after it, so a test
that checks a transcript from the top of a clip proves nothing. Every assertion
below that touches a timestamp reads the END of the audio.

The fake engine reports one word per second of whatever it is handed, timed in
that pass's own compacted timeline, which is what a real recogniser does: it
has no idea a window is a window. So the arithmetic under test is entirely this
service's.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import asr, glossary, pipeline, vad
from app.main import app

RATE = 16_000


def wav(seconds: float) -> bytes:
    """A real WAV of the requested length, built without an audio library.

    numpy rather than test_parity's per-sample loop: the clips here are minutes
    long, and a Python loop over six million samples turns a test suite into a
    coffee break.
    """
    frames = int(seconds * RATE)
    tone = (8000 * np.sin(2 * np.pi * 220 * np.arange(frames) / RATE))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(tone.astype("<i2").tobytes())
    return buffer.getvalue()


def speech_of(spans: tuple[tuple[float, float], ...]) -> vad.Speech:
    """A Speech built from (start, end) pairs in SECONDS of the original clip.

    The samples are numbered so that a mis-sliced window is visible: sample n
    of the concatenation holds n, so a window's first sample says exactly where
    in the compacted timeline it was cut.
    """
    offsets = tuple((int(lo * RATE), int(hi * RATE)) for lo, hi in spans)
    total = sum(hi - lo for lo, hi in offsets)
    return vad.Speech(samples=np.arange(total, dtype=np.float32),
                      spans=offsets, kept=1.0)


class WindowedParakeet:
    """The default engine's capability profile, plus a word clock.

    Records the size of every pass it is given, and reports one word per whole
    second of it timed from zero -- the compacted timeline of THAT PASS, which
    is all a recogniser ever sees. Words are named after the pass they came
    from so a dropped or duplicated window is a text assertion, not a count.
    """

    name = "parakeet"
    accepts_vocabulary = True
    accepts_boost = True
    vocabulary_unavailable = None
    accepts_language = False
    accepts_temperature = False
    can_translate = False
    can_stream = False
    reports_language = False
    reports_segments = False
    reports_token_logprobs = True
    reports_token_ids = False

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.seen: list[asr.Options] = []

    def vocabulary_problems(self, terms):  # noqa: ANN001, ANN201
        del terms
        return ()

    def transcribe(self, samples, opts):  # noqa: ANN001, ANN201
        index = len(self.calls)
        self.calls.append(int(samples.size))
        self.seen.append(opts)
        words = tuple(
            asr.Word(f"w{index}-{n}", float(n), float(n) + 0.5, -0.1)
            for n in range(int(samples.size // RATE))
        )
        return asr.Recognition(
            text=" ".join(word.word for word in words),
            words=words,
            logprobs=tuple(asr.TokenLogprob(f" {w.word}", -0.1, (32,))
                           for w in words),
            boosted=opts.vocabulary if opts.boost else (),
        )

    def stream(self, samples, opts):  # noqa: ANN001, ANN201
        raise NotImplementedError(self.name)


class SlicingVad:
    """A VAD that keeps `keep` seconds out of every `period`, at fixed places.

    Stands in for Silero so the spans under test are known exactly. The real
    one is 2 MB of ONNX and its boundaries move with the audio, which is the
    wrong thing to assert a cut against.
    """

    def __init__(self, keep: float = 4.0, period: float = 5.0) -> None:
        self.keep = keep
        self.period = period

    def speech_only(self, samples, **kwargs):  # noqa: ANN001, ANN003, ANN201
        del kwargs
        spans: list[tuple[int, int]] = []
        step = int(self.period * RATE)
        for lo in range(0, len(samples), step):
            hi = min(lo + int(self.keep * RATE), len(samples))
            if hi > lo:
                spans.append((lo, hi))
        kept = np.concatenate([samples[lo:hi] for lo, hi in spans])
        return vad.Speech(samples=kept, spans=tuple(spans),
                          kept=kept.size / samples.size)


class FakeClock:
    """Stands in for the `time` module inside pipeline, and only there.

    Hands out the readings it was given and then holds the last one, so a call
    the test did not anticipate cannot turn a failed assertion into a
    StopIteration from somewhere else entirely.
    """

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def monotonic(self) -> float:
        return self._readings.pop(0) if len(self._readings) > 1 \
            else self._readings[0]


@pytest.fixture
def engine() -> WindowedParakeet:
    return WindowedParakeet()


@pytest.fixture
def client(engine: WindowedParakeet) -> TestClient:
    """The app with a fake engine and NO VAD, so the spans are the whole clip.

    TestClient without its context manager, for test_parity's reason: entering
    it runs the lifespan, and the lifespan downloads a 460 MB model and
    overwrites the fake.
    """
    pipeline.state.clear()
    pipeline.state["asr"] = engine
    pipeline.state["rules"] = glossary.compile_rules({})
    yield TestClient(app)
    pipeline.state.clear()


def transcribe(client: TestClient, seconds: float, **fields):  # noqa: ANN003, ANN201
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", wav(seconds), "audio/wav")},
        # The bracketed spelling and both granularities, which is what
        # openai-python sends when a caller asks for either.
        data={"model": "whisper-1", "response_format": "verbose_json",
              "timestamp_granularities[]": ["word", "segment"], **fields},
    )


# ── the ceiling ───────────────────────────────────────────────────────────────

def test_the_default_ceiling_is_under_the_length_that_failed() -> None:
    """The shipped default has to be a number the deployed host survived.

    7.7 min of audio returned the ONNXRuntimeError; 6.6 min returned 200. A
    default at or above 462 s would ship the defect this file is named after,
    and one derived by interpolation rather than from a measured success would
    be a guess wearing a measurement's clothes. 300 s is a measured 200.
    """
    assert pipeline.MAX_WINDOW_SECONDS <= 396.0, "6.6 min is the last measured 200"
    assert pipeline.MAX_WINDOW_SECONDS < 462.0, "7.7 min is the measured 500"
    assert pipeline.MAX_WINDOW_SECONDS > 0


def test_speech_past_the_ceiling_is_split_rather_than_sent_whole() -> None:
    speech = speech_of(((0.0, 30.0),))
    windows = pipeline._windows(speech, 10.0)
    assert len(windows) == 3
    assert all(window.samples.size <= 10 * RATE for window in windows)


def test_a_ceiling_of_zero_switches_windowing_off() -> None:
    """The documented escape hatch, and the shape that produced the 500."""
    speech = speech_of(((0.0, 30.0),))
    assert pipeline._windows(speech, 0.0) == (speech,)


# ── one window is not a second code path ──────────────────────────────────────

def test_speech_under_the_ceiling_is_the_same_object_not_a_rebuilt_copy() -> None:
    """Identity, not equality, and the distinction is the point.

    Everything downstream of a short clip behaves as it did before windowing
    existed BY CONSTRUCTION when the object is the same one. Rebuilding an
    equal Speech would make that a claim to be re-checked after every edit to
    _windows.
    """
    speech = speech_of(((0.0, 4.0), (5.0, 9.0)))
    assert pipeline._windows(speech, 300.0)[0] is speech


def test_a_clip_just_under_the_ceiling_still_takes_one_pass(
        client: TestClient, engine: WindowedParakeet,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    assert transcribe(client, 9.5).status_code == 200
    assert engine.calls == [int(9.5 * RATE)], "one pass, over the whole clip"


def test_one_window_leaves_the_response_exactly_as_it_was(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The golden for the single-window path, written out rather than derived.

    Every combining step in run() is meant to be the identity on one element:
    the join over one part, the id renumbering of segments already numbered
    from zero, the logprob concatenation, the boost de-duplication. A golden is
    the only way to notice one of them stopped being an identity, because
    comparing two windowed runs against each other would agree while both were
    wrong.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    body = transcribe(client, 3.0).json()
    assert body["text"] == "w0-0 w0-1 w0-2"
    assert body["words"] == [
        {"word": "w0-0", "start": 0.0, "end": 0.5},
        {"word": "w0-1", "start": 1.0, "end": 1.5},
        {"word": "w0-2", "start": 2.0, "end": 2.5},
    ]
    assert [segment["id"] for segment in body["segments"]] == [0]
    assert body["segments"][0]["seek"] == 0


# ── the offsets, read at the END of the clip ──────────────────────────────────

def test_word_times_in_a_later_window_are_not_the_first_windows(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shift bug, which is invisible anywhere but the end of the audio.

    Thirty seconds, no VAD, a ten-second ceiling: three passes, and the fake
    reports every pass timed from zero. If the window's start is not carried,
    the last word comes back at 9.0 s -- a perfectly plausible number, inside
    the clip, and wrong by twenty seconds.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    words = transcribe(client, 30.0).json()["words"]
    assert len(words) == 30
    assert words[0] == {"word": "w0-0", "start": 0.0, "end": 0.5}
    assert words[10] == {"word": "w1-0", "start": 10.0, "end": 10.5}
    assert words[-1] == {"word": "w2-9", "start": 29.0, "end": 29.5}
    # Monotonic throughout, which is what a subtitle track needs and what a
    # per-window reset breaks at every boundary.
    starts = [word["start"] for word in words]
    assert starts == sorted(starts)


def test_the_shift_is_carried_by_the_spans_and_never_added_twice() -> None:
    """Where the offset lives, asserted on the mapping rather than the route.

    A window's spans are offsets into the ORIGINAL clip, so Speech.original
    walking them from a compacted zero already lands on the client's timeline.
    Adding a window start on top of that is the other way this goes wrong, and
    it doubles rather than omits -- which puts the last word past the end of
    the audio instead of at the start of it.
    """
    speech = speech_of(((0.0, 30.0),))
    first, second, third = pipeline._windows(speech, 10.0)
    assert first.original(0.0) == 0.0
    assert second.original(0.0) == 10.0
    assert third.original(0.0) == 20.0
    # The last moment of the last window is the last moment of the clip, not
    # twice its start and not the length of one window.
    assert third.original(10.0) == 30.0


def test_the_end_of_a_clip_cut_at_pauses_maps_back_to_the_end_of_the_clip(
        client: TestClient, engine: WindowedParakeet,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The real shape: silence removed first, then windowed, then mapped back.

    Four seconds of speech in every five, thirty seconds of clip, an
    eight-second ceiling. The compacted timeline is 24 s long and the client's
    is 30 s, so a window offset applied to the wrong one of the two is off by
    six seconds at the end -- the exact drift the VAD mapping was written to
    remove, reintroduced one layer up.
    """
    pipeline.state["vad"] = SlicingVad()
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 8.0)
    body = transcribe(client, 30.0).json()
    assert engine.calls == [8 * RATE, 8 * RATE, 8 * RATE]
    words = body["words"]
    # The last word is the eighth second of the third pass, which is the
    # second span of that pass, three seconds in: 25.0 + 3.0.
    assert words[-1] == {"word": "w2-7", "start": 28.0, "end": 28.5}
    assert words[-1]["end"] <= body["duration"], "no word outside the audio"


# ── where the cuts land ───────────────────────────────────────────────────────

def test_windows_are_cut_at_the_vads_pauses_rather_than_mid_word() -> None:
    """A cut through a word costs the word in both halves.

    The VAD has already found the real pauses, which is the same argument
    _segments_from_words makes for cutting segments at them. So a window
    boundary is a span boundary whenever one is available, even though that
    leaves a window shorter than the ceiling.
    """
    spans = ((0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0))
    windows = pipeline._windows(speech_of(spans), 10.0)
    original = speech_of(spans).spans
    for window in windows:
        for span in window.spans:
            assert span in original, "a span was split with a pause available"
    assert [len(window.spans) for window in windows] == [2, 2]


def test_a_speech_run_longer_than_the_ceiling_is_cut_by_length() -> None:
    """The fallback, and the only place a word can be split.

    A single run past the ceiling has no pause inside it by definition. Cutting
    it by length loses at most the word on the boundary; not cutting it is the
    500 this file is named after.
    """
    windows = pipeline._windows(speech_of(((0.0, 25.0),)), 10.0)
    assert [window.samples.size for window in windows] == [
        10 * RATE, 10 * RATE, 5 * RATE]
    # Still contiguous in the original clip: the pieces tile the run exactly,
    # with nothing dropped between them.
    assert [span for window in windows for span in window.spans] == [
        (0, 10 * RATE), (10 * RATE, 20 * RATE), (20 * RATE, 25 * RATE)]


def test_no_audio_is_dropped_or_repeated_between_windows() -> None:
    """Concatenating the passes reproduces the speech, in order.

    An off-by-one in the slicing is otherwise a few silent milliseconds nobody
    notices until a word falls in the gap.
    """
    speech = speech_of(((0.0, 4.0), (5.0, 9.0), (10.0, 21.0)))
    windows = pipeline._windows(speech, 7.0)
    rebuilt = np.concatenate([window.samples for window in windows])
    assert np.array_equal(rebuilt, speech.samples)


# ── what the response still has to carry ──────────────────────────────────────

def test_the_transcript_is_every_window_joined_once(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    text = transcribe(client, 25.0).json()["text"]
    assert text.split() == (
        [f"w0-{n}" for n in range(10)]
        + [f"w1-{n}" for n in range(10)]
        + [f"w2-{n}" for n in range(5)]
    )


def test_segment_ids_run_from_zero_across_the_whole_transcript(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ids that restart at each window are three segments called 0.

    A client indexes by this field, and the specification's own example
    numbers a whole transcription rather than a decoder window.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    segments = transcribe(client, 30.0).json()["segments"]
    assert [segment["id"] for segment in segments] == list(range(len(segments)))
    assert len(segments) == 3


def test_logprobs_from_every_window_reach_the_response(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", wav(30.0), "audio/wav")},
        data={"model": "whisper-1", "response_format": "json",
              "include[]": "logprobs"},
    )
    assert response.status_code == 200
    assert len(response.json()["logprobs"]) == 30


def test_the_realtime_factor_describes_the_whole_job_not_one_window(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured over the clip the client sent and every pass it took.

    Reporting the last window's share would make a long transcription look
    identical to a short one, which is the number's only use.

    The clock is faked because the fake engine is instantaneous: three real
    passes over a fake model round to 0.00 s, and a factor of 141259.5x
    asserts nothing. Two readings, three seconds apart, and 30 s of audio has
    to come back as 10.0x -- the whole clip over the whole job.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    # Rebinding pipeline's OWN name, not setattr on the time module: the real
    # time.monotonic is shared with anyio, httpx and the test client's portal,
    # and replacing it there hangs the whole run rather than failing it.
    monkeypatch.setattr(pipeline, "time", FakeClock(100.0, 103.0))
    response = client.post("/transcribe",
                           files={"file": ("clip.wav", wav(30.0), "audio/wav")})
    body = response.json()
    assert body["audio_seconds"] == 30.0
    assert body["speech_seconds"] == 30.0
    assert body["compute_seconds"] == 3.0
    assert body["realtime_factor"] == 10.0


def test_the_glossary_repairs_a_term_in_the_last_window_too(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The repair half of a glossary, asserted where a shift bug would hide it.

    Applied to the joined transcript, so it reaches every window, and to each
    word, so `words[]` and `text` cannot disagree about what was said.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    pipeline.state["rules"] = glossary.compile_rules({"w2-9": "final"})
    body = transcribe(client, 30.0).json()
    assert body["text"].endswith("final")
    assert body["words"][-1]["word"] == "final"
    assert body["words"][-1]["start"] == 29.0


def test_every_window_gets_the_requests_vocabulary(
        client: TestClient, engine: WindowedParakeet,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode-time biasing is per pass, so it has to be handed to every pass.

    A boost list applied to the first window only would bias the opening of a
    recording and leave the rest of it unhelped -- and would say so nowhere,
    because x-boost-applied names the terms, not the passes.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    response = transcribe(client, 30.0, prompt="Parakeet", boost="true")
    assert response.status_code == 200
    assert len(engine.seen) == 3
    assert all("Parakeet" in opts.vocabulary for opts in engine.seen)
    assert all(opts.boost for opts in engine.seen)
    # Named once on the header, not once per window.
    assert response.headers["x-boost-applied"] == "Parakeet"


def test_a_windowed_clip_still_produces_a_usable_subtitle_track(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end reason the offsets matter, in the format that shows it.

    SRT is where a shift bug becomes visible to a person: a cue at 00:00:09
    for speech at 00:00:29 is a subtitle track twenty seconds out of step.
    """
    monkeypatch.setattr(pipeline, "MAX_WINDOW_SECONDS", 10.0)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", wav(30.0), "audio/wav")},
        data={"model": "whisper-1", "response_format": "srt"},
    )
    assert response.status_code == 200
    cues = [line for line in response.text.splitlines() if "-->" in line]
    assert len(cues) == 3
    assert cues[0].startswith("00:00:00,000")
    assert cues[-1].endswith("00:00:29,500")
