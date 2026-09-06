"""`stream_format: "sse"` on the wire.

The framing rules asserted here were verified against openai-python 3.6.0's
SSEDecoder, and the first of them is the most damaging bug this endpoint can
have: a stream whose FINAL event ends in a single newline instead of a blank
line has that event silently dropped — no error, no warning, and a client that
never sees `speech.audio.done`.

The incrementality test is the one that matters most, because it is the claim
the whole feature rests on. Chunking a finished buffer into fake deltas would
pass every other assertion in this file and would be a lie a client builds
timing assumptions on.
"""

from __future__ import annotations

import base64
import json
import os
import time

import pytest


def _events(raw: bytes) -> list[dict]:
    """Parse a stream the way a client must, and assert the framing on the way.

    Deliberately strict: this is the wire format, not a convenience parser.
    """
    text = raw.decode()
    assert text.endswith("\n\n"), "the LAST event must end in a blank line too"
    events = []
    for block in text.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        assert len(lines) == 1, f"one line per event, got {lines!r}"
        line = lines[0]
        if line.startswith(":"):
            continue  # a comment: keepalive
        assert line.startswith("data: "), f"bare data frames only: {line!r}"
        events.append(json.loads(line[len("data: "):]))
    return events


TEXT = ("Open your configuration file. Find the section marked network. "
        "Change the listen address and save it. Restart the service now.")

# Long enough to become several chunks: each sentence is over the merge target,
# so the chunker leaves them alone and the stream has something to be
# incremental about.
LONG = " ".join(
    f"Sentence number {n} explains one more step of the setup in a plain and "
    f"unhurried way, without hurrying over the part that matters." 
    for n in range(4))


def test_the_stream_is_shaped_the_way_the_schema_says(speech):
    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": TEXT, "response_format": "pcm",
                             "stream_format": "sse"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "no-cache" in response.headers["cache-control"]
        raw = b"".join(response.iter_bytes())

    events = _events(raw)
    deltas = [e for e in events if e["type"] == "speech.audio.delta"]
    done = [e for e in events if e["type"] == "speech.audio.done"]

    assert deltas, "a stream with no audio in it is not a stream"
    assert len(done) == 1 and events[-1] is done[0], "done is last, and once"
    for delta in deltas:
        # Two properties, exactly. No index, no id, no sequence_number.
        assert set(delta) == {"type", "audio"}
        base64.b64decode(delta["audio"], validate=True)
    usage = done[0]["usage"]
    assert set(usage) == {"input_tokens", "output_tokens", "total_tokens"}
    assert all(isinstance(v, int) for v in usage.values())
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    # NOT the transcription usage shape: no `type`, no input_token_details.
    assert "type" not in usage


def test_no_done_sentinel_and_no_event_lines(speech):
    """Bare `data:` frames, and nothing invented after the last event.

    Nothing authoritative says OpenAI emits `data: [DONE]` for this endpoint,
    and the one verbatim OpenAI audio SSE transcript in the schema — the
    transcription example — carries no `event:` name lines. Both SDKs tolerate
    either; emitting one would be a guess presented as a fact.
    """
    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": "One short line.", "stream_format": "sse",
                             "response_format": "pcm"}) as response:
        raw = b"".join(response.iter_bytes()).decode()
    assert "[DONE]" not in raw
    assert "event:" not in raw


def test_the_deltas_are_the_buffered_body(speech):
    """The promise: decode every delta, concatenate, and you have the file."""
    body = {"input": TEXT, "response_format": "pcm"}
    with speech.stream("POST", "/v1/audio/speech",
                       json={**body, "stream_format": "sse"}) as response:
        raw = b"".join(response.iter_bytes())
    streamed = b"".join(base64.b64decode(e["audio"]) for e in _events(raw)
                        if e["type"] == "speech.audio.delta")

    buffered = speech.post("/v1/audio/speech", json=body)
    assert buffered.status_code == 200, buffered.text
    assert streamed == buffered.content


@pytest.mark.skipif(os.getenv("TTS_SKIP_TIMING") == "1",
                    reason="timing test disabled")
def test_the_first_delta_leaves_before_generation_finishes(live, monkeypatch):
    """The claim the feature rests on, measured rather than asserted.

    Through a real socket, because TestClient runs the application to
    completion before handing back a response and would report every stream as
    perfectly buffered.

    Each chunk is given a real delay, so a response that generated everything
    and then sliced the buffer would show its first delta at the very end.
    Anything at or above 0.8 of the total is buffering wearing a costume.
    """
    import httpx

    from app import synth as synth_module

    original = synth_module.Synth._speak

    def slow(self, *args, **kwargs):
        time.sleep(0.4)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(synth_module.Synth, "_speak", slow)

    started = time.monotonic()
    first = None
    with httpx.Client(timeout=60) as client:
        with client.stream("POST", f"{live}/v1/audio/speech",
                           json={"input": LONG, "response_format": "pcm",
                                 "stream_format": "sse"}) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if ("speech.audio.delta" in line and line.startswith("data:")
                        and first is None):
                    first = time.monotonic() - started
    total = time.monotonic() - started

    assert first is not None
    assert first < total * 0.8, (
        f"first delta at {first:.2f}s of {total:.2f}s — that is a buffer, "
        f"not a stream")


def test_a_failure_after_the_headers_is_reported_in_band(speech, monkeypatch):
    """Once 200 has gone out, a top-level `error` key is the only channel left.

    openai-python raises APIError(message=data["error"]["message"]) off it and
    stops. Without it the client sees a truncated stream and no reason.
    """
    from app import synth as synth_module

    def boom(self, *args, **kwargs):
        raise RuntimeError("the model fell over")

    monkeypatch.setattr(synth_module.Synth, "_speak", boom)

    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": "One short line.", "stream_format": "sse",
                             "response_format": "pcm"}) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes())
    events = _events(raw)
    assert events[-1]["error"]["message"]
    assert set(events[-1]["error"]) == {"message", "type", "param", "code"}


def test_a_stream_that_waits_sends_keepalive_comments(speech, monkeypatch):
    """A second caller serialises behind the first: one job runs at a time.

    It must not look dead while it waits. `:` lines are comments — ignored by
    openai-python's decoder, and enough to keep a proxy and a read timeout from
    giving up on a stream that is queued rather than broken.
    """
    from app import synth as synth_module

    original = synth_module.Synth._speak

    def slow(self, *args, **kwargs):
        time.sleep(0.6)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(synth_module.Synth, "_speak", slow)

    speech.post("/jobs", json={"text": TEXT})  # occupies the single worker
    comments = 0
    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": "A second stream.", "stream_format": "sse",
                             "response_format": "pcm"}) as response:
        for line in response.iter_lines():
            comments += line.startswith(":")
    assert comments >= 1


def test_a_streamed_job_is_still_collectable_from_jobs(speech):
    """A dropped connection must not throw away minutes of CPU.

    The file on disk is the canonical buffered encoding, whatever the stream
    sent, so /jobs/<id>/audio hands over a complete file.
    """
    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": TEXT, "response_format": "wav",
                             "stream_format": "sse"}) as response:
        job_id = response.headers["x-job-id"]
        b"".join(response.iter_bytes())

    audio = speech.get(f"/jobs/{job_id}/audio")
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"
    # The complete header, with the real length in it — which is exactly what
    # the stream could not send.
    assert int.from_bytes(audio.content[40:44], "little") == len(audio.content) - 44


def test_a_pcm_delta_is_one_segment_and_its_length_is_that_segment(speech):
    """Where each sentence starts, derivable from the stream with no extra
    frame, and only in `pcm`.

    `_RawEncoder.write` returns the piece whole and `close()` returns nothing,
    so one delta is exactly one segment and the bytes that arrived before it
    are exactly where it starts: len / 2 / 24000 seconds, headerless s16le at
    24 kHz mono. Those figures must equal the `offsets` the job records, which
    are accumulated sample counts and not an interpolation.

    A client that follows the text as it plays depends on this and cannot check
    it. It does NOT hold for the other formats -- mp3 lags by up to
    TTS_ENCODER_FLUSH_WAIT and adds a tail delta after the last segment, so
    counting deltas there would drift a sentence at a time -- which is why a
    caller that wants boundaries out of the stream must ask for pcm.
    """
    with speech.stream("POST", "/v1/audio/speech",
                       json={"input": LONG, "response_format": "pcm",
                             "stream_format": "sse"}) as response:
        job_id = response.headers["x-job-id"]
        raw = b"".join(response.iter_bytes())

    deltas = [base64.b64decode(e["audio"]) for e in _events(raw)
              if e["type"] == "speech.audio.delta"]
    starts, running = [], 0
    for delta in deltas:
        starts.append(round(running / 2 / 24_000, 3))
        running += len(delta)

    job = speech.get(f"/jobs/{job_id}").json()
    assert len(deltas) == job["chunks"] > 1, "one delta per segment"
    assert starts == job["offsets"], "the stream and the job disagree on where"
