"""Every parameter OpenAI's schema declares, and what this service does with it.

The rule is one line long: **nothing is accepted and dropped.** Each field is
either honoured or refused by name, in the four-field envelope, and every
assertion below is a behaviour that was measured on the wire and found wrong.
"""

from __future__ import annotations

import pytest


def _error(response):
    body = response.json()["error"]
    # The schema requires all four, with param and code required-but-nullable.
    assert set(body) == {"message", "type", "param", "code"}, body
    return body


def test_every_error_carries_all_four_fields(speech):
    """`param` was missing from every error this service produced."""
    _error(speech.post("/v1/audio/speech", json={}))


def test_a_missing_field_is_named(speech):
    response = speech.post("/v1/audio/speech", json={})
    assert response.status_code == 400
    body = _error(response)
    assert body["param"] == "input"
    assert body["code"] == "missing_required_parameter"


def test_an_unknown_url_under_v1_is_in_the_envelope(speech):
    """`{"detail":"Not Found"}` is a shape openai-python reads no message off."""
    response = speech.post("/v1/audio/transcriptions", json={})
    assert response.status_code == 404
    assert _error(response)["message"] == "Invalid URL (POST /v1/audio/transcriptions)"


def test_the_wrong_method_under_v1_is_in_the_envelope(speech):
    """GET on the speech route answered `{"detail":"Method Not Allowed"}`."""
    response = speech.get("/v1/audio/speech")
    assert response.status_code == 405
    assert _error(response)["code"] == "method_not_allowed"
    # Starlette's Allow header must survive the reshaping.
    assert response.headers["allow"] == "POST"


def test_the_native_routes_keep_their_own_error_shape(speech):
    """/jobs is an older contract and something out there parses `detail`."""
    assert speech.get("/jobs/nope").json() == {"detail": "no such job"}


def test_instructions_is_refused_rather_than_ignored(speech):
    """"speak cheerfully" returned ordinary audio, no error, no warning."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "instructions": "speak cheerfully"})
    assert response.status_code == 400
    assert _error(response)["param"] == "instructions"


def test_an_unknown_voice_is_refused(speech):
    """alloy, nova and custom_voice_xyz all returned the same audio and a 200."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "voice": "custom_voice_xyz"})
    assert response.status_code == 400
    assert _error(response)["param"] == "voice"


def test_the_voice_object_form_is_accepted(speech):
    """The schema's VoiceIdsOrCustomVoice, which the minimal SDK call sends."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "voice": {"id": "nova"},
                                 "response_format": "pcm"})
    assert response.status_code == 200, response.text


def test_a_malformed_voice_object_names_voice_and_not_a_union_branch(speech):
    """`param` has to be a parameter a client can act on.

    pydantic reports a union one error PER BRANCH, tagging each with the
    branch's type name, so reading the first error the way every other field is
    read produced `param: "voice.str"` — a name that appears nowhere in
    OpenAI's schema or in the caller's request. Both branches' complaints are
    kept: a caller who mistyped a key inside the object needs to hear about the
    object form, not only about the string one.
    """
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi",
                                 "voice": {"id": "nova", "typo": 1}})
    assert response.status_code == 400
    error = _error(response)
    assert error["param"] == "voice"
    assert "typo" in error["message"]
    assert "valid string" in error["message"]


def test_a_union_collapse_does_not_swallow_two_ordinary_field_errors():
    """Two errors on two ordinary sub-fields are not a union and stay apart.

    The guard on the collapse above. `segments.0.text` and
    `segments.0.pause_after` share a prefix and diverge at the same depth that
    union branches do; only the branch TAG tells the two cases apart, so this
    asserts the tag test rather than the shape.
    """
    from app.envelope import _union_branches

    union = [{"loc": ("body", "voice", "str"), "msg": "Input should be a "
                                                      "valid string"},
             {"loc": ("body", "voice", "CustomVoice", "typo"),
              "msg": "Extra inputs are not permitted"}]
    assert _union_branches(union) == (
        "voice", ["Input should be a valid string",
                  "typo: Extra inputs are not permitted"])

    ordinary = [{"loc": ("body", "segments", 0, "text"), "msg": "a"},
                {"loc": ("body", "segments", 0, "pause_after"), "msg": "b"}]
    assert _union_branches(ordinary) is None
    assert _union_branches([{"loc": ("body", "input"), "msg": "c"}]) is None


def test_the_voice_actually_used_is_reported(speech):
    """This service has one voice. The response says which one it used."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "voice": "nova",
                                 "response_format": "pcm"})
    assert response.headers["x-voice"] == "default"


def test_an_unknown_field_is_refused_by_name(speech):
    """`{"stream": true}` and `{"totally_unknown_field": 1}` both returned 200.

    `stream` is the TRANSCRIPTION-side switch: a client sending it here was
    asking for a stream and got a buffered file with nothing to read that said
    so.
    """
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "stream": True})
    assert response.status_code == 400
    body = _error(response)
    assert body["param"] == "stream"
    assert body["code"] == "unknown_parameter"


def test_speed_is_refused(speech):
    """Chatterbox has no rate control and resampling would shift the pitch."""
    response = speech.post("/v1/audio/speech", json={"input": "hi", "speed": 1.5})
    assert response.status_code == 400
    assert _error(response)["param"] == "speed"


def test_input_is_capped_at_the_schema_maximum(speech):
    """A 5000-character input was accepted and queued for synthesis."""
    response = speech.post("/v1/audio/speech", json={"input": "x" * 4097})
    assert response.status_code == 400
    assert _error(response)["param"] == "input"
    assert speech.post("/v1/audio/speech",
                       json={"input": "x" * 4096, "response_format": "pcm"}
                       ).status_code in (200, 202)


def test_an_unsupported_language_is_refused_before_the_queue(speech):
    """generate() raises on it, so the caller used to queue to learn about a typo."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "language": "xx"})
    assert response.status_code == 400
    assert _error(response)["param"] == "language"


def test_an_unknown_stream_format_is_refused(speech):
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "stream_format": "chunks"})
    assert response.status_code == 400
    assert _error(response)["param"] == "stream_format"


def test_the_default_format_is_the_one_the_schema_names(speech):
    """It defaulted to wav while an EXPLICIT mp3 was refused with a 400.

    A caller who omitted the field believing it had asked for mp3 was handed
    wav, labelled audio/wav, with no error anywhere.
    """
    from app.encoders import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg is not installed here")
    response = speech.post("/v1/audio/speech", json={"input": "One short line."})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.parametrize("fmt", ["wav", "flac", "pcm"])
def test_the_formats_that_need_no_encoder_always_work(speech, fmt):
    from app.encoders import MEDIA_TYPES

    response = speech.post("/v1/audio/speech",
                           json={"input": "One short line.", "response_format": fmt})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == MEDIA_TYPES[fmt]


def test_an_unavailable_format_is_refused_by_name(speech):
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "response_format": "ogg"})
    assert response.status_code == 400
    assert _error(response)["param"] == "response_format"


def test_no_content_disposition_on_the_openai_route(speech):
    """Not in the schema, and it forces a download instead of playback."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "One short line.", "response_format": "pcm"})
    assert "content-disposition" not in response.headers
    # The native route keeps it: there, a filename is the point.
    job = speech.post("/jobs", json={"text": "One short line."}).json()
    _wait(speech, job["id"])
    assert "content-disposition" in speech.get(f"/jobs/{job['id']}/audio").headers


def test_the_response_is_chunked_rather_than_a_whole_file(speech):
    """The schema declares Transfer-Encoding: chunked on the speech 200."""
    response = speech.post("/v1/audio/speech",
                           json={"input": "One short line.", "response_format": "pcm"})
    assert "content-length" not in response.headers


def test_a_full_queue_is_a_429_with_retry_after(speech, monkeypatch):
    """The only error response the schema declares for this path, and it did not exist."""
    from app import main

    monkeypatch.setattr(main, "MAX_QUEUE", 1)
    monkeypatch.setattr(main, "_full", lambda: True)
    response = speech.post("/v1/audio/speech",
                           json={"input": "hi", "response_format": "pcm"})
    assert response.status_code == 429
    body = _error(response)
    assert body["type"] == "rate_limit_error"
    assert body["code"] == "rate_limit_exceeded"
    assert int(response.headers["retry-after"]) >= 1


def test_long_input_returns_a_job_and_says_so(speech, monkeypatch):
    """The endorsed deviation. It must at least be legible when it happens."""
    from app import main

    monkeypatch.setattr(main, "SYNC_MAX_CHARS", 10)
    response = speech.post("/v1/audio/speech",
                           json={"input": "This is much longer than ten characters.",
                                 "response_format": "pcm"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == f"/jobs/{body['id']}"
    assert int(response.headers["retry-after"]) >= 1
    assert "README" in body["message"]


def test_the_synchronous_boundary_accounts_for_the_queue(speech, monkeypatch):
    """The documented boundary has to be the real one.

    The old code compared a character count against a constant built on the
    wrong realtime factor and ignored the queue entirely, so a request UNDER
    the documented threshold silently became a 202.
    """
    from app import main

    monkeypatch.setattr(main, "SYNC_TIMEOUT", 1.0)
    monkeypatch.setattr(main, "_backlog_seconds", lambda: 1000.0)
    response = speech.post("/v1/audio/speech",
                           json={"input": "Two words.", "response_format": "pcm"})
    assert response.status_code == 202


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def test_long_native_input_is_no_longer_truncated(speech):
    """1690 characters came back as exactly 40.0 seconds, silently.

    generate() stops at 1000 speech tokens. The fix is in the chunker; this is
    the assertion that it reaches the native route too, which is where the
    long jobs actually are.
    """
    text = " ".join(f"Sentence number {n} explains one more step of the setup "
                    f"in a plain and unhurried way." for n in range(20))
    job = speech.post("/jobs", json={"text": text}).json()
    assert job["chunks"] > 1
    finished = _wait(speech, job["id"])
    assert finished["status"] == "done"
    assert finished["audio_seconds"] > 41


def test_a_queued_job_can_be_cancelled(speech):
    """A 202 handed out an id and the worker ground through it regardless."""
    job = speech.post("/jobs", json={"text": "One short line."}).json()
    assert speech.delete(f"/jobs/{job['id']}").json()["status"] in {
        "cancelling", "deleted"}


def test_a_finished_job_can_be_discarded(speech):
    """Nothing was ever removed: the dict and /output grew for the process life."""
    job = speech.post("/jobs", json={"text": "One short line."}).json()
    _wait(speech, job["id"])
    assert speech.delete(f"/jobs/{job['id']}").json()["status"] == "deleted"
    assert speech.get(f"/jobs/{job['id']}").status_code == 404


def test_the_voice_list_is_discoverable(speech):
    """Unknown voices are a 400 now, so what is known has to be readable."""
    body = speech.get("/voices").json()
    assert "default" in body["voices"]
    assert "alloy" in body["openai_aliases"]


def test_a_voice_clip_becomes_a_real_voice(speech, tmp_path, monkeypatch):
    """The path out of the deviation: a file in TTS_VOICE_DIR is a voice."""
    from app import voices

    registry = voices.load_registry(tmp_path, strict=True)
    assert registry.resolve("alloy") is None
    (tmp_path / "alloy.wav").write_bytes(b"not really a wav, only a name")
    registry = voices.load_registry(tmp_path, strict=True)
    name, reference = registry.resolve("alloy")
    assert name == "alloy" and reference.endswith("alloy.wav")
