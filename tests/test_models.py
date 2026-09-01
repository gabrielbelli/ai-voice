"""The two bases, and the reason each config points the way it does."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voice_common.models import OpenAISpeechRequest, Segment


def test_segment_bounds_are_the_ones_both_services_already_publish() -> None:
    """0.0 to 10.0 seconds, declared twice today with nothing checking either.

    A drift in either direction is a silent change to a documented API.
    """
    assert Segment(text="hi").pause_after == 0.0
    assert Segment(text="hi", pause_after=10.0).pause_after == 10.0
    for bad in (-0.1, 10.1):
        with pytest.raises(ValidationError):
            Segment(text="hi", pause_after=bad)


def test_a_segment_rejects_an_unknown_field_rather_than_dropping_it() -> None:
    """tts-stack documented a per-segment `voice` while silently ignoring it.

    The caller got the default voice back and had nothing to read that said
    why. A typo belongs in a 422, not in the audio.
    """
    with pytest.raises(ValidationError):
        Segment(text="hi", pause_afetr=1.0)


def test_the_openai_body_accepts_a_field_it_has_never_heard_of() -> None:
    """OpenAI keeps adding them: `instructions`, `stream_format`.

    A service whose whole purpose is answering OpenAI's clients must not
    reject one for speaking a newer version of the dialect it claims to
    speak. tts-stack reasoned this out; tts-long gets it only by pydantic's
    default, which is luck rather than a decision.
    """
    req = OpenAISpeechRequest(input="hi", instructions="speak slowly",
                              stream_format="sse")
    assert req.input == "hi"
    assert req.model_extra["instructions"] == "speak slowly"


def test_the_two_configs_point_opposite_ways_and_both_are_right() -> None:
    """Segment is a native API: strict. OpenAISpeechRequest is compatibility:
    permissive. Getting either backwards is a real bug in one direction and a
    broken client in the other."""
    assert Segment.model_config["extra"] == "forbid"
    assert OpenAISpeechRequest.model_config["extra"] == "allow"


def test_model_is_accepted_and_ignored() -> None:
    """There is one model per service. Rejecting "tts-1" breaks every client
    that sends it; claiming to honour it would be a lie."""
    assert OpenAISpeechRequest(model="tts-1", input="hi").model == "tts-1"
    assert OpenAISpeechRequest(input="hi").model == "default"


def test_voice_is_optional_unlike_upstream() -> None:
    """A service with a configured default voice has an answer when the field
    is absent, and tts-long has no named voices at all."""
    assert OpenAISpeechRequest(input="hi").voice is None


def test_no_response_format_here_because_the_encoder_set_is_an_image_decision(
        ) -> None:
    """tts-long has no ffmpeg in an image already carrying torch, so mp3 and
    opus are not on offer there. Each service subclasses and declares the
    formats its image can actually produce."""
    assert "response_format" not in OpenAISpeechRequest.model_fields


def test_speed_is_unconstrained_here_because_the_range_is_per_service() -> None:
    """tts-stack clamps OpenAI's 0.25-4.0 into Kokoro's 0.5-2.0 because a
    client cannot be told to send something else; tts-long refuses any speed
    but 1.0 because Chatterbox has no rate control and resampling would shift
    pitch. A shared bound would have to be overridden by both."""
    assert OpenAISpeechRequest(input="hi", speed=9.0).speed == 9.0
