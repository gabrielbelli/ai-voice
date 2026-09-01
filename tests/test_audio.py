"""The three wire-format helpers, and the failure each one closes."""

from __future__ import annotations

import numpy as np
import pytest

from voice_common.audio import SAMPLE_RATE, check_rate, pcm_bytes, splice


def test_an_overshoot_is_clipped_not_wrapped_it_would_have_been_a_click() -> None:
    """A sample above 1.0 scaled straight into int16 wraps to the opposite
    extreme: +1.5 becomes a large NEGATIVE number, so one overshoot is an
    audible click rather than an inaudibly loud sample."""
    wrapped = np.frombuffer(pcm_bytes(np.array([1.5], dtype=np.float32)),
                            dtype="<i2")
    assert wrapped[0] == 32767


def test_an_undershoot_is_clipped_too() -> None:
    clipped = np.frombuffer(pcm_bytes(np.array([-1.5], dtype=np.float32)),
                            dtype="<i2")
    assert clipped[0] == -32767


def test_the_pcm_bytes_are_16_bit_little_endian_as_openai_specifies() -> None:
    """Headerless 24 kHz 16-bit LE mono. A byte order mistake here is not an
    error anywhere, it is noise in the caller's speakers."""
    data = pcm_bytes(np.array([0.0, 1.0], dtype=np.float32))
    assert len(data) == 4
    assert data == b"\x00\x00\xff\x7f"


def test_silence_round_trips_to_zero() -> None:
    assert pcm_bytes(np.zeros(3, dtype=np.float32)) == b"\x00\x00" * 3


def test_a_changed_model_sample_rate_is_refused_rather_than_mislabelled() -> None:
    """tts-long has no such guard: it asserts 24 kHz in a comment and writes
    whatever Chatterbox returns into a wav header that says 24000. A model
    update that changed the rate would ship every file at the wrong pitch and
    nothing would report an error — the files play, they are simply wrong."""
    with pytest.raises(RuntimeError, match="unexpected sample rate 16000"):
        check_rate(16_000)
    check_rate(SAMPLE_RATE)


def test_splice_inserts_real_silence_of_the_length_asked_for() -> None:
    """Pauses are generated here rather than asked of the model: no TTS model
    reliably produces a beat you can act inside."""
    one = np.ones(10, dtype=np.float32)
    out = splice([(one, 0.5), (one, 0.0)])
    assert out.size == 10 + int(SAMPLE_RATE * 0.5) + 10
    assert out[10] == 0.0


def test_splice_of_nothing_returns_an_empty_array_not_a_valueerror() -> None:
    """np.concatenate([]) raises. Both existing implementations return
    zeros(0), because a request of nothing but pauses is odd and not an
    error."""
    assert splice([]).size == 0
    assert splice([(None, 0.0)]).size == 0


def test_a_segment_with_no_audio_still_contributes_its_pause() -> None:
    """An empty text with a pause after it is how a caller asks for a beat."""
    assert splice([(None, 1.0)]).size == SAMPLE_RATE


def test_splice_keeps_float32_so_nothing_upcasts_on_the_way_to_the_encoder(
        ) -> None:
    assert splice([(np.ones(4, dtype=np.float32), 0.1)]).dtype == np.float32
