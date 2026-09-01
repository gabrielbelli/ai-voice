"""The promise that makes `stream_format: "sse"` honest.

Concatenating every delta must reproduce the buffered body. That is provable
here rather than assertable in prose: the same samples go through the same
encoder both ways, whole and in pieces, and the bytes are compared.

Where it cannot hold, the difference is asserted EXACTLY, so the README's
deviation is a measurement rather than a hedge. wav and flac state the total
length in a header the encoder patches on close, which a stream has already
sent — that difference and no other.
"""

from __future__ import annotations

import numpy as np
import pytest
from voice_common.audio import SAMPLE_RATE

from app.encoders import (MEDIA_TYPES, available_formats, encode, ffmpeg_available,
                          make_encoder)

EXACT = ("pcm", "mp3", "aac")
CONTAINER = ("wav", "flac")


def _audio(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _streamed(audio: np.ndarray, fmt: str, pieces: int = 7) -> bytes:
    encoder = make_encoder(fmt)
    step = audio.size // pieces
    out = [encoder.write(audio[i * step:(i + 1) * step if i < pieces - 1 else audio.size])
           for i in range(pieces)]
    out.append(encoder.close())
    return b"".join(out)


@pytest.mark.parametrize("fmt", EXACT)
def test_the_deltas_are_the_buffered_body(fmt):
    """Byte for byte, for every format that has no length field to patch."""
    if fmt not in available_formats():
        pytest.skip("ffmpeg is not installed here")
    audio = _audio()
    assert _streamed(audio, fmt) == encode(audio, fmt)[0]


def test_wav_differs_only_in_the_two_riff_size_fields():
    """Measured, not asserted loosely: offsets 4 and 40, and nothing else."""
    audio = _audio()
    streamed, buffered = _streamed(audio, "wav"), encode(audio, "wav")[0]
    assert len(streamed) == len(buffered)
    differing = {i for i, (a, b) in enumerate(zip(streamed, buffered)) if a != b}
    assert differing <= set(range(4, 8)) | set(range(40, 44))
    # Everything after the 44-byte canonical header is the same audio.
    assert streamed[44:] == buffered[44:]


def test_flac_differs_only_inside_streaminfo():
    """Total samples and the MD5, which are only known when the last chunk is."""
    audio = _audio()
    streamed, buffered = _streamed(audio, "flac"), encode(audio, "flac")[0]
    assert len(streamed) == len(buffered)
    differing = {i for i, (a, b) in enumerate(zip(streamed, buffered)) if a != b}
    # The STREAMINFO metadata block: 4 bytes of "fLaC", a 4-byte block header,
    # then 34 bytes of stream information.
    assert differing <= set(range(8, 42))
    assert streamed[42:] == buffered[42:]


@pytest.mark.parametrize("fmt", [*EXACT, *CONTAINER, "opus"])
def test_every_format_produces_something_playable(fmt):
    if fmt not in available_formats():
        pytest.skip("ffmpeg is not installed here")
    data, media_type = encode(_audio(1.0), fmt)
    assert media_type == MEDIA_TYPES[fmt]
    assert len(data) > 100


def test_pcm_is_raw_little_endian_at_the_declared_rate():
    """OpenAI's pcm is headerless 24 kHz 16-bit mono. Two bytes per sample."""
    audio = _audio(1.0)
    data, media_type = encode(audio, "pcm")
    assert media_type == "audio/pcm"
    assert len(data) == audio.size * 2


def test_a_missing_ffmpeg_removes_formats_rather_than_faking_them():
    """The list is probed, so a checkout without ffmpeg answers 400, not 500."""
    formats = available_formats()
    if ffmpeg_available():
        assert set(formats) == set(MEDIA_TYPES)
    else:
        assert set(formats) == {"wav", "flac", "pcm"}
