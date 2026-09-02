"""Wire-format helpers only. An optional extra: voice-common[audio].

Three small things, each of which is either already duplicated byte for byte
or missing from a service that needs it.

An extra rather than a core dependency for two reasons. stt-stack needs none
of it and should not take numpy just to import the auth module. And numpy is
the one pin that genuinely conflicts across the estate: tts-stack pins
2.3.4, and tts-long deliberately leaves it unpinned because chatterbox-tts
requires numpy<2 on Python 3.12, so a hard dependency here would make one of
the two resolves impossible.

Everything above the wire format stays out. tts-stack's ffmpeg table and its
bitrate reasoning — 32k opus, 64k mp3 and aac, chosen for one voice at 24 kHz
— is an image decision, not a wire decision: tts-long has no ffmpeg on purpose,
and stt-stack has none so that a 44.1 kHz caller is told rather than quietly
resampled.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SAMPLE_RATE", "pcm_bytes", "check_rate", "splice"]

# OpenAI's `pcm` is headerless 24 kHz 16-bit little-endian mono. It is also
# the native rate of both Kokoro and Chatterbox, so nothing is resampled on
# the way out of either service.
SAMPLE_RATE = 24_000


def pcm_bytes(audio: np.ndarray) -> bytes:
    """Float samples to OpenAI's headerless PCM.

    The expression is currently byte-identical in tts-stack/app/main.py:217
    and tts-long/app/main.py:108.

    Clipped BEFORE scaling, and that ordering is the whole content of this
    function. A sample above 1.0 scaled straight into int16 wraps to the
    opposite extreme, so a single overshoot becomes a click rather than a loud
    sample — an audible defect from an inaudible one. libsndfile already does
    this for wav and flac; the raw path has to do it itself.
    """
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def check_rate(actual: int, expected: int = SAMPLE_RATE) -> None:
    """Refuse audio at a rate the caller is about to mislabel.

    tts-stack/app/synth.py:67 has this guard. tts-long has NOT: it asserts
    24 kHz in a comment and writes whatever Chatterbox returns into a wav
    header that says 24000. A model update that changed the rate would ship
    every file at the wrong pitch and nothing would report an error — the
    files play, they are simply wrong. That is precisely the failure a cheap
    integer comparison is for.
    """
    if actual != expected:
        raise RuntimeError(
            f"unexpected sample rate {actual}, expected {expected}: the "
            f"output would be written with the wrong rate in its header and "
            f"play at the wrong pitch")


def splice(chunks: list[tuple[np.ndarray | None, float]],
           rate: int = SAMPLE_RATE) -> np.ndarray:
    """Join audio chunks, inserting real silence between them.

    Each item is (audio or None, pause_after in seconds); None is a segment
    whose text was empty, which contributes its pause and nothing else.

    Pauses are generated here rather than asked of the model, and that is the
    shared reasoning worth keeping in one place: no TTS model reliably
    produces a beat you can act inside. Punctuation buys a breath; an
    instruction needs a gap. Measured by ear on the same voice and the same
    words, inserted silence is what separates audio that sounds like
    instructions from audio that sounds like narration.

    The empty case returns zeros(0) rather than raising, matching both
    existing implementations: a request of nothing but pauses is odd but not
    an error, and np.concatenate on an empty list is a ValueError.
    """
    parts: list[np.ndarray] = []
    for audio, pause_after in chunks:
        if audio is not None and audio.size:
            parts.append(audio)
        if pause_after > 0:
            parts.append(np.zeros(int(rate * pause_after), dtype=np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)
