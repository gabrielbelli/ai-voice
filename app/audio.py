"""Audio in, 16 kHz mono float32 out.

The specification lists nine input formats — flac, mp3, mp4, mpeg, mpga, m4a,
ogg, wav and webm — and this service decoded four. libsndfile reads none of the
MPEG-4 family, so an m4a and a webm were both answered `400 could not decode
audio: ... Format not recognised`. Those two are exactly what an iOS client and
a browser MediaRecorder produce, so the formats most likely to arrive were the
ones certain to fail.

libav decodes all nine, and resamples and downmixes on the way through.
Measured here on the same 14.2 s clip in every container, decoding costs 5–20 ms
once the codec is warm — mp3 0.013 s, m4a 0.009 s, webm 0.021 s, 44.1 kHz wav
0.018 s — which is under half a percent of the 5 s Parakeet spends on the same
clip. It arrives with faster-whisper (`av>=11`), so the image already carried
it; using it directly costs an explicit pin and lets libsndfile leave.

The source sample rate is reported rather than hidden, because the two routes
disagree about it on purpose: /v1 resamples, because no OpenAI client expects
anything else, and the native route still refuses. See pipeline.decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(ValueError):
    """The bytes could not be decoded. Carries the reason for the client."""


@dataclass(frozen=True)
class Decoded:
    """The audio, plus what it was before this module touched it."""

    samples: np.ndarray  # float32, mono, SAMPLE_RATE
    source_rate: int
    source_channels: int


def decode(raw: bytes) -> Decoded:
    """Decode any container libav reads into 16 kHz mono float32.

    Resampling always happens; whether it is *allowed* is the caller's
    decision, which is why source_rate comes back rather than being consumed
    here. The cost of resampling a clip that turns out to be rejected is the
    0.018 s measured above, and doing it unconditionally keeps one decode path
    instead of two.
    """
    import av  # noqa: PLC0415 - libav's import pulls the codec tables; keep it off module load

    try:
        with av.open(BytesIO(raw), metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise AudioError("no audio stream")
            stream = container.streams.audio[0]
            source_rate = int(stream.rate or 0)
            source_channels = int(getattr(stream, "channels", 0) or 0)

            # "flt" is packed float32, which is what both recognisers want, so
            # nothing here converts a second time. Mono is a downmix, matching
            # what the previous soundfile path did with .mean(axis=1).
            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=SAMPLE_RATE
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # Flush: swresample holds a tail of samples inside its filter delay,
            # and dropping it truncates the last few milliseconds of every clip.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise AudioError(str(exc)) from exc

    samples = (
        np.concatenate(chunks).astype(np.float32)
        if chunks
        else np.zeros(0, dtype=np.float32)
    )
    return Decoded(samples=samples, source_rate=source_rate,
                   source_channels=source_channels)
