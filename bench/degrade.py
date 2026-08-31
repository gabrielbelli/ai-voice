"""Degradations applied to clean audio, to measure the curve rather than a point.

A single WER on studio audio says almost nothing about a real dictation setup.
The failure that actually cost the most in practice was a laptop microphone
inside a closed lid — no model recovers audio that muffled, and no clean-speech
benchmark predicts it.

Each degradation is applied to the SAME clips at several levels, so the result
is WER as a function of condition. Where the curve bends is the useful part:
it tells you which side of the pipeline to spend on. If WER is flat to 10 dB
SNR and collapses at 5, a better microphone beats a better model.

ffmpeg is required for the codec round-trips only. Everything else is numpy.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------- noise ----

def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise. Closer than white to room tone, fans and traffic."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    return np.fft.irfft(spectrum / np.sqrt(freqs), n).astype(np.float32)


def babble(clips: list[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """Overlapping speech, built from the corpus itself so nothing is downloaded.

    Babble is the hardest noise for ASR because it is speech: the model has to
    decide which voice to follow, and Whisper in particular will happily
    transcribe the wrong one.
    """
    out = np.zeros(length, dtype=np.float32)
    for _ in range(6):
        c = clips[rng.integers(len(clips))]
        if c.size < length:
            c = np.tile(c, int(np.ceil(length / c.size)))
        start = rng.integers(0, max(1, c.size - length))
        out += c[start:start + length]
    return out / 6.0


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix to a target signal-to-noise ratio, measured on the clean signal."""
    if noise.size < clean.size:
        noise = np.tile(noise, int(np.ceil(clean.size / noise.size)))
    noise = noise[:clean.size]
    p_sig = float(np.mean(clean ** 2)) or 1e-12
    p_noi = float(np.mean(noise ** 2)) or 1e-12
    scale = np.sqrt(p_sig / (p_noi * 10 ** (snr_db / 10)))
    return np.clip(clean + noise * scale, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------- microphone band ----

def bandlimit(x: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Brick-wall band limit in the frequency domain.

    Stands in for microphone quality. A closed-lid laptop mic, a cheap headset
    and a phone line all lose the same thing first: high-frequency energy, and
    with it the fricatives and stops that distinguish similar words. This is
    why "TLDR" loses its leading T.
    """
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / SAMPLE_RATE)
    spec[(freqs < low_hz) | (freqs > high_hz)] = 0
    return np.fft.irfft(spec, x.size).astype(np.float32)


def reverb(x: np.ndarray, decay: float = 0.35, delay_ms: float = 45) -> np.ndarray:
    """Cheap exponential-decay impulse response — far-field / hard room.

    Not a measured RIR. It is enough to show whether a setup degrades
    gracefully when the microphone is across the desk rather than at the mouth.
    """
    d = int(delay_ms * SAMPLE_RATE / 1000)
    ir = np.zeros(d * 6, dtype=np.float32)
    for k in range(6):
        ir[k * d] = decay ** k
    out = np.convolve(x, ir)[: x.size]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * float(np.max(np.abs(x)))).astype(np.float32)


def clip_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Overdrive then hard-clip — an input gain set far too hot."""
    return np.clip(x * (10 ** (gain_db / 20)), -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- codecs ------

def codec(x: np.ndarray, name: str, bitrate: str) -> np.ndarray:
    """Encode and decode through a real codec, back to 16 kHz mono float32.

    Matters because audio reaching a self-hosted endpoint has usually crossed
    a network. Opus at 16 kbps is a plausible phone-to-server setting and is
    NOT transparent for speech recognition, even where it sounds fine.
    """
    if not HAVE_FFMPEG:
        raise RuntimeError("ffmpeg not found; codec degradations unavailable")
    ext = {"opus": "opus", "mp3": "mp3", "amr": "amr"}[name]
    with tempfile.TemporaryDirectory() as td:
        raw, enc, dec = Path(td) / "a.wav", Path(td) / f"b.{ext}", Path(td) / "c.wav"
        import soundfile as sf
        sf.write(raw, x, SAMPLE_RATE)
        rate = "8000" if name == "amr" else str(SAMPLE_RATE)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-b:a", bitrate, "-ar", rate, str(enc)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc),
                        "-ar", str(SAMPLE_RATE), "-ac", "1", str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float32")
    return y


# --------------------------------------------------------------- matrix ----

# Every condition applied to the same clips. `clean` is the control and must
# always run first — a degraded WER means nothing without it.
MATRIX: list[tuple[str, dict]] = [
    ("clean",            {}),

    ("pink-20db",        {"noise": "pink",   "snr": 20}),
    ("pink-10db",        {"noise": "pink",   "snr": 10}),
    ("pink-5db",         {"noise": "pink",   "snr": 5}),
    ("babble-10db",      {"noise": "babble", "snr": 10}),
    ("babble-5db",       {"noise": "babble", "snr": 5}),

    # 6.5 kHz  a decent headset
    # 4 kHz    a cheap or distant laptop microphone
    # 300-3400 telephone band, the floor
    ("mic-good",         {"band": (60, 6500)}),
    ("mic-cheap",        {"band": (100, 4000)}),
    ("mic-phoneline",    {"band": (300, 3400)}),
    ("mic-closed-lid",   {"band": (150, 3000), "reverb": True}),

    ("room-reverb",      {"reverb": True}),
    ("hot-gain",         {"gain_db": 14}),

    ("opus-32k",         {"codec": ("opus", "32k")}),
    ("opus-16k",         {"codec": ("opus", "16k")}),
    ("mp3-32k",          {"codec": ("mp3", "32k")}),
    ("amr-nb",           {"codec": ("amr", "12.2k")}),
]


def apply(x: np.ndarray, spec: dict, pool: list[np.ndarray],
          rng: np.random.Generator) -> np.ndarray:
    """Apply one MATRIX entry. Order is deliberate and mirrors physics:
    the room acts before the microphone, the microphone before the codec."""
    y = x
    if spec.get("reverb"):
        y = reverb(y)
    if "noise" in spec:
        n = babble(pool, y.size, rng) if spec["noise"] == "babble" else _pink(y.size, rng)
        y = add_noise(y, n, spec["snr"])
    if "band" in spec:
        y = bandlimit(y, *spec["band"])
    if "gain_db" in spec:
        y = clip_gain(y, spec["gain_db"])
    if "codec" in spec:
        y = codec(y, *spec["codec"])
    return y
