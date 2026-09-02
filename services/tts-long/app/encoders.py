"""One encoder, used both ways: all at once, or a chunk at a time.

`stream_format: "sse"` promises that base64-decoding every `speech.audio.delta`
and concatenating the result reproduces the body the same request would have
returned buffered. The only way to keep that promise honestly is to have one
implementation and feed it either the whole array or a sequence of pieces, so
this module exposes exactly that: `Encoder.write()` returns whatever bytes are
ready, `Encoder.close()` returns the rest.

Measured here on ffmpeg 9.0.1 with three seconds of tone, whole versus seven
chunks through the same command: **mp3, aac, flac and wav are byte-identical**.
`opus` is not, and that is not a chunking artefact — the Ogg serial number is
random per stream, so two whole-file encodes of the same samples differ too.
Inside one request it does not matter: a streamed response is a single encode,
partitioned, so its deltas always concatenate to exactly what that encode
produced.

**The two container formats carry a deviation, and it is measured rather than
estimated.** WAV and FLAC state the total length in their header, which is not
known until generation ends. libsndfile patches those fields on close; a
stream has already sent them. Diffed byte for byte:

  * `wav`  — the streamed bytes differ from the buffered file in the two RIFF
    size fields only, at offsets 4 and 40. Byte 44 onward is identical.
  * `flac` — the streamed bytes differ inside STREAMINFO only (total samples
    and the MD5 of the audio, offsets 8 to 41). Every audio frame is identical.

Both remain playable — a zero length is how a WAV or FLAC stream of unknown
duration is written — and both are noted in the README as deviations. `pcm`,
`mp3` and `aac` have no such field and are exact.

The bitrates are tts-stack's, deliberately: 32k opus, 64k mp3 and aac, chosen
for one voice at 24 kHz rather than for music. Two services in one estate
answering the same request with different bitrates is drift nobody would
notice until they compared files.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import threading
from collections import deque

import numpy as np
import soundfile as sf
from voice_common.audio import SAMPLE_RATE, pcm_bytes

__all__ = ["FORMATS", "MEDIA_TYPES", "available_formats", "encode",
           "make_encoder", "ffmpeg_available"]

# Every format OpenAI's schema names, with the MIME this estate answers with.
#
# `audio/pcm` is a local decision — the OpenAI schema specifies no per-format
# MIME for this endpoint — and it is written down here because it has to be
# THE SAME STRING in tts-stack and stt-stack or the estate drifts. It is the
# one tts-stack already sends. `audio/ogg` for opus rather than `audio/opus`
# for the same reason: what comes back is an Ogg stream, and that is what
# tts-stack calls it.
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# What ffmpeg is asked for. Raw 24 kHz mono s16le in, the named container out.
_FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"],
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
}

# libsndfile's, and no external process involved.
_SOUNDFILE_FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", None)}

# How long a streaming write waits for ffmpeg to hand back the bytes for the
# samples it was just given. See _FfmpegEncoder.write.
_FLUSH_WAIT = float(os.getenv("TTS_ENCODER_FLUSH_WAIT", "0.5"))

FORMATS = tuple(MEDIA_TYPES)


def ffmpeg_available() -> bool:
    """Is the ffmpeg binary on PATH?

    Probed rather than assumed. The image installs it, but this service also
    runs from a checkout, and an encoder that is missing should produce a 400
    naming the format rather than a 500 out of a subprocess.
    """
    return shutil.which("ffmpeg") is not None


def available_formats() -> tuple[str, ...]:
    """The formats this process can actually produce, in schema order."""
    if ffmpeg_available():
        return FORMATS
    return tuple(f for f in FORMATS if f not in _FFMPEG_ARGS)


class Encoder:
    """Incremental encoder. `write` returns what is ready, `close` the rest."""

    media_type = "application/octet-stream"

    def write(self, audio: np.ndarray) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class _RawEncoder(Encoder):
    """OpenAI's `pcm`: headerless 24 kHz 16-bit mono little-endian.

    Chatterbox's native rate, so nothing is resampled. The clip-before-scale
    that keeps a single overshoot from wrapping into a click is
    voice_common.audio.pcm_bytes.
    """

    media_type = MEDIA_TYPES["pcm"]

    def write(self, audio: np.ndarray) -> bytes:
        return pcm_bytes(audio)

    def close(self) -> bytes:
        return b""


class _SoundFileEncoder(Encoder):
    """wav and flac, through libsndfile, drained as it writes.

    The buffer is a BytesIO because libsndfile wants a seekable target: it
    seeks back on close to patch the length into the header. Everything sent
    before that point is already on the wire, which is the deviation described
    in the module docstring — and the reason the drained prefix is tracked by
    offset rather than re-read from the start.
    """

    def __init__(self, fmt: str) -> None:
        container, subtype = _SOUNDFILE_FORMATS[fmt]
        self.media_type = MEDIA_TYPES[fmt]
        self._buffer = io.BytesIO()
        self._sent = 0
        kwargs = {"subtype": subtype} if subtype else {}
        self._file = sf.SoundFile(self._buffer, mode="w",
                                  samplerate=SAMPLE_RATE, channels=1,
                                  format=container, **kwargs)

    def _drain(self) -> bytes:
        current = self._buffer.getbuffer().nbytes
        if current <= self._sent:
            return b""
        data = self._buffer.getvalue()[self._sent:current]
        self._sent = current
        return data

    def write(self, audio: np.ndarray) -> bytes:
        self._file.write(audio)
        self._file.flush()
        return self._drain()

    def close(self) -> bytes:
        if not self._file.closed:
            self._file.close()
        return self._drain()


class _FfmpegEncoder(Encoder):
    """mp3, opus and aac, through one long-lived ffmpeg process.

    One process for the whole request, not one per chunk: a fresh encoder per
    chunk would emit its own headers and padding, and the concatenation would
    no longer be the file a buffered request returns.

    stdout is drained by a thread. Without it, ffmpeg fills the pipe buffer and
    blocks on its own write while this side is still writing samples to stdin,
    which is a deadlock with no timeout on either end.
    """

    def __init__(self, fmt: str) -> None:
        if not ffmpeg_available():
            raise RuntimeError(
                f"{fmt} needs ffmpeg and the binary is not on PATH")
        self.media_type = MEDIA_TYPES[fmt]
        self._process = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
             *_FFMPEG_ARGS[fmt], "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self._chunks: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        while True:
            # read1, not read: BufferedReader.read(n) blocks until it has all
            # n bytes or the pipe closes, so a 64 KB request held the first
            # delta back until 64 KB of ENCODED audio existed — sixteen seconds
            # of it at opus's 32 kbps, which is a whole chunk late. read1
            # returns whatever one underlying read gives.
            data = self._process.stdout.read1(65536)
            if not data:
                self._arrived.set()
                return
            with self._lock:
                self._chunks.append(data)
            self._arrived.set()

    def _take(self) -> bytes:
        with self._lock:
            data = b"".join(self._chunks)
            self._chunks.clear()
            self._arrived.clear()
        return data

    def write(self, audio: np.ndarray) -> bytes:
        assert self._process.stdin is not None
        self._process.stdin.write(pcm_bytes(audio))
        self._process.stdin.flush()
        # Wait, briefly, for the encoder to catch up with what it was just
        # given. Without this the return raced ffmpeg by a few milliseconds
        # and every chunk's bytes were emitted with the NEXT one — measured on
        # opus, where a two-chunk stream produced its first delta at 5.8 s of a
        # 5.9 s request rather than at 3.0 s. Half a second is nothing against
        # a chunk that takes tens of seconds to generate, and an empty return
        # after it is normal: a lossy encoder is entitled to hold a frame.
        self._arrived.wait(_FLUSH_WAIT)
        return self._take()

    def close(self) -> bytes:
        assert self._process.stdin is not None
        if not self._process.stdin.closed:
            self._process.stdin.close()
        self._reader.join(timeout=30)
        code = self._process.wait(timeout=30)
        data = self._take()
        if code != 0:
            stderr = (self._process.stderr.read() or b"").decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed ({code}): {stderr.strip()[:500]}")
        return data

    def kill(self) -> None:
        """Abandon the process. For a cancelled job, where close() is a wait."""
        if self._process.poll() is None:
            self._process.kill()


def make_encoder(fmt: str) -> Encoder:
    """An encoder for `fmt`. Raises RuntimeError if this image cannot do it."""
    if fmt == "pcm":
        return _RawEncoder()
    if fmt in _SOUNDFILE_FORMATS:
        return _SoundFileEncoder(fmt)
    if fmt in _FFMPEG_ARGS:
        return _FfmpegEncoder(fmt)
    raise RuntimeError(f"unknown format {fmt!r}")


def encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    """The whole array at once. This is the canonical body for `fmt`.

    wav and flac go straight to libsndfile rather than through the streaming
    encoder above, and that is the difference the module docstring quantifies:
    the buffered file gets its header patched with the real length on close,
    which a stream has already sent and cannot take back. Everything else —
    pcm, mp3, opus, aac — is the same encoder either way, so the streamed and
    buffered bodies are the same bytes.
    """
    if fmt in _SOUNDFILE_FORMATS:
        container, subtype = _SOUNDFILE_FORMATS[fmt]
        buffer = io.BytesIO()
        kwargs = {"subtype": subtype} if subtype else {}
        sf.write(buffer, audio, SAMPLE_RATE, format=container, **kwargs)
        return buffer.getvalue(), MEDIA_TYPES[fmt]

    encoder = make_encoder(fmt)
    try:
        head = encoder.write(audio) if audio.size else b""
        return head + encoder.close(), encoder.media_type
    except Exception:
        if isinstance(encoder, _FfmpegEncoder):
            encoder.kill()
        raise
