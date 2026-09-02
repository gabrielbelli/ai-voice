"""Encoding, incremental. One generator, both stream formats.

    [ndarray, ndarray, …]  ->  encode_stream(chunks, fmt)  ->  b"", b"", …

**Why this is a generator and not a function returning bytes.** `stream_format`
gives one request two possible shapes for the same audio: a buffered body, and
a run of `speech.audio.delta` events whose base64 payloads a client
concatenates. Those two have to agree — a client that streams and a client that
does not must be able to compare files — and the only way to guarantee that is
for both to come out of the same encoder. So the route drains this generator for
a buffered response and frames each yield as a delta for an SSE one, and the two
cannot drift because there is nothing to drift.

**One sample conversion, `voice_common.audio.pcm_bytes`, feeding every format.**
Before this, `wav` went through libsndfile and `pcm` through pcm_bytes, and the
two rounded differently: measured over one utterance, 54.6% of samples differed,
by up to 2 LSB. Both routes claimed to carry the same samples and did not. The
44-byte header written below is byte-identical to the one libsndfile wrote —
verified against `sf.write` output — so the only change to a `wav` body is that
its samples now match the `pcm` body of the same request exactly.

**mp3, aac, opus and flac go to ffmpeg on a pipe rather than through a temporary
file.** A file has to be complete before it can be read, which is the whole
defect this module exists to remove. Measured feeding a 10 s clip in five
pieces, encoded bytes come back after every piece: mp3 15020 bytes after the
first, aac 15025, flac 22938, opus 135. `-flush_packets 1` changes none of
those numbers, so it is not passed.

Three container-level consequences of encoding to a pipe, all measured against
the same input encoded to a file, all documented in the README:

  * mp3  loses the 192-byte Xing/`Info` frame. It records the total frame count
         and is written by seeking back to the start, which a pipe cannot do.
  * flac STREAMINFO carries zero for total-samples and for the MD5, the
         standard streaming convention for the same reason.
  * aac  is byte-identical either way.
  * opus is byte-identical for a given process, and never reproducible between
         two of them: the Ogg serial number is randomised per stream, so two
         identical buffered requests already returned different bytes today.
  * wav  is the one format whose buffered form is NOT the concatenation of its
         streamed form — see `encode`.
"""

from __future__ import annotations

import collections
import struct
import subprocess
import threading
from collections.abc import Iterable, Iterator

import numpy as np
from voice_common.audio import SAMPLE_RATE, pcm_bytes

__all__ = ["CONTENT_TYPE", "FORMATS", "encode", "encode_stream"]

# Bitrates are set for one voice at 24 kHz, not for music. 32k is what opus
# shipped with here; mp3 and aac get double because their psychoacoustic models
# predate Opus by fifteen years and more, and neither is clean at 32k on speech.
#
# The muxer name is explicit because the output is a pipe: ffmpeg guesses a
# container from the output file's extension, and `pipe:1` has none.
_FFMPEG: dict[str, tuple[list[str], str]] = {
    "opus": (["-b:a", "32k"], "opus"),
    "mp3": (["-b:a", "64k"], "mp3"),
    "aac": (["-b:a", "64k"], "adts"),
    "flac": ([], "flac"),
}

CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    # OpenAI's `pcm` is headerless 24 kHz 16-bit little-endian mono, which is
    # Kokoro's own sample rate — nothing is resampled on the way out.
    "pcm": "audio/pcm",
}

# The order the OpenAI schema lists them in, which is also the order the README
# table uses. `mp3` first because it is the schema's default.
FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

# RIFF and `data` chunk sizes when the length is not yet known. A stream cannot
# write a length before its last sample exists, so the streamed form of a wav
# carries this and the buffered form overwrites it — see `encode`.
_UNKNOWN_LENGTH = 0xFFFFFFFF

# Byte offsets of those two fields inside the 44-byte header.
_RIFF_SIZE_AT = 4
_DATA_SIZE_AT = 40


def _wav_header(data_bytes: int | None) -> bytes:
    """The canonical 44-byte header, or its unknown-length form for a stream.

    Written here rather than taken from libsndfile because libsndfile cannot
    produce one until it knows the length. The bytes are identical to what
    `sf.write` produced for the same rate and sample count, checked field by
    field, so nothing that reads a wav from this service sees a new shape.
    """
    size = _UNKNOWN_LENGTH if data_bytes is None else data_bytes
    riff = _UNKNOWN_LENGTH if data_bytes is None else 36 + data_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE,
                          SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", size))


class _Ffmpeg:
    """One ffmpeg process, fed samples and drained as it produces bytes.

    stdout is drained by a thread because ffmpeg writes while we are still
    writing to its stdin, and a pipe that nobody reads fills and blocks the
    encoder, which then stops reading stdin, which blocks us: a deadlock that
    only appears on inputs long enough to fill 64 KB, which is to say in
    production and not in a test.

    The thread reads with `read1`, not `read`. `read(n)` on a buffered pipe
    blocks until it has n bytes or sees EOF, which would hold every chunk back
    until the end and reintroduce exactly the latency this module removes —
    measured: with `read`, zero bytes were visible after four of five feeds.
    """

    def __init__(self, fmt: str) -> None:
        args, muxer = _FFMPEG[fmt]
        try:
            self._proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error",
                 "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-i", "pipe:0", *args, "-f", muxer, "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            # ffmpeg absent from PATH entirely, which is the one failure this
            # class had nothing to say about: `finish()` turns a non-zero exit
            # into ffmpeg's own words, but a binary that never started has no
            # stderr to quote, so Popen's bare "No such file or directory:
            # 'ffmpeg'" was all that reached the 500.
            #
            # It cost a full CI round to read. The conformance job had no
            # ffmpeg, so the default mp3 request 500ed, and the failures
            # surfaced as KeyError on X-Speed-Clamped and X-Ignored-Parameters
            # in tests that never mention encoding — sending the search to the
            # header code in app.main, which was correct all along. Naming the
            # missing dependency and the format that wanted it puts the next
            # reader at the cause rather than three frames downstream of it.
            raise RuntimeError(
                f"ffmpeg is not on PATH, and {fmt} is encoded by it. Install "
                "ffmpeg, or ask for a format that needs no encoder: wav and "
                "pcm are written here.") from exc
        # deque.append and popleft are atomic, so the reader thread and the
        # request thread need no lock between them.
        self._out: collections.deque[bytes] = collections.deque()
        self._err: list[bytes] = []
        self._readers = [threading.Thread(target=self._drain, daemon=True),
                         threading.Thread(target=self._drain_err, daemon=True)]
        for thread in self._readers:
            thread.start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        while chunk := self._proc.stdout.read1(65536):
            self._out.append(chunk)

    def _drain_err(self) -> None:
        assert self._proc.stderr is not None
        # stderr is drained for the same reason stdout is: at `-loglevel error`
        # it is normally empty, but a pipe nobody reads is a deadlock waiting
        # for the one input that makes ffmpeg complain.
        while chunk := self._proc.stderr.read1(65536):
            self._err.append(chunk)

    def _take(self) -> bytes:
        parts = []
        while self._out:
            parts.append(self._out.popleft())
        return b"".join(parts)

    def feed(self, samples: np.ndarray) -> bytes:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(pcm_bytes(samples))
            self._proc.stdin.flush()
        except BrokenPipeError:
            # ffmpeg died mid-stream. Its own words are more use than
            # "broken pipe", so finish() is left to raise them.
            pass
        return self._take()

    def finish(self) -> bytes:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        for thread in self._readers:
            thread.join()
        code = self._proc.wait()
        if code != 0:
            message = b"".join(self._err).decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg exited {code}: {message or 'no output'}")
        return self._take()

    def abort(self) -> None:
        """Kill the process when the caller gives up part way through.

        A client that disconnects mid-stream leaves the generator unfinished,
        and without this the ffmpeg process survives it.
        """
        if self._proc.poll() is None:
            self._proc.kill()
        for thread in self._readers:
            thread.join(timeout=1.0)


def encode_stream(chunks: Iterable[np.ndarray], fmt: str) -> Iterator[bytes]:
    """Encode audio chunks to `fmt`, yielding bytes as soon as there are any.

    Empty yields are suppressed: a codec with a lookahead window returns
    nothing for the first sample it is handed, and an SSE delta carrying an
    empty string is a frame that says nothing.
    """
    if fmt == "pcm":
        for chunk in chunks:
            if chunk.size:
                yield pcm_bytes(chunk)
        return

    if fmt == "wav":
        first = True
        for chunk in chunks:
            body = pcm_bytes(chunk) if chunk.size else b""
            if first:
                first = False
                yield _wav_header(None) + body
            elif body:
                yield body
        if first:
            # No audio at all. The header still has to go out, or the caller
            # gets a zero-byte body where a valid empty wav was asked for.
            yield _wav_header(None)
        return

    encoder = _Ffmpeg(fmt)
    try:
        for chunk in chunks:
            if not chunk.size:
                continue
            if data := encoder.feed(chunk):
                yield data
        if data := encoder.finish():
            yield data
    except GeneratorExit:
        encoder.abort()
        raise
    except Exception:
        encoder.abort()
        raise


def encode(chunks: Iterable[np.ndarray], fmt: str) -> bytes:
    """The buffered body: the same bytes, with a wav length filled in.

    For mp3, aac, flac and pcm this is exactly `b"".join(encode_stream(…))`, and
    the deltas of an SSE response for the same request concatenate to it byte
    for byte — asserted in tests/test_openai_speech.py for every format. `opus`
    agrees everywhere but the Ogg page serial and the CRC covering it, which
    the muxer randomises per stream: 40 bytes of 8980 on a five-page stream,
    and already true of two identical buffered requests before any of this.

    `wav` is the one exception and it is a physical one: the RIFF and `data`
    chunk sizes cannot be written before the last sample exists. A buffered
    response knows them, so it writes them; a streamed one leaves them at
    0xFFFFFFFF, the streaming convention. The two therefore differ in exactly
    eight bytes, at offsets 4 and 40, and in no sample. Leaving the placeholder
    in the buffered body instead was measured and rejected: soundfile, ffprobe
    and CoreAudio all read it, but Python's `wave` module then reports
    2147483647 frames.
    """
    body = b"".join(encode_stream(chunks, fmt))
    if fmt != "wav":
        return body
    data_bytes = len(body) - len(_wav_header(None))
    return (body[:_RIFF_SIZE_AT] + struct.pack("<I", 36 + data_bytes)
            + body[_RIFF_SIZE_AT + 4:_DATA_SIZE_AT]
            + struct.pack("<I", data_bytes) + body[_DATA_SIZE_AT + 4:])
