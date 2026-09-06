"""Self-hosted text-to-speech. Kokoro, CPU only.

    text or segments -> phonemise -> Kokoro -> wav, opus, mp3, aac, flac, pcm

There is one model and it stays resident: at 330 MB and 4x realtime on CPU it
is too cheap to unload. Chatterbox, the long-form alternative, is deliberately
not here — it needs 5.3 GB and runs below realtime, so it belongs behind a
separate service that can be started on demand.

Two request shapes reach the same synthesiser. `/speak` is the native one and
the one to prefer: it takes segments with explicit pauses and a voice each,
and returns the realtime factor in headers. `/v1/audio/speech` is OpenAI's
shape, kept alongside so existing clients need only a base URL change; it can
express none of those things.

`/v1/audio/speech` answers `stream_format: "sse"` with real server-sent events.
The synthesiser produces a chunk at a time, so a delta leaves as soon as the
first chunk is encoded rather than when the last one is: measured on the
schema's own 4096-character maximum, the first frame goes out after 5.49 s of a
55.06 s generation, against 58.98 s before a buffered response sends anything at
all. That parameter used to be accepted and dropped — a caller that asked for a
stream got a single buffered mp3, HTTP 200, no error, and no way to tell.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from voice_common import auth, logging as voice_logging
from voice_common.errors import error_response, install_errors
from voice_common.health import install_health
from voice_common.models import OpenAISpeechRequest, Segment as BaseSegment

from .audio_out import CONTENT_TYPE, FORMATS, encode, encode_stream
from .openai_api import (VOICE_ALIASES, custom_voice_id, language_for_voice,
                         resolve_voice, unmapped_aliases)
from .synth import FRAME_SAMPLES, MAX_CHUNK_PHONEMES, SAMPLE_RATE, Synth

MODEL_DIR = Path(os.getenv("TTS_MODEL_DIR", "/models"))
MODEL_URL = os.getenv(
    "TTS_MODEL_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
VOICES_URL = os.getenv(
    "TTS_VOICES_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
DEFAULT_VOICE = os.getenv("TTS_VOICE", "bm_george")
DEFAULT_LANG = os.getenv("TTS_LANGUAGE", "en-us")
THREADS = int(os.getenv("TTS_THREADS", "4"))

# How much of the utterance goes into one model pass, in phonemes, and so how
# often an SSE delta leaves. The default is the model's own context window less
# the one row that cannot be indexed, which is also the batching upstream would
# have chosen on any text it handled correctly, so the audio is what this
# service returned before streaming existed.
#
# Lowering it buys latency and costs length. Measured on a 4096-character input:
# 509 gives 7 chunks, 154.5 s of audio and 6.21 s to the first; 200 gives 17
# chunks, 174.5 s and 2.25 s; 100 gives 39 chunks, 180.8 s and 1.29 s. The 17%
# growth is the duration predictor seeing less context, not silence at the
# seams, so it is a real change to the speech and not a packaging one.
CHUNK_PHONEMES = min(max(int(os.getenv("TTS_CHUNK_PHONEMES",
                                       str(MAX_CHUNK_PHONEMES))), 1),
                     MAX_CHUNK_PHONEMES)

# OpenAI's own maximum for `input`, and the schema's. It was not enforced, so a
# single synchronous request had no upper bound on how long it could run.
MAX_INPUT_CHARS = 4096

# What actually synthesises, and so the one `model` value that is not a
# deviation. There is one model in this image; see _deviations for why any
# other name is answered rather than rejected.
MODEL_NAME = "kokoro"

# Same one line of configuration this always had, plus the TTS_LOG_LEVEL switch
# it never had: getting DEBUG out of a running container used to mean editing
# the source and rebuilding the image, which is exactly the moment that is
# impossible. Unset still means INFO, so nothing changes for anyone who has not
# asked for it.
log = voice_logging.setup("tts-stack", "TTS")

state: dict[str, object] = {}


class _Rate:
    """The observed realtime factor of THIS container, as an EMA.

    Reported in /health so a client does not have to write the number down.
    It is a property of a machine and of its configuration, not of Kokoro: the
    same model on the same NAS measured 1.83x realtime at TTS_THREADS=4 and
    2.79x at 8, so any constant a caller keeps is a claim about a deployment
    that a rebuild can make false without telling anyone.

    UNSEEDED, AND THAT IS THE POINT. tts-long seeds its EMA from a constant so
    that its queue arithmetic always has a number, which means its
    realtime_factor is never absent and a seed is indistinguishable from a
    measurement. A client deciding whether audio can be played as it arrives
    must be able to tell those apart, so this one reports nothing at all until
    something has actually been synthesised, and `value` stays None until then.

    The first observation is taken whole rather than blended into a seed that
    was never measured. Later ones move it by 0.3, the same weight tts-long
    uses, so one unusually short request cannot swing the figure a client is
    about to plan a playback buffer against.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self.samples = 0
        self._lock = threading.Lock()

    def observe(self, audio_seconds: float, compute_seconds: float) -> None:
        if audio_seconds <= 0 or compute_seconds <= 0:
            return
        measured = audio_seconds / compute_seconds
        with self._lock:
            self.samples += 1
            self.value = (measured if self.value is None
                          else self.value + 0.3 * (measured - self.value))


rate = _Rate()


def _fetch(url: str, dest: Path) -> None:
    if dest.is_file():
        return
    log.info("downloading %s", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSfL", "-o", str(tmp), url], check=True)
    tmp.rename(dest)  # rename last, so a killed download never looks complete


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    started = time.monotonic()
    model = MODEL_DIR / "kokoro.onnx"
    voices = MODEL_DIR / "voices.bin"
    _fetch(MODEL_URL, model)
    _fetch(VOICES_URL, voices)
    synth = Synth(str(model), str(voices))
    state["synth"] = synth

    # The alias table names Kokoro voices from outside the model file, so a
    # different voices.bin could leave one of the thirteen OpenAI names
    # pointing at nothing. Complain at load rather than at the first request
    # that uses it — the same failure the espeak wiring exists to avoid.
    missing = unmapped_aliases(synth.voices)
    if missing:
        log.warning("OpenAI voice names absent from this voices file: %s; "
                    "requests using them will be rejected", ", ".join(missing))

    log.info("ready in %.1fs, %d threads", time.monotonic() - started, THREADS)
    yield
    state.clear()


app = FastAPI(title="tts-stack",
              description="Kokoro text-to-speech, CPU only.",
              lifespan=lifespan)

# Installed on the app rather than route by route, so a route added later is
# covered without anyone having to remember to ask for it. The variable name is
# a parameter of the shared middleware precisely so that TTS_API_KEYS did not
# have to be renamed for this service to stop keeping its own copy.
auth.install(app, "TTS_API_KEYS")

# The /v1 error envelope — all four fields, and the 404, 405 and 500 handlers
# that used to escape it. This repo carried app/errors.py to add `param` and
# those handlers on top of a shared package it could not change; both now live
# upstream, so there is nothing to register second. The native routes keep
# FastAPI's own `{"detail": ...}`: clients are already written against that
# shape and /v1 is the only compatibility boundary here.
install_errors(app)


class Segment(BaseSegment):
    """A native-API segment: shared text and pause, plus this service's voice.

    `text` and the 0.0–10.0 second `pause_after` come from
    voice_common.models.Segment, where the bounds are a published part of two
    services' APIs and no longer free to drift in one of them.
    """

    # A voice is a 510 KB embedding against 310 MB of shared weights, so
    # changing it between segments costs nothing once the model is loaded.
    # Absent means the request's voice. The phonemiser language is not
    # inferred from it: `language` stays a property of the request, as it has
    # always been on this route. Kokoro-specific, so it stays here.
    voice: str | None = None


class SpeakRequest(BaseModel):
    # Unknown fields are rejected rather than dropped. A per-segment `voice`
    # was documented for as long as it was silently ignored, and the caller
    # got the default voice back with nothing to read that said why. A typo
    # belongs in a 422, not in the audio.
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    segments: list[Segment] | None = None
    voice: str | None = None
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="wav", pattern="^(wav|opus|mp3|aac|flac|pcm)$")


class SpeechRequest(OpenAISpeechRequest):
    """OpenAI's /v1/audio/speech body, with the parts that are Kokoro's.

    `model`, `input`, `voice` and the `extra="allow"` rule come from
    voice_common.models.OpenAISpeechRequest. That last one is the reason the
    base class exists: OpenAI keeps adding fields, and a service whose whole
    purpose is to answer OpenAI's clients must not reject one for speaking a
    newer version of the dialect it claims to speak.

    `extra="allow"` is not licence to drop what arrives, though, and it used to
    be. `stream: true` — a field the speech schema forbids, and the classic
    confusion with the transcription one — came back 200 with an ordinary
    buffered mp3 and no signal of any kind. Anything unrecognised is now named
    in `X-Ignored-Parameters` on the response, so the answer to "did that do
    anything?" is on the wire rather than in this file.

    Everything else here is a property of this image rather than of the shape:
    the encoder list is what ffmpeg can produce in this image, the speed range
    is OpenAI's published one, and the two enums are the schema's own.
    """

    # "kokoro" rather than the base's "default", because that is what this
    # service answers with and something may well be reading it back. Any other
    # name is accepted and named in X-Ignored-Parameters: see _deviations.
    model: str = MODEL_NAME

    # maxLength 4096 is in the schema and was not enforced here: bodies well
    # past it were accepted and attempted, which left no upper bound at all on
    # how long one synchronous request could run.
    input: str = Field(max_length=MAX_INPUT_CHARS)

    # Accepted, never honoured, and now said so out loud. Kokoro-82M has no
    # style, prosody or emotion conditioning — a voice is a 510 KB embedding
    # selected by name, and the ONNX graph takes tokens, style and speed and
    # nothing else — so there is no input to route a sentence of direction
    # into. Proved rather than assumed: two requests differing only by
    # instructions="Speak in an extremely angry shouting voice, very fast"
    # returned byte-identical audio, SHA-1 b477f15864f99dc5.
    #
    # Ignored rather than rejected, and the README says so in a row of its own.
    # OpenAI documents that `instructions` does not work with tts-1 or
    # tts-1-hd either, so a client that sends it already tolerates no effect;
    # a 400 would break clients for a field the upstream API also ignores.
    # The response names it in X-Ignored-Parameters, so the silence is gone
    # even though the behaviour has not changed.
    instructions: str | None = Field(default=None, max_length=MAX_INPUT_CHARS)

    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal[FORMATS] = "mp3"  # type: ignore[valid-type]

    # Validated against its enum, which it was not: `stream_format:
    # "nonsense_value"` returned 200 with the same mp3 as `"audio"`, so a
    # client that misspelled "sse" got exactly the same silence as one that
    # spelled it right.
    stream_format: Literal["audio", "sse"] = "audio"

    @field_validator("voice", mode="before")
    @classmethod
    def _unwrap_custom_voice(cls, value: object) -> object:
        """`{"id": "voice_1234"}` is the schema's other form of `voice`."""
        return custom_voice_id(value)


def _health() -> dict[str, object]:
    """The body, unchanged. The route and its auth exemption are shared.

    install_health registers `/health` AND exempts exactly that string from
    authentication, so a rename cannot lock a container healthcheck out — the
    two used to be independent literals in two modules. It also makes the route
    a coroutine: a sync one shares AnyIO's forty-thread pool with /speak, and
    enough concurrent synthesis requests took the pool and stopped /health
    answering while the service was merely busy. Nothing here blocks.
    """
    s = state.get("synth")
    out: dict[str, object] = {"status": "ok" if s else "loading",
                              "voices": len(getattr(s, "voices", [])),
                              "default_voice": DEFAULT_VOICE,
                              "threads": THREADS}
    # HOW FAST THIS CONTAINER SPEAKS, measured, and ABSENT UNTIL IT IS.
    #
    # A client that plays audio as it arrives has to know whether generation
    # outruns playback before it starts, and the only alternative to this field
    # is a constant written into the client -- which is a claim about a machine
    # and a thread count that the client cannot check. tts-long has reported
    # its own factor all along; this service reported none, so the fast engine
    # was the one being guessed about.
    #
    # Missing means "not measured on this process yet", which is a real state
    # and not a zero. A client that treats an absent field as a number is
    # exactly the failure this shape exists to prevent.
    if rate.value is not None:
        out["realtime_factor"] = round(rate.value, 2)
        # How many requests are behind it, because one observation is a
        # warm-up and a caller may reasonably want more before it commits.
        out["realtime_factor_samples"] = rate.samples
    return out


install_health(app, _health)


@app.get("/voices")
def voices() -> dict[str, object]:
    s = state.get("synth")
    if not s:
        raise HTTPException(503, "model still loading")
    all_voices = s.voices  # type: ignore[attr-defined]
    # Kokoro encodes locale in the prefix: p = Portuguese, a/b = US/UK English.
    return {"voices": all_voices,
            "pt_br": [v for v in all_voices if v.startswith(("pf_", "pm_"))],
            "en_us": [v for v in all_voices if v.startswith(("af_", "am_"))],
            "en_gb": [v for v in all_voices if v.startswith(("bf_", "bm_"))],
            "openai_aliases": VOICE_ALIASES}


def _headers(duration: float, compute: float) -> dict[str, str]:
    return {"X-Audio-Seconds": f"{duration:.2f}",
            "X-Compute-Seconds": f"{compute:.2f}",
            "X-Realtime-Factor": f"{duration / compute:.1f}" if compute else "0"}


def _deviations(req: SpeechRequest, speed: float) -> dict[str, str]:
    """Headers naming every part of the request that did not reach the audio.

    Nothing here changes what the service does. It changes what it admits to,
    which is the entire complaint: `speed: 4` was clamped to 2.0 and answered
    with byte-identical audio to `speed: 2` and no hint on the wire, and
    `instructions` was absorbed the same way. A client had no way to
    distinguish a parameter that worked from one that was dropped.

    A header rather than a field in the body because the body is audio, and
    rather than a 400 because every one of these is either a model limit the
    caller cannot act on or a field OpenAI itself ignores.

    `model` is in the list for the same reason `instructions` is. There is one
    model in this image and `tts-1` cannot be rejected — every OpenAI client
    sends a name, and refusing them is refusing the compatibility this route
    exists for — but a request that asked for `gpt-4o-mini-tts` and got Kokoro
    was told nothing at all. Named only when it differs from what actually
    synthesised, so a caller that asks for `kokoro` gets no noise for having
    been right.
    """
    headers: dict[str, str] = {}
    ignored = set(req.model_extra or ())
    if req.instructions is not None:
        ignored.add("instructions")
    if req.model != MODEL_NAME:
        ignored.add("model")
    if ignored:
        # Sorted as a whole. It used to sort the unknown fields and then append
        # the known ones, so `stream` came out before `instructions` and the
        # order depended on which kind of field it was.
        headers["X-Ignored-Parameters"] = ", ".join(sorted(ignored))
    if speed != req.speed:
        headers["X-Speed-Clamped"] = f"{req.speed:g} to {speed:g}"
    return headers


def _frame(event: dict[str, object]) -> bytes:
    """One SSE event: a bare `data:` line and the blank line that ends it.

    **The blank line is not optional and its absence is silent.** Fed a stream
    whose last event ended in a single newline, openai-python's SSEDecoder
    dropped that event with no error and no warning — so a `speech.audio.done`
    written without it simply never happens as far as the client is concerned.

    No `event:` name line. The schema models the JSON payload only and gives no
    `event` field, unlike `ErrorEvent` in the same file which models one
    explicitly, and the only verbatim OpenAI audio SSE transcript in the spec —
    the transcription example — uses bare `data:` lines. openai-python ignores
    the name for these types and dispatches on the JSON `type` anyway, so
    emitting one would be harmless and would still be a guess about what
    OpenAI sends. Consumers dispatch on `type`.

    No trailing `data: [DONE]` either. Both SDK decoders tolerate one, nothing
    authoritative says OpenAI emits one for this endpoint, and the terminal
    event is already `speech.audio.done`.
    """
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


class ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that closes its generator when the response ends.

    **This exists because otherwise a client that hangs up leaks an ffmpeg
    process, and leaks it for good.** Measured: abandon an SSE request after
    the first delta and the encoder is still running 200 s later, on a stream
    that would have finished in 29 s — blocked on a pipe read, holding its
    memory, waiting for a stdin that will never close.

    Starlette stops iterating the generator on disconnect, correctly. What it
    does not do is close it, so `encode_stream`'s `except GeneratorExit` — the
    branch that kills the encoder — runs only when the suspended generator is
    finally collected. Proved with an instrumented generator under uvicorn:
    after the client closed the socket, iteration stopped immediately, and
    `GeneratorExit` did not arrive until a `gc.collect()` was forced by hand.
    It is a reference cycle, so refcounting never frees it, and on a service
    that is not allocating hard the cyclic collector may not run for minutes.

    Closing it here makes that deterministic and owes nothing to the garbage
    collector: whichever way the response ended — finished, disconnected, or
    raised — `close()` throws GeneratorExit in at the yield, and the encoder
    is killed before this coroutine returns.

    Safe to call at this point, and only at this point: starlette awaits each
    chunk before sending it, and anyio waits for an in-flight worker thread
    before propagating a cancellation, so by the time `__call__` returns the
    generator is suspended at a yield rather than mid-`next()` — which would
    raise "generator already executing" instead of cleaning anything up.
    """

    def __init__(self, content: Iterator[bytes], *args: object,
                 **kwargs: object) -> None:
        # Kept because `self.body_iterator` is the async wrapper starlette
        # builds around this, and closing that is GC-dependent all over again.
        self._source = content
        super().__init__(content, *args, **kwargs)  # type: ignore[arg-type]

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        try:
            await super().__call__(scope, receive, send)
        finally:
            # In the threadpool because closing kills a process and joins its
            # reader threads, and the event loop should not wait on either.
            # A generator that already finished ignores this.
            await run_in_threadpool(self._source.close)


# Deliberately `def`, not `async def`: synthesis is blocking CPU work, and on
# the event loop it would starve /health during any sustained load.
@app.post("/speak")
def speak(req: SpeakRequest) -> Response:
    synth = state.get("synth")
    if not synth:
        raise HTTPException(503, "model still loading")
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")

    # Accepts the thirteen OpenAI names as well, so a caller that learned them
    # from /v1/audio/speech does not have to unlearn them to use segments.
    voice = resolve_voice(req.voice, synth.voices, DEFAULT_VOICE)  # type: ignore[attr-defined]
    if voice is None:
        raise HTTPException(
            400, f"unknown voice {(req.voice or DEFAULT_VOICE)!r}; see GET /voices")
    language = req.language or DEFAULT_LANG

    # Each segment may name its own voice, resolved the same way and falling
    # back to the request's. Resolved here rather than in the synthesiser so
    # that an unknown name is a 400 naming the segment, not a failure part way
    # through a long synthesis.
    segments: list[tuple[str, float, str]] = []
    for index, segment in enumerate(req.segments or ()):
        seg_voice = resolve_voice(segment.voice, synth.voices, voice)  # type: ignore[attr-defined]
        if seg_voice is None:
            raise HTTPException(
                400,
                f"unknown voice {segment.voice!r} in segment {index}; "
                "see GET /voices")
        segments.append((segment.text, segment.pause_after, seg_voice))

    started = time.monotonic()
    offsets: list[float] = []
    try:
        if segments:
            audio, offsets = synth.speak_segments(segments, language, req.speed)  # type: ignore[attr-defined]
        else:
            audio = synth.speak(req.text or "", voice, language, req.speed)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(500, f"synthesis failed: {exc}") from exc

    compute = time.monotonic() - started
    duration = audio.size / SAMPLE_RATE
    # Handled for the same reason synthesis is: ffmpeg can be missing or can
    # fail, and outside the handler that reached the caller as a 500 with
    # nothing in it to act on.
    try:
        data = encode([audio], req.format)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        raise HTTPException(500, f"encoding to {req.format} failed: {exc}") from exc
    mime = CONTENT_TYPE[req.format]
    log.info("%.1fs audio in %.2fs (%.1fx) voice=%s",
             duration, compute, duration / compute if compute else 0.0, voice)

    # The same two numbers the header carries, so /health and
    # X-Realtime-Factor can never disagree about how fast this machine is.
    rate.observe(duration, compute)
    headers = _headers(duration, compute)
    if offsets:
        # WHERE EACH SEGMENT STARTS, so a client can follow the text as it
        # plays. In a header because the body is audio and there is nowhere
        # else to put it without inventing a second response shape for a
        # route that has clients.
        #
        # Offsets only, not the text: the caller sent the segments and knows
        # them in order, so repeating them would be the request echoed back.
        # Three decimals is a millisecond, and ~7 bytes a segment keeps a
        # hundred-segment reading well inside any header limit.
        headers["X-Segment-Offsets"] = ",".join(f"{o:.3f}" for o in offsets)
    return Response(content=data, media_type=mime, headers=headers)


def _usage(input_tokens: int, samples: int) -> dict[str, int]:
    """The `usage` object `speech.audio.done` is required to carry.

    The schema makes it required with three required integers, and Kokoro has
    no notion of an OpenAI token in either direction. It has espeak phonemes
    and it has audio samples, so any number here is a mapping this service
    chose. The two chosen are the least arbitrary available, and both are the
    model's own units rather than invented ones:

      input_tokens   phonemes as the model's 114-symbol vocabulary counts
                     them, which is literally the tensor it is fed
      output_tokens  25 ms frames. Measured, not assumed: the greatest common
                     divisor of five untrimmed outputs of different lengths is
                     exactly 600 samples at 24 kHz, so 600 is the model's own
                     output granularity

    The event cannot be omitted — a done event without usage violates the
    schema — so the rule is written down here and in the README instead, and
    `X-Audio-Seconds` remains the number to trust for anything that matters.
    """
    output_tokens = round(samples / FRAME_SAMPLES)
    return {"input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _sse_body(synth: Synth, chunks: list[str], voice: str, language: str,
              speed: float, fmt: str, input_tokens: int) -> Iterator[bytes]:
    """The event stream: a delta per encoded piece, then done.

    Genuinely incremental, and that is the whole point: the loop synthesises
    one chunk, encodes it and yields it before touching the next, so the first
    frame leaves after the first chunk rather than after the last. Chunking a
    finished buffer into deltas would produce an identical-looking stream and a
    lie a client would build timing assumptions on.

    Concatenating the base64 of every delta reproduces the buffered body for
    the same request byte for byte — same encoder, same chunks, see
    app/audio_out.py — for every format but wav, where it differs in the two
    length fields a stream cannot know.
    """
    samples = 0
    compute = 0.0

    def audio() -> Iterator[np.ndarray]:
        nonlocal samples, compute
        for phonemes in chunks:
            started = time.monotonic()
            piece = synth.speak_chunk(phonemes, voice, language, speed)
            # THE MODEL'S TIME AND NOTHING ELSE. Timing the whole generator
            # instead would include however long starlette waited for the
            # client to take the last frame, so one reader on a slow link
            # would teach every other client that this machine is slow.
            compute += time.monotonic() - started
            samples += piece.size
            yield piece

    # NO KEEPALIVE COMMENT, AND NO OPENING ONE EITHER, AND BOTH ARE CHOICES.
    #
    # The headers are already on their way before this generator is first
    # pulled -- starlette sends http.response.start and only then iterates --
    # so an opening `: comment` would tell a client nothing it does not have.
    #
    # A periodic one is a different question and the answer is the same for a
    # different reason: this is a synchronous generator, blocked inside the
    # model for the whole of speak_chunk, so it cannot emit anything between
    # chunks without a second thread. What that would buy is a stream that
    # survives a proxy read timeout on one very slow chunk, and the timeout in
    # front of this service is 300 s (GATEWAY_TTS_TIMEOUT) against a chunk of
    # at most CHUNK_PHONEMES phonemes, measured at 6.21 s for the first of
    # seven on a 4096-character input. tts-long keepalives because it queues
    # for minutes before it starts; this service starts at once.
    try:
        for data in encode_stream(audio(), fmt):
            yield _frame({"type": "speech.audio.delta",
                          "audio": base64.b64encode(data).decode("ascii")})
        rate.observe(samples / SAMPLE_RATE, compute)
        yield _frame({"type": "speech.audio.done",
                      "usage": _usage(input_tokens, samples)})
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        # The in-band error channel, and the only one left: 200 and the headers
        # are long gone by the time synthesis fails. openai-python raises
        # APIError(message=data["error"]["message"]) on any frame whose JSON
        # has a top-level `error` key and stops reading, which is exactly the
        # behaviour wanted here.
        log.exception("sse synthesis failed")
        yield _frame({"error": {"message": f"synthesis failed: {exc}",
                                "type": "server_error",
                                "param": None,
                                "code": "synthesis_failed"}})


# Same reasoning as /speak: blocking work, so a worker thread rather than the
# event loop. A StreamingResponse handed a sync generator is iterated in that
# same pool, so the SSE path does not put synthesis on the loop either.
@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest) -> Response:
    synth = state.get("synth")
    if not synth:
        return error_response(503, "model still loading",
                              type_="server_error", code="model_loading")

    voice = resolve_voice(req.voice, synth.voices, DEFAULT_VOICE)  # type: ignore[attr-defined]
    if voice is None:
        # Rejected rather than accepted as a custom voice id. The schema allows
        # an arbitrary string because OpenAI has custom voices; this service
        # has 54 fixed ones and nothing to map an unknown id onto, so saying so
        # beats synthesising in a voice nobody asked for. All thirteen of the
        # published names are accepted first, which is the part that was
        # missing: seven of them used to be a 400.
        return error_response(
            400,
            f"Unknown voice {(req.voice or DEFAULT_VOICE)!r}. Accepted: "
            f"{', '.join(sorted(VOICE_ALIASES))}, or any Kokoro voice from "
            "GET /voices.",
            code="invalid_value", param="voice")

    # OpenAI's schema allows 0.25 to 4.0. kokoro_onnx.Kokoro.create carries a
    # hard `assert speed >= 0.5 and speed <= 2.0` before it will run, so the
    # clamp is what stops that assertion becoming a 500; the full range is only
    # reachable by time-stretching afterwards, which without a phase vocoder
    # shifts pitch and sounds worse than the clamp. Clamped rather than
    # rejected because an OpenAI client cannot be told to send something else.
    #
    # What has changed is that it no longer happens in silence: X-Speed-Clamped
    # names the value asked for and the value used. `speed: 4` and `speed: 2`
    # returned byte-identical audio and nothing to tell them apart.
    speed = min(max(req.speed, 0.5), 2.0)
    language = language_for_voice(voice, DEFAULT_LANG)
    headers = _deviations(req, speed)
    if headers:
        log.info("ignored or adjusted: %s", "; ".join(
            f"{k}={v}" for k, v in sorted(headers.items())))

    try:
        chunks = synth.plan(req.input, language, CHUNK_PHONEMES)  # type: ignore[attr-defined]
        input_tokens = synth.token_count(chunks)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        return error_response(500, f"phonemisation failed: {exc}",
                              type_="server_error", code="synthesis_failed")

    if req.stream_format == "sse":
        # No Content-Length, so uvicorn frames it chunked, which is what the
        # schema declares on this endpoint. X-Accel-Buffering is for a reverse
        # proxy in front: nginx buffers a proxied response by default and would
        # hold every delta until the last, undoing the entire feature without
        # changing a byte of it.
        return ClosingStreamingResponse(
            _sse_body(synth, chunks, voice, language, speed,  # type: ignore[arg-type]
                      req.response_format, input_tokens),
            media_type="text/event-stream",
            headers={**headers,
                     "Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"})

    started = time.monotonic()
    try:
        pieces = [synth.speak_chunk(phonemes, voice, language, speed)  # type: ignore[attr-defined]
                  for phonemes in chunks]
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        return error_response(500, f"synthesis failed: {exc}",
                              type_="server_error", code="synthesis_failed")

    # Handled, not left to FastAPI: an ffmpeg that is missing or fails used to
    # return the plain "Internal Server Error" body, which openai-python shows
    # as a bare status code. Every other error on this route is an envelope
    # and this one has to be as well.
    try:
        data = encode(pieces, req.response_format)
    except Exception as exc:  # noqa: BLE001 - the client needs the reason
        return error_response(
            500, f"encoding to {req.response_format} failed: {exc}",
            type_="server_error", code="encoding_failed")

    compute = time.monotonic() - started
    duration = sum(piece.size for piece in pieces) / SAMPLE_RATE
    log.info("%.1fs audio in %.2fs (%.1fx) voice=%s openai",
             duration, compute, duration / compute if compute else 0.0, voice)
    rate.observe(duration, compute)

    # The buffered body keeps its Content-Length and its realtime factor. Both
    # are more use to a client than chunked framing would be — voice-gateway
    # logs X-Realtime-Factor per request and reads it from the header — and a
    # caller that wants bytes as they are made has `stream_format: "sse"` to
    # ask for them by name.
    return Response(content=data, media_type=CONTENT_TYPE[req.response_format],
                    headers={**_headers(duration, compute), **headers})
