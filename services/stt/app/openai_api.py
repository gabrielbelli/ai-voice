"""OpenAI-compatible transcription and translation, alongside the native route.

Anything that already speaks to OpenAI's /v1/audio/transcriptions — the
openai-python client, Open WebUI, a dictation app with a base-URL field —
works against this service unchanged. That is the whole reason this module
exists.

What such a client cannot see is everything /transcribe returns and this
specification has no field for: `realtime_factor`, the `raw` transcript before
glossary repair, and `repaired`, the list of terms that were rewritten. That
last one matters most — a silent substitution is worse than no substitution.
Prefer /transcribe wherever you control the client.

THE RULE THIS MODULE IS BUILT AROUND
------------------------------------
Every field in the specification is either honoured or refused by name. None is
accepted and dropped. That is not a style preference: a dropped field is a
client that believes something false about the transcript it just received, and
this surface used to drop eleven of them.

  timestamp_granularities[]  now honoured — words and segments both
  chunking_strategy          now honoured — it tunes the VAD this service
                             already runs, which was the sharpest of the drops:
                             the client's settings looked like they had landed
                             on a service that visibly does VAD, and had not
  include[]=logprobs         honoured on the engine that reports per-token
                             logprobs, refused by name on the one that does not
  stream                     honoured on the engine that can genuinely emit
                             before it finishes, refused by name on the one
                             that cannot — see the streaming note below
  language, prompt,          honoured on Whisper, refused by name on Parakeet,
  keywords[], temperature    which has no mechanism for any of them
  languages[], diarisation   refused by name; nothing here can do them
  unknown fields             refused by name. CreateTranscriptionRequest sets
                             additionalProperties: false, and lenience here is
                             the mechanism by which every field above was
                             silently swallowed rather than surfaced

`model` is the one exception, and it is a deliberate one. It is required, as
the specification requires it, and its value cannot choose an engine: Parakeet
needs 1.4 GB resident and Whisper large-v3 2.9 GB, and holding both does not
fit the memory this is deployed under. Refusing `whisper-1` on a Parakeet
deployment would reject every existing client — Open WebUI sends it, this
repository's own README sends it — to make a point about a name. So the request
is answered, and every response on this surface carries `x-stt-engine` naming
the engine that actually ran. Honesty rather than obedience; /health says the
same thing.

STREAMING
---------
faster-whisper yields each segment as CTranslate2 finishes the 30-second window
it belongs to, so a stream of deltas is genuinely incremental there. Measured
with tiny/int8 on four threads: a 297 s clip's first delta left 7.8 s into a
68.6 s transcription — 11% of the way — and the same clip's transcription is
one buffered body otherwise. A 14.2 s clip is inside a single window, so its
first and last deltas arrive together at 2.6 s; that is the model's granularity
showing through, not a shortcut here.

Parakeet encodes the whole waveform and then runs a decode loop that emits
nothing until it ends: 5.07 s to the first and only output on that same 14.2 s
clip. There is no partial transcript to send, so `stream=true` is refused by
name under it. Chunking a finished transcript into timed fake deltas would be a
lie a client builds timing assumptions on, and the specification itself notes
that streaming is ignored for whisper-1, so a refusal has precedent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import asr, languages, pipeline
from .errors import (
    CODE_INVALID,
    CODE_MISSING,
    CODE_UNKNOWN_PARAM,
    CODE_UNSUPPORTED_PARAM,
    CODE_UNSUPPORTED_VALUE,
    ApiError,
)

log = logging.getLogger("stt-stack.openai")

FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
TRANSLATION_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
GRANULARITIES = ("word", "segment")

# Fields CreateTranscriptionRequest defines. Anything else is refused, because
# the schema sets additionalProperties: false and because leniency here is what
# turned every unhonoured field into silence.
TRANSCRIPTION_FIELDS = frozenset({
    "file", "model", "language", "languages", "keywords", "prompt",
    "response_format", "temperature", "include", "timestamp_granularities",
    "stream", "chunking_strategy", "known_speaker_names",
    "known_speaker_references",
})
TRANSLATION_FIELDS = frozenset({
    "file", "model", "prompt", "response_format", "temperature",
})

# No auth dependency: the key check is voice_common's ASGI middleware, applied
# to the whole app in main.py. A dependency has to be remembered on every
# route added from here on; middleware cannot be forgotten.
router = APIRouter(prefix="/v1")


# ── the wire, read by hand ────────────────────────────────────────────────────
#
# Every field is parsed from the raw form rather than declared as a FastAPI
# Form parameter, for three reasons that all bite this endpoint in particular:
# the reference client sends arrays as `timestamp_granularities[]` and objects
# as `chunking_strategy[type]`, which no plain declaration matches; an unknown
# field has to be *seen* to be refused, and FastAPI drops it before the handler
# runs; and a type error has to name the field, which pydantic's default body
# does not. The schema the two routes advertise is written out below instead.


def _keys(form) -> set[str]:  # noqa: ANN001 - starlette FormData
    """Field names as sent, with `[]` and `[child]` reduced to the parent."""
    names = set()
    for key in form:
        names.add(key.split("[", 1)[0] if "[" in key else key)
    return names


def _values(form, name: str) -> list[str]:  # noqa: ANN001
    """Every value sent for an array field, bracketed spelling or bare.

    openai-python serialises multipart arrays with array_format="brackets", so
    it sends `timestamp_granularities[]` once per value. curl users send the
    bare name. Both are read, because refusing one of them would be a parity
    gap of its own.
    """
    out: list[str] = []
    for key in (f"{name}[]", name):
        out.extend(str(value) for value in form.getlist(key))
    return out


def _value(form, name: str) -> str | None:  # noqa: ANN001
    raw = form.get(name)
    if raw is None:
        return None
    return str(raw)


def _bad(message: str, *, param: str | None = None,
         code: str = CODE_INVALID) -> ApiError:
    return ApiError(400, message, code=code, param=param)


def _unsupported(param: str, why: str) -> ApiError:
    """A field this deployment cannot honour, refused in the client's words."""
    return ApiError(400, f"Unsupported parameter: '{param}' {why}",
                    code=CODE_UNSUPPORTED_PARAM, param=param)


def _reject_unknown(form, allowed: frozenset[str]) -> None:  # noqa: ANN001
    for name in sorted(_keys(form) - allowed):
        raise _bad(f"Unrecognized request argument supplied: {name}",
                   param=name, code=CODE_UNKNOWN_PARAM)


def _model(form) -> str:  # noqa: ANN001
    """`model` is required by the specification and was optional here.

    It does not choose an engine — see the module docstring — but a server that
    accepts its absence accepts a request the real API rejects, and a client
    written against that difference breaks on the way back.
    """
    value = (_value(form, "model") or "").strip()
    if not value:
        raise _bad("Missing required parameter: 'model'.", param="model",
                   code=CODE_MISSING)
    return value


def _response_format(form, allowed: tuple[str, ...]) -> str:  # noqa: ANN001
    value = (_value(form, "response_format") or "json").strip()
    if value == "diarized_json":
        raise ApiError(
            400,
            "Unsupported value: 'response_format' does not support "
            "'diarized_json'. This service has no speaker-embedding or "
            "clustering component and neither engine produces speaker labels, "
            "so there is nothing to annotate segments with.",
            code=CODE_UNSUPPORTED_VALUE, param="response_format")
    if value not in allowed:
        raise _bad(
            f"Unsupported value: 'response_format' does not support '{value}'. "
            f"Supported values are: {', '.join(repr(f) for f in allowed)}.",
            param="response_format", code=CODE_UNSUPPORTED_VALUE)
    return value


def _float(form, name: str) -> float | None:  # noqa: ANN001
    raw = _value(form, name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise _bad(f"'{name}' must be a number, got {raw!r}.",
                   param=name) from exc


def _temperature(form, engine) -> float | None:  # noqa: ANN001
    """Absent and 0 are different requests, so the default is None.

    A pinned temperature disables Whisper's [0.0 … 1.0] fallback ladder, which
    is the retry on low-confidence output. Honouring the field is therefore a
    real reliability trade, and one only a client that named it has asked for.
    """
    value = _float(form, "temperature")
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise _bad("'temperature' must be between 0 and 1.", param="temperature")
    if not engine.accepts_temperature:
        raise _unsupported(
            "temperature",
            f"is not supported by the '{engine.name}' engine: a TDT decoder "
            "has no sampling temperature. Omit it, or deploy with "
            "STT_MODEL=whisper.")
    return value


def _language(form, engine) -> str | None:  # noqa: ANN001
    value = (_value(form, "language") or "").strip()
    if not value:
        return None
    if not engine.accepts_language:
        raise _unsupported(
            "language",
            f"is not supported by the '{engine.name}' engine, which detects "
            "the language itself and takes no hint. It used to be accepted "
            "and echoed back in verbose_json as if it had steered the "
            "decode, which was a claim about the output rather than a "
            "setting. Omit it, or deploy with STT_MODEL=whisper.")
    if not languages.known(value):
        raise _bad(
            f"'language' must be an ISO-639-1 code, got {value!r}.",
            param="language")
    return value


def _vocabulary(form, engine) -> str | None:  # noqa: ANN001
    """`prompt` and `keywords[]`, which reach the same decoder argument.

    Both are vocabulary biasing, and this service has a vocabulary-biasing
    mechanism — on one engine. Whisper takes hotwords at decode time and
    measurably benefits; Parakeet's TDT decoder has no such argument, and
    onnx-asr exposes none, so the field is refused there rather than accepted
    and ignored. Post-decode glossary repair still runs on both, and cannot
    recover a word the acoustic model never approached.
    """
    prompt = (_value(form, "prompt") or "").strip()
    keywords = [k.strip() for k in _values(form, "keywords") if k.strip()]
    if not prompt and not keywords:
        return None

    param = "prompt" if prompt else "keywords"
    if not engine.accepts_vocabulary:
        raise _unsupported(
            param,
            f"is not supported by the '{engine.name}' engine: its TDT decoder "
            "takes no vocabulary at decode time and onnx-asr exposes no "
            "biasing argument for it. Deploy with STT_MODEL=whisper to steer "
            "the decoder, or rely on the glossary's post-decode repair.")
    if not pipeline.HOTWORDS_ENABLED:
        raise _unsupported(
            param,
            "cannot be honoured: this deployment runs with STT_HOTWORDS=0, "
            "which switches decode-time biasing off entirely so that a "
            "benchmark measures the model rather than the vocabulary.")
    return ", ".join([prompt, *keywords]) if prompt else ", ".join(keywords)


def _granularities(form, response_format: str) -> tuple[str, ...]:  # noqa: ANN001
    values = [v.strip() for v in _values(form, "timestamp_granularities") if v.strip()]
    if not values:
        # The specification's default, and the reason `segments` appears on a
        # verbose_json body nobody asked a question about.
        return ("segment",) if response_format == "verbose_json" else ()
    for value in values:
        if value not in GRANULARITIES:
            raise _bad(
                f"Unsupported value: 'timestamp_granularities' does not "
                f"support '{value}'. Supported values are: 'word', 'segment'.",
                param="timestamp_granularities", code=CODE_UNSUPPORTED_VALUE)
    if response_format != "verbose_json":
        raise _bad(
            "'timestamp_granularities' requires response_format=verbose_json.",
            param="timestamp_granularities")
    return tuple(dict.fromkeys(values))


def _include(form, engine, response_format: str) -> bool:  # noqa: ANN001
    values = [v.strip() for v in _values(form, "include") if v.strip()]
    if not values:
        return False
    for value in values:
        if value != "logprobs":
            raise _bad(
                f"Unsupported value: 'include' does not support '{value}'. "
                "Supported values are: 'logprobs'.",
                param="include", code=CODE_UNSUPPORTED_VALUE)
    if response_format != "json":
        raise _bad("'include[]=logprobs' requires response_format=json.",
                   param="include")
    if not engine.reports_token_logprobs:
        raise _unsupported(
            "include",
            f"cannot be honoured by the '{engine.name}' engine: faster-whisper "
            "reports one average logprob per segment and one probability per "
            "word, but no per-token logprob, and returning a differently "
            "shaped number under the same name would be worse than refusing.")
    return True


def _stream(form, engine, response_format: str) -> bool:  # noqa: ANN001
    raw = (_value(form, "stream") or "").strip().lower()
    if raw in {"", "false", "0", "none", "null"}:
        return False
    if raw not in {"true", "1"}:
        raise _bad(f"'stream' must be a boolean, got {raw!r}.", param="stream")
    if not engine.can_stream:
        raise _unsupported(
            "stream",
            f"is not supported by the '{engine.name}' engine: it encodes the "
            "whole waveform and then runs a decode loop that emits nothing "
            "until it ends — measured 5.07 s to the first and only output on "
            "a 14.2 s clip — so there is no partial transcript to send. "
            "Cutting a finished transcript into timed deltas would be a lie "
            "about latency. Deploy with STT_MODEL=whisper to stream.")
    if response_format != "json":
        raise _bad(
            "'stream' requires response_format=json: the stream carries "
            "transcript text events, which the subtitle and verbose formats "
            "have no representation for.",
            param="stream")
    return True


def _chunking(form) -> pipeline.Tuning:  # noqa: ANN001
    """chunking_strategy, honoured against the VAD this service already runs.

    The three server_vad knobs map one-to-one onto Silero's, which is why this
    is honourable at all: threshold is the same number, prefix_padding_ms is
    the pad before a speech run, silence_duration_ms is how much silence ends
    one. The defaults stay this service's own (0.5 / 100 ms / 300 ms), not the
    specification's, because those are what every measurement in the README
    was taken with.
    """
    strategy = _value(form, "chunking_strategy")
    children = {key.split("[", 1)[1].rstrip("]"): str(value)
                for key, value in form.multi_items()
                if key.startswith("chunking_strategy[")}
    if strategy is None and not children:
        return pipeline.Tuning()

    if not pipeline.VAD_ENABLED:
        raise _unsupported(
            "chunking_strategy",
            "cannot be honoured: this deployment runs with STT_VAD=0, so "
            "there is no voice activity detection to configure.")

    kind = (children.get("type") or strategy or "auto").strip()
    if kind == "auto":
        return pipeline.Tuning()
    if kind != "server_vad":
        raise _bad(
            f"Unsupported value: 'chunking_strategy' does not support "
            f"'{kind}'. Supported values are: 'auto', 'server_vad'.",
            param="chunking_strategy", code=CODE_UNSUPPORTED_VALUE)

    for name in children:
        if name not in {"type", "threshold", "prefix_padding_ms",
                        "silence_duration_ms"}:
            raise _bad(
                f"Unrecognized request argument supplied: "
                f"chunking_strategy[{name}]",
                param=f"chunking_strategy[{name}]", code=CODE_UNKNOWN_PARAM)

    def number(name: str, low: float, high: float) -> float | None:
        raw = children.get(name)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise _bad(f"'chunking_strategy[{name}]' must be a number, "
                       f"got {raw!r}.",
                       param=f"chunking_strategy[{name}]") from exc
        if not low <= value <= high:
            raise _bad(f"'chunking_strategy[{name}]' must be between "
                       f"{low:g} and {high:g}.",
                       param=f"chunking_strategy[{name}]")
        return value

    threshold = number("threshold", 0.0, 1.0)
    padding = number("prefix_padding_ms", 0.0, 5_000.0)
    silence = number("silence_duration_ms", 0.0, 30_000.0)
    return pipeline.Tuning(
        threshold=threshold,
        min_silence_ms=None if silence is None else int(silence),
        speech_pad_ms=None if padding is None else int(padding),
    )


def _reject_diarisation(form) -> None:  # noqa: ANN001
    for name in ("known_speaker_names", "known_speaker_references"):
        if _values(form, name):
            raise _unsupported(
                name,
                "is not supported: this service has no speaker-embedding or "
                "clustering component, and neither engine produces speaker "
                "labels.")


def _reject_languages(form) -> None:  # noqa: ANN001
    if _values(form, "languages"):
        raise _unsupported(
            "languages",
            "is not supported: neither engine takes a set of candidate "
            "languages. Send 'language' with a single ISO-639-1 code on a "
            "Whisper deployment, which is a hint the decoder can act on.")


# ── the bodies ────────────────────────────────────────────────────────────────


def _clock(seconds: float, decimal: str) -> str:
    """HH:MM:SS with the fraction separator each subtitle format insists on.

    Computed in whole milliseconds because 4.1 seconds is not 4.1 in binary:
    formatting the fractional part directly renders it as 04,099.
    """
    total = int(max(seconds, 0.0) * 1000 + 0.5)
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def _cues(result: pipeline.Result) -> list[tuple[float, float, str]]:
    """One cue per utterance, or one cue for the clip if there is nothing else.

    This used to be a single cue spanning the whole recording, which made a
    five-minute file into a five-minute cue: true, and unusable as a subtitle
    track. Both engines can do better — Whisper reports segments, Parakeet
    reports token timings the VAD's own boundaries cut into utterances.
    """
    cues = [(s.start, s.end, s.text.strip()) for s in result.segments
            if s.text.strip()]
    if cues:
        return cues
    return [(0.0, result.audio_seconds, result.text)] if result.text else []


def _srt(result: pipeline.Result) -> str:
    # SubRip terminates every block with a blank line, the last one included.
    # The body used to end `day.\n`, which most parsers tolerate on a
    # single-cue file and none tolerate once there is more than one.
    return "".join(
        f"{index}\n{_clock(start, ',')} --> {_clock(end, ',')}\n{text}\n\n"
        for index, (start, end, text) in enumerate(_cues(result), start=1)
    )


def _vtt(result: pipeline.Result) -> str:
    cues = "".join(
        f"{_clock(start, '.')} --> {_clock(end, '.')}\n{text}\n\n"
        for start, end, text in _cues(result)
    )
    return "WEBVTT\n\n" + cues


def _usage(result: pipeline.Result) -> dict[str, Any]:
    """The duration variant, which costs nothing: the number is already here.

    Not the token variant — neither engine is billed by tokens and neither
    reports an input token count, so `input_tokens` would be invented.
    """
    return {"type": "duration", "seconds": round(result.audio_seconds)}


def _word_json(word: asr.Word) -> dict[str, Any]:
    return {"word": word.word, "start": round(word.start, 2),
            "end": round(word.end, 2)}


def _segment_json(segment: asr.Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "seek": segment.seek,
        "start": round(segment.start, 2),
        "end": round(segment.end, 2),
        "text": segment.text,
        "tokens": list(segment.tokens),
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob,
    }


def _body(result: pipeline.Result, response_format: str,
          granularities: tuple[str, ...], want_logprobs: bool) -> Response:
    if response_format == "text":
        # The specification's text format is the bare transcript. This used to
        # append a newline: harmless to the SDK, visible to anything diffing.
        return PlainTextResponse(result.text)

    if response_format == "srt":
        return PlainTextResponse(_srt(result))

    if response_format == "vtt":
        # text/vtt is what the format is, and what a browser needs to see
        # before it will treat the body as a track. The specification declares
        # no content type for it; text/plain was a convention, not a rule.
        return PlainTextResponse(_vtt(result), media_type="text/vtt; charset=utf-8")

    if response_format == "verbose_json":
        body: dict[str, Any] = {
            "task": result.task,
            # The language of the INPUT audio, as the field is defined, in the
            # spelling the specification's own example uses. It used to echo
            # the request, so language=pt on English audio came back claiming
            # "pt" — a fabricated assertion about a decode it had not steered.
            "language": languages.name(result.language),
            "duration": result.audio_seconds,
            "text": result.text,
        }
        if "segment" in granularities:
            body["segments"] = [_segment_json(s) for s in result.segments]
        if "word" in granularities:
            body["words"] = [_word_json(w) for w in result.words]
        body["usage"] = _usage(result)
        return JSONResponse(body)

    body = {"text": result.text}
    if want_logprobs and result.logprobs is not None:
        body["logprobs"] = [
            {"token": entry.token, "logprob": entry.logprob,
             "bytes": list(entry.bytes)}
            for entry in result.logprobs
        ]
    body["usage"] = _usage(result)
    return JSONResponse(body)


# ── server-sent events ────────────────────────────────────────────────────────
#
# Framing rules, all verified against openai-python 3.6.0's SSEDecoder:
#
#   * every event ends with a BLANK LINE, the last one included. A final event
#     terminated by a single \n is silently DROPPED — no error, no warning.
#   * bare `data:` frames, no `event:` name line. The schema models the JSON
#     payload only and gives no event field, the one verbatim OpenAI audio SSE
#     transcript in the specification uses bare data lines, and the client
#     dispatches on the JSON `type` either way.
#   * no `[DONE]` sentinel. Nothing authoritative says OpenAI emits one for
#     this endpoint; both SDK decoders tolerate its absence.
#   * a top-level `error` key in any event makes openai-python raise APIError
#     and stop. That is the in-band error channel once 200 has gone out.

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx buffers proxied responses by default, which would hold the deltas
    # back until the end and undo the whole point.
    "X-Accel-Buffering": "no",
}


def _event(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _sse(stream: pipeline.Stream, release) -> Iterator[bytes]:  # noqa: ANN001
    try:
        for delta in stream.deltas:
            yield _event({"type": "transcript.text.delta", "delta": delta})
        # `usage` is optional on this event and omitted deliberately: the
        # schema pins it to the token variant, and neither engine reports an
        # input token count to put in it.
        yield _event({"type": "transcript.text.done", "text": stream.text})
    except Exception as exc:  # noqa: BLE001 - 200 has gone; this is the only channel
        log.exception("stream failed after %d bytes", len(stream.text))
        yield _event({"error": {
            "message": f"transcription failed: {exc}",
            "type": "server_error", "param": None, "code": None}})
    finally:
        release()


# ── routes ────────────────────────────────────────────────────────────────────

_MULTIPART = "multipart/form-data"


def _schema(fields: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """The request body, written out because this module parses by hand.

    /docs is otherwise reduced to "file" and nothing else, which would make the
    schema dump a worse description of the surface than the code.
    """
    return {"requestBody": {"required": True, "content": {
        _MULTIPART: {"schema": {"type": "object", "properties": fields,
                                "required": required,
                                "additionalProperties": False}}}}}


_TRANSCRIPTION_SCHEMA = _schema({
    "file": {"type": "string", "format": "binary"},
    "model": {"type": "string"},
    "language": {"type": "string"},
    "languages": {"type": "array", "items": {"type": "string"}},
    "prompt": {"type": "string"},
    "keywords": {"type": "array", "items": {"type": "string"}},
    "response_format": {"type": "string", "enum": list(FORMATS)},
    "temperature": {"type": "number", "minimum": 0, "maximum": 1},
    "include": {"type": "array", "items": {"type": "string", "enum": ["logprobs"]}},
    "timestamp_granularities": {"type": "array",
                                "items": {"type": "string",
                                          "enum": list(GRANULARITIES)}},
    "stream": {"type": "boolean"},
    "chunking_strategy": {"type": "string"},
    "known_speaker_names": {"type": "array", "items": {"type": "string"}},
    "known_speaker_references": {"type": "array", "items": {"type": "string"}},
}, ["file", "model"])

_TRANSLATION_SCHEMA = _schema({
    "file": {"type": "string", "format": "binary"},
    "model": {"type": "string"},
    "prompt": {"type": "string"},
    "response_format": {"type": "string", "enum": list(TRANSLATION_FORMATS)},
    "temperature": {"type": "number", "minimum": 0, "maximum": 1},
}, ["file", "model"])


def _headers(engine) -> dict[str, str]:  # noqa: ANN001
    """Which engine actually ran. See the module docstring on `model`."""
    return {"x-stt-engine": engine.name}


def _translate_pipeline_error(exc: HTTPException) -> ApiError:
    """Only this side re-shapes errors.

    The native route's bodies are part of a contract that already has clients,
    so they stay exactly as FastAPI renders them.
    """
    param = "file" if exc.status_code == 400 else None
    return ApiError(
        exc.status_code, str(exc.detail),
        type_="invalid_request_error" if exc.status_code < 500 else "server_error",
        code=CODE_INVALID if exc.status_code == 400 else None,
        param=param)


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise _bad("'file' is empty.", param="file")
    return data


@router.post("/audio/transcriptions", openapi_extra=_TRANSCRIPTION_SCHEMA)
async def transcriptions(request: Request,
                         file: UploadFile = File(...)) -> Response:
    engine = pipeline.engine()
    form = await request.form()

    _reject_unknown(form, TRANSCRIPTION_FIELDS)
    _model(form)
    response_format = _response_format(form, FORMATS)
    _reject_diarisation(form)
    _reject_languages(form)
    streaming = _stream(form, engine, response_format)
    granularities = _granularities(form, response_format)
    want_logprobs = _include(form, engine, response_format)
    tuning = _chunking(form)

    want_segments = "segment" in granularities or response_format in {"srt", "vtt"}
    opts = asr.Options(
        language=_language(form, engine),
        hotwords=_vocabulary(form, engine),
        temperature=_temperature(form, engine),
        task="transcribe",
        # The engine that reports no segments of its own has them cut from its
        # word timings, so a subtitle needs words there and not on the other.
        want_words=("word" in granularities
                    or (want_segments and not engine.reports_segments)),
        want_segments=want_segments,
        want_logprobs=want_logprobs,
    )
    data = await _read(file)

    if streaming:
        return await _stream_response(data, opts, tuning, engine)
    return await _run(data, opts, tuning, engine, response_format,
                      granularities, want_logprobs)


@router.post("/audio/translations", openapi_extra=_TRANSLATION_SCHEMA)
async def translations(request: Request,
                       file: UploadFile = File(...)) -> Response:
    """Speech in any language, English text out.

    Implemented on the engine that genuinely has the task — faster-whisper
    takes task="translate" — and refused by name on the one that does not.
    Parakeet has no translate task and no target-language conditioning, so
    there is nothing to route the request to; silently transcribing instead
    would answer a different question in a shape that looks like an answer to
    this one.
    """
    engine = pipeline.engine()
    if not engine.can_translate:
        raise ApiError(
            400,
            f"Unsupported value: translation requires an engine with a "
            f"translate task, and this deployment loaded '{engine.name}', "
            "which has none — it has no target-language conditioning either, "
            "so there is nothing to translate with. Deploy with "
            "STT_MODEL=whisper, or use /v1/audio/transcriptions.",
            code=CODE_UNSUPPORTED_VALUE, param="model")

    form = await request.form()
    _reject_unknown(form, TRANSLATION_FIELDS)
    _model(form)
    response_format = _response_format(form, TRANSLATION_FORMATS)
    granularities = ("segment",) if response_format == "verbose_json" else ()

    opts = asr.Options(
        hotwords=_vocabulary(form, engine),
        temperature=_temperature(form, engine),
        task="translate",
        want_segments=response_format in {"verbose_json", "srt", "vtt"},
    )
    data = await _read(file)
    return await _run(data, opts, tuning=pipeline.Tuning(), engine=engine,
                      response_format=response_format,
                      granularities=granularities, want_logprobs=False)


async def _run(data: bytes, opts: asr.Options, tuning: pipeline.Tuning,
               engine, response_format: str, granularities: tuple[str, ...],  # noqa: ANN001
               want_logprobs: bool) -> Response:
    try:
        with pipeline.slot():
            # Blocking CPU work, kept off the event loop: declared inline it
            # would starve /health until the transcription finished, and a
            # container healthcheck that times out restarts a service that is
            # working correctly.
            result = await run_in_threadpool(
                pipeline.run, data, opts, allow_resample=True, tuning=tuning)
    except pipeline.Busy as exc:
        raise _busy() from exc
    except HTTPException as exc:
        raise _translate_pipeline_error(exc) from exc

    response = _body(result, response_format, granularities, want_logprobs)
    response.headers.update(_headers(engine))
    return response


async def _stream_response(data: bytes, opts: asr.Options,
                           tuning: pipeline.Tuning, engine) -> Response:  # noqa: ANN001
    """Open the stream before returning, so a failure is still a real status.

    Decoding and the VAD pass happen here rather than inside the generator:
    both can fail on the client's input, and a 400 that arrives as an event
    inside a 200 is a worse answer than a 400.
    """
    try:
        pipeline.acquire()
    except pipeline.Busy as exc:
        raise _busy() from exc
    try:
        stream = await run_in_threadpool(
            pipeline.open_stream, data, opts, allow_resample=True, tuning=tuning)
    except HTTPException as exc:
        pipeline.release()
        raise _translate_pipeline_error(exc) from exc
    except Exception:
        pipeline.release()
        raise

    return StreamingResponse(
        _sse(stream, pipeline.release),
        media_type="text/event-stream; charset=utf-8",
        headers={**SSE_HEADERS, **_headers(engine)},
    )


def _busy() -> ApiError:
    return ApiError(
        429,
        "Rate limit reached: this deployment allows "
        f"{pipeline.MAX_CONCURRENT} transcription(s) at a time. Retry shortly.",
        type_="requests", code="rate_limit_exceeded",
        headers={"Retry-After": "1"})
