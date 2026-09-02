# tts-stack

Self-hosted text-to-speech. Kokoro, CPU only, no torch.

```text
text or segments  →  /speak             native: pauses, realtime factor
text              →  /v1/audio/speech   OpenAI's shape, buffered or SSE
  ↓  espeak-ng    phonemise
  ↓  Kokoro-82M   ONNX Runtime, CPU, a chunk at a time
  ↓  wav  opus  mp3  aac  flac  pcm
```

Sibling of [services/stt](../stt/README.md), same conventions.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 \
  ghcr.io/gabrielbelli/ai-voice-tts:pre
```

First start downloads ~340 MB into the volume. Later starts are immediate.

```bash
curl -X POST localhost:8001/speak -H 'content-type: application/json' \
  -d '{"text":"Here is the change to make.","voice":"bm_george"}' \
  --output out.wav
```

## Segments, and why they matter more than the voice

`/speak` accepts a list of segments with explicit pauses:

```json
{
  "voice": "bm_george",
  "speed": 0.95,
  "segments": [
    {"text": "Three steps.",                       "pause_after": 0.75},
    {"text": "One. Open your config file.",        "pause_after": 0.75},
    {"text": "Two. Set the model to nothing.",     "pause_after": 0.75},
    {"text": "Three. Restart the container.",      "pause_after": 0.45}
  ]
}
```

**The silence is generated here, not asked of the model.** No TTS model
reliably produces a beat you can act inside — punctuation buys a breath, an
instruction needs a gap.

Tested by ear on the same voice and the same words, three ways: flowing prose,
short declaratives, and short declaratives with 0.75 s of inserted silence.
Only the third sounds like instructions. Nothing changed but the writing and
the gaps.

So the model is not the interesting variable. Write for the ear and place the
pauses; any competent voice will then do.

A segment may name its own `voice`, and uses the request's when it does not. A
voice is a 510 KB embedding over weights that are already resident, so
alternating between two costs nothing:

```json
{
  "voice": "bm_george",
  "segments": [
    {"text": "The reviewer asked:",             "pause_after": 0.4},
    {"text": "why is this not in the volume?",  "pause_after": 0.6, "voice": "af_nova"},
    {"text": "Because it is 310 megabytes."}
  ]
}
```

`language` stays a property of the request rather than the segment, so a
segment in another language wants its own request.

**Unknown fields are rejected, not ignored.** `/speak` answers `422` for a
field it does not recognise, at the top level or inside a segment. A misspelt
`pause_after` was previously dropped in silence and heard as a missing pause.
`/v1/audio/speech` is deliberately the other way round — see below.

## Voices

54 voices, all sharing one 310 MB model. **A voice is a 510 KB embedding**, so
switching costs nothing after load — you can use a different voice per
request, or per segment.

```bash
curl -s localhost:8001/voices | python3 -m json.tool
```

| Prefix | Locale |
|---|---|
| `af_` `am_` | en-US female / male |
| `bf_` `bm_` | en-GB female / male |
| `pf_` `pm_` | pt-BR female / male |

Only three are Brazilian Portuguese: `pf_dora`, `pm_alex`, `pm_santa`.

For explanations, the lower and calmer voices work better than the bright
defaults — `bm_george`, `am_onyx`. Slowing slightly (`"speed": 0.95`) reads as
deliberate rather than performed.

## OpenAI-compatible API

`POST /v1/audio/speech` accepts OpenAI's body, so anything already written
against `api.openai.com` reaches this service by changing a base URL.

```bash
curl -X POST localhost:8001/v1/audio/speech \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer sk-your-key' \
  -d '{"model":"tts-1","input":"Here is the change to make.","voice":"fable","response_format":"mp3"}' \
  --output out.mp3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-your-key")

response = client.audio.speech.create(
    model="tts-1",
    voice="fable",
    input="Here is the change to make.",
    response_format="mp3",
)
response.write_to_file("out.mp3")
```

`api_key` is required by the client library even when this service accepts
anything; send any non-empty string when authentication is off.

| Field | Handling |
|---|---|
| `model` | **Accepted and ignored, and the response says so.** There is one model in this image. Rejecting `tts-1` would break every client that sends it, so anything is answered — and named in `X-Ignored-Parameters` unless it is `kokoro`, which is what actually synthesised |
| `input` | Required. **Rejected over 4096 characters**, the schema's own maximum. Empty or whitespace-only is legal — the schema sets no `minLength` — and returns `200` with the empty form of the container asked for, where it used to be a `500`. See [Deviations](#deviations) |
| `instructions` | **Accepted and ignored, and the response says so.** See [Deviations](#deviations); the reasoning is that Kokoro has nothing to route a direction into, and OpenAI documents that `instructions` does not work with `tts-1` either |
| `voice` | All thirteen OpenAI names, the `{"id": "…"}` object form, **or** any of the 54 Kokoro names. Optional, unlike upstream: `TTS_VOICE` answers when it is absent. An id this service does not know is a `400` naming `voice` |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`. `pcm` is headerless 24 kHz 16-bit mono, which is Kokoro's own rate — nothing is resampled |
| `speed` | Accepted across OpenAI's 0.25–4.0 and **clamped to 0.5–2.0**, where `kokoro_onnx` will run at all. Clamped rather than rejected, but no longer in silence: `X-Speed-Clamped` names both values |
| `stream_format` | `audio` (default) or `sse`. Validated against the enum — a misspelling is a `400` naming the field, where it used to return the same audio as a correct value |

Unknown fields are accepted, unlike on `/speak`, because OpenAI keeps adding
them and a service whose entire purpose is to answer OpenAI's clients must not
reject one for speaking a newer dialect. **Accepted is not dropped**, though:
every field that reached no audio is listed in `X-Ignored-Parameters` on the
response, so `{"stream": true}` — the classic confusion with the transcription
schema — no longer returns an ordinary mp3 with nothing to say it did nothing.

Three kinds of field end up in that header, and the rule is the same for all
three: **if it did not reach the audio, it is named.** An unrecognised field
like `stream`; `instructions`, which this model has nothing to route into;
and `model` itself when it asked for something other than `kokoro`. The last
one is the reason the header is worth reading on an otherwise ordinary
request — `{"model": "gpt-4o-mini-tts"}` is answered by an 82M-parameter
model, and a client that never sees that said so has no way to find out.

| Header | When |
|---|---|
| `X-Ignored-Parameters` | Any field that reached no audio, alphabetically: unknown fields, `instructions`, and `model` when it named something other than `kokoro` |
| `X-Speed-Clamped` | `4 to 2`, when the requested speed was outside Kokoro's range |

Errors come back in OpenAI's envelope, all four keys, `param` included:

```json
{"error": {"message": "Unknown voice 'voice_1234'. Accepted: alloy, ash, …",
           "type": "invalid_request_error",
           "param": "voice",
           "code": "invalid_value"}}
```

`param` is required-but-nullable in the schema and used to be absent entirely,
which made every error this service emitted schema-invalid — while the field
name was known and thrown away. It is now present as an explicit `null` when no
one field is at fault, and names the field when one is.

`404` and `405` under `/v1` are in the envelope too, `{"error": …}` rather than
FastAPI's `{"detail": …}`, with the same `Invalid URL (GET /v1/audio/speech)`
wording `api.openai.com` uses. The native routes keep FastAPI's shape: clients
are already written against it. That covers a failure to encode as well as a
failure to synthesise — `ffmpeg` missing or failing is a `500` with a message on
both routes, in each route's own shape.

### Streaming

`stream_format: "sse"` returns server-sent events, and they are generated as
they are sent rather than sliced off a finished file.

```python
import base64, json, httpx

with httpx.stream("POST", "http://localhost:8001/v1/audio/speech", json={
        "model": "tts-1", "voice": "fable", "input": text,
        "response_format": "mp3", "stream_format": "sse"}) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "speech.audio.delta":
                player.write(base64.b64decode(event["audio"]))
```

On the wire, verbatim, for a two-sentence input:

```text
data: {"type":"speech.audio.delta","audio":"SUQzBAAAAAAAIlRTU0UAAAAOAA…"}

data: {"type":"speech.audio.delta","audio":"//OExCIuy+KRstvK3Z3WQk334S…"}

data: {"type":"speech.audio.done","usage":{"input_tokens":53,"output_tokens":147,"total_tokens":200}}

```

**It is genuinely incremental, and here is the measurement.** The schema's own
4096-character maximum, `bm_fable`, mp3, on an M2 Max: 188.4 s of speech,
1 507 820 bytes. Median of three runs each way.

```text
buffered   first byte at 58.98 s, complete at 58.98 s
streamed   first byte at  5.49 s, complete at 55.06 s   11 deltas
```

The first delta arrives **10% into the generation, 10.7x sooner** than the
buffered body's first byte, and the stream finishes no later than the buffered
response does. The remaining deltas land every 4 to 6 seconds, each carrying
roughly 17 s of audio — the box generates at about 3x realtime, so a player
started on the first delta never catches up with the encoder.

Before this, the same request returned one buffered blob after the whole
utterance was synthesised and encoded, with zero bytes reaching the client until
the end. `stream_format: "sse"` was accepted and dropped: HTTP 200,
`content-type: audio/mpeg`, no frames, no error. And because `openai-python`
casts the result to `HttpxBinaryResponseContent` for both stream formats and
never inspects the content type, an SDK-based test could not see it — the client
wrote the mp3 to disk and reported success.

**Concatenating the base64 of every delta reproduces the buffered body byte for
byte.** That is not a coincidence and not a promise: both stream formats come
out of one encoder in `app/audio_out.py`, drained for a buffered response and
framed for a streamed one, so the two cannot drift. `wav` and `opus` are the two
measured exceptions and both are in [Deviations](#deviations).

Three framing decisions the schema does not settle, and what this service does:

| Question | Here | Why |
|---|---|---|
| `event:` name lines | No, bare `data:` frames | The schema models the JSON payload only and gives no `event` field, unlike `ErrorEvent` in the same file which models one explicitly. The only verbatim OpenAI audio SSE transcript in the spec — the transcription example — uses bare `data:` lines. `openai-python` ignores the name for these types and dispatches on the JSON `type`, so sending one would be harmless and still a guess |
| trailing `data: [DONE]` | No | Both SDK decoders tolerate one; nothing authoritative says OpenAI emits one for this endpoint, and `speech.audio.done` is already terminal |
| chunk size | One model pass, `TTS_CHUNK_PHONEMES` | See [Configuration](#configuration) |

Every event ends in a blank line, the last one included. That is not a stylistic
choice: fed a stream whose final event ended in a single newline,
`openai-python`'s `SSEDecoder` **dropped that event silently** — no error, no
warning, and a client that never sees `speech.audio.done`.

If a synthesis fails after the headers have gone out, the stream carries an
event with a top-level `error` key, which is the one in-band channel a 200 has
left. `openai-python` raises `APIError` on it and stops reading.

**A client that hangs up takes its encoder with it.** Starlette stops iterating
a generator when the socket closes but never closes the generator, so the
`GeneratorExit` that kills ffmpeg used to wait on the cyclic collector — and on
a service that is not allocating hard, that wait has no upper bound. Measured:
an SSE request abandoned after its first delta still had a live ffmpeg 200 s
later, blocked on a stdin that would never close, on a stream that would have
finished in 29 s. The response now closes its own generator as it ends, however
it ends; the same request cleans up in 9.3 to 12.4 s, which is the chunk already
inside the model finishing. That last part is not removable without cancelling
a running ONNX call, which the runtime does not offer.

`Transfer-Encoding: chunked` and `X-Accel-Buffering: no` go out with the stream.
The second is for a reverse proxy in front: nginx buffers a proxied response by
default and would hold every delta until the last, undoing the whole feature
without changing a byte of it.

The buffered path keeps its `Content-Length` and its `X-Realtime-Factor`, which
are more use to a client writing a file than chunked framing would be. A caller
that wants bytes as they are made asks for them by name.

### Which route to prefer

**`/speak`, unless a client you do not control is asking for the other one.**

| | `/speak` | `/v1/audio/speech` |
|---|---|---|
| Segments with pauses | yes | no field for it |
| Per-segment voice | yes | no |
| Unknown fields | `422` | accepted, and named in `X-Ignored-Parameters` |
| `language` override | yes | inferred from the voice |
| `X-Realtime-Factor` | yes | sent, but clients ignore it |
| Speed range | 0.5–2.0, rejected outside | 0.25–4.0, clamped, clamp announced |
| Incremental delivery | no | `stream_format: "sse"` |

The OpenAI body cannot express a pause, and pauses are the thing that makes
audio sound like instructions rather than narration. `/v1/audio/speech` is a
translation layer over the same synthesiser, not a second interface.

### Voice names

**All thirteen published names are accepted.** Seven of them used to be a `400`
— `ash`, `ballad`, `coral`, `sage`, `verse`, `marin`, `cedar` — which was never
a limitation of the model. With 54 voices loaded it was a table nobody had
extended, and a client written against the published enum had no way to know
which half of it this service would take.

Five rows are Kokoro's own borrowings and are exact. The other eight have no
counterpart in the model and are mapped by ear against OpenAI's description of
each voice, weighted towards the voices Kokoro itself grades highly — `af_heart`
is its only A.

| OpenAI | Kokoro | |
|---|---|---|
| `alloy` | `af_alloy` | en-US |
| `ash` | `am_fenrir` | en-US, by ear: firm, darker male |
| `ballad` | `bm_lewis` | en-GB, by ear: the expressive British male |
| `coral` | `af_sarah` | en-US, by ear: bright and warm |
| `echo` | `am_echo` | en-US |
| `fable` | `bm_fable` | en-GB |
| `onyx` | `am_onyx` | en-US |
| `nova` | `af_nova` | en-US |
| `sage` | `af_heart` | en-US, by ear: calm and measured |
| `shimmer` | `af_bella` | en-US, by ear: warm American female |
| `verse` | `am_puck` | en-US, by ear: expressive male |
| `marin` | `af_aoede` | en-US, by ear: the plainer newer female |
| `cedar` | `am_michael` | en-US, by ear: the plainer newer male |

OpenAI's custom-voice object is accepted too — `"voice": {"id": "fable"}` is the
schema's other form and the reference client sends it on its minimal call. An id
this service has no voice for is a `400` naming `voice`, rather than a substitute
voice nobody asked for: there are 54 fixed voices here and nothing to map an
unknown id onto.

Native Kokoro names take precedence, so `af_nova` reaches `af_nova` whatever
the table says, and `GET /voices` returns the live table under
`openai_aliases`.

The phonemiser language is inferred from the voice prefix, because OpenAI's
body has no language field — asking for `fable` and getting a British voice
read as American would be worse than inferring. `/speak` still takes an
explicit `language`.

## Deviations

Where this service cannot do what OpenAI's schema says, with the measurement
that forces it. Nothing here is hidden and nothing is faked.

### `instructions` is accepted and ignored

**The decision: ignore it, not reject it.** Kokoro-82M has no style, prosody or
emotion conditioning. A voice is a 510 KB embedding tensor selected by name, and
the ONNX graph takes tokens, style and speed and nothing else — there is no
input a sentence of direction could be routed into.

Measured rather than assumed: two requests differing only by
`instructions: "Speak in an extremely angry shouting voice, very fast,
whispering is forbidden"` returned **byte-identical audio**, SHA-1 prefix
`b477f15864f99dc5`.

Ignoring beats rejecting because OpenAI documents that `instructions` does not
work with `tts-1` or `tts-1-hd` either, so a client that sends one already
tolerates no effect, and a `400` would break clients for a field the upstream
API also drops. What has changed is that the response now names it in
`X-Ignored-Parameters`, so the caller can tell.

### An unknown custom voice id is rejected rather than substituted

`VoiceIdsOrCustomVoice` is `anyOf[string, {"id": string}]`, so the schema lets
`voice` carry any string at all — OpenAI has custom voices and an id like
`voice_1234` is legal on the wire. **Both forms are accepted here**, and the
object form used to be a `400` reading `voice: Input should be a valid string`,
which was a schema error for a request the schema allows. It is unwrapped before
validation now, so the error `loc` stays flat and `param` reads `voice` rather
than `voice.str`.

What is still refused is an id this service has no voice for. There are 54
fixed voices in the image and nothing to map an unknown id onto, so the choice
is between a `400` naming `voice` and synthesising in a voice nobody asked for;
the second is worse, and it is worse silently. The rejection is an
`invalid_value` in the envelope naming the field and listing what is accepted,
not a schema error — and the thirteen published names are all accepted first,
which is the half that was actually missing: seven of them used to be a `400`.

### `speed` outside 0.5–2.0 is clamped

`kokoro_onnx.Kokoro.create` carries a hard
`assert speed >= 0.5 and speed <= 2.0` before it will run, so the clamp is what
stops that assertion becoming a `500`. The full 0.25–4.0 range is only reachable
by time-stretching afterwards, which without a phase vocoder shifts pitch and
would sound worse than the clamp.

The clamp is not the defect; the silence was. `speed: 0.25` and `speed: 0.5`
returned the same audio, SHA `0c859dd1e014`, and `speed: 4`, `3` and `2` all
returned `a0e4bbb6e278`, with nothing on the wire to say so. `X-Speed-Clamped:
4 to 2` now goes out with the response.

### `usage` in `speech.audio.done` is a mapping this service chose

The schema makes `usage` required with three required integers, and Kokoro has
no notion of an OpenAI token in either direction. The event cannot be omitted —
a done event without `usage` violates the schema — so the rule is written down
instead. Both numbers are the model's own units rather than invented ones:

| Field | Rule |
|---|---|
| `input_tokens` | Phonemes as the model's 114-symbol vocabulary counts them: literally the tensor it is fed |
| `output_tokens` | 25 ms frames. **Measured**: the greatest common divisor of five untrimmed outputs of different lengths is exactly 600 samples at 24 kHz, so 600 is the model's own output granularity |
| `total_tokens` | The sum |

`X-Audio-Seconds` remains the number to trust for anything that matters.

### The 510-phoneme context window is a hard model property

Any utterance longer than that must be split, which is a constraint to work
with rather than a bug — but the way it used to fail was one. `_split_phonemes`
in `kokoro-onnx` breaks only on `[.,!?;]` and bounds the phoneme *string*, so a
long run between two full stops arrived at the model as one oversized batch,
truncated to exactly 510 phonemes, tokenised to exactly 510 tokens, and indexed
a voice tensor of shape (510, 1, 256) at row 510.

**Measured**: 400 characters of unpunctuated English phonemise to 518 symbols
and returned `500`, `index 510 is out of bounds for axis 0 with size 510` — from
ordinary prose well inside the 4096 characters the schema allows, on both
routes. `app/synth.py` now bounds every chunk at 509 and splits a
punctuation-free run at a space, so that index cannot be reached. The split is
also what makes streaming possible: it is the unit a delta carries.

### The streamed body differs from the buffered one in two formats

Both come out of one encoder, so for `mp3`, `aac`, `flac` and `pcm` the
concatenated deltas equal the buffered body byte for byte. Two exceptions,
both physical:

| Format | Difference | Why |
|---|---|---|
| `wav` | 8 bytes: the RIFF size at offset 4 and the `data` size at offset 40 | A length cannot be written before the last sample exists. The buffered body knows it and writes it; the stream leaves `0xFFFFFFFF`. Leaving the placeholder in the buffered body instead was measured and rejected — soundfile, ffprobe and CoreAudio all read it, but Python's `wave` reports 2147483647 frames |
| `opus` | The 4-byte Ogg page serial and the 4-byte CRC covering it, 40 bytes of 8980 on a five-page stream | The muxer randomises the serial per stream, and `-serial_offset` only offsets that random base — measured, two runs still differ. Two identical buffered requests already returned different bytes before any of this |

**The samples are untouched on both routes.** The phoneme chunking that makes
streaming possible is upstream's greedy fill line for line, checked against
`Kokoro._split_phonemes` on 400 randomly generated texts and identical on all
400, so `/speak` and a buffered `/v1/audio/speech` return the audio they always
returned — bit-identical to what `create()` produced for the same text. What
changed is the crash it used to reach on a long unpunctuated run.

Encoding to a pipe rather than a temporary file — which is what makes streaming
possible at all, since a file has to be complete before it can be read — has
three container-level consequences on **both** routes:

- `mp3` has no Xing/`Info` frame. It records the total frame count and is
  written by seeking back to the start, which a pipe cannot do. Duration is
  still computable from the file size at a constant bitrate.
- `flac` STREAMINFO carries zero for total-samples and for the MD5, the standard
  streaming convention. `ffprobe` reports `N/A` for duration; the audio decodes
  intact, checked by decoding rather than by reading the header.
- `aac` is byte-identical either way.

### `wav` and `pcm` now carry the same samples

They did not. `wav` went through libsndfile and `pcm` through
`voice_common.audio.pcm_bytes`, and the two rounded differently: over one
utterance, **54.6% of samples differed, by up to 2 LSB**. Inaudible, and not a
spec violation, but the two paths claimed to carry the same samples and did not.
One conversion now feeds every format, and a `wav` body is its 44-byte header
followed by exactly the `pcm` body of the same request. The header is
byte-identical to the one libsndfile wrote.

### An empty `input` returns an empty container

`""` and `"   "` used to be `500`, "need at least one array to concatenate",
from a schema-legal request — `CreateSpeechRequest` sets no `minLength`. They
now return `200`, and what arrives is whatever the requested container's empty
form is: 44 bytes of `wav` header, zero bytes of `pcm`, an Ogg Opus header with
no audio pages, a `flac` header and its padding block. `mp3` and `aac` come back
with no frames in them, which a player will not open — there are no samples to
put in one. A `400` was the alternative and was rejected as the less faithful
answer to a request the schema allows.

### Two fields are optional here that the schema requires

`model` and `voice` are required in `CreateSpeechRequest` and optional here.
There is one model per service, so a request that omits it has an unambiguous
answer, and `TTS_VOICE` answers for the voice. Accepting a body OpenAI would
reject breaks nothing that works against the real API.

A `model` that names something else is accepted too, for the same reason: every
OpenAI client sends a name, and refusing `tts-1` refuses the compatibility this
route exists for. It reaches nothing, though, so it is named in
`X-Ignored-Parameters` like anything else that did not — a request answered by
an 82M-parameter model it did not ask for used to be told nothing at all.

## Authentication

Off by default. Set `TTS_API_KEYS` to a comma-separated list to turn it on:

```bash
docker run -p 8001:8001 -v tts-models:/models \
  -e TTS_API_KEYS=sk-workstation,sk-laptop \
  ghcr.io/gabrielbelli/ai-voice-tts:pre
```

```bash
curl -X POST localhost:8001/speak \
  -H 'authorization: Bearer sk-workstation' \
  -H 'content-type: application/json' \
  -d '{"text":"Authenticated."}' --output out.wav
```

- **Unset means every request is accepted**, and the service says so at
  `WARNING` on every start. It neither refuses to boot nor runs open in
  silence. This service already runs on a LAN with no keys anywhere, and an
  upgrade that starts rejecting every existing caller is a worse outage than a
  warning nobody reads.
- **Set but naming no key refuses to start.** `TTS_API_KEYS=`, `,` and `,,  ,`
  all exit with a sentence saying so. Unset is a choice; set to nothing is an
  accident — `-e TTS_API_KEYS=$SECRET` with `SECRET` unset hands the container
  an empty value — and it used to leave the service open to anyone under a
  warning that claimed the variable was unset.
- One list, no per-key identity or scopes. Rotation is: add the new key, move
  the callers, drop the old one.
- Surrounding whitespace is trimmed from each key, so `k1, k2` works as
  written. A key is therefore never surrounded by spaces, which is just as
  well: HTTP strips a field value's trailing whitespace, so one could not be
  presented even if it were configured.
- Non-ASCII keys work — the comparison is done on the bytes that crossed the
  wire, and the configured key is encoded as UTF-8 to match. HTTP does not
  require a client to send UTF-8, though, so a start with one logs a warning
  and an ASCII key avoids the question.
- Keys are compared with `hmac.compare_digest` against every configured key,
  with no early exit on a match. `==` returns at the first differing byte and
  leaks the matching prefix; stopping at the match would leak how far down the
  list a valid key sits.
- **`/health` is never authenticated**, `/health/` included. Container
  healthchecks have no key and no way to be given one, and a probe written
  with the trailing slash must not go permanently unhealthy the moment keys
  are set. Everything else needs one, `/docs` and `/openapi.json` included.
- Enforcement is middleware, not a per-route dependency, so a route added later
  is protected without anyone remembering to ask.

A rejected request gets `401` in OpenAI's envelope:

```json
{"error": {"message": "Incorrect API key provided. Send it as 'Authorization: Bearer <key>'.",
           "type": "invalid_request_error",
           "code": "invalid_api_key"}}
```

That envelope is used on the native routes too, unlike every other error. A
rejection happens before routing, so there is no route yet whose conventions
it could follow, and the client most likely to be turned away is the one that
only reads `error.message`.

## TLS

Off by default. Set both `TTS_TLS_CERT` and `TTS_TLS_KEY` to PEM paths and
uvicorn serves HTTPS on the same port:

```bash
docker run -p 8001:8001 -v tts-models:/models \
  -v /etc/letsencrypt/live/tts.example.net:/certs:ro \
  -e TTS_TLS_CERT=/certs/fullchain.pem \
  -e TTS_TLS_KEY=/certs/privkey.pem \
  -e TTS_API_KEYS=sk-workstation \
  ghcr.io/gabrielbelli/ai-voice-tts:pre
```

> **No self-signed certificate is ever generated.** A certificate that appears
> by magic is one nobody validates, and a client taught to skip verification
> keeps skipping it against the real certificate too. Bring a real one, or
> terminate TLS at a reverse proxy and leave this service on plain HTTP behind
> it.

What the entrypoint does with the pair, before it drops privileges:

| Situation | Result |
|---|---|
| Only one of the two variables set | Exits. Half a configuration must not become a silent downgrade to plain HTTP |
| Either file unreadable **by uid 1000** | Exits with the path. Readability is tested as the user that will open the file, not as root — a key mounted `0600 root:root` reads fine for the entrypoint and not at all for uvicorn |
| `setpriv` itself cannot run | Exits quoting what `setpriv` said. Its own failure is not a permissions problem, and reporting it as one sends an operator to inspect a file whose mode was fine |
| Both set, but the command is not `uvicorn` | Exits. Only uvicorn is handed the certificate, so a `command:` override with TLS configured would have served plain HTTP while printing that it was serving HTTPS |
| Both set and readable | Appends `--ssl-certfile` and `--ssl-keyfile` to the uvicorn command |

The flags are appended to the command rather than read in Python, so the `CMD`
baked into the image is exactly what it always was. Running uvicorn directly,
outside the container, the environment variables do nothing — pass the flags
yourself:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 \
  --ssl-certfile /certs/fullchain.pem --ssl-keyfile /certs/privkey.pem
```

Certificate renewal is the host's business. uvicorn reads the PEM files once at
start, so a renewed certificate needs a container restart.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TTS_VOICE` | `bm_george` | Default when a request omits one |
| `TTS_LANGUAGE` | `en-us` | `en-us`, `en-gb`, `pt-br`, … |
| `TTS_THREADS` | `4` | Must match your CPU limit — see below |
| `TTS_MODEL_DIR` | `/models` | Volume for weights |
| `TTS_API_KEYS` | unset | Comma-separated accepted keys. Unset means no authentication; set but naming none refuses to start |
| `TTS_TLS_CERT` | unset | PEM certificate chain. Both TLS variables or neither |
| `TTS_TLS_KEY` | unset | PEM private key, readable by uid 1000 |
| `TTS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. An unrecognised value logs a warning and stays at `INFO` rather than refusing to start |
| `TTS_CHUNK_PHONEMES` | `509` | Phonemes per model pass, and so how often an SSE delta leaves. See below |

### Chunk size and streaming latency

The default is the model's context window (510) less the one row of the voice
tensor that cannot be indexed. It is also the batching `kokoro-onnx` would have
chosen on any text it handled correctly, so the audio a buffered request returns
is what this service always returned.

Lowering it buys latency and costs length. Measured on a 4096-character input:

```text
TTS_CHUNK_PHONEMES   chunks   audio     first chunk
       509              7     154.5 s      6.21 s
       200             17     174.5 s      2.25 s
       100             39     180.8 s      1.29 s
```

The 17% growth is the duration predictor seeing less context per chunk, not
silence accumulating at the seams — it is a real change to the speech, which is
why the default does not move. Lower it only if a caller needs the first audio
sooner than a whole model pass and can live with slightly slower delivery.

## Limiting CPU use

Set the container's CPU limit **and** `TTS_THREADS` to the same number. ONNX
Runtime sizes its thread pool from the host's core count, not the cgroup, so a
`--cpus` limit alone leaves the container spawning a thread per host core and
then contending for the slice it is allowed — slower than simply using fewer
threads.

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 --memory 2g \
  ghcr.io/gabrielbelli/ai-voice-tts:pre
```

Steady state is about 400 MB, so 2 GB is generous.

Every buffered response carries `X-Realtime-Factor`. If it drops when you raise
`TTS_THREADS`, coordination is costing more than the extra cores return. An SSE
response cannot carry it — the headers leave before the audio exists — so
`stream_format: "audio"` is the one to measure with.

## Performance

Measured on an M2 Max, CPU only:

```text
20.7 s of speech generated in 5.0 s   =  4.1x realtime
17.1 s of speech generated in 4.0 s   =  4.3x realtime
```

~330 MB resident. The GPU is never touched.

## What is not here

**Chatterbox**, the long-form alternative, is deliberately absent. It needs
5.3 GB and runs below realtime, so it belongs behind a separate service that
can be started on demand rather than sitting resident beside a model a
thousandth its size. Whether it is viable on CPU at all is still being
measured.

**Qwen3-TTS** is excluded for a different reason: it is Mandarin-first, and
its English and Portuguese carry a Chinese accent. **F5-TTS** and **XTTS-v2**
are excluded for their non-commercial licences.

## Shared code

The API key middleware, the OpenAI error envelope, the `/health` route, the
`Segment` and OpenAI request models, the PCM and splicing helpers, the logging
setup and the TLS entrypoint are not written here. They come from
[packages/common](../../packages/common/README.md), installed as a path
dependency in `requirements.txt` and shared with `services/stt` and
`services/tts-long`.

That is not tidiness. The three services each carried their own copy of
`app/auth.py`; the copies drifted by 170 to 197 lines, and one review round
found three *different* defects, one per repo, because each had drifted
separately. Two of the three were this repo's:

- a non-ASCII `TTS_API_KEYS` value could never authenticate — the correct key
  was rejected as the wrong one
- `GET /health/` came back `401` the moment keys were configured, so a probe
  written with the trailing slash went permanently unhealthy

Both are fixed here and both are now assertions in a suite the package ships.
`tests/test_conformance.py` is four lines of fixture; the tests come from
voice-common and run in CI against the app object this repo actually builds,
so a bad bump of that pin fails at the build rather than on the box:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

`python -m pytest` rather than `pytest`, because the module form puts the
working directory on `sys.path` and `app` is a source tree here, not an
installed package.

Nothing in it loads Kokoro — the suite never enters the app's lifespan, so CI
never downloads the 340 MB of weights.

What stays here is what is actually this service's: the Kokoro voice table and
its aliases, the ffmpeg encoder set and its bitrates, the format enum, the
`0.5`–`2.0` speed clamp, the phoneme chunking, the SSE event shape, the segment
`voice` field, and every route.

`app/errors.py` is the exception, and a temporary one. The `param` key, the
`/v1` 404 and 405 handlers and the middleware that fills `param` into the shared
auth middleware's `401` all belong in `voice_common.errors` — that is where the
envelope lives and all three services need them. They are here because the
shared package used to be pinned to a commit tarball, so a change there reached
this image only on the next bump, and this service was live. **That blocker is
gone**: the package is a path dependency from this same tree now, and moving the
module across is a follow-up rather than something a pin forbids. It is written
to be lifted unchanged.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
