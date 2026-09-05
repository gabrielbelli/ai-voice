# stt-stack

Self-hosted speech-to-text. One container, whole pipeline, CPU only.

```text
audio
  ↓  VAD          Silero — drop silence first
  ↓  recogniser   Parakeet TDT 0.6B v3 by default, Whisper large-v3 on request
  ↓  glossary     repair known terms — opt-in, per request
text
```

No CUDA, no torch. One model at a time, chosen by `STT_MODEL`.

## Which model

**Parakeet is the default.** Measured across 25 conditions and five Brazilian
Portuguese corpora on identical audio:

| Engine | pt-BR WER | English WER | rtf | RAM | Disk |
|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v3** | **0.144** | **0.121** | **47–63×** | 1.4 GB | 461 MB |
| Whisper large-v3 | 0.250 | 0.131 | 0.5–0.9× | 2.9 GB | 2.9 GB |

Parakeet won 21 of 25 conditions at roughly seventy times the speed. It also
degrades far more gracefully — band-limiting to 4 kHz, which is what a cheap
or distant microphone does, cost Whisper +206% WER on CORAA and Parakeet +41%.

**Whisper is worth choosing** for clean read speech, where it genuinely leads,
and for translation. It used to be the only engine that could take a vocabulary
at decode time; that is no longer true. Both engines now bias their decoder
from the same terms — Whisper always, Parakeet when a request adds
`boost=true`. See [Decode-time biasing on Parakeet](#decode-time-biasing-on-parakeet).

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
docker run -p 8000:8000 -v stt-models:/models \
  --cpus 4 -e STT_THREADS=4 \
  ghcr.io/gabrielbelli/ai-voice-stt:pre
```

First start downloads the selected model into the volume. Later starts are
immediate.

```bash
curl -F file=@clip.wav http://localhost:8000/transcribe
```

```json
{
  "text": "I need to make a comet on the theory dashboard",
  "raw": "I need to make a comet on the theory dashboard",
  "repaired": [],
  "model": "parakeet",
  "audio_seconds": 4.1,
  "speech_seconds": 3.2,
  "compute_seconds": 2.4,
  "realtime_factor": 1.7
}
```

**No glossary is applied unless you ask for one.** `text` equals `raw` above
because this request selected no profile — see [Glossary profiles](#glossary-profiles).
Select one and the repair happens:

```bash
curl -F file=@clip.wav -F glossary=dictation http://localhost:8000/transcribe
```

```json
{
  "text": "I need to make a commit on the theory dashboard",
  "raw": "I need to make a comet on the theory dashboard",
  "repaired": ["commit"]
}
```

**This route takes 16 kHz only.** Any container libav reads is fine — flac,
mp3, m4a, mp4, ogg, wav, webm — and stereo is downmixed, but a clip at another
sample rate is rejected rather than resampled, so a client sending 44.1 kHz
finds out immediately instead of quietly getting worse transcripts. That rule
is this route's alone: `/v1` resamples, because no OpenAI client anticipates
it.

## OpenAI-compatible API

Two routes speak OpenAI's audio shape, so anything already pointed at that API
works here by changing a base URL:

```text
POST /v1/audio/transcriptions
POST /v1/audio/translations
```

```bash
curl -H "Authorization: Bearer $STT_API_KEY" \
     -F file=@clip.wav -F model=whisper-1 \
     http://localhost:8000/v1/audio/transcriptions
```

```json
{"text": "I need to make a comet on the theory dashboard",
 "usage": {"type": "duration", "seconds": 4}}
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")

with open("clip.wav", "rb") as clip:
    print(client.audio.transcriptions.create(model="whisper-1", file=clip).text)
    # Add extra_body={"glossary": "tech"} to select a profile — see below.
```

`response_format` takes `json`, `text`, `verbose_json`, `srt` and `vtt`. All
nine input formats the specification lists are accepted — flac, mp3, mp4, mpeg,
mpga, m4a, ogg, wav and webm — at any sample rate, in mono or stereo. Decoding
is libav, and it resamples: **this route does not have the native route's
16 kHz rule**, because no OpenAI client anticipates one and a 44.1 kHz mp3 is
the ordinary case there.

### Every field is honoured, or refused by name

Nothing is accepted and dropped. A dropped field is a client believing
something false about the transcript it just received, and this surface used to
drop eleven of them — `timestamp_granularities[]`, `chunking_strategy`,
`include[]`, `language`, `temperature`, `prompt`, `keywords[]`, `languages[]`,
`stream`, both diarisation fields — each with a 200 and a body that gave no
sign. `stream=true` was the worst: the client's event loop completed having
seen no events and raised nothing.

| Field | Behaviour |
|---|---|
| `file` | All nine formats, any rate, mono or stereo |
| `model` | **Required**, and it does not choose an engine. See the deviation below |
| `response_format` | `json`, `text`, `verbose_json`, `srt`, `vtt`. `diarized_json` is refused |
| `timestamp_granularities[]` | Honoured. `segment` is the default, `word` adds word timings |
| `chunking_strategy` | Honoured — `server_vad`'s `threshold`, `prefix_padding_ms` and `silence_duration_ms` tune the VAD this service already runs |
| `stream` | Honoured on Whisper, refused on Parakeet. See below |
| `language`, `temperature` | Honoured on Whisper, refused on Parakeet |
| `prompt`, `keywords[]` | Honoured on **both**, in whichever halves the engine has. See [A request's own vocabulary](#a-requests-own-vocabulary) |
| `glossary` | **Extension.** Named profiles, `glossary=tech,dictation`. Honoured on both engines. See [Glossary profiles](#glossary-profiles) |
| `include[]=logprobs` | Honoured on Parakeet, refused on Whisper |
| `languages[]` | Refused: neither engine takes a candidate set |
| `known_speaker_names[]`, `known_speaker_references[]` | Refused: nothing here diarises |
| anything else | Refused by name. `CreateTranscriptionRequest` sets `additionalProperties: false` |

Every refusal is a 400 in OpenAI's envelope, naming the field in `param` and
saying which engine could do it:

```json
{"error": {"message": "Unsupported parameter: 'language' is not supported by the 'parakeet' engine, which detects the language itself and takes no hint. …", "type": "invalid_request_error", "param": "language", "code": "unsupported_parameter"}}
```

### What each engine can do

The two recognisers are not interchangeable, and the compatibility layer
answers for the difference rather than papering over it. `/health` reports
`translations` and `streaming` for the engine that is loaded, so a client can
find out without spending a request on a refusal.

| | Parakeet (default) | Whisper |
|---|---|---|
| `stream=true` | refused — batch decoder | transcript deltas, one per window |
| `/v1/audio/translations` | refused — no translate task | `task="translate"` |
| `language` | refused — takes no hint | honoured |
| `prompt`, `keywords[]` | honoured — post-decode repair, *and* decode-time biasing with `boost=true` | joined into hotwords, *and* the same repair |
| `glossary` | honoured — repair, *and* biasing with `boost=true` | honoured — repair *and* hotwords |
| `boost` | honoured — switches decode-time biasing on for this request | refused — hotwords here are unconditional, there is nothing to switch |
| `temperature` | refused — no sampling temperature | honoured, and it disables the fallback ladder |
| `include[]=logprobs` | per-token logprobs | refused — only a per-segment average exists |
| `verbose_json.language` | `"unknown"` — no language ID in the image | the detected language, e.g. `"english"` |
| `segments` | cut at the VAD's own pauses | Whisper's own, with its own `seek` and `no_speech_prob` |
| `words` | from TDT token timings | from Whisper's word timestamps |

### Streaming

`stream=true` is genuinely incremental on Whisper and refused on Parakeet.

faster-whisper yields each segment as CTranslate2 finishes the 30-second window
it belongs to. Measured over HTTP with `tiny`/int8 on four threads, a 297-second
clip:

```text
first delta   6.31 s
last event   85.07 s      the first event is 7% of the way in
```

A clip shorter than one window has one window, so its first and last deltas
arrive together — 2.6 s on a 14.2-second clip. That is the model's granularity
showing through, not a shortcut here.

Parakeet encodes the whole waveform and then runs a decode loop that emits
nothing until it ends: 5.07 s to the first and only output on that same 14.2
second clip. There is no partial transcript to send, so the field is refused
rather than answered with one delta at the end. Cutting a finished transcript
into timed deltas would be a lie a client builds timing assumptions on, and the
specification itself notes that streaming is ignored for `whisper-1`, so a
refusal has precedent.

On the wire: bare `data:` frames, no `event:` name lines, every event
terminated by a blank line including the last, and no `[DONE]` sentinel. The
schema models the JSON payload only, and openai-python dispatches on the JSON
`type`; the trailing blank line is the one that matters, because its decoder
drops a final event ending in a single newline silently, with no error. The
concatenation of every `transcript.text.delta` equals the `transcript.text.done`
text exactly — glossary repair runs per segment as it is emitted, so a client
that renders deltas and a client that waits for the terminal event see the same
transcript.

### Errors

Every `/v1` error is OpenAI's envelope with all four keys — `type`, `message`,
`param` and `code` — where `param` and `code` are present as `null` rather than
missing. That includes a 404 on an unknown path and a 405 on a wrong method,
both of which used to leak FastAPI's `{"detail": ...}`, which openai-python
reports as a bare "unknown error". A rejected request is a **400**, which is
what the real API answers and what a client written against it branches on.

`STT_MAX_CONCURRENT` puts a ceiling on how many transcriptions run at once;
past it, `/v1` answers **429** with `Retry-After` instead of queueing. It is
unset by default — a limit picked here rather than measured on your host would
start refusing work the service is doing happily today.

### Which route to prefer

**`/transcribe`, wherever you control the client.** It is the same pipeline,
and it returns what the OpenAI shape has no field for:

| | `/transcribe` | `/v1/audio/transcriptions` |
|---|---|---|
| Transcript | yes | yes |
| Which glossary terms fired | `repaired` | `x-glossary-repaired` |
| `raw` — the transcript before repair | yes | — |
| `realtime_factor` and the timings behind it | yes | — |
| Segments, words, subtitles, streaming | — | yes |
| Works with an unmodified OpenAI client | — | yes |

A silent substitution is worse than no substitution: if a glossary rule is
wrong, you want to see it named. The native route puts that list in `repaired`;
the compatible route could not, and now answers it in a header instead —
**`x-glossary-repaired`**, present only when something was actually rewritten,
listing the terms comma-separated and percent-encoded (a header value is
latin-1 on the wire and a glossary term is not). A body key would have changed
the response shape, which ADR 0001 forbids an extension from doing, and `text`,
`srt` and `vtt` have nowhere to put one anyway.

It is absent on `stream=true`: the headers go out before the first delta is
decoded, so at that point nothing has been repaired. The deltas themselves are
already repaired.

### Deviations

What cannot be 1:1, why, and the measurement that forces it. Nothing here is
hidden and nothing is faked.

**`model` does not choose an engine.** It is required, as the specification
requires it, and its value is not obeyed: Parakeet needs 1.4 GB resident and
Whisper large-v3 2.9 GB, holding both does not fit the memory this is deployed
under, and a cold load is minutes. Refusing `whisper-1` on a Parakeet
deployment would reject every existing client — Open WebUI sends it, the
example above sends it — to make a point about a name. So the request is
answered and **every `/v1` response carries `x-stt-engine`** naming the engine
that actually ran. Honesty rather than obedience.

**No streaming, translation, `language` or `temperature` under Parakeet.** All
four are refusals, not silences, and all four are properties of a TDT decoder
that has no such mechanism and nothing downstream that can stand in for one.
Deploy with `STT_MODEL=whisper` if you need them, and pay the order of
magnitude in latency.

`prompt` and `keywords[]` used to be a fifth and sixth, on the grounds that
Parakeet's decoder took no vocabulary. That was wrong twice over: they reach
the post-decode repair, and since decode-time biasing landed they reach the
decoder too. The lesson is in the list above — a refusal has to name a
mechanism that genuinely does not exist, and "the library exposes no argument
for it" is not that.

**No diarisation.** `diarized_json`, `known_speaker_names[]` and
`known_speaker_references[]` are refused. There is no speaker-embedding or
clustering component in this image and neither engine produces speaker labels.

**Two segment fields are synthesised under Parakeet.** `seek` is a Whisper
30-second window offset and there are no windows — the encoder runs over the
whole waveform — so it is `0`. `no_speech_prob` is a Whisper decoder output
that a TDT decoder does not produce, so it is `0.0`; the VAD has already
removed what it believed was silence. `tokens` is empty, because onnx-asr maps
token ids to strings inside its decoder and returns only the strings, and an
array of invented integers would be worse than an empty one.

**Word timings mean slightly different things.** Whisper reports a start and an
end per word. Parakeet's TDT decoder reports the frame at which each token was
*emitted*, quantised to 0.08 s, so a word's `end` is its last token's
timestamp and its `start` is the timestamp of the token before it. Words are
therefore contiguous, with no invented gaps.

**`usage` is the duration variant only** — `{"type": "duration", "seconds": N}`
— because neither engine is billed by tokens or reports an input token count.
The terminal streaming event omits `usage` entirely for the same reason: the
schema pins it to the token variant there, and inventing `input_tokens` would
be worse than an absent optional field.

**`glossary` is an extension, and it is off unless asked for.** OpenAI has no
glossary-profile field, so this one travels the way ADR 0001 requires an
extension to travel — a body field an SDK reaches with `extra_body`, sitting in
the allowlist beside `keywords[]` and `languages[]`, absent by default. A
client that knows nothing about it behaves exactly as it would against OpenAI,
and the four `/glossaries` routes that manage the profiles are deliberately
native rather than `/v1`.

**Glossary repair happens on strings**, when a profile was selected or a
request sent terms of its own. It is applied to the whole transcript,
and separately to each segment and each word, so a rule spanning a boundary —
`cloud code` split across two segments — fires in `text` and cannot fire inside
the smaller unit. `/transcribe` names every rule that fired in `repaired`, and
`/v1` in `x-glossary-repaired`.

**`seek` is not remapped.** With the VAD on, the recogniser sees speech runs
concatenated, and every start and end is mapped back onto the clip you sent.
`seek` indexes a decoder window rather than naming a moment, so it is passed
through as the decoder reported it. `id` is *not* passed through: faster-whisper
numbers its segments from 1 and the specification's own example numbers from 0,
so both engines are renumbered from 0 here rather than answering the same field
differently depending on which one is loaded.

**Two conventions chosen by fiat.** `vtt` is served as `text/vtt; charset=utf-8`
— the specification declares no content type for it, and a browser needs that
one before it will treat the body as a track. There is no `[DONE]` sentinel on
the stream: nothing authoritative says the real API emits one, and both SDK
decoders tolerate its absence.


## Authentication

`STT_API_KEYS` is a comma-separated list of accepted keys. Clients send
`Authorization: Bearer <key>`, which is what OpenAI clients already do.

```bash
docker run -p 8000:8000 -v stt-models:/models \
  -e STT_API_KEYS="$(openssl rand -hex 32)" \
  ghcr.io/gabrielbelli/ai-voice-stt:pre
```

**Unset means no authentication**, and the service says so at every startup:

```text
WARNING STT_API_KEYS is unset: authentication is DISABLED and every request is
accepted, including /v1. Set STT_API_KEYS to a comma-separated list of keys to
require Authorization: Bearer.
```

That is deliberate. This already runs on a LAN, and an upgrade that refused to
start, or refused every request, would break a working deployment in order to
protect it. Open is allowed; quietly open is not.

**Set to nothing, or to separators alone, is a different thing and refuses to
start.** `STT_API_KEYS=" , , "` and `STT_API_KEYS=""` parse to no keys at all,
and announcing either as "unset" would report a typo as a decision — whoever
wrote it wanted authentication and would have got none. `-e
STT_API_KEYS=$SECRET` with `SECRET` unset reaches this by ordinary accident:

```text
STT_API_KEYS=' , , ' is set but names no key: it is empty, or only commas and
whitespace. Set it to a comma-separated list of keys, or unset it entirely to
run without authentication.
```

`/health` is the only unauthenticated route. Container healthchecks call it and
have no key, and requiring one turns a working service into a restart loop.

`/openapi.json`, `/docs` and `/redoc` need the key like everything else — an
unauthenticated schema is a free map of the service — and so does any path that
does not exist, because the check runs ahead of routing. In a browser the two
pages then render empty, because the browser sends no bearer token when it
fetches the schema; read the schema with `curl` and a key.

Rejections are 401 in OpenAI's envelope, which is the shape openai-python
reads — with FastAPI's default `{"detail": ...}` it reports "unknown error":

```json
{"error": {"message": "Incorrect API key provided. Send it as 'Authorization: Bearer <key>'.", "type": "invalid_request_error", "code": "invalid_api_key"}}
```

Keys are compared with `hmac.compare_digest`, and every configured key is
compared rather than stopping at the first match, so neither a key's value nor
its position in the list leaks through the response time. The comparison is
done on the bytes the client put on the wire, so a key containing an accent
authenticates — provided the client sends the header as UTF-8, which HTTP does
not guarantee and the service warns about at startup. ASCII keys avoid the
question.

The list is read once, at startup: rotating a key is a restart.

The benchmark in `bench/` is a client like any other. Export `STT_API_KEY` for
it when the service it points at has keys configured.

## TLS

Set both `STT_TLS_CERT` and `STT_TLS_KEY` to PEM files and the container
serves HTTPS. Leave them unset and it serves plain HTTP, as before.

```yaml
services:
  stt:
    environment:
      STT_TLS_CERT: /etc/letsencrypt/live/stt.example.net/fullchain.pem
      STT_TLS_KEY: /etc/letsencrypt/live/stt.example.net/privkey.pem
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt:ro
```

**Mount `/etc/letsencrypt`, not `live/<domain>`.** The files under `live` are
relative symlinks into `../../archive/<domain>/`. Mount the `live` directory
alone and the symlinks arrive intact with their targets left outside the
container, where they resolve to nothing and uvicorn exits on a missing file.

Setting only one variable is a configuration error, and the container refuses
to start. It used to warn and serve plain HTTP, which is the failure that
hides: the operator reads "TLS is configured" from their own compose file and
believes it, while a bearer token crosses the LAN in the clear on every
request. Failing to start is the loud version.

Overriding the container's `command:` with TLS configured is refused outright.
Only uvicorn is given the certificate, so any other command would serve plain
HTTP with the deployment believing otherwise — the same silent failure, and
the one thing worse than being told is not being told.

Both files must be readable by uid 1000, which is what the service drops to,
and the entrypoint checks that **as uid 1000** before starting rather than
leaving uvicorn to fail on it with a traceback. Certbot leaves `archive/` mode
`0700` and the private key `0600`, both owned by root, so the straight mount
above needs either those permissions widened or a deploy hook that copies the
pair somewhere owned by uid 1000.

**Nothing here generates a certificate.** A self-signed certificate that
appears by magic is one every client is eventually told to stop verifying,
which is worse than the plain HTTP it replaced. Use a real one — an internal
CA, or Let's Encrypt over DNS-01 for a host with no public address — or
terminate TLS at a reverse proxy in front of the container and leave both
variables unset.

Under TLS the healthcheck probe needs the `https://` URL and something that
can verify the certificate.

## Why there is no second model

A consensus pass was built, measured, and removed. The idea was that two
models failing differently would flag the words worth doubting.

In practice, across every disagreement observed, **the second model was the
wrong one**. Not once did it catch a real error. A second opinion only informs
when it is roughly as good as the first; when it is reliably worse, its
dissent reduces to "the weaker model is wrong again" — noise, at about 40% of
throughput.

The cost landed worst where it mattered. On short clips, which is what
dictation consists of, fixed per-request cost dominates:

```text
long clips (read corpora)          rtf 1.26 – 1.60
short clips (spontaneous corpora)  rtf 0.39 – 0.47
```

## Why no LLM cleanup

The obvious next stage, and a trap at any size that fits beside a recogniser
on a CPU box. Tested at 4B with an explicit prompt forbidding it, the cleanup
stage still:

- inverted meaning — "makes no sense to be available to me" became "are not
  available to me"
- reversed pronouns, turning "those tasks for you" into "those tasks for me"
- deleted content it was told twice to preserve
- leaked its own reasoning into the output ("No, that's not right")

Reliable adherence starts around 14B, which needs 10–20 GB of VRAM. Below
that the raw transcript is more faithful than the cleaned one, and a faithful
transcript is the product.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STT_MODEL` | `parakeet` | `parakeet` or `whisper` |
| `STT_MODEL_ID` | model default | Override the specific checkpoint |
| `STT_QUANTISATION` | `int8` | Whisper also takes `int8_float32`, `float32` |
| `STT_LANGUAGE` | unset | Leave unset if you code-switch. See below |
| `STT_THREADS` | `4` | Must match your CPU limit. See below |
| `STT_VAD` | `1` | Silence removal |
| `STT_HOTWORDS` | `1` | `0` disables decode-time biasing entirely, for A/B tests. See below |
| `STT_MAX_CONCURRENT` | `0` | Transcriptions allowed at once. `0` is no limit; past a limit, `/v1` answers 429 with `Retry-After` |
| `STT_GLOSSARY_BUILTIN` | `/etc/ai-voice/glossaries` | Read-only profiles baked into the image |
| `STT_GLOSSARY_DIR` | `/glossaries` | Writable profiles. Mount a volume here or the write routes answer 503 |
| `STT_GLOSSARY_DEFAULT` | unset | Profiles applied when a request selects none. **Leave it unset.** See below |
| `STT_GLOSSARY` | unset | One extra file, loaded as a profile named after it. The pre-profile variable |
| `STT_API_KEYS` | unset | Comma-separated accepted keys. Unset means no auth |
| `STT_LOG_LEVEL` | `INFO` | `DEBUG`, `WARNING`, … An unrecognised value falls back to `INFO` |
| `STT_TLS_CERT` | unset | PEM certificate. With `STT_TLS_KEY`, serves HTTPS |
| `STT_TLS_KEY` | unset | PEM private key, readable by uid 1000 |

### Glossary profiles

A glossary is a **named profile**, selected per request. **Nothing is applied
unless a request asks for it.**

That default is measured, not tidy. Across 25 cells, a glossary whose
terms do **not** occur in the audio raised WER by **28% on Whisper and 28% on
Whisper**. Irrelevant terms are not inert — they actively cost accuracy — which
is why one always-on list is the worst available shape and why **selecting
several profiles at once is discouraged**. Select the one that matches what is
being said.

Two profiles ship in the image, read-only:

| Profile | Contents |
|---|---|
| `dictation` | Mishearings any dictating user hits: `ldr = TLDR`, `dts = STT`, `tex to speak = text-to-speech` |
| `tech` | **General** technical vocabulary — Kubernetes, nginx, PostgreSQL, ONNX. Vendor and tool names anyone in the field would say |

**Neither contains anyone's project names**, and that is the point of the split
rather than a stylistic preference. A term in `tech` is paid for by every
request that selects `tech`, so a name only one person ever says makes the
profile worse for everybody else. Your own names go in a profile you supply;
`examples/glossaries/personal.txt` is a worked example.

#### Selecting one

```bash
# native route
curl -F file=@clip.wav -F glossary=tech http://localhost:8000/transcribe

# OpenAI-compatible route
curl -F file=@clip.wav -F model=whisper-1 -F glossary=tech \
     http://localhost:8000/v1/audio/transcriptions
```

```python
client.audio.transcriptions.create(
    model="whisper-1", file=clip,
    extra_body={"glossary": "tech"},          # named profiles, an extension
    prompt="Theoria, Catallaxy",              # one-off terms, the spec's own field
)
```

`prompt` is the specification's own field, defined as text that guides the
model, and it is what a one-off should use — it needs no extension. `glossary`
is the extension, allowlisted beside `keywords[]` and `languages[]`. **A
request naming a profile that does not exist is a 400 naming it**, never a
silent no-op.

#### Managing them

```text
GET    /glossaries            every profile: name, source, term count
GET    /glossaries/{name}     its terms, and the file text they came from
PUT    /glossaries/{name}     create or replace a custom profile
DELETE /glossaries/{name}     remove one
```

```bash
curl -X PUT --data-binary @mine.txt http://localhost:8000/glossaries/mine
curl http://localhost:8000/glossaries
```

These are **native routes, not `/v1`**: OpenAI has no concept of a glossary
profile, so there is nothing to be 1:1 with, and claiming `/v1/glossaries`
would take specification territory that does not exist.

**Writability follows the volume.** Built-ins live in the image and are
read-only — a `PUT` or `DELETE` on `tech` or `dictation` is a **409**, not a
silent shadow, because a profile whose contents depend on which directory won
is a profile nobody can reason about. Custom profiles live in `/glossaries`,
and if nothing is mounted there the write routes answer **503 naming the
reason** while the built-ins carry on serving. A `PUT` that evaporated on the
next restart would be worse than a refusal.

> **Set `STT_API_KEYS` before you mount that volume.** Authentication is
> `STT_API_KEYS`, unchanged, and on a deployment that leaves it unset **a write
> API is an unauthenticated write API**.

A profile written over the API applies to the **next request** — no restart.
Editing a file in the mounted directory by hand works the same way; the
registry stats the directory and its files and rescans only when one changes.

#### Two line forms, because they are two different jobs

```text
catalaxy = Catallaxy    a replacement AND a hotword
Catallaxy               a hotword only
```

Use the bare form when the likely mishearing is an ordinary word. "Belli" is
heard as "belly", but a `belly = Belli` rule would corrupt any sentence that
genuinely says belly. Biasing the decoder is safe; rewriting is not.

Decoder biasing is much the stronger of the two. Measured against real
recordings, hotwords alone fixed every technical term — `commit` (heard as
"comet"), `Theoria` ("theory"), `FreeBSD` ("free BSD"), `Belli` ("Belly") — and
the post-decode replacement never had to fire. It can also recover a word
string replacement never sees, because the wrong spelling was never in the
list.

This paragraph used to end **"Parakeet has no such mechanism"**, and a
hotword-only line really did nothing on the default engine. Both are now false;
see below.

### Decode-time biasing on Parakeet

Add `boost=true` to a request and its vocabulary — the profiles it selected,
its own `prompt` and `keywords[]` — is compiled into a boosting automaton and
fused into Parakeet's TDT decoding loop, one frame before the argmax. A
hotword-only line finally does something on the engine this service actually
deploys.

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@clip.wav -F model=whisper-1 \
  -F glossary=tech -F boost=true
```

**It is off unless you ask.** Not because it is expensive — it is not, see
below — but because the thing it buys is small and the caller is the only one
who knows whether they are about to say any of the words. `STT_BOOST=1` changes
the default for a deployment that wants it on everything.

#### What it costs and what it buys, measured

`bench/boost_bench.py`, 145 clips and 942 s over CORAA, FLEURS pt-BR,
LibriSpeech clean and Earnings-22, on Parakeet TDT 0.6B v3 at int8. Pooled
relative WER against no boosting, 95% paired bootstrap over clips:

| the list you send | WER | change | 95% CI |
|---|---|---|---|
| none (plain) | 0.1135 | — | — |
| **the shipped `tech`+`dictation` profile, on audio containing none of it** | 0.1135 | **byte-identical** | zero false fires |
| 200 unrelated phrases — `STT_BOOST_MAX_PHRASES` exactly | 0.1140 | +0.4% | [−0.8, +1.7] |
| the words that ARE in the audio | 0.1077 | **−5.2%** | [−9.2, −1.3] |
| those same words, padded to 200 phrases with another language's | 0.1077 | −5.2% | [−9.5, −1.4] |

**Irrelevant vocabulary is inert in this decoder.** At
`STT_BOOST_START_WEIGHT = 0` a phrase has to be entered on acoustics, and a word
that is not in the audio never is, so carrying it changes nothing. The last two
rows are the same number to four decimal places: padding a clip's handful of
real terms out to the full 200-phrase ceiling with words from another language
costs exactly nothing.

That is **not** true of the other biasing paths this project has measured, and
the difference is why the number below is worth stating separately. Whisper's
hotwords cost **+28% WER** when the terms are absent (`baseline` 0.3208 against
`whisper-nohotwords-orko` 0.2499), and FluidAudio's CoreML `--custom-vocab` on
Parakeet cost **+12%** (`parakeet-plain` 0.1443 against `parakeet-vocab`
0.1620). Both are real, both are other implementations, and neither describes
this one. See ADR 0005.

**And the win is real but small: −5.2% is fourteen words out of 2,378.** Term
recall goes 0.908 → 0.922, about a sixth of what plain missed. Quote the
fourteen words alongside the percentage or you are overselling it. Throughput is
unaffected — 17.8x plain against 18.0x with a full 200-phrase automaton, inside
a ±8% measurement spread.

Two caveats that belong next to those numbers. **No jargon corpus exists**, so
the terms-present column is a proxy — rare words from public corpora, not
`Catallaxy` and `Theoria` in the voice that says them. And **CORAA, the corpus
closest to real dictation here, moved by nothing at any weight**: its clips are
3.4 s and 10 words, and 14 of 40 hold no word distinctive enough to boost.

Separately, on a synthetic probe — mechanism, not WER:

| | transcript |
|---|---|
| unboosted | Anthropic released clode code, and the **Thearia** dashboard uses Ghost Pepper for dictation. |
| `boost=true` | Anthropic released clode code, and the **Theoria** dashboard uses Ghost Pepper for dictation. |
| terms absent from the audio | byte-identical to unboosted |

`clode code` was **not** recovered, and that is the same fact as the free absent
axis seen from the other side: a phrase must be entered on acoustics, so
boosting finishes words it cannot start. See `STT_BOOST_START_WEIGHT`.

A response carries `x-boost-applied` naming the phrases that reached the
decoder, and nothing when none did. A term can fail to get there three ways:

| | what happens |
|---|---|
| a character the model has no piece for (`日本語`, `café ☕`) | **400** naming the phrase and the character |
| shorter than `STT_BOOST_MIN_PHRASE_CHARS` (default 4) | dropped, absent from `x-boost-applied` |
| over `STT_BOOST_MAX_PHRASES` (default 200) | dropped, absent from `x-boost-applied` |

The knobs, all with measured defaults for **Parakeet TDT 0.6B v3 at int8**.
They are calibrated against that model's raw logit scale and do not transfer to
another model or quantisation:

| variable | default | what it does |
|---|---|---|
| `STT_BOOST` | `0` | the default for requests that do not send `boost` |
| `STT_BOOST_WEIGHT` | `3.0` | bonus per character for a token that CONTINUES a match. **The measured optimum**: the WER curve peaks here and decays either side, and 3.0 is the largest weight whose confidence interval still excludes zero (1.0 → −2.6%, 2.0 → −4.1%, 3.0 → −5.2%, 4.5 → −3.7%, 6.0 → −2.2%) |
| `STT_BOOST_START_WEIGHT` | `0.0` | bonus on a phrase's FIRST token. At 0 a phrase must be entered on acoustics and is only helped to finish. **Raising it is now measured, and it is the one setting that ruins the transcript**: at 1.5 it lifts term recall 0.908 → 0.944, and on a realistic list — the right terms plus ninety wrong ones — WER goes **+81%** [+49, +122]. At 3.0 with the shipped profile, CORAA goes 0.2195 → 0.8005. The recall win is an oracle's; leave this at 0 |
| `STT_BOOST_GATE` | `6.0` | a bonus applies only to a token already within this many logits of the winner. **This is the guard, not the weight ceiling** — clamping the weight and leaving the gate open still destroyed a transcript |
| `STT_BOOST_MAX_WEIGHT` | `6.0` | hard ceiling, clamping all three above |
| `STT_BOOST_MAX_PHRASES` | `200` | bounds the collateral-damage surface, not the runtime. A full 200-phrase automaton of words absent from the audio costs +0.4% WER [−0.8, +1.7] and no measurable throughput |
| `STT_BOOST_MIN_PHRASE_CHARS` | `4` | short terms match constantly; `US` would rewrite "he told us" |

`STT_HOTWORDS=0` switches the whole thing off on both engines, and `boost=true`
is then refused by name rather than answered with an unbiased transcript.

### A request's own vocabulary

`prompt` and `keywords[]` are one-off terms for one request. They are the
specification's own way to guide the model — `prompt` is defined as text that
does exactly that — so they need no extension, and `keywords[]` is the same
list already shaped as a list. Both are read as terms, split on commas and
newlines.

```bash
curl -H "Authorization: Bearer $STT_API_KEY" \
  -F file=@clip.wav -F model=whisper-1 \
  -F 'prompt=Theoria, Ghost Pepper, Catallaxy' \
  http://localhost:8000/v1/audio/transcriptions
```

| | Parakeet | Whisper |
|---|---|---|
| joined into the decoder's `hotwords` | — | yes |
| compiled into this request's repair | yes | yes |

**What the repair half can and cannot do.** A bare term spells its own intended
form and names no wrong one, so the only rule derivable from it is over its own
letters: wherever the transcript already contains the term, spell it the way
you wrote it. That fixes `theoria` → `Theoria` and `ghost pepper` → `Ghost
Pepper`, which is what Parakeet actually gets wrong on a proper noun it heard
correctly. It does **not** turn `entropic` into `Anthropic` — nothing here
names `entropic`. Only a `heard = intended` rule in a profile does that.

Two shapes are skipped, and the skips matter more than they look:

- **A term that is already all lower case.** Its rule is a no-op at best and
  harmful at worst — matching is case-insensitive, so `sync` as a term would
  rewrite a correct sentence-initial `Sync` down to lower case. This is the
  common case, not an exotic one: `commit`, `nginx` and `kubectl` are all
  hotwords in the shipped profiles.
- **A term under three characters.** `US` would turn "he told us" into "he told
  US" on every request that named it.

**A profile's own bare hotwords are still never turned into a rewrite**, and
the asymmetry is deliberate. A profile's author can write `heard = intended`
when they want one, and both shipped files promise in their headers that a bare
term biases the decoder and never rewrites the text. A `prompt` cannot express
a replacement at all, so its choice is between the weak repair and nothing —
which is what it used to get, as a 400.

**A profile and a prompt compose.** Send both and you get both, with the
profiles' terms first and the request's own last in both halves, so a one-off
term is never dropped in favour of a server-side profile. Whatever actually
fired comes back in `x-glossary-repaired`.

Up to 500 terms, the same ceiling a profile file is held to and for the same
reason: every term is a compiled regex run against every word of the
transcript. Over it is a 400 naming the count.

#### What a `PUT` refuses, and why

A bad rule corrupts silently, so writes are validated and **nothing is written
if any line was rejected** — the response lists every rejected line with its
number and its reason, and the file on disk is untouched. A `PUT` that
half-succeeded is a failure this stack has already been bitten by three times
in other forms, and a 200 carrying a `rejected` array is trivially ignored by a
script.

| Refused | Reason | Way out |
|---|---|---|
| single-word left-hand side | `belly = Belli` rewrites every innocent "belly" | `?force=true`, or use the bare hotword form |
| duplicate left-hand side | two rules for one heard term is a conflict, not last-one-wins | fix the file |
| over 500 terms, or 64 KiB | every entry is a compiled regex run against every word of every transcript | split the profile |

```json
{"detail": {"message": "1 line(s) rejected; nothing was written. 1 term(s) would have been accepted.",
            "accepted": 1,
            "rejected": [{"line": 2, "text": "belly = Belli",
                          "reason": "'belly' is a single word, so this rule would rewrite any sentence that says it correctly. …"}]}}
```

The single-word check is deliberately blunt and says so in its own message: it
is not a dictionary lookup, because there is no word list in a slim image and
an embedded list of the commonest few thousand words would not contain "belly"
— it sits around rank 4000 — and so would pass the one case the check exists
for. It over-refuses instead (`catalaxy` is not an English word and is refused
anyway), and it under-refuses too: `my sequel = MySQL` is multi-word, is
accepted, and would eat "my sequel to the book". Nothing local to a rule can
see that.

#### Migrating from `STT_GLOSSARY`

`services/stt/glossary.txt` no longer exists, and `STT_GLOSSARY` is unset in
the image. If you set it to a file, that file is loaded as a profile named
after it — `/etc/mine.txt` becomes `mine` — and is **selected per request like
any other**, not applied to everything.

To restore the old always-on behaviour, name the profiles explicitly:

```bash
STT_GLOSSARY_DEFAULT=dictation,tech
```

That is opting in to the WER cost above on every request, including the ones
whose audio contains none of those terms. Prefer selecting per request.

For Brazilian Portuguese, `alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx`
drops in via `STT_MODEL_ID`, with `STT_MODEL` left at `parakeet`.

### Switching biasing off

`STT_HOTWORDS=0` is absolute **about the decoder, on both engines**: no
glossary hotwords from any profile, no boosting automaton on Parakeet, and the
terms in a `prompt` or `keywords[]` are dropped rather than passed to either.
A request that sends `boost=true` is refused by name. Half an off switch is
worse than none — a benchmark run that sets `prompt` would get biasing back on
exactly the requests that carried one, and measure something other than what it
thinks. It covered Whisper only until Parakeet acquired a decoder to switch
off, which would have made it exactly that half-open door on the DEFAULT
engine.

Text repair is unaffected either way, and that includes the repair a request's
own terms compile into. The switch exists so a benchmark can separate what the
vocabulary contributes from what the model does, and a rewrite applied to the
model's finished output changes neither — dropping it too would make the switch
mean something it has never claimed. `/health` reports the current setting as
`hotwords`.

## Language

Leave `STT_LANGUAGE` unset unless every clip is in one language.

Pinning the wrong language does not degrade the transcript — it **translates**
it. English speech sent with `language=pt` comes back as fluent Portuguese
that reads like a working transcript and silently is not one:

```text
spoken     Look, there is a big problem here. Small tasks and more operational...
pinned pt  Veja, há um grande problema aqui, tarefas pequenas e mais operacionais...
```

Parakeet is unaffected — it detects on its own and takes no language argument
— so this trap exists on the Whisper path only. Nothing in the response marks
it: a translated transcript reads exactly like a working one, which is the
whole problem.

On `/v1` the same field is refused under Parakeet rather than accepted. It used
to be echoed back in `verbose_json` as `"language"`, so `language=pt` on English
audio returned the English transcript alongside a claim that Portuguese had
steered it — a field the request set, reported as a property of the output.

Override per request when you do know:

```bash
curl -F file=@clip.wav -F language=en http://localhost:8000/transcribe
```

## Volume ownership

A bind mount arrives with the **host** directory's ownership, which overrides
anything the image sets. On a NAS that usually means root, and the service
runs as uid 1000.

The container handles this itself: it starts as root, takes ownership of
`/models` if it does not already have it, and drops to uid 1000 before running
anything. Nothing in the service runs as root.

To manage ownership on the host instead, chown the directory and pin the user
— the entrypoint then skips the chown entirely:

```bash
chown -R 1000:1000 /mnt/tank/apps/stt-stack/models
```

```yaml
    user: "1000:1000"
```

Named volumes need none of this; Docker creates them with the right owner.

## Limiting CPU use

Set the container's CPU limit **and** `STT_THREADS` to the same number. A
limit on its own does not help: CTranslate2 and ONNX Runtime both size their
thread pools from the host's core count, not the cgroup, so on a 22-core box
they still spawn 22 threads each and then fight for the slice they are
allowed. More threads than allotted CPU is slower than fewer, not merely
capped.

```bash
docker run -p 8000:8000 -v stt-models:/models \
  --cpus 4 -e STT_THREADS=4 \
  ghcr.io/gabrielbelli/ai-voice-stt:pre
```

Pin to specific cores when the host is shared, so the service cannot be
scheduled onto whatever else is busy:

```bash
docker run -p 8000:8000 -v stt-models:/models \
  --cpuset-cpus 0-3 -e STT_THREADS=4 \
  ghcr.io/gabrielbelli/ai-voice-stt:pre
```

Compose:

```yaml
services:
  stt:
    image: ghcr.io/gabrielbelli/ai-voice-stt:pre
    ports: ["8000:8000"]
    volumes: ["stt-models:/models"]
    environment:
      STT_THREADS: "4"
    cpuset: "0-3"
    mem_limit: 6g
volumes:
  stt-models:
```

Steady state is about 1.4 GB on Parakeet and 2.9 GB on Whisper; 6 GB leaves
room for a long clip without letting a runaway request take the host down.

Every response carries `realtime_factor`. If it drops when you raise
`STT_THREADS`, you have crossed the point where coordination costs more than
the extra cores return — around 8 on most hosts, earlier on older ones.

## Performance

Rough figures, `int8`, one 15-second clip on 4 modern cores: Parakeet returns
in about 3 seconds. Whisper large-v3 runs at 0.5–0.9× realtime on the same CPU
and dominates any wait it is part of, which is why the default engine is the
other one. Choosing Whisper buys decode-time vocabulary and clean-read-speech
accuracy, and costs an order of magnitude in latency.

## Shared code

The API keys and the 401, OpenAI's error envelope, the `/health` route, the log
level switch and the container entrypoint all come from
[packages/common](../../packages/common/README.md), which this service shares
with `services/tts` and `services/tts-long`. Each of those used to be a
hand-vendored copy in every repo; the three copies of `auth.py` alone differed
by 170–197 lines and had drifted into three *different* defects, one per copy.

It is a **path dependency** in `requirements.txt` — `./packages/common`, from
the working tree. It used to be a tarball pinned by SHA, which meant a change
there did not reach this service until it was pushed, tagged and re-pinned.
One repository removes that gap: the commit is the pin. What keeps it honest is
`tests/test_conformance.py`, which runs the suite the package ships against the
app this service actually builds, so a bad change to the shared code fails in
CI rather than in production.

`app/errors.py` is gone, and it existed because of the old pin. The
specification requires `param` on every error and an envelope on 404 and 405,
`voice_common.errors` emitted neither, and a pin cannot name a commit that has
not been published — so this service completed the envelope on its own. All of
it is upstream now. Two things this service used to answer changed with the
move, both under `/v1` and both because the three copies had disagreed: a 405
now reads `Invalid URL (GET …)` rather than `Invalid method (GET …)`, which is
the wording the other two chose and the one the real API sends, and it carries
`code: "method_not_allowed"` where it used to carry null — a 404 and a 405 now
share a message shape, so `code` is what tells them apart.
`tests/test_parity.py` asserts the result, along with every other field on the
compatible surface, against a recogniser that is not a model, so CI runs all of
it without downloading 460 MB.

Install from the repository root — pip resolves `./packages/common` against the
working directory — and run pytest from here, because `import app.main` needs
this directory as the rootdir:

```bash
pip install -r services/stt/requirements.txt './packages/common[conformance]'
cd services/stt && python -m pytest tests -q
```

Everything particular to speech-to-text stays here: the recognisers, the VAD,
the glossary, the 16 kHz input rule and both routes.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
