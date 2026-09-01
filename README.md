# stt-stack

Self-hosted speech-to-text. One container, whole pipeline, CPU only.

```text
audio
  ↓  VAD          Silero — drop silence first
  ↓  recogniser   Parakeet TDT 0.6B v3 by default, Whisper large-v3 on request
  ↓  glossary     repair known terms
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
and when you need decode-time vocabulary: it accepts `hotwords`, Parakeet does
not. For Parakeet the glossary is post-decode repair only, which is weaker —
it cannot recover a word the acoustic model never approached.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
docker run -p 8000:8000 -v stt-models:/models \
  --cpus 4 -e STT_THREADS=4 \
  ghcr.io/gabrielbelli/stt-stack:pre
```

First start downloads the selected model into the volume. Later starts are
immediate.

```bash
curl -F file=@clip.wav http://localhost:8000/transcribe
```

```json
{
  "text": "I need to make a commit on the Theoria dashboard",
  "raw": "I need to make a comet on the theory dashboard",
  "repaired": ["Theoria dashboard", "commit"],
  "model": "parakeet",
  "audio_seconds": 4.1,
  "speech_seconds": 3.2,
  "compute_seconds": 2.4,
  "realtime_factor": 1.7
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
{"text": "I need to make a commit on the Theoria dashboard",
 "usage": {"type": "duration", "seconds": 4}}
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")

with open("clip.wav", "rb") as clip:
    print(client.audio.transcriptions.create(model="whisper-1", file=clip).text)
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
| `language`, `prompt`, `keywords[]`, `temperature` | Honoured on Whisper, refused on Parakeet |
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
| `prompt`, `keywords[]` | refused — no decode-time vocabulary | joined into hotwords |
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
| `repaired` — which glossary terms fired | yes | — |
| `raw` — the transcript before repair | yes | — |
| `realtime_factor` and the timings behind it | yes | — |
| Segments, words, subtitles, streaming | — | yes |
| Works with an unmodified OpenAI client | — | yes |

`repaired` is the one worth caring about. A silent substitution is worse than
no substitution: if a glossary rule is wrong, you want to see it named, and on
the compatible route you cannot.

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

**No streaming, translation, `language`, `prompt`, `keywords[]` or
`temperature` under Parakeet.** All six are refusals, not silences, and all six
are properties of a TDT decoder that has no such mechanism. Deploy with
`STT_MODEL=whisper` if you need them, and pay the order of magnitude in
latency.

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

**Glossary repair happens on strings.** It is applied to the whole transcript,
and separately to each segment and each word, so a rule spanning a boundary —
`cloud code` split across two segments — fires in `text` and cannot fire inside
the smaller unit. `/transcribe` names every rule that fired in `repaired`.

**`seek` is not remapped.** With the VAD on, the recogniser sees speech runs
concatenated, and every start and end is mapped back onto the clip you sent.
`seek` indexes a decoder window rather than naming a moment, so it is passed
through as the decoder reported it.

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
  ghcr.io/gabrielbelli/stt-stack:pre
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
| `STT_GLOSSARY` | `/etc/stt-stack/glossary.txt` | See below |
| `STT_API_KEYS` | unset | Comma-separated accepted keys. Unset means no auth |
| `STT_LOG_LEVEL` | `INFO` | `DEBUG`, `WARNING`, … An unrecognised value falls back to `INFO` |
| `STT_TLS_CERT` | unset | PEM certificate. With `STT_TLS_KEY`, serves HTTPS |
| `STT_TLS_KEY` | unset | PEM private key, readable by uid 1000 |

### Glossary

Two line forms, because they are two different jobs:

```text
catalaxy = Catallaxy    a replacement AND a hotword
Catallaxy               a hotword only
```

Use the bare form when the likely mishearing is an ordinary word. "Belli" is
heard as "belly", but a `belly = Belli` rule would corrupt any sentence that
genuinely says belly. Biasing the decoder is safe; rewriting is not.

Decoder biasing is much the stronger of the two. Measured against real
recordings, hotwords alone fixed every technical term — `commit` (heard as
"comet"), `Theoria` ("theory"), `FreeBSD` ("free BSD"), `Belli` ("Belly") —
and the post-decode replacement never had to fire. It can also recover a word
string replacement never sees, because the wrong spelling was never in the
list.

For Brazilian Portuguese, `alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx`
drops in via `STT_MODEL_ID`, with `STT_MODEL` left at `parakeet`.

### Switching biasing off

`STT_HOTWORDS=0` is absolute: no glossary hotwords, and a `prompt` sent to
`/v1/audio/transcriptions` is dropped rather than passed to the decoder. Half
an off switch is worse than none — a benchmark run that sets `prompt` would
get biasing back on exactly the requests that carried one, and measure
something other than what it thinks. Text repair is unaffected either way, and
`/health` reports the current setting as `hotwords`.

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
  ghcr.io/gabrielbelli/stt-stack:pre
```

Pin to specific cores when the host is shared, so the service cannot be
scheduled onto whatever else is busy:

```bash
docker run -p 8000:8000 -v stt-models:/models \
  --cpuset-cpus 0-3 -e STT_THREADS=4 \
  ghcr.io/gabrielbelli/stt-stack:pre
```

Compose:

```yaml
services:
  stt:
    image: ghcr.io/gabrielbelli/stt-stack:pre
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
[voice-common](https://github.com/gabrielbelli/voice-common), which this
service shares with `tts-stack` and `tts-long`. Each of those used to be a
hand-vendored copy in every repo; the three copies of `auth.py` alone differed
by 170–197 lines and had drifted into three *different* defects, one per copy.

It is pinned by SHA in `requirements.txt`, so a change there does not force a
rebuild here. What keeps the pin honest is `tests/test_conformance.py`: the
package ships its own pytest suite and this repo runs it against the app it
actually builds, so a bad bump fails at the build rather than in production.

That pin is also why `app/errors.py` exists. The specification requires `param`
on every error and an envelope on 404 and 405, and voice-common emits neither
yet; a fix there is not a fix here until the pin moves, so this service
completes the envelope on its own — over the shared code rather than instead of
it. `tests/test_parity.py` asserts the result, along with every other field on
the compatible surface, against a recogniser that is not a model, so CI runs
all of it without downloading 460 MB.

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt pytest httpx
./.venv/bin/python -m pytest tests -q
```

Everything particular to speech-to-text stays here: the recognisers, the VAD,
the glossary, the 16 kHz input rule and both routes.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
