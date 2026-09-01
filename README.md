# tts-stack

Self-hosted text-to-speech. Kokoro, CPU only, no torch.

```text
text or segments  →  /speak             native: pauses, realtime factor
text              →  /v1/audio/speech   OpenAI's shape
  ↓  espeak-ng    phonemise
  ↓  Kokoro-82M   ONNX Runtime, CPU
  ↓  wav  opus  mp3  aac  flac  pcm
```

Sibling of [stt-stack](https://github.com/gabrielbelli/stt-stack), same
conventions.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 \
  ghcr.io/gabrielbelli/tts-stack:pre
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
| `model` | Accepted and ignored — there is one model here. Rejecting `tts-1` would break every client that sends it; claiming to honour it would be a lie |
| `input` | Required |
| `voice` | The six OpenAI names, **or** any of the 54 Kokoro names. Optional, unlike upstream: `TTS_VOICE` answers when it is absent |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`. `pcm` is headerless 24 kHz 16-bit mono, which is Kokoro's own rate — nothing is resampled |
| `speed` | Accepted across OpenAI's 0.25–4.0 and **clamped to 0.5–2.0**, where Kokoro's duration predictor still tracks punctuation. A client asking for 4.0 wants fast speech, not a 400 it cannot act on |

Unknown fields are accepted and ignored here, unlike on `/speak`. OpenAI keeps
adding them — `instructions`, `stream_format` — and a service whose entire
purpose is to answer OpenAI's clients must not reject one for speaking a newer
version of the dialect it claims to speak.

Errors come back in OpenAI's envelope, `{"error": {"message", "type", "code"}}`,
because `openai-python` reads `error.message` and renders FastAPI's
`{"detail": …}` as a bare status code. The native routes keep FastAPI's shape:
clients are already written against it. That covers a failure to encode as well
as a failure to synthesise: `ffmpeg` missing or failing is a `500` with a
message on both routes, in each route's own shape.

### Which route to prefer

**`/speak`, unless a client you do not control is asking for the other one.**

| | `/speak` | `/v1/audio/speech` |
|---|---|---|
| Segments with pauses | yes | no field for it |
| Per-segment voice | yes | no |
| Unknown fields | `422` | accepted and ignored |
| `language` override | yes | inferred from the voice |
| `X-Realtime-Factor` | yes | sent, but clients ignore it |
| Speed range | 0.5–2.0, rejected outside | 0.25–4.0, clamped |

The OpenAI body cannot express a pause, and pauses are the thing that makes
audio sound like instructions rather than narration. `/v1/audio/speech` is a
translation layer over the same synthesiser, not a second interface.

### Voice names

The five names Kokoro borrowed from OpenAI map straight through. `shimmer` has
no counterpart in the model, so that row is mapped by ear and is the only
judgement call in the table.

| OpenAI | Kokoro | |
|---|---|---|
| `alloy` | `af_alloy` | en-US |
| `echo` | `am_echo` | en-US |
| `fable` | `bm_fable` | en-GB |
| `onyx` | `am_onyx` | en-US |
| `nova` | `af_nova` | en-US |
| `shimmer` | `af_bella` | en-US, mapped by ear |

Native Kokoro names take precedence, so `af_nova` reaches `af_nova` whatever
the table says, and `GET /voices` returns the live table under
`openai_aliases`.

The phonemiser language is inferred from the voice prefix, because OpenAI's
body has no language field — asking for `fable` and getting a British voice
read as American would be worse than inferring. `/speak` still takes an
explicit `language`.

## Authentication

Off by default. Set `TTS_API_KEYS` to a comma-separated list to turn it on:

```bash
docker run -p 8001:8001 -v tts-models:/models \
  -e TTS_API_KEYS=sk-workstation,sk-laptop \
  ghcr.io/gabrielbelli/tts-stack:pre
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
  ghcr.io/gabrielbelli/tts-stack:pre
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

## Limiting CPU use

Set the container's CPU limit **and** `TTS_THREADS` to the same number. ONNX
Runtime sizes its thread pool from the host's core count, not the cgroup, so a
`--cpus` limit alone leaves the container spawning a thread per host core and
then contending for the slice it is allowed — slower than simply using fewer
threads.

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 --memory 2g \
  ghcr.io/gabrielbelli/tts-stack:pre
```

Steady state is about 400 MB, so 2 GB is generous.

Every response carries `X-Realtime-Factor`. If it drops when you raise
`TTS_THREADS`, coordination is costing more than the extra cores return.

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
[voice-common](https://github.com/gabrielbelli/voice-common), pinned to a
commit in `requirements.txt` and shared with stt-stack and tts-long.

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
`0.5`–`2.0` speed clamp, the segment `voice` field, and every route.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
