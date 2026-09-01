# tts-long

Long-form text-to-speech. Chatterbox on CPU, as a **job queue**.

Sibling of [tts-stack](https://github.com/gabrielbelli/tts-stack), which runs
Kokoro and answers requests directly. This one cannot: it is roughly twenty
times slower and twenty times heavier, so it takes work and hands back an id.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Why a queue and not an endpoint

Measured on an M2 Max, CPU, `exaggeration=0.3 cfg_weight=0.3 temperature=0.6`:

| Threads | rtf (en) | rtf (pt) | Peak RSS |
|---|---|---|---|
| 4 | 0.213× | 0.212× | 6.8 GB |
| 8 | 0.208× | 0.187× | 6.6 GB |
| 16 | 0.217× | **0.223×** | 6.5 GB |

**Threads do not help.** From 4 to 16 the rate moved under 5% — autoregressive
token generation is sequential, so cores cannot parallelise it.

At ~0.21× realtime a ten-minute recording takes about 45 minutes. An HTTP
request waiting for that would time out long before the audio existed.

Against its sibling:

```text
Kokoro       4.1x realtime,  0.33 GB     tts-stack, answers requests
Chatterbox   0.21x realtime, 6.6 GB      here, answers with a job id
```

## Run

```bash
docker run -p 8002:8002 \
  -v tts-long-models:/models -v tts-long-out:/output \
  --cpus 8 -e TTS_THREADS=8 --memory 10g \
  ghcr.io/gabrielbelli/tts-long:pre
```

First job downloads ~3 GB of weights. The model **loads lazily and unloads
after ten minutes idle** — 6.5 GB resident is not something to leave sitting
on a shared host between jobs.

```bash
# submit
curl -s -X POST localhost:8002/jobs -H 'content-type: application/json' \
  -d '{"segments":[
        {"text":"Three steps.","pause_after":0.75},
        {"text":"One. Open your config file.","pause_after":0.75}]}'
# {"id":"...","status":"queued","estimated_seconds":38}

# poll
curl -s localhost:8002/jobs/<id>

# collect
curl -s localhost:8002/jobs/<id>/audio --output out.wav
```

## Segments and pauses

Same contract as tts-stack. `segments` carries explicit `pause_after` values,
and the silence is **generated here, not asked of the model** — no TTS model
reliably produces a beat you can act inside. Punctuation buys a breath; an
instruction needs a gap.

## OpenAI-compatible API

`POST /v1/audio/speech` accepts OpenAI's request shape, on the same queue.

**It is synchronous only where it can honestly be.** At 0.21× realtime a long
input cannot come back inside an HTTP request, so the endpoint splits:

| Input | Response |
|---|---|
| ≤ `TTS_OPENAI_SYNC_MAX_CHARS` (300) | Blocks, then **200** with the audio |
| Longer, or the wait ran out | **202** with a job id and a `Location` header |

Roughly 14 characters become a second of speech, and a second of speech costs
about 4.8 seconds of CPU — a third of a second per character. 300 characters
is therefore around 100 seconds of work, inside openai-python's 600 s default.

A 202 does not cancel anything. The job is queued, `Location` points at
`/jobs/<id>`, and the audio arrives at `/jobs/<id>/audio` as usual.

```bash
# short: returns the audio
curl -s localhost:8002/v1/audio/speech \
  -H 'authorization: Bearer sk-your-key' \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Open your config file."}' \
  --output out.wav

# long: returns a job
curl -si localhost:8002/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"<two thousand words>"}'
# HTTP/1.1 202 Accepted
# location: /jobs/6f0c...
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="sk-your-key")

client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Open your config file.",
    response_format="wav",
    # Not OpenAI fields. The OpenAI shape has no room for the knobs that
    # decide how this reads, so they go through extra_body.
    extra_body={"exaggeration": 0.3, "cfg_weight": 0.3, "language": "en"},
).stream_to_file("out.wav")
```

> openai-python cannot read a 202 — it will write the JSON body into your
> `.wav`. Keep input under the threshold, raise the threshold and the client's
> timeout together, or use `/jobs`.

### Which route to prefer

**Prefer `/jobs`.** The OpenAI shape has no field for most of what this
service knows, and the compatibility route drops it:

| Native `/jobs` | `/v1/audio/speech` |
|---|---|
| `segments` with explicit `pause_after` | Flat `input` only |
| `realtime_factor`, `compute_seconds`, `audio_seconds` | — |
| `queued_ahead`, `estimated_seconds` on every poll | Only on a 202 |
| Always returns a job, never blocks | Blocks under the threshold |

`/v1/audio/speech` exists so an OpenAI client works unmodified. It is the
lossy route, not the good one.

### What is accepted and what is not

| Field | Behaviour |
|---|---|
| `input` | Required |
| `model` | Accepted, ignored — there is one model here |
| `voice` | **Accepted and ignored.** Chatterbox clones from a reference clip and has no named voices; mapping `alloy` onto something would be an invention |
| `response_format` | `wav` (default), `flac`, `pcm`. `mp3`/`opus`/`aac` return 400 — no encoder beyond libsndfile is in the image |
| `speed` | Only `1.0`. Anything else returns 400: Chatterbox has no rate control, and resampling would shift pitch |

`pcm` is OpenAI's raw 24 kHz 16-bit mono little-endian, which is exactly what
Chatterbox emits — no resampling involved.

Errors use OpenAI's envelope, so openai-python raises something readable:

```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "invalid_api_key"}}
```

## Authentication

Set `TTS_API_KEYS` to a comma-separated list of accepted keys. Send one as
`Authorization: Bearer <key>` — what OpenAI clients already do.

```bash
docker run -p 8002:8002 -e TTS_API_KEYS='sk-alpha,sk-beta' ... tts-long
```

**Unset or empty means authentication is disabled**, and the startup log says
so at WARNING:

```text
WARNING TTS_API_KEYS is unset: every request is accepted without a key, including /v1/audio/speech
```

That is deliberate. This already runs on a LAN with callers that have no key,
and an upgrade that started refusing them would turn a feature into an outage.
Refusing to boot is the tidier position and the worse one.

Keys are compared with `hmac.compare_digest`, and every configured key is
compared even after one matches — short-circuiting would leak which key was
presented through the response time. A LAN is not a threat-free network.

`/health` stays open, because the container healthcheck calls it and has no
key. Everything else needs one, `/docs` and `/openapi.json` included.

## TLS

Set both `TTS_TLS_CERT` and `TTS_TLS_KEY` to PEM paths and uvicorn serves
HTTPS. Set neither and it serves plain HTTP, as before. Setting one alone logs
a warning and serves plain HTTP — a certificate without its key would
otherwise fail at bind time with the container already reported healthy.

```bash
docker run -p 8002:8002 \
  -v /etc/ssl/tts:/certs:ro \
  -e TTS_TLS_CERT=/certs/fullchain.pem \
  -e TTS_TLS_KEY=/certs/privkey.pem \
  -e TTS_API_KEYS='sk-alpha' \
  ghcr.io/gabrielbelli/tts-long:pre
```

**Nothing generates a certificate.** A cert that appears by magic is a cert
nobody validates — it teaches every client on the network to pass `--insecure`
permanently, and then the next one is not checked either. Use a real one from
your internal CA or Let's Encrypt, or terminate TLS in a reverse proxy in
front of this and leave the container on HTTP.

The flags are assembled in `entrypoint.sh`, so this applies to the image. If
you run uvicorn directly, pass `--ssl-certfile` and `--ssl-keyfile` yourself.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TTS_THREADS` | `8` | Match your CPU limit, but expect little from raising it |
| `TTS_IDLE_TIMEOUT` | `600` | Seconds before the 6.5 GB model is unloaded |
| `TTS_EXAGGERATION` | `0.3` | Stock is 0.5 and reads as over-cheerful |
| `TTS_CFG_WEIGHT` | `0.3` | Lower is slower, more deliberate |
| `TTS_TEMPERATURE` | `0.6` | Stock 0.8 varies more than an explanation wants |
| `TTS_API_KEYS` | *(unset)* | Comma-separated accepted keys. Unset means **no auth** |
| `TTS_TLS_CERT` | *(unset)* | PEM certificate. Both this and the key are needed for HTTPS |
| `TTS_TLS_KEY` | *(unset)* | PEM private key |
| `TTS_OPENAI_SYNC_MAX_CHARS` | `300` | Longest `/v1/audio/speech` input answered synchronously; `0` always returns 202 |
| `TTS_OPENAI_SYNC_TIMEOUT` | `180` | Seconds to wait before giving up and returning 202 instead |

Chatterbox's shipped defaults (0.5 / 0.5 / 0.8) are tuned for expressive
delivery. For instructions and explanations they sound performed. These are
calmer; raise them if you want more life.

## One job at a time

Deliberate. The model is 6.5 GB and generation is sequential, so a second
concurrent job would double the memory and slow both.

## Torch, and why CPU

Chatterbox has no ONNX build, so torch is unavoidable. The **CPU wheel** is
used on purpose: the CUDA wheels add several gigabytes, and the GPU this would
otherwise target is a GTX 1060 — Pascal, whose FP16 runs at 1/64 rate. It
would not help even where one exists, and 6.5 GB does not fit in 6 GB of VRAM
at fp32 regardless.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
