# tts-long

Long-form text-to-speech. Chatterbox on CPU, as a **job queue** and an
**SSE stream**.

Sibling of [services/tts](../tts/README.md), which runs
Kokoro and answers requests directly. This one cannot: it is roughly twenty
times slower and twenty times heavier. So it either streams the audio as it is
made — `stream_format: "sse"`, OpenAI's own shape — or takes the work and
hands back an id.

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

Re-measured on the deployed instance on 2026-09-01, which is where the numbers
in the rest of this file come from:

```text
65 characters, one string    ->   6.6 s of audio in  21.9 s   (0.303×)
1690 characters, one string  ->  40.0 s of audio in 184.6 s   (0.217×)
1690 characters, 20 segments -> 100.2 s of audio in 338.1 s   (0.296×)
```

Read the last two rows together. **The same 1690 characters produced 40.0
seconds of audio as one string and 100.2 seconds as twenty segments** — see
below.

Speech rate across those samples runs from 9.8 characters per second (one short
sentence, where the silence at each end dominates) to 19.3 (a 336-character
passage). Estimates here use 15 and say so; the chunk ceiling is checked
against the slowest of them, because that is the direction that truncates.

Against its sibling:

```text
Kokoro       4.1x realtime,  0.33 GB     tts-stack, answers requests
Chatterbox   0.21x realtime, 6.6 GB      here, streams, or answers with a job id
```

### The 40-second ceiling, and why everything is chunked now

Look at the second measurement again. 1690 characters is about 170 seconds of
speech. It came back as **exactly 40.0 seconds**, with no error and no warning.

`generate()` stops after 1000 speech tokens (chatterbox-tts 0.1.7,
`chatterbox/mtl_tts.py:297`) and S3 speech tokens run at 25 Hz
(`chatterbox/models/s3tokenizer/s3tokenizer.py:18`). One call therefore cannot
produce more than forty seconds of audio however much text it is given, and
this service used to hand it the whole input. Anything longer was **silently
truncated** — the caller got the first part of their text and a file that ends
mid-sentence, after paying for the whole thing in CPU.

The third measurement above is the same text through the segment path, which
was never truncated: **100.2 seconds instead of 40.0**, two and a half times as
much speech from the same words.

Every input is now split into sentence-sized chunks and the pieces are spliced,
on `/jobs` and on `/v1/audio/speech` alike. `app/chunking.py` carries the
reasoning; `chunks` in the job body says how many a request became.
## Run

```bash
docker run -p 8002:8002 \
  -v tts-long-models:/models -v tts-long-out:/output \
  --cpus 8 -e TTS_THREADS=8 --memory 10g \
  ghcr.io/gabrielbelli/ai-voice-tts-long:pre
```

First job downloads ~3 GB of weights. The model **loads lazily and unloads
after ten minutes idle** — 6.5 GB resident is not something to leave sitting
on a shared host between jobs.

**The image carries no `HEALTHCHECK` instruction**, and one here would not
survive the build: `HEALTHCHECK` is not a field in the OCI image spec, and CI
builds with buildah, whose default format is `oci`, so the instruction is
dropped silently on the way to the registry and `docker inspect` on a published
image shows none. The probe is passed at run time instead, which is where all
four images have theirs. [`compose.yaml`](../../compose.yaml) at the repository
root carries it, and that copy is the only one there is — delete it and this
container has no probe.

Its `start_period` there is **900 s**, not the two minutes an ordinary service
would want, because the first job downloads ~3 GB of weights before it can
answer anything; a shorter grace kills the container mid-download and the next
one starts the download again. `/health` itself is answered on the event loop
and never waits on the queue, so a service that is merely busy still reports
healthy.

```bash
# submit
curl -s -X POST localhost:8002/jobs -H 'content-type: application/json' \
  -d '{"segments":[
        {"text":"Three steps.","pause_after":0.75},
        {"text":"One. Open your config file.","pause_after":0.75}]}'
# {"id":"...","status":"queued","chunks":2,"estimated_seconds":38}

# poll
curl -s localhost:8002/jobs/<id>

# collect
curl -s localhost:8002/jobs/<id>/audio --output out.wav

# cancel a queued job, or discard a finished one and its file
curl -s -X DELETE localhost:8002/jobs/<id>
```

A queued job stops immediately; a running one stops at its next chunk
boundary, because `generate()` has no interruption point inside it. Finished
jobs and their audio are swept after `TTS_JOB_TTL` — until now nothing was ever
removed and `/output` grew for the lifetime of the process.

`jobs` is a dict in one process, so a restart empties it while the audio sits
in the volume and survives. Finished jobs are rebuilt from `/output` at
startup, and each one writes a **metadata sidecar** — `{id}.json` beside
`{id}.{format}` — so what comes back is the whole row and not a filename.
Before it, the voice, the language, the generation parameters, the chunk count
and the realtime factor lived only in the dict and every recovered job read
*voice unknown*. Both files are removed together, by `DELETE` and by the sweep.
A job finished before this release has no sidecar and still recovers with what
its name carries; the record says `recovered: true` either way and never
invents a voice it does not know.

## Segments and pauses

Same contract as tts-stack — the same class, now, rather than the same idea
written out twice: `text` and a `pause_after` of 0.0 to 10.0 seconds come from
`voice_common.models.Segment`, so the published range cannot drift in one
service and not the other.

The silence is **generated here, not asked of the model** — no TTS model
reliably produces a beat you can act inside. Punctuation buys a breath; an
instruction needs a gap.

A field that is not `text` or `pause_after` is now a **422** rather than
silently ignored. tts-stack documented a per-segment `voice` for as long as it
was quietly dropped and gave the caller the default voice with nothing to read
that said why; a typo belongs in an error, not in the audio.

## OpenAI-compatible API

`POST /v1/audio/speech` accepts OpenAI's request shape, on the same queue, and
answers in one of three ways.

| What you send | What you get |
|---|---|
| `stream_format: "sse"` | **200**, `text/event-stream`, audio as it is generated |
| Input the arithmetic can finish in time | Blocks, then **200** with the audio |
| Anything longer, or a wait that runs out | **202** with a job id and `Location` |

### Streaming, which is the one to use

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="sk-your-key")

with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice="alloy", response_format="mp3",
        stream_format="sse", input=long_text) as response:
    for line in response.iter_lines():
        ...  # data: {"type":"speech.audio.delta","audio":"<base64>"}
```

Frames are OpenAI's: bare `data:` lines, each ending in a blank line, carrying
`{"type":"speech.audio.delta","audio":"<standard base64>"}` and finally one
`{"type":"speech.audio.done","usage":{...}}`. Comment lines (`: keepalive`)
appear while the stream is waiting and are ignored by every SSE decoder.

**It is genuinely incremental.** The first delta leaves when the first sentence
finishes generating, not when the request does. Measured through a socket with
a four-chunk request: first delta at **0.42 s of a 1.66 s** response — the
assertion in `tests/test_streaming.py` fails if the first delta ever arrives in
the last fifth of the response, which is what buffering-and-slicing would look
like.

What streaming does **not** do is make this service fast. The compute is
unchanged: 4096 characters is still around 400 seconds of speech and therefore
around half an hour of CPU. What it removes is the dead air and the client-side
timeout — a stream that keeps producing frames resets a read timeout, and the
keepalives cover the gaps between sentences. The realistic floor for the first
sound is **one sentence**, so roughly 3–8 s of audio and therefore 15–40 s of
compute. Anyone promising sub-second first audio from this model is wrong:
`generate()` is autoregressive over the whole string it is given and emits
nothing part way through.

Concurrent streams serialise. One job runs at a time, so a second caller's
stream sends keepalives until the first finishes; `queued_ahead` on `/jobs` and
`queued` on `/health` say how deep that is, and past `TTS_MAX_QUEUE` the answer
is a **429 with `Retry-After`**.

### Buffered, and the 202

```bash
# short: returns the audio
curl -s localhost:8002/v1/audio/speech \
  -H 'authorization: Bearer sk-your-key' \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Open your config file."}' \
  --output out.mp3

# long: returns a job
curl -si localhost:8002/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"<two thousand words>"}'
# HTTP/1.1 202 Accepted
# location: /jobs/6f0c...
# retry-after: 412
```

The synchronous boundary is now arithmetic rather than a constant. A request
blocks only when `TTS_OPENAI_SYNC_TIMEOUT` is enough for **this host at the
rate it is currently achieving**, less what is already in the queue, less a
model load if the model is not resident. The rate is measured from finished
jobs rather than assumed, because the same code ran at 0.217× on the deployed
instance and 0.138× on a loaded laptop, and the old fixed 0.21 turned requests
under the documented threshold into 202s with nothing to read that said why.

> openai-python treats a 202 as success and will write the JSON body into your
> `.wav`. Send `stream_format: "sse"` instead, or use `/jobs`. This is a
> deliberate deviation — see below.

### Which route to prefer

**`/v1/audio/speech` with `stream_format: "sse"` for anything interactive;
`/jobs` for batch.** The OpenAI shape has no field for most of what this
service knows:

| Native `/jobs` | `/v1/audio/speech` |
|---|---|
| `segments` with explicit `pause_after` | Flat `input` only |
| `realtime_factor`, `compute_seconds`, `audio_seconds` | — |
| `queued_ahead`, `estimated_seconds`, `chunks` when accepted | The same, but only on a 202 |
| `DELETE` to cancel or discard | Close the stream, which cancels |
| Never blocks | Blocks, streams, or hands back a job |

### Following a job while it runs

`GET /jobs/{id}` carries `offsets` **from the first segment onwards**, not only
when the job is done:

```json
{"status": "running", "chunks": 34, "offsets": [0.0, 3.1, 7.4, 11.9]}
```

Four of thirty-four segments are spoken and 11.9 seconds of audio exist. Both
are exact: each boundary is a running total of samples that were produced, not
`duration x (characters so far / characters total)`, which is wrong from the
first sentence because the pause after a segment is a fixed number of seconds
whatever its length.

At ~0.21x realtime a job runs for minutes, and before this the only progress a
poller could show was elapsed time against an estimate. `offsets[-1]` divided
by the time since `started_at` is also this job's own realtime factor, live,
which corrects the remaining estimate without waiting for the service average.

### What is accepted and what is not

Nothing is accepted and ignored. Every field is honoured or refused **by
name**, in OpenAI's error envelope with `param` set.

| Field | Behaviour |
|---|---|
| `input` | Required, 1 to 4096 characters, as the schema says |
| `model` | Accepted, ignored — there is one model here, and that is stated rather than implied |
| `voice` | Resolved against the voice registry; a string or `{"id": "..."}`. Unknown names are a **400**. See *Voices* |
| `response_format` | All six: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | Only `1.0`. Anything else is a **400**: Chatterbox has no rate control, and resampling would shift pitch |
| `instructions` | **400.** Chatterbox has no instruction conditioning; use `exaggeration`, `cfg_weight` and `temperature` through `extra_body` |
| `stream_format` | `audio` or `sse` |
| Anything else | **400**, naming the field, as OpenAI's own API answers it |

`exaggeration`, `cfg_weight`, `temperature` and `language` are vendor fields,
sent through `extra_body`, and are the only extras accepted.

Errors are OpenAI's envelope with **all four fields**, `param` included:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "voice", "code": "unsupported_value"}}
```

That includes a body rejected before it reaches any check — a missing `input`,
an `exaggeration` of 9 — and now also **404 and 405 under `/v1`**, which used
to escape as `{"detail": "Not Found"}` and tell openai-python nothing. The
native `/jobs` routes keep the `detail` list and the 422: they are the older
contract.

## Voices

Chatterbox has no named voices. It **clones from a reference clip**, so a voice
here is a file:

```bash
docker run -v /srv/voices:/voices ... ghcr.io/gabrielbelli/ai-voice-tts-long:pre
# /voices/alloy.wav  -> voice "alloy"
# /voices/gabriel.wav -> voice "gabriel"
```

`GET /voices` lists what is available; `default` is the model's own speaker and
is always there. A name that is neither a clip nor an alias is a **400**.

> **Deviation.** With no clips installed there is exactly one voice, and
> OpenAI's thirteen documented names all resolve to it. Refusing `alloy`
> outright would break every unmodified client, so instead the substitution is
> declared: every response carries `X-Voice` naming the voice actually used,
> startup logs the aliased names at WARNING, and `TTS_VOICE_STRICT=1` turns the
> aliases off and makes them 400s like any other unknown name.

## Deviations from OpenAI, and the measurements that force them

Nothing here is hidden and nothing here is faked.

| Deviation | Why | Measurement |
|---|---|---|
| Long input answers **202** with a job id instead of audio | A buffered synchronous answer is not slow, it is impossible | 4096 characters ≈ 400 s of speech ≈ 1900 s of CPU at 0.217×; openai-python's default timeout is 600 s and proxies give up at 60–120 s |
| `speed` is a **400** | Chatterbox has no rate control; resampling shifts pitch and an external time-stretch is not in this image | — |
| `instructions` is a **400** | No instruction conditioning exists in the model | — |
| The thirteen OpenAI voice names map to one voice | Chatterbox clones from clips and none ship here | Reported per response in `X-Voice`; closable by adding clips |
| Streamed `wav` and `flac` differ from the buffered file in their header | Both state the total length, which is not known until the last sentence is generated | Diffed byte for byte: `wav` differs at offsets 4 and 40 only and is identical from byte 44; `flac` differs inside STREAMINFO only (offsets 8–41). `pcm`, `mp3` and `aac` are exact |
| Streamed `opus` is not byte-comparable between requests | The Ogg serial number is random per stream, so two encodes of the same samples differ anyway | Within one request the deltas are a partition of a single encode |
| `sse` is accepted for every `model` value | OpenAI restricts `stream_format` to `gpt-4o-mini-tts`; there is one model here and refusing the field on the strength of a name would be theatre | — |
| Unknown fields are **400**, not ignored | OpenAI's schema sets `additionalProperties: false` and its API answers the same way | The cost: a genuinely new OpenAI field is refused here until it is added |
| `audio/pcm` as the content type for `response_format: "pcm"` | The schema names no per-format MIME. This is an **estate-wide decision**, written down so tts-stack, tts-long and stt-stack cannot drift: `pcm` → `audio/pcm`, `opus` → `audio/ogg`, `mp3` → `audio/mpeg`, `aac` → `audio/aac` | — |

Things that are **not** deviations any more: `param` in the error envelope, the
4096-character cap, 429 with `Retry-After`, `Transfer-Encoding: chunked`, the
absence of `Content-Disposition` on `/v1`, and mp3 as the default format.
## Authentication

Set `TTS_API_KEYS` to a comma-separated list of accepted keys. Send one as
`Authorization: Bearer <key>` — what OpenAI clients already do.

```bash
docker run -p 8002:8002 -e TTS_API_KEYS='sk-alpha,sk-beta' ... tts-long
```

**Unset means authentication is disabled**, and the startup log says so at
WARNING:

```text
WARNING TTS_API_KEYS is unset: authentication is DISABLED and every request is accepted, including /v1. Set TTS_API_KEYS to a comma-separated list of keys to require Authorization: Bearer.
```

That is deliberate. This already runs on a LAN with callers that have no key,
and an upgrade that started refusing them would turn a feature into an outage.
Refusing to boot is the tidier position and the worse one.

A `TTS_API_KEYS` that is *set* but names no key — `''`, `','`, `'  '`, `',,'`
— **refuses to start**. All four are reached by ordinary accident: `-e
TTS_API_KEYS=$SECRET` with `SECRET` unset hands the container an empty value.
Unset means "I am not using this"; a value that is present and yields nothing
means someone meant to configure keys, and reading that as "off" turns an
operator's intent to require keys into a service open to anyone. *(The empty
string used to disable authentication silently. It now exits with the sentence
above.)*

Keys are compared with `hmac.compare_digest`, and every configured key is
compared even after one matches — short-circuiting would leak which key was
presented through the response time. A LAN is not a threat-free network.

A key with non-ASCII characters in it authenticates. It is compared against
the bytes the client actually put on the wire, because Starlette decodes a
header as latin-1 and re-encoding that as UTF-8 produced different bytes for
every accented key — so the *correct* key came back as "Incorrect API key
provided". Startup warns about such a key anyway: it works only with clients
that send the header as UTF-8, which HTTP does not guarantee.

`/health` stays open, because the container healthcheck calls it and has no key
and no way to be given one — that probe lives in `compose.yaml` rather than in
the image, for the reason under [Run](#run) — and so does `/health/`, with the
trailing slash. The check runs before routing, so FastAPI's 307 to `/health`
never happens and a probe written that way used to go permanently 401 the day
keys were configured. Everything else needs a key, `/docs` and `/openapi.json`
included.

## TLS

Set both `TTS_TLS_CERT` and `TTS_TLS_KEY` to PEM paths and uvicorn serves
HTTPS. Set neither and it serves plain HTTP, as before.

**Any half-configuration refuses to start.** Setting one of the two, or
setting both while overriding `CMD` to something that is not uvicorn, used to
print a line and serve plain HTTP — so an operator who had written the TLS
variables into their compose file read them back, believed the port was
encrypted, and sent a bearer token across the LAN in cleartext on every
request. A container that will not start is noticed in seconds. Plaintext
under a configuration that claims otherwise is noticed when someone else has
the key.

```bash
docker run -p 8002:8002 \
  -v /etc/ssl/tts:/certs:ro \
  -e TTS_TLS_CERT=/certs/fullchain.pem \
  -e TTS_TLS_KEY=/certs/privkey.pem \
  -e TTS_API_KEYS='sk-alpha' \
  ghcr.io/gabrielbelli/ai-voice-tts-long:pre
```

**Nothing generates a certificate.** A cert that appears by magic is a cert
nobody validates — it teaches every client on the network to pass `--insecure`
permanently, and then the next one is not checked either. Use a real one from
your internal CA or Let's Encrypt, or terminate TLS in a reverse proxy in
front of this and leave the container on HTTP.

The flags are assembled in `voice-entrypoint.sh`, which
[packages/common](../../packages/common/README.md) installs into
`/usr/local/bin`, so this applies to the image — and only to the uvicorn
command it knows how to add them to. If you run uvicorn directly, pass
`--ssl-certfile` and `--ssl-keyfile` yourself.

The certificate and the key are also checked for readability **as uid 1000**,
the user the server drops to, rather than as root. A key mounted `0600
root:root` reads fine to the entrypoint and not at all to the process that
opens it, and uvicorn's failure at that point is a traceback rather than a
sentence.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TTS_THREADS` | `8` | Match your CPU limit, but expect little from raising it |
| `TTS_IDLE_TIMEOUT` | `600` | Seconds before the 6.5 GB model is unloaded |
| `TTS_EXAGGERATION` | `0.3` | Stock is 0.5 and reads as over-cheerful |
| `TTS_CFG_WEIGHT` | `0.3` | Lower is slower, more deliberate |
| `TTS_TEMPERATURE` | `0.6` | Stock 0.8 varies more than an explanation wants |
| `TTS_API_KEYS` | *(unset)* | Comma-separated accepted keys. Unset means **no auth**; set but naming no key (`''`, `','`) refuses to start |
| `TTS_LOG_LEVEL` | `INFO` | Root log level. An unrecognised name warns and falls back to `INFO` rather than refusing to start |
| `TTS_TLS_CERT` | *(unset)* | PEM certificate. Both this and the key are needed for HTTPS; half a pair refuses to start |
| `TTS_TLS_KEY` | *(unset)* | PEM private key. Only applied to the `uvicorn` command; overriding `CMD` with TLS set refuses to start |
| `TTS_VOICE_DIR` | `/voices` | Reference clips. `<name>.wav` becomes voice `<name>`. Rescanned when the directory changes, so a clip copied in is usable on the next request rather than after a restart |
| `TTS_VOICE_STRICT` | *(off)* | Refuse OpenAI voice names that have no clip, instead of aliasing them to `default` |
| `TTS_OPENAI_SYNC_MAX_CHARS` | `300` | Hard ceiling on input answered synchronously; `0` always returns 202 |
| `TTS_OPENAI_SYNC_TIMEOUT` | `180` | Seconds to wait before giving up and returning 202 instead |
| `TTS_REALTIME_FACTOR` | `0.21` | Seed for the measured rate. Corrected by every job that finishes |
| `TTS_COLD_LOAD_SECONDS` | `60` | Charged against the synchronous budget when the model is not resident |
| `TTS_CHARS_PER_SECOND` | `15` | Measured between 9.8 and 19.3. Only affects estimates and chunk sizing |
| `TTS_CHUNK_MAX_CHARS` | `280` | Hard ceiling per `generate()` call; must stay under 40 s of speech |
| `TTS_CHUNK_TARGET_CHARS` | `160` | Short sentences merge up to this. Lower means sooner first audio and choppier prosody |
| `TTS_MAX_QUEUE` | `32` | Queue depth past which both routes answer **429** with `Retry-After` |
| `TTS_JOB_TTL` | `86400` | Seconds a finished job and its audio survive. `0` keeps them for the process lifetime, as before |
| `TTS_SSE_KEEPALIVE` | `10` | Seconds between `:` comment lines on a waiting stream |

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

## Borrowing somebody's GPU, optionally

Chatterbox is the only component in this stack slower than realtime, which is
the entire reason this section exists. **It is off by default and unset means
local-only**: with no `TTS_RUNNER_HOST`, every job runs on this CPU and
`app/remote.py` is imported and never used.

The other end is an `idlegpu` agent (its own repository, not yet published): a
small program on a machine with a spare GPU, most likely somebody's gaming PC, that runs work while
nobody is using the card and hands it straight back when they are. There is no
broker and no cluster. This service talks to that machine directly over TLS.

```yaml
TTS_RUNNER_HOST: "192.0.2.11"      # example only
TTS_RUNNER_FINGERPRINT: ""         # `idlegpu fingerprint` on that machine
TTS_RUNNER_API_KEY_FILE: /run/secrets/runner-key
```

### What it buys, measured

Chatterbox on an RTX 3070: **0.56 to 0.72x realtime**, against **0.275x** on
this container's eight threads. Two and a half times faster on the slowest thing
in the stack. Worth having, and deliberately written down rather than rounded up
to the order of magnitude somebody might assume.

### What it costs

The GPU goes away, several times an evening, whenever its owner sits down. That
is the runner working correctly rather than failing: it yields in **under three
seconds**, measured with the model resident, and the job comes back as
**queued** rather than `failed`. This service waits `TTS_RUNNER_MAX_WAIT` and
then speaks it on the CPU instead.

`app/synth.py` is therefore never going anywhere. It is the fallback for exactly
the case the runner is designed to create.

### Nothing is ever spoken twice

The obvious response to the GPU going away is to retry, and it is how the same
sentence gets generated twice. A short yield is **waited out** rather than
retried, which closes the three routes to that at once: the lease on the runner
survives under an idempotency key that is the local job id, so it does not
regenerate what it already made; the client does not collect an artefact it
already holds; and `_run` is never re-entered, so no encoder is rebuilt and no
open SSE stream is sent a second file header half way through.

Only the bounded wait running out ends the remote attempt, and the local re-run
that follows happens **only when nothing has been sent to a stream yet**. A job
that had already streamed audio is told what happened instead of being replayed
from the beginning.

### The three things that stay here

**Encoding**, because `test_wav_differs_only_in_the_two_riff_size_fields` and
`test_flac_differs_only_inside_streaminfo` are byte-exact facts about *this*
container's ffmpeg. Moving encoding to a stranger's Windows box would make them
facts about a build nobody here controls.

**Chunking**, because the 40-second `generate()` ceiling was measured on this
stack and belongs to it. The runner is sent already-segmented text and carries no
text policy at all.

**The file write.** `OUT_DIR / f"{job['id']}.{job['format']}"` is what `/jobs`,
`_recover`, `_discard` and the sweeper all reach through. Nothing the runner says
about where a file is ever reaches that field.

### And the one that would be easy to get wrong

**The realtime-factor EMA is per backend.** `rate` decides whether an incoming
request is answered synchronously or handed a 202. A GPU at 0.7x and this CPU at
0.275x sharing one average describes no machine that exists, and `_sync_budget`
would then accept a synchronous request the CPU cannot finish — at exactly the
moment the GPU disappears, because that is when jobs come back here. `/health`
reports `realtime_factor_by_backend` for the same reason.

`tests/test_remote.py` covers all of this with a fake runner and opens no socket.

## Shared code

Authentication, the OpenAI error envelope, the `/health` contract, `Segment`,
the PCM byte cast and the entrypoint all live in
[packages/common](../../packages/common/README.md), installed as a path
dependency in `requirements.txt` and shared with `services/tts` and
`services/stt`. Three hand-vendored copies of that code had drifted by 170 to
197 lines and carried three *different* bugs, two of which were this service's:
a key with an accent in it could never authenticate, and `GET /health/`
answered 401.

`tests/test_conformance.py` is four lines and runs the suite the package ships
against this service's own `app.main:app`, so a change to the shared code fails
here rather than on the host. The rest of `tests/` is this service's own: the
chunker, the encoders' byte-for-byte identity, and the SSE wire format. None of
it needs a model or torch — `Synth._speak` is the single method faked, so the
routes, the queue, the chunker and the encoders under test are the real ones:

Install from the repository root, because pip resolves `./packages/common`
against the working directory; run pytest from here, because `import app.main`
needs this directory as the rootdir:

```bash
pip install './packages/common[audio,conformance]' fastapi soundfile 'uvicorn[standard]'
cd services/tts-long && python -m pytest tests
```

`app/envelope.py` is gone. It was `voice_common.errors` plus the `param` field
OpenAI's schema requires and the 404/405/500 handlers the shared package had no
equivalent of, and it was here rather than upstream only because of the pin — a
commit SHA on a GitHub tarball cannot name a commit that has not been
published. There is no pin now, so all of it moved up, tests included. It was
the best of the three vendored copies and it is the one the shared code was
built from: the union-branch collapse, the `extra_forbidden` wording, the 500
handler and the `.items()` that keeps a 405's `Allow` header all came from
here. Measured on the wire before and after, **this service emits byte-identical
errors** — the change was entirely to the other three.

What stays here is what is actually this service's: the job queue, the
streaming path, the chunker that works around the model's 40-second ceiling,
the Chatterbox knobs and the watermarker stub.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
