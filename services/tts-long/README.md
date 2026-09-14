# tts-long

Long-form text-to-speech. Chatterbox on CPU, as a **job queue** and an
**SSE stream**.

Two engines, chosen by the `model` string: `chatterbox`, the 23-language
checkpoint with the expressive controls, and `chatterbox-turbo`, 2.36x quicker
on a GPU and English-only with no expressive controls at all. **Both answer with
a job id**, and both run here with the optional GPU switched off. See
*Two engines*.

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
  ghcr.io/gabrielbelli/calliope-tts-long:pre
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
boundary, because `generate()` has no interruption point inside it.

**Audio and records expire on separate clocks, and that is the point.** The
audio goes after `TTS_AUDIO_TTL` (a day) and the row stays, marked
`audio.state: "expired"`; the record itself goes after `TTS_RECORD_TTL` (a
month). One TTL used to take both, so what was said, which voice said it and
which machine ran it — a few hundred bytes — was thrown away with the
megabytes every twenty-four hours, and the job list could only ever show today.
`TTS_JOB_TTL` still works and still means the audio one.

### The record is the index and the audio is an attachment

`jobs` is a dict in one process, so a restart empties it while the files sit in
the volume and survive. Every run writes a **record** — `runs/{id}.json` — and
`_recover` walks those, not the audio. Before records existed, a recovered row
was rebuilt from a filename and read *voice unknown*.

**That walk used to go the other way round, and it had to be inverted.** A
record with no audio behind it was an orphan and was deleted — a rule that
already needed one exemption, for audio deleted on purpose. Instant speech and
transcriptions keep no audio here at all, so records with no file are now the
majority, and a rule needing a second exemption is the wrong rule.

A run finished before this release has no `kind`, and **absent means `clone`**,
so every one of them is a valid record with no migration of file contents
anywhere. Records that were written beside the audio are moved into `runs/`
once, at startup. The record says `recovered: true` and never invents a voice it
does not know.

Audio state is stored as two booleans and read as one object, because the page
kept guessing at it:

| `audio.state` | means |
|---|---|
| `present` | there is a file; `format`, `bytes` and `url` come with it |
| `pending` | the job has not finished. It has no audio **yet**, which is not the same as having none |
| `deleted` | somebody pressed the button |
| `expired` | `TTS_AUDIO_TTL` ran out, or the file went from outside |
| `never` | an instant run or a transcription. **The audio was not lost, it was never kept** |

`never` is load-bearing: telling a reader "deleted" about a file that never
existed is the lie the enum exists to prevent.

### Records from the rest of the stack

`services/tts` (Kokoro) and `services/stt` (Parakeet) answer a request and
forget it. They now `POST /runs` here with a finished record and forget that
too — one request, no retry, dropped on a full queue, never on the caller's
clock. This is the only service in the stack that keeps a record of anything,
so it is the only place the three kinds can be listed together.

`POST /runs` is **service to service and is deliberately absent from the page's
proxy table and from the gateway.** A mutable log with the browser as a writer
is not a log.

Two rules make the two halves shippable in either order:

* **unknown keys are dropped, never 400ed** — a field added to a newer sender
  must not refuse an older receiver;
* **a server-only field is a 400 that names it.** `path` is the argument to
  `open()` on `/jobs/{id}/audio`; a sender that tries to set it is a bug in that
  sender, not a compatibility case.

The shape is pinned by `packages/common/tests/fixtures/run_records.json`, which
both sides read. If the code and the fixture disagree, the fixture is right.

### Filtering the listing

`GET /jobs` takes `limit` (50, max 200), `kind`, `audio` and `status`, all
comma-separable, and returns `counts` computed over **every** record before any
filter — so a client can label `Everything (412)` without asking twice.

Filtering happens here rather than in the browser because the listing is capped:
a morning of Kokoro presses would push last night's clone off the end of the
response before any client-side filter saw it, and a cap applied before a filter
is a cap on the wrong set.

Two rules the filters never break. **A live job is returned by every
combination**, before the limit — hiding the thing somebody is waiting for is
the worst possible reading of a filter. **A failed job is returned by
`audio=present` too** — a clone that fails has no audio, and making it vanish at
the moment its owner is watching reads as data loss.

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
| `model` | **Honoured.** `chatterbox` and `chatterbox-turbo` name an engine; `tts-long` and OpenAI's three names resolve to `TTS_DEFAULT_ENGINE`; anything else is a **400**. See *Two engines* |
| `voice` | Resolved against the voice registry; a string or `{"id": "..."}`. Unknown names are a **400**. See *Voices* |
| `response_format` | All six: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | Only `1.0`. Anything else is a **400**: Chatterbox has no rate control, and resampling would shift pitch |
| `instructions` | **400.** Chatterbox has no instruction conditioning; use `exaggeration`, `cfg_weight` and `temperature` through `extra_body` |
| `stream_format` | `audio` or `sse` |
| Anything else | **400**, naming the field, as OpenAI's own API answers it |

`exaggeration`, `cfg_weight`, `temperature` and `language` are vendor fields,
sent through `extra_body`, and are the only extras accepted. **Which of them an
engine honours is a property of the engine**, and one it does not is a 400 that
names the field, the reason and the engine that does — never a value accepted
and dropped. See *Two engines*.

Errors are OpenAI's envelope with **all four fields**, `param` included:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "voice", "code": "unsupported_value"}}
```

That includes a body rejected before it reaches any check — a missing `input`,
an `exaggeration` of 9 — and now also **404 and 405 under `/v1`**, which used
to escape as `{"detail": "Not Found"}` and tell openai-python nothing. The
native `/jobs` routes keep the `detail` list and the 422: they are the older
contract.

## Two engines: baseline and turbo

Two checkpoints, named on the wire by `model`. **Both answer with a job id**,
and **both run on this container's own processor** as well as on the runner's
card.

| `model` | engine | voices | languages | delivery controls | cold load |
|---|---|---|---|---|---|
| `chatterbox` | the multilingual checkpoint, 801M | **clones any clip** | 23 | `exaggeration`, `cfg_weight`, `temperature` | 22.2 s |
| `chatterbox-turbo` | Chatterbox Turbo, 743M | **clones a clip over 5 s** | **English only** | `temperature` **only** | 67.5 s |

Both cold-load figures were timed on `spring`, an RTX 3070; turbo warm-loads in
20.0 s. Both checkpoints are MIT.

**A third engine is in the catalogue and this deployment does not enable it.**
`voxtral` — Mistral `Voxtral-4B-TTS-2603`, int4, twenty preset voices in nine
languages — was built, wired to this wire, measured on the card it was for and
**retired**, for three measured reasons: it peaks at 8265–8288 MiB loading on
an 8192 MiB card and **segfaults** whenever the desktop holds any VRAM; its
transcript match against the script ranged 1.000 down to **0.682, with whole
sentences missing**; and it is **0.104x realtime**, the slowest figure here,
while unable to clone at all. The decision and the evidence are
[ADR 0010](../../docs/adr/0010-the-third-engine-was-measured-and-retired.md),
which supersedes
[ADR 0009](../../docs/adr/0009-a-third-engine-that-cannot-run-here.md).
**The engine code is not deleted**, it is not enabled: the sections below
describe what a deployment that turns it on gets, and how.

| `model` string | resolves to |
|---|---|
| `chatterbox` | that engine, `engine_reason: "pinned"` |
| `chatterbox-turbo` | that engine, `engine_reason: "pinned"` |
| `voxtral` | **400**, `code: model_not_found`, naming `TTS_ENGINES` — a catalogue name this deployment has not enabled, which is a different sentence from a name nothing has heard of |
| `tts-long` | `TTS_DEFAULT_ENGINE`, `engine_reason: "alias:tts-long"` |
| `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` | `TTS_DEFAULT_ENGINE`, `engine_reason: "alias:openai"` |
| absent or `null` | `TTS_DEFAULT_ENGINE`, `engine_reason: "default"` |
| anything else | **400**, `code: model_not_found`, `param: model` |

**No alias resolves to an engine without a local lane, and the service refuses
to start if `TTS_DEFAULT_ENGINE` has not got one.** That is what keeps a
runner-only engine from becoming a hole in the whole API where one is enabled:
it is reached by typing its name, so a client that has never heard of it cannot
be made to fail by somebody switching a gaming PC off. **No engine enabled here
is runner-only**, so nothing on this deployment can fail that way at all.

Comparison is `model.strip().lower()`. Every response carries **`x-tts-engine`**
beside `X-Voice` — on the 202, on the synchronous 200 and on the SSE response —
because that header is the only way a client holding audio can tell what made it.

### What turbo is faster at, measured on this project's own hardware

| configuration | rtf | vs baseline | VRAM |
|---|---|---|---|
| chatterbox fp32 | 0.6531x | 1.00 | 4199 MiB |
| **turbo fp32** | **1.5426x** | **2.36** | 3805 MiB |
| turbo fp16 | 1.5361x | 2.35 | 3393 MiB |
| bf16 | 0.6140x | 0.94 | 3259 MiB |
| fp16 | 0.6035x | 0.92 | 3259 MiB |

`spring`, an RTX 3070 with 8 GB. **Precision is a VRAM lever and not a speed
lever here**: the cast genuinely happens — weights fall from 3057 to 2035 MiB —
and it is still slower, because the model is autoregressive at batch one and
bound by kernel-launch latency rather than by matmul throughput. There is no
precision setting in this service and there is not going to be one.

### Turbo is past realtime and it is still a job

1.5426x realtime is comfortably past the point where a short request could be
answered down the socket, and **it stays a job anyway, permanently.**

That figure is the one on *that card*, and the card belongs to somebody who is
often using it. A synchronous turbo route would answer in about nine seconds
when the runner is idle and in about four minutes when its owner is gaming — the
same request, the same voice, a twenty-five-fold spread decided by whether
somebody started a game. A route that is *sometimes* fast is worse than one that
is always a job id.

`_sync_budget` still applies to both engines exactly as it does today, so a short
input may still finish inside the window — but it is computed from the **local**
rate and never from a runner rate. This service does not make promises on another
machine's behalf. The decision is
[ADR 0008](../../docs/adr/0008-two-engines-and-both-stay-jobs.md).

### What turbo cannot do, and what this service does about it

Turbo **accepts `exaggeration` and `cfg_weight` as keyword arguments and
discards them with a logged warning** — it fired on all 21 segments of the trial
generation — and its `generate()` has no `language_id` parameter at all, so the
call this service makes today raises `TypeError` against it. This service does
not pass that failure on to its own callers. Every one is a **400 naming the
field**:

| condition | status | `code` | `param` |
|---|---|---|---|
| `model` is not a known engine, alias or OpenAI name | 400 | `model_not_found` | `model` |
| `model` is a catalogue engine this deployment has not enabled | 400 | `model_not_found` | `model` |
| unknown `voice` | 400 | `unsupported_value` | `voice` |
| `language` outside the engine's set | 400 | `unsupported_value` | `language` |
| `exaggeration` sent to an engine that has no such control | 400 | `unsupported_value` | `exaggeration` |
| `cfg_weight` sent to an engine that has no such control | 400 | `unsupported_value` | `cfg_weight` |
| reference clip shorter than the engine's minimum | 400 | `unsupported_value` | `voice` |
| the engine is enabled but no lane can serve it now | 503 + `Retry-After` | `engine_unavailable` | `model` |

Evaluated in that order, so a client fixing one field at a time converges. The
503 is issued **before a job id exists** — a 202 that nothing can serve is a
progress bar that never moves.

The turbo refusals read like this, and the wording is the point:

```text
exaggeration is not supported by chatterbox-turbo: the emotion conditioning
layer is not built in this checkpoint (hp.emotion_adv is False), so the value
would be accepted by generate(), logged as a warning and discarded. Delivery on
this engine is controlled with temperature alone. Send model='chatterbox' if you
need exaggeration and cfg_weight.
```

**Equality with a deployment default is not consent.** `{"model":
"chatterbox-turbo", "exaggeration": 0.3}` is a 400 even though `0.3` is exactly
what `TTS_EXAGGERATION` is set to here, because the caller typed the field and
believes it did something. Defaults are read **after** the engine is known, from
that engine's own control set, so an engine without a control gets no default and
no environment key — which is why `TTS_EXAGGERATION` stays set on this deployment
and a turbo request never carries it.

`POST /jobs` refuses unknown top-level fields too, as pydantic's 422 in the
existing `{"detail": [...]}` shape. It used to accept and silently discard them,
and the page uses `/jobs` exclusively: `model` on `/v1/audio/speech` without this
would be a page that asks for turbo and gets baseline with no error anywhere.

### The rest of the bill

* **Turbo's output is about 11 dB quieter, by design.** It normalises to
  −27 LUFS; measured RMS is 0.070–0.104 against the current model's 0.132–0.170.
  Nothing here compensates for it.
* **It asserts a reference clip longer than 5 s.** `GET /voices` publishes each
  clip's `reference_seconds` — read from the file header at scan time, never a
  decode — and which engines it can be used with, so a short clip is refused
  before a job id exists rather than by an assert inside a running job.
* **It reloads slowly.** 67.5 s cold against 22.2 s for the current model, both
  on spring. A lane that yields the card and reloads pays that every time.

### Voxtral: twenty fixed voices, nine languages, no clip — and not enabled here

> **This section describes an engine this deployment does not offer.** It is in
> the catalogue and in the code, and every refusal, record field and `/health`
> row below is exercised by the suite. It is switched off here because of the
> three measurements in
> [ADR 0010](../../docs/adr/0010-the-third-engine-was-measured-and-retired.md),
> and the section is kept because a deployment with a bigger card can switch it
> on — see **[How to enable Voxtral](#how-to-enable-voxtral)** at the end. Read
> everything below as *"if it is enabled"*.

`voxtral` is Mistral's `Voxtral-4B-TTS-2603` served through the
`github.com/TheMHD1/voxtral-int4` wrapper, quantised to int4 with HQQ over
torchao's tinygemm kernels.

**It cannot clone a voice, and that is not a gap somebody can close.** Its
speakers are twenty `.pt` tensors shipped beside the weights; there is **no
speaker encoder anywhere in the checkpoint** and **no reference-audio parameter
in any of the wrapper's nine source files**. So a clip cannot be honoured on
this engine even in part. It is a preset-voice engine and belongs conceptually
beside Kokoro, whatever the fact that it lives in this service suggests.

The twenty, and **the language is a property of the voice** rather than a field
beside it:

| language | voices |
|---|---|
| Arabic | `ar_male` |
| German | `de_female`, `de_male` |
| English | `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male` |
| Spanish | `es_female`, `es_male` |
| French | `fr_female`, `fr_male` |
| Hindi | `hi_female`, `hi_male` |
| Italian | `it_female`, `it_male` |
| Dutch | `nl_female`, `nl_male` |
| **Portuguese** | **`pt_female`, `pt_male`** |

`GET /health` publishes them under `engines.voxtral.voices`, each with its own
language, so nothing anywhere parses the `de_` prefix to work that out. Prefix
parsing is exactly how a voice list ends up resolving `de_female` to `en-us` by
coincidence.

**Voice names are scoped by engine.** A clip in `TTS_VOICE_DIR` called
`pt_male.wav` does not collide with the preset: the pair `(engine, voice)` is
the key, which is already why `min_reference_seconds` is a property of the pair.
Such a name logs a warning at boot and nothing else — a service that refuses to
start because somebody named a file badly is worse than the collision.

#### It is 0.104x realtime and there is no setting at which it is not a job

Measured on `spring`, an RTX 3070, at the settings ADR 0009 tuned by ear and
ADR 0010 preserves. **0.104x is the slowest figure in this repository**, and it
is the third of the three reasons the engine is switched off here:

| engine | rtf | where |
|---|---|---|
| Kokoro | 2.2–2.7x | orko's processor, **no GPU at all** |
| Chatterbox Turbo | 1.594x | spring |
| Chatterbox baseline | 0.635x | spring |
| **Voxtral, `flow_steps=32`** | **0.104x** | **spring, and nowhere else** |
| Voxtral at the wrapper's own `flow_steps=8` default | 0.354x | — |

Those two Chatterbox figures are a **same-passage** re-run so that all four rows
compare to each other; ADR 0008's 0.6531x and 1.5426x are the trial that decided
turbo. They agree within a per cent and neither replaces the other.

**Twenty seconds of speech costs about 192 seconds of card.** Voxtral is about
six times slower than baseline, fifteen times slower than turbo, and twenty
times slower than Kokoro running on a container with no graphics card in it.
Even at the wrapper's own default of 8 flow steps it is 0.354x, so unlike turbo
this is not a decision anybody made — **there is no setting at which this
engine could have answered down a socket.**

`stream_format: "sse"` is still accepted and still honoured, because a socket
held open for four minutes is already true of baseline and refusing it here
would be theatre. **The page never chooses it**, because the page decides from
the engine's published rate and 0.104x is far below its streaming threshold.

#### Its two settings, and they were chosen by ear

| setting | the value that was bought by ear | why |
|---|---|---|
| `flow_steps` | **32** | **The main quality knob.** 32 clearly best on the owner's ear, 16 close behind, 8 and 4 audibly worse. The wrapper ships 8. |
| `cfg_alpha` | **1.2** | Full classifier-free guidance. 1.0 is documented upstream as faster and garbled. |

Both are request fields **and** deployment keys (`TTS_VOXTRAL_FLOW_STEPS`,
`TTS_VOXTRAL_CFG_ALPHA`, and the global `TTS_FLOW_STEPS`), and both are recorded
on the job, so a tuning log says exactly which numbers produced which sound.

**Neither key is set in this repository's `compose.yaml` any more**, because a
per-engine key is read only for an engine `TTS_ENGINES` names and a key nothing
reads is worse than no key. The two values above are the ones to start from
when the engine is enabled again; they are preserved in ADR 0010 with the rest
of the settings, including the four the runner owns.

Four more settings live on the runner rather than on the wire, because changing
any of them costs more than the current job: `group_size` (**32**, against the
wrapper's 64 — finer, and it cost no VRAM) and `max_frames` (**2000**) are
consumed inside the 63-second load and the KV-cache allocation respectively;
`fade_ms` (**120**) and `low_pass_hz` (**0**, off) shape the artefact rather
than the generation and must be identical across every segment of a listening
comparison. All four are published on `/health.engines.voxtral.runner.settings`
and copied onto the record, and all four are **refused by name** as job
parameters rather than swallowed.

Two of those four are corrections to upstream rather than preferences:

* **The 10 kHz low-pass is off.** `audio_postprocess.postprocess_audio` runs a
  6th-order Butterworth over everything it touches. The owner heard it
  immediately and called it *"sounds like a pilot mic"*. Resample plus peak
  normalise, and nothing else.
* **The warm-up glitch is faded, never trimmed.** Upstream's
  `trim_warmup_frames` only catches a run of *identical* leading codes and
  misses the rest. **Dropping frames instead was tried here and it ate the first
  word.** A squared fade-in over the first 120 ms attenuates the artefact and
  every sample of speech survives.

#### Audio from this engine is 24 kHz, and the repository it came from gets that wrong

`postprocess_audio` resamples **24000 → 48000**, and then `generate.py`,
`generate_fast.py` and `benchmark_all.py` all write the result back **at 24000**
— only `serve.py` is right. A file that inherits that mistake **plays at half
speed**, which is what happened here the first time round.

This service emits **raw 24 kHz and never post-processes to 48 kHz.** The codec
is 24 kHz natively, `voice_common/audio.py` fixes that rate estate-wide, the
rate is published as `native_sample_rate` on the engine row, and it is asserted
at load and again on the wire.

#### What it cannot honour, refused by name

Same house rule, same shape as turbo's, with one addition worth calling out:

| the caller sends | what comes back |
|---|---|
| a `voice` outside the twenty | **400** on `voice`, saying the speakers are embeddings baked into the weights with no speaker encoder of any kind, listing the ones in the language asked for and then all twenty, and naming `model='chatterbox'` for cloning |
| a clip upload, or an OpenAI voice alias | **400** on `voice`. The aliases resolve to reference clips here and are **not** mapped onto a preset by ear |
| `exaggeration`, `temperature` | **400** naming the field: a flow-matching checkpoint with no emotion conditioning and a sampler that does not read temperature |
| `cfg_weight` | **400** on `cfg_weight`, **naming `cfg_alpha`** — a different scale and a different solver |
| `flow_steps` or `cfg_alpha` sent to either Chatterbox | **400** naming the field: neither has a flow-matching decoder |
| `language` contradicting the chosen voice | **400** on `language`, naming a voice in the language asked for |

`cfg_weight` is the near-miss a real caller hits most, so it is the one refusal
that names the other field rather than only saying no.

#### The 63 seconds are paid every time, and the flag that looks like the fix is a decoy

There is **no pre-quantised checkpoint shipped**. Every load reads 8.01 GB of
BF16 into RAM and quantises it to int4, and **28.8 s of the 63 is the
quantisation alone**. `--quantized` is not the way out: it routes to an
unrelated **TurboQuant** experiment importing a package that is not in the
repository, not on PyPI and not installed, whose own docstring says it
dequantises back to BF16 before taking the identical memory spike. Removing the
63 s means serialising torchao's int4 state dict — new code, and the
highest-value follow-up this engine has.

**The memory spike is the real operational risk: 3.5–3.7 GB resident, but a
measured 8265 MiB peak during load on an 8192 MiB card** — 73 MiB *over* what
the card physically holds. It completed only because the desktop was near idle.
The runner therefore publishes its manifest **first**, loads on the first job,
and fails **that job** with both numbers in the sentence rather than killing the
process; a controller that exits before publishing is relaunched every 500 ms
for ever while the client watches `queued`.

It also needs **torch 2.11+**, which the `chatterbox` service's `torch==2.6.0`
pin makes impossible to share — so it is a second virtual environment and a
second CUDA torch on the runner, about 2.5 GB more.

### How to enable Voxtral

**`TTS_DEFAULT_ENGINE` must have a local lane, and no alias may resolve to an
engine without one.** That is the rule, and it is narrower than the one ADR 0008
wrote — *"every engine this service advertises is renderable on the local
lane"* — because Voxtral was the first engine that could not satisfy it. The
narrowed rule is kept now that it is retired: it is the stricter statement of
the same promise, and it is what the boot checks enforce.

`TTS_ENGINES` naming anything `TTS_LOCAL_ENGINES` cannot run is a **refusal to
start**, not a warning, unless `TTS_ALLOW_RUNNER_ONLY_ENGINES=1` says the
operator accepts what it buys. **This deployment does not set it**, and with no
runner-only engine enabled there is nothing here for it to excuse — leaving it
set would be a disarmed safety catch waiting for the next such engine.

Enabling Voxtral is therefore three settings, all in `compose.yaml`:

1. add `voxtral` to **`TTS_ENGINES`** on `tts-long`;
2. add `voxtral` to **`GATEWAY_LONG_MODELS`** on `voice-gateway`. These two must
   agree, minus the `tts-long` alias, and `docs/tests/test_deployment.py`
   asserts that they do;
3. set **`TTS_ALLOW_RUNNER_ONLY_ENGINES=1`**, since the engine has no local
   lane.

Start from the settings table in ADR 0010 rather than from the wrapper's
defaults, and know what the third switch buys:

* **A `voxtral` request with the runner away is a 503 `engine_unavailable`,
  before a job id exists and before a queue slot is taken**, naming the engine,
  the `idlegpu service install voxtral` command, and `model='chatterbox'`.
* **An already-accepted job that has had no eligible lane for
  `TTS_RUNNER_ONLY_DEADLINE_SECONDS` (900) reaches a terminal `failed` record**
  with the runner's own reason in it, and its queue slot is released. The clock
  counts *time with no eligible lane*, never time in the queue: at 0.104x a
  two-minute script is over nineteen minutes of card and is perfectly healthy.
* **A request that does not type `voxtral` is unaffected.** Baseline and turbo
  keep answering with the runner switched off, and that is asserted rather than
  hoped for.

**A Voxtral job never falls back**, and `fell_back` is never set for it. Nothing
may render *"after spring gave up"* about a machine that never took the job.

Why not simply hold it queued: `TTS_MAX_QUEUE` is **32 and deployment-wide**, so
thirty-two overnight Voxtral submissions would answer **429 to every Chatterbox
caller on a completely idle lane.** Why not substitute Kokoro: that is the house
rule inverted, and `de_female` has no Kokoro equivalent at all. It is right as
an *offer* on the page and wrong as a substitution here.

The fallback is gentler than the one already shipping: turbo on a processor is
3.0–3.2x baseline on the same processor, so turbo-on-card to turbo-here costs
about 2.5x, while baseline-on-card to baseline-here costs about 3x and happens
several evenings a week.

Voxtral has no such fallback and never will; that is the whole of ADR 0009 and
the section above.

There is still **one runner lane**. The agent runs at most one controller per
device group, so the card holds one process; the engine rides on the job and
resolves to an `idlegpu` service id at submit
(`TTS_RUNNER_SERVICE_<ENGINE>`). When engines contend for the card **baseline
wins**, because it has the worse fallback: 23 languages and two expressive
controls with no substitute anywhere, against turbo's local floor 2.5x away. A
runner-only engine registers below both — losing the card costs it a wait, and
it was always going to be a job.

**On this deployment the two engines that exist are both local**, so `spring`
is speed and never availability: with that machine switched off, unplugged or
lying about itself, every request still finishes on this container's eight
threads.

## Voices

**A voice on this service is a file, for both engines it offers.** Chatterbox
and Chatterbox Turbo have no named voices at all: they **clone from a reference
clip**, so a voice is a file. The other kind — presets baked into the weights,
with no clip anywhere in the path — is what `voxtral` would bring back if it
were enabled, and the pair `(engine, voice)` is the key precisely so that the
two namespaces cannot collide when it is.

```bash
docker run -v /srv/voices:/voices ... ghcr.io/gabrielbelli/calliope-tts-long:pre
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
| A catalogue `model` this deployment has not enabled is **400 `model_not_found`**, not a fall-through | `voxtral` is in the catalogue and switched off here, and an unrecognised name that reached the default engine would hand a caller audio from an engine they did not name | The refusal names `TTS_ENGINES`, so the sentence says *"that exists and this box has not turned it on"* rather than *"never heard of it"*. See [ADR 0010](../../docs/adr/0010-the-third-engine-was-measured-and-retired.md) |
| `speed` is a **400** | **No engine in the catalogue has a rate control**, so the sentence names no engine; resampling to fake one shifts pitch with it, and an external time-stretch is not in this image | — |
| `instructions` is a **400** | **No engine in the catalogue has instruction conditioning.** Also engine-neutral, and checked before the engine is even known | — |
| The thirteen OpenAI voice names map to one voice | Chatterbox clones from clips and none ship here | Reported per response in `X-Voice`; closable by adding clips |
| Streamed `wav` and `flac` differ from the buffered file in their header | Both state the total length, which is not known until the last sentence is generated | Diffed byte for byte: `wav` differs at offsets 4 and 40 only and is identical from byte 44; `flac` differs inside STREAMINFO only (offsets 8–41). `pcm`, `mp3` and `aac` are exact |
| Streamed `opus` is not byte-comparable between requests | The Ogg serial number is random per stream, so two encodes of the same samples differ anyway | Within one request the deltas are a partition of a single encode |
| `sse` is accepted for every `model` value | OpenAI restricts `stream_format` to `gpt-4o-mini-tts`. Both engines here can stream, so refusing the field on the strength of a name would be theatre. A slow engine holds the socket longer, which is a difference of degree and not of kind — `X-Job-Id` is on the response, so a caller that gives up can poll | — |
| `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` resolve to `TTS_DEFAULT_ENGINE` | They are documented **aliases**, not fields accepted and dropped. That is what they have always meant here, and it preserves every direct caller that reaches this service without the gateway | `x-tts-engine` on every response names the engine that actually ran |
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
  ghcr.io/gabrielbelli/calliope-tts-long:pre
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
| `TTS_EXAGGERATION` | `0.3` | Stock is 0.5 and reads as over-cheerful. Read **only** for engines that declare the control, so it never reaches `chatterbox-turbo` |
| `TTS_CFG_WEIGHT` | `0.3` | Lower is slower, more deliberate. Same rule |
| `TTS_TEMPERATURE` | `0.6` | Stock 0.8 varies more than an explanation wants |
| `TTS_ENGINES` | `chatterbox` | Which catalogue engines this deployment offers. A name that is not in `voice_common.engines.CATALOGUE` **refuses to start** |
| `TTS_DEFAULT_ENGINE` | first of `TTS_ENGINES` | What `tts-long`, the three OpenAI names and an absent `model` resolve to |
| `TTS_LOCAL_ENGINES` | value of `TTS_ENGINES` | Which engines this container's own CPU may run. Anything in `TTS_ENGINES` and not here refuses to start |
| `TTS_ALLOW_RUNNER_ONLY_ENGINES` | `0` | Opt out of that refusal and accept a 503 at submit whenever the runner is away. **Not set here**: no engine this deployment offers needs it, and leaving it set would disarm the check for the next one that does. It never lets an *alias* reach a runner-only engine, and `TTS_DEFAULT_ENGINE` must still have a local lane |
| `TTS_RUNNER_ONLY_DEADLINE_SECONDS` | `900` | How long an already-accepted job may go with **no eligible lane** before it is failed with a terminal record. Counts time with no lane, never time in the queue. Its clock can only start for an engine with no local lane, so nothing here can trip it and the key is **not set** |
| `TTS_VOXTRAL_FLOW_STEPS` | *(unset here)* | **Voxtral's main quality knob**, chosen by ear on an RTX 3070: 32 clearly best, 16 close behind, 8 and 4 audibly worse. The upstream wrapper ships 8, which runs 0.354x against 0.104x. Also a request field. Read only where `voxtral` is enabled, which is not here — set it to **32** if you enable it |
| `TTS_VOXTRAL_CFG_ALPHA` | *(unset here)* | Full classifier-free guidance at **1.2**. 1.0 is documented upstream as faster and **garbled**. Not `cfg_weight`: different scale, different solver. Also a request field, and read only where `voxtral` is enabled |
| `TTS_LOCAL_RESIDENT_MAX` | `1` | Local checkpoints kept loaded at once. Two engines resident is ~6.6 GB plus 1.83 GB against a 10 GB limit |
| `TTS_<ENGINE>_<FIELD>` | the global default | Per-engine default for one control, e.g. `TTS_CHATTERBOX_TURBO_TEMPERATURE`. Setting one for a field that engine has no control for is **fatal at boot**, naming the key and the way out |
| `TTS_RUNNER_PORT` | `47600` | The runner's port. Only read when `TTS_RUNNER_HOST` is set |
| `TTS_RUNNER_POLL` | `2` | Seconds between polls of a lease the runner is already working on. Not the readiness probe — that is `TTS_RUNNER_PROBE_S` |
| `TTS_RUNNER_SERVICE_<ENGINE>` | the engine id | The `idlegpu` service id per engine. `TTS_RUNNER_SERVICE` is the legacy spelling and now means the `chatterbox` engine only. For an engine with no local lane an uninstalled service is a 503 rather than a slow local job, because there is no local job |
| `TTS_REALTIME_FACTOR_<LANE>_<ENGINE>` | the catalogue seed | Rate seed per (lane, engine) pair. Falls back pair → lane → global |
| `TTS_COLD_LOAD_SECONDS_<ENGINE>` | the catalogue seed | Cold load per engine; `chatterbox-turbo` seeds 68 from the 67.5 s measured on spring |
| `TTS_API_KEYS` | *(unset)* | Comma-separated accepted keys. Unset means **no auth**; set but naming no key (`''`, `','`) refuses to start |
| `TTS_LOG_LEVEL` | `INFO` | Root log level. An unrecognised name warns and falls back to `INFO` rather than refusing to start |
| `TTS_TLS_CERT` | *(unset)* | PEM certificate. Both this and the key are needed for HTTPS; half a pair refuses to start |
| `TTS_TLS_KEY` | *(unset)* | PEM private key. Only applied to the `uvicorn` command; overriding `CMD` with TLS set refuses to start |
| `TTS_VOICE_DIR` | `/voices` | Reference clips. `<name>.wav` becomes voice `<name>`. Rescanned when the directory changes, so a clip copied in is usable on the next request rather than after a restart |
| `TTS_VOICE_STRICT` | *(off)* | Refuse OpenAI voice names that have no clip, instead of aliasing them to `default` |
| `TTS_OPENAI_SYNC_MAX_CHARS` | `300` | Hard ceiling on input answered synchronously; `0` always returns 202 |
| `TTS_OPENAI_SYNC_TIMEOUT` | `180` | Seconds to wait before giving up and returning 202 instead |
| `TTS_REALTIME_FACTOR` | `0.21` | Seed for the measured rate. Corrected by every job that finishes |
| `TTS_COLD_LOAD_SECONDS` | `60` | Charged against the synchronous budget when the model is not resident. Covers the ~3 GB first-ever download as well as the load; the load alone timed 22.2 s on spring |
| `TTS_CHARS_PER_SECOND` | `15` | Measured between 9.8 and 19.3. Only affects estimates and chunk sizing |
| `TTS_CHUNK_MAX_CHARS` | `280` | Hard ceiling per `generate()` call; must stay under 40 s of speech |
| `TTS_CHUNK_TARGET_CHARS` | `160` | Short sentences merge up to this. Lower means sooner first audio and choppier prosody |
| `TTS_MAX_QUEUE` | `32` | Queue depth past which both routes answer **429** with `Retry-After` |
| `TTS_AUDIO_TTL` | `86400` | Seconds the AUDIO survives. The record stays and says `expired`. `TTS_JOB_TTL` is still read and means this |
| `TTS_RECORD_TTL` | `2592000` | Seconds the RECORD survives. Thirty days. `0` on either disables that half |
| `TTS_BACKEND_ORDER` | `runner,local` | Which lanes exist. A membership test, not an order: leaving a name out switches that lane off, and `local` alone is local-only |
| `TTS_RUNNER_MAX_WAIT` | `300` | Seconds a job may sit queued on the runner before it is given up on and spoken here. Was 900 when a yield stalled the whole service; with two lanes it is five minutes of a still progress bar against fifteen |
| `TTS_DISPATCH_MARGIN` | `1.25` | How much better a remote lane must be before a job crosses the network |
| `TTS_RUNNER_HOP_S` | `8` | The fixed cost of the handover, added rather than multiplied |
| `TTS_RUNNER_PROBE_S` | `10` | Seconds between asking the runner whether it is free, on a thread nobody waits on |
| `TTS_RUNNER_OFFER_TIMEOUT` | `3` | That question's own timeout, separate from the 30 s job timeout |
| `TTS_RUNNER_COOLDOWN_S` | `30` | How long a lane is left alone after it hands a job back |
| `TTS_STREAM_ON_RUNNER` | `0` | Streamed jobs stay on this host. A yield mid-stream cannot be re-run |
| `TTS_SHUTDOWN_GRACE_S` | `20` | How long shutdown waits for the lanes before marking what is left `cancelled` |
| `TTS_RUNLOG_ACCEPT` | `1` | Whether `POST /runs` accepts records from `tts` and `stt`. `0` answers 404 |
| `TTS_RUNLOG_MAX_RECORDS` | `5000` | Ceiling on stored records, counted against a cache refreshed every 60 s |
| `TTS_RUNLOG_RATE` | `120` | Accepted records per minute, per sending service |
| `TTS_RUNLOG_TEXT` | `1` | `0` stores the length of what was said and not the words |
| `AIV_HOST_LABEL` | `platform.node()` | What this machine is called, on every record it writes |
| `TTS_SSE_KEEPALIVE` | `10` | Seconds between `:` comment lines on a waiting stream |

`<ENGINE>` in a key name is the engine id uppercased with hyphens turned into
underscores: `chatterbox-turbo` is `CHATTERBOX_TURBO`. **No key spells `turbo`
by hand anywhere**, which is what keeps a third engine a catalogue row rather
than a branch.

Chatterbox's shipped defaults (0.5 / 0.5 / 0.8) are tuned for expressive
delivery. For instructions and explanations they sound performed. These are
calmer; raise them if you want more life. They apply to `chatterbox` alone:
`chatterbox-turbo` has no expressive controls at all and takes `temperature`,
and `voxtral` has neither and takes `flow_steps` and `cfg_alpha`.

**A per-engine key for a control that engine has not got is fatal at boot**, and
so is **a global key that reaches no enabled engine** — `TTS_EXAGGERATION` on a
box running only `voxtral` is the same untruth as `TTS_VOXTRAL_EXAGGERATION`,
one file away. `TTS_VOXTRAL_LANGUAGE` is fatal too, with its own reason:
Voxtral's language is carried by the voice embedding, so a deployment-wide
default would either agree with the voice or contradict it. Pick a voice.

**Four of Voxtral's settings are not in this table because they are not this
service's.** `group_size` (32), `max_frames` (2000), `fade_ms` (120) and
`low_pass_hz` (0, off) are installed on the runner in `worker.ini` and are
refused **by name** if sent as job parameters. Where the engine is enabled they
are readable without an SSH session — the controller publishes them and this
service echoes them at `/health.engines.voxtral.runner.settings`.
## One job at a time

Deliberate. The model is 6.5 GB and generation is sequential, so a second
concurrent job would double the memory and slow both.

Two engines does not change that: `TTS_LOCAL_RESIDENT_MAX` is `1`, so switching
between `chatterbox` and `chatterbox-turbo` on the local lane evicts the outgoing
checkpoint and pays its cold load on the way back. Raise it only on a host with
the memory for both — 6.6 GB plus turbo's 1.83 GB of weights and an
`AutoTokenizer`, against this deployment's 10 GB limit.

## Torch, and why CPU

Chatterbox has no ONNX build, so torch is unavoidable. The **CPU wheel** is
used on purpose: the CUDA wheels add several gigabytes, and the GPU this would
otherwise target is a GTX 1060 — Pascal, whose FP16 runs at 1/64 rate. It
would not help even where one exists, and 6.5 GB does not fit in 6 GB of VRAM
at fp32 regardless.

## Borrowing somebody's GPU, optionally

Everything in this service is slower than realtime on this container's
processor, which is the entire reason this section exists. **It is off by
default and unset means local-only**: with no `TTS_RUNNER_HOST`, every job runs
on this CPU and `app/remote.py` is imported and never used.

**That is true of both engines this deployment offers**, which is what makes
the runner speed and never availability here. It is not true of every engine in
the catalogue: `voxtral` has no local path at all, so for it the runner is not
speed but existence, and leaving `TTS_RUNNER_HOST` unset on a deployment that
offers it means every `voxtral` request is a 503. The honest thing on such a
box is to take it out of `TTS_ENGINES` and `GATEWAY_LONG_MODELS` rather than
advertise it — which is what this deployment has now done, for reasons that
outgrew the runner entirely. See
[ADR 0010](../../docs/adr/0010-the-third-engine-was-measured-and-retired.md).

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

**Chatterbox Turbo on the same card is 1.5426x realtime** — 2.36x baseline's
0.6531x, measured this week at fp32 — which is the first time anything long-form
in this stack has been faster than the speech it produces. It is **still a job**,
because that figure belongs to a card somebody else is often using. See
*Turbo is past realtime and it is still a job*.

### The runner prices each service separately, so one boolean is not the answer

A game takes the card and leaves twelve threads idle. A compile takes every
thread and leaves the card at five per cent. So the runner prices its services
separately, and this side asks it about the one it wants rather than working the
answer out.

This is about **asking**, not about routing: nothing here has ever sent a job to
the runner's processor, and [there is no such lane](#the-processor-rung-is-gone).

`GET /v1/services` carries a per-service `available` field, already resolved
against the right gate: the GPU detector for a GPU service, the runner's own
state ladder for a CPU one, and the free-memory check for both. This service
reads that and nothing else. It does **not** read the machine-wide
`gpu_available` and reason from a device name, because that would be this side
reimplementing a decision the runner already makes correctly, including a memory
headroom check that is invisible from here.

A runner that predates the split publishes no `available` at all, and a missing
field is not a `false`: the machine-wide flag is the fallback, because refusing a
perfectly good older runner for ever is a worse failure than being slightly
conservative.

The status panel shows what the runner is currently willing to give up, because
a job that takes four times as long while its owner is at their desk is the
system working, not a fault, and a panel that cannot say so sends people looking
for one.

### Two lanes, and nothing asks a runner a question on a job's own thread

There were three rungs on a ladder and there are now **two lanes**, each one
job wide, chosen per job:

| lane | engine | what it is | rate |
|---|---|---|---|
| `local` | `chatterbox` | this container, eight Xeon threads | 0.23x, measured |
| `local` | `chatterbox-turbo` | the same eight threads | 0.45x, **an estimate** |
| `runner` | `chatterbox` | the runner's card, an RTX 3070 | 0.70x, measured |
| `runner` | `chatterbox-turbo` | the same card | 1.54x, measured |

**Two lanes, four rates.** The lanes are places; the engines ride on the job. A
third engine adds two rows and no lane.

The rule, with `A` the seconds of speech a job is:

```
finish(lane) = work already in the lane + A / rate(lane) + handover(lane)
choose the smallest, subject to: a remote lane must beat local by 1.25x
```

Worked at the measured rates, on 300 seconds of speech with both lanes idle:
1304 s here against 429 s there plus an 8 s handover, and 437 × 1.25 = 546 is
still comfortably under 1304 — so it goes. **The handover is added, not
multiplied.** A multiplicative margin hides a fixed cost; this one is real, and
it is why a one-second job stays here.

**The margin is asymmetric on purpose.** Losing a remote lane mid-job costs a
whole re-speak, because `_run` re-enters with a fresh encoder and a fresh
offsets list. Losing the local one costs only slowness. So a remote lane has to
be clearly better, not marginally better.

**The second job starts now.** With one worker, a ten-minute job handed to the
runner's card held this host's completely idle CPU for the whole ten minutes,
and the next job started twenty-one minutes late. Two lanes is the entire win,
and it is asserted in `tests/test_dispatch.py`.

**Local's width is one and is not configurable.** `Synth._speak` holds one lock
across `_ensure_loaded()` and `generate()`, so two local jobs do not overlap —
they interleave at segment granularity for no extra throughput and double the
latency of each. Every concurrency here comes from the second machine.

#### Asking is a background job now

`offer()` used to be called on the worker's own thread, once per job per rung,
deliberately uncached — and `RunnerConfig.timeout` is thirty seconds. A runner
that accepts a TCP connection and then says nothing therefore added **thirty
seconds of silence to a job that was always going to run here**. Switched off is
the cheap case: that is a refused connection. Asleep, wedged, or behind a
firewall that drops rather than rejects is the expensive one.

So `LaneProbe` asks every `TTS_RUNNER_PROBE_S` on its own thread, with its own
`TTS_RUNNER_OFFER_TIMEOUT`, and the chooser reads the answer. An answer older
than three intervals shuts the lane, because a probe stuck on a socket must not
leave the last good answer standing. Ten seconds of staleness costs at most one
wrong lane choice, which the yield path already recovers.

**A yield puts the job back at the head of the queue** and cools the lane for
`TTS_RUNNER_COOLDOWN_S`. The job has already waited its turn once; sending it to
the tail would let everything submitted since overtake it because somebody
started a game.

#### The processor rung is gone

`runner_cpu` had a whole arithmetic — read the published cap, derate by it,
compare against the backlog — and eight passing tests. **It had never run a
single job and could not.** The server set
`TTS_RUNNER_CPU_SERVICE=chatterbox-cpu` and the agent on that machine registers
`echo` and `chatterbox` and has never offered anything else; every offer was
`no_such_service`. It also sat below `local` on the ladder, and `local` is a
rung that is always willing, so nothing below it was ever reached.

The two-lane arithmetic refuses it anyway without a special case: 300 s of
speech is 1250 s on the runner's processor against 1304 s here, and 1250 × 1.25
is more than 1304. `TTS_BACKEND_ORDER` survives as a membership test — naming a
lane turns it on — and naming `runner_cpu` now does nothing.

`compose.yaml` no longer sets `TTS_RUNNER_CPU_SERVICE` and no longer recommends
an order containing `runner_cpu`, and none of the four `TTS_RUNNER_CPU_*` keys
appears in it as a live knob. **Be exact about which of them actually died**,
because rounding that up is the same class of falsehood as the rung was:
`TTS_RUNNER_CPU_MIN_PCT` and `TTS_RUNNER_CPU_WHEN_BACKLOG_S` are gone from the
code. `TTS_RUNNER_CPU_SERVICE`, `TTS_RUNNER_CPU_MAX_WAIT` and
`TTS_REALTIME_FACTOR_RUNNER_CPU` are **still parsed** and still reach fields
nothing can route work to — including `runner.cpu_service` on `/health`, which
names a service the agent on that machine has never registered. That deletion is
still owed.

The decision, its measurements and the dated correction about what was and was
not deleted are all in
[ADR 0007](../../docs/adr/0007-two-lanes-not-three-rungs.md), which supersedes
[ADR 0006](../../docs/adr/0006-three-way-speech-routing.md).

### What it costs

The GPU goes away, several times an evening, whenever its owner sits down. That
is the runner working correctly rather than failing: it yields in **under three
seconds**, measured with the model resident, and the job comes back as
**queued** rather than `failed`. This service waits `TTS_RUNNER_MAX_WAIT` and
then speaks it on the CPU instead.

Since the runner started selling its processor, "the GPU went away" is no longer
the only reason it declines, and neither is it always the end of the job there:
while somebody is at their machine the runner throttles rather than stopping, so
the work continues at a tenth of the speed instead of coming back. That is why
the fallback log says "the runner is not free" and quotes the runner's own
reason, rather than asserting a game is running.

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

**The realtime-factor EMA is per (lane, engine) pair.** `rate` decides whether an
incoming request is answered synchronously or handed a 202. A GPU at 0.7x and
this CPU at 0.275x sharing one average describes no machine that exists, and
`_sync_budget` would then accept a synchronous request the CPU cannot finish — at
exactly the moment the GPU disappears, because that is when jobs come back here.

**The second engine makes that argument sharper, not softer.** Baseline and turbo
are **2.36x apart on the same card** — 0.6531x against 1.5426x on spring's
3070 — so a per-lane average over both describes no configuration that exists
either. `/health` publishes `realtime_factor_by_engine` keyed `lane/engine`, and
keeps `realtime_factor_by_backend` beside it narrowed to `TTS_DEFAULT_ENGINE`
rather than left as the average of two things a factor of two apart.

**And each has its own seed, with the count of what is behind it.** Every
backend used to start from the local CPU constant, which for a GPU is
pessimistic and self-corrects on the first finished job. A lane seeded from
`local` is worse than pessimistic: it is indistinguishable from `local`, so it
is never chosen, never measured, and its seed is permanent — which is how
`runner_cpu` kept a hypothesis as a number for its whole life. `/health`
publishes `engine_observations` beside the rates: a count of zero says the figure
next to it is a documented measurement from another machine rather than this
stack's own.

**One seed in this deployment is not a measurement, and it says so.**
`local/chatterbox-turbo` is seeded at 0.45. Turbo on a processor has never been
run on this host: it measured 0.53–0.55 on an M-series laptop, and turbo is
3.0–3.2x baseline on the same processor with the same threads and the same clip,
which puts this container somewhere around 0.5–0.7. It is seeded at the bottom of
that range on purpose, because a pessimistic rate defers to a 202 rather than
promising a synchronous answer this container cannot deliver. **One turbo job
finished here replaces it**, and until one is, `engine_observations` reports 0
beside it.

**The estimate stays a local estimate, and it stays conservative.**
`estimated_seconds` is frozen at enqueue from `_rates["local"]`, whatever ends up
running the job. That is the safe direction with two lanes: the only lane a job
can be routed to is the card, at three times this host's rate, so a job that
leaves here finishes early against the number its caller was given, never late.

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

**One marker is deselected by default and the run summary says so.**
`checkpoint` marks the tests that need the real weights installed — there is
exactly one, and it asks whether the wheel on *this* machine is the checkpoint
`voice_common.engines.CATALOGUE` describes. Run it where the package is:

```bash
cd services/tts-long && python -m pytest tests -m checkpoint
```

It is deselected rather than skipped on purpose. A `skipif` on "are the weights
here" is silently absent on the one machine it matters on, which is how
`chatterbox-cpu` became a rung nothing ever offered a job. Everything else
about engine identity — that the request reaches the right catalogue row, and
that the row loads the checkpoint it names — runs on every commit with no
weights and no card. See `tests/test_engine_identity.py`.

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
