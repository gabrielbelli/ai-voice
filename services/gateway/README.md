# voice-gateway

One port, one key, three speech services.

```text
                        :8080  voice-gateway
                          │
  /v1/audio/transcriptions├──────────────────────────►  stt-stack:8000
  /transcribe             │                             Parakeet, 8.5-10.4x
                          │
  /v1/audio/speech  model=│kokoro tts-1 tts-1-hd …  ─►  tts-stack:8001
  /speak  /voices         │                             Kokoro, 1.2-1.5x
                          │
  /v1/audio/speech  model=│chatterbox tts-long      ─►  tts-long:8002
  /jobs  /jobs/{id}[/audio]                             Chatterbox, 0.138x
                          │
  /v1/models              ├─  answered here, from a static table
  /health                 └─  all three, fanned out, no key required
```

Sibling of [services/stt](../stt/README.md), [services/tts](../tts/README.md)
and [services/tts-long](../tts-long/README.md), same conventions. No
torch, no model, no volume, no state — the image is `fastapi` and `httpx` on
the slim base.

## Why this exists

Two reasons, and they are the only two.

**One auth boundary.** The three backends each carry their own copy of an auth
module. The three copies have already diverged — one enforces via ASGI
middleware, one via per-route dependencies *plus* a re-registration of
`/openapi.json`, `/docs` and `/redoc` behind the key, one carries a third
variant — and the same two bugs were found and fixed in only one of them:
`/health/` with a trailing slash missing the exemption and 401-ing a healthcheck
that had always worked, and header bytes decoded latin-1 then re-encoded UTF-8
so a correct non-ASCII key was rejected as "Incorrect API key". Three
enforcement points is three places for the fourth bug. Here the backends run
open, only `:8080` is published, and one file checks a token.

**One health answer.** Knowing whether the stack is up currently means polling
three ports. `GET /health` fans out and returns all three, unauthenticated, in
one call.

Routing is a consequence of those two, not the point.

## Status

Not deployed yet. `main` carries validated versions only; work happens on
`prerelease`, which publishes `:pre` and never `:latest`.

**This component does not pay for itself until the app config stops publishing
8000, 8001 and 8002.** Until then it is a fourth container and an extra hop in
front of three ports that are still open to the LAN, and the single auth
boundary is aspirational rather than real. See *Deploying* below.

## Run

```bash
docker run -p 8080:8080 \
  -e GATEWAY_STT_URL=http://stt-stack:8000 \
  -e GATEWAY_TTS_URL=http://tts-stack:8001 \
  -e GATEWAY_TTS_LONG_URL=http://tts-long:8002 \
  -e GATEWAY_API_KEYS=sk-workstation,sk-laptop \
  ghcr.io/gabrielbelli/ai-voice-gateway:pre
```

```bash
curl -s localhost:8080/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "gateway": "ok",
  "backends": {
    "stt": {"url": "http://stt-stack:8000", "reachable": true, "http_status": 200,
            "health": {"status": "ok", "model": "parakeet", "threads": 8}},
    "tts": {"url": "http://tts-stack:8001", "reachable": true, "http_status": 200,
            "health": {"status": "ok", "voices": 54, "default_voice": "bm_george"}},
    "tts_long": {"url": "http://tts-long:8002", "reachable": true, "http_status": 200,
                 "health": {"status": "ok", "model_loaded": false, "queued": 0}}
  }
}
```

Each backend's own body is inlined rather than summarised: `model_loaded:
false` on a cold tts-long and `status: "loading"` on a starting tts-stack are
the answers to the question an operator is about to ask next.

## Routes

Everything else is `404` with an OpenAI envelope. There is no catch-all
pass-through: `/docs`, `/redoc` and `/openapi.json` are **not** proxied,
because stt-stack deliberately put its own behind a key, and a wildcard route
here would quietly undo that.

| Route | Backend | Body |
|---|---|---|
| `POST /v1/audio/transcriptions` | stt-stack | streamed through |
| `POST /transcribe` | stt-stack | streamed through |
| `POST /v1/audio/speech` | by `model` — see below | buffered, to read `model` |
| `POST /speak` | tts-stack | streamed through |
| `GET /voices` | tts-stack | — |
| `POST /jobs` | tts-long | streamed through |
| `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/audio` | tts-long | — |
| `GET /v1/models` | answered here | — |
| `GET /health` | all three | — |

**Native routes mount flat and unprefixed, and nothing is rewritten.** That is
load-bearing rather than cosmetic. tts-long answers a long request with
`Location: /jobs/{id}` and a body field `audio_url: /jobs/{id}/audio`, both
relative to its own root; mounted here at the same paths they stay correct with
zero header rewriting. A prefixed design (`/tts-long/jobs/…`) would need a rule
that rewrites a header *and* a JSON field, and that rule rots the first time the
backend adds a field.

Upload bodies stream straight through. An hour of wav is 100 MB+, and buffering
it here would double resident memory on a host that already keeps 6.5 GB of
Chatterbox around. `POST /v1/audio/speech` is the exception — the gateway must
read `model` out of that body to route it, and that body is text measured in
kilobytes. Responses stream in every case.

## Which TTS backend a request reaches

**The criterion is the `model` string and nothing else.**

| `model` | Backend |
|---|---|
| `kokoro`, `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` | tts-stack |
| absent, empty, or **any unrecognised value** | tts-stack |
| `chatterbox`, `tts-long` | tts-long |

`GET /v1/models` returns that table as OpenAI's model list, with the backend in
`owned_by`. It is answered from a static table with no backend call: the names
are a property of the routing contract, not of any backend's state, and a
client most wants to know what it can send while a backend is restarting.

**Input length is not a routing key**, and that is the decisive rejection.
Length is a proxy for a quality choice, and the cost of getting it wrong is not
symmetric: a caller sending a perfectly ordinary 400-character request would
have a 17-second call auto-escalated into a ten-minute job, with nothing in the
request that asked for it.

The rejection used to rest on a second, sharper fact that has since expired,
and it is recorded here rather than quietly dropped: tts-long's image carried no
ffmpeg, so its formats were `wav`, `flac` and `pcm` and an mp3 request to it was
a hard `400` — while `mp3` is exactly what OpenAI's `response_format` defaults
to. That image carries ffmpeg now and answers all six formats at the same
bitrates tts-stack uses, so the two backends no longer differ on format. The
timing asymmetry above is what still rules length out on its own.

**A header was rejected too.** `X-TTS-Backend` works, and is invisible in the
surface that matters: Open WebUI and every other OpenAI-shaped client has a
`model` field in its settings and no custom-header field. A routing key nobody
can set is not a routing key.

**An unknown model goes fast rather than `400`.** The two wrong answers are
asymmetric — Kokoro on long-form costs some quality, Chatterbox on an ordinary
request turns 17 seconds into a job nobody asked for. Default to the
recoverable mistake, and do not break a client that sends whatever string its
UI was left holding.

**Length is a guard-rail, never a router.** There is no auto-escalation in
either direction. Long input on the fast path stays there, bounded by the
timeout rather than by an invented cap the backend does not have: 300 s at the
orko-measured 1.2x realtime is ~360 s of speech, about 900 words. Past that,
send `chatterbox` or split the text.

### The deviation, stated plainly

**With `model=chatterbox`, `POST /v1/audio/speech` may answer `202` with JSON
instead of audio bytes.**

The synchronous OpenAI contract cannot be honoured by tts-long. At the
orko-measured 0.138x realtime, a 200-word request is 80 s of speech and 580 s
of compute — 9.7 minutes. tts-long already made this call: input under
`TTS_OPENAI_SYNC_MAX_CHARS` is waited on and returned as audio, anything longer
or any wait that expires returns `202` with `{id, status, queued_ahead,
estimated_seconds, audio_url}` and a `Location` header. The audio is collected
from `GET /jobs/{id}/audio`.

The gateway neither invents this nor re-implements it. Its only job is to not
break it, which is why the long read timeout (240 s) sits **above** tts-long's
own `SYNC_TIMEOUT` (180 s): a gateway that timed out first would return `504`
for a job that is still running and will produce audio, and would throw away
the job id — a lie plus a leak.

The cost, honestly: `openai-python` does not raise on a 2xx, so it hands that
JSON to the caller as if it were audio, and `stream_to_file` will happily write
JSON into a `.wav`. Two things blunt it — the deviation is reachable only
through an opt-in model name, so no unmodified client meets it by accident, and
the `Content-Type` is `application/json` rather than `audio/*`, so a client that
checks can tell.

## Authentication

```bash
docker run -p 8080:8080 -e GATEWAY_API_KEYS=sk-workstation,sk-laptop \
  ghcr.io/gabrielbelli/ai-voice-gateway:pre
```

```bash
curl -H 'authorization: Bearer sk-workstation' localhost:8080/voices
```

The rules are tts-stack's, inherited deliberately rather than rewritten:

- **Unset means every request is accepted**, and the service says so at
  `WARNING` on every start — for all three backends, since this is the only
  thing checking a token. The stack runs on a LAN with no keys today, and an
  upgrade that starts rejecting every existing caller is a worse outage than a
  warning nobody reads.
- **Set but naming no key refuses to start.** `GATEWAY_API_KEYS=`, `,` and
  `,,  ,` all exit with a sentence. `-e GATEWAY_API_KEYS=$SECRET` with `SECRET`
  unset is a real accident, and it matters more here than in a backend: this
  process is the only thing checking a token for three services, so the
  accident opens all of them at once.
- **Enforcement is middleware over the whole app**, not per route, so a route
  added later is protected by default.
- **`/health` is the only exemption**, `/health/` included, matched after
  stripping a trailing slash. The TrueNAS healthcheck has no key and no way to
  be given one.
- Keys are compared with `hmac.compare_digest` over every configured key with
  no early break, on the latin-1 wire bytes against the UTF-8-encoded
  configured key. Both of those are corrections already paid for in tts-stack.
- `401` carries the OpenAI envelope with `code: invalid_api_key` and a
  `WWW-Authenticate: Bearer` header, and is answered before any backend is
  contacted.
- **The client's `Authorization` header is stripped before forwarding and is
  not replaced.** Relaying a key to three services that run with their keys
  unset achieves nothing except copying the secret into three more log streams.

### What this trades away

Anything already on the container network can call the backends
unauthenticated. On a single-user NAS where the three containers are one app,
an attacker on that network has already won — so the failure this invites is
operational, not adversarial: someone publishes 8001 to the LAN for a debugging
session and forgets.

Two cheap guards, in order of preference: publish nothing but 8080 and debug
with `docker exec` or curl from inside the app; or, if a backend port must be
published, set that one service's `*_API_KEYS` at the same moment and have the
gateway inject it.

A permanent shared internal token from gateway to backends was **rejected for
now**. It costs one env var per container and no new code, which is genuinely
cheap, but it re-creates the duplication this component exists to remove and
adds a secret to rotate, to defend a boundary already inside the trust domain.
Revisit if the stack stops being one app, or if a backend port becomes
permanently published.

## Failures

This service installs `packages/common` for one module: `voice_common.errors`.
It used to have a fourth hand-written copy of the envelope, which built three
keys where OpenAI's schema requires four — `param` is required-but-nullable and
was absent from every error this gateway ever emitted — and whose
`error_response(status, message, type_, code)` took the two strings
*positionally*, which is the exact swap the shared function is keyword-only to
prevent. Auth, health and the entrypoint stay local: this service has its own
env var, fans health out to three backends rather than reporting on itself, and
has no TLS or volume for the shared entrypoint to handle.

Under `/v1` a 404 and a 405 now read exactly as they do from the three
backends. The **native** routes keep the wording and the `code` values they
had, `method_not_supported` included: `/transcribe`, `/speak`, `/voices` and
`/jobs` have clients that may already branch on them. Their only change is the
`param` key the schema requires.

**Governing rule: if the backend produced an answer, forward it unchanged** —
status, body and all. All three services already emit the OpenAI envelope under
`/v1` with distinct `code` values (`model_loading`, `invalid_value`,
`unsupported_value`, `synthesis_failed`, `missing_required_parameter`), and
re-wrapping destroys the code the client switches on. Note stt-stack answers a
rejected body with `422` where the two TTS services answer `400`: both pass
through as-is. Normalising that difference would make the gateway a second,
lying source of truth.

| Failure | Answer |
|---|---|
| Backend down, restarting, no DNS | `503`, `Retry-After: 30`, `code: backend_unavailable`, **naming the service** |
| Backend still loading (fast path) | its own `503 model_loading`, untouched, plus `Retry-After: 10` |
| Backend still loading (long path) | **not an error** — see below |
| Read timeout | `504`, `code: backend_timeout`, message carries the measured rate *and* the way out |
| Non-JSON body on a 5xx | `502`, `code: backend_error`, first 200 bytes quoted, whole body in the log |
| Malformed JSON on `/v1/audio/speech` | `400`, `code: invalid_value` — the only body validation performed |
| Missing or wrong bearer token | `401` + `WWW-Authenticate: Bearer`, before any backend is contacted |
| Client disconnects mid-request | upstream connection closed and logged; a tts-long job is **not** cancelled |
| Anything not in the route table | `404`, `code: unknown_url` |

`503` rather than `502` for an unreachable container, because `502` claims the
upstream answered badly and it did not answer at all — and because
`openai-python` retries 5xx twice by default, which for a container mid-restart
is exactly right and costs nothing, since a refused connection fails in
microseconds. The service is named because with three backends behind one URL,
"upstream failed" is unactionable.

**A cold tts-long is never gated on `model_loaded`.** It reports
`model_loaded: false` for minutes on a cold start — 6.5 GB, lazy, unloaded
after 600 s idle, ~3 GB downloaded on the very first job — and `POST /jobs`
accepts work regardless, because the queue absorbs it and the worker loads the
model. A gateway that read that field and synthesised a `503` would reject the
request that was about to warm the model, for the entire cold-start window,
turning a working design into an outage.

Timeouts are per route, from the two measured rates, not one global number:

| Route | Read timeout | Why that number |
|---|---|---|
| STT | 900 s | 8.5-10.4x realtime: two hours of audio is ~847 s of compute |
| Fast TTS | 300 s | 1.2x realtime: ~360 s of speech, about 900 words |
| Long TTS | 240 s | only to sit above tts-long's own 180 s `SYNC_TIMEOUT` |
| Connect | 2 s | a container on the same host either accepts immediately or is not there |

A client disconnecting mid-request does **not** cancel a tts-long job: the job
runs to completion, the audio lands on disk, and the id was already handed over
in the `202`. Abandoning half-finished 6.5 GB work to save disk would be the
worse trade.

## Logging

One line per request, and that is the entire observability budget:

```text
route=/v1/audio/speech POST backend=tts-stack model=tts-1 status=200 duration=3.872 rtf=0.6
```

Route, backend chosen, the `model` string as the client sent it, upstream
status, the duration this process observed, and the backend's own
`X-Realtime-Factor` where it sends one. `grep` answers every question asked so
far, which is why there is no Prometheus, no OpenTelemetry and no sidecar for
three containers and one user.

A request that never reached a backend still writes its line, with `backend=-`
and a status naming what went wrong instead of a code — `client-disconnect`
when the caller hung up mid-upload (answered `499`, nginx's code for it, which
never leaves this process) and `400-badjson` when the body could not be parsed
to route it. Both are on `POST /v1/audio/speech`, the one route that has to
read the whole body before it can choose. `duration` on those lines is the
time the client actually waited, not zero.

`rtf` is populated from the `X-Realtime-Factor` **header**, which only
tts-stack sends. stt-stack and tts-long report `realtime_factor` in their JSON
bodies, and the gateway does not read it: reading it would mean buffering a
response this service exists not to buffer. Those numbers are still in the body
the client received.

httpx's own per-request log line is silenced — in a proxy it is exactly one
duplicate per request, carrying neither the model nor the duration. uvicorn's
access line is left alone; pass `--no-access-log` if one line per request is
meant literally.

## Deploying

The three services and this gateway are **one TrueNAS app**, which is the whole
reason the single boundary is free: the containers already share a network.

1. Publish `8080` **only**. 8000, 8001 and 8002 stay on the app-internal
   network. This is the step that makes the boundary real; without it this
   component is an extra hop and nothing else.
2. Leave `STT_API_KEYS` and `TTS_API_KEYS` unset on all three backends.
3. Set `GATEWAY_API_KEYS` here.
4. Leave each container's own healthcheck pointed at **its own** localhost
   `/health`, not at the gateway's aggregate. A container must not be
   restarted because a sibling is down.

`GET /health` here always answers `200`, even when a backend is unreachable —
read `status`, not the status code. The healthcheck for *this* container calls
this endpoint, and a `503` because tts-long is cold would have the orchestrator
restart the gateway for a sibling's fault.

One backend setting is worth changing at the same time, and it belongs in
tts-long rather than here: **set `TTS_OPENAI_SYNC_MAX_CHARS=180` on orko.** The
300 default was derived from 0.21x realtime, which is the M2 Max number in
tts-long's own README; orko measures 0.138x, so 300 characters is ~21 s of
speech and ~155 s of compute against a 180 s `SYNC_TIMEOUT` — no headroom at
all, and none whatsoever for a cold start that loads 6.5 GB. At 180 characters
it is ~13 s of speech and ~93 s of compute. The gateway must not compensate for
this in code.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_API_KEYS` | unset | Comma-separated accepted keys. Unset means no authentication; set but naming none refuses to start |
| `GATEWAY_STT_URL` | `http://stt-stack:8000` | |
| `GATEWAY_TTS_URL` | `http://tts-stack:8001` | |
| `GATEWAY_TTS_LONG_URL` | `http://tts-long:8002` | |
| `GATEWAY_STT_TIMEOUT` | `900` | Read timeout, seconds |
| `GATEWAY_TTS_TIMEOUT` | `300` | Read timeout, seconds |
| `GATEWAY_TTS_LONG_TIMEOUT` | `240` | Must stay above tts-long's `TTS_OPENAI_SYNC_TIMEOUT` |
| `GATEWAY_CONNECT_TIMEOUT` | `2` | |
| `GATEWAY_HEALTH_TIMEOUT` | `5` | Per backend, fanned out concurrently |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

63 tests against mock backends wired in through httpx's own transport layer, so
every one of them runs the real proxy code — header filtering, streaming,
timeout mapping, auth — with only the socket replaced.

`tests/test_live.py` runs the real app against the three live containers on
orko over real sockets, and **skips itself** when they are unreachable, which
from a CI runner they should be. It queues no Chatterbox work: tts-long runs
one job at a time on a 6.5 GB model, so the long path is exercised read-only
through `GET /jobs`.

## What is not here

Rejected deliberately, each against the same budget: a 2-second dictation clip
is 190-240 ms of recognition at the measured 8.5-10.4x, and a feature that adds
20 ms to that has taken 10% of the interactive path.

- **Retries.** Both TTS routes are non-idempotent — retrying `POST /jobs`
  enqueues a second 6.5 GB job, retrying `POST /speak` burns a second full
  synthesis on a box that is already CPU-bound — there is nowhere to retry to,
  and `openai-python` already retries 5xx twice, so a gateway budget would
  multiply with the client's rather than replace it.
- **Caching.** The hit rate on arbitrary dictated text is approximately zero,
  and the bodies are audio. If a cache is ever justified it belongs *inside*
  tts-stack, keyed on `(text, voice, language, speed, format)` where those are
  already resolved — at the gateway the key would drift from tts-stack's own
  voice-alias and language-inference tables the first time either changed.
- **Rate limiting.** One user, one host. The real limiter is physical and
  already implemented: tts-long runs exactly one job at a time by design, and
  the other two block on CPU in worker threads. A token bucket would only
  convert "slow" into "rejected".
- **Load balancing.** One instance of each service, one CPU. A second Kokoro
  replica would halve the ONNX thread pool available to each.
- **Circuit breakers**, which would do active harm: a breaker cannot tell
  tts-long's minutes-long cold start from an outage, so it would trip on the
  slow first request and then reject exactly the requests that warm the model.
  The `503` with `Retry-After` does the whole job.
- **A health pre-flight before each request.** It doubles the request count on
  the interactive path, it races, and for the long path `model_loaded: false`
  is a normal state in which `/jobs` must still be accepted.
- **Body rewriting beyond reading `model`.** No voice mapping, no language
  inference, no format translation, no error normalisation. tts-stack already
  maps all thirteen OpenAI voice names onto Kokoro voices, and tts-long keeps
  its own clip registry with a different mechanism again; a second table here
  means two tables and a guaranteed drift. The last time that table changed it
  went from six names to thirteen, which is exactly the drift a copy would have
  missed.
- **A unified job abstraction over both TTS backends.** It means the gateway
  holds job state, and then needs storage, restart survival, its own `/jobs`
  endpoints, and an answer for in-flight jobs when it redeploys. tts-long
  already has all of that, and the flat mount makes its URLs work unchanged.
- **Gateway-side streaming or SSE synthesis.** Both TTS backends stream for
  real now — `stream_format: "sse"` on `/v1/audio/speech` emits
  `speech.audio.delta` events as each chunk is encoded, through ffmpeg on a
  pipe rather than a temporary file — and this service already forwards a
  response body as it arrives, so those events reach the client untouched with
  nothing added here. What stays rejected is a gateway-side *fake*: buffering a
  complete response and dribbling it out to pretend at a capability a backend
  does not have. There is nothing to pretend at, and inventing an SSE frame
  here would put a second copy of the event shape in front of the one the
  backend defines.
- **TLS, CORS and certificates**, unlike the siblings, which do carry TLS. It
  is LAN traffic on one host; if it is ever exposed beyond the LAN that belongs
  to whatever reverse proxy already terminates TLS for the NAS.
- **WebSocket or realtime STT.** stt-stack is file-in, transcript-out with a
  VAD stage; there is no partial-hypothesis interface to expose.

## Honest limitations

- **It is not deployed, and it earns nothing until the app config stops
  publishing 8000, 8001 and 8002.** Everything above about a single auth
  boundary is a property of that config change, not of this code.
- **The hop is not free, though it is close.** Measured on a laptop over
  loopback with a trivial JSON body, 300 requests: 0.32 ms median direct
  against 1.17 ms through the gateway — **0.85 ms added**. That is under 1% of
  a 200 ms dictation turn. It has not been measured on orko, whose CPU is
  slower and busier, and it will be larger there.
- **`rtf` in the log is only ever tts-stack's**, because it is the only backend
  that sends the header. The other two put `realtime_factor` in the body, and
  reading a body to log it is exactly the buffering this service avoids.
- **A `202` from the long path will be written into a `.wav` by an unmodified
  OpenAI client.** Documented above; reachable only through an opt-in model
  name.
- **Client-disconnect handling is asserted by reading, not by a test.** The
  code path closes the upstream connection and logs, and the mocked suite
  cannot cut a connection mid-flight through an in-process ASGI transport.
- **No test covers a backend that starts a response and then stalls**
  mid-body. The code logs it and truncates, because the status line has
  already gone out and there is nothing left to change.
- **`/health` fans out on every call with no cache.** Three concurrent local
  requests, 5 s timeout each. Something polling it once a second would triple
  that rate onto the backends; nothing does today.
- **Something already answers on `orko:8080`.** The host port for this app has
  to be chosen at deploy time; the container port is what the contract fixes.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
