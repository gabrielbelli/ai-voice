# voice-ui

One page in front of the gateway, and the three things a browser cannot do for
itself.

```
  http://orko.gabrielbelli.com:30081/ui
        |
        |  every XHR, same origin, Authorization: Bearer <key>
        v
  voice-ui:8090  ──────────────► voice-gateway:8080 ──► stt-stack / tts-stack / tts-long
        │
        ├─ /ui/resolve /commit /abandon /progress /fetch ──► MeTube (by host address)
        └─ /ui/clips                                     ──► the shared `voices` volume
```

Three tabs — **Transcribe**, **Speak**, **Jobs** — an easy mode that needs no
manual, and an expert toggle that reveals the real knobs beneath the controls
already on screen without swapping pages or losing what you typed.

---

## The short version

| | |
|---|---|
| **Reached at** | `http://<host>:30081/ui` — 30081, next to the gateway's 30080 |
| **Talks to** | the gateway, and MeTube. Never `:8000`, `:8001` or `:8002` |
| **Auth** | the gateway's. This service holds no key list and compares no token |
| **Image** | 320 MB, measured. The gateway is 286 MB on the same machine and `python:3.13-slim-trixie` is 215 MB |
| **Build step** | none. One HTML file, inline CSS and JS, no framework, no `node_modules`, no CDN |
| **Degrades** | MeTube down or unset → link box hidden or disabled, uploads and TTS unaffected. Gateway down → the page still loads and says so |

---

## Why this is a fifth container, and what it costs

The framework survey argued for serving the page from the gateway itself, and
the argument was good: the page is static, the gateway is already the only
published port, and this estate is organised around not adding containers.

Two things decided otherwise.

The gateway's Containerfile makes a specific promise — **283 MB**, described in
its own comment as *"the cheap check on this file — an audio or model library
arriving by accident moves it by gigabytes, not megabytes"*. Ingestion needs
`yt-dlp`. Putting `yt-dlp` in the gateway would make the process that holds
every API key also the process that spawns a subprocess on a URL a browser
chose. Separate images keep that blast radius where it is.

And the brief asked for `services/ui` with a Containerfile, a compose entry, a
workflow matrix row and a README like its four siblings.

**What it costs is a second `ports` entry**, and `compose.yaml` says at the top
that one appearing means the file has stopped doing its job. That sentence was
written about 8000, 8001 and 8002 — three backends reachable *directly*,
skipping the only process in the stack that checks a token. Those three are
still closed. This one is a browser origin, not a bypass:

- it reaches the gateway and nothing else;
- it forwards the caller's `Authorization` header untouched;
- it holds no key list — it asks the gateway whether a presented key is good,
  using `GET /v1/models`, the cheapest authenticated call in the stack (a
  static table, no backend contacted), cached for 60 s;
- it can answer nothing the gateway would not have answered.

If a future edit gives this container a URL for `stt-stack`, `tts-stack` or
`tts-long`, that is the moment the paragraph above stops being true.

### Why the page's XHRs come back here rather than going straight to :30080

One origin means no CORS on the gateway, no preflight on every upload, and no
second base URL for someone to get wrong. The forwarding table in `app/main.py`
is a **fixed allowlist** — no wildcard, no catch-all, for the reason the
gateway has none: a wildcard would proxy `/docs` and `/openapi.json` to
services that deliberately do not publish them.

---

## Security: read this before setting `UI_METUBE_URL`

**MeTube has no authentication of any kind.** Its configuration has no `auth`,
`user`, `password` or `token` key; `/add`, `/start`, `/delete`, `/retry` and
`/history` are all open, and an unauthenticated `GET /history` from off-NAS
answers 200. That is true today, with or without this service.

This service does not widen it — our ingestion routes sit behind
`GATEWAY_API_KEYS`, so we are a strictly narrower client of something already
open to the LAN. But **shipping a UI that makes MeTube load-bearing is the
moment to close it**: after this deploys, an outage or an abuse of port 30097
becomes an ai-voice outage. The fix is not in this code. Unpublish 30097, or
firewall it to the NAS, and point `UI_METUBE_URL` at the LAN IP.

### The SSRF story, in three layers

`app/guard.py` carries the full reasoning; the short form:

1. **Our stdlib pre-filter.** http/https only, no userinfo, ports 80 and 443
   only, then `getaddrinfo` and a check of **every** address the name resolves
   to — loopback, RFC 1918, ULA, link-local (which includes
   `169.254.169.254`), CGNAT, multicast, reserved, unspecified, and IPv4-mapped
   IPv6 of any of those. `localhost*`, `*.local`, `*.internal` and
   `metadata.google.internal` are refused by name before resolution is even
   attempted.
2. **MeTube's own `url_guard.validate_url`,** which runs inside its `POST /add`
   and is better than ours: ingress validation *plus* a connect-time
   `getaddrinfo` hook installed in the download subprocess, so it covers
   redirects and DNS rebinding during the download itself.
3. **Our probe, third,** and only on a URL both guards have already accepted.

That ordering is deliberate and it has a cost: because `POST /add` runs before
the probe, every link the user declines has left a pending record in MeTube. So
`/ui/abandon` is not a nicety, it is the other half of `/ui/resolve` — and it
**verifies**, because MeTube's `/delete` answers `{"status":"ok"}` when it
deletes nothing at all.

**Still open, stated rather than hidden.** yt-dlp's extraction follows
redirects, and neither guard covers a redirect *during* extraction to an
internal host; MeTube documents that exact limitation in its own docstring.
The impact is *blind* SSRF — the probe's output is parsed into five scalars,
nothing is written to disk, and no response body is ever returned to a caller.
**The real backstop is network isolation:** this container has no business
reaching the NAS's other services, and an egress rule on the ai-voice app is
the fix. Write it down; do not assume it.

`UI_PROBE=0` removes the probe entirely, at the cost of a title-only confirm
card.

---

## Ingestion: what MeTube can and cannot do

`auto_start:false` on `POST /add` **genuinely resolves without downloading** —
verified against MeTube's source and live. `POST /start {ids}` commits;
`POST /delete {ids, where}` abandons. `download_type:"audio"` means a two-hour
4K video never has its video stream pulled: about 1 MB a minute at opus, so
~131 MB for a 2h14m podcast rather than tens of gigabytes.

**But there is no duration and no size in what MeTube exposes.** Its
`DownloadInfo` has no duration field anywhere in its source, `size` stays
`null` until the download finishes, and the full yt-dlp info-dict is stored and
then deliberately stripped (`_PUBLIC_EXCLUDED_FIELDS`). MeTube resolves
duration and throws it away.

So the confirm dialog needs a **metadata probe** of our own —
`yt-dlp -J --skip-download --no-config --no-cache-dir`, in a subprocess with a
20 s hard kill and a `cwd` it cannot write, whose output is reduced to five
scalars. That is not embedding a downloader: there is no output template, no
writable path, no format selection, no post-processing, no cookies, no
concurrency and no disk. Its failure mode is a missing estimate, never a
blocked fetch.

### Sharp edges, all verified

| Fact | Consequence |
|---|---|
| `ids` in `/start` and `/delete` are **URLs**, not the short `id` field | A wrong id returns `{"status":"ok"}` and silently does nothing |
| An `auto_start:false` item lands in **`pending`**, not `queue` | `/history` returns three lists; reading `queue` finds nothing |
| `auto_start` defaults to **true** when the field is `None` | It is always sent explicitly |
| `filename` is `OUTPUT_TEMPLATE` sanitised and byte-trimmed | **Never predicted** — read back from `/history` and percent-encoded |
| Terminal success is `status == "finished"` | Anything else in `done` means the status was rewritten to `error` and `filename` nulled |
| A *rejected* add still creates a record, in `done` | Abandon clears both queues |
| `DELETE_FILE_ON_TRASHCAN` defaults false and is unset here | `/delete where=done` clears the record and **leaves the file** — see cleanup below |
| `AUDIO_DOWNLOAD_DIR` defaults to `%%DOWNLOAD_DIR`, unset here | `/download/` and `/audio_download/` are the same directory, which is why `UI_METUBE_FOLDER` is mandatory in effect |
| `CORS_ALLOWED_ORIGINS` is empty | A browser **cannot** call MeTube at all. Every call is server-side from this container, which is the right shape anyway |

**Cleanup is the one thing delegation does not solve.** Files accumulate in
`stt-ingest/` and nothing removes them: we have no shared volume to delete
through, and setting `DELETE_FILE_ON_TRASHCAN=true` is global and would make
the user's own trashcan button delete their music. Start with a TrueNAS cron
pruning that directory by mtime. Move to a second, dedicated MeTube instance
with its own dataset if ingest volume ever gets real. Do not silently pick the
global flag.

---

## Uploads, including video

Video files work and always did: `services/stt/app/audio.py` opens the bytes
with PyAV, takes `container.streams.audio[0]` and ignores the video entirely.
There is no ffmpeg package and no shell-out anywhere in this feature.

Two things the page handles because the backend does not.

**There is no size ceiling in the stack.** `services/stt/app/main.py:138` is a
bare `file.file.read()` on an `UploadFile` — no `Content-Length` check, no cap,
no streaming — so a 4 GB MKV is buffered whole into a container limited to
6 GB, and the failure is an OOM kill rather than a message. This service
rejects on `Content-Length` above `UI_MAX_UPLOAD_BYTES` (2 GiB by default)
before a byte is forwarded.

**The browser extracts the audio first.** `decodeAudioData` →
`OfflineAudioContext` at 16 kHz mono → a hand-written WAV header turns a 2 GB
MKV into about 15 MB before it crosses the network. That also unlocks the
native `/transcribe` route, which calls `decode(allow_resample=False)` and 400s
with *"expected 16000 Hz, got 44100 Hz"* — video audio is always 44.1 or
48 kHz, so without this the richer response and its `repaired` glossary field
are unreachable for every video anyone owns. Above ~500 MB the file is uploaded
raw with a warning, because `decodeAudioData` needs the whole thing resident.

The same fifty lines serve the reference-clip path at 24 kHz. **One transcoder,
two features, zero packages added to any image.**

---

## Voice cloning, and the two small changes it needed

On this stack a voice **is a file**: `TTS_VOICE_DIR/gabriel.wav` is the voice
`gabriel`, resolved by tts-long's registry and handed to Chatterbox as
`audio_prompt_path`. There is no per-request reference field on the wire at any
layer.

So "use my own voice" is: record or drop a clip, the browser transcodes it to
24 kHz mono WAV, this service writes it into a volume shared with tts-long, and
the voice is selectable. In the picker it is one more row — the user never
chooses between Kokoro and Chatterbox, because **picking the voice picks the
engine**, and each row carries its own speed tag.

Two changes elsewhere made that possible:

- **`compose.yaml` gained a named `voices:` volume.** The deployed compose
  mounted none, so tts-long's registry was `["default"]` only and all thirteen
  OpenAI names aliased to the built-in speaker. Without a *named* volume every
  cloned voice would die on the next restart, which is worse than not shipping
  the feature. This service is the only writer; tts-long mounts it read-only.
- **`services/tts-long/app/voices.py` rescans on mtime** instead of once at
  startup. The original argument — no directory listing per request — is kept:
  it is one `stat()`, and a listing happens only on the request after something
  in the directory changed. What it removes is the restart, which on a
  container holding 6.5 GB of Chatterbox costs a ~60 s model load on its next
  job. This is a strict improvement to tts-long on its own terms: a clip copied
  in over SMB now works too.

The write sequence is devnen/Chatterbox-TTS-Server's (MIT, `server.py:670-753`)
transplanted in shape — sanitise, extension allowlist, write, validate duration,
unlink on failure, return the refreshed list — with three things added that
upstream lacks and that matter behind a gateway: a size cap, a collision
policy, and a name built from a `[a-z0-9_-]` **whitelist** rather than from a
sanitiser applied to the uploaded filename. Upstream's `sanitize_filename` is
the only thing between a caller and path traversal; "the only thing" is not
where a write path belongs.

Only `.wav` is accepted, because the browser always sends one. That is what
lets the server-side validation be the stdlib `wave` module rather than
librosa.

---

## Long jobs

Chatterbox runs at about **0.138× realtime**, so a five-minute voice note is
about thirty-six minutes of compute. The whole answer is that the user closes
the tab.

- Submission is **`POST /jobs`**, never `/v1/audio/speech`. `/jobs` answers 202
  every time — no 200-vs-202-vs-SSE to branch on — and hands back `chunks` and
  its own `estimated_seconds`, which the page prefers over its own arithmetic.
  It costs the format field (`/jobs` always produces WAV) and that is the right
  trade: an SSE stream dies when a laptop lid closes, and **closing the stream
  cancels the job**.
- The list is rebuilt from `GET /jobs` **and** reconciled with the ids this
  browser remembers, so it survives a reload, a browser restart and a different
  device, and it covers the window where a 202 landed before the server's list
  caught up.
- Polling is a backoff ladder (2 s → 10 s → 30 s) **layered on
  `document.visibilityState`**, so a background tab does not hammer a NAS that
  is simultaneously running CPU-only Parakeet.
- **The progress bar is honest.** `GET /jobs/{id}` carries `chunks` but no
  per-chunk counter, so the only bar available is time against
  `estimated_seconds`. It is capped at 95 % until the job says `done`, and it
  relabels itself to *"still going"* when it overruns. A bar that reaches the
  end and sits there teaches people to distrust every bar you ever show them.
  The real fix is a completed-chunk counter in tts-long's worker — a handful of
  lines, and it would turn this estimate into a fact.
- **Cancel exists now.** The button reads *"Stop and keep what's done"*,
  because `/jobs/{id}/audio` serves a `cancelled` job as happily as a `done`
  one. tts-long has had `DELETE /jobs/{job_id}` all along; the gateway simply
  never routed it, and this change adds those three lines.
- Before submission, anything over ten minutes of compute gets a strip that
  **offers the cheap path as a button** — *"a Kokoro voice would do it in
  about 30 seconds — [Use it instead] [Queue it anyway]"* — and the ETA is in
  the button label itself. Nobody presses *"Generate — about 36 minutes"* by
  accident. Kokoro's button just says *Speak*; the asymmetry is the message.

STT gets none of this ceremony, deliberately. Parakeet is fast enough that a
spinner is the correct UI.

---

## The estimate, and the number this repository contradicts itself about

The confirm dialog quotes a transcription time, and the rate behind it is
stated three different ways in this repository:

| Source | Claim |
|---|---|
| root `README.md:95` | 47–63× realtime |
| `services/gateway/app/main.py:117-123` | **8.5–10.4×** — and the 900 s `GATEWAY_STT_TIMEOUT` and the 504 help text are built on this one |
| `services/stt/README.md:590` | about 5× on four cores |

A factor of twelve apart. At 47× the brief's flagship example — a 2h14m podcast
— is about two minutes; at 8.5× it is about sixteen minutes and **946 s of
compute, which exceeds the gateway's own 900 s ceiling**, so the honest dialog
for that file says *"this will not finish in one request — trim it"*.

So: nothing is hardcoded. `UI_STT_RTF` seeds the **conservative** figure, the
page keeps its own EMA in `localStorage` corrected by the `realtime_factor`
every native transcription returns, the number is labelled an estimate, and the
dialog warns whenever `duration / rtf` crosses `UI_STT_BUDGET`.

**Someone must re-measure on orko before this is trusted**, and correct
`main.py:121`'s `timeout_help` in the same change — otherwise this page and the
gateway's own 504 message will disagree in front of the same user.

The two halves are **never blended into one figure**. Download and transcribe
are separate lines, because for long media the download is the slow half and
one merged number hides which half to blame. The download half is shown as a
**size**, not a time: this container has no idea what the source's bandwidth
is, and MeTube reports the real `speed` and `eta` once it is actually
downloading.

---

## What the page hides, and why each one is right

- **`language`, in either mode.** It is a 400 `unsupported_parameter` on `/v1`
  under Parakeet and silently ignored on `/transcribe`
  (`accepts_language = False`). `STT_LANGUAGE` is deliberately unset on this
  deployment because pinning it breaks the English/Portuguese code-switching
  the user actually does — agreement collapsed to 0.017 when the service
  translated instead of transcribing. A control that does nothing, or does
  harm, is worse than no control.
- **A denoise toggle.** There is nothing to toggle: the pipeline is
  decode → VAD → ASR → glossary, with no preprocessing stage anywhere.
  Denoising measured **+26 % mean WER**, worse in 9 of 13 conditions, one case
  above WER 1.0 from hallucination. Expert mode carries this as a note so
  nobody adds it back.
- **A per-request glossary box.** There is no such field — glossary is a
  startup file, and `prompt`/`keywords[]` are both 400 on Parakeet. An
  *irrelevant* glossary cost +12 % WER on Parakeet and +28 % on Whisper, which
  is a finding no slider can express.
- **`model` on STT.** Required by `/v1` validation, but it does not choose an
  engine — Parakeet runs regardless and says so in `x-stt-engine`.
- **The TTS language dropdown, on the fast path.** It is *inferred* from the
  voice prefix (`a`→en-us, `b`→en-gb, `p`→pt-br, …). This is the single best
  easy-mode win available: `/speak` defaults to `en-us`, so a UI that simply
  omitted the field would mispronounce every Portuguese request while looking
  entirely correct.

### What expert mode shows

STT `response_format`, `timestamp_granularities[]` (auto-switching to
`verbose_json`), `include[]=logprobs` (auto-switching to `json`), the three
real Silero VAD knobs **at this deployment's defaults** (0.5 / 100 / 300, which
are not OpenAI's), and the route choice — with the native route *disabled with
a stated reason* when the selected file is not 16 kHz.

Kokoro: the `language` override, `format` (noting the field is `format` on
`/speak` and `response_format` on `/v1`, with *different defaults*),
`stream_format`, a segments editor with `pause_after` and per-segment voice,
and the route choice — surfacing `X-Ignored-Parameters` and `X-Speed-Clamped`,
because a field that is accepted and ignored is the worst kind.

Chatterbox: `exaggeration`, `cfg_weight` and `temperature`, each labelled
**"this deployment is calmer than stock"** (0.3 / 0.3 / 0.6 against 0.5 / 0.5 /
0.8), with a *Reset to deployment defaults* button — three sliders at non-stock
values is exactly the state people get lost in. Resemble's demo offers
exaggeration 0.25–2.0; our backend validates `ge=0.0, le=1.0`, so those ranges
are reconciled rather than copied.

**What expert mode still does not show**, because expert mode is not every
environment variable: `STT_VAD`, `STT_HOTWORDS`, `STT_THREADS`,
`STT_MAX_CONCURRENT`, `STT_QUANTISATION`, `TTS_THREADS`, `TTS_MAX_QUEUE`,
`TTS_JOB_TTL`, every `GATEWAY_*` timeout, and every `/v1` field that is an
unconditional 400 on this stack. Startup configuration is startup
configuration. A control that cannot change the outcome is a lie with a slider
on it.

---

## Configuration

Every variable is optional and every default degrades rather than fails.

| Variable | Default | What it does |
|---|---|---|
| `UI_GATEWAY_URL` | `http://voice-gateway:8080` | The only speech address this service knows |
| `UI_METUBE_URL` | *(unset)* | MeTube, **by host address**. Unset hides the link box entirely |
| `UI_METUBE_FOLDER` | `stt-ingest` | Mandatory in effect — see the table above |
| `UI_METUBE_FORMAT` | `opus` | ~1 MB a minute. MeTube 400s on any `quality` but `best` for it |
| `UI_PROBE` | on | `0` removes yt-dlp from the running system; the card degrades to a title |
| `UI_PROBE_TIMEOUT` | `20` | Hard kill, not a suggestion |
| `UI_MAX_UPLOAD_BYTES` | 2 GiB | Checked on `Content-Length` before a byte is forwarded |
| `UI_CONFIRM_SECONDS` | `600` | Below this **and** the size threshold, no dialog |
| `UI_CONFIRM_BYTES` | 50 MiB | The second gate, not an alternative |
| `UI_STT_RTF` | `8.5` | The conservative seed. The page measures its own |
| `UI_STT_BUDGET` | `900` | `GATEWAY_STT_TIMEOUT`. Crossing it warns |
| `UI_VOICE_DIR` | `/voices` | The reference-clip store, shared with tts-long |
| `UI_MAX_CLIP_BYTES` | 25 MiB | |
| `UI_MAX_CLIP_SECONDS` | `30` | Trimmed client-side, enforced server-side |
| `UI_RESOLVE_PER_MINUTE` | `12` | So `/ui/resolve` is not a free scanner |
| `UI_VOLUMES` | `/voices` | What the entrypoint takes ownership of before dropping to uid 1000 |

### Why the confirm thresholds are those numbers

Ten minutes of audio is ~10 MB at opus and, at the conservative 8.5×, about
71 s of transcription — an order of magnitude inside the gateway's 900 s
ceiling and well inside anyone's patience. A dialog there is pure friction, and
**a dialog that fires on everything is a dialog people dismiss without
reading** — which is exactly how the three-hour stream gets through. The size
threshold is a second gate rather than an alternative, because a short video
with an enormous audio stream is still a real download. An *unknown* duration
always confirms: not knowing is the case the dialog exists for.

---

## Running it

```bash
# Build, from the repository root — the context is the root for every service
docker build -f services/ui/Containerfile -t ai-voice-ui .

# Run, on the network the gateway shares
docker run -p 30081:8090 \
  -e UI_GATEWAY_URL=http://voice-gateway:8080 \
  -e UI_METUBE_URL=http://192.0.2.10:30097 \
  -v voices:/voices \
  ai-voice-ui
```

`compose.yaml` at the repository root wires all of it, including the healthcheck
— which is **not** in the Containerfile, and that is not an oversight:
`HEALTHCHECK` is not a field in the OCI image spec, so an OCI-format build
drops it silently, and OCI is buildah's default format, which is what CI runs.
The four sibling images have none for the same reason.

### Tests

```bash
pip install -r services/ui/requirements-dev.txt   # from the repository root
cd services/ui && pytest -q
```

**No test starts a server, and none may.** The suite is
`fastapi.testclient.TestClient` over an httpx `MockTransport` standing in for
both the gateway and MeTube, so the whole resolve → confirm → fetch flow, the
forwarding table, the upload ceiling and the clip store run in-process with no
socket anywhere. `yt-dlp` is never spawned — `app.probe.run` is replaced.

---

## Layout

| File | |
|---|---|
| `app/static/ui.html` | The whole UI. Inline CSS and JS, no build step, no external request of any kind — it works on a NAS with no internet |
| `app/main.py` | The forwarding allowlist, the key check, the upload ceiling, the clip routes |
| `app/ingest.py` | Resolve, commit, abandon, progress, fetch |
| `app/metube.py` | A narrow client, with every verified trap written down |
| `app/probe.py` | Five scalars out of a URL, and not one byte of media |
| `app/guard.py` | What a pasted URL has to survive. **Read this before relaxing anything in it** |
| `app/clips.py` | The reference-clip store |
| `app/config.py` | Every knob, with the measurement behind each default |

## What we reused rather than wrote

- **MeTube** (AGPL, called over HTTP so no licence reach) — the entire
  downloader: extractors, cookies, retries, concurrency, format selection,
  audio-only extraction, an SSRF guard better than ours, and file serving with
  Range support. Delegating means no extractor rot to chase and, because
  `/audio_download/` is HTTP, **no shared volume between two TrueNAS apps**.
- **devnen/Chatterbox-TTS-Server** (MIT, `server.py:670-753`) — the
  reference-audio upload sequence, transplanted in shape.
- **resemble-ai/chatterbox** (MIT) **as a specification only** — the parameter
  ranges and the four-visible-plus-accordion layout. Its ranges are reconciled
  against ours, not copied.
- **speaches-ai/speaches** (MIT, `src/speaches/ui/app.py:14-88`) — the
  `localStorage` API-key box with show/hide, lifted as a pattern into vanilla
  JS. The rest of that fork carries imports into `speaches.config` and two
  dropdown wirings that are wrong for our API.
- **PyAV**, already in the stt image — video upload already worked; nothing was
  added for it.
- **tts-long's job queue, chunking and ETA arithmetic** — the page reads them,
  it does not reimplement them.
- **`<dialog>.showModal()`, `OfflineAudioContext`, `Notification`** — the
  modal, the transcoder and the ping. Zero dependencies for all three.
