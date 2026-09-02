# ai-voice

Self-hosted speech-to-text and text-to-speech. Four images and the package they
share, one repository, deployed as a single app behind one published port.

```text
                       :8080  services/gateway
                         │
  /v1/audio/transcriptions├───────────────────────►  services/stt       :8000
  /transcribe             │                          Parakeet, ONNX, no torch
                          │
  /v1/audio/speech  model=│kokoro tts-1 …       ──►  services/tts       :8001
  /speak  /voices         │                          Kokoro-82M, 54 voices
                          │
  /v1/audio/speech  model=│chatterbox tts-long  ──►  services/tts-long  :8002
  /jobs  /jobs/{id}[/audio]                          Chatterbox, a job queue
                          │
  /v1/models              ├─ answered at the gateway
  /health                 └─ all three, fanned out, no key required

  packages/common            the wire contract all four services import
```

Every service keeps its own README, and those are the reference: what each
route accepts, every OpenAI deviation and the measurement forcing it, the
configuration table, the deployment notes. This file is about the repository.

| | | |
|---|---|---|
| [`services/stt`](services/stt/README.md) | `ai-voice-stt` | Parakeet TDT 0.6B v3 by default, Whisper large-v3 on request |
| [`services/tts`](services/tts/README.md) | `ai-voice-tts` | Kokoro-82M, 54 voices, six output formats |
| [`services/tts-long`](services/tts-long/README.md) | `ai-voice-tts-long` | Chatterbox, a queue and an SSE stream |
| [`services/gateway`](services/gateway/README.md) | `ai-voice-gateway` | One address, one key, one health answer |
| [`packages/common`](packages/common/README.md) | — | Auth, the error envelope, `/health`, the entrypoint |

## Why one repository

Five repositories, and `packages/common` was consumed by three of them as a
GitHub tarball pinned to a commit SHA. That pin was correct: a moving ref in a
`requirements.txt` is a supply-chain decision made by whoever pushed last, and
an image build has to produce the same bytes next month.

It also meant a one-line fix in the shared auth middleware reached a service
only after being pushed, tagged, re-pinned in that service's requirements, and
built. Three times. The three copies of `app/auth.py` that preceded the shared
package had already proved what happens in that gap — they drifted by 170 to
197 lines and carried three *different* defects, one per repository:

- a **non-ASCII API key could never authenticate**: Starlette decodes header
  bytes as latin-1, the code compared UTF-8, so the *correct* key was rejected
  with a message saying it was wrong
- **`GET /health/` returned 401** once keys were configured, because the check
  runs before routing, so any probe written with the trailing slash went
  permanently unhealthy
- **`TTS_API_KEYS=','` silently disabled authentication** and logged that the
  variable was unset

The shared package fixed all three. The pin then reintroduced the same shape of
problem one level up: `packages/common` and the services that embed it could
sit at different revisions, and nothing in a single commit could say so.

Here the commit is the pin. `packages/common` is a **path dependency**, every
service installs it from the working tree, and a change to it is one commit and
one CI run that rebuilds all four images.

The first thing that removed was a second instance of exactly the same shape.
OpenAI's error schema requires four fields on every error — `type`, `message`,
`param` and `code`, the last two required-but-**nullable**, so present as JSON
null rather than absent — and `voice_common.errors` built three. It also
registered no handler for `StarletteHTTPException`, so a mistyped path under
`/v1` returned FastAPI's `{"detail": "Not Found"}`, which openai-python reads
no message off and reports as a bare "unknown error".

Three services noticed and each vendored its own fix, each docstring saying the
code belonged upstream and that the pin was the only reason it was not. The
fourth, `services/gateway`, had none of it — and it is the process an SDK
client actually talks to. Nobody looked, which is the same failure as the
non-ASCII API key: fixed in one service, found still live in two others months
later, because the fix round only patched where a reviewer happened to look.

All of it is in `voice_common.errors` now, the three copies are deleted, and
`voice_common.conformance` carries the assertion that would have caught the
gateway: every `/v1` error, in every service, has all four fields. A fifth
service cannot repeat it silently.

## Why four services and not one

Each one is a different shape of process, and the differences are measured
rather than assumed.

| | resident | rate | torch |
|---|---|---|---|
| `stt` — Parakeet | 1.4 GB | 47–63× realtime | no |
| `tts` — Kokoro | 0.33 GB | 4.1× realtime | no |
| `tts-long` — Chatterbox | 6.6 GB | 0.21× realtime | yes, CPU wheel |
| `gateway` | ~50 MB | — | no |

**Chatterbox cannot share an image with Kokoro.** It is twenty times heavier
and twenty times slower, so it loads lazily and unloads after ten minutes idle;
a model a thousandth its size must not wait behind that. It is a job queue for
the same reason — at 0.21× realtime a ten-minute recording takes about
three-quarters of an hour, and an HTTP request waiting for that times out long
before the audio exists.

**Neither TTS service can share an image with the recogniser.** `stt` and `tts`
carry no torch at all: ONNX Runtime carries Parakeet, Silero and Kokoro, and
CTranslate2 carries Whisper. Together they cost a fraction of a single torch
wheel, and adding `tts-long`'s torch to either would multiply an image that is
currently under half a gigabyte.

**The gateway is separate because it is the auth boundary.** It publishes the
only port; 8000, 8001 and 8002 stay on the app-internal network. Measured over
loopback with a trivial body, 300 requests: 0.32 ms direct against 1.17 ms
through the gateway — **0.85 ms added**, under 1% of a 200 ms dictation turn.

## Build

The build context is the **repository root** for every service, and each
Containerfile is named by path:

```bash
docker build -f services/stt/Containerfile      -t ai-voice-stt .
docker build -f services/tts/Containerfile      -t ai-voice-tts .
docker build -f services/tts-long/Containerfile -t ai-voice-tts-long .
docker build -f services/gateway/Containerfile  -t ai-voice-gateway .
```

That is the whole cost of the path dependency: `packages/common` has to be
inside the context, and a context of `services/stt` does not contain it.
Building from inside a service directory fails on the first `COPY`.

Everything else about these images is unchanged. No torch in `stt` or `tts`;
every service drops to uid 1000 before it serves anything; no model is baked
in — each downloads into the volume at `/models` on first start, which is why
the first start is slow and the rest are not.

**None of the four carries a `HEALTHCHECK` instruction.** `HEALTHCHECK` is not
a field in the OCI image spec, and CI builds with buildah, whose default format
is `oci`, so the instruction is dropped silently on the way to the registry.
The probes live in `compose.yaml`, which is where they take effect.

## Test

Install from the repository root, run pytest from inside the service:

```bash
pip install -r services/stt/requirements.txt './packages/common[conformance]'
cd services/stt && python -m pytest tests -q
```

Both halves are load-bearing.

- **Install from the root**, because `requirements.txt` names
  `./packages/common` and pip resolves a path requirement against the process's
  working directory rather than against the file it read it from. From
  `services/stt` the same command fails with `Expected package name at the
  start of dependency specifier`, which reads like a syntax error and is not
  one.
- **Test from the service directory**, because every suite does `import
  app.main` and that resolves only when the service's own directory is pytest's
  rootdir. There is deliberately no pytest configuration at the repository
  root: running pytest there would put four different `app` packages in scope
  at once.

To edit `packages/common` and see it in a service without reinstalling, put an
editable install over the top: `pip install -e './packages/common[audio,conformance]'`.

Each service runs `voice_common.conformance`, the suite the shared package
ships, against the app object it actually builds. That is what makes one tree
safe to share: a change to `packages/common` fails at a consumer's test job
rather than on the host. The gateway runs one assertion out of it rather than
the whole suite — it carries its own auth module and publishes no
`/openapi.json`, so most of the rest does not describe it — and the one it runs
is the four-field envelope check it used to fail.

## CI

One workflow, [`.github/workflows/build.yml`](.github/workflows/build.yml). The
five nested ones the imported repositories brought with them are gone — GitHub
reads `.github` at the repository root and nowhere else, so between the import
and this file, nothing built at all.

It keeps the conventions each of those five established: buildah, native
per-architecture runners (`ubuntu-24.04` and `ubuntu-24.04-arm`) rather than
qemu, `STORAGE_DRIVER=vfs`, a per-architecture tag then a manifest job,
`prerelease` publishing `:pre` and never `:latest`, `main` publishing
`:latest`.

What is new is the filter. A change under `services/tts` builds `tts` and
nothing else; a change under `packages/common` builds **all four**, because
three of them have its code inside them and the fourth is rebuilt anyway rather
than relying on a "this one is exempt" fact staying true. A change to the
workflow or to `.dockerignore` also builds all four. A tag, a manual run or a
new branch has no base to diff against and builds everything.

Each image keeps its per-architecture runtime import check. Those are not
ceremony — they caught two real defects:

- **torch would not import on arm64.** The amd64 wheels vendor their own
  `libgomp.so.1` and the arm64 ones do not, so the image built cleanly on both
  architectures and then failed at run time on one of them. `libgomp1` is in
  the `tts-long` Containerfile because of that check.
- **a stale `soundfile` assertion failed both `tts` builds** with
  `ModuleNotFoundError` for a module the image is correct not to have, after
  every encoder moved to feeding ffmpeg on a pipe and the library left with
  `libsndfile`.

## Deploy

[`compose.yaml`](compose.yaml) is the stack as one app. Only the gateway
publishes a port; the three backends are internal-only, which is what makes the
single auth boundary real rather than aspirational — a client that skipped
`:8080` skipped the only process here that checks a token. Every healthcheck
points at its own container's `127.0.0.1`: a container must never be restarted
because a different container is down.

`GATEWAY_API_KEYS` is deliberately not set in that file. Unset means
authentication is **off** and every request is accepted, and the gateway says
so at WARNING on every start. Inventing a key in a file that gets deployed is
how a placeholder becomes production credentials.

## The findings worth keeping

Each of these is measured, and each one is why some part of this repository is
shaped the way it is. The service READMEs carry the full working.

**Parakeet beats Whisper large-v3 on this workload, by a lot.** Across 25
conditions and five Brazilian Portuguese corpora on identical audio: 0.144
against 0.250 pt-BR WER, at 47–63× realtime against 0.5–0.9×. It won 21 of the
25, on 1.4 GB against 2.9 GB. It also degrades far more gracefully —
band-limiting to 4 kHz, which is what a cheap or distant microphone does, cost
Whisper +206% WER and Parakeet +41%. Whisper stays available because it is
genuinely better on clean read speech and it is the only one of the two that
accepts a vocabulary at decode time.

**A second recogniser as a cross-check does not work.** Built, measured,
removed: across every disagreement observed, the second model was the wrong
one. Not once did it catch a real error. A second opinion only informs when it
is roughly as good as the first; when it is reliably worse its dissent reduces
to noise, at about 40% of throughput — and worst on short clips, which is what
dictation is (rtf 0.39–0.47 against 1.26–1.60 on long ones).

**LLM cleanup of a transcript is a trap at any size that fits on this box.**
Tested at 4B with an explicit prompt forbidding it, the cleanup stage still
inverted meaning, reversed pronouns, deleted content it was told twice to
preserve, and leaked its own reasoning into the output. Reliable adherence
starts around 14B, which needs 10–20 GB of VRAM. Below that the raw transcript
is more faithful than the cleaned one, and a faithful transcript is the
product.

**Chatterbox silently truncated everything longer than 40 seconds.**
`generate()` stops after 1000 speech tokens and S3 speech tokens run at 25 Hz,
so one call cannot produce more than forty seconds of audio however much text
it is given. 1690 characters — about 170 seconds of speech — came back as
exactly 40.0 seconds, with no error and no warning, after paying for the whole
thing in CPU. The same text through the chunker produces 100.2 seconds. Every
input is chunked and spliced now, on both routes.

**Threads do not help Chatterbox.** From 4 to 16 the rate moved under 5%
(0.213× to 0.217× on English), because autoregressive token generation is
sequential. That is why it is a queue rather than a bigger box.

**The pauses matter more than the voice.** Tested by ear on the same voice and
the same words, three ways — flowing prose, short declaratives, and short
declaratives with 0.75 s of inserted silence. Only the third sounds like
instructions. The silence is generated in `services/tts`, not asked of the
model: no TTS model reliably produces a beat you can act inside. A voice is a
510 KB embedding over weights that are already resident, so switching between
54 of them costs nothing.

**The gateway earns nothing until the three backend ports are closed.**
Everything above about a single auth boundary is a property of the deployment,
not of the code. Until `compose.yaml` applies, it is a fourth container and an
extra hop in front of three ports that are still open to the LAN.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

All 42 commits from the five original repositories are intact — the import used
`git subtree`, so `git log --follow` on any file reaches back through it.
(`git rev-list --count HEAD` counts more: the five merge commits the imports
made, and this repository's own commits on top of them.)

## Licence

BSD 2-Clause, throughout. Each service and the package keep their own `LICENSE`.
