# voice-common

The shared wire contract for [services/stt](../../services/stt/README.md),
[services/tts](../../services/tts/README.md),
[services/tts-long](../../services/tts-long/README.md) and, for the error
envelope, [services/gateway](../../services/gateway/README.md).

```text
voice_common.auth         API keys, the exemption set, the 401
voice_common.errors       OpenAI's error envelope — all four fields — and the
                          validation, 404/405 and unhandled-500 handlers
voice_common.health       the health CONTRACT: async, and one string for route
                          and exemption
voice_common.models       Segment, OpenAISpeechRequest — two bases, no more
voice_common.logging      one basicConfig line, plus a level switch
voice_common.audio        [audio] extra: pcm_bytes, check_rate, splice
voice_common.conformance  a pytest suite each consumer runs against its own app
voice-entrypoint.sh       installed to /usr/local/bin: TLS, chown, setpriv
```

## Why this exists

`app/auth.py` existed in all three repositories, once each, before this package
replaced them. Diffed pairwise at that point:

| | differing lines |
|---|---|
| stt-stack vs tts-stack | 197 |
| tts-stack vs tts-long | 187 |
| stt-stack vs tts-long | 170 |

They implement the same idea and drifted, and **the drift was the defects**. One
adversarial review round found three *different* bugs, one set per copy, because
each copy had drifted separately:

- **A non-ASCII API key could never authenticate** (tts-stack only). Starlette
  decodes header bytes as latin-1; the code compared UTF-8 bytes, so the
  *correct* key was rejected with a message saying it was wrong.
- **`GET /health/` returned 401 once keys were on** (tts-stack only). The check
  runs before routing, so FastAPI's 307 never happens, and any probe written
  with the trailing slash goes permanently unhealthy.
- **`TTS_API_KEYS=','` silently disabled authentication** (tts-stack and
  tts-long, not stt-stack) and logged that the variable was *unset*.

Two of the three outlived their discovery: the fix round patched only the copy
a reviewer happened to be reading, and `stt-stack/app/auth.py:115` and
`tts-long/app/auth.py:68` were both still encoding UTF-8, with
`tts-long/app/auth.py:103` still matching `OPEN_PATHS` as an exact string, when
this package landed. All three copies are deleted now — the only `app/auth.py`
left in the tree is `services/gateway`'s, which is its own module by design:
that service has a different env var and fans health out to three backends
rather than reporting on itself.

This package is the union of every fix. The argument for it is measured, not
predicted.

## What belongs here

**The wire contract.** What a client sees, what an operator configures, what a
healthcheck probes. If two services must answer the same way or an operator
must configure them the same way, it belongs here.

## What deliberately does not

Each of these was considered and declined. The list is part of the design, not
an accident of scope.

| Stays behind | Why |
|---|---|
| Kokoro voice handling (`VOICE_ALIASES`, `resolve_voice`, …) | A property of one model's `voices.bin`, and the `shimmer`→`af_bella` row is a judgement by ear. tts-long has its own registry and it is a different mechanism — clips on disk cloned per request, not embeddings in a weights file, so `TTS_VOICE_STRICT` and the `X-Voice` header have no counterpart here. stt-stack has no voices at all. |
| Parakeet and Whisper model loading | Nothing else in the estate loads a recogniser. |
| Silero VAD | One consumer; the 512-sample window and 16 kHz constant are properties of one ONNX model. |
| Glossary repair | A regex table with no STT-specific machinery, so it *looks* shareable. Nobody has asked for it twice. "Could plausibly be wanted by a second service" is exactly how a shared package becomes a junk drawer. |
| Chatterbox's job queue | The whole reason tts-long exists as a separate service, and none of it generalises to a 4x-realtime synthesiser. |
| Per-wheel workarounds (`_wire_espeak`, `_stub_watermarker`) | Each is a scar from one dependency in one image. Sharing them would put tts-long's torch problem in stt-stack's import path. |
| Audio encoding beyond the PCM byte contract | The ffmpeg table and its bitrates are an image decision. tts-stack and tts-long both carry the binary and both encode the same three lossy formats at the same bitrates, but each keeps its own table: the formats, the sample rate and the container each service offers are part of *its* wire contract, and stt-stack has no encoder at all so a 44.1 kHz caller is told rather than quietly resampled. Worth revisiting if a third service starts encoding. |
| The 16 kHz input guard | Despite the name, not the same thing as the output rate guard. It is a policy about what input the service accepts, in the only service that takes audio in. |
| The concrete request models | They diverge where it matters: tts-stack clamps speed into 0.5–2.0, tts-long refuses any speed but 1.0. A shared base would need overriding on nearly every field. |
| Containerfiles and CI workflows | Base image, ports, ENV defaults, tuning constants. Deployment machinery, and reusable-workflow territory — a separate decision. |

## Using it

```python
from fastapi import FastAPI
from voice_common import auth, errors, health, logging as voice_logging

log = voice_logging.setup("tts-stack", "TTS")

app = FastAPI(title="tts-stack")
errors.install_errors(app)                  # ApiError, validation, 404/405, 500
health.install_health(app, details=lambda: {"threads": THREADS})
auth.install(app, "TTS_API_KEYS")           # the env var name is the parameter
```

`install_health` registers the route **and** exempts exactly that path from
authentication, so the two can never name different strings. The order of the
two calls does not matter.

Raise `errors.ApiError` from a handler; return `errors.error_response(...)` from
middleware. `type_`, `code` and `param` are keyword-only, because two sibling
repos passed the first two positionally in opposite orders and nothing caught
it — both are strings, both produce a valid-looking envelope, and the only
symptom is a client reading a code out of the field that names a category.

Every body carries all four fields OpenAI's schema requires. `param` and `code`
are serialised even when null, because the schema marks them
required-but-nullable and a generated client may read the key rather than test
for its presence.

**`install_errors` stops at `/v1`.** Native routes keep FastAPI's
`{"detail": ...}` and its 422, and an unhandled error on one stays a plain-text
500. Those routes have clients that never touch the compatibility layer, and
`errors.v1_path` is the one place that line is drawn.

### The conformance suite

The three backends run the whole suite this package ships, against the app each
actually builds. That is what stops behaviour drifting in the parts each service
still writes itself, and it makes a bad `voice-common` bump fail at the
consumer's build rather than in production. The gateway runs one assertion out
of it rather than the whole suite — it carries its own auth module and publishes
no `/openapi.json`, so most of the rest does not describe it — and the one it
runs, `assert_four_field_envelope`, is exported for exactly that.

```python
# tests/test_conformance.py
import pytest
from voice_common.conformance import *          # noqa: F401,F403
from voice_common.conformance import Service, module_app

@pytest.fixture
def voice_service():
    return Service(env_var="TTS_API_KEYS",
                   build=module_app("app.main"),
                   v1_path="/v1/audio/speech")
```

The star import is deliberate: it puts the tests in the consumer's own tree, so
its conftest and fixtures apply. `pytest --pyargs` would collect them out of
site-packages, where they cannot see the consumer's fixtures.

### The entrypoint

`voice-entrypoint.sh` installs to `/usr/local/bin`, so a Containerfile drops its
own `entrypoint.sh` and its `COPY` line:

```dockerfile
ENV VOICE_TLS_PREFIX=TTS \
    VOICE_CHOWN_DIRS="/models /output"
ENTRYPOINT ["voice-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

It is the union of the strictest rule from the three copies it replaces: exit 1
on half a TLS configuration, exit 1 when the command is not uvicorn, readability
probed as the target uid with setpriv's own stderr used to tell an unreadable
file from a setpriv failure, then `setpriv --reuid --regid --init-groups
--inh-caps=-all`. **Nothing generates a certificate.** One that appears by magic
is one every client is taught to stop validating.

## Installing

One line in each consuming service's `requirements.txt`, a path into this same
tree:

```text
./packages/common            # services/stt and services/gateway — no extras
./packages/common[audio]     # services/tts and services/tts-long — numpy
```

The gateway takes it for `voice_common.errors` and nothing else: it keeps its
own auth module, its own health fan-out and its own entrypoint, so it needs no
extras and does not install the audio helpers.

**The path is relative to the working directory, not to the file.** pip
resolves a path requirement against the process's cwd, so every install runs
from the repository root — which is what the Containerfiles' `WORKDIR` and
every CI job do. Reading a `requirements.txt` from inside its own service
directory fails with `Expected package name at the start of dependency
specifier`, which reads like a syntax error in the file and is not one.

Naming the path twice with different extras is not a conflict: pip's resolver
unions them, which is how `services/tts/requirements-dev.txt` adds
`[conformance]` on top of the `[audio]` its `-r requirements.txt` already
brought in.

### Why a path, when the SHA pin was right

The pin was right *while this was a separate repository*. Its reasoning is
worth keeping, because it is the reasoning that a fourth consumer outside this
tree would revive:

- **A tarball URL, not `git+https://`**: `python:3.x-slim-trixie` has no git
  binary, so a git dependency costs an extra apt layer in three images.
- **A SHA, not a tag**: a tag can be moved; the archive-by-SHA URL is genuinely
  immutable.
- **Not PyPI**: three consumers, one namespace, none external. A published
  wheel buys release ceremony and nothing else. This is still a valid sdist, so
  the move is a `twine upload` away if a fourth consumer appears.
- **Not vendoring**: vendoring *was* the status quo, and it is what produced
  197/187/170 differing lines and three different bug sets.
- **Not a submodule**: the pin lands in `.gitmodules` as a raw SHA nobody reads
  in review, every CI checkout needs `submodules: true`, and the dependencies
  still have to be duplicated into each `requirements.txt` by hand.

What the pin bought was reproducibility, and a path in one repository gives the
same thing more cheaply: **the commit is the pin**. A checkout of any revision
of this repository builds the services against exactly the shared code that
revision contains, and no state outside `git log` records which version that
was.

What it cost was the gap. A fix here reached a service only after a push, a
tag, a re-pin and a rebuild, three times over — so a change was **not** forced
on the consumers, and the one case where it should be, a security-relevant fix,
was the case that did not happen at all. That inversion is now the other way
round: a change here rebuilds all four images in one CI run, and there is
nowhere for a stale copy to hide.

The honest remaining cost is unchanged in kind: every change here has to be
validated against four consumers now, not three — the gateway installs this
package too. `voice_common.conformance` is the answer to that, and it now runs
in the same pipeline as this package's own tests rather than in four other
repositories.

## Tests

Run from this directory, so pyproject's `testpaths` and its
`filterwarnings = ["error"]` apply:

```bash
python -m venv .venv && ./.venv/bin/pip install -e '.[audio,conformance]'
cd packages/common && ../../.venv/bin/python -m pytest -q
```

Every test is named after the failure it prevents, so the next person reading
one can see why it exists.

## Licence

BSD-2-Clause. Same as the four services.
