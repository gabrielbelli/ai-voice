# voice-common

The shared wire contract for [stt-stack](https://github.com/gabrielbelli/stt-stack),
[tts-stack](https://github.com/gabrielbelli/tts-stack) and
[tts-long](https://github.com/gabrielbelli/tts-long).

```text
voice_common.auth         API keys, the exemption set, the 401
voice_common.errors       OpenAI's error envelope, and the validation handler
voice_common.health       the health CONTRACT: async, and one string for route
                          and exemption
voice_common.models       Segment, OpenAISpeechRequest — two bases, no more
voice_common.logging      one basicConfig line, plus a level switch
voice_common.audio        [audio] extra: pcm_bytes, check_rate, splice
voice_common.conformance  a pytest suite each consumer runs against its own app
voice-entrypoint.sh       installed to /usr/local/bin: TLS, chown, setpriv
```

## Why this exists

`app/auth.py` exists in all three repos. Diffed pairwise:

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

Two of the three are still live today in the copies that never got the fix:
`stt-stack/app/auth.py:115` and `tts-long/app/auth.py:68` both encode UTF-8,
and `tts-long/app/auth.py:103` still matches `OPEN_PATHS` as an exact string.

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
| Kokoro voice handling (`VOICE_ALIASES`, `resolve_voice`, …) | A property of one model's `voices.bin`, and the `shimmer`→`af_bella` row is a judgement by ear. tts-long has no named voices; stt-stack has no voices. |
| Parakeet and Whisper model loading | Nothing else in the estate loads a recogniser. |
| Silero VAD | One consumer; the 512-sample window and 16 kHz constant are properties of one ONNX model. |
| Glossary repair | A regex table with no STT-specific machinery, so it *looks* shareable. Nobody has asked for it twice. "Could plausibly be wanted by a second service" is exactly how a shared package becomes a junk drawer. |
| Chatterbox's job queue | The whole reason tts-long exists as a separate service, and none of it generalises to a 4x-realtime synthesiser. |
| Per-wheel workarounds (`_wire_espeak`, `_stub_watermarker`) | Each is a scar from one dependency in one image. Sharing them would put tts-long's torch problem in stt-stack's import path. |
| Audio encoding beyond the PCM byte contract | The ffmpeg table and its bitrates are an image decision: tts-long has no ffmpeg on purpose, stt-stack none so a 44.1 kHz caller is told rather than quietly resampled. |
| The 16 kHz input guard | Despite the name, not the same thing as the output rate guard. It is a policy about what input the service accepts, in the only service that takes audio in. |
| The concrete request models | They diverge where it matters: tts-stack clamps speed into 0.5–2.0, tts-long refuses any speed but 1.0. A shared base would need overriding on nearly every field. |
| Containerfiles and CI workflows | Base image, ports, ENV defaults, tuning constants. Deployment machinery, and reusable-workflow territory — a separate decision. |

## Using it

```python
from fastapi import FastAPI
from voice_common import auth, errors, health, logging as voice_logging

log = voice_logging.setup("tts-stack", "TTS")

app = FastAPI(title="tts-stack")
errors.install_errors(app)                  # ApiError + RequestValidationError
health.install_health(app, details=lambda: {"threads": THREADS})
auth.install(app, "TTS_API_KEYS")           # the env var name is the parameter
```

`install_health` registers the route **and** exempts exactly that path from
authentication, so the two can never name different strings. The order of the
two calls does not matter.

Raise `errors.ApiError` from a handler; return `errors.error_response(...)` from
middleware. `type_` and `code` are keyword-only, because two sibling repos pass
them positionally in opposite orders today and nothing catches it.

### The conformance suite

Every consumer runs the suite this package ships, against the app it actually
builds. That is what stops behaviour drifting in the parts each service still
writes itself, and it makes a bad `voice-common` bump fail at the consumer's
build rather than in production.

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

One identical line in each service's `requirements.txt`, with the readable tag
in a comment above it:

```text
# voice-common v1.0.0 — https://github.com/gabrielbelli/voice-common
voice-common @ https://github.com/gabrielbelli/voice-common/archive/<40-char-sha>.tar.gz
```

Plus `voice-common[audio] @ ...` for the two TTS services, which need numpy
anyway.

- **A tarball URL, not `git+https://`**: `python:3.x-slim-trixie` has no git
  binary, so a git dependency costs an extra apt layer in three images.
- **A SHA, not a tag**: a tag can be moved; the archive-by-SHA URL is genuinely
  immutable. The tag survives as the human-readable half in the comment, which
  the release workflow should emit as a pair so the line is generated rather
  than hand-edited.
- **Not PyPI**: three consumers, one namespace, none external. A published wheel
  buys release ceremony and nothing else. This is already a valid sdist, so the
  move is a `twine upload` away if a fourth consumer appears.
- **Not vendoring or a subtree**: that *is* the status quo, and it is what
  produced 197/187/170 differing lines and three different bug sets.
- **Not a submodule**: the pin lands in `.gitmodules` as a raw SHA nobody reads
  in review, every CI checkout needs `submodules: true`, and the dependencies
  still have to be duplicated into each `requirements.txt` by hand.

A change here does **not** force three rebuilds. Each service pins its own SHA
and bumps on its own schedule. What *should* force three rebuilds is a
security-relevant fix — which is precisely the case that today does not happen
at all.

The honest ongoing cost: this repo needs its own CI, its own tests and a
release, and every change has to be validated against three consumers.
`voice_common.conformance` is the answer to that last part.

## Adoption order

1. **tts-stack** — it already has the corrected auth and the strictest TLS
   readability check, so it is the smallest diff and the reference.
2. **tts-long**.
3. **stt-stack** last: it is the only one that inverts its enforcement model
   from per-route dependencies to middleware, and that should go after the
   middleware has run in two services.

## Tests

```bash
python -m venv .venv && ./.venv/bin/pip install -e '.[audio,conformance]'
./.venv/bin/python -m pytest -q
```

Every test is named after the failure it prevents, so the next person reading
one can see why it exists.

## Licence

BSD-2-Clause. Same as the three services.
