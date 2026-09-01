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

Audio must be **16 kHz mono**. Anything else is rejected rather than resampled
in-process, so a client sending 44.1 kHz finds out immediately instead of
quietly getting worse transcripts.

## OpenAI-compatible API

`POST /v1/audio/transcriptions` speaks OpenAI's transcription shape, so
anything already pointed at that endpoint works here by changing a base URL.

```bash
curl -H "Authorization: Bearer $STT_API_KEY" \
     -F file=@clip.wav -F model=whisper-1 \
     http://localhost:8000/v1/audio/transcriptions
```

```json
{"text": "I need to make a commit on the Theoria dashboard"}
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")

with open("clip.wav", "rb") as clip:
    print(client.audio.transcriptions.create(model="whisper-1", file=clip).text)
```

`response_format` takes `json`, `text`, `verbose_json`, `srt` and `vtt`. The
same 16 kHz mono rule applies; the audio is not resampled here either.

Errors on this route are OpenAI's envelope, and a rejected body is a **400**
carrying `error.message` — that is what the real API answers, and what a client
written against it branches on. This route answered 422 until the shared
package below took the handler over. The native `/transcribe` is untouched: its
FastAPI `{"detail": [...]}` bodies are a contract that already has clients.

### Which route to prefer

**`/transcribe`, wherever you control the client.** It is the same pipeline,
and it returns what the OpenAI shape has no field for:

| | `/transcribe` | `/v1/audio/transcriptions` |
|---|---|---|
| Transcript | yes | yes |
| `repaired` — which glossary terms fired | yes | — |
| `raw` — the transcript before repair | yes | — |
| `realtime_factor` and the timings behind it | yes | — |
| Works with an unmodified OpenAI client | — | yes |

`repaired` is the one worth caring about. A silent substitution is worse than
no substitution: if a glossary rule is wrong, you want to see it named, and on
the compatible route you cannot.

### What the compatible route cannot honour

Documented rather than faked:

| Field | Behaviour |
|---|---|
| `model` | Accepted, ignored. The engine is fixed at startup by `STT_MODEL`; loading a second one per request does not fit the memory this runs in |
| `prompt` | Added to the glossary's hotwords **on Whisper only**. Parakeet has no decode-time vocabulary at all, so under the default engine it does nothing. `STT_HOTWORDS=0` drops it on both engines |
| `temperature` | Accepted, ignored. Parakeet has no sampling temperature, and pinning one on Whisper disables its retry on low-confidence output |
| `verbose_json` | No `segments`, and `language` echoes the request — neither engine reports segment timings or a detected language through this interface |
| `srt`, `vtt` | One cue spanning the clip, for the same reason. True, if not very useful |

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

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt pytest httpx
./.venv/bin/python -m pytest tests -q
```

Everything particular to speech-to-text stays here: the recognisers, the VAD,
the glossary, the 16 kHz input rule and both routes.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
