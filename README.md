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

First start downloads both models into the volume. Later starts are immediate.

```bash
curl -F file=@clip.wav http://localhost:8000/transcribe
```

```json
{
  "text": "I need to make a commit on the <Theoria|theory> dashboard",
  "primary": "I need to make a comet on the Theoria dashboard",
  "secondary": "I need to make a commit on the theory dashboard",
  "disagreements": [{"primary": "Theoria", "secondary": "theory"}],
  "agreement": 0.889,
  "repaired": ["commit"],
  "audio_seconds": 4.1,
  "speech_seconds": 3.2,
  "compute_seconds": 2.4,
  "realtime_factor": 1.7
}
```

Audio must be **16 kHz mono**. Anything else is rejected rather than resampled
in-process, so a client sending 44.1 kHz finds out immediately instead of
quietly getting worse transcripts.

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

The obvious next stage, and a trap at any size that fits beside two
recognisers on a CPU box. Tested at 4B with an explicit prompt forbidding it,
the cleanup stage still:

- inverted meaning — "makes no sense to be available to me" became "are not
  available to me"
- reversed pronouns, turning "those tasks for you" into "those tasks for me"
- deleted content it was told twice to preserve
- leaked its own reasoning into the output ("No, that's not right")

Reliable adherence starts around 14B, which needs 10–20 GB of VRAM. Below
that the raw transcript is more faithful than the cleaned one. The consensus
pass does the useful half of the job — flagging uncertainty — without the half
that invents confidence.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STT_MODEL` | `parakeet` | `parakeet` or `whisper` |
| `STT_MODEL_ID` | model default | Override the specific checkpoint |
| `STT_QUANTISATION` | `int8` | Whisper also takes `int8_float32`, `float32` |
| `STT_LANGUAGE` | unset | Leave unset if you code-switch. See below |
| `STT_THREADS` | `4` | Must match your CPU limit. See below |
| `STT_VAD` | `1` | Silence removal |
| `STT_HOTWORDS` | `1` | `0` disables decode-time biasing, for A/B tests |
| `STT_MARKER` | `<{a}\|{b}>` | Disagreement format |
| `STT_GLOSSARY` | `/etc/stt-stack/glossary.txt` | See below |

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
drops in via `STT_SECONDARY`.

## Language

Leave `STT_LANGUAGE` unset unless every clip is in one language.

Pinning the wrong language does not degrade the transcript — it **translates**
it. English speech sent with `language=pt` comes back as fluent Portuguese
that reads like a working transcript and silently is not one:

```text
spoken     Look, there is a big problem here. Small tasks and more operational...
pinned pt  Veja, há um grande problema aqui, tarefas pequenas e mais operacionais...
```

Parakeet is unaffected — it detects on its own and has no language argument —
so the consensus `agreement` score collapses toward zero when this happens.
A near-zero score on speech that clearly transcribed is the signature.

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

Steady state is about 3 GB with both models loaded; 6 GB leaves room for a
long clip without letting a runaway request take the host down.

Every response carries `realtime_factor`. If it drops when you raise
`STT_THREADS`, you have crossed the point where coordination costs more than
the extra cores return — around 8 on most hosts, earlier on older ones.

## Performance

Rough figures, `int8`, both models, one 15-second clip:

| Host | Wait |
|---|---|
| 8 modern cores | ~9 s |
| 4 modern cores | ~15 s |
| 4 cores, primary disabled (Parakeet only) | ~3 s |

Whisper dominates the cost. `STT_SECONDARY=` alone is not the speed lever —
`STT_PRIMARY=` cannot be emptied, so for a fast configuration set
`STT_PRIMARY=small` or run Parakeet-class models at both ends and accept the
loss of hotwords.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
