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

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TTS_THREADS` | `8` | Match your CPU limit, but expect little from raising it |
| `TTS_IDLE_TIMEOUT` | `600` | Seconds before the 6.5 GB model is unloaded |
| `TTS_EXAGGERATION` | `0.3` | Stock is 0.5 and reads as over-cheerful |
| `TTS_CFG_WEIGHT` | `0.3` | Lower is slower, more deliberate |
| `TTS_TEMPERATURE` | `0.6` | Stock 0.8 varies more than an explanation wants |

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
