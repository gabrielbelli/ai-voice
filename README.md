# tts-stack

Self-hosted text-to-speech. Kokoro, CPU only, no torch.

```text
text or segments
  ↓  espeak-ng    phonemise
  ↓  Kokoro-82M   ONNX Runtime, CPU
  ↓  wav or opus
```

Sibling of [stt-stack](https://github.com/gabrielbelli/stt-stack), same
conventions.

## Status

`main` carries validated versions only. Work happens on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 \
  ghcr.io/gabrielbelli/tts-stack:pre
```

First start downloads ~340 MB into the volume. Later starts are immediate.

```bash
curl -X POST localhost:8001/speak -H 'content-type: application/json' \
  -d '{"text":"Here is the change to make.","voice":"bm_george"}' \
  --output out.wav
```

## Segments, and why they matter more than the voice

`/speak` accepts a list of segments with explicit pauses:

```json
{
  "voice": "bm_george",
  "speed": 0.95,
  "segments": [
    {"text": "Three steps.",                       "pause_after": 0.75},
    {"text": "One. Open your config file.",        "pause_after": 0.75},
    {"text": "Two. Set the model to nothing.",     "pause_after": 0.75},
    {"text": "Three. Restart the container.",      "pause_after": 0.45}
  ]
}
```

**The silence is generated here, not asked of the model.** No TTS model
reliably produces a beat you can act inside — punctuation buys a breath, an
instruction needs a gap.

Tested by ear on the same voice and the same words, three ways: flowing prose,
short declaratives, and short declaratives with 0.75 s of inserted silence.
Only the third sounds like instructions. Nothing changed but the writing and
the gaps.

So the model is not the interesting variable. Write for the ear and place the
pauses; any competent voice will then do.

## Voices

54 voices, all sharing one 310 MB model. **A voice is a 510 KB embedding**, so
switching costs nothing after load — you can use a different voice per
request, or per segment.

```bash
curl -s localhost:8001/voices | python3 -m json.tool
```

| Prefix | Locale |
|---|---|
| `af_` `am_` | en-US female / male |
| `bf_` `bm_` | en-GB female / male |
| `pf_` `pm_` | pt-BR female / male |

Only three are Brazilian Portuguese: `pf_dora`, `pm_alex`, `pm_santa`.

For explanations, the lower and calmer voices work better than the bright
defaults — `bm_george`, `am_onyx`. Slowing slightly (`"speed": 0.95`) reads as
deliberate rather than performed.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TTS_VOICE` | `bm_george` | Default when a request omits one |
| `TTS_LANGUAGE` | `en-us` | `en-us`, `en-gb`, `pt-br`, … |
| `TTS_THREADS` | `4` | Must match your CPU limit — see below |
| `TTS_MODEL_DIR` | `/models` | Volume for weights |

## Limiting CPU use

Set the container's CPU limit **and** `TTS_THREADS` to the same number. ONNX
Runtime sizes its thread pool from the host's core count, not the cgroup, so a
`--cpus` limit alone leaves the container spawning a thread per host core and
then contending for the slice it is allowed — slower than simply using fewer
threads.

```bash
docker run -p 8001:8001 -v tts-models:/models \
  --cpus 4 -e TTS_THREADS=4 --memory 2g \
  ghcr.io/gabrielbelli/tts-stack:pre
```

Steady state is about 400 MB, so 2 GB is generous.

Every response carries `X-Realtime-Factor`. If it drops when you raise
`TTS_THREADS`, coordination is costing more than the extra cores return.

## Performance

Measured on an M2 Max, CPU only:

```text
20.7 s of speech generated in 5.0 s   =  4.1x realtime
17.1 s of speech generated in 4.0 s   =  4.3x realtime
```

~330 MB resident. The GPU is never touched.

## What is not here

**Chatterbox**, the long-form alternative, is deliberately absent. It needs
5.3 GB and runs below realtime, so it belongs behind a separate service that
can be started on demand rather than sitting resident beside a model a
thousandth its size. Whether it is viable on CPU at all is still being
measured.

**Qwen3-TTS** is excluded for a different reason: it is Mandarin-first, and
its English and Portuguese carry a Chinese accent. **F5-TTS** and **XTTS-v2**
are excluded for their non-commercial licences.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
