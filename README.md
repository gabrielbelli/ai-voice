# parakeet-stt

CPU-only speech-to-text over HTTP. Parakeet TDT 0.6B v3 on ONNX Runtime, no
CUDA and no torch.

Returns the raw transcript plus glossary repair. There is no LLM cleanup
stage, deliberately — see [Why no cleanup](#why-no-cleanup).

## Status

`main` carries validated versions only. Active work is on `prerelease`, which
publishes `:pre` and never `:latest`.

## Run

```bash
podman run -p 8000:8000 -v parakeet-models:/models \
  ghcr.io/gabrielbelli/parakeet-stt:pre
```

First start downloads ~600 MB into the volume. Later starts are immediate.

```bash
curl -F file=@clip.wav http://localhost:8000/transcribe
```

```json
{
  "text": "I need to make a commit on the Theoria dashboard",
  "audio_seconds": 3.4,
  "compute_seconds": 0.7,
  "realtime_factor": 4.9,
  "repaired": ["commit", "Theoria dashboard"]
}
```

Audio must be **16 kHz mono**. Anything else is rejected rather than resampled
in-process, so a client that sends 44.1 kHz finds out immediately instead of
quietly getting worse transcripts.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STT_MODEL` | `istupakov/parakeet-tdt-0.6b-v3-onnx` | Any onnx-asr model id |
| `STT_QUANTISATION` | `int8` | `int8` or `fp32` |
| `STT_THREADS` | `4` | ONNX Runtime scales sub-linearly past ~8 |
| `STT_GLOSSARY` | `/etc/parakeet-stt/glossary.txt` | `heard = intended`, one per line |

For Brazilian Portuguese, `alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx`
is a fine-tune of the same architecture and drops in via `STT_MODEL`.

## Performance

Rough figures, `int8`, one 15-second clip:

| Host | Wait |
|---|---|
| 4 cores, modern desktop CPU | ~3 s |
| 8 cores | ~2 s |
| 4 cores, older Xeon | ~5 s |

Parakeet is CTC/TDT, not encoder-decoder, which is why this is viable on a CPU
at all — Whisper `large-v3` on the same 8 cores runs at roughly 1x realtime,
meaning a 15-second clip costs 15 seconds.

## Why no cleanup

An LLM pass over the transcript is the obvious next stage and it is a trap at
the sizes that fit beside this on a CPU box. Tested on a 4B model with a
well-written, explicit prompt forbidding it, the cleanup stage still:

- inverted meaning — "makes no sense to be available to me" became "are not
  available to me"
- reversed pronouns, turning "those tasks for you" into "those tasks for me"
- deleted content it was told twice to preserve
- leaked its own reasoning into the output ("No, that's not right")

Instruction-following strong enough for this task starts around 14B, which
needs 10-20 GB of VRAM. Below that, the raw transcript is more faithful than
the cleaned one. Whatever consumes this text — an editor, an agent, a person —
resolves ambiguity better than a small model rewriting sentences does.

Glossary repair stays because it is a lookup, not a judgement.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
