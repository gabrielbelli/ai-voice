# ADR 0010 — The third engine was built, measured on the card it was for, and retired from this deployment

**Status:** accepted
**Date:** 2026-09-13
**Supersedes:** [ADR 0009](0009-a-third-engine-that-cannot-run-here.md), which
stands as the record of what was built and why it was shaped that way. Every
engineering fact in it is still true. What it got wrong is its own first
sentence — that this deployment offers a third engine — and that is what this
record replaces.
**Unchanged:** [ADR 0007](0007-two-lanes-not-three-rungs.md), still two lanes,
and [ADR 0008](0008-two-engines-and-both-stay-jobs.md), whose summary table is
correct again: two engines, both of them local, both of them jobs.
**Evidence:** Linear GAB-634 carries the full run log.

## Decision

`services/tts-long` offers **two engines**, `chatterbox` and
`chatterbox-turbo`. `voxtral` is **in the catalogue, tested, and not enabled
here**.

| | |
|---|---|
| `TTS_ENGINES` | `chatterbox,chatterbox-turbo` |
| `GATEWAY_LONG_MODELS` | `chatterbox,tts-long,chatterbox-turbo` |
| `TTS_ALLOW_RUNNER_ONLY_ENGINES` | **unset**, so back to the shipped `0` |
| `voice_common.engines.CATALOGUE` | **three rows, unchanged** |

**The engine code is not deleted and this record is not a licence to delete
it.** The catalogue row, the runner client, the refusals that name
`flow_steps` and `cfg_alpha`, the preset-voice path and every test over them
stay exactly as they are. They are correct, they pass, and a machine with a
bigger card is the deployment they were written for. What is retired is the
**claim that this deployment offers the engine**, and that claim lived in two
configuration keys.

Retiring it is therefore **two lines of configuration**, not surgery. Enabling
it again is the same two lines plus the opt-out, because the engine has no
local lane and the availability check refuses that pairing by default.

## Why, in three measurements taken on the card this engine was for

`spring` is the owner's gaming PC: Windows, Ryzen 7 5700X3D, **RTX 3070, 8 GB**.
Every figure below was taken there, on the settings ADR 0009 tuned by ear.
Nothing here is a vendor claim, and nothing here is an opinion about the
checkpoint's quality in the abstract.

### 1. It does not fit on the card, and the failure is a segfault rather than an error

| | |
|---|---|
| measured load peak | **8265–8288 MiB** |
| what the card physically has | **8192 MiB** |
| resident once loaded | 3.5–3.7 GB, comfortable |
| what happens when the desktop holds any VRAM at all | **SEGFAULT** |

ADR 0009 recorded the 8265 MiB peak as *"an operational risk, not a footnote"*
and designed around it: preflight the VRAM, fail **that job** with both numbers
in the sentence, keep the process alive. Repeated runs found the peak is a
**range** rather than a number — 8265 to 8288 MiB — and that the failure mode
past the margin is not the `torch.cuda.OutOfMemoryError` the design catches.
It is a **segmentation fault inside the quantised kernels**, which no Python
handler sees and no preflight can promise to avoid: the load completed at all
only while the desktop was near idle, and the desktop belongs to somebody who
is often using it.

That is the reason the other two do not share. Chatterbox baseline is 4199 MiB
and turbo is 3805 MiB on the same card, both with room for a browser and a
game.

### 2. It dropped whole sentences, and the round trip is what caught it

The check is the one this repository already trusts: speak a script through the
deployed API, transcribe the audio back through Parakeet, and diff the two.
Nothing else in this stack reads what the audio **says**.

| | |
|---|---|
| best transcript match against the script | **1.000** |
| worst | **0.682** |
| what the gap is made of | **whole sentences missing from the audio** |

A match that ranges from perfect to 0.682 on the same script is worse than a
uniformly mediocre one, because it cannot be heard in a spot check and it
cannot be tuned away: the settings were identical across the runs. Chatterbox
and turbo were measured the same way on the same script and neither loses a
sentence.

**A dropped sentence is the defect class this project keeps finding, arriving
from the audio side.** Every label, every header and every record row said the
right thing while the speech was missing a sentence. `x-tts-engine` cannot
catch that and neither can a suite.

### 3. It is the slowest thing measured here, and it cannot clone

| engine | rtf | where |
|---|---|---|
| Kokoro | 2.2–2.7x | orko's processor, **no GPU at all** |
| Chatterbox Turbo | 1.594x | spring |
| Chatterbox baseline | 0.635x | spring |
| **Voxtral, `flow_steps=32`** | **0.104x** | **spring, and nowhere else** |

**0.104x is the slowest figure in this repository.** Twenty seconds of speech
costs about 192 seconds of card — about six times slower than baseline, fifteen
times slower than turbo, and twenty times slower than Kokoro on a container
with no graphics card in it at all. At the wrapper's own default of 8 flow
steps it is 0.354x, still under realtime, so there is no setting that changes
the shape of this.

And it buys none of what the other two are slow **for**: it has **no speaker
encoder anywhere in the checkpoint**, so it cannot clone a voice at all. The
expensive engine on this stack is expensive because it clones. This one was
paying Chatterbox's price for Kokoro's capability, on a card that could not
hold it.

## What this costs, stated plainly

**Portuguese.** That was the reason the engine was built: `pt_female` and
`pt_male` are two of its twenty presets, nine languages between them, and
Chatterbox has never been good at Portuguese. Retiring the engine gives that
up, and no substitution is offered in its place — Kokoro has no Portuguese
voice, and putting a caller's `pt_male` request onto an English voice is the
house rule inverted.

The gap is recorded rather than papered over. `model=voxtral` is a **404 at the
gateway naming `GATEWAY_LONG_MODELS`**, not a quiet fall-through to Kokoro; the
mechanism that makes it a 404 is the one ADR 0008 built for turbo and ADR 0009
proved generalises, and it is the reason this retirement is safe to do in
configuration alone.

## What was deleted, and what was kept

| | |
|---|---|
| deleted from `spring` | the **12 GB install** — a 7.49 GB checkpoint plus a second CUDA torch, which could not share `services/chatterbox`'s `torch==2.6.0` pin |
| deleted from `compose.yaml` | `voxtral` in the two model keys, its runner service id, its cold-load and rate seeds, its two quality keys, and the opt-out that existed only for it |
| kept | the catalogue row, the engine code, the refusals, the tests, and this record |

**The per-engine keys had to go with it.** A per-engine key is read only for an
engine `TTS_ENGINES` names, so leaving `TTS_VOXTRAL_FLOW_STEPS` set would have
been a live-looking setting that nothing parses and nothing honours — the same
failure as the `runner_cpu` lane this repository documented for months after
the code behind it had gone, arriving from the other direction.

## The settings, preserved here because they were bought by ear

Losing these would cost the evenings that were spent listening. Anybody
re-enabling the engine starts from this table, **not** from the wrapper's
defaults.

| setting | value | why this value |
|---|---|---|
| `TTS_VOXTRAL_FLOW_STEPS` | **32** | the main quality knob: 32 clearly best, 16 close behind, 8 and 4 audibly worse. The wrapper ships 8 |
| `TTS_VOXTRAL_CFG_ALPHA` | **1.2** | full classifier-free guidance. 1.0 is documented upstream as faster and garbled |
| `TTS_REALTIME_FACTOR_RUNNER_VOXTRAL` | **0.104** | measured, at the two settings above |
| a local rate | **none, deliberately** | with no local rate the synchronous branch is structurally unreachable rather than arithmetically lucky |
| `TTS_COLD_LOAD_SECONDS_VOXTRAL` | **63** | paid on **every** load: 8.01 GB of BF16 read and quantised to int4 at load time, 28.8 s of it the quantisation alone |
| `group_size` | **32** | on the runner. The wrapper ships 64; 32 is finer and cost no VRAM |
| `max_frames` | **2000** | on the runner. 160 s at the codec's 12.5 Hz; the fast path's 500 is a 40-second ceiling |
| `fade_ms` | **120** | on the runner. A squared fade-in over the warm-up artefact. **Trimming frames instead was tried and it ate the first word** |
| `low_pass_hz` | **0**, off | on the runner. Upstream's 6th-order Butterworth at 10 kHz *"sounds like a pilot mic"* |

The two traps ADR 0009 recorded are still traps and are repeated here so that
nobody re-derives them on a busy card: `--quantized` is **not** a pre-quantised
checkpoint and does not remove the 63 seconds, and upstream resamples
**24000 → 48000** and then writes the result back at **24000**, which makes the
file play at half speed.

## How to enable it again

Three settings, and they are three rather than two on purpose:

1. add `voxtral` to `TTS_ENGINES` on `tts-long`;
2. add `voxtral` to `GATEWAY_LONG_MODELS` on `voice-gateway` — **these two must
   agree**, minus the `tts-long` alias, and `docs/tests/test_deployment.py`
   asserts that they do;
3. set `TTS_ALLOW_RUNNER_ONLY_ENGINES=1`, because the engine has no local lane
   and the service refuses to start otherwise.

The third is the one that must stay deliberate. It is the switch that says an
engine may exist only while somebody's gaming PC is on, and on a card that can
hold the load it buys a 503 at submit rather than a segfault.

## Consequences

* **`GET /v1/models` loses exactly one id**, and `model=voxtral` becomes a
  **404 naming `GATEWAY_LONG_MODELS`** at the gateway and a **400
  `model_not_found` naming `TTS_ENGINES`** at tts-long. Neither is a
  fall-through to Kokoro, which was the trap both refusals were built for.
* **The page needs no edit.** Its groups, controls, languages and rates are
  built from `/health.engines`, and no engine id is written as a string
  anywhere in it. The Voxtral group, its two sliders and its disabled Language
  control disappear because the data stopped carrying them.
* **The availability invariant is enforced again rather than excused.** With
  the opt-out unset, `TTS_ENGINES` naming an engine `TTS_LOCAL_ENGINES` cannot
  run is once more a refusal to start. The server does not rely on spring, and
  now no engine does either.
* **ADR 0008's summary table is true again**, in both halves: two engines, and
  both of them run locally. ADR 0009 narrowed that rule to the default engine
  to make room for a third; the narrowed rule is kept, because it is the
  stricter statement of the same promise and it is what the boot checks
  already enforce.
* **Chatterbox Turbo is untouched by this.** It ships: 1.594x realtime against
  baseline's 0.635x on the same passage, and it still clones — its pitch
  tracked the reference clip to within 6 Hz. It is English-only and has no
  `exaggeration` or `cfg_weight`, so it is an **additional** engine and never a
  replacement.
* **Both clone engines stay jobs**, turbo included, for the reason ADR 0008
  gave and this record does not revisit: the card belongs to somebody who is
  often using it.
* **The measurement that decided this is not in any suite.** Every test stayed
  green for the whole life of this engine, on both sides, because a suite reads
  labels and records and not audio. The round trip through Parakeet is what
  read the speech, and it is the only tool here that does.
