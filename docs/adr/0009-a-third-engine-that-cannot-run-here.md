# ADR 0009 — A third engine, and it is the first one this container cannot run

**Status:** superseded by [ADR 0010](0010-the-third-engine-was-measured-and-retired.md)
**Date:** 2026-09-10
**Builds on:** [ADR 0008](0008-two-engines-and-both-stay-jobs.md), which stands
in every part except its own summary table. That table says *two* engines and
says **both run locally**. Both halves stop being true here, and the second half
was an invariant rather than a fact, so this record has to say what replaces it.
Also [ADR 0007](0007-two-lanes-not-three-rungs.md), which is untouched: this is
still two lanes.

## Decision

`services/tts-long` offers a **third engine**, `voxtral`, named on the wire by
the `model` field exactly as the other two are. It answers with a job id, it
**cannot clone a voice**, it **cannot run on this container's processor**, and
its twenty voices are fixed in the weights.

| `model` | engine | voices | languages | delivery controls | lanes |
|---|---|---|---|---|---|
| `chatterbox` | multilingual Chatterbox, 801M | any reference clip | 23 | `exaggeration`, `cfg_weight`, `temperature` | runner **and local** |
| `chatterbox-turbo` | Chatterbox Turbo, 743M | any clip over 5 s | 1 | `temperature` | runner **and local** |
| **`voxtral`** | **Mistral Voxtral-4B-TTS-2603, int4** | **twenty, fixed** | **9** | **`flow_steps`, `cfg_alpha`** | **runner only** |

`voxtral` is reached by typing `voxtral` and by nothing else. No alias resolves
to it: `tts-long`, an absent `model`, and OpenAI's three documented names all
resolve to `TTS_DEFAULT_ENGINE`, which this record makes the service refuse to
start without a local lane for. **A caller cannot be handed this engine, and a
caller cannot lose the service because this engine's machine is off.**

## What Voxtral is, and the one word that decides the whole design

Mistral's `Voxtral-4B-TTS-2603` checkpoint, served through the third-party
`github.com/TheMHD1/voxtral-int4` wrapper, which quantises it to int4 with HQQ
over torchao's tinygemm kernels.

**It is not a cloning engine.** That is not a limitation of the wrapper and it
is not a parameter somebody forgot to expose:

* Its speakers are **twenty `.pt` tensors** shipped beside the weights —
  embeddings, already trained, one file each.
* There is **no speaker encoder anywhere in the checkpoint**. There is nothing
  that could turn a clip into one of those tensors.
* **No reference-audio parameter exists in any of the wrapper's nine source
  files.** A clip cannot be honoured in part, degraded, or approximated.

So it sits beside **Kokoro** conceptually — a preset-voice engine — and not
beside Chatterbox, whatever the fact that it lives in `tts-long` suggests. The
twenty:

```
ar_male  casual_female  casual_male  cheerful_female  de_female  de_male
es_female  es_male  fr_female  fr_male  hi_female  hi_male  it_female
it_male  neutral_female  neutral_male  nl_female  nl_male  pt_female  pt_male
```

Nine languages, **including Portuguese**, which is the one this deployment's
owner actually needs and the one Chatterbox has never been good at.

**The language is a property of the voice, not a field beside it.** `de_female`
is German because of which tensor it is. Nothing parses the `de_` prefix to find
that out, here or in the browser, because prefix-parsing a voice name is exactly
how Kokoro's `PREFIX` map resolves `de_female` to `en-us` by coincidence.

## Every number here was measured on this project's own hardware

`spring` is the owner's gaming PC: Windows, **RTX 3070, 8 GB**. Every figure
below was taken on that card this week, through the settings this deployment
ships. Nothing is a vendor claim and nothing is re-derived from a datasheet.

| engine | configuration | rtf | where |
|---|---|---|---|
| Kokoro | orko, eight Xeon threads, **no GPU at all** | 2.2–2.7x | the fast path |
| Chatterbox Turbo | spring, RTX 3070 | 1.594x | the runner |
| Chatterbox baseline | spring, RTX 3070 | 0.635x | the runner |
| **Voxtral** | **spring, RTX 3070, int4, `flow_steps=32`** | **0.104x** | **the runner, and only there** |
| Voxtral | the same card at the wrapper's own default `flow_steps=8` | 0.354x | — |

**Those Chatterbox figures are a same-passage comparison and not the ones in
ADR 0008.** 0008 records 0.6531x and 1.5426x from the trial that decided turbo;
the 0.635x and 1.594x above come from re-running the *same passage* the Voxtral
runs used, so that the four rows are comparable to each other. They agree to
within a per cent and neither supersedes the other. Two numbers for one thing,
each honest about what it measured, is better than one number quietly reused.

What 0.104x means in the units a caller feels:

* **20 seconds of speech costs about 192 seconds of card.** Three and a bit
  minutes of a GPU somebody else is trying to play games on.
* It is about **six times slower than Chatterbox baseline**, **fifteen times
  slower than turbo**, and **twenty times slower than Kokoro on a processor with
  no graphics card in it at all.**
* Even at the wrapper's own default of 8 flow steps it is 0.354x — still under
  realtime. **There is no setting at which this engine could have been a
  synchronous route.** It is a job, and unlike turbo that is not a decision
  somebody made; it is arithmetic.

## The settings, and that they were chosen by ear

The owner tuned this by listening, after several wrong turns, and this section
exists so that nobody re-derives it from a benchmark.

| setting | value | why this value |
|---|---|---|
| `flow_steps` | **32** | **The main quality knob.** 32 was clearly best; 16 close behind; 8 and 4 audibly worse. The wrapper ships 8. |
| `cfg_alpha` | **1.2** | Full classifier-free guidance. 1.0 is documented upstream as faster and garbled. |
| `group_size` | **32** | The quantiser's group size. The wrapper defaults to 64; 32 is finer and **cost no VRAM at all**. |
| `max_frames` | **2000** | 160 s at the codec's 12.5 Hz. The wrapper's fast path defaults to 500 — a **40-second ceiling** — and the standard path uses 2000. |
| low-pass | **off** | See below. |
| fade-in | **120 ms, squared** | See below. |

**The low-pass filter is removed, and that is a correction to upstream rather
than a preference.** `audio_postprocess.postprocess_audio` runs a 6th-order
Butterworth at 10 kHz over everything it touches. The owner heard it
immediately and called it *"sounds like a pilot mic"*. What replaces it is
resample plus peak normalise and nothing else. The filter is still reachable —
`--low-pass-hz` is a runner flag and `0` means off — because removing a knob is
how the next person has to patch code to hear what it did.

**The warm-up glitch is faded out, never trimmed, and the difference ate a
word.** There is a known upstream warm-up artefact in the first frames
(HF discussion #20). Upstream's `trim_warmup_frames` only catches a run of
*identical* leading codes and misses the rest of it. **Dropping frames instead
was tried here and it ate the first word** — the owner caught it. The fix that
ships is a **squared fade-in over the first 120 ms**: the artefact is attenuated
and **every sample of speech survives**. A fade is not a trim and the two must
never be confused again in this repository.

## Why this engine cannot run here, and what was tried

> **ADR 0008's rule was: every engine this deployment advertises is renderable
> on the local lane. Voxtral is the first engine that breaks it, and the rule is
> narrowed rather than abandoned.**

The narrowed rule, and it is enforced at boot rather than hoped for:

> **`TTS_DEFAULT_ENGINE` must have a local lane, and no alias may resolve to an
> engine that has not got one. A request that does not name Voxtral cannot fail
> because spring is off.**

So the service still does not rely on spring. **One engine does**, visibly, by
name, and only for callers who typed that name.

### A processor lane does not exist, and not merely slowly

This was the first thing tried, because it is what makes turbo safe. It is not
available here, and three independent walls say so:

1. **`torchao/.../int4_tile_packed_to_4d_tensor.py:124` calls
   `torch.cuda.get_device_capability()` inside `from_hp`**, before any device
   dispatch. There is no CPU path through the quantised weights.
2. **HQQ is asserted to require exactly `TILE_PACKED_TO_4D`**, so the layout
   cannot be swapped for one that has a processor kernel.
3. **The BF16 escape hatch calls `torch.cuda.synchronize()` unconditionally on
   the decode path.** Even the unquantised fallback is CUDA-only.

And a fourth that matters more than the three, because it means a processor
lane would not even be the same engine: **`flow_steps` and `cfg_alpha` exist
only on the fast path.** The standard path is first-order Euler
(`model.py:335-347`); the fast path is the second-order midpoint solver
(`generate_fast.py:253-273`) that the 32 was chosen against by ear. A processor
lane would silently be a different solver at a setting nobody listened to.

Estimated cost if all four were solved: **0.002–0.004x — a 20-second clip in one
and a half to three hours**, holding 8 GB resident and evicting Chatterbox from
`TTS_LOCAL_RESIDENT_MAX=1` for the whole time. Inventing a `local_seed` for that
would be the `chatterbox-cpu` trap a second time: a lane configured, published
and unable to do work.

### The three other answers to "spring is off", and why each is worse

**Hold the job queued until the runner comes back.** *This is the option that
actually breaks the rule.* `_pick` returns `None` and loops with no deadline;
`_sweep` skips anything with `finished_at is None`; `TTS_MAX_QUEUE` is **32 and
deployment-wide**. Thirty-two Voxtral jobs submitted overnight make `_full()`
answer **429 to every Chatterbox caller on a completely idle lane.** A 202
nothing can serve is a progress bar that never moves, and here it takes the
other two engines down with it.

**Substitute Kokoro.** The house rule inverted: a field accepted and quietly
turned into something else. `pt_male` has no Kokoro equivalent and `de_female`
has none at all. It is right as an **offer** and wrong as a substitution — the
page already has `nearestKokoro()` and a *"use this instead"* button, and at
0.104x the *"this will take over ten minutes"* warning trips at about **62
seconds of audio**, so nearly every Voxtral request shows it. The button appears
only when there is a Kokoro voice in the resolved language, and is **absent
rather than wrong** otherwise.

**What ships instead:** a **503 `engine_unavailable` at submit, before a job id
exists and before a queue slot is taken**, using the refusal that already exists
for the runner-only case; and for a job already accepted whose lane then
disappears, a **deadline** — `TTS_RUNNER_ONLY_DEADLINE_SECONDS`, 900 by default
— after which the job reaches a **terminal `failed` record** with the runner's
own reason in it. The clock counts *time with no eligible lane*, never time in
the queue, because a legitimately long render must not trip it.

**A Voxtral job never falls back and `fell_back` is never set for it.** Nothing
may ever render *"…after spring gave up"* about a machine that never took it.

## What it costs to install, all of it measured

| cost | figure | consequence |
|---|---|---|
| weights | **7.49 GB**, `mistralai/Voxtral-4B-TTS-2603`, ungated | a real download on somebody's home connection |
| cold load | **63 s, every single time** | 8.01 GB of BF16 read into RAM and quantised to int4 at load, **28.8 s of it the quantisation alone**. There is no pre-quantised checkpoint shipped. |
| VRAM resident | 3.5–3.7 GB | comfortable |
| **VRAM peak at load** | **8265 MiB on an 8192 MiB card** | **73 MiB over what the card physically has.** It completed only because the desktop was near idle. This is an operational risk, not a footnote. |
| torch | **2.11+ required** | `services/chatterbox` pins `torch==2.6.0`, so **this cannot share that virtual environment**. A second CUDA torch is about 2.5 GB more on somebody's gaming PC. |
| undeclared dependencies | `tiktoken`, `scipy` | the wrapper's README omits both and it imports both |
| a Linux-only bug | `generate.py` opens `tekken.json` with no `encoding=` | Python on Windows uses cp1252 and the tokeniser dies. `PYTHONUTF8=1` fixes it **without patching upstream**, which is why the runner passes `-X utf8`. |

### The 63 seconds are not removable with a flag, and the flag that looks like it is not

`--quantized` exists in the wrapper and it is **not** a pre-quantised checkpoint
path. It routes to `load_quantized_model`, an unrelated **TurboQuant** experiment
that imports a package which is not in the repository, not on PyPI and not
installed — an immediate `ImportError` — and whose own docstring says it
dequantises back to BF16, after which it performs the identical
`model.to(device, torch.bfloat16)` and takes the identical spike.

Removing the 63 s means **serialising torchao's int4 state dict**: new code, not
a flag. It is recorded here as future work precisely so that the next person
does not spend an afternoon discovering that the flag is a decoy. It would
remove both the load cost and the memory spike, which is the single highest-value
follow-up this engine has.

### What happens when it runs out of memory, and why the job dies rather than the process

The margin is 73 MiB on a card whose owner may start a game at any moment, so
this has to have a designed answer rather than a hope.

Nothing in the runner agent checks it for us. `MemoryAllows` (`Agent.cs:1292`)
is a **system-memory** paging admission check, and `ForeignVramBusyMiB` decides
whether *the owner* is using the card, not whether *our* job fits. `StartBest`
reads **no exit code, has no backoff and has no failure counter**.

So: **the manifest is published first, the model is loaded lazily on the first
job, and a load that cannot fit fails _that job_ with both numbers in the
sentence.** A VRAM preflight runs before the load and a `torch.cuda.OutOfMemoryError`
that gets through anyway lands in the same place. **The process stays alive.**

The alternative — check and exit — is worse in a way that is invisible: a
controller that dies before publishing its manifest never writes `service.json`,
leaves its lease in `pending/`, and **is relaunched every 500 ms for ever** while
the client sits on `queued`. Loud in `logs\`, silent everywhere a person looks,
and thirty-two of them answer 429 to every Chatterbox caller. Failing the job
turns a silent infinite loop into one terminal record with a readable sentence.

> **OPEN, AND IT BELONGS TO THE RUNNER PACKAGE.** The VRAM floor is written as
> `--min-free-vram-mib 8400` and is checked against `torch.cuda.mem_get_info()`.
> On this card that call reports a **total** of 8192 MiB, so free memory can
> never reach 8400 and the preflight would refuse **every** job before it ever
> loaded. The floor has to be reconciled with the card before this ships —
> either as a number below the physical total, or by comparing against a budget
> that counts the shared-memory spill the 8265 MiB peak evidently used. The
> number is recorded here as measured rather than quietly adjusted, because
> adjusting it in the documentation would hide a defect in the code.

## The output is 24 kHz, and that is a wire contract rather than a preference

**There is a sample-rate bug in the upstream repository and this stack must not
reproduce it.** `postprocess_audio` resamples **24000 → 48000**, and then
`generate.py`, `generate_fast.py` and `benchmark_all.py` all write the result
back **at 24000**. Only `serve.py` gets it right. Audio that has been through
post-processing is 48 kHz; audio from the non-fast `generate_speech` is raw
24 kHz. Getting this wrong makes the file **play at half speed**, which is
exactly what happened here the first time round.

The decision that closes it: **this stack emits raw 24 kHz and never
post-processes to 48 kHz.** `voice_common/audio.py:31` fixes `SAMPLE_RATE =
24_000` estate-wide and `remote.py:942` already sends that number, the codec
produces 24 kHz natively, and the 48 kHz only ever existed because upstream's
post-processing resampled. So there is no resample, no rate mismatch and one
fewer place for the half-speed bug. The rate is published as a fact on the
engine row (`native_sample_rate`), asserted at load, carried on the wire, and
written into the record.

## Every field this engine cannot honour is refused by its own name

`services/stt/app/openai_api.py` states the house rule and this whole surface is
built on it: **every field is either honoured or refused by name.** None is
accepted and dropped.

Voxtral has no reference clip, no `exaggeration`, no `cfg_weight` and no
`temperature`, and its voices are twenty fixed names. All of it is a **400
naming the field**, never a shrug:

| the caller sends | what comes back |
|---|---|
| a voice that is not one of the twenty | 400 on `voice`, saying this checkpoint carries its speakers as embeddings baked into the weights and has no speaker encoder of any kind, listing the twenty and the ones in the language asked for, and naming `model='chatterbox'` for cloning |
| a clip upload, or one of the OpenAI voice aliases | 400 on `voice`. The aliases resolve to reference clips here, and **they are not mapped onto a preset by ear** |
| `exaggeration` or `temperature` | 400 naming the field: this is a flow-matching checkpoint with no emotion conditioning and a sampler that does not read temperature |
| `cfg_weight` | 400 on `cfg_weight`, **naming `cfg_alpha` as the near-miss it is** — a different scale and a different solver, where 1.2 is full guidance and 1.0 is documented upstream as faster and garbled |
| `flow_steps` or `cfg_alpha` to either Chatterbox | 400 naming the field: these are flow-matching solver settings and neither Chatterbox has a flow-matching decoder |
| `language` that contradicts the chosen voice | 400 on `language`, naming a voice in the language asked for |
| `speed`, `instructions` | 400, in a sentence that names **no engine** — no engine in the catalogue has a rate control or instruction conditioning, so the reason is a property of the catalogue |

**Equality with a deployment default is still not consent**, exactly as ADR 0008
decided. And the rule now cuts one step further out: **a global key that reaches
no enabled engine is fatal at boot.** `TTS_EXAGGERATION` on a box running only
Voxtral is the same lie as a per-engine key for a control that does not exist,
relocated one file away.

**`TTS_VOXTRAL_LANGUAGE` is fatal at boot with its own reason**, and it is the
one nobody caught: the defaults builder unconditionally stamps a deployment-wide
`language` onto every job, which would have written `en` onto a `pt_male`
request. Voxtral's language is carried by the voice embedding, so a
deployment-wide default would either agree with the voice or contradict it.

## Consequences

* **`GET /v1/models` gains exactly one id, and only where it is enabled.** The
  gateway knows every name the catalogue holds, so a deployment that has not
  enabled `voxtral` answers **404 naming the variable**, rather than falling off
  the end of the routing table and serving Kokoro. That trap was closed for
  turbo in ADR 0008 and this is the row that proves it generalises.
* **No new route anywhere.** A third model string is a field value, not an
  endpoint. The gateway's three allowlists are untouched.
* **The page never streams it and never offers "generate and listen".** At
  0.104x the stream threshold refuses it on the rate alone, which is the correct
  mechanism — the page must decide from the engine's published rate and not from
  the engine's name.
* **A preset engine's voices are published on `/health.engines[*].voices` with
  the language on each voice.** That is the only honest home for twenty names: a
  list in the browser would be the page's fifth model table, and the page's own
  comment says why there must not be a fifth.
* **`TTS_CHARS_PER_SECOND` stays one number, again.** Voxtral renders slower; it
  does not *talk* slower. This is the second engine to test that sentence and it
  is still true.
* **The record carries the settings that produced the sound** — `flow_steps`,
  `cfg_alpha` and the runner's load-time settings — as additional columns on the
  existing allow-list. A tuning log is worthless if two rows can differ in a way
  that is not written down, which is also why the fade and the low-pass are
  identical across every segment of every job and are not per-request fields.
* **Four load-time settings are refused as job parameters by the runner, by
  name.** `group_size` is consumed once inside a 63-second load; `max_frames`
  pre-allocates the KV cache and is a VRAM decision taken on a card with a
  73 MiB margin. **The caller does not get to cap the model with a number they
  picked.** All four are still readable from a browser, on `/health`, so
  answering *"what is spring actually serving"* needs no SSH session.
* **The engine that ran must be asserted to be the engine that was asked for.**
  An audit last round found **four single-token edits that produce baseline audio
  from a turbo request** while the response header, the job record and
  `engine_reason` all still say turbo — and **all four keep every suite green.**
  A third engine widens that hole. One test closes it for all three: generate
  with each and assert something only that checkpoint can produce. For Voxtral
  that is its 12.5 Hz frame grid, which nothing else in this stack has.

## What this deliberately does not add

| left out | why |
|---|---|
| a synchronous or streamed Voxtral route | 0.104x. Not a decision, arithmetic |
| a Voxtral processor lane | three walls in torchao, a CUDA-only decode path, and a different ODE solver at the one setting that was tuned by ear |
| a fourth `kind` on the run record | the record's `kind` is derived from whether the engine reads a reference clip; a fourth value is a fourth table that must agree |
| `fade_ms` and `low_pass_hz` on the wire | they shape the artefact, not the generation, and they must be identical across every segment of a listening comparison. Published and recorded either way |
| a pre-quantised checkpoint | the highest-value follow-up, and it is new code rather than the decoy flag |
| a Kokoro catalogue row | it would make the gateway 404 Kokoro the moment it lands, so the row and the routing change must be one commit. **Named debt** |
| re-benchmarking any of this | the card is finishing a render. Every number here was measured on it this week, and the observation count published beside each seed says out loud that a seed is a seed |
