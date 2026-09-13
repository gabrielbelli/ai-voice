# ADR 0008 — Long-form speech offers two engines on the `model` string, and both of them stay jobs

**Status:** accepted
**Date:** 2026-09-10
**Builds on:** [ADR 0007](0007-two-lanes-not-three-rungs.md), which stands. Two
lanes was a decision about *machines*; this is a decision about *checkpoints*,
and it changes exactly one of 0007's assumptions — that there is one engine to
put in a lane.

## Decision

`services/tts-long` offers **two engines**, named on the wire by the `model`
field, and **both of them answer with a job id**.

| `model` | engine | languages | delivery controls | on spring's 3070 |
|---|---|---|---|---|
| `chatterbox` | the multilingual checkpoint that ships today | 23 | `exaggeration`, `cfg_weight`, `temperature` | 0.6531x realtime |
| `chatterbox-turbo` | Chatterbox Turbo | **English only** | `temperature` **and nothing else** | 1.5426x realtime |

`tts-long`, an absent `model`, and OpenAI's three documented names resolve to
`TTS_DEFAULT_ENGINE`, which is `chatterbox` on this deployment. Anything else is
a **400 naming the field**. There is no substitution, no negotiation and no
preference: **a caller cannot receive an engine it did not name.**

Both engines run on the local CPU as well as on the card, and that is an
invariant the service refuses to start without.

## Every number here was measured on this project's own hardware

`spring` is the owner's gaming PC: Windows, Ryzen 7 5700X3D, **RTX 3070, 8 GB**.
`orko` is the NAS the containers run on: Xeon E5-2697 v4, eight threads to this
container. Nothing below is a vendor claim.

| configuration | rtf | vs baseline | VRAM | note |
|---|---|---|---|---|
| chatterbox fp32 | 0.6531x | 1.00 | 4199 MiB | what ships today |
| **turbo fp32** | **1.5426x** | **2.36** | 3805 MiB | past realtime |
| turbo fp16 | 1.5361x | 2.35 | 3393 MiB | no gain over fp32 |
| tf32 on | 0.6468x | 0.99 | 4199 MiB | no gain; left off |
| bf16 | 0.6140x | 0.94 | 3259 MiB | **slower** |
| fp16 | 0.6035x | 0.92 | 3259 MiB | **slower** |
| fp16 autocast | 0.5713x | 0.88 | 4199 MiB | slower still |
| `torch.compile` | failed | — | — | Triton is absent on Windows |

On processors: baseline runs at **0.230x on orko** and 0.271x on spring, both at
eight threads. Turbo on a processor has never been measured on either box.

**Precision is a VRAM lever and not a speed lever, and that is settled here so
nobody re-derives it.** The cast genuinely happens — weights fall from 3057 to
2035 MiB — and it is still slower, because this model is autoregressive at batch
one and bound by kernel-launch latency rather than by matmul throughput. There
is no precision knob in this design and there is not going to be one: a control
that cannot help is a control that costs a reader time for ever.

## Why the engine is the `model` string

The owner's words were *"we are going to use baseline and turbo as the options
available … make available on the api"*. A named engine on `model` is the literal
reading, and it is the only candidate where a caller can never be handed an
engine it did not ask for.

Two alternatives were designed out and both are worth recording.

**The voice is not the selector.** Registering `gabriel-turbo.wav` beside
`gabriel.wav` would double the voice list, would require registering a second
artefact to change one parameter, and would refuse to put `chatterbox-turbo` on
the API at all — which is the thing that was asked for.

**A preference field is not the selector.** `prefer=speed` is a third option
nobody asked for, and the day it fires it hands back audio about 11 dB quieter,
in a different language set, from a checkpoint the caller never named. The
explicability debt gets paid in a record field the API caller never reads.

The engine lives in one table, `packages/common/voice_common/engines.py`, which
both `tts-long` and the gateway import. A third engine is a row in that table
rather than a branch anywhere.

## Why turbo stays a job, permanently

**1.5426x realtime is past the threshold where a request could be answered down
the socket, and the owner has decided it stays a job anyway.** That is not a
deferral and it is not pending a benchmark.

The reason is that **1.54x is the figure on that card, and that card belongs to
somebody who is often using it.** The runner hands its GPU back the instant its
owner touches the machine. A synchronous route would therefore answer in about
nine seconds when spring is idle and in about four minutes when spring is
gaming — the same request, the same text, the same voice, a twenty-five-fold
spread decided by whether somebody started a game. **A route that is sometimes
fast is worse than a route that is always a job id**, because the caller cannot
plan around it and the failure arrives as a timeout in somebody else's client.

Nobody is to improve this into a synchronous route, a streamed route, or a
special case that sometimes waits.

`_sync_budget` still applies to both engines identically — a short input may
finish inside the existing window exactly as baseline does today — but that
budget is computed from the **local** rate and never from a runner rate. This
server does not make promises on spring's behalf.

## What turbo costs, and why every cost is a refusal rather than a footnote

`services/stt/app/openai_api.py` states the house rule and this whole surface is
built on it: **every field is either honoured or refused by name.** None is
accepted and dropped. A dropped field is a client that believes something false
about the audio it just received.

Turbo breaks that rule *inside the package*, which is exactly why this service
must not pass it through:

| what is missing | established how | what this service does |
|---|---|---|
| `language` | `generate()` has **no `language_id` parameter**; the deployed controller call raises `TypeError` immediately. 23 languages collapse to English. | 400 on `language`, naming the engine that does speak it |
| `exaggeration` | `hp.emotion_adv` is `False`, so the conditioning layer is never built | 400 on `exaggeration` |
| `cfg_weight` | there is no classifier-free-guidance path in `inference_turbo` at all | 400 on `cfg_weight` |

Both are **accepted as keyword arguments and silently discarded with a logged
warning** — it fired on all 21 segments of the trial generation. The deployment
sets `TTS_EXAGGERATION=0.3` and `TTS_CFG_WEIGHT=0.3`, so without this decision
every turbo request would arrive carrying two fields turbo cannot honour, *from a
configuration file rather than from a person*, and lose them without an error
anywhere in the stack.

The rule that follows, and it is the one a reviewer will push on:
**equality with a deployment default is not consent.** `{"model":
"chatterbox-turbo", "exaggeration": 0.3}` is a 400 even though `0.3` is what
`compose.yaml` sets, because the caller typed the field and believes it did
something. Defaults are resolved **after** the engine is known, from that
engine's own control set, so an engine with no such control gets no default, no
environment key and no slider.

The rest of the bill, all of it real:

* **Output is about 11 dB quieter, by design.** Turbo normalises to −27 LUFS;
  measured RMS is 0.070–0.104 against the current model's 0.132–0.170. Nothing
  compensates for this — a gain stage applied on our side would be this service
  disagreeing with the checkpoint about what it produced.
* **Cold load is 67.5 s against baseline's 22.2 s**, both timed on spring; 20.0 s
  warm. A 1.83 GB checkpoint plus an `AutoTokenizer`. A service that yields the
  card and reloads pays this every time, which is why the runner section runs
  turbo with a 240 s idle window rather than the default 60 — a 60 s idle window
  against a 68 s reload is a card that spends more time loading than speaking.
* **A five-second floor on the reference clip**, which turbo asserts itself. It
  is a property of the *pair*, not of the engine and not of the voice, so it is
  published per voice on `GET /voices` and checked before a job id exists.
* **743M parameters, not the 350M advertised.** The current model is 801M. The
  2.36x is architectural, not a size difference worth repeating.
* **MIT licence, the same as the current model.** Nothing changes here.

## Spring is speed, never availability

> **Every engine this deployment advertises is renderable on the local lane.**

This is the rule the owner has repeated all week, and turbo is the first thing
that could have broken it. An engine that only exists while somebody's gaming PC
is switched on is an engine that disappears, and the first Friday evening that
hole eats a job.

It is enforced at startup rather than hoped for: `TTS_ENGINES` naming anything
`TTS_LOCAL_ENGINES` cannot run is a **refusal to start**, not a warning, unless
`TTS_ALLOW_RUNNER_ONLY_ENGINES=1` says the operator accepts a 503 at submit
whenever the runner is away. A job is never created that nothing can serve: an
option honestly and visibly absent is fine, but a 202 nothing can finish is a
progress bar that never moves, and thirty-two of them answer 429 to every
*baseline* caller on a completely idle lane.

**The fallback is gentler than the one already shipping.** Turbo on a processor
is 3.0–3.2x baseline on the same processor, so falling from turbo-on-card to
turbo-on-orko costs about 2.5x. Falling from baseline-on-card to baseline-on-orko
costs about 3x, and this deployment does that several evenings a week and calls
it normal.

## One lane, two engines, and baseline wins the card

There is still **one runner lane**. The agent starts at most one controller per
device group and does not pre-empt, so the card physically holds one process; a
second lane would let the dispatcher believe it has a slot it will never get.
The engine rides on the job and resolves to an `idlegpu` service id at submit.

When both engines want the card, **baseline wins**: turbo is registered at
`Priority = 9` against baseline's `10`. The reason is not that baseline is more
important but that **it has the worse fallback**. Baseline serves 23 languages
and two expressive controls that have no substitute anywhere; turbo always has a
local floor only ~2.5x slower. Reversing this starves baseline off the card under
a steady turbo trickle, and every starved baseline job then reports
`machine_busy` — *the same string a game produces* — so it would be
indistinguishable from a week of gaming.

## Consequences

* **`GET /v1/models` gains exactly one id.** An OpenAI-shaped client that has
  never heard of turbo sends `tts-1` and reaches Kokoro, unchanged; sends
  `chatterbox` and gets today's behaviour byte for byte; and never reaches turbo
  without typing its name.
* **The three OpenAI names become documented aliases for this service's default
  engine**, not fields accepted and dropped. That is what they have always meant
  here, and `x-tts-engine` on every response says which engine actually ran.
  `services/tts-long/README.md` records it as a deviation, because it is one.
* **`POST /jobs` starts refusing unknown fields.** It has always accepted and
  silently discarded them, and the page uses `/jobs` exclusively. Shipping
  `model` on `/v1/audio/speech` without it would ship a page that asks for turbo
  and gets baseline with no error anywhere — the same shape as two bugs already
  shipped this week. `extra="forbid"` and the `model` field are one commit.
* **The realtime-factor EMA is keyed per (lane, engine).** ADR 0007's argument
  for per-lane rates applies one level down and harder: the two engines are 2.36x
  apart on the *same* card, so one average over both describes no configuration
  that exists. `/health` keeps `realtime_factor_by_backend` narrowed to the
  default engine and adds `realtime_factor_by_engine` beside it.
* **`TTS_CHARS_PER_SECOND` stays one number.** It measures how long the speech
  *is*, not how long it takes to make. Turbo renders faster; it does not talk
  faster. Splitting it would double-count the speed-up in every estimate.
* **A per-engine configuration key naming a control that engine does not have is
  fatal at boot**, with a message that names the key, the reason and the way out.
  A key that does nothing is the house-rule failure with a longer fuse.
* **The adapter checks the installed package against the catalogue at load** —
  `inspect.signature` for `language_id`, `hp.emotion_adv` for the conditioning
  layer. A turbo point release that adds either must fail the container at start
  with a named message, not one job at a time.
* **Turbo ships local-first.** The first slice puts it on the API running on
  orko's CPU with `idlegpu` untouched. That proves the local-floor rule instead
  of promising it, and it means nobody can later argue the card into being a
  requirement.

## What this deliberately does not add

| left out | why |
|---|---|
| a synchronous or streamed turbo route | ruled out above, permanently |
| a precision knob | measured and settled: fp16 and bf16 are both *slower* |
| a "turbo preferred, baseline acceptable" field | a guess with a voice attached, and the guess is the thing that produces audio the caller did not ask for |
| a second dispatcher lane | one card, one controller, one lane; the engine rides on the job |
| per-engine `chars_per_second` | no measurement supports a difference |
| backfilling `engine` on historical records by inference | `_recover` writes `chatterbox` because that is *true* — it was the only engine. Anything beyond that is a guess indistinguishable from a measurement a year later |
