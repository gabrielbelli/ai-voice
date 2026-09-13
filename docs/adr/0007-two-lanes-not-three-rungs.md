# ADR 0007 — Long-form speech has two lanes, and the runner's processor is deleted rather than left switched off

**Status:** accepted
**Date:** 2026-09-09
**Supersedes:** [ADR 0006](0006-three-way-speech-routing.md)
**Extended by:** [ADR 0008](0008-two-engines-and-both-stay-jobs.md) — one lane,
two engines. This ADR's lanes are unchanged by it.

## Decision

`services/tts-long` has **two lanes**, each one job wide, chosen per job by
arithmetic rather than walked in order.

| lane | what it is | seed | measured on |
|---|---|---|---|
| `local` | this container | 0.230x | orko, Xeon E5-2697 v4, 8 threads |
| `runner` | the runner's card | 0.70x | spring, an RTX 3070, midpoint of a measured 0.644 to 0.746 |

```
finish(lane) = work already in the lane + A / rate(lane) + handover(lane)
choose the smallest, subject to: a remote lane must beat local by 1.25x
```

`runner_cpu` is **not a lane**, and it is not a lane that is switched off: the
lane key, its gate, its promotion rule and two of its four environment variables
are gone. (Its *client* is not gone, and neither are the other two variables.
See the dated correction in **Consequences**.)
`TTS_BACKEND_ORDER` survives as a membership test — naming
a lane turns it on, leaving it out turns it off, `local` alone is local-only —
and naming `runner_cpu` in it does nothing at all.

## Why the third rung went, and neither reason is "it was complicated"

**One: it was five per cent, not two or three times.** Chatterbox is
autoregressive — its transformer emits speech tokens one at a time at batch size
one — so it is bound by single-thread latency rather than by throughput, and a
thread sweep on orko says so (0.077x at 2 threads, 0.230x at 8, 0.285x at 16:
per-thread efficiency halves). That is the shape a 2016 Broadwell-EP server part
is worst at and a 2022 Zen 3 desktop part with a large cache is best at, which
is why the hypothesis was worth measuring. Measured through the shipped path:
**0.271x on spring's Ryzen 7 5700X3D against 0.230x here.** ADR 0006 recorded the
same comparison as 0.24 against 0.23 and drew the same conclusion — the rung is a
fallback for when the card is busy, not a faster machine.

The two-lane arithmetic therefore refuses it without needing a special case. On
300 s of speech: 1304 s here, and roughly 1100 s on spring's processor plus an
8 s hop — and 1108 × 1.25 is more than 1304, so it would stay here.

**Two, and this is the one that settles it: the rung was never reachable.** The
`idlegpu` agent on spring registers exactly two services, `echo` and
`chatterbox`. It has never offered `chatterbox-cpu`. Every offer this service
ever made to that rung was answered `no_such_service`, for the whole life of the
rung, in the only deployment there is. Nothing failed and nothing was logged as
a fault, because a rung that refuses is indistinguishable from a rung that is
busy.

It could not have been reached even if the service had existed. `local` is a
rung and `local` is always willing, so on a ladder nothing ordered after it is
ever asked — and the shipped order put the runner's processor last.

## Why the whole thing is deleted, and not left as a default-off knob

Eight tests passed against code that had never run a job. `compose.yaml`
documented `TTS_RUNNER_CPU_MIN_PCT` and `TTS_RUNNER_CPU_WHEN_BACKLOG_S` as live
knobs and recommended `TTS_BACKEND_ORDER: "runner,local,runner_cpu"` in a
comment, so **an operator following this repository's own advice configured a
lane that silently did nothing.** A knob that is read by no code is worse than
no knob: it survives every grep, it reads as a supported feature, and the only
way to find out it is inert is to measure a deployment and find the work
somewhere else.

The same argument is why a measured five per cent is written down here rather
than rounded up. The next person to have this idea should find the number that
killed it, not the absence of the feature.

## What replaces it, and why that is worth having

The ladder's real cost was never the third rung, which did nothing. It was that
one worker thread walked it:

* **A runner that answers slowly delayed a job that was always going to run
  here.** `offer()` ran on the worker's own thread, once per job per rung,
  deliberately uncached, against a 30 s job timeout. Spring switched off is the
  cheap case — a refused connection, instantly. Spring **asleep**, wedged, or
  behind a firewall that drops rather than rejects is the expensive one: the
  full timeout, charged to a local job. `LaneProbe` now asks on its own thread
  with its own 3 s clock, and the chooser only reads the answer it left behind.
  An answer older than three intervals shuts the lane, because a probe stuck on
  a socket must not leave the last good answer standing.
* **The second job waited for the first even when the first had left the
  building.** One worker meant a ten-minute job on spring's card held orko's
  completely idle CPU for the whole ten minutes; measured, the next job started
  twenty-one minutes late. Two lanes is the entire win.

Width is one on both lanes, and local's is not configurable: `Synth._speak`
holds one lock across `_ensure_loaded()` and `generate()`, so two local jobs
would interleave at segment granularity for no extra throughput and double the
latency of each. Every concurrency here comes from the second machine.

## Consequences

* **`TTS_RUNNER_MAX_WAIT` drops from 900 to 300, and `compose.yaml` stops
  overriding it.** Fifteen minutes was correct when a yield stopped the whole
  service: one worker, one job, so giving up meant re-speaking on this CPU with
  everything submitted since queued behind it. With two lanes a yield costs one
  lane and this host carries on, so the trade is five minutes of a progress bar
  that has not moved against fifteen — and fifteen reads as a hang followed by
  an unexplained restart. The default moved in `app/remote.py` under an explicit
  `"900"` in compose, so production kept the old number; that override is now
  gone.
* **A yield puts the job back at the head of the queue** and cools that lane for
  `TTS_RUNNER_COOLDOWN_S`. The job has already waited its turn once; sending it
  to the tail would let everything submitted since overtake it because somebody
  started a game.
* `compose.yaml` no longer sets `TTS_RUNNER_CPU_SERVICE`, no longer recommends
  an order containing `runner_cpu`, and no longer documents
  `TTS_RUNNER_CPU_MIN_PCT`, `TTS_RUNNER_CPU_WHEN_BACKLOG_S`,
  `TTS_RUNNER_CPU_MAX_WAIT` or `TTS_REALTIME_FACTOR_RUNNER_CPU`. It documents the
  lane knobs that are read instead.
* **Correction, 2026-09-10.** This bullet used to claim that
  `RTF_SEED_RUNNER_CPU`, `RunnerConfig.cpu_service`, `cpu_max_wait`,
  `for_cpu()`, the `runner_cpu` client built at startup and the `cpu_service`
  field on the `/health` runner snapshot had gone with the rung. **None of them
  did.** All six are still in the working tree, and `TTS_RUNNER_CPU_SERVICE`,
  `TTS_RUNNER_CPU_MAX_WAIT` and `TTS_REALTIME_FACTOR_RUNNER_CPU` are still
  parsed. `TTS_RUNNER_CPU_MIN_PCT` and `TTS_RUNNER_CPU_WHEN_BACKLOG_S` are the
  only two environment variables that genuinely went.

  What that costs today is small and it is not nothing: a second `RunnerClient`
  is constructed at startup and never given work, `_SEEDS` carries a rate for a
  lane that cannot be chosen, and `/health` publishes
  `runner.cpu_service: "chatterbox-cpu"` — **a service id the agent on spring
  has never registered, advertised to the page by the deployment, for a rung
  this document says is gone.** That is the same failure this ADR was written
  about, one level in: a knob nothing consumes reads as a supported feature and
  survives every grep.

  It is left recorded rather than quietly amended because an ADR that describes
  a deletion nobody performed is worse than no ADR. The deletion is still the
  right change and it belongs in `app/remote.py` and `app/main.py`.
  `compose.yaml` sets none of the three and says so at the two-lanes block. A
  seed nothing consumes is a number that reads as a measurement.
* The default `TTS_BACKEND_ORDER` is `runner,local`.
* `estimated_seconds` is still frozen at enqueue from the local rate. That is
  the safe direction with two lanes: the only lane a job can be routed to is
  three times faster than the estimate, so a routed job finishes early against
  the number its caller was given, never late.
* **Spring may be switched off, unplugged, asleep, or wrong about itself, and
  orko still speaks everything.** That is the requirement the ladder quietly
  broke by asking a sleeping machine a question on a job's own thread, and it is
  what the probe thread, the cooldown and the yield-to-local path exist to keep
  true.

## What is not revisited

ADR 0006's reason for reading `/v1/services` rather than the machine-wide
boolean still holds and is still what `offer()` does: `available` is resolved by
the runner against the right gate for the service being asked about, including a
free-memory check that is invisible from here. What changes is only which
services this side ever asks about — one, the card.
