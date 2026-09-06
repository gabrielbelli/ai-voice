# ADR 0006 — Long-form speech is routed between three machines, on the cap the runner publishes rather than on a boolean

**Status:** accepted
**Date:** 2026-09-06

## Decision

`services/tts-long` offers a job to three backends in a configured order, and
`_backend_for` is still the only place that choice is made.

| key | what it is | seed | measured on |
|---|---|---|---|
| `runner` | the runner's card | 0.70x | an RTX 3070, midpoint of a measured 0.644 to 0.746 |
| `local` | this container | 0.23x | the NAS, Xeon E5-2697 v4, 8 threads |
| `runner_cpu` | the runner's processor | 0.24x | the same desktop, Ryzen 7 5700X3D, 8 threads |

| | |
|---|---|
| order | `TTS_BACKEND_ORDER`, default `runner,local,runner_cpu` |
| the second rung's address | none: `TTS_RUNNER_CPU_SERVICE`, a service id on the same host, port, pin and key |
| the gate | `TTS_RUNNER_CPU_MIN_PCT`, default 50, against the cap the runner publishes |
| the promotion | `TTS_RUNNER_CPU_WHEN_BACKLOG_S`, default 120 |
| unset | every one of these has a default that reproduces today's behaviour, and no `TTS_RUNNER_HOST` still means local only |

## The expectation this overturns

The reason for measuring at all was that the runner's processor looked like it
should win. Chatterbox is autoregressive: its transformer emits speech tokens
one at a time at batch size one, and a thread sweep on the NAS barely moved with
more threads (0.077x at 2 threads, 0.230x at 8, 0.285x at 16, so per-thread
efficiency halves). That is the signature of work bound by single-thread latency
rather than by throughput, which is the shape a 2016 Broadwell-EP server part is
worst at and a 2022 Zen 3 desktop part with a large cache is best at. Two or
three times faster was the hypothesis.

Measured through the shipped path on the runner: **0.24x against the NAS's
0.23x**. About five per cent.

So the runner's processor is a fallback for when its card is busy, not a faster
machine, and the whole design follows from that being written down rather than
rounded up.

## Why the cap and not the boolean

The runner sells its card and its processor separately, and prices them per
machine state. `available: true` while it is giving a hundred per cent of a
sixteen-thread machine and `available: true` while it is giving five per cent
are the same field and a twenty to one difference in delivered speech.

A router that reads only the boolean sends a ten-minute job to a machine that
will take over an hour over it, on somebody's desktop, while they are sitting at
it. Nothing reports a fault: the job is running, the runner is behaving
correctly, and the only symptom is a number nobody is watching.

So `offer()` carries `cpu_pct` out of `/v1/services` and `_cpu_rung_worth_it`
compares an expected delivered rate against this host's. The derate is
deliberately pessimistic (measured rate times the cap, when the real curve is
much flatter), because being wrong in that direction costs a job that could have
gone there and did not, and being wrong in the other costs somebody hours.

**An unknown cap is a refusal.** It is the one place in this service that fails
closed on a missing field. Everywhere else a runner that does not publish
something gets the benefit of the doubt, because the cost of being wrong is a
fallback to a CPU that works.

## Why the rung sits below an always-willing floor

`local` is always willing, so anything ordered after it is unreachable. That is
the property that makes the floor a floor, and it is why the shipped order puts
the runner's processor last: at 0.24x against 0.23x there is no reason to cross
a network for it on a quiet stack.

A queue changes the arithmetic, because 0.24x starting now beats 0.23x starting
four minutes from now. That promotion is also the only escape from a
self-fulfilling refusal: a rung that is never chosen is never measured, so its
seed would be permanent. `/health` publishes `backend_observations` so a count of
zero says out loud that the figure beside it is a hypothesis.

## Consequences

* `_worker`'s fallback is a ladder rather than a hard-coded bottom rung. A job
  can go card, then processor, then here; `fell_back_from` records the chain and
  `segments_from_runner` accumulates instead of being assigned.
* `SIDECAR_KEYS` gains `fell_back_from`, so the chain survives a restart.
* `/health` gains `backend_observations` and `backend_order`.
* The runner snapshot now reads `/v1/services` as well as `/v1/status`, because
  `device` and `available` have never been published on the latter. The status
  panel was reading two nulls and no test could see it.
* `estimated_seconds` is still frozen at enqueue from the local rate. That stays
  the safe direction: the card is three times faster and the processor rung is
  only taken when it will finish sooner than here, so a routed job finishes
  early against the number its caller was given, never late.
* The page cannot yet draw a third backend. `ranOn()` returns an empty string
  for `runner_cpu`, and `paintRunner()` has "GPU runner" baked into it. Both are
  in `services/ui`, which another change owns; the items are listed in the
  commit that introduced this.
