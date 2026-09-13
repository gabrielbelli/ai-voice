"""Long-form text-to-speech. Chatterbox on CPU, as a job queue.

Kokoro (tts-stack) answers requests. This does not: at roughly 0.21x realtime
a ten-minute recording takes three quarters of an hour, so an HTTP request
that waits for the audio would time out long before it arrived.

So /jobs accepts work and returns immediately with an id. ONE JOB AT A TIME
PER MACHINE, and the reason is not the one this docstring used to give. It said
the model is 6.5 GB and concurrency would double the memory; the real
constraint is that Synth._speak holds one threading.Lock across _ensure_loaded
and generate(), so two local jobs do not run side by side at all — they
interleave at segment granularity, for no extra throughput and double the
latency of each. A second machine is therefore the only thing that can make two
jobs overlap, and app/dispatch.py is how: one lane here, one lane on the
runner, chosen per job rather than walked as a ladder.

/v1/audio/speech sits on top of the same queue for OpenAI clients, and it now
answers in all three of the ways that endpoint can honestly be answered here:

  * `stream_format: "sse"` streams the audio as it is generated. This is the
    route worth using for anything long. The first `speech.audio.delta` leaves
    when the FIRST SENTENCE finishes rather than when the whole input does, so
    a caller hears sound in tens of seconds instead of waiting out a silence
    measured in minutes. It does not make the service faster — the compute is
    unchanged — it removes the dead air and the client-side timeout.
  * input short enough that the arithmetic allows it is waited on and returned
    as one buffered body, which is what an unmodified OpenAI client needs.
  * anything longer gets 202 and a Location header pointing at the native job.
    That is a deliberate deviation from OpenAI's contract, documented in the
    README with the measurement that forces it.

Every input is split into sentence-sized chunks before it reaches the model,
and that is a bug fix rather than a streaming detail: generate() stops after
1000 speech tokens, which is 40 seconds of audio, and 1690 characters measured
on the deployed instance came back as exactly 40.0 seconds with no error. See
app/chunking.py.

The native routes stay the ones to prefer for batch work — realtime_factor,
per-segment pauses and the queue position have no field in the OpenAI shape
and are dropped there.

Authentication, the health contract, the error envelope, `Segment` and the
logging setup are voice_common's. app/envelope.py is gone: it was this
service's private copy of the envelope, written while voice-common was pinned
by tarball SHA, and every line of it now lives in voice_common.errors — it was
the best of the three vendored copies and the one the shared code was built
from, so this service's wire output did not change. app/encoders.py is the one
encoder both the buffered and the streamed paths use. What is left in this file
is the queue, which is the whole reason this service exists separately from
tts-stack.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import platform
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.concurrency import run_in_threadpool
from voice_common import auth, logging as voice_logging
from voice_common.errors import error_response, install_errors
from voice_common.health import install_health
from voice_common.models import OpenAISpeechRequest, Segment

from voice_common.engines import CATALOGUE, CONTROL_RANGES, WIRE_CONTROLS, slug

from . import voices as voice_registry
from .chunking import chunk_text, speech_seconds
from .engines import (DEFAULT_ENGINE, ENGINES, LOCAL_ENGINES,
                      LOCAL_RESIDENT_MAX, PRESET_VOICE_ENGINE, Refusal,
                      defaults_for, engine_rows, refuse, refuse_unavailable,
                      spec_for, warn_voice_collisions)


def _control_field(name: str):
    """One Pydantic field for a wire control, bounded by the shared table.

    THE BOUNDS ARE NOT WRITTEN HERE AND THAT IS THE POINT. They were:
    `Field(ge=0.0, le=1.0)` in two request models, against a bare `float(raw)`
    in the config loader, so a deployment could set a value at boot that every
    request was then refused for. voice_common.engines.CONTROL_RANGES is read
    by both, so the wire and compose.yaml cannot disagree about what is legal.

    None IS THE ONLY VALUE THAT CAN MEAN "THE CALLER DID NOT SAY", so every one
    of these defaults to it and the engine's own default is resolved after the
    engine is known.
    """
    kind, low, high = CONTROL_RANGES[name]
    return (kind | None, Field(default=None, ge=low, le=high))


def _controls_model(name: str):
    """A base model carrying EVERY wire control, generated from the table.

    Both request bodies get all five. A control the chosen engine cannot honour
    is refused BY NAME with the model's own reason -- which is the house rule
    -- and that refusal cannot happen if Pydantic has already answered 422
    "extra fields not permitted" for a field this service does carry. The two
    answers are different: 422 says "this API has no such field", 400 says
    "this ENGINE has no such dial, here is why, and here is the one that does".
    """
    return create_model(name, **{f: _control_field(f) for f in WIRE_CONTROLS})
from .dispatch import FINISHED, YIELDED, Dispatcher, LaneProbe
from .encoders import MEDIA_TYPES, available_formats, encode, make_encoder
from .remote import (RemoteSynth, RemoteUnavailable, RemoteYield, RunnerClient,
                     RunnerConfig)
from .synth import SAMPLE_RATE, Synth, speech_tokens

OUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/output"))
THREADS = int(os.getenv("TTS_THREADS", "8"))
IDLE_TIMEOUT = float(os.getenv("TTS_IDLE_TIMEOUT", "600"))
# TTS_LANGUAGE, TTS_EXAGGERATION, TTS_CFG_WEIGHT AND TTS_TEMPERATURE ARE READ
# IN app/engines.py NOW, and the move is the point rather than tidiness. A
# default is only meaningful once the engine is known: an engine with no
# exaggeration control must get no exaggeration default, or compose.yaml's 0.3
# arrives on every request to it, written by a config file rather than by a
# person. Module-level constants here could not know which engine was coming.
# Stock defaults (0.5 / 0.5 / 0.8) read as animated and over-cheerful, which is
# the wrong register for an explanation; engines.py keeps the calmer ones.

# OpenAI's schema caps `input` at 4096 characters. It was not enforced: a
# 5000-character body was accepted and queued.
MAX_INPUT_CHARS = 4096

# The rate this host is achieving, seeded from the measurement and then
# corrected by every job that finishes. Seeded rather than hardcoded because
# the same code measured 0.217x on the deployed instance and 0.138x on a
# loaded laptop, and an estimate built on the wrong one of those is wrong by
# 60% — which is how a request UNDER the documented synchronous threshold used
# to turn into a 202 with no explanation.
RTF_SEED = float(os.getenv("TTS_REALTIME_FACTOR", "0.21"))

# ONE SEED PER PLACE THE WORK CAN GO, and the third one is why they exist.
#
# Every backend used to be seeded from RTF_SEED, the local CPU constant. For a
# GPU that is pessimistic and self-corrects on the first finished job. For the
# RUNNER'S PROCESSOR it is a self-fulfilling refusal: the seed is
# indistinguishable from local's, so a chooser comparing rates never picks it,
# so it never finishes a job, so it is never measured, so the seed becomes
# permanent. `backend_observations` in /health is the other half of the fix --
# a count of zero says out loud that the figure beside it is a hypothesis.
#
# The numbers are measurements, and each names the machine it came from:
#   local       0.23x  the NAS, Xeon E5-2697 v4, Chatterbox at 8 threads
#   runner      0.70x  a desktop RTX 3070, midpoint of a measured 0.644-0.746
#   runner_cpu  0.24x  the same desktop's Ryzen 7 5700X3D, 8 threads, uncapped
# The third is the unflattering one and it is not a typo: Chatterbox is
# autoregressive at batch one, so it is bound by single-thread latency and a
# 2022 desktop part beats a 2016 server part by about five per cent rather than
# by the two or three times the core counts suggest. The runner's processor is
# a fallback for when its card is busy, not a faster machine.
RTF_SEED_LOCAL = float(os.getenv("TTS_REALTIME_FACTOR_LOCAL") or RTF_SEED)
RTF_SEED_RUNNER = float(os.getenv("TTS_REALTIME_FACTOR_RUNNER") or 0.70)
RTF_SEED_RUNNER_CPU = float(os.getenv("TTS_REALTIME_FACTOR_RUNNER_CPU") or 0.24)

# WHICH LANES EXIST. It was an ORDER -- a ladder walked top to bottom -- and a
# ladder is why `runner_cpu` never ran a single job: `local` is a rung, it is
# always willing, and anything below an always-willing rung is unreachable.
# Lanes have no order, so this is now only a membership test: naming a lane
# turns it on, leaving it out turns it off. "local" alone is local-only.
#
# `runner_cpu` IS NO LONGER A LANE and naming it does nothing. The server sets
# TTS_RUNNER_CPU_SERVICE=chatterbox-cpu and the runner registers `echo` and
# `chatterbox` and nothing else, so that rung has never been offered a job and
# could not have been; at its measured 0.24x against this host's 0.23x the
# arithmetic in dispatch.py refuses it anyway, without a special case.
BACKEND_ORDER = tuple(x.strip() for x in
                      (os.getenv("TTS_BACKEND_ORDER") or "runner,local").split(",")
                      if x.strip())

# HOW MUCH BETTER A REMOTE LANE HAS TO BE BEFORE A JOB CROSSES THE NETWORK.
# Asymmetric on purpose: losing the runner mid-job costs a whole re-speak,
# because _run re-enters with a fresh encoder and a fresh offsets list, while
# losing the local lane costs only slowness. Worked at the measured rates, 300
# seconds of speech is 1304 s here and 437 s there, and 437 x 1.25 = 546 is
# still comfortably under 1304 -- the margin refuses the close calls, not the
# obvious ones.
DISPATCH_MARGIN = float(os.getenv("TTS_DISPATCH_MARGIN") or 1.25)
# The fixed cost of putting a job on another machine: the reference clip, the
# lease, the first poll. ADDITIVE, because a multiplicative margin hides a
# fixed cost instead of charging for it.
RUNNER_HOP_S = float(os.getenv("TTS_RUNNER_HOP_S") or 8.0)
# How often the runner is asked whether it is free, on a thread nobody is
# waiting on. See dispatch.LaneProbe: the chooser reads the answer and never
# makes the call, which is what stops a wedged runner from charging its connect
# timeout to a job that was always going to run here.
RUNNER_PROBE_S = float(os.getenv("TTS_RUNNER_PROBE_S") or 10.0)
# How long a lane is left alone after it hands a job back. A yield means the
# runner's owner came back; bouncing the next job off the same machine one
# second later is how a queue turns into a stampede.
RUNNER_COOLDOWN_S = float(os.getenv("TTS_RUNNER_COOLDOWN_S") or 30.0)
# HOW LONG AN ACCEPTED JOB MAY SIT WITH NO LANE ABLE TO RUN IT BEFORE IT IS
# FAILED. The mid-run half of the spring-off rule: a runner-only engine is
# refused at SUBMIT with a 503 while the runner is away, but the runner can
# also go away AFTER a job is accepted, and that job had no clock at all --
# `_pick` returned None and looped, `_sweep` skips anything unfinished, so it
# sat `queued` until the process died. Fifteen minutes is long enough that a
# reboot or a game finishing is survived and short enough that nobody watches a
# dead progress bar all evening. 0 is not offered: "wait for ever" is the state
# this exists to remove.
RUNNER_ONLY_DEADLINE_S = max(
    1.0, float(os.getenv("TTS_RUNNER_ONLY_DEADLINE_SECONDS") or 900.0))
# STREAMED JOBS STAY ON THIS HOST. A RemoteYield with `delivered > 0` on a
# stream is re-raised and fails the job, because a second run would send a
# second file header into the middle of somebody's file. Refusing to reach that
# state is cheaper and more honest than resumable streaming, which nothing has
# asked for. 1 turns it on for anybody who wants to find out.
STREAM_ON_RUNNER = (os.getenv("TTS_STREAM_ON_RUNNER") or "0") not in {"0", "false", "no"}
# How long shutdown waits for the lanes before giving up on them. What is still
# running past this is marked `cancelled`, never `failed`.
SHUTDOWN_GRACE_S = float(os.getenv("TTS_SHUTDOWN_GRACE_S") or 20.0)
# What a cold start costs the caller who triggers it: 6.5 GB off disk, or a
# ~3 GB download on a truly cold image. Counted against the synchronous
# budget rather than ignored, because it lands inside whatever the caller is
# waiting on.
COLD_LOAD_SECONDS = float(os.getenv("TTS_COLD_LOAD_SECONDS", "60"))

# How much text /v1/audio/speech will block for, as a hard ceiling. The
# arithmetic that decides whether a given request under this ceiling can
# actually be finished in time is _sync_budget(): fifteen characters is a
# second of speech (measured — see app/chunking.py) and a second of speech
# costs 1/rtf seconds of compute, so 300 characters is around 95 s at 0.21x.
# Set to 0 to always answer 202.
SYNC_MAX_CHARS = int(os.getenv("TTS_OPENAI_SYNC_MAX_CHARS", "300"))
# When it runs out the job is not cancelled — the caller gets 202 and can
# collect the audio from /jobs.
SYNC_TIMEOUT = float(os.getenv("TTS_OPENAI_SYNC_TIMEOUT", "180"))

# Queue depth past which /v1 answers 429 and /jobs answers 429. It is the only
# error response OpenAI's schema declares for this path, and there was no
# backpressure of any kind here: an unbounded number of multi-minute jobs
# could be queued, and the queue is the memory and the disk of one process.
MAX_QUEUE = int(os.getenv("TTS_MAX_QUEUE", "32"))
# TWO SWEEPS, BECAUSE A RECORD AND A FILE ARE NOT THE SAME THING and one TTL
# destroyed both. The audio is megabytes and is worth reclaiming daily; the
# record is a few hundred bytes and is the only evidence the job ever happened.
# The old single TTS_JOB_TTL took the record with the audio at twenty-four
# hours, so the history somebody now wants to filter was being deleted every
# day -- splitting these is a precondition for the Jobs tab, not a nicety.
#
# TTS_JOB_TTL IS STILL HONOURED as the audio TTL. It is set in compose files
# and in people's shells, and silently ignoring a variable somebody set is how
# a disk fills up quietly.
AUDIO_TTL = float(os.getenv("TTS_AUDIO_TTL") or os.getenv("TTS_JOB_TTL") or 86400)
# Thirty days. The record outlives its audio by a month and then goes too,
# because "keep everything for ever" is the growth the sweeper exists to stop.
RECORD_TTL = float(os.getenv("TTS_RECORD_TTL") or 2592000)
# Both 0 disable, exactly as TTS_JOB_TTL=0 always has.

# WHAT THIS MACHINE IS CALLED, on every record it writes. Required on every
# kind, including the ones that have only ever run in one place: it is what
# makes a second machine a data change rather than a change to the page.
HOST_LABEL = (os.getenv("AIV_HOST_LABEL") or platform.node() or "unknown").strip()

# WHETHER OTHER SERVICES MAY POST A FINISHED RUN HERE. tts-long is the only
# service in the stack that keeps a record of anything, so Kokoro and Parakeet
# post theirs to /runs. 0 answers 404 and the senders drop the record, which is
# what they are built to do -- a log must never be able to stall the thing it
# logs.
RUNLOG_ACCEPT = (os.getenv("TTS_RUNLOG_ACCEPT") or "1") not in {"0", "false", "no"}
# The ceiling on stored records. Counted against a cached count refreshed on a
# timer, never an os.scandir per write: a scandir per record is O(n) per write
# and quadratic over a batch, and `runs/` exists as its own directory precisely
# to keep this off _recover's audio walk.
RUNLOG_MAX_RECORDS = int(os.getenv("TTS_RUNLOG_MAX_RECORDS") or 5000)
RUNLOG_COUNT_TTL = float(os.getenv("TTS_RUNLOG_COUNT_TTL") or 60.0)
# Accepted records per minute, per service. A sender in a loop is a full disk.
RUNLOG_RATE = int(os.getenv("TTS_RUNLOG_RATE") or 120)
# Whether what was said is kept. 0 stores the LENGTH and nothing else, for
# anyone who would rather this service did not hold a transcript of every voice
# note they have ever dictated.
RUNLOG_TEXT = (os.getenv("TTS_RUNLOG_TEXT") or "1") not in {"0", "false", "no"}

# Seconds between `:` comment lines on an idle SSE stream. Comments are
# ignored by openai-python's SSEDecoder (verified) and by every other SSE
# client; they exist so a stream that is waiting — for the model to load, for
# a job ahead of it in the queue, or for a long sentence — does not look dead
# to a proxy or trip a read timeout.
SSE_KEEPALIVE = float(os.getenv("TTS_SSE_KEEPALIVE", "10"))
# Whether hanging up on a stream cancels the work behind it. Off, because the
# page promises the opposite in as many words. See the comment where it is read.
CANCEL_ON_DISCONNECT = os.getenv("TTS_SSE_CANCEL_ON_DISCONNECT", "0") not in {"0", "false", "no"}

# The same one line of configuration this always had, plus the TTS_LOG_LEVEL
# switch it never had: getting DEBUG out of a running container used to mean
# editing the source and rebuilding a 6.5 GB image, which is exactly the moment
# that is impossible. Unset still means INFO, so nothing changes for anyone who
# has not asked for it.
log = voice_logging.setup("tts-long", "TTS")

jobs: dict[str, dict] = {}
state: dict[str, object] = {}
# Built at import and STARTED in lifespan, so a test can reach into the lanes
# before a thread has ever run. See app/dispatch.py for the whole argument;
# what matters here is that the `queue.Queue` and the single `_worker` this
# replaces could only ever run one job anywhere, so a job on the runner's card
# held this host's idle CPU for the whole of its ten minutes.
dispatch: Dispatcher
# Only the OpenAI route registers here, and only while it is waiting on a job.
# Kept out of the job dict so nothing unserialisable can reach /jobs.
#
# An asyncio.Event rather than a threading one, paired with the loop that owns
# it: the waiter is a coroutine on the event loop and the setter is the worker
# thread, and asyncio primitives are not thread-safe to touch from outside.
events: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}

# A module-level singleton, and the CMD is a bare `uvicorn app.main:app` with
# no --workers, so there is exactly ONE process holding exactly one of these.
# That is what makes Registry.refresh worth having: a clip written into the
# shared volume by services/ui becomes resolvable on the next request rather
# than on the next restart of a container that carries 6.5 GB of Chatterbox.
VOICES = voice_registry.load_registry()
# A CLIP THAT SHARES A NAME WITH A PRESET VOICE, SAID OUT LOUD ONCE AND NEVER
# REFUSED. Both stay reachable, because a voice here is a PAIR: `pt_male` on a
# preset engine is the checkpoint's embedding and `pt_male` on a cloning engine
# is the file. A service that would not start over a filename would be worse
# than the collision it was refusing.
warn_voice_collisions(VOICES.clips, log)
FORMATS = available_formats()


@dataclass
class Stream:
    """The pipe from the worker thread to one SSE response.

    Same reasoning as `events` above: the producer is the worker thread and
    the consumer is a coroutine, so every hand-off goes through
    call_soon_threadsafe onto the loop that owns the queue.
    """

    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def put(self, kind: str, payload: object) -> None:
        with suppress(RuntimeError):
            # The loop is gone: the client disconnected and the response was
            # torn down. The job carries on to disk, where /jobs can collect
            # it, and nothing here needs to hear about it.
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, payload))


class _Rate:
    """The realtime factor this host is actually achieving.

    An exponential moving average over finished jobs, because the alternative
    — a constant — was wrong by 60% on the machine the audit ran on, and every
    estimate and every synchronous/202 decision is built on it. Weighted
    towards recent jobs so a host that gets busier is noticed within a couple
    of jobs rather than averaged away.
    """

    def __init__(self, seed: float) -> None:
        self.value = seed
        # HOW MANY JOBS ARE BEHIND THE NUMBER. Zero means the value is the seed
        # and nothing has ever measured this backend, which is the question a
        # router and a person reading /health both have and neither could ask.
        # It is published as `backend_observations` beside the rates.
        self.count = 0
        self._lock = threading.Lock()

    def observe(self, audio_seconds: float, compute_seconds: float) -> None:
        if audio_seconds <= 0 or compute_seconds <= 0:
            return
        with self._lock:
            self.value += 0.3 * (audio_seconds / compute_seconds - self.value)
            self.count += 1


# WHICH LANE-AND-ENGINE PAIR A SEED BELONGS TO.
#
# The lane-wide keys are an operator's measurement of ONE engine on ONE lane
# and they predate there being two, so they apply to the DEFAULT engine and
# nothing else. Handing TTS_REALTIME_FACTOR_RUNNER=0.70 -- a measurement of the
# multilingual model on a 3070 -- to turbo would understate it by 2.36x, and an
# understated rate is not a harmless one: it is the number that decides whether
# a caller is answered synchronously or handed a 202.
_LANE_ENV = {"local": ("TTS_REALTIME_FACTOR_LOCAL", "TTS_REALTIME_FACTOR"),
             "runner": ("TTS_REALTIME_FACTOR_RUNNER",),
             "runner_cpu": ("TTS_REALTIME_FACTOR_RUNNER_CPU",)}
_LANE_SEEDS = {"local": RTF_SEED_LOCAL,
               "runner": RTF_SEED_RUNNER,
               "runner_cpu": RTF_SEED_RUNNER_CPU}


def _seed_for(lane: str, engine: str) -> float:
    """The documented seed for one (lane, engine) pair.

    TTS_REALTIME_FACTOR_<LANE>_<SLUG> is the specific key and wins outright.
    Below it the catalogue's measurement for that engine on that kind of
    machine, and below that -- for the default engine only -- the lane-wide key
    an operator may already have set.
    """
    pair = os.getenv(f"TTS_REALTIME_FACTOR_{lane.upper()}_{slug(engine)}")
    if pair:
        return float(pair)
    facts = CATALOGUE.get(engine)
    if engine == DEFAULT_ENGINE:
        for key in _LANE_ENV.get(lane, ()):
            raw = os.getenv(key)
            if raw:
                return float(raw)
    if facts is not None:
        if lane == "local":
            # None MEANS NO CPU FIGURE HAS EVER BEEN MEASURED ON ANY MACHINE IN
            # THIS STACK, which is the honest value and not a pessimistic one.
            # Returning 0.0 makes `_compute_seconds` divide by the 1e-3 floor
            # and hand back a finite number, which is a promise; returning
            # nothing at all is what makes the synchronous branch structurally
            # unreachable rather than arithmetically lucky. See `_seeded_lanes`
            # -- no `local/<engine>` row is published for such an engine, so
            # /health cannot print a figure nobody measured.
            return facts.local_seed if facts.local_seed is not None else 0.0
        return facts.runner_seed
    return _LANE_SEEDS.get(lane, RTF_SEED)


def _has_local_rate(engine: str | None) -> bool:
    """Has anybody ever measured this engine on a processor. See `_seed_for`."""
    facts = CATALOGUE.get(engine or DEFAULT_ENGINE)
    if facts is None:
        return True
    if engine == DEFAULT_ENGINE and any(os.getenv(k)
                                        for k in _LANE_ENV["local"]):
        return True
    return (facts.local_seed is not None
            or bool(os.getenv(
                f"TTS_REALTIME_FACTOR_LOCAL_{slug(engine or DEFAULT_ENGINE)}")))


def _pair(backend: str, engine: str | None) -> str:
    """The key one EMA lives under. One string, so nothing has to hold a tuple."""
    return f"{backend}/{engine or DEFAULT_ENGINE}"


rate = _Rate(_seed_for("local", DEFAULT_ENGINE))

# ONE EMA PER BACKEND, AND THE SEPARATION IS NOT TIDINESS.
#
# `rate` drives _estimate, _compute_seconds, _pending_work, _retry_after and
# _sync_budget: it is the number that decides whether a request is answered
# synchronously or handed a 202. A GPU runner at 20x realtime and this CPU at
# 0.275x sharing one average means _sync_budget accepts a synchronous request the
# CPU can never finish, and it accepts it at exactly the worst moment - the
# instant somebody sits down at the gaming machine and the GPU goes away. Every
# estimate this host makes must be an estimate about the host that will actually
# do the work.
#
# _rates["local"] IS `rate`, the same object, so nothing that reads `rate`
# changes and test_chatterbox_is_the_slower_talker_and_the_constant_says_so
# still pins what it always pinned.
# KEYED BY PAIR, NOT BY LANE, and it is the same argument one level down. A
# GPU at 20x must never enter the average that decides whether the CPU can
# answer the next request synchronously -- and neither must a 1.54x renderer
# enter the average of a 0.65x one on the SAME card. The two engines on that
# machine are 2.36x apart; one figure over both describes neither.
#
# _rates["local/<default engine>"] IS `rate`, the same object, so everything
# that reads `rate` is unchanged and rate_for("local") still returns it.
_rates: dict[str, _Rate] = {_pair("local", DEFAULT_ENGINE): rate}
_rates_lock = threading.Lock()

def rate_for(backend: str, engine: str | None = None) -> _Rate:
    """The observed realtime factor of one (lane, engine) pair, on first use.

    `engine` defaults to this deployment's default engine, so every existing
    caller -- and every existing test -- asks the question it always asked and
    gets the object it always got.
    """
    engine = engine or DEFAULT_ENGINE
    key = _pair(backend, engine)
    with _rates_lock:
        r = _rates.get(key)
        if r is None:
            r = _Rate(_seed_for(backend, engine))
            _rates[key] = r
        return r


# EVERY LANE THIS PROCESS HAS, SEEDED AT IMPORT FOR THE DEFAULT ENGINE.
#
# THE DEFECT THIS PREVENTS: a lane disappearing from `realtime_factor_by_backend`
# because the first job that ran there named the other engine. The page draws
# that map, `backend_observations` is what tells it a figure is a seed rather
# than a measurement, and a lane that is simply absent reads as a lane that does
# not exist. Creating them here rather than on first use also means the shipped
# seeds are visible on /health before anything has run.
# ONE ROW PER (LANE, ENGINE) PAIR THE LANE CAN ACTUALLY CARRY, and the second
# half of that condition is what stops /health printing a number nobody has
# ever measured. A lane with no implementation of an engine has no rate for it
# -- not zero, not the other engine's -- so `local/<a runner-only engine>` is
# ABSENT, and absent is the only honest way to say "nobody has run this on a
# processor". `engine_observations` beside each figure says out loud which of
# the rest are seeds rather than measurements.
for _lane in ("local", *BACKEND_ORDER):
    rate_for(_lane)
    for _engine in ENGINES:
        if _lane == "local" and not (ENGINES[_engine].local
                                     and _has_local_rate(_engine)):
            continue
        rate_for(_lane, _engine)


def _compute_seconds(chars: int, engine: str | None = None) -> float:
    """Estimated CPU seconds to speak `chars` characters, at the observed rate.

    THE LOCAL RATE, ALWAYS, and that is not an oversight. This feeds the
    synchronous/202 decision, and a promise made on the strength of somebody
    else's gaming PC is a promise this service cannot keep the moment they sit
    down at it. The engine still matters, because the two Chatterbox rows are
    2.36x apart on the same processor.

    INFINITE WHERE NOTHING HAS EVER BEEN MEASURED ON A PROCESSOR, and that is
    the whole mechanism by which an engine with no local lane can never be
    answered synchronously. The arithmetic already agrees -- 300 characters is
    25 s of speech and 240 s of compute at 0.104x, against a budget of
    180 less the queue less a 63-second load -- but agreeing is not the same as
    being unable to disagree. `inf` is what stops it passing on somebody else's
    faster card, or the day an operator raises TTS_OPENAI_SYNC_TIMEOUT, and it
    does it WITHOUT a branch on an engine name: `test_no_branch_on_an_engine_name`
    stays green because there is no name here to branch on.
    """
    if not _has_local_rate(engine):
        return math.inf
    return speech_seconds(chars) / max(rate_for("local", engine).value, 1e-3)


def _job_chars(job: dict) -> int:
    """How much text a job is, in characters.

    `.get`, because this is now asked on the way IN to a job as well as about
    jobs already accepted: `_order_for` sizes the work before choosing a
    backend for it. A job with no segments is zero characters, which is the
    true answer and not a missing one.
    """
    return sum(len(text) for text, _ in (job.get("segments") or ()))


# ----------------------------------------------------------------- lanes --


def _job_work(job: dict) -> float:
    """Seconds of speech this job is. The unit every lane is compared in."""
    return speech_seconds(_job_chars(job))


def _refused(job: dict, backend: str, why: str) -> None:
    """Record that a lane was offered this job and would not take it.

    ACCUMULATED, not overwritten. A job can be handed back by the runner and
    then run here, and a record that keeps only the last reason cannot say
    whether the card was busy or missing. These reasons are what a person reads
    when they ask why a job that should have gone to a fast machine ran here.
    """
    job["fell_back"] = True
    said = f"{backend}: {why}" if why else backend
    prior = job.get("fell_back_reason")
    job["fell_back_reason"] = f"{prior}; {said}" if prior else said


class _Synths:
    """The local checkpoints this process keeps loaded, and the ceiling on them.

    ONE Synth PER ENGINE AND A CEILING ON HOW MANY ARE RESIDENT AT ONCE. There
    is one lane here and it is one job wide, so two loaded checkpoints buy no
    throughput at all -- they buy the ABSENCE of a reload when work alternates
    between engines, and they cost the resident memory of a second model on a
    shared host. TTS_LOCAL_RESIDENT_MAX is where that trade is made, and 1 is
    the default because a NAS running this beside everything else is the
    deployment this service was built for.

    EVICTION HOLDS THE OUTGOING SYNTH'S OWN LOCK. `Synth._speak` holds that
    lock across `_ensure_loaded` and `generate()`, so dropping a model without
    it would be tearing several gigabytes out from under a thread that is using
    them -- which is the failure `dispatch.drain` exists to prevent at
    shutdown, reached by a different route.
    """

    def __init__(self, max_resident: int) -> None:
        self._max = max(1, max_resident)
        self._synths: dict[str, Synth] = {}
        # Insertion order IS use order: `get` moves a hit to the end, so the
        # first key is always the least recently used.
        self._lock = threading.Lock()

    def get(self, spec) -> Synth:
        with self._lock:
            found = self._synths.pop(spec.id, None)
            if found is not None:
                self._synths[spec.id] = found
                return found
            fresh = Synth(idle_timeout=IDLE_TIMEOUT, threads=THREADS, spec=spec)
            self._synths[spec.id] = fresh
            while len(self._synths) > self._max:
                name, evicted = next(iter(self._synths.items()))
                del self._synths[name]
                self._evict(name, evicted)
            return fresh

    @staticmethod
    def _evict(name: str, synth: Synth) -> None:
        with synth._lock:  # noqa: SLF001 - see the class docstring
            synth._model = None  # noqa: SLF001
            synth.close()
        import gc
        gc.collect()
        log.info("unloaded %s to stay within TTS_LOCAL_RESIDENT_MAX", name)

    def resident(self, engine: str) -> bool:
        with self._lock:
            return (self._synths.get(engine) is not None
                    and self._synths[engine].loaded)

    @property
    def loaded(self) -> bool:
        """Is ANY checkpoint resident. What /health has always published."""
        with self._lock:
            return any(s.loaded for s in self._synths.values())

    def close(self) -> None:
        with self._lock:
            for synth in self._synths.values():
                synth.close()


def _local_synth(engine: str) -> Synth:
    """The loaded checkpoint for one engine, on this host."""
    pool: _Synths = state["synths"]  # type: ignore[assignment]
    return pool.get(ENGINES[engine])


# ONE CLIENT PER SERVICE, KEPT, and rebuilt when the runner itself is replaced.
#
# THE DEFECT THIS PREVENTS IS PAID PER JOB. `RunnerClient` holds the set of
# reference clips it has already uploaded, so a fresh client per job re-asks
# `HEAD /v1/assets/<sha256>` for a clip that machine has held for a week --
# one extra round trip over the LAN on every single job, to learn something
# this process already knew. Keyed on the base client's identity as well as
# the service id, because a test attaching a fake runner mid-process must not
# be shadowed by a client built for the one before it.
_runners: dict[tuple[int, str], object] = {}
_runners_lock = threading.Lock()


def _runner_for(client, service: str):
    """The same machine's client for one service id, built once.

    Through `RunnerClient.for_service`, which copies rather than constructs:
    one TLS context, one pin, and a transport a test has replaced stays
    replaced. See that method.
    """
    if service == client.cfg.service:
        return client
    key = (id(client), service)
    with _runners_lock:
        found = _runners.get(key)
        if found is None:
            found = client.for_service(service)
            # BOUNDED. A long-lived process attaching many runners would grow
            # this for ever; there are at most a few engines and one runner, so
            # anything past that is a test, and a test's clients are cheap to
            # rebuild.
            if len(_runners) > 16:
                _runners.clear()
            _runners[key] = found
        return found


def _backend_for(job: dict, lane: str = "local"):
    """The thing that speaks this job on the lane it was already given.

    IT NO LONGER CHOOSES AND IT NO LONGER ASKS. This used to walk the ladder,
    call `offer()` on each remote rung and decide -- on the worker's own
    thread, with a thirty second socket timeout, once per job per rung. A
    runner that accepts a connection and then says nothing therefore added
    thirty seconds of silence to a job that was always going to run here. The
    asking now happens on dispatch.LaneProbe's thread and the choosing in
    dispatch.Dispatcher; what is left is the mapping from a lane name to an
    object with `speak_segments`, which is all `_run` ever needed.

    WITH NO RUNNER CONFIGURED THIS RETURNS THE LOCAL Synth, always, and no
    remote code runs at all. That is load-bearing rather than a convenience:
    the whole test suite rests on conftest monkeypatching Synth._speak, so a
    chooser that bypassed Synth in the unconfigured case would leave every test
    in this repository quietly exercising nothing.
    """
    engine = job.get("engine") or DEFAULT_ENGINE
    local: Synth = _local_synth(engine)
    # STAMPED BEFORE ANYTHING IS DECIDED, so no path can leave without saying
    # where it ran. There were three ways to reach the local synth and exactly
    # one of them recorded the fact, so an ordinary local job carried no
    # `backend` at all and `job.get("backend", "local")` quietly supplied the
    # answer downstream.
    job["backend"] = "local"
    if lane == "local":
        return local
    client = state.get(lane)
    if client is None:
        # The lane was chosen and the client went away between the choice and
        # here -- a test detaching its fake, or a reconfiguration. The floor is
        # still the floor.
        _refused(job, lane, "no client is configured any more")
        return local
    # THE ENGINE RIDES ON THE JOB AND RESOLVES TO A SERVICE ID HERE. One host,
    # one port, one pin, one key -- and the runner's other speech service. NOT
    # a second lane: the agent starts at most one controller per device group,
    # so believing there are two slots on that card is believing in a
    # concurrency the machine will never give.
    client = _runner_for(client, ENGINES[engine].runner_service)
    job["backend"] = lane
    job["runner_host"] = "%s:%d" % (client.cfg.host, client.cfg.port)
    job["runner_service"] = client.cfg.service
    job["waiting_since"] = None

    def on_wait(waiting: bool) -> None:
        """Say `queued` again while the runner's owner is at their machine.

        The job really is queued: the lease is sitting on the runner and will
        resume. Reporting `running` for a quarter of an hour with nothing
        happening would be a lie a poller cannot see through, and this is the
        state the whole design calls the NORMAL case rather than an error.

        WHEN IT STARTED WAITING, not just that it is. "Waiting" with no clock
        beside it is indistinguishable from "hung", and the row that says
        "waiting for the runner since 14:02" is the one that stops somebody
        restarting a service that is working correctly.
        """
        job["status"] = "queued" if waiting else "running"
        job["waiting_since"] = time.time() if waiting else None

    # The local job id is the idempotency key, bound here because this is the
    # one place that has both the job and the client. See RemoteSynth.__init__:
    # it cannot be a parameter of speak_segments without breaking the local path.
    return RemoteSynth(client, job["id"], on_wait=on_wait,
                       spec=ENGINES[engine])


def _failed_here(job: dict, why: str) -> str:
    """End the job badly, tell whoever is listening, and leave a row behind.

    THE RECORD IS THE INDEX AND A FAILED RUN HAS NOTHING ELSE. It made no
    audio, so `_recover` has no file to adopt; without the record written here
    the only trace of a run that went wrong is a traceback in a log that
    scrolls away, and the row is gone at the next restart. "What did we run and
    where did it go wrong" is the question the run log exists to answer, and it
    was losing exactly the runs worth asking it about.
    """
    job.update(status="failed", error=why, finished_at=time.time())
    stream = job.get("stream")
    if stream is not None:
        stream.put("error", why)
    _write_record(job)
    return FINISHED


def _cannot_restart(job: dict, why: str, delivered: int) -> str:
    """The one handback that cannot be handed back: audio already sent.

    `_run` re-enters with a fresh encoder and a fresh offsets list, so a second
    attempt would send a second file header into the middle of somebody's file.
    Failing it is cheaper and more honest than resumable streaming, which
    nothing has asked for. Unreachable while TTS_STREAM_ON_RUNNER is 0, because
    dispatch pins streamed jobs to the local lane.
    """
    log.warning("%s: %s, and %d segments were already streamed, "
                "so it cannot be restarted", job["id"][:8], why, delivered)
    return _failed_here(job, why)


def _hand_back(job: dict, lane: str, why: str, delivered: int) -> str:
    """Give the job back to the dispatcher and say which lane lost it.

    THE JOB IS NOT FINISHED AND MUST NOT READ AS THOUGH IT WERE. It goes back
    to `queued` here; the dispatcher returns it to the HEAD of the deque,
    records that this lane would not keep it and cools that lane. The local
    lane can raise neither of the two things that reach here, so the walk
    always terminates on it.
    """
    log.info("%s: %s; putting it back at the head of the queue",
             job["id"][:8], why)
    # WHICH LANE IT LEFT, which a single `fell_back` flag could not say.
    prior = job.get("fell_back_from")
    job["fell_back_from"] = f"{prior}, {lane}" if prior else lane
    _refused(job, lane, why)
    # ACCUMULATED, NOT ASSIGNED. Each attempt delivers its own segments
    # before giving up, and a plain assignment would report only the last.
    job["segments_from_runner"] = job.get("segments_from_runner", 0) + delivered
    job["status"] = "queued"
    job["waiting_since"] = None
    return YIELDED


def _execute_on_lane(job: dict, lane: str) -> str:
    """Run one job on one lane, and say whether the lane kept it.

    A YIELD IS NOT A FAILURE. It means the runner's owner came back and the
    lease was handed over, which is the case the whole design calls normal. The
    dispatcher puts the job back at the HEAD of the deque and cools the lane;
    the local lane cannot yield, so the walk terminates.

    NEITHER IS LOSING THE MACHINE, AND THAT IS THE ONE THIS USED TO GET WRONG.
    A runner that is unplugged mid-job, asleep since the probe last answered,
    out of memory, or reachable and producing nothing raises RemoteUnavailable,
    and `_run` wrote it onto the job as `failed` -- a job thrown away because a
    machine this service does not need was not there. It is a fact about the
    LANE, so it is handed back exactly like a yield: the dispatcher cools the
    lane, records the refusal, and the local CPU speaks the job.
    Nobody asked for a second machine and nobody should have to notice it went.

    WHY THE PROBE CANNOT COVER IT. `LaneProbe.ok` accepts an answer up to three
    intervals old -- thirty seconds at the shipped TTS_RUNNER_PROBE_S -- so
    there is a half-minute in which every job is dispatched at a machine that
    is already gone. Being TOLD is the only thing that shuts the lane inside
    that window; waiting for the probe to notice is thirty seconds of jobs sent
    at a sleeping desktop, which is measured: five submitted, five failed.
    """
    synth = None
    try:
        synth = _backend_for(job, lane)
        _run(synth, job)
        return FINISHED
    except RemoteYield as yielded:
        if yielded.delivered and job.get("stream") is not None:
            return _cannot_restart(job, str(yielded), yielded.delivered)
        return _hand_back(job, lane, str(yielded), yielded.delivered)
    except RemoteUnavailable as gone:
        # OFF THE SYNTH, NOT OFF THE EXCEPTION. RemoteUnavailable is raised
        # from six places in the client and from the socket layer under all of
        # them, so it cannot carry a count only the poll loop has; RemoteSynth
        # publishes `delivered` as segments land for exactly this reason.
        # `synth` is None when _backend_for itself could not build a client,
        # which is a lane that lost the job before it started.
        delivered = int(getattr(synth, "delivered", 0) or 0)
        if delivered and job.get("stream") is not None:
            return _cannot_restart(job, str(gone), delivered)
        return _hand_back(job, lane, str(gone), delivered)
    except Exception as exc:  # noqa: BLE001 - surfaced on the job, never raised
        # A JOB THAT CANNOT BE STARTED IS A FAILED JOB, NOT A DEAD SERVICE.
        # _run catches its own failures, but everything around it -- resolving
        # the client, building the RemoteSynth -- used to run outside any
        # handler, so anything raised there left the one worker thread there
        # was and every later job sat `queued` for ever with no error anywhere.
        log.exception("%s: could not be started on the %s lane",
                      job["id"][:8], lane)
        return _failed_here(job, str(exc) or exc.__class__.__name__)


def _woken(job_id: str) -> None:
    """Wake whoever is waiting on this job, however it ended.

    Called by the dispatcher after every terminal outcome and never after a
    yield, because a yielded job has not ended -- waking its caller then would
    hand a 202 to somebody whose audio was about to arrive.
    """
    waiter = events.pop(job_id, None)
    if waiter is not None:
        loop, event = waiter
        with suppress(RuntimeError):
            # The loop shut down while this job ran. Nothing is waiting on the
            # other end any more, and the audio is on disk.
            loop.call_soon_threadsafe(event.set)


def _lane_allows(lane: str, engine: str | None) -> bool:
    """Could this lane ever carry this engine. Not "is it free right now".

    Only the local lane can answer no: a runner lane's own probe already says
    which services that machine registered, per engine, and answers "no runner
    lane is configured" as an ordinary shut lane. This host's answer is
    `TTS_LOCAL_ENGINES`, which is configuration -- and below it the catalogue's
    `local_class is None`, which is not.
    """
    if lane != "local":
        return True
    spec = ENGINES.get(engine or DEFAULT_ENGINE)
    return spec is None or spec.local


def _expire_stranded(job_id: str, waited: float) -> None:
    """A job no lane has been able to run for the deadline. Terminal, and named.

    THE ONE THING THIS MUST NOT DO IS SAY IT FELL BACK. `fell_back` means a
    machine took this job and gave it up; nothing took this one. `ranOn()` on
    the page renders "after spring gave up" off that flag, and printing it
    about a machine that never had the job is a story that did not happen.
    """
    job = jobs.get(job_id)
    if job is None:
        return
    spec = ENGINES.get(job.get("engine") or DEFAULT_ENGINE)
    lane = dispatch.lanes.get("runner")
    probe = lane.probe if lane is not None else None
    why = (probe.why_for(job.get("engine")) if probe is not None
           else "no runner lane is configured")
    minutes = round(waited / 60.0)
    _failed_here(
        job,
        f"{job.get('engine')} needs the runner and no lane has been able to "
        f"run it for {minutes} minute{'s' if minutes != 1 else ''} "
        f"({why}); nothing here can run it. Send "
        f"model='{DEFAULT_ENGINE}', or bring "
        f"'{spec.runner_service if spec else ''}' back up on the runner.")


def _build_dispatch() -> Dispatcher:
    """The lanes this process has, built before any of them runs.

    THE RUNNER LANE ALWAYS EXISTS AND ITS PROBE DECIDES WHETHER IT IS OPEN.
    Building the lane only when `state["runner"]` is set at startup would tie
    the shape of the dispatcher to the order two lines of `lifespan` happen in,
    and would make a runner attached later -- by a test, or by a
    reconfiguration -- invisible for the life of the process. The probe reads
    `state` on every round instead, and answers "no runner is configured" as an
    ordinary shut lane.
    """
    d = Dispatcher(execute=_execute_on_lane, job_of=jobs.get,
                   work_of=_job_work,
                   rate_of=lambda name, engine=None: rate_for(name, engine).value,
                   finished=_woken, log=log, margin=DISPATCH_MARGIN,
                   cooldown=RUNNER_COOLDOWN_S, pin_streams=not STREAM_ON_RUNNER,
                   # WHICH LANE MAY CARRY WHICH ENGINE, off the catalogue. The
                   # local lane has no probe, so it was willing to take
                   # anything -- including an engine whose `local_class` is
                   # None, which would have imported None inside somebody's
                   # job. `TTS_LOCAL_ENGINES` is a deployment's answer to the
                   # same question and this is the one place it is enforced at
                   # dispatch time rather than at boot.
                   lane_allows=_lane_allows,
                   expire=_expire_stranded,
                   stranded_deadline=RUNNER_ONLY_DEADLINE_S)
    d.add_lane("local")
    if "runner" in BACKEND_ORDER:
        d.add_lane("runner", hop=RUNNER_HOP_S,
                   # THE PROBE IS TOLD WHICH SERVICE CARRIES WHICH ENGINE, so
                   # a missing turbo service shuts turbo and nothing else. It
                   # asks the runner about EVERY service in the same request it
                   # was already making -- which is what `chatterbox-cpu` never
                   # had: a detector that existed and that nothing ran.
                   probe=LaneProbe(lambda: state.get("runner"),
                                   RUNNER_PROBE_S, log,
                                   services={e: s.runner_service
                                             for e, s in ENGINES.items()}))
    return d


dispatch = _build_dispatch()



# How far `frames / frame_rate` may sit from the audio's real length before the
# two are called disagreeing. A splice inserts real silence BETWEEN segments and
# the frame count covers only what the model generated, so the audio is always
# the longer of the two by exactly the pauses -- which is why the pauses are
# subtracted before the comparison rather than absorbed into a wide tolerance.
# The margin left is for the frame grid itself: one frame at 12.5 Hz is 80 ms.
FRAME_TOLERANCE_S = float(os.getenv("TTS_FRAME_TOLERANCE_SECONDS") or 0.5)


def _record_frames(job: dict, synth, duration: float) -> None:
    """Copy what the lane reported about the generation, and check it adds up.

    THE DEFECT THIS PREVENTS SHIPS AUDIO THAT PLAYS AT THE WRONG SPEED AND
    REPORTS NOTHING. The wrapper Voxtral runs through resamples 24000 -> 48000
    in post-processing and then writes the result labelled 24000 in three of
    its four writers; a file like that is playable, is exactly half the right
    length, and nothing anywhere raises. `frames / frame_rate` against the real
    duration is arithmetic this service can do on its own evidence, and a
    factor of two is not a rounding error.

    NOT FATAL WHEN THE LANE SAYS NOTHING. Every runner built before these keys
    existed reports neither, which is "this lane did not say" and not "this
    lane disagreed" -- the columns are simply absent from those rows.
    """
    frames, rate = getattr(synth, "frames", None), getattr(synth, "frame_rate", None)
    settings = getattr(synth, "runner_settings", None)
    if settings is not None:
        job["runner_settings"] = settings
    if not frames or not rate:
        return
    job["frames"], job["frame_rate"] = int(frames), float(rate)
    generated = duration - sum(pause for _, pause in (job.get("segments") or ()))
    claimed = int(frames) / float(rate)
    if abs(claimed - generated) > FRAME_TOLERANCE_S:
        raise RuntimeError(
            f"the lane reported {frames} frames at {rate} Hz, which is "
            f"{claimed:.1f}s of speech, and it delivered {generated:.1f}s. "
            f"Audio that has been resampled and then labelled with the old "
            f"rate looks exactly like this and plays at the wrong speed.")


def _run(synth: Synth, job: dict) -> None:
    stream: Stream | None = job.get("stream")
    if job["cancelled"]:
        # Cancelled while it sat in the queue. Nothing was generated, so there
        # is no audio to write and nothing to stream.
        job.update(status="cancelled", finished_at=time.time())
        if stream is not None:
            stream.put("error", "the job was cancelled before it started")
        # THE ROW IS STILL A RUN. This one produced no audio at all, so it is
        # the record or nothing: _recover indexes records and adopts audio, and
        # a job cancelled in the queue has neither unless this line writes one.
        # It used to vanish at the next restart, which reads as a job that was
        # never submitted rather than one somebody stopped.
        _write_record(job)
        return

    started = time.time()
    # HOW LONG IT SAT BEFORE ANYTHING TOUCHED IT. compute_seconds has always
    # measured synthesis and nothing else, which is right, but it left the
    # commonest complaint unanswerable: a job that took ten minutes when the
    # estimate said two spent eight of them queued behind another job, and the
    # record could not say so. created_at and started_at were both there and
    # nobody was subtracting them.
    job.update(status="running", started_at=started,
               queued_seconds=round(max(0.0, started - job["created_at"]), 1))
    encoder = None
    try:
        started = time.monotonic()
        # WHETHER THIS JOB PAID FOR A COLD LOAD, counted rather than inferred.
        # Turbo is 67.5 s off disk against 22.2 s, so "it was slow" and "it was
        # slow because it loaded a model" are different stories and the record
        # could not tell them apart.
        loads_before = int(getattr(synth, "loads", 0) or 0)
        if stream is not None:
            # The streaming encoder is created here rather than in the route so
            # that an ffmpeg process is never started for a request that then
            # waits ten minutes in the queue.
            encoder = make_encoder(job["format"])

        # WHERE EACH SEGMENT STARTS, accumulated as it is made. speak_segments
        # synthesises and splices one segment at a time and hands each piece
        # here, so its length is known at that moment and the running total is
        # an EXACT boundary -- not duration x (chars so far / chars total),
        # which is wrong from the first sentence because the pause after a
        # segment is a fixed number of seconds regardless of its length.
        #
        # These were being computed and discarded. Recording them is what lets
        # a client follow the text as the audio plays.
        offsets: list[float] = []
        samples = 0

        def on_chunk(piece) -> None:  # noqa: ANN001 - numpy array
            nonlocal samples
            offsets.append(round(samples / SAMPLE_RATE, 3))
            samples += piece.size
            # PUBLISHED HERE, NOT WHEN THE JOB ENDS. A clone job is minutes of
            # compute on this CPU, and until this line a poller had `chunks`
            # and no way to learn how many of them existed yet -- so the only
            # progress available was elapsed / estimated_seconds, a guess,
            # while the exact answer sat in a local list until the job
            # finished. A mid-run GET /jobs/{id} now carries a growing list of
            # exact boundaries: len(offsets) segments are spoken and
            # offsets[-1] seconds of audio are made.
            #
            # `list(offsets)`, never the list itself. _public snapshots a job
            # with dict(job), which copies the mapping and NOT the values, so
            # publishing the live object would hand the JSON encoder a list
            # this worker thread is still appending to. The copy is a few
            # floats per segment and it is what makes the snapshot a snapshot.
            job["offsets"] = list(offsets)
            if encoder is None or stream is None:
                return
            data = encoder.write(piece)
            if data:
                stream.put("delta", data)

        # ALWAYS passed now, not only when streaming. It used to be handed over
        # `if stream is not None`, so a job collected from /jobs -- which is
        # most of them, since Chatterbox is minutes of compute -- produced no
        # boundaries at all.
        # ONE MAPPING, BUILT FROM THE COLUMNS. Positional on purpose: `_run` is
        # handed a backend and must not learn which one it got, so the local
        # and remote signatures have to stay byte for byte identical. A control
        # the engine has no value for is absent from the dict rather than None,
        # because `generate(exaggeration=None)` is the same silently-discarded
        # keyword as `generate(exaggeration=0.3)`.
        spoken = synth.speak_segments(
            job["segments"], job["language"],
            {f: job[f] for f in WIRE_CONTROLS if job.get(f) is not None},
            job["reference"],
            on_chunk=on_chunk,
            cancelled=lambda: bool(job["cancelled"]))
        # The final value, and a copy for the same reason. on_chunk has already
        # published every boundary; this line is what covers a job that made no
        # segments at all, and it leaves the published list independent of the
        # local one for good.
        job["offsets"] = list(offsets)
        # WALL CLOCK AND OCCUPANCY ARE TWO NUMBERS AND THEY WERE ONE.
        # `compute_seconds` fed rate_for(...).observe, which is the figure that
        # decides whether the next request can be answered synchronously -- and
        # on the runner it was being charged for every second the runner spent
        # handing its GPU back to its owner. A machine that yields twice reads
        # as a machine that is slow, permanently, in the average.
        #
        # `waited` is what RemoteSynth already accumulated and discarded.
        # Occupancy is what the rate is built from; wall clock is what a person
        # wants when a job looks hung, and `lane_seconds - compute_seconds` is
        # exactly "how long we spent waiting for somebody else's machine".
        elapsed = time.monotonic() - started
        compute = max(0.0, elapsed - float(getattr(synth, "waited", 0.0) or 0.0))
        if int(getattr(synth, "loads", 0) or 0) > loads_before:
            job["load_seconds"] = float(getattr(synth, "load_seconds", 0.0) or 0.0)

        if encoder is not None and stream is not None:
            tail = encoder.close()
            if tail:
                stream.put("delta", tail)
            encoder = None

        # The file on disk is always the buffered encoding, headers and all,
        # even for a streamed request: /jobs/<id>/audio has to hand over a
        # complete file, and a client whose stream dropped half way should
        # still find the whole thing there.
        path = OUT_DIR / f"{job['id']}.{job['format']}"
        data, _ = encode(spoken.audio, job["format"])
        # TEMP THEN RENAME, because _recover TRUSTS THIS FILE. A plain
        # write_bytes interrupted by a restart leaves a truncated file, and
        # _recover walks /output and rebuilds whatever it finds as `status:
        # "done"` with `bytes` set to the truncated length -- a job that reads
        # finished and plays as silence or half a sentence, with nothing
        # anywhere reporting a problem. rename is atomic within a filesystem,
        # so the file is either absent or whole and _recover cannot see a
        # partial one. The runner's own lease already did it this way.
        staging = path.with_suffix(path.suffix + ".part")
        staging.write_bytes(data)
        staging.replace(path)

        duration = spoken.audio.size / SAMPLE_RATE
        # WHAT THE LANE SAID IT DID, AND WHETHER THAT AGREES WITH THE AUDIO.
        # Read off the synth for the reason `waited` is: `_run` is handed a
        # backend and must not learn which one it got, so the two signatures
        # stay identical and anything only a remote lane knows arrives on the
        # object. Absent stays absent -- a lane that reports nothing writes no
        # columns rather than wrong ones.
        _record_frames(job, synth, duration)
        # PER BACKEND. See _rates: a GPU's 20x must never enter the average that
        # decides whether the CPU can answer the next request synchronously.
        rate_for(job.get("backend", "local"), job.get("engine")).observe(
            duration, compute)
        usage = {
            "input_tokens": spoken.input_tokens,
            "output_tokens": speech_tokens(spoken.audio.size),
            "total_tokens": spoken.input_tokens + speech_tokens(spoken.audio.size),
        }
        job.update(status="cancelled" if job["cancelled"] else "done",
                   path=str(path),
                   # HOW BIG THE FILE IS, SET WHERE THE FILE IS WRITTEN. This
                   # was set in _recover and nowhere else, so the `audio`
                   # object published `bytes: 0` for every job this process had
                   # actually run and the true size only after a restart -- a
                   # field that was wrong until the one event that is supposed
                   # to lose nothing. `data` IS what went to disk, so this
                   # cannot disagree with the file the way a second stat could.
                   bytes=len(data),
                   audio_seconds=round(duration, 1),
                   compute_seconds=round(compute, 1),
                   lane_seconds=round(elapsed, 1),
                   realtime_factor=round(duration / compute, 3) if compute else 0.0,
                   usage=usage,
                   waiting_since=None,
                   finished_at=time.time())
        # Beside the audio, so a restart can rebuild this row rather than a
        # filename. See _write_record.
        _write_record(job)
        log.info("%s %s: %.1fs audio in %.0fs (%.2fx)", job["id"][:8],
                 job["status"], duration, compute,
                 duration / compute if compute else 0)
        if stream is not None:
            stream.put("done", usage)
    except RemoteUnavailable:
        # THE LANE WENT AWAY, AND THAT IS NOT THIS JOB GOING WRONG. Unplugged
        # mid-poll, asleep since the probe answered, out of memory, or
        # answering "running" for ever and producing nothing: every one of
        # those arrives here, and every one of them used to be written onto the
        # job by the handler below as `failed`, with somebody else's errno as
        # the reason. The text was fine and this host could speak it. Re-raised
        # for the same reason RemoteYield is: which backend, and what to do
        # about losing one, is _execute_on_lane's business and not _run's.
        #
        # The local Synth cannot raise this either, so with no runner
        # configured this clause is unreachable.
        raise
    except RemoteYield:
        # NOT A FAILURE, so it must not fall into the handler below. The bounded
        # wait for somebody else's GPU ran out; _worker speaks the job on this
        # host instead. Re-raised rather than handled here because "which
        # backend" is _worker's and _backend_for's business, and _run stays a
        # function that is handed one and uses it.
        #
        # The local Synth cannot raise this, so with no runner configured this
        # clause is unreachable and the path below is byte for byte the one that
        # ran before any of this existed.
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced on the job, not raised
        job.update(status="failed", error=str(exc), finished_at=time.time())
        log.exception("%s failed", job["id"][:8])
        if stream is not None:
            stream.put("error", str(exc))
        # THE RUN WORTH KEEPING IS THE ONE THAT WENT WRONG, and it was the one
        # run with nothing on disk: no audio for _recover to adopt and, until
        # this line, no record to index either. See _failed_here.
        _write_record(job)
    finally:
        if encoder is not None:
            # Only reached on the failure path: close() on a half-fed ffmpeg
            # would wait on a process nobody is going to read.
            kill = getattr(encoder, "kill", None)
            if kill is not None:
                kill()


async def _sweeper() -> None:
    """Two sweeps, because a record and a file are not the same thing.

    THE DEFECT THIS PREVENTS IS A HISTORY THAT DELETES ITSELF DAILY. One TTL
    took the record with the audio at twenty-four hours, so "which voice said
    that, how long did it take, which machine ran it" -- a few hundred bytes --
    was thrown away with the megabytes, every day, and the Jobs tab could only
    ever show today.

    The audio goes at TTS_AUDIO_TTL and the row stays, marked `audio_expired`
    so the page can say "expired" rather than "deleted": nobody pressed
    anything, and telling a reader somebody did is a lie. The record goes at
    TTS_RECORD_TTL, a month later. Either at 0 disables that half.
    """
    if AUDIO_TTL <= 0 and RECORD_TTL <= 0:
        return
    tick = min([t for t in (AUDIO_TTL, RECORD_TTL, 300.0) if t > 0])
    while True:
        await asyncio.sleep(tick)
        _sweep(time.time())


def _sweep(now: float) -> tuple[int, int]:
    """One pass. Split out so a test can run it without waiting out a sleep."""
    expired = swept = 0
    for job_id, job in list(jobs.items()):
        finished = job.get("finished_at")
        if finished is None:
            # Still queued or running. A live job has no age yet, and reading
            # `time.time()` as its default -- which this used to do -- made
            # every unfinished job permanently one tick from the cutoff.
            continue
        if RECORD_TTL > 0 and finished < now - RECORD_TTL:
            _discard(job)
            jobs.pop(job_id, None)
            swept += 1
            log.info("%s: record swept after %.0fs", job_id[:8], RECORD_TTL)
            continue
        if (AUDIO_TTL > 0 and finished < now - AUDIO_TTL
                and job.get("path") and not job.get("audio_deleted")):
            _expire_audio(job)
            expired += 1
            log.info("%s: audio expired after %.0fs; the record stays",
                     job_id[:8], AUDIO_TTL)
    return expired, swept


def _expire_audio(job: dict) -> None:
    """The file goes, the row stays, and the row says which of the two happened.

    `audio_expired` rather than `audio_deleted`, and the two are not
    interchangeable. One means somebody pressed a button; the other means a
    clock ran out. A page that reports the second as the first is telling its
    reader they did something they did not do.
    """
    path = job.get("path")
    if path:
        with suppress(OSError):
            Path(path).unlink()
    job["path"] = None
    job["bytes"] = 0
    job["audio_expired"] = True
    _write_record(job)


def _discard(job: dict) -> None:
    """The audio AND the metadata beside it.

    Both, or every swept or deleted job would leave its {id}.json behind: the
    same unbounded growth of /output the sweeper exists to stop, in smaller
    files, and orphans that _recover would then have to reason about.
    """
    path = job.get("path")
    if path:
        with suppress(OSError):
            Path(path).unlink()
    job_id = job.get("id")
    if job_id:
        with suppress(OSError):
            _sidecar(job_id).unlink()


# WHAT SURVIVES A RESTART, AND IT IS NOW THE INDEX RATHER THAN A NOTE BESIDE
# THE AUDIO. One record per run, in OUT_DIR/runs/{id}.json.
#
# THE INVERSION AND WHY IT WAS FORCED. `_recover` used to walk the AUDIO files
# and read a sidecar for each one, so a record with no audio behind it was an
# orphan by definition -- and that rule already needed one exemption, for a job
# whose audio had been deleted on purpose. Instant speech and transcriptions
# keep no audio at all, so records with no file are now the MAJORITY, and a
# rule that needs a second exemption is the wrong rule. The record is the
# index; the audio is an attachment to it.
#
# `reference` is deliberately absent. It is a server-side path to a voice clip,
# _public strips it from every response, and the voice NAME is what identifies
# the job to a reader. `segments` is absent too: _said() flattens it into
# `text`, which is what the listing previews and what GET /jobs/{id} returns.
RECORD_KEYS = (
    # -- identity, and `kind` is the field the whole listing turns on ---------
    # ABSENT MEANS "clone". That default is what makes every sidecar written
    # before this release a valid record with no migration of its contents.
    "kind", "service", "engine", "host", "route", "client",
    # WHY THAT ENGINE, beside which one. "pinned" means the caller typed the
    # name; "default" means they typed nothing. Without it a change of
    # TTS_DEFAULT_ENGINE is invisible in every row written either side of it.
    "engine_reason",
    # WHAT THE COLD LOAD COST THIS RUN, measured rather than assumed. Turbo is
    # 67.5 s off disk against the multilingual model's 22.2 s, and a job that
    # looks slow because it paid for a load is a different story from one that
    # is slow.
    "load_seconds",
    "status", "format", "voice", "language", "chunks", "cancelled",
    # EVERY CONTROL AS ITS OWN COLUMN, spread from the shared table. Additive
    # and migration-free: a record written before this release has none of the
    # new ones and is still a valid record, because `_write_record` drops what
    # is None and `_read_record` filters rather than requires. THE EXACT
    # flow_steps AND cfg_alpha THAT PRODUCED THE SOUND is the whole point --
    # the owner intends to retune by ear, and a tuning log whose rows cannot
    # say which setting made which audio is not a log.
    *WIRE_CONTROLS,
    "speed", "model_requested",
    # WHAT RATE THE AUDIO IS. The wrapper Voxtral runs through resamples
    # 24000 -> 48000 in post-processing and then writes 24000 in three of its
    # four writers, which ships a file that plays at half speed and reports
    # nothing. One number on the row is what makes that arguable after the fact.
    "sample_rate",
    # HOW MANY ACOUSTIC FRAMES, AND AT WHAT GRID. `frames / frame_rate` must
    # equal `audio_seconds`, which is the cheapest assertion in this service
    # that the audio is the length the generator thinks it is -- and, on a
    # 12.5 Hz grid nothing else in this stack has, evidence about WHICH
    # checkpoint produced it.
    "frames", "frame_rate",
    # WHAT THE RUNNER WAS ACTUALLY SERVING when this row was made: the
    # load-time settings it published in its manifest. They are not caller
    # fields and never will be -- group_size is consumed inside a 63-second
    # load, max_frames pre-allocates the KV cache on a card with a 73 MiB
    # margin -- but "what produced this audio" is unanswerable without them.
    "runner_settings",
    "created_at", "started_at", "finished_at", "error",
    "audio_seconds", "speech_seconds", "compute_seconds", "realtime_factor",
    "usage", "offsets", "text", "chars",
    # WALL CLOCK BESIDE OCCUPANCY. compute_seconds is what the rate is built
    # from and excludes time spent waiting for somebody else's machine;
    # lane_seconds includes it, and the difference is the number a person wants
    # when a job looks hung.
    "lane_seconds",
    # Kept so a recovered row can still say what it was promised. It is only
    # read for a live job, and a recovered one is finished by definition, but
    # dropping it would make a restart the one way to lose the number a client
    # was given -- exactly the class of loss this record exists to stop.
    "estimated_seconds",
    # WHERE IT RAN, AND WHAT HAPPENED ON THE WAY. Recovered rows used to come
    # back with no backend at all, so every job in the list read the same
    # whether it had been on a GPU across the LAN or on this CPU, and the one
    # question worth asking of a two-machine setup had no answer after a
    # restart.
    "backend", "runner_host", "runner_service",
    "fell_back", "fell_back_reason", "segments_from_runner",
    # WHICH LANE IT LEFT. `fell_back` says something went wrong somewhere and
    # `backend` says where it ended up; neither can say a job started on the
    # card and finished here.
    "fell_back_from",
    "queued_seconds",
    # WHEN THE WAITING STARTED, not merely that it is happening. "Waiting" with
    # no clock beside it is indistinguishable from "hung".
    "waiting_since",
    # THE TWO WAYS AUDIO GOES, KEPT APART. `audio_deleted` means somebody
    # pressed a button; `audio_expired` means a clock ran out. Reporting the
    # second as the first tells a reader they did something they did not do.
    "audio_deleted", "audio_expired",
)

# WHAT NO SENDER MAY SET, and this list is the whole security boundary of
# POST /runs. `_read_sidecar`'s filter comment applies verbatim and more so:
# that file was in a writable volume, this arrives over HTTP, and accepting
# `path` would let a sender choose the argument to open() on /jobs/{id}/audio.
# `text_preview`, `text_length` and `audio` are derived by _public and a sender
# that supplies them is describing a job it did not run.
SERVER_ONLY_KEYS = frozenset({
    "path", "bytes", "reference", "segments", "stream", "parts", "recovered",
    "text_preview", "text_length", "audio",
})

# What a sender's `text` is truncated to on the way in. TRUNCATED, NEVER
# REFUSED: a transcript that is one character over the limit is still the only
# record that the transcription happened, and 413 for a log line is absurd.
MAX_RECORD_TEXT = 4096


def _sidecar(job_id: str) -> Path:
    """Where one record lives.

    ITS OWN DIRECTORY, and that is a performance decision rather than a tidy
    one. `_recover` walks OUT_DIR for audio, and every instant-speech and
    transcription record would otherwise be a directory entry that walk has to
    look at and reject. Records outnumber audio files by design now.
    """
    return OUT_DIR / "runs" / f"{job_id}.json"


def _write_record(job: dict) -> None:
    """Record a run, for _recover to read back.

    Never fatal. For a clone job the audio is the artefact and it is already on
    disk by the time this runs, so a record that could not be written costs the
    row its voice and nothing else -- which is the state every job was in
    before this existed.
    """
    data = {key: job[key] for key in RECORD_KEYS if job.get(key) is not None}
    said = _said(job)
    if said:
        data["text"] = said
    if not RUNLOG_TEXT:
        # THE LENGTH IS NOT THE TEXT. Somebody who has switched this off does
        # not want a transcript of every voice note on disk; they still want to
        # know the job existed and how big it was.
        data.pop("text", None)
        if said:
            data["chars"] = len(said)
    try:
        # Temp then rename, for the reason the audio write gives: _recover
        # reads this back and json.loads a truncated file raises, which costs
        # the row its voice for no reason other than when the power went.
        target = _sidecar(job["id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(".json.part")
        staging.write_text(json.dumps(data), encoding="utf-8")
        staging.replace(target)
    except OSError as exc:
        log.warning("%s: could not write the run record (%s); a restart "
                    "will recover this job's audio without its voice",
                    job["id"][:8], exc)


def _read_record(job_id: str) -> dict:
    """What _write_record left, or {} when there is nothing usable.

    Silent about a missing file: every job finished before this release has
    audio and no record, and those still recover with what the filename
    carries. A corrupt one is logged instead, because a warning in the log is
    the only sign of it anybody would ever get.

    Filtered through RECORD_KEYS on the way IN as well as out. This file lives
    in a writable volume, and merging it into the record unfiltered would let
    whatever is in it set `path` -- which is the argument to open() on
    /jobs/{id}/audio.
    """
    try:
        data = json.loads(_sidecar(job_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("%s: unreadable run record (%s); recovering the "
                    "audio without it", job_id[:8], exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in RECORD_KEYS}


def _migrate_legacy_records() -> int:
    """Move every OUT_DIR/{id}.json into OUT_DIR/runs/, once.

    Records written before this release sat beside the audio. Leaving them
    where they were would mean two places to look for the same fact for ever,
    and _recover's audio walk would keep tripping over them. Anything that does
    not read back as a record is left exactly where it is rather than moved
    into the index -- a stray .json in the output volume is somebody else's.
    """
    moved = 0
    target_dir = OUT_DIR / "runs"
    for legacy in sorted(OUT_DIR.glob("*.json")):
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Corrupt or unreadable. _read_record's rule stands: half a file is
            # not a fact, and this one cannot even be identified.
            with suppress(OSError):
                legacy.unlink()
            continue
        if not isinstance(data, dict):
            continue
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            legacy.replace(target_dir / legacy.name)
        except OSError as exc:
            log.warning("could not move %s into runs/ (%s)", legacy.name, exc)
            continue
        moved += 1
    return moved


def _recover() -> int:
    """Rebuild every known run from disk. THE RECORD IS THE INDEX.

    `jobs` is a dict in this process, so a restart forgets every run -- while
    the records and the audio sit in a volume and survive. Four things went
    wrong with that before any of this existed, and the fifth is why the walk
    is now over records rather than over audio:

      * a finished job became unreachable. The file was there, named after the
        job, and nothing would serve it.
      * the page remembers job ids in localStorage, so a job the service no
        longer knew about rendered as PENDING for ever -- queued behind
        nothing, waiting for a worker that had already finished it.
      * the sweeper only removes files it has a job for, so every restart
        orphaned another day's audio permanently.
      * what the filename carries is the id and the format, so every recovered
        row read "voice unknown" -- GAB-629.
      * AND NOW: a run may have no audio at all. Instant speech and
        transcriptions never produce a file here, and a clone job whose audio
        was deleted or expired is a row people still want. Walking audio would
        have deleted every one of those as an orphan, quietly, on the next
        restart.

    Five steps, in this order: migrate, index, attach, adopt, sweep.
    """
    moved = _migrate_legacy_records()
    if moved:
        log.info("moved %d record(s) into %s", moved, OUT_DIR / "runs")

    found = 0
    # 1-2. INDEX. Every record is a row, whether or not anything is beside it.
    for record in sorted((OUT_DIR / "runs").glob("*.json")):
        job_id = record.stem
        if job_id in jobs:
            continue
        kept = _read_record(job_id)
        if not kept:
            continue
        job = {"id": job_id, "status": "done", "cancelled": False,
               "path": None, "bytes": 0, "recovered": True}
        job.update(kept)
        if job.get("service") == "tts-long":
            # ACCURATE FOR HISTORY, NOT AN INFERENCE. Every record this service
            # wrote before there were two engines was `chatterbox`, because it
            # was the only engine -- so this is a fact rather than a guess, and
            # a row with no engine at all makes the whole list read the same
            # whether it was one model or the other. Anything beyond restating
            # what was true would be indistinguishable from a measurement a
            # year from now. Only for THIS service's rows: Kokoro and Parakeet
            # post their own and name their own engines.
            job.setdefault("engine", "chatterbox")
        # THE ROW'S OWN IDENTITY IS NOT THE FILE'S TO SET. See
        # test_a_sidecar_cannot_point_the_audio_route_somewhere_else.
        job.update(id=job_id, recovered=True, path=None, bytes=0)
        jobs[job_id] = job
        found += 1

    # 3. ATTACH. The audio is an attachment to a record, and the FILE settles
    # what format it is and how many bytes -- never the record's claim about it.
    have_audio: set[str] = set()
    for path in sorted(OUT_DIR.glob("*.*")):
        job_id, suffix = path.stem, path.suffix.lstrip(".")
        if suffix not in FORMATS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        job = jobs.get(job_id)
        if job is None:
            # 4. ADOPT: audio with no record at all. Every job finished before
            # records existed is this shape, and it is rebuilt from the only
            # thing left -- the filename -- exactly as it always was. Named,
            # not guessed: a reader can tell "one chunk" from "we do not know".
            jobs[job_id] = {
                "id": job_id, "status": "done", "format": suffix,
                "path": str(path), "bytes": stat.st_size,
                "created_at": stat.st_mtime, "started_at": stat.st_mtime,
                "finished_at": stat.st_mtime, "cancelled": False,
                "recovered": True,
            }
            found += 1
            have_audio.add(job_id)
            continue
        if job.get("audio_deleted"):
            # The record says the file was deleted on purpose and a file is
            # here anyway: a re-run, or something outside put it back. The
            # record is what a person said; believe it and leave the file for
            # the sweeper rather than silently un-deleting a row.
            continue
        job.update(path=str(path), bytes=stat.st_size, format=suffix)
        job.pop("audio_expired", None)
        have_audio.add(job_id)

    # A CLONE RECORD WITH NO AUDIO BEHIND IT HAS LOST ITS AUDIO, and saying so
    # is the whole point of the split. Deleted stays deleted -- somebody
    # pressed that button. Anything else is expired: the sweeper took it, or it
    # went from outside, and either way there is nothing to play.
    #
    # ONLY FOR `clone`. An instant-speech or transcription record never had a
    # file here, and telling a reader that a file which never existed has
    # expired is the lie the `never` state exists to prevent.
    for job in jobs.values():
        if job["id"] in have_audio or job.get("audio_deleted"):
            continue
        if job.get("kind", "clone") == "clone" and job["status"] == "done":
            job["audio_expired"] = True

    # 5. SWEEP. A half-written audio or record file from a restart mid-write.
    # _recover ignores the suffix by design, so without this nothing would ever
    # remove them and /output would grow by one on every unlucky restart.
    for partial in list(OUT_DIR.glob("*.part")) + list((OUT_DIR / "runs").glob("*.part")):
        with suppress(OSError):
            partial.unlink()
    return found



@asynccontextmanager
async def lifespan(app: FastAPI):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recovered = _recover()
    if recovered:
        log.info("recovered %d finished job(s) from %s; their audio is "
                 "downloadable again and the sweeper can now expire them",
                 recovered, OUT_DIR)
    # ONE Synth PER ENGINE, WITH A CEILING. See _Synths: the lane is one job
    # wide, so a second resident checkpoint buys no throughput -- it buys the
    # absence of a 68 second reload when work alternates, and it costs the
    # memory of a second model on a shared host.
    state["synths"] = _Synths(LOCAL_RESIDENT_MAX)
    # UNSET MEANS LOCAL ONLY, and that is the default. RunnerConfig.from_env
    # returns None when TTS_RUNNER_HOST is not set, state["runner"] stays None,
    # and _backend_for returns the local Synth without importing anything else.
    runner_cfg = RunnerConfig.from_env()
    state["runner"] = RunnerClient(runner_cfg) if runner_cfg else None
    # THE SAME MACHINE'S OTHER SERVICE, and a second client rather than a second
    # host. It shares the certificate pin, the key and the port; only the
    # service id and the wait differ. None when the runner has no processor rung
    # configured, which is what TTS_RUNNER_CPU_SERVICE="" says.
    cpu_cfg = runner_cfg.for_cpu() if runner_cfg else None
    state["runner_cpu"] = RunnerClient(cpu_cfg) if cpu_cfg else None
    if runner_cfg:
        log.info("a runner is configured at %s:%d; its card is service %s. "
                 "Lanes: %s.", runner_cfg.host, runner_cfg.port,
                 runner_cfg.service, ", ".join(dispatch.lanes))
    # ONE LANE PER PLACE THE WORK CAN GO, each one thread wide. Local's width is
    # 1 by construction and is not a knob: Synth._speak holds one lock across
    # _ensure_loaded and generate(), so two local jobs interleave at segment
    # granularity for no extra throughput and double the latency of each. The
    # docstrings that used to say the constraint was the 6.5 GB model were
    # wrong -- it is that lock, and every concurrency this delivers comes from
    # the second machine.
    dispatch.start()
    sweeper = asyncio.create_task(_sweeper())
    log.info("ready, %d threads, idle timeout %.0fs, formats %s, voices %s, "
             "engines %s (default %s; models load on first job)",
             THREADS, IDLE_TIMEOUT, ", ".join(FORMATS), ", ".join(VOICES.names),
             ", ".join(ENGINES), DEFAULT_ENGINE)
    if VOICES.aliased:
        log.warning("no reference clip for %s: those names answer with the "
                    "built-in voice and every response says so in X-Voice. "
                    "Drop <name>.wav into TTS_VOICE_DIR to give them their own "
                    "voice, or set TTS_VOICE_STRICT=1 to refuse them.",
                    ", ".join(VOICES.aliased))
    if len(FORMATS) < len(MEDIA_TYPES):
        # Including the schema's DEFAULT, which is mp3 — so on a checkout
        # without ffmpeg every request that omits response_format is refused.
        # Refused rather than quietly answered in wav: handing a caller who
        # believes it asked for mp3 a wav file with no error is the exact
        # defect this release removed. The image installs ffmpeg and the build
        # checks for it, so this is a warning for a hand-rolled environment.
        log.warning("ffmpeg is not on PATH: %s are unavailable and requests "
                    "for them are refused with a 400 — mp3 included, which is "
                    "the default, so a request that omits response_format will "
                    "be refused too",
                    ", ".join(f for f in MEDIA_TYPES if f not in FORMATS))
    yield
    sweeper.cancel()
    # DRAIN BEFORE CLOSE, and the order is the whole fix. This used to put one
    # sentinel on the queue, join nothing, and close the model on the next line
    # -- so a shutdown during synthesis tore a 6.5 GB model out from under the
    # thread that was using it, and whatever that produced was neither logged
    # nor recoverable.
    if not dispatch.drain(SHUTDOWN_GRACE_S):
        # CANCELLED, NEVER FAILED. `failed` means the synthesis went wrong;
        # this one was going fine and the service was asked to stop.
        for job in list(jobs.values()):
            if job["status"] in {"queued", "running"}:
                job.update(status="cancelled", finished_at=time.time(),
                           error="the service was shutting down")
        log.warning("the lanes did not come back clean within %.0fs; what was "
                    "still running or still waiting is marked cancelled",
                    SHUTDOWN_GRACE_S)
    state["synths"].close()  # type: ignore[attr-defined]
    state.clear()


app = FastAPI(
    title="tts-long",
    description="Chatterbox long-form speech, CPU, as a job queue.",
    lifespan=lifespan,
)

# Installed on the app rather than route by route, so a route added later is
# covered without anyone having to remember to ask for it. TTS_API_KEYS is a
# parameter of the shared middleware precisely so that no operator-visible
# variable had to be renamed for this service to stop keeping its own copy —
# a copy that, among other things, could never authenticate a key with an
# accent in it and answered 401 to a probe written `/health/`.
auth.install(app, "TTS_API_KEYS")

# The /v1 error envelope, in the four-field shape the schema requires, plus
# the 404, 405 and 500 handlers that used to escape it. Order against
# auth.install no longer matters: the middleware's 401 is built by the same
# error_response as everything else, so it carries `param` without this
# service rebinding a name to put it there. The native routes keep FastAPI's
# own `{"detail": ...}` and its 422: /jobs is the older contract and something
# out there already parses it.
install_errors(app)


class JobRequest(_controls_model("_JobControls")):
    """The native job body.

    **`extra="forbid"`, and it ships in the same commit as `model`.** This
    model had no `model_config` at all, so `POST /jobs {"model": "..."}` was
    ACCEPTED AND SILENTLY DISCARDED -- before any of this existed. The page
    uses /jobs exclusively and always will, so adding an engine selector to
    /v1/audio/speech and not to this one would ship a page that asks for turbo,
    gets the other engine, and reports no error anywhere in the stack. That is
    the same shape as two defects this project has already shipped, and it is
    the reason these two lines are one commit.

    FastAPI renders the refusal as `{"detail": [...]}` with 422, which is this
    route's existing error shape, so nothing new is invented and nothing that
    parses it has to change.

    **`None` IS THE ONLY VALUE THAT CAN MEAN "THE CALLER DID NOT SAY."** A
    float default makes every caller look like they asked for the deployment's
    number, and compose.yaml sets TTS_EXAGGERATION=0.3 -- so every turbo
    request would arrive carrying two fields turbo cannot honour, written by a
    config file rather than by a person. The defaults are resolved AFTER the
    engine is known, from that engine's own controls.

    Chosen over `model_fields_set` because None survives the record round trip
    and the page's retry path, which rebuilds a body from a stored record: with
    `model_fields_set` a retry would resurrect a field the original caller
    never sent.
    """

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    # voice_common.models.Segment: the same `text` and the same 0.0–10.0
    # second `pause_after`, which is a published part of both this API and
    # tts-stack's and no longer free to drift in one of them.
    segments: list[Segment] | None = None
    # WHICH ENGINE. Absent resolves to TTS_DEFAULT_ENGINE, which is exactly
    # what every /jobs caller has had until now.
    model: str | None = None
    language: str | None = None
    # exaggeration, cfg_weight, temperature, flow_steps and cfg_alpha come from
    # the generated base above, with their bounds off CONTROL_RANGES.
    voice: str | None = None


def _runner_settings(engine: str) -> dict | None:
    """What the runner says this engine's service was installed with.

    Off the cached snapshot -- /health has already paid for it, and asking the
    runner again here would put a second LAN round trip on the event loop that
    `install_health` writes in capitals must never block.

    None where the runner does not publish any, which is every runner built
    before manifests carried them. Absent is not empty: a page that drew "{}"
    would be reporting that the machine is configured with nothing.
    """
    snapshot = _last_runner_snapshot()
    if not snapshot:
        return None
    service = ENGINES[engine].runner_service
    for row in (snapshot.get("services") or []):
        if row.get("id") == service:
            return row.get("settings") or None
    return None


def _last_runner_snapshot() -> dict | None:
    """The snapshot `_health` already fetched, without fetching another."""
    runner = state.get("runner")
    return getattr(runner, "_snap", None) if runner is not None else None


async def _health() -> dict[str, object]:
    """The body, unchanged apart from the queue's new ceiling.

    install_health registers `/health` AND exempts exactly that string from
    authentication, so a rename can no longer lock the container healthcheck
    out — the two used to be independent literals in two modules.

    It also makes the route `async def`, which this service had already paid to
    learn: a sync route runs on AnyIO's forty-thread pool, shared with
    /v1/audio/speech, which held a thread for up to TTS_OPENAI_SYNC_TIMEOUT
    seconds each. Forty concurrent speech requests starved the pool, /health
    stopped answering, and an orchestrator restarted a service that was merely
    busy.

    WHICH IS WHY THE RUNNER IS ASKED OFF THE LOOP. `install_health` says it in
    capitals -- details must not block, it runs on the event loop -- and the
    runner panel broke it: `RunnerClient.snapshot()` is a cache-missing
    BLOCKING call over the LAN to a desktop that is allowed to be switched off.
    Measured: /health took 3.0 s and GET /jobs took 2.7 s WHILE IT WAITED. It
    was never one slow route. Nothing else ran at all -- not the listing, not
    /v1/audio/speech, not the SSE routes -- so a machine that drops packets
    rather than refusing them stopped a service that could speak perfectly
    well. `snapshot()` is on the three-second offer clock now as well, because
    moving a thirty-second wait off the loop still leaves it holding a
    healthcheck open past its own timeout.
    """
    pool: _Synths = state["synths"]  # type: ignore[assignment]
    runner = state.get("runner")
    snapshot = (await run_in_threadpool(runner.snapshot)
                if runner is not None else None)
    lane = dispatch.lanes.get("runner")
    probe = lane.probe if lane is not None else None
    return {
        "status": "ok",
        "model_loaded": pool.loaded,
        "threads": THREADS,
        "queued": dispatch.depth(),
        "queue_capacity": MAX_QUEUE,
        "running": sum(1 for j in list(jobs.values()) if j["status"] == "running"),
        "realtime_factor": round(rate.value, 3),
        # Named per backend rather than merged, for the same reason the EMAs are
        # separate: "this host does 0.28x and the runner does 20x" is two facts,
        # and averaging them describes no machine that exists.
        # KEPT, AND NARROWED TO THE DEFAULT ENGINE rather than left as an
        # average of two things 2.36x apart. ui.html and test_interface.py both
        # read these and both keep being true, which is what lets the server
        # half of this ship before the page half.
        "realtime_factor_by_backend": {
            k.split("/", 1)[0]: round(v.value, 3) for k, v in _rates.items()
            if k.endswith(f"/{DEFAULT_ENGINE}")},
        # HOW MANY JOBS ARE BEHIND EACH OF THOSE FIGURES. A rate with zero
        # observations is the seed and nothing else -- a documented measurement
        # from another machine, published so that a router which has not tried a
        # backend yet can tell a hypothesis from a fact. Without this the third
        # backend is a number nobody can weigh, and the first thing anybody does
        # with an unweighable number is trust it.
        "backend_observations": {
            k.split("/", 1)[0]: v.count for k, v in _rates.items()
            if k.endswith(f"/{DEFAULT_ENGINE}")},
        # AND THE UNNARROWED TRUTH BESIDE IT, one row per (lane, engine). A
        # count of zero says out loud that the figure next to it is a
        # hypothesis rather than a measurement, which is the whole reason the
        # observation counts exist: the first thing anybody does with an
        # unweighable number is trust it.
        "realtime_factor_by_engine": {k: round(v.value, 3)
                                      for k, v in _rates.items()},
        "engine_observations": {k: v.count for k, v in _rates.items()},
        # WHAT EACH ENGINE IS AND WHERE IT CAN RUN RIGHT NOW. `controls` and
        # `languages` are the honest surface of the checkpoint, so a page can
        # remove a slider that would do nothing rather than draw one that lies;
        # the two lane rows are why an option is unavailable, in the runner's
        # own words, so it can be shown DISABLED WITH A REASON rather than
        # hidden.
        "engines": {
            engine: {
                **row,
                "local": {"ready": ENGINES[engine].local,
                          "why": "" if ENGINES[engine].local
                                 else "not in TTS_LOCAL_ENGINES",
                          "resident": pool.resident(engine)},
                "runner": {
                    "ready": bool(probe is not None and probe.ok_for(engine)),
                    "why": (probe.why_for(engine) if probe is not None
                            else "no runner lane is configured"),
                    "service": ENGINES[engine].runner_service,
                    # WHAT THAT MACHINE IS ACTUALLY SERVING, echoed from its
                    # own manifest. The load-time settings -- quantisation
                    # group size, the frame ceiling, the fade, the low-pass --
                    # are not caller fields and never will be, but they decide
                    # how the audio sounds, and the only other way to read them
                    # is an SSH session that lands in Session 0 and cannot see
                    # the desktop. `curl /health | jq` answers it instead.
                    "settings": _runner_settings(engine)},
            }
            for engine, row in engine_rows().items()},
        "default_engine": DEFAULT_ENGINE,
        # The order work is offered in, so the page and a person reading /health
        # can see the policy rather than infer it from where jobs ended up.
        "backend_order": list(BACKEND_ORDER),
        # WAS A BOOLEAN, AND A BOOLEAN ANSWERED THE WRONG QUESTION. "A runner is
        # configured" is not what anybody wants to know; "is it up, is it free,
        # and if not why not" is. The page draws this, so it is the whole
        # snapshot and not a flag. None when no runner is configured at all,
        # which is still distinguishable from one that is configured and down.
        "runner": snapshot,
        # ONE ROW PER LANE, and it is what makes "why did that job run here"
        # answerable without reading the log. A shut lane says which of the
        # three reasons shut it: nothing configured, the probe cannot reach it,
        # or it handed a job back and is cooling.
        "dispatch": dispatch.snapshot(),
        "host_label": HOST_LABEL,
    }


install_health(app, _health)


def _estimate(chars: int, engine: str | None = None) -> int:
    """Seconds until `chars` is spoken, WHEREVER IT IS ABOUT TO BE SPOKEN.

    Three things have been wrong with this number in turn. It used a fixed
    0.21x, which under-predicted by 1.5x on the machine the audit measured; it
    counted whitespace-separated words, so a 5000-character string with no
    spaces in it was estimated at two seconds; and -- until lanes -- it divided
    by the LOCAL rate whatever the destination, so a job bound for the runner's
    card was promised twenty-one minutes and delivered in seven. Somebody told
    twenty-one minutes goes away.
    """
    return max(0, round(dispatch.estimate_for(speech_seconds(chars),
                                              engine)[1]))


def _enqueue(*, segments: list[tuple[str, float]], language: str | None,
             controls: dict | None = None,
             voice: str, reference: str | None, fmt: str = "wav",
             text: str | None = None,
             spec=None, model_requested: str | None = None,
             engine_reason: str = "default",
             waiter: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None,
             stream: Stream | None = None) -> str:
    spec = spec if spec is not None else ENGINES[DEFAULT_ENGINE]
    controls = controls or {}
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id, "status": "queued", "created_at": time.time(),
        # THE THREE FIELDS THAT MAKE THIS ROW COMPARABLE WITH ANOTHER
        # SERVICE'S. tts-long is no longer the only thing that produces a
        # record -- Kokoro and Parakeet post theirs to /runs -- so a row that
        # cannot say what kind of work it was, which engine did it or which
        # machine it was on is a row the listing cannot filter or explain.
        # `host` is required even here, where there has only ever been one
        # machine: it is what makes a second one a data change rather than a
        # change to the page.
        #
        # `kind` IS ONE EXPRESSION OFF THE CATALOGUE AND NOT A FOURTH VALUE.
        # An engine that has no speaker encoder is not cloning anything, so a
        # preset job is `speech` -- the same word Kokoro's rows use, for the
        # same reason. A fourth KINDS value would be a fourth table for three
        # services to agree about, for a distinction the `engine` column
        # already makes.
        "kind": "clone" if spec.facts.reference_audio else "speech",
        "service": "tts-long", "engine": spec.id,
        # WHAT THE CALLER TYPED AND WHY THEY GOT WHAT THEY GOT. Without the
        # second of these a row that says `chatterbox` cannot distinguish "the
        # caller named this engine" from "the caller named nothing and the
        # deployment's default happened to be this" -- which is how a change of
        # default becomes invisible in six months of history.
        "model_requested": model_requested, "engine_reason": engine_reason,
        "host": HOST_LABEL, "route": "/jobs",
        "text": text, "language": language, "segments": segments,
        # ONE COLUMN PER CONTROL, SPREAD FROM THE MAPPING. Flat on the job and
        # flat in the record: RECORD_KEYS is an allow-list and run_records.json
        # is a pinned wire shape that both halves of POST /runs read, so extra
        # columns are additive and migration-free while a nested `controls`
        # object is neither. Absent is None, which is what "the caller did not
        # say and this engine has no default for it" means everywhere else.
        **{field: controls.get(field) for field in WIRE_CONTROLS},
        # WHAT RATE THE AUDIO OF THIS JOB IS, recorded rather than assumed. The
        # 24k/48k confusion in the wrapper Voxtral runs through is the failure
        # that ships audio playing at half speed with nothing reporting it, so
        # the number goes on the row a person can read.
        "sample_rate": spec.facts.native_sample_rate,
        "format": fmt, "voice": voice,
        "reference": reference, "cancelled": False, "stream": stream,
        "chunks": len(segments),
        # ON THE JOB, not only in the 202. It was computed for the POST
        # response and thrown away, so GET /jobs never carried it -- and the
        # page reads `job.estimated_seconds || 0` from the LISTING to size its
        # progress bar. The bar, the elapsed/remaining line and the "past the
        # estimate" state were therefore dead for every client that polled or
        # reloaded, which is all of them after the first render.
        #
        # Frozen at enqueue rather than recomputed per poll: it is the promise
        # the caller was given in the 202, and a bar whose total moves under it
        # as the EMA drifts is worse than one that is a little wrong.
        "estimated_seconds": _estimate(sum(len(t) for t, _ in segments),
                                       spec.id),
    }
    # Registered before the job is visible to the worker, or a fast job could
    # finish and find nothing to wake.
    if waiter is not None:
        events[job_id] = waiter
    dispatch.submit(job_id)
    return job_id


def _full() -> bool:
    """Is the ceiling reached, COUNTING WHAT IS RUNNING.

    `queue.qsize()` counted only the jobs still waiting, so the service
    admitted MAX_QUEUE on top of however many lanes were busy and the ceiling
    was never the ceiling.
    """
    return MAX_QUEUE > 0 and dispatch.depth() >= MAX_QUEUE


def _retry_after() -> int:
    """Seconds a rejected caller should wait: how long the backlog will take.

    Through the dispatcher, so the answer is about the lane the work would
    actually go to. Quoting the local rate for a stack whose backlog is sitting
    on a GPU tells a caller to come back three times too late.
    """
    return max(1, round(dispatch.estimate_for(_pending_work())[1]))


def _pending_work() -> float:
    """Seconds of speech accepted and not yet finished, over a snapshot.

    Over a `list()` snapshot, and so is every other walk of `jobs`. The dict is
    read from the event loop and from AnyIO's thread pool -- every sync route
    runs there -- while DELETE pops from it and the sweeper pops from it, and a
    plain `.values()` iteration interrupted by a pop raises RuntimeError:
    dictionary changed size during iteration. That would be a 500 on
    /v1/audio/speech caused by an unrelated DELETE landing in the same
    millisecond.
    """
    return sum(_job_work(job) for job in list(jobs.values())
               if job["status"] in {"queued", "running"})


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest) -> dict[str, object]:
    if not req.text and not req.segments:
        raise HTTPException(400, "provide either text or segments")
    # THE SAME SEQUENCE AS /v1/audio/speech AND THE SAME REFUSALS, rendered in
    # this route's own shape. /jobs keeps `{"detail": ...}` and /v1 keeps
    # OpenAI's four-field envelope -- one function decides WHAT is refused,
    # each route decides how its own callers are told, and unifying the two
    # would break something out there that already parses this one.
    chosen = _choose(model=req.model, voice=req.voice, language=req.language,
                     controls=_wire_controls(req))
    if isinstance(chosen, Refusal):
        # NOTHING IS CREATED. R9 in particular is refused here rather than
        # queued: a 202 nothing can serve is a progress bar that never moves,
        # and thirty-two of them make _full() answer 429 to every caller of the
        # OTHER engine on a completely idle lane.
        headers = {"Retry-After": "60"} if chosen.status == 503 else None
        raise HTTPException(chosen.status, chosen.message, headers=headers)
    if _full():
        # The queue is one process's memory and disk. Nothing bounded it
        # before, so a client in a loop could accept an hour of work in a
        # second and then wait an hour for the first of it.
        retry = _retry_after()
        raise HTTPException(429, f"queue is full ({MAX_QUEUE} jobs); retry in "
                                 f"about {retry}s",
                            headers={"Retry-After": str(retry)})
    segments = _segments(req.text, req.segments)
    params = chosen.params
    job_id = _enqueue(segments=segments, text=req.text,
                      language=params.get("language"), controls=params,
                      voice=chosen.voice, reference=chosen.reference,
                      spec=chosen.spec, model_requested=req.model,
                      engine_reason=chosen.engine_reason)
    # Read back off the job rather than computed a second time, so the 202 and
    # every later GET /jobs quote the same number. Two calls to _estimate()
    # either side of a finished job would not.
    return {"id": job_id, "status": "queued",
            "queued_ahead": max(0, dispatch.position(job_id)),
            "chunks": len(segments),
            # WHICH ENGINE THIS JOB IS FOR, in the 202 rather than only on the
            # row. A caller that named nothing learns what it got without a
            # second request, and a caller that named turbo can see it landed.
            "engine": chosen.spec.id,
            "estimated_seconds": jobs[job_id]["estimated_seconds"]}


# ------------------------------------------------------- imported records --
#
# tts-long is the only service in this stack that keeps a record of anything.
# Kokoro answers a request and forgets it; Parakeet returns a transcript and
# forgets it. So the two of them POST a finished run here and forget THAT too:
# one request, no retry, and a dropped record on a full queue. See
# voice_common.runlog for the sending half and the reason it must never be on
# the caller's clock.


# HOW MANY RECORDS EACH SENDER MAY POST PER MINUTE. A sender in a loop is a
# full disk, and the disk is shared with the audio.
_runlog_hits: dict[str, list[float]] = {}
_runlog_lock = threading.Lock()
# The stored-record count, cached. NEVER an os.scandir per write: that is O(n)
# per write and quadratic over a batch, and `runs/` is its own directory
# precisely so this stays off _recover's audio walk.
_runlog_count: dict[str, float] = {"n": 0.0, "at": 0.0}


def _runlog_allowed(service: str) -> bool:
    now = time.monotonic()
    with _runlog_lock:
        hits = [t for t in _runlog_hits.get(service, ()) if now - t < 60.0]
        if len(hits) >= RUNLOG_RATE:
            _runlog_hits[service] = hits
            return False
        hits.append(now)
        _runlog_hits[service] = hits
        return True


def _runlog_room() -> bool:
    """Is there room for another record, against a count refreshed on a timer."""
    now = time.monotonic()
    with _runlog_lock:
        if now - _runlog_count["at"] > RUNLOG_COUNT_TTL:
            try:
                _runlog_count["n"] = float(sum(
                    1 for _ in (OUT_DIR / "runs").glob("*.json")))
            except OSError:
                _runlog_count["n"] = 0.0
            _runlog_count["at"] = now
        if _runlog_count["n"] >= RUNLOG_MAX_RECORDS:
            return False
        _runlog_count["n"] += 1
        return True


@app.post("/runs", status_code=201)
def import_run(record: dict) -> dict[str, str]:
    """Accept one finished run from another service in this stack.

    A `dict` rather than a pydantic model ON PURPOSE, and it is the whole
    mid-upgrade story. UNKNOWN KEYS ARE DROPPED SILENTLY: a field added to a
    newer sender must never 400 an older receiver, or every deploy becomes an
    ordering problem and the two halves of this feature cannot ship
    independently. A model with extra="forbid" would do exactly that.

    WHAT IS NOT FORGIVEN is a sender setting a server-only field. `path` is the
    argument to open() on /jobs/{id}/audio; `bytes` and `text_preview` describe
    work this service did. A sender that tries is a bug in that sender, not a
    compatibility case, and a 400 that names the field is how it gets fixed
    rather than silently ignored.

    SERVICE TO SERVICE, NEVER FROM THE BROWSER. This route is deliberately
    absent from the page's proxy table and from the gateway: a mutable log with
    the page as a writer is not a log.
    """
    if not RUNLOG_ACCEPT:
        # 404 rather than 403, and the senders treat any failure the same way:
        # drop the record and carry on. Switching this off must not make
        # another service noisy.
        raise HTTPException(404, "this service is not accepting run records")
    if not isinstance(record, dict):
        raise HTTPException(400, "a run record is a JSON object")
    forbidden = sorted(SERVER_ONLY_KEYS & set(record))
    if forbidden:
        raise HTTPException(400, f"{forbidden} are set by this service and "
                                 f"cannot be sent")
    service = str(record.get("service") or "unknown")[:64]
    if not _runlog_allowed(service):
        raise HTTPException(429, f"{service} has posted more than {RUNLOG_RATE} "
                                 f"records in a minute",
                            headers={"Retry-After": "60"})
    if not _runlog_room():
        raise HTTPException(429, f"this service is holding its ceiling of "
                                 f"{RUNLOG_MAX_RECORDS} records",
                            headers={"Retry-After": "3600"})
    kept = {k: v for k, v in record.items() if k in RECORD_KEYS}
    if (kept.get("kind", "clone") != "clone"
            and (kept.get("audio_deleted") or kept.get("audio_expired"))):
        # THE SAME LIE AS DELETE /jobs/{id}/audio, REACHED FROM THE OTHER END.
        # Both booleans mean something happened to a file THIS service was
        # holding: one that somebody deleted it, one that a clock took it.
        # Kokoro and Parakeet keep no file here, so a sender that sets either
        # is describing an event that cannot have happened, and the row then
        # reads "deleted" to everyone who looks at it. SERVER_ONLY_KEYS cannot
        # carry this rule, because for a clone row these two are exactly what a
        # re-imported record is meant to bring back.
        #
        # Only a TRUE value is refused. A sender that faithfully copies its
        # whole row, false fields included, is saying nothing untrue and is not
        # worth a 400 -- the defaults below are already false.
        raise HTTPException(400, "audio_deleted and audio_expired describe "
                                 "audio this service stored, and a "
                                 f"{kept.get('kind')} run has none here")
    job_id = str(record.get("id") or "").strip() or str(uuid.uuid4())
    if job_id in jobs or "/" in job_id or job_id.startswith("."):
        # A SENDER DOES NOT GET TO NAME AN EXISTING ROW, and it does not get to
        # name a path either. `id` is the audio filename for a clone job, so a
        # collision would attach somebody else's file to this record and a
        # separator would put the record outside runs/.
        job_id = str(uuid.uuid4())
    text = kept.get("text")
    if isinstance(text, str):
        # TRUNCATED BY THE RECEIVER TOO, not only by the sender. The sender is
        # asked to do it; a log that can be made to hold a megabyte per line by
        # a sender that forgot is a disk somebody loses.
        kept["text"] = text[:MAX_RECORD_TEXT]
    kept.setdefault("kind", "clone")
    kept.setdefault("status", "done")
    kept.setdefault("created_at", time.time())
    kept.setdefault("started_at", kept["created_at"])
    job = {"id": job_id, "cancelled": False, "recovered": False,
           "path": None, "bytes": 0,
           "audio_deleted": False, "audio_expired": False}
    job.update(kept)
    job.update(id=job_id, path=None, bytes=0, recovered=False)
    jobs[job_id] = job
    _write_record(job)
    log.info("%s recorded a %s run on %s (%s)", job.get("service"),
             job.get("kind"), job.get("host"), job_id[:8])
    return {"id": job_id}


def _segments(text: str | None,
              segments: list[Segment] | None) -> list[tuple[str, float]]:
    """Everything the worker will speak, as (text, pause_after) pairs.

    Every piece goes through the chunker, segments included: a `segments` entry
    of 1200 characters hits generate()'s 40-second ceiling exactly as a flat
    `text` of 1200 characters does. The pause belongs to the LAST piece of a
    segment, so splitting a segment does not insert a gap that was never asked
    for.
    """
    if segments:
        out: list[tuple[str, float]] = []
        for segment in segments:
            pieces = chunk_text(segment.text)
            if not pieces:
                # A segment with no text is a pause and nothing else, which
                # voice_common.audio.splice already handles.
                out.append(("", segment.pause_after))
                continue
            for index, piece in enumerate(pieces):
                last = index == len(pieces) - 1
                out.append((piece, segment.pause_after if last else 0.0))
        return out
    return [(piece, 0.0) for piece in chunk_text(text or "")]


def _said(job: dict) -> str:
    """Everything this job speaks, as one string.

    Over `segments` AS THE WORKER HOLDS THEM, which are (text, pause_after)
    pairs -- not the Segment models the request carried. This was reading them
    as if they were dicts, so `seg.get` raised AttributeError on a tuple and a
    single segments-only job turned GET /jobs into a 500 for every job in the
    list. The page sends segments whenever the text has paragraph pauses, so
    the whole Jobs tab went blank; measured with a one-segment request against
    the real routes, see test_jobs.py.
    """
    if job.get("text"):
        return job["text"]
    return " ".join(text for text, _ in (job.get("segments") or []))


def _public(job: dict) -> dict:
    """The job as /jobs reports it, over a snapshot for the reason above.

    `dict(job)` rather than `job.items()`: the worker thread ADDS keys to a
    running job — started_at, then path and the timings — so a comprehension
    over the live dict can be interrupted mid-walk by the job it is reporting
    on finishing, which is a 500 on GET /jobs/<id> at the exact moment a caller
    is most likely to be polling it.
    """
    out = {k: v for k, v in dict(job).items()
           if k not in {"segments", "text", "stream", "reference"}}
    # ABSENT MEANS CLONE, everywhere, and this is the line that makes every
    # record written before this release a valid one. There is no migration of
    # file contents anywhere in this service.
    out["kind"] = job.get("kind", "clone")
    out["audio"] = _audio(job)
    # A PREVIEW OF WHAT WAS SAID, because a list of uuids identifies nothing.
    # Every job in that list looked the same: the same voice name, a different
    # random id, and no way to tell which was the one you wanted without
    # downloading each in turn.
    #
    # A preview rather than the text: `text` is capped at 4096 characters and
    # this route returns fifty jobs, so sending it whole would be a couple of
    # hundred kilobytes on a poll that runs every few seconds. GET /jobs/{id}
    # carries the full text -- see below -- which is what a disclosure control
    # in a client should ask for.
    source = _said(job)
    if source:
        source = " ".join(source.split())
        out["text_preview"] = source[:140] + ("…" if len(source) > 140 else "")
        out["text_length"] = len(source)
    return out


def _audio(job: dict) -> dict[str, object]:
    """The one thing a reader wants to know about a row's audio, as one object.

    STORED AS TWO BOOLEANS, READ AS ONE STATE, because the page kept guessing.
    It had `hasAudio`, which tested `path` and then, when that was not enough,
    guessed at the server's sweep from `finished_at` and a TTL constant it kept
    its own copy of -- and got it wrong every time the two disagreed.

    `never` IS LOAD-BEARING. An instant run's audio was not lost, it was never
    kept here; telling somebody "deleted" about a file that never existed is
    the exact lie this enum exists to prevent. `pending` is the other one: a
    job that has not finished has no audio YET, which is not the same as
    having none.

    `format`, `bytes` and `url` appear ONLY in state `present`. A url for a
    file that is gone is a 409 waiting to be clicked.
    """
    if job.get("audio_deleted"):
        state_ = "deleted"
    elif job.get("audio_expired"):
        state_ = "expired"
    elif job.get("path"):
        state_ = "present"
    elif job["status"] in {"queued", "running"}:
        state_ = "pending"
    elif job.get("kind", "clone") != "clone":
        state_ = "never"
    else:
        state_ = "expired"
    if state_ != "present":
        return {"state": state_}
    return {"state": "present", "format": job.get("format"),
            "bytes": job.get("bytes") or 0,
            "url": f"/jobs/{job['id']}/audio"}


# What `?audio=` and `?kind=` and `?status=` accept. Named here rather than
# inline so an unknown value can be REFUSED with the list -- a filter that
# silently matches nothing looks exactly like an empty stack.
AUDIO_STATES = ("present", "deleted", "expired", "never", "pending")
KINDS = ("clone", "speech", "transcribe")
STATUSES = ("queued", "running", "done", "failed", "cancelled", "live")
LIVE = ("queued", "running")


def _wanted(raw: str | None, allowed: tuple[str, ...], name: str) -> set[str] | None:
    """A comma-separated filter value, or None for "everything"."""
    if raw is None or not raw.strip():
        return None
    values = {v.strip() for v in raw.split(",") if v.strip()}
    unknown = values - set(allowed)
    if unknown:
        raise HTTPException(400, f"unknown {name} {sorted(unknown)!r}; this "
                                 f"service has {', '.join(allowed)}")
    return values


@app.get("/jobs")
def list_jobs(limit: int = 50, kind: str | None = None,
              audio: str | None = None,
              status: str | None = None) -> dict[str, object]:
    """The listing, filtered ON THIS SIDE and counted before the filter.

    FILTERING HERE RATHER THAN IN THE BROWSER IS WHAT MAKES THE DEFAULT
    SHIPPABLE. The listing is capped, and once instant speech and
    transcriptions are recorded a morning of Kokoro presses would push last
    night's clone off the end of the response before any client-side filter
    ever saw it. A cap applied before a filter is a cap on the wrong set.

    A LIVE JOB IS RETURNED BY EVERY COMBINATION, ALWAYS, and before the limit.
    It has no audio YET, which is not the same as having none, and hiding the
    thing somebody is waiting for is the worst possible reading of a filter.

    A FAILED JOB IS RETURNED BY `audio=present` TOO. A clone that fails has no
    audio; a naive filter makes it vanish at the moment its owner is watching
    it, which reads as data loss rather than as a failure.

    `counts` is over EVERY record, before any filter, so the page can label
    "Everything (412)" without a second request -- and so a default that hides
    rows can say how many it is hiding.
    """
    want_kind = _wanted(kind, KINDS, "kind")
    want_audio = _wanted(audio, AUDIO_STATES, "audio state")
    want_status = _wanted(status, STATUSES, "status")
    if want_status and "live" in want_status:
        want_status = (want_status - {"live"}) | set(LIVE)

    counts: dict[str, int] = {k: 0 for k in ("all",) + KINDS + AUDIO_STATES
                              + ("failed", "cancelled", "live")}
    keep = []
    for job in sorted(list(jobs.values()), key=lambda j: -j["created_at"]):
        shape = _public(job)
        state_ = shape["audio"]["state"]  # type: ignore[index]
        counts["all"] += 1
        counts[shape["kind"]] = counts.get(shape["kind"], 0) + 1  # type: ignore[index]
        counts[state_] += 1
        if job["status"] in LIVE:
            counts["live"] += 1
        elif job["status"] in {"failed", "cancelled"}:
            counts[job["status"]] += 1
        live = job["status"] in LIVE
        if want_kind and shape["kind"] not in want_kind:
            continue
        if want_status and job["status"] not in want_status and not live:
            continue
        if want_audio and state_ not in want_audio and not (live or
                                                            job["status"] == "failed"):
            continue
        keep.append(shape)
    limit = max(1, min(limit, 200))
    return {"jobs": keep[:limit], "counts": counts,
            "truncated": len(keep) > limit}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """One job, with the full text this time.

    The listing carries a 140-character preview so fifty of them stay small on
    a poll; asking for one job is the moment you want all of it, and it is the
    request a client makes when someone expands a row rather than every few
    seconds for everything.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    out = _public(job)
    # _said, not job["text"], so a segments-only job answers with what it will
    # say. `text` is null on those, and the page's expandable row reads exactly
    # this field -- so every job the Speak tab submits with paragraph pauses
    # said "the text was not kept for this job" while the text was right there
    # in `segments`.
    said = _said(job)
    if said:
        out["text"] = said
    if job.get("segments"):
        out["segments"] = job["segments"]
    return out


@app.delete("/jobs/{job_id}/audio")
def delete_job_audio(job_id: str) -> dict:
    """Free the disk and keep the record.

    THE TWO THINGS A FINISHED JOB IS, SEPARATED. Deleting used to mean both:
    the audio went and the row went with it, so reclaiming a gigabyte also
    threw away what was said, which voice said it, how long it took and which
    machine did the work. Those are the only record that any of it happened,
    they cost a few hundred bytes, and they are what makes a list of past jobs
    worth having at all.

    The row stays, without a player, and says its audio is gone. The sidecar is
    rewritten rather than removed, so this survives a restart the same way
    everything else does: _recover walks audio files, so a record with no audio
    is not rebuilt from disk and would otherwise come back looking finished
    with a file behind it.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] not in {"done", "failed", "cancelled"}:
        raise HTTPException(409, "that job has not finished; cancel it instead")
    if _audio(job)["state"] == "never":
        # `never` IS LOAD-BEARING and this route manufactured the lie it
        # exists to prevent. An instant-speech or transcription row keeps no
        # audio HERE at all; deleting it answered 200, set audio_deleted, and
        # the row then read "deleted" -- telling a reader somebody removed a
        # file that never existed. The page never issues this, because the
        # button is gated on the state, but the route is proxied and it is in
        # the gateway, so any keyed client reaches it.
        #
        # Asked of `_audio` rather than of `kind`, so the route and the enum it
        # publishes cannot drift apart.
        raise HTTPException(409, f"a {job.get('kind', 'clone')} run keeps no "
                                 "audio on this service, so there is nothing "
                                 "here to delete")
    path = job.get("path")
    if path:
        with suppress(OSError):
            Path(path).unlink()
    job["path"] = None
    job["bytes"] = 0
    job["audio_deleted"] = True
    # NOT `audio_expired`. Both mean "there is nothing to play"; only one of
    # them means somebody chose it, and a row that says "expired" about a file
    # its owner deleted is telling them a clock did something they did.
    job.pop("audio_expired", None)
    _write_record(job)
    # THE SAME `audio` OBJECT EVERY OTHER READ OF THIS ROW CARRIES. The old
    # `{"status": "audio_deleted"}` was a fourth spelling of a state that has
    # exactly one -- a caller had to know that "audio_deleted" here meant
    # `audio.state == "deleted"` in the listing, and nothing said so.
    return {"id": job_id, "audio": _audio(job)}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Cancel a queued job, or discard a finished one.

    There was no way to do either: a 202 handed out an id and the worker ground
    through it whatever happened at the other end. A job that has already
    started stops at its next chunk boundary — generate() has no interruption
    point inside it, so a sentence already in flight is finished and kept.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job["cancelled"] = True
    if job["status"] in {"done", "failed", "cancelled"}:
        _discard(job)
        jobs.pop(job_id, None)
        return {"id": job_id, "status": "deleted"}
    return {"id": job_id, "status": "cancelling"}


def _file_stream(path: str) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        while True:
            block = handle.read(65536)
            if not block:
                return
            yield block


@app.get("/jobs/{job_id}/audio")
def get_audio(job_id: str) -> Response:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] not in {"done", "cancelled"} or not job.get("path"):
        raise HTTPException(409, f"job is {job['status']}")
    fmt = job["format"]
    # Content-Disposition stays on the NATIVE route: this one hands over a
    # finished artefact and a filename is useful. It is gone from /v1, where
    # OpenAI's schema has no such header and it forced a download instead of
    # inline playback.
    return StreamingResponse(
        _file_stream(job["path"]), media_type=MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{job_id}.{fmt}"'})


@app.get("/voices")
def list_voices() -> dict[str, object]:
    """What `voice` may name. Unknown names are a 400, so they are listed.

    Refreshed first, because this is the call a UI makes to build a voice
    picker and a picker that cannot see a clip someone just added is the whole
    reason the registry stopped being immutable.
    """
    VOICES.refresh()
    return {"voices": VOICES.names,
            "openai_aliases": {name: voice_registry.BUILTIN
                               for name in VOICES.aliased},
            "strict": VOICES.strict,
            # ADDITIVE. `voices` keeps its exact shape -- a list of strings --
            # because a picker is reading it right now and a second engine is
            # not a reason to break one.
            #
            # WHICH ENGINES CAN ACTUALLY USE EACH VOICE, and why not when they
            # cannot. The five-second reference minimum belongs to the PAIR, so
            # this is the only place it can be published; a page that hides a
            # short-clip voice from the turbo option, rather than disabling it
            # with the reason on the line, is how `chatterbox-cpu` stayed
            # invisible for its whole life.
            "detail": [_voice_detail(name) for name in VOICES.names]}


def _voice_detail(name: str) -> dict[str, object]:
    seconds = VOICES.seconds_for(name)
    usable, excluded = [], {}
    for engine, spec in ENGINES.items():
        if not spec.facts.reference_audio:
            # A PRESET ENGINE IS EXCLUDED FROM EVERY CLIP, AND NOT BECAUSE THE
            # CLIP IS TOO SHORT. `reference_seconds: null` on this row means
            # "the length could not be read", which a page turns into "probably
            # fine"; this engine has no speaker encoder at all, so the sentence
            # has to be about the checkpoint rather than about the file. The
            # one exception is a clip whose NAME matches one of that engine's
            # presets, which is a different voice reachable by the same word --
            # the pair is the key, so both stay usable.
            if name in spec.voice_names:
                usable.append(engine)
            else:
                excluded[engine] = (f"{engine} has no speaker encoder and "
                                    f"cannot read a reference clip; its voices "
                                    f"are fixed")
            continue
        floor = spec.min_reference_seconds
        if seconds is not None and floor > 0 and seconds < floor:
            excluded[engine] = (f"reference clip is {seconds:.1f} s; "
                                f"{engine} needs {floor:.1f} s")
        else:
            usable.append(engine)
    return {"name": name, "reference_seconds": (round(seconds, 1)
                                                if seconds is not None else None),
            "engines": usable, "excluded": excluded}


# ------------------------------------------------------- OpenAI compatible --
#
# Added alongside the native routes, not in place of them. /jobs stays the one
# to prefer for batch work, for the reasons in the module docstring.


class CustomVoice(BaseModel):
    """OpenAI's custom-voice object: `{"id": "voice_1234"}`.

    The schema's VoiceIdsOrCustomVoice is anyOf[string, {id: string}] and
    openai-python's `Voice` alias includes it. It was rejected here with
    "voice: Input should be a valid string" — including for the minimal SDK
    call, which sends exactly this form.

    Declared as a model rather than unwrapped by a validator so that this
    service's own /openapi.json keeps the anyOf, which is what a generated
    client reads to learn the object form is accepted. The name is visible on
    the wire — pydantic tags a failed union branch with the class name, and
    voice_common.errors puts that tag in the message — so it is OpenAI's word
    for the shape rather than an internal one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str


class SpeechRequest(OpenAISpeechRequest, _controls_model("_SpeechControls")):
    """OpenAI's /v1/audio/speech body, with the parts that are Chatterbox's.

    `model`, `input` and `speed` come from voice_common.models.OpenAISpeechRequest.

    **`extra="forbid"`, overriding the base.** The base allows unknown fields
    so that a client speaking a newer dialect of OpenAI's API is not rejected
    for it, and that reasoning is sound for a field OpenAI adds. It was not
    sound for what it actually did here: `{"stream": true}` and
    `{"totally_unknown_field": 123}` both returned 200 with audio, and
    `stream` is the TRANSCRIPTION-side switch, so a client that sent it
    expecting a stream got a buffered file and no way to tell. OpenAI's own
    schema sets additionalProperties: false and its API answers "Unrecognized
    request argument supplied"; matching that is what makes the reply
    trustworthy. The cost is that a genuinely new OpenAI field is a 400 until
    it is added here, which is the trade the README states.

    `response_format` and the speed range are deliberately NOT in the base.
    Both are properties of this image.
    """

    model_config = ConfigDict(extra="forbid")

    # "tts-1" rather than the base's "default", because that is what this
    # service has always answered with and something may be reading it back.
    #
    # HONOURED NOW, AND THE THREE OPENAI NAMES ARE ALIASES FOR THIS SERVICE'S
    # DEFAULT ENGINE. That is what they have always meant here; the difference
    # is that saying so is now a decision with an alternative rather than a
    # description of the only thing there was. x-tts-engine on the response
    # says which engine actually ran. Documented as a deviation in the README.
    model: str = "tts-1"
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    # str or {"id": ...}; resolved against app/voices.py, which is also what
    # decides whether an unknown name is a 400.
    voice: str | CustomVoice | None = None
    # OpenAI's default, and now this service's, because the image carries an
    # encoder for it. It used to default to wav while an EXPLICIT mp3 was
    # refused with a 400 — so a caller who omitted the field believing it had
    # asked for mp3 was handed wav with no error at all.
    response_format: str = "mp3"
    # "audio" (one buffered body) or "sse". Validated in the route so the
    # message can explain, rather than as a Literal that produces pydantic's.
    stream_format: str = "audio"
    # Refused rather than ignored. See the route.
    instructions: str | None = None
    # Not OpenAI fields. Accepted because the OpenAI shape has no room for the
    # knobs that decide how this reads, and extra_body is how openai-python
    # passes vendor options through. Declared rather than left to extra="allow"
    # now that unknown fields are refused.
    # None RATHER THAN A DEPLOYMENT DEFAULT. See JobRequest: a float default
    # makes every caller look like they asked for compose.yaml's number, and a
    # value the chosen engine cannot honour is then either refused for
    # something nobody wrote or discarded in silence.
    language: str | None = None
    # exaggeration, cfg_weight, temperature, flow_steps and cfg_alpha come from
    # the generated base, with their bounds off CONTROL_RANGES -- one table,
    # read here and by the config loader, so a value legal at boot cannot be
    # illegal on the wire.


@dataclass
class Chosen:
    """Everything settled before a job id exists: the voice, and the engine.

    Returned as one object because the five parts have to travel together --
    the resolved reference clip decides whether the engine will take the voice,
    the engine decides what the unset fields default to, and for an engine
    whose voices carry their own language the VOICE decides what goes in
    `params["language"]`. None of the three can be settled without the others.
    """

    voice: str
    reference: str | None
    spec: object
    engine_reason: str
    params: dict


def _wire_controls(req) -> dict[str, float | None]:
    """Every control the caller could have named, absent-never-missing.

    THE UNION, NOT THE ENGINE'S SUBSET, because this is read BEFORE the engine
    is known and a field left out here is a field silently dropped -- the exact
    failure the house rule exists to stop. `refuse` is what narrows it, by
    name, with the model's own reason.

    Built by walking WIRE_CONTROLS rather than by naming five attributes, so a
    control added to the catalogue's table reaches the refusal with no line
    here to edit.
    """
    return {field: getattr(req, field, None) for field in WIRE_CONTROLS}


def _all_controls_sentence() -> str:
    """Every control any engine here has, as prose. `a, b and c`."""
    names = sorted({f for spec in ENGINES.values() for f in spec.controls})
    if len(names) < 2:
        return names[0] if names else "no"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _render(refusal: Refusal) -> JSONResponse:
    """A Refusal in OpenAI's four-field envelope, for the /v1 routes."""
    return error_response(refusal.status, refusal.message,
                          code=refusal.code, param=refusal.param)


def _choose(*, model: str | None, voice, language: str | None,
            controls: dict) -> Chosen | Refusal:
    """The engine, then the voice inside that engine, then the pair.

    THE ENGINE IS RESOLVED FIRST NOW AND THE SWAP IS A DEFECT FIX. The old
    docstring's reasoning was that the five-second reference minimum is a
    property of the PAIR, so nothing can be said about a request until the clip
    behind the voice is known -- and that is still true, which is why the pair
    check is still LAST. What it did not say, because it could not, is that
    "the voice" was one directory of reference clips: the only kind of voice
    there was. A preset engine's voices are a property of the CHECKPOINT, so
    the name cannot even be looked up until the engine is known, and until this
    swap a Voxtral request naming `pt_male` died at `VOICES.resolve()` with a
    sentence about TTS_VOICE_DIR.

    `spec_for` depends on nothing but `model`, so putting it first costs
    nothing and orders the three questions the way a caller can act on them:
    which model, which voice on that model, then whether the two go together.

    ONE FUNCTION FOR BOTH ROUTES. /jobs and /v1/audio/speech ask exactly the
    same questions and used to ask two different subsets of them in two
    different orders -- which is how /jobs came to accept a language check the
    other route spelled differently and no `instructions` check at all.
    """
    chosen = spec_for(model)
    if isinstance(chosen, Refusal):
        return chosen
    spec, engine_reason = chosen

    requested = voice.id if isinstance(voice, CustomVoice) else voice
    if spec.voices is not None:
        # A CLOSED LIST OFF THE CHECKPOINT, NOT THE CLIP DIRECTORY. The name is
        # carried through unresolved: there is no file behind it, and `refuse`
        # is what decides whether the checkpoint has a speaker by that name.
        #
        # AN ABSENT VOICE IS REFUSED HERE AND NOT ANSWERED WITH THE FIRST ONE.
        # A cloning engine has a built-in speaker, so `default` is a real
        # answer to an unasked question; a checkpoint carrying twenty
        # equal embeddings has no such thing, and picking one -- whichever
        # happens to sort first, which is `ar_male` -- would be a value chosen
        # for a caller who would never have chosen it, in a language they did
        # not ask for. That is the house rule inverted: not a field dropped,
        # a field invented.
        if not (requested or "").strip():
            return Refusal(
                400,
                f"voice is required for {spec.id}: it carries "
                f"{len(spec.voice_names)} fixed speaker embeddings and no "
                f"built-in default among them, and its language is a property "
                f"of which one you pick. Choose one of: "
                f"{', '.join(spec.voice_names)}.",
                "unsupported_value", "voice")
        name = requested.strip()
        reference = None
    else:
        VOICES.refresh()
        resolved = VOICES.resolve(requested)
        if resolved is None:
            owner = PRESET_VOICE_ENGINE.get((requested or "").strip())
            if owner is not None:
                # Not "unknown voice": it is a real voice on this service, on
                # another engine. `refuse` says which one, in R0d's own words.
                named = refuse(spec, language=language, controls=controls,
                               voice_seconds=None,
                               voice_name=(requested or "").strip(),
                               voice_resolved=False)
                if named is not None:
                    return named
            return Refusal(
                400,
                f"unknown voice '{requested}': this service has "
                f"{', '.join(VOICES.names)} for {spec.id}. It clones from a "
                f"reference clip, so a voice is a file in TTS_VOICE_DIR.",
                "unsupported_value", "voice")
        name, reference = resolved

    refusal = refuse(spec, language=language, controls=controls,
                     voice_seconds=VOICES.seconds_for(name),
                     voice_name=name,
                     voice_is_object=isinstance(voice, CustomVoice),
                     voice_is_alias=(requested or "") in voice_registry.OPENAI_VOICES)
    if refusal is not None:
        return refusal

    lane = dispatch.lanes.get("runner")
    refusal = refuse_unavailable(spec, lane.probe if lane is not None else None)
    if refusal is not None:
        return refusal

    # THE DEFAULTS COME LAST, when the engine is known. An engine with no such
    # control gets no default, so the field stays None and never reaches
    # generate() at all.
    params = defaults_for(spec)
    for field_name, value in (("language", language), *controls.items()):
        if value is not None:
            params[field_name] = value
    # THE LANGUAGE COMES OFF THE VOICE FOR AN ENGINE THAT CARRIES IT THERE, and
    # this is the other half of the fix in `_defaults_from_env`. There is no
    # deployment default to fall back on -- there cannot be one, it could only
    # agree with the voice or contradict it -- so the record would otherwise
    # carry no language at all beside a Portuguese speaker. R7 has already
    # refused a `language` that disagrees, so this can only ever confirm.
    spoken = spec.language_of(name)
    if spoken is not None:
        params["language"] = spoken
    return Chosen(voice=name, reference=reference, spec=spec,
                  engine_reason=engine_reason, params=params)


def _validate(req: SpeechRequest) -> Chosen | JSONResponse:
    """Everything that can be refused before a single token is generated.

    Every parameter OpenAI's schema declares is either honoured or refused here
    by name. Nothing is accepted and dropped.
    """
    if not req.input.strip():
        return error_response(400, "input must not be empty",
                              code="invalid_value", param="input")
    if req.instructions is not None:
        # NO ENGINE HERE HAS INSTRUCTION CONDITIONING OF ANY KIND. Accepting
        # "speak cheerfully" and returning the same flat delivery is the
        # failure the audit found: no error, no warning header, nothing.
        #
        # CHECKED BEFORE THE ENGINE IS KNOWN, DELIBERATELY, AND THE SENTENCE
        # STOPPED SAYING "Chatterbox". Moving it after `_choose` would change
        # which refusal a doubly-invalid request gets first, and tests assert
        # that ordering; deriving the sentence from the catalogue instead costs
        # nothing and cannot go stale, because an engine that HAD instruction
        # conditioning would put it in `controls` and this list would grow.
        return error_response(
            400, "instructions is not supported: no engine on this service "
                 "has instruction conditioning. Delivery is controlled with "
                 f"the {_all_controls_sentence()} vendor fields, which "
                 "openai-python sends through extra_body, and which of them "
                 "apply depends on the engine -- see /health.engines.",
            code="unsupported_value", param="instructions")
    if req.response_format not in FORMATS:
        known = req.response_format in MEDIA_TYPES
        return error_response(
            400,
            f"response_format '{req.response_format}' is not available here: "
            + ("ffmpeg is not on PATH in this container."
               if known else
               f"OpenAI's formats are {', '.join(MEDIA_TYPES)}.")
            + f" This image produces {', '.join(FORMATS)}.",
            code="unsupported_value", param="response_format")
    if abs(req.speed - 1.0) > 1e-6:
        # Silently ignoring it would return audio of the wrong length, which
        # is worse than refusing. Engine-neutral for the same reason as
        # `instructions`: no engine in the catalogue has a rate control, so the
        # sentence is derived rather than named after the one there used to be.
        return error_response(400, "speed is not supported: no engine on this "
                                   "service has a rate control, and resampling "
                                   "to fake one shifts pitch with it.",
                              code="unsupported_value", param="speed")
    if req.stream_format not in {"audio", "sse"}:
        return error_response(
            400, f"stream_format '{req.stream_format}' is not one of 'audio' "
                 f"or 'sse'.", code="invalid_value", param="stream_format")
    # THE VOICE, THEN THE ENGINE, THEN EVERY FIELD THAT ENGINE CANNOT HONOUR.
    # `language` used to be checked here against one module-level list, which
    # was correct while there was one engine and is a substitution the moment
    # there are two: the languages a request may name are a property of the
    # checkpoint it is about to reach.
    chosen = _choose(model=req.model, voice=req.voice, language=req.language,
                     controls=_wire_controls(req))
    if isinstance(chosen, Refusal):
        return _render(chosen)
    return chosen


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest) -> Response:
    """OpenAI's speech endpoint: streamed, synchronous, or a job id.

    OpenAI's contract is request/response. This service runs at roughly 0.21x
    realtime, so for anything but short text a buffered synchronous answer is
    not slow, it is impossible — the socket dies long before the audio exists.
    Three answers, in the order a caller should want them:

    - `stream_format: "sse"` streams deltas as sentences finish. Nothing is
      buffered and nothing is faked: the first frame leaves when the first
      sentence is generated.
    - input the arithmetic says can be finished inside TTS_OPENAI_SYNC_TIMEOUT
      is waited on and returned as one body.
    - anything longer, or a wait that runs out, returns 202 with the job id
      and a Location header for the native route. openai-python treats a 202
      as success and will write that JSON into the caller's file; the README
      says so, and Retry-After plus Location are there for the clients that
      can act on them.
    """
    resolved = _validate(req)
    if isinstance(resolved, JSONResponse):
        return resolved
    voice_name, reference = resolved.voice, resolved.reference
    spec, params = resolved.spec, resolved.params

    if _full():
        # The only error response OpenAI's schema declares for this path, and
        # there was none of any kind here. Retry-After is the half that makes
        # it actionable: openai-python honours it when it retries a 429.
        retry = _retry_after()
        response = error_response(
            429, f"the queue is full ({MAX_QUEUE} jobs ahead). This service "
                 f"generates one job at a time on CPU; retry in about "
                 f"{retry}s.",
            type_="rate_limit_error", code="rate_limit_exceeded")
        response.headers["Retry-After"] = str(retry)
        return response

    text = req.input.strip()
    segments = _segments(text, None)
    fmt = req.response_format

    if req.stream_format == "sse":
        return _sse_response(req, resolved, segments, text, fmt)

    chars = len(text)
    budget = _sync_budget(spec)
    done = (asyncio.Event()
            if 0 < chars <= SYNC_MAX_CHARS
            and _compute_seconds(chars, spec.id) <= budget
            else None)
    job_id = _enqueue(
        segments=segments, text=text, language=params.get("language"),
        controls=params, voice=voice_name, reference=reference, fmt=fmt,
        spec=spec, model_requested=req.model,
        engine_reason=resolved.engine_reason,
        waiter=(asyncio.get_running_loop(), done) if done else None)

    # `async def` and an asyncio wait, not a sync route blocking on a
    # threading.Event. A sync route holds one of AnyIO's 40 worker threads for
    # the whole wait — up to TTS_OPENAI_SYNC_TIMEOUT seconds — and this
    # service is slow by design, so 40 concurrent callers took every thread
    # and /health, itself a sync route on that pool, stopped answering. An
    # orchestrator then restarted a service that was only busy. Waiting on the
    # event loop costs a coroutine instead, and nothing else queues behind it.
    if done is not None:
        try:
            await asyncio.wait_for(done.wait(), SYNC_TIMEOUT)
        except TimeoutError:
            # Deregister before falling through to the 202, so the worker does
            # not later wake an event nobody is holding. `pop` is the whole
            # handshake: if the worker got there first this is a no-op.
            events.pop(job_id, None)
        else:
            job = jobs[job_id]
            if job["status"] == "done":
                return StreamingResponse(
                    _file_stream(job["path"]), media_type=MEDIA_TYPES[fmt],
                    # WHAT MADE THIS AUDIO, beside whose voice it is. It is the
                    # only way a client holding a file can tell which of two
                    # engines produced it, and it mirrors x-stt-engine, which
                    # the transcription side has published for the same reason.
                    headers={"X-Voice": voice_name,
                             "x-tts-engine": job.get("engine", spec.id)})
            return error_response(500, job.get("error", "synthesis failed"),
                                  type_="server_error",
                                  code="synthesis_failed")

    # Either too long to wait for, or the wait ran out. The job is untouched
    # and still queued, so nothing has been wasted — Location points at where
    # it will appear.
    estimate = _estimate(chars, spec.id)
    return JSONResponse(
        status_code=202,
        headers={"Location": f"/jobs/{job_id}",
                 "Retry-After": str(max(1, estimate)),
                 "X-Voice": voice_name,
                 "x-tts-engine": spec.id},
        content={"id": job_id, "status": "queued",
                 "queued_ahead": max(0, dispatch.position(job_id)),
                 "estimated_seconds": estimate,
                 "audio_url": f"/jobs/{job_id}/audio",
                 "message": "This input is too long to answer inside one "
                            "request on this hardware. The audio is being "
                            "generated; collect it from audio_url, or send "
                            "stream_format='sse' to receive it as it is made. "
                            "This 202 is a documented deviation from OpenAI's "
                            "contract — see the README."},
    )


def _sync_budget(spec=None) -> float:
    """Seconds of compute a synchronous request may still spend.

    The timeout, less what is already queued, less the model load if it is not
    resident. The old code compared a character count against a constant and
    ignored both, so a request UNDER the documented threshold turned into a
    202 whenever the queue was busy or the model was cold — and the README
    table promised otherwise.
    """
    engine = (spec.id if spec is not None else DEFAULT_ENGINE)
    pool: _Synths | None = state.get("synths")  # type: ignore[assignment]
    resident = pool is not None and pool.resident(engine)
    # PER ENGINE, AND CHARGED ONLY WHEN THAT ENGINE IS NOT ALREADY RESIDENT.
    # Turbo is 67.5 s off disk against the multilingual model's 22.2 s, and
    # that difference lands inside whatever the caller is holding a socket
    # open for. One global constant charged a turbo request the wrong number in
    # both directions depending on which way it was set.
    cold = 0.0 if resident else _cold_load_seconds(engine)
    return (SYNC_TIMEOUT - dispatch.estimate_for(_pending_work(), engine)[1]
            - cold)


def _cold_load_seconds(engine: str) -> float:
    """What loading this checkpoint costs here.

    TTS_COLD_LOAD_SECONDS is still honoured, as the figure for the default
    engine, because it is set in compose files and in people's shells and a
    variable that silently stops applying is its own defect. Per engine below
    it, from the catalogue's measurement.
    """
    spec = ENGINES.get(engine)
    if spec is None:
        return COLD_LOAD_SECONDS
    if engine == DEFAULT_ENGINE and os.getenv("TTS_COLD_LOAD_SECONDS"):
        return COLD_LOAD_SECONDS
    return spec.cold_load_seconds


# ------------------------------------------------------------------- SSE --
#
# Framing rules, verified against openai-python 3.6.0's SSEDecoder:
#
#   * every event ends with a BLANK LINE, the last one included. A final event
#     terminated by a single \n is silently DROPPED, with no error;
#   * two data: lines with no blank line between them are joined into one
#     event and then fail to parse;
#   * a line starting with `:` is a comment and is ignored — which is what the
#     keepalives below are;
#   * a top-level `error` key makes the client raise APIError and stop, which
#     is the only way to report a failure after the 200 headers have gone.
#
# Bare `data:` frames, with no `event:` name line. The schema models the JSON
# payload only and gives no event field, and the one verbatim OpenAI audio SSE
# transcript in the same spec — the transcription example — uses bare data
# lines. openai-python dispatches on the JSON `type` either way. No `[DONE]`
# sentinel is sent: nothing authoritative says OpenAI emits one for this
# endpoint, and inventing a frame is worse than omitting one both SDKs treat
# as optional.


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_response(req: SpeechRequest, chosen: "Chosen",
                  segments: list[tuple[str, float]],
                  text: str, fmt: str) -> StreamingResponse:
    stream = Stream(loop=asyncio.get_running_loop())
    params = chosen.params
    job_id = _enqueue(
        segments=segments, text=text, language=params.get("language"),
        controls=params, voice=chosen.voice, reference=chosen.reference,
        spec=chosen.spec, model_requested=req.model,
        engine_reason=chosen.engine_reason,
        fmt=fmt, stream=stream)
    log.info("%s streaming %d chunks as %s", job_id[:8], len(segments), fmt)
    return StreamingResponse(
        _sse_events(job_id, stream),
        media_type="text/event-stream",
        headers={
            # no-store as well as no-cache: an intermediary that cached a
            # partial stream would replay someone else's audio.
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would hold
            # every delta until the response ended and turn a stream back into
            # the silence it exists to remove.
            "X-Accel-Buffering": "no",
            "X-Voice": chosen.voice,
            "x-tts-engine": chosen.spec.id,
            "X-Job-Id": job_id,
        })


async def _sse_events(job_id: str, stream: Stream) -> AsyncIterator[str]:
    """The frames themselves. One delta per encoded chunk, then one done.

    The audio in each delta is a slice of a single encode, so concatenating
    every decoded delta reproduces the buffered body byte for byte — for wav
    and flac, everything except the length fields in the header, which are not
    knowable until the last sentence is generated. app/encoders.py has the
    measured diff.
    """
    # Before anything is generated, so the headers reach the client now rather
    # than when the first sentence is ready. A comment line is legal SSE and
    # ignored by every decoder.
    yield ": tts-long stream open\n\n"
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(stream.queue.get(),
                                                       SSE_KEEPALIVE)
            except TimeoutError:
                # Waiting on the model to load, on a job ahead in the queue, or
                # on a long sentence. Concurrent callers serialise: this
                # service generates one job at a time, so a second stream sits
                # here producing keepalives until the first finishes.
                yield ": keepalive\n\n"
                continue
            if kind == "delta":
                yield _frame({"type": "speech.audio.delta",
                              "audio": base64.b64encode(payload).decode("ascii")})
            elif kind == "done":
                yield _frame({"type": "speech.audio.done", "usage": payload})
                return
            else:
                # The only in-band error channel there is once 200 has gone
                # out. openai-python raises APIError(message=...) off this.
                yield _frame({"error": {"message": str(payload),
                                        "type": "server_error",
                                        "param": None,
                                        "code": "synthesis_failed"}})
                return
    finally:
        job = jobs.get(job_id)
        if job is not None and job["status"] in {"queued", "running"}:
            # THE JOB SURVIVES THE SOCKET, and it used not to.
            #
            # Hanging up cancelled the work, on the reasoning that ten more
            # minutes of one CPU for audio nobody is holding a socket for is
            # waste. That is true of a client that has gone away for good, and
            # false of the only client this service actually has: the Jobs tab
            # says "Close this page if you want. The job runs on the server."
            # That promise was already false for anything streamed, and the
            # cheapest way to find out was to close a tab and lose ten minutes.
            #
            # It matters more now that the page streams a cloned voice, which
            # is the longest work here: a reader who starts listening, hears
            # enough and closes the tab would have thrown away the file they
            # were about to be able to download.
            #
            # TTS_SSE_CANCEL_ON_DISCONNECT=1 restores the old behaviour for
            # anyone whose clients really are throwaway. The sweeper still
            # bounds the cost: an abandoned job is finished, written, and
            # expired by JOB_TTL like any other.
            if CANCEL_ON_DISCONNECT:
                job["cancelled"] = True
                log.info("%s client disconnected, cancelling", job_id[:8])
            else:
                log.info("%s client disconnected; the job keeps running and "
                         "its audio will be collectable from /jobs", job_id[:8])
