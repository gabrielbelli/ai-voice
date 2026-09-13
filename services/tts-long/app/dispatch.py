"""Where a job runs, and why this is lanes rather than a queue and a ladder.

WHAT THIS REPLACES AND THE FAILURE IT PREVENTS. One `queue.Queue` fed one
worker thread, and that thread walked a LADDER: it asked the runner whether it
was free, waited for the answer, and only then started work. Two things
followed and both were measured.

  * A RUNNER THAT ANSWERS SLOWLY DELAYED A LOCAL JOB. `RunnerConfig.timeout`
    is thirty seconds, `offer()` was called on the worker's own thread, once
    per job per rung, and it was deliberately not cached. A runner that accepts
    a TCP connection and then says nothing therefore added thirty seconds of
    silence to a job that was always going to run here. With the machine
    switched off it is a refused connection and it is fast; with the machine
    ASLEEP, or wedged, or behind a firewall that drops rather than rejects, it
    is the full timeout. That is INVARIANT L below, and it is a test.
  * THE SECOND JOB WAITED FOR THE FIRST EVEN WHEN THE FIRST HAD LEFT THE
    BUILDING. One worker means one job at a time, so a ten-minute job handed to
    the runner's card held this host's completely idle CPU hostage for the
    whole ten minutes. Measured: 300 seconds of speech is 1304 s here and 437 s
    there, so the second job started twenty-one minutes late for no reason at
    all.

THE SHAPE. `pending` is a deque under one Condition. ONE chooser thread is the
only thing that reads a lane's readiness and the only thing that hands a job to
a lane; each lane has ONE runner thread that does nothing but execute what it
is given. The chooser NEVER makes a network call -- `LaneProbe` does that on
its own thread, and the chooser reads the tuple it left behind. Ten seconds of
staleness costs at most one wrong lane choice, which the yield path already
recovers; a network call on the job's thread costs the job.

WIDTH IS ONE ON BOTH LANES AND LOCAL'S IS NOT CONFIGURABLE. `Synth._speak`
holds one `threading.Lock` across `_ensure_loaded()` and `generate()`, so two
local jobs do not overlap -- they interleave at segment granularity for zero
extra throughput and double the latency of each. Every concurrency this module
delivers comes from the SECOND MACHINE, not from a second thread here.

`runner_cpu` IS NOT A LANE. The server sets TTS_RUNNER_CPU_SERVICE=chatterbox-cpu
and the runner registers only `echo` and `chatterbox`, so that rung has never
been offered a job and never can be; at its measured 0.24x against this host's
0.23x the arithmetic below would refuse it anyway. A lane nothing can reach is
a lane nothing tests.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

# What a lane's execute callback says happened, so the chooser does not have to
# know what a RemoteYield is.
FINISHED = "finished"
YIELDED = "yielded"


@dataclass
class Lane:
    """One place work can go, and everything the chooser knows about it."""

    name: str
    # 1 everywhere, and see the module docstring for why local's is not a knob.
    width: int = 1
    # The fixed cost of the handover: uploading the reference clip, submitting
    # the lease, polling for the first artefact. ADDITIVE, never a multiplier --
    # a multiplicative margin hides a fixed cost, and this one is real.
    hop: float = 0.0
    # Set when this lane hands a job back, so the chooser stops offering it work
    # for a while rather than bouncing the next job off the same busy machine.
    cooldown_until: float = 0.0
    # None on `local`, which is always willing by construction.
    probe: "LaneProbe | None" = None
    # The job this lane is executing, or None. Width 1, so one slot is the whole
    # occupancy model.
    slot: str | None = None
    started_at: float = 0.0
    work: float = 0.0
    thread: threading.Thread | None = None

    def free(self, now: float) -> bool:
        if self.slot is not None:
            return False
        if now < self.cooldown_until:
            return False
        return self.probe is None or self.probe.ok()

    def remaining(self, now: float, rate: float) -> float:
        """Seconds of work still owed by whatever is in this lane."""
        if self.slot is None:
            return 0.0
        total = self.work / max(rate, 1e-3)
        return max(0.0, total - (now - self.started_at))


class LaneProbe:
    """Ask a runner whether it is free, on a thread nobody is waiting on.

    A DELIBERATE REVERSAL of `RunnerClient.offer`'s "not cached, asked once per
    job" rule, and the reversal is the whole point. That rule is right about
    staleness and wrong about who pays for it: the asking used to happen on the
    thread that was about to speak, so an unreachable runner charged its whole
    connect timeout to a job that was going to run here anyway.

    `client()` is a callable rather than a client, because `state["runner"]` is
    rebound -- at startup, and by any test that attaches a fake -- and a probe
    holding the object it was built with would answer about a runner that is no
    longer configured.

    AN ANSWER THAT IS TOO OLD IS NOT AN ANSWER. Three intervals with no reply
    means the probe thread is stuck on a socket, and reporting the last good
    result would route jobs at a machine that stopped talking.
    """

    def __init__(self, client: Callable[[], object | None], interval: float,
                 log, name: str = "runner",
                 services: dict[str, str] | None = None) -> None:
        self._client = client
        self._interval = max(0.01, interval)
        self._log = log
        self._name = name
        # {engine id: the runner's service id for it}. The probe asks about a
        # MACHINE and a job names an ENGINE, and this is the only place the two
        # vocabularies meet. Empty means one engine and one service, which is
        # every deployment that has not enabled a second.
        self._services = dict(services or {})
        self._lock = threading.Lock()
        self.ready = False
        self.why = "not asked yet"
        self.cpu_pct: int | None = None
        # WHAT THE RUNNER SAID ABOUT EVERY SERVICE IT LISTS, not only about the
        # one this client is pinned to. A missing turbo service must shut turbo
        # and nothing else: shutting the whole lane would take the card away
        # from baseline for a service baseline never needed.
        self.by_service: dict[str, tuple[bool, str]] = {}
        self.at = 0.0
        # WHAT WAS SAID LAST TIME, so this logs a CHANGE and not a heartbeat.
        # The probe runs every ten seconds for the life of the process; a line
        # per round would be eight and a half thousand a day saying the same
        # thing, and a log nobody reads is a log that hides the one line that
        # mattered.
        self._said = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"probe-{self._name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.once()
            self._stop.wait(self._interval)

    def once(self) -> None:
        """One round, synchronously. Called by the thread, and by the tests."""
        client = self._client()
        if client is None:
            # Not a failure and not worth a log line: this is the shipped
            # default. No runner is configured, so the lane is simply shut.
            with self._lock:
                self.ready, self.why, self.cpu_pct = False, "no runner is configured", None
                self.at = time.monotonic()
            return
        try:
            offer = client.offer()
        except Exception as exc:  # noqa: BLE001 - a runner that is down is not an error
            with self._lock:
                self.ready = False
                self.why = f"unreachable: {type(exc).__name__}"
                self.cpu_pct = None
                # EMPTIED, NOT LEFT BEHIND. A stale per-service map would let
                # ok_for() answer yes about a machine that has stopped talking,
                # which is the exact staleness `ok()` refuses one line further
                # down.
                self.by_service = {}
                self.at = time.monotonic()
            self._say(f"unreachable: {type(exc).__name__}", warn=False)
            return
        with self._lock:
            self.ready = bool(getattr(offer, "ready", False))
            self.why = getattr(offer, "why", "") or ""
            self.cpu_pct = getattr(offer, "cpu_pct", None)
            self.by_service = dict(getattr(offer, "by_service", None) or {})
            self.at = time.monotonic()
        if self.ready:
            self._say("free", warn=False)
        else:
            # NOT INSTALLED IS NOT THE SAME ANSWER AS BUSY, and the difference
            # is what a person is told rather than where the job goes -- both
            # fall back to this host, so the audio is fine either way and that
            # is exactly why this is easy to get wrong and never notice. "The
            # runner has no speech service" will never clear on its own and is
            # fixed by one command on that machine. "Somebody is at the
            # keyboard" clears in a minute and is nobody's problem. Logging
            # both at the same level trains people to ignore the one that
            # matters.
            self._say(self.why, warn=self.why in {"not_installed", "not_enabled",
                                                  "no_such_service"})

    def _say(self, what: str, *, warn: bool) -> None:
        if what == self._said:
            return
        self._said = what
        if warn:
            self._log.warning("the %s lane is shut: the runner has no usable "
                              "speech service (%s). Fix it with `idlegpu "
                              "service install` on that machine.",
                              self._name, what)
        else:
            self._log.info("the %s lane is now: %s", self._name, what)

    def _fresh(self) -> bool:
        """Was the last answer recent enough to act on. Call under the lock.

        AN ANSWER THAT IS TOO OLD IS NOT AN ANSWER: three intervals with no
        reply means the probe thread is stuck on a socket, and reporting the
        last good result would route jobs at a machine that stopped talking.
        """
        if not self.at:
            return False
        return time.monotonic() - self.at <= 3 * self._interval

    def ok(self) -> bool:
        """Is this lane worth offering ANY job to.

        ANY ENGINE, NOT THIS CLIENT'S ONE. There are two speech services on
        that machine now and they fail independently: a turbo service that was
        never installed must not take the card away from baseline, which is
        installed, enabled and idle. `ready` alone is the pinned service's
        answer, so it is kept as the first clause -- with an older runner, or a
        fake that publishes no per-service map, this is byte for byte the
        condition that shipped. `_eligible` does the per-engine half.
        """
        with self._lock:
            if not self._fresh():
                return False
            if self.ready:
                return True
            return any(ready for ready, _why in self.by_service.values())

    def ok_for(self, engine: str | None) -> bool:
        """Will this runner run THIS engine, right now.

        An engine with no per-service answer falls back to the lane-wide one.
        That is not laxity: a runner that predates the per-service map, or a
        deployment with one engine and one service, has exactly one answer and
        it is `ready`. Inventing a refusal from a field the runner does not
        publish would be this side manufacturing an outage.
        """
        with self._lock:
            if not self._fresh():
                return False
            if not engine or not self.by_service:
                return self.ready or any(
                    ready for ready, _why in self.by_service.values())
            service = self._services.get(engine or "", engine or "")
            found = self.by_service.get(service)
            if found is None:
                return self.ready if service == "" else False
            return found[0]

    def why_for(self, engine: str | None) -> str:
        """Why not, in the runner's own words, for one engine."""
        with self._lock:
            if not self._fresh():
                return self.why or "the runner has not answered recently"
            service = self._services.get(engine or "", engine or "")
            found = self.by_service.get(service)
            if found is None:
                return self.why if service == "" or not self.by_service \
                    else "no_such_service"
            return found[1]

    def snapshot(self) -> dict:
        with self._lock:
            age = round(time.monotonic() - self.at, 1) if self.at else None
            return {"ready": self.ready, "why": self.why,
                    "cpu_pct": self.cpu_pct, "answered_seconds_ago": age,
                    # ONE ROW PER SERVICE, so "why did that job run here" is
                    # answerable per engine rather than per machine. Absent on
                    # a runner that publishes no per-service map.
                    "by_service": {k: {"ready": v[0], "why": v[1]}
                                   for k, v in self.by_service.items()}}


class Dispatcher:
    """The deque, the chooser and one thread per lane.

    Everything about WHAT a job is stays in main.py and reaches here as four
    callables. This module knows how to choose and how to wait; it does not
    know what Chatterbox is, and that is what makes the choosing testable
    without a model.
    """

    def __init__(self, *, execute: Callable[[dict, str], str],
                 job_of: Callable[[str], dict | None],
                 work_of: Callable[[dict], float],
                 rate_of: Callable[[str, str | None], float],
                 finished: Callable[[str], None],
                 log,
                 margin: float = 1.25,
                 cooldown: float = 30.0,
                 pin_streams: bool = True,
                 lane_allows: Callable[[str, str | None], bool] | None = None,
                 expire: Callable[[str, float], None] | None = None,
                 stranded_deadline: float = 900.0) -> None:
        self._execute = execute
        self._job_of = job_of
        self._work_of = work_of
        self._rate_of = rate_of
        self._finished = finished
        self._log = log
        self._margin = margin
        self._cooldown = cooldown
        self._pin_streams = pin_streams
        # WHETHER A LANE COULD EVER CARRY THIS ENGINE, which is a different
        # question from whether it is free. `Lane.free()` and the probe both
        # answer "right now"; this one answers "at all", and it exists because
        # the local lane has no probe and was therefore willing to take
        # anything -- including an engine with no local implementation, whose
        # first job would have imported None and failed after a job id already
        # existed. Default: every lane takes everything, which is what one
        # engine with a CPU path has always meant.
        self._lane_allows = lane_allows or (lambda lane, engine: True)
        # WHAT TO DO WITH A JOB NO LANE CAN CARRY. See `_stranded_since`.
        self._expire = expire
        self._stranded_deadline = stranded_deadline
        # WHEN EACH STRANDED JOB WAS FIRST FOUND WITH NO CAPABLE LANE.
        #
        # THE CLOCK COUNTS TIME WITH NO LANE, NEVER TIME IN THE QUEUE, and the
        # distinction is the whole reason this is a dict rather than a
        # comparison against `created_at`. A legitimately long render behind
        # another legitimately long render is a job that has waited an hour and
        # is perfectly well; a job whose only machine went home is a job
        # nothing will ever pick up. Only the second has an entry here, and an
        # entry is dropped the instant a lane comes back.
        self._stranded_since: dict[str, float] = {}
        self._cond = threading.Condition()
        self._pending: deque[str] = deque()
        self._lanes: dict[str, Lane] = {}
        # WHICH LANES HAVE ALREADY HANDED THIS JOB BACK. Without it a runner
        # that yields instantly and cools for thirty seconds would still be
        # offered the same job again the moment the cooldown lapsed, and a job
        # could bounce between the two for ever. `local` cannot yield, so the
        # exclusion set is what bounds the walk.
        self._refused: dict[str, set[str]] = {}
        self._stopping = False
        self._chooser: threading.Thread | None = None

    # -- construction -------------------------------------------------------

    def add_lane(self, name: str, *, hop: float = 0.0,
                 probe: LaneProbe | None = None) -> Lane:
        lane = Lane(name=name, hop=hop, probe=probe)
        self._lanes[name] = lane
        return lane

    @property
    def lanes(self) -> dict[str, Lane]:
        return self._lanes

    def start(self) -> None:
        for lane in self._lanes.values():
            if lane.probe is not None:
                lane.probe.start()
            lane.thread = threading.Thread(target=self._lane_loop, args=(lane,),
                                           daemon=True, name=f"lane-{lane.name}")
            lane.thread.start()
        self._chooser = threading.Thread(target=self._choose_loop, daemon=True,
                                         name="chooser")
        self._chooser.start()

    # -- the queue, as everything outside sees it ---------------------------

    def submit(self, job_id: str) -> None:
        with self._cond:
            self._pending.append(job_id)
            self._cond.notify_all()

    def depth(self) -> int:
        """Work accepted and not yet finished: waiting PLUS in flight.

        `queue.qsize()` counted only the waiting half, so `_full()` admitted
        MAX_QUEUE jobs on top of however many were running -- the ceiling was
        never the ceiling.
        """
        with self._cond:
            return len(self._pending) + sum(1 for l in self._lanes.values()
                                            if l.slot is not None)

    def position(self, job_id: str) -> int:
        """How many jobs are in front of this one. -1 once it has left the deque."""
        with self._cond:
            try:
                return self._pending.index(job_id)
            except ValueError:
                return -1

    def running(self) -> list[str]:
        with self._cond:
            return [l.slot for l in self._lanes.values() if l.slot is not None]

    # -- the arithmetic -----------------------------------------------------

    def finish(self, lane: Lane, work: float, now: float | None = None,
               engine: str | None = None) -> float:
        """When this lane would be done with `work` seconds of speech.

        ADDITIVE, and the handover is a term rather than a fudge factor:
        remaining work in the lane, plus the speech divided by the rate that
        lane actually achieves, plus the fixed cost of getting there.

        PER LANE AND PER ENGINE, because the two engines on that card are
        2.36x apart and one average over both describes neither. It is the same
        argument the per-backend split already makes one level up: a figure
        that mixes a 1.54x renderer with a 0.65x one accepts a synchronous
        request the slow one can never finish, at exactly the worst moment.

        `lane.remaining` still divides by the rate of the engine being ASKED
        ABOUT rather than the one occupying the lane. That is a known
        approximation and the direction is safe: the lane's occupant is
        recomputed to nothing the moment it finishes, and the alternative --
        holding the occupant's engine on the Lane -- buys accuracy in a term
        that is already an estimate.
        """
        now = time.monotonic() if now is None else now
        rate = self._rate_of(lane.name, engine)
        return lane.remaining(now, rate) + work / max(rate, 1e-3) + lane.hop

    def estimate_for(self, work: float,
                     engine: str | None = None) -> tuple[str, float]:
        """Where this much speech would go, and how long it would take there.

        The same argmin the chooser runs, WITHOUT taking a lease. Every quote
        this service gives used to divide by the LOCAL rate whatever the
        destination, so a job bound for the runner's card was promised
        twenty-one minutes and delivered in seven -- and the caller who was
        told twenty-one minutes went away.
        """
        now = time.monotonic()
        best, best_at = "local", None
        for name, lane in self._lanes.items():
            # A LANE WITH NO IMPLEMENTATION OF THIS ENGINE IS NOT A CANDIDATE
            # AND MUST NOT BE QUOTED. Without this the local lane -- which has
            # no probe and is therefore always considered -- answers with a
            # rate nobody ever measured, and that number goes out in a 202's
            # `estimated_seconds` and sizes somebody's progress bar.
            if not self._lane_allows(name, engine):
                continue
            if lane.probe is not None:
                # NAMED ENGINE, NAMED ANSWER. With no engine this is the old
                # question -- is this lane open at all -- which is what the
                # queue-wide quotes (`_retry_after`, `_pending_work`) are
                # asking. With one, the runner may be perfectly free and still
                # have no service for it.
                open_ = (lane.probe.ok_for(engine) if engine
                         else lane.probe.ok())
                if not open_:
                    continue
            at = self.finish(lane, work, now, engine)
            if best_at is None or at < best_at:
                best, best_at = name, at
        if best_at is None:
            # EVERY CANDIDATE LANE IS SHUT. Quote the best of them anyway, shut
            # or not, because a quote is what a caller was already promised and
            # "how long once it opens" is the only honest answer left. Falling
            # back to `local` unconditionally -- which this did -- would quote
            # a lane that cannot run this engine at all.
            fallback = next((l for name, l in self._lanes.items()
                             if self._lane_allows(name, engine)), None)
            return ((fallback.name, self.finish(fallback, work, now, engine))
                    if fallback is not None else ("local", work))
        return best, best_at

    # -- choosing -----------------------------------------------------------

    def _capable(self, job: dict, job_id: str) -> list[Lane]:
        """Every lane that could carry this job at all, free or not.

        THE DIFFERENCE BETWEEN "BUSY" AND "NEVER" IS THE WHOLE POINT OF THIS
        FUNCTION, and `_assign` acts on it in opposite ways. A head job whose
        lanes are merely busy must block the queue -- nothing behind it could
        move either, and letting it be overtaken would turn the deque into a
        lottery. A head job that NO lane can carry must not block anything: it
        is waiting on a machine that has gone home, and thirty-two of those on
        a shared 32-slot queue answer 429 to every caller of an engine that
        runs perfectly well on the lane sitting idle beside them.
        """
        refused = self._refused.get(job_id, ())
        engine = job.get("engine")
        # `engine` GUARDS THE PROBE EXACTLY AS IT DOES IN `_eligible`. With no
        # engine named this is the question that has always been asked -- is
        # this lane's machine there at all -- and a lane that is merely shut
        # still COULD carry the job, so it blocks the queue rather than
        # stranding it. Only a named engine the runner does not register makes
        # a lane incapable rather than busy.
        return [l for l in self._lanes.values()
                if l.name not in refused
                and self._lane_allows(l.name, engine)
                and not (l.probe is not None and engine
                         and not l.probe.ok_for(engine))]

    def _eligible(self, job: dict, job_id: str, now: float) -> list[Lane]:
        refused = self._refused.get(job_id, ())
        engine_wanted = job.get("engine")
        lanes = [l for l in self._lanes.values()
                 if l.name not in refused and l.free(now)
                 # AND THIS LANE HAS AN IMPLEMENTATION OF THIS ENGINE. `free()`
                 # cannot ask: local has no probe and is always willing, so
                 # without this a runner-only engine is handed to a lane that
                 # would import None.
                 and self._lane_allows(l.name, engine_wanted)]
        # THE LANE IS OPEN AND THIS ENGINE IS NOT NECESSARILY ON IT. `free()`
        # asks whether the machine will take work at all; a job names one
        # engine, and the runner carries one speech service per engine. Without
        # this line a turbo job is dispatched at a runner that has no turbo
        # service, comes back `no_such_service`, and bounces -- once per job,
        # for ever, on a lane that was working perfectly for baseline.
        engine = job.get("engine")
        if engine:
            lanes = [l for l in lanes
                     if l.probe is None or l.probe.ok_for(engine)]
        if self._pin_streams and job.get("stream") is not None:
            # A STREAMED JOB IS PINNED HERE, AND IT IS A REFUSAL TO BUILD
            # RESUMABLE STREAMING. A RemoteYield with `delivered > 0` on a
            # stream cannot be re-run: `_run` re-enters with a fresh encoder and
            # a fresh `offsets`, so the client would get a second file header in
            # the middle of the first file. The job is failed instead. Never
            # reaching that state is cheaper than engineering around it.
            lanes = [l for l in lanes if l.name == "local"]
        return lanes

    def _pick(self, job: dict, job_id: str, now: float) -> Lane | None:
        free = self._eligible(job, job_id, now)
        if not free:
            return None
        work = self._work_of(job)
        engine = job.get("engine")
        scored = [(self.finish(lane, work, now, engine), lane) for lane in free]
        local = next((l for _, l in scored if l.name == "local"), None)
        if local is not None:
            here = self.finish(local, work, now, engine)
            # THE ASYMMETRIC MARGIN, and it is asymmetric on purpose. Losing a
            # remote lane costs a WHOLE RE-SPEAK -- `_run` re-enters with a
            # fresh offsets list and the segments already made on the runner are
            # kept but the local rung starts again. Losing a local lane costs
            # only slowness. So a remote lane has to be clearly better, not
            # marginally better, before the job crosses the network.
            scored = [(at, lane) for at, lane in scored
                      if lane.name == "local" or at * self._margin < here]
        scored.sort(key=lambda pair: (pair[0], pair[1].name != "local"))
        return scored[0][1]

    def _assign(self, now: float) -> bool:
        """Hand at most one waiting job to one lane. True if anything moved."""
        skipped = 0
        while len(self._pending) > skipped:
            job_id = self._pending[skipped]
            job = self._job_of(job_id)
            if job is None:
                # Popped between submit and here -- DELETE and the sweeper both
                # remove jobs. Dropping it here rather than letting the lane
                # thread trip over a None is what keeps a missing job from
                # costing a lane.
                del self._pending[skipped]
                self._refused.pop(job_id, None)
                self._stranded_since.pop(job_id, None)
                continue
            if not self._capable(job, job_id):
                # NO LANE CAN CARRY THIS ONE, so it must not hold the queue.
                # See `_capable`: this is the machine having gone home, not a
                # machine being busy, and the two have opposite right answers.
                if self._strand(job_id, job, now):
                    continue
                skipped += 1
                continue
            self._stranded_since.pop(job_id, None)
            lane = self._pick(job, job_id, now)
            if lane is None:
                # Capable lanes exist and every one of them is busy or cooling.
                # Nothing behind this job could move either, so the queue waits
                # rather than letting a later job overtake a job that has
                # already waited its turn.
                return False
            del self._pending[skipped]
            lane.slot = job_id
            lane.started_at = now
            lane.work = self._work_of(job)
            return True
        return False

    def _strand(self, job_id: str, job: dict, now: float) -> bool:
        """Start or check this job's no-lane clock. True if it was expired.

        A JOB WAITING ON A MACHINE THAT NEVER COMES BACK NEEDS AN ANSWER AND
        THE ONE IT HAD WAS SILENCE. `_pick` returned None and looped with no
        deadline, and `_sweep` skips anything whose `finished_at` is None, so
        the job sat `queued` until the process died: a progress bar that never
        moves, a queue slot held for ever, and -- on a deployment-wide ceiling
        of thirty-two -- 429 to every caller of a working engine.

        Terminal, with the runner's own reason in the sentence, so somebody
        reading the row learns which machine and why rather than "failed".
        """
        started = self._stranded_since.setdefault(job_id, now)
        waited = now - started
        if self._expire is None or waited < self._stranded_deadline:
            return False
        del self._pending[self._pending.index(job_id)]
        self._stranded_since.pop(job_id, None)
        self._refused.pop(job_id, None)
        self._log.warning("%s: no lane has been able to run %s for %.0fs; "
                          "failing it rather than holding a queue slot",
                          job_id[:8], job.get("engine"), waited)
        self._expire(job_id, waited)
        self._finished(job_id)
        return True

    def stranded_for(self, job_id: str, now: float | None = None) -> float:
        """How long this job has had no lane at all. 0.0 if it has one.

        Published so a poller can tell "queued behind other work" from "queued
        with nothing able to run it", which read identically before.
        """
        started = self._stranded_since.get(job_id)
        if started is None:
            return 0.0
        return max(0.0, (now if now is not None else time.monotonic()) - started)

    def _choose_loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopping and not self._assign(time.monotonic()):
                    # A BOUNDED WAIT, NOT AN OPEN ONE. Two of the three things
                    # that can make a shut lane open again -- a cooldown
                    # lapsing and a probe answering -- happen on other threads
                    # and do not touch this Condition. Waking anyway is what
                    # stops a job sitting in the deque with an idle runner
                    # beside it.
                    self._cond.wait(0.25)
                if self._stopping:
                    # UNCONDITIONALLY, even with jobs still waiting. Testing
                    # `and not self._pending` here reads as tidier and spins
                    # this thread at a hundred per cent of a core for the whole
                    # of a shutdown that still has a deque behind it, because
                    # `_assign` refuses and the loop above exits on `stopping`
                    # rather than waiting. Jobs left in the deque never started
                    # and are the caller's to mark.
                    return
                self._cond.notify_all()

    # -- running ------------------------------------------------------------

    def _lane_loop(self, lane: Lane) -> None:
        while True:
            with self._cond:
                while lane.slot is None and not self._stopping:
                    self._cond.wait()
                if lane.slot is None:
                    return
                job_id = lane.slot
            self._work(lane, job_id)

    def _work(self, lane: Lane, job_id: str) -> None:
        outcome = FINISHED
        try:
            job = self._job_of(job_id)
            if job is not None:
                outcome = self._execute(job, lane.name)
        except Exception:  # noqa: BLE001 - a lane must survive its own job
            # THE FAILURE THIS PREVENTS IS A SERVICE THAT GOES QUIET. The old
            # worker caught its exceptions inside the loop for exactly this
            # reason: anything escaping killed the one thread there was, and
            # every later job then sat `queued` for ever with nothing to run it
            # and no error anywhere. With lanes it would kill one lane silently,
            # which is worse -- the service would look half-well.
            self._log.exception("%s: the %s lane raised", job_id[:8], lane.name)
        finally:
            with self._cond:
                lane.slot = None
                lane.work = 0.0
                if outcome == YIELDED:
                    # BACK TO THE HEAD, not the tail. This job has already
                    # waited its turn once; sending it to the back would let
                    # every job submitted since overtake it because a machine
                    # somewhere else got busy.
                    self._pending.appendleft(job_id)
                    self._refused.setdefault(job_id, set()).add(lane.name)
                    lane.cooldown_until = time.monotonic() + self._cooldown
                else:
                    self._refused.pop(job_id, None)
                self._cond.notify_all()
            if outcome != YIELDED:
                self._finished(job_id)

    # -- shutdown -----------------------------------------------------------

    def drain(self, grace: float) -> bool:
        """Stop taking work, let the lanes finish, and say whether they did.

        THE MODEL MUST NOT CLOSE UNDER A RUNNING JOB. `lifespan` used to put one
        sentinel on the queue, join nothing, and call `synth.close()` straight
        after -- so a shutdown during synthesis tore the model out from under
        the thread using it. Returns False when the grace ran out, and the
        caller marks what is left `cancelled`, never `failed`: a service that
        was shutting down is not a synthesis that went wrong.

        THE DEQUE IS PART OF THE ANSWER AND IT WAS NOT IN IT. This read only
        the lane slots, so a shutdown with work still waiting reported exactly
        like a shutdown with nothing left to do, and the caller -- which marks
        nothing when this says True -- left every waiting job `queued` for ever
        with no error anywhere. The job that reaches that state most often is
        one the runner hands back DURING the drain: `_work` returns it to the
        head of the deque deliberately without waking the chooser, and by then
        the chooser has already returned on `_stopping`. `_choose_loop` says
        those jobs "are the caller's to mark"; this is how the caller is told.
        """
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for lane in self._lanes.values():
            if lane.probe is not None:
                lane.probe.stop()
        deadline = time.monotonic() + grace
        for lane in self._lanes.values():
            if lane.thread is not None:
                lane.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._chooser is not None:
            self._chooser.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._cond:
            # AN ID THAT NO LONGER NAMES A JOB IS NOT WORK. DELETE and the
            # sweeper both remove jobs while they sit in the deque, and
            # `_assign` drops those ids when it reaches them -- so counting the
            # deque's length would report a dirty shutdown over rows nobody can
            # be told about, and print a warning naming jobs that are gone.
            waiting = any(self._job_of(job_id) is not None
                          for job_id in self._pending)
            return (not waiting
                    and not any(l.slot is not None for l in self._lanes.values()))

    def snapshot(self) -> dict:
        """What /health publishes: one row per lane, plus what is waiting."""
        now = time.monotonic()
        with self._cond:
            return {
                "waiting": len(self._pending),
                "lanes": {
                    name: {
                        "width": lane.width,
                        "busy": lane.slot is not None,
                        "free": lane.free(now),
                        "cooling_for": (round(lane.cooldown_until - now, 1)
                                        if lane.cooldown_until > now else 0.0),
                        "probe": lane.probe.snapshot() if lane.probe else None,
                    }
                    for name, lane in self._lanes.items()
                },
            }
