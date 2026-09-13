"""Two lanes, and the four ways one machine used to hold up the other.

NOTHING HERE OPENS A SOCKET AND NOTHING HERE LOADS A MODEL. Half of these
drive `Dispatcher` directly with stub callables, because the choosing is
arithmetic and deserves to be read as arithmetic; the other half go through the
real routes with a fake runner attached, because the two failures that hurt --
a slow runner delaying a local job, and a second job waiting for a first that
had left the building -- are only visible end to end.

Every test is named after the mistake it prevents.
"""

from __future__ import annotations

import threading
import time

from app.dispatch import FINISHED, YIELDED, Dispatcher, LaneProbe
from app.main import DISPATCH_MARGIN, RUNNER_COOLDOWN_S, RUNNER_HOP_S
from app.remote import RunnerConfig

from test_remote import FakeClient, runner


# ------------------------------------------------------------------ stubs ---


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def exception(self, *a, **k): pass


class _Probe:
    """A LaneProbe as far as Lane.free can tell, with the answer set by hand.

    `services` is {engine: ready}, empty for the runner that carries one. An
    engine with no entry falls back to the machine-wide answer, which is what a
    runner publishing no per-service map really means.
    """

    def __init__(self, ready=True, services=None):
        self.ready = ready
        self.services = dict(services or {})

    def start(self):
        pass

    def stop(self):
        pass

    def ok(self):
        return self.ready or any(self.services.values())

    def ok_for(self, engine):
        if not engine or engine not in self.services:
            return self.ok()
        return self.services[engine]

    def why_for(self, engine):
        return "" if self.ok_for(engine) else "no_such_service"

    def snapshot(self):
        return {"ready": self.ready}


def _dispatcher(*, rates=None, work=1.0, execute=None, jobs=None,
                hop=RUNNER_HOP_S, margin=DISPATCH_MARGIN,
                cooldown=RUNNER_COOLDOWN_S, runner_ready=True,
                pin_streams=True, lane_allows=None, expire=None,
                stranded_deadline=900.0):
    """A Dispatcher over stubs, with the two lanes the service really has.

    THE THREE NUMBERS COME OUT OF app.main, and they used to be typed again
    here. That is not tidiness: every worked example below is arithmetic on the
    shipped margin and the shipped handover, so a helper that restated 1.25 and
    8.0 tested this file against itself. Measured -- main.py's margin was
    mutated to 1.0 and its hop to 0.0, and the whole suite passed both times,
    which means the two constants the routing rests on had no test at all.
    """
    rates = rates or {"local": 0.23, "runner": 0.70}
    jobs = jobs if jobs is not None else {}
    done: list[str] = []
    d = Dispatcher(
        execute=execute or (lambda job, lane: FINISHED),
        job_of=jobs.get,
        work_of=lambda job: job.get("work", work),
        rate_of=lambda name, engine=None: rates[name],
        finished=done.append,
        log=_Log(), margin=margin, cooldown=cooldown, pin_streams=pin_streams,
        # DEFAULT: EVERY LANE TAKES EVERYTHING, which is what one engine with a
        # CPU path has always meant and what every test above this line is
        # about. A test that wants a runner-only engine says so.
        lane_allows=lane_allows or (lambda lane, engine: True),
        expire=expire, stranded_deadline=stranded_deadline)
    d.add_lane("local")
    d.add_lane("runner", hop=hop, probe=_Probe(runner_ready))
    d.finished_ids = done            # type: ignore[attr-defined]
    d.job_table = jobs               # type: ignore[attr-defined]
    return d


def _settle(predicate, timeout=5.0, why="the dispatcher never got there"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(why)


# --------------------------------------------------------- the arithmetic ---


def test_the_worked_example_sends_a_long_job_to_the_card():
    """THE NUMBER THE WHOLE DESIGN RESTS ON, asserted rather than asserted-to.

    300 seconds of speech, both lanes idle, at the two measured rates: 1304 s
    here against 429 s there plus an 8 s handover. 437 x 1.25 = 546, still
    comfortably under 1304, so the job crosses the network.
    """
    d = _dispatcher()
    now = time.monotonic()
    local, remote = d.lanes["local"], d.lanes["runner"]
    assert round(d.finish(local, 300.0, now)) == 1304
    assert round(d.finish(remote, 300.0, now)) == 437
    assert d._pick({"work": 300.0}, "j", now).name == "runner"


def test_a_short_job_stays_here_because_the_handover_is_a_real_cost():
    """THE DEFECT THIS PREVENTS: a margin that hides a fixed cost.

    A second of speech is 4.3 s here and 1.4 s of compute there -- but getting
    it there costs an 8 s handover, so it arrives later. A MULTIPLICATIVE
    margin cannot express that: 1.4 x 1.25 is under 4.3 and the job would go.
    The handover is a term, not a factor, which is the whole reason the rule is
    written `remaining + A/r + HOP` and not `A/r x something`.
    """
    d = _dispatcher()
    now = time.monotonic()
    assert d._pick({"work": 1.0}, "j", now).name == "local"


def test_a_marginal_gain_is_not_worth_a_whole_re_speak():
    """The margin is asymmetric ON PURPOSE and this is the case it exists for.

    Losing a remote lane mid-job costs a whole re-speak -- _run re-enters with
    a fresh encoder and a fresh offsets list. Losing the local one costs only
    slowness. So a remote lane that is barely better does not get the job.
    """
    # Same rate on both machines: the remote one can only ever be worse, and it
    # is worse by exactly the handover.
    d = _dispatcher(rates={"local": 0.23, "runner": 0.23})
    assert d._pick({"work": 300.0}, "j", time.monotonic()).name == "local"
    # And a runner only 20% faster does not clear 1.25 either.
    d = _dispatcher(rates={"local": 0.23, "runner": 0.276}, hop=0.0)
    assert d._pick({"work": 300.0}, "j", time.monotonic()).name == "local"


def test_an_idle_local_lane_is_never_held_open_for_a_faster_machine():
    """THE DEFECT THIS PREVENTS is the one the whole feature exists to remove.

    The runner is busy with somebody else's job. A chooser that waits for the
    better lane leaves this host's CPU idle for the length of a job it could
    have finished -- which is exactly what one worker and a ladder did, for
    twenty-one minutes, measured.
    """
    d = _dispatcher()
    d.lanes["runner"].slot = "somebody-elses-job"
    picked = d._pick({"work": 300.0}, "j", time.monotonic())
    assert picked is not None and picked.name == "local"


def test_a_shut_runner_lane_is_not_a_reason_to_stop_working():
    d = _dispatcher(runner_ready=False)
    assert d._pick({"work": 300.0}, "j", time.monotonic()).name == "local"


def test_a_quote_is_about_the_lane_the_job_will_actually_go_to():
    """THE DEFECT THIS PREVENTS: a caller told twenty-one minutes for a job
    that takes seven, and going away.

    Every estimate this service gives -- the 202's `estimated_seconds`, the
    Retry-After on a 429, the synchronous budget -- divided by the LOCAL rate
    whatever the destination.
    """
    d = _dispatcher()
    lane, seconds = d.estimate_for(300.0)
    assert lane == "runner"
    assert 400 < seconds < 500, seconds
    # And with the runner shut it is honest about this host instead.
    d.lanes["runner"].probe.ready = False
    lane, seconds = d.estimate_for(300.0)
    assert lane == "local"
    assert 1250 < seconds < 1350, seconds


# ------------------------------------------------------- yields and stalls ---


def test_a_yield_returns_the_job_to_the_head_and_cools_the_lane():
    """A YIELD IS NOT A FAILURE and it is not the back of the queue either.

    The runner's owner came back, which the design calls normal. The job has
    already waited its turn once; sending it to the tail would let everything
    submitted since overtake it because somebody else started a game. And the
    lane that just said no must not be offered the next job one millisecond
    later, or a queue becomes a stampede against a machine in use.
    """
    jobs = {"a": {"id": "a", "work": 300.0}}
    calls: list[str] = []

    def execute(job, lane):
        calls.append(lane)
        return YIELDED if lane == "runner" else FINISHED

    d = _dispatcher(execute=execute, jobs=jobs)
    d.start()
    d.submit("a")
    _settle(lambda: calls[:1] == ["runner"], why="it never reached the runner")
    _settle(lambda: calls == ["runner", "local"],
            why="the yielded job did not come back to a lane")
    assert d.lanes["runner"].cooldown_until > time.monotonic(), \
        "the lane that handed the job back was offered work again at once"
    assert d.finished_ids == ["a"], "a yield woke the caller as if it had ended"
    d.drain(2.0)


def test_a_job_that_yields_is_never_offered_the_same_lane_twice():
    """The bound on the walk, and it is not the cooldown.

    A cooldown expires. Without a per-job record of which lanes have already
    handed THIS job back, a runner that yields instantly would be offered the
    same job again thirty seconds later, for ever, and the job would never run.
    `local` cannot yield, so the exclusion set is what terminates the walk.
    """
    jobs = {"a": {"id": "a", "work": 300.0}}
    calls: list[str] = []

    def execute(job, lane):
        calls.append(lane)
        return YIELDED if lane == "runner" else FINISHED

    d = _dispatcher(execute=execute, jobs=jobs, cooldown=0.0)
    d.start()
    d.submit("a")
    _settle(lambda: d.finished_ids == ["a"])
    assert calls == ["runner", "local"], calls
    d.drain(2.0)


def test_a_lane_survives_a_job_that_raises():
    """THE FAILURE THIS PREVENTS IS A SERVICE THAT LOOKS HALF-WELL.

    Anything escaping the old single worker killed the only thread there was
    and every later job sat `queued` for ever with no error anywhere. With two
    lanes the same mistake kills ONE of them silently, which is worse: the
    service keeps answering, at half speed, and nothing says why.
    """
    jobs = {"a": {"id": "a", "work": 1.0}, "b": {"id": "b", "work": 1.0}}
    seen: list[str] = []

    def execute(job, lane):
        seen.append(job["id"])
        if job["id"] == "a":
            raise RuntimeError("the model fell over")
        return FINISHED

    d = _dispatcher(execute=execute, jobs=jobs, runner_ready=False)
    d.start()
    d.submit("a")
    d.submit("b")
    _settle(lambda: seen == ["a", "b"], why="the lane died on the first job")
    d.drain(2.0)


def test_a_job_that_vanished_costs_the_chooser_nothing():
    """DELETE and the sweeper both pop from `jobs`. An id with no row behind it
    is dropped in the chooser, before a lane is ever given it -- a lane that
    trips over a missing job is a lane out of action."""
    jobs = {"b": {"id": "b", "work": 1.0}}
    seen: list[str] = []
    d = _dispatcher(execute=lambda job, lane: seen.append(job["id"]) or FINISHED,
                    jobs=jobs, runner_ready=False)
    d.start()
    d.submit("gone")
    d.submit("b")
    _settle(lambda: seen == ["b"])
    d.drain(2.0)


# ------------------------------------------------------------- the ceiling ---


def test_the_ceiling_counts_what_is_running_as_well_as_what_is_waiting():
    """`queue.qsize()` counted only the waiting half, so the service admitted
    MAX_QUEUE jobs on TOP of whatever the lanes were doing and the ceiling was
    never the ceiling."""
    held = threading.Event()
    jobs = {"a": {"id": "a", "work": 1.0}}
    d = _dispatcher(execute=lambda job, lane: held.wait(5) and FINISHED,
                    jobs=jobs, runner_ready=False)
    d.start()
    d.submit("a")
    try:
        _settle(lambda: d.lanes["local"].slot == "a")
        assert d.depth() == 1, "a running job stopped being counted"
        assert d.position("a") == -1, "and it is no longer ahead of anybody"
    finally:
        held.set()
        d.drain(2.0)


# --------------------------------------------------------------- shutdown ---


def test_shutdown_stops_every_lane_before_the_model_closes():
    """THE DEFECT THIS PREVENTS: closing a 6.5 GB model under the thread using
    it.

    `lifespan` put ONE sentinel on the queue, joined nothing, and called
    `synth.close()` on the next line. Whatever that produced was neither logged
    nor recoverable. drain() waits for every lane and SAYS whether they all
    finished, so the caller can mark what is left `cancelled` -- never
    `failed`, which means the synthesis went wrong.
    """
    running = threading.Event()
    release = threading.Event()
    jobs = {"a": {"id": "a", "work": 1.0}}

    def execute(job, lane):
        running.set()
        release.wait(5)
        return FINISHED

    d = _dispatcher(execute=execute, jobs=jobs, runner_ready=False)
    d.start()
    d.submit("a")
    running.wait(5)
    assert d.drain(0.2) is False, "it claimed the lanes were clear while one ran"
    release.set()
    assert d.drain(5.0) is True


def test_shutdown_with_a_queue_behind_it_does_not_spin():
    """A `while` that exits on `stopping` and then loops without waiting burns a
    whole core for the length of a shutdown that still has a deque behind it,
    silently. The chooser returns instead."""
    d = _dispatcher(jobs={}, runner_ready=False)
    d.start()
    for i in range(5):
        d.submit(f"never-created-{i}")
    started = time.monotonic()
    assert d.drain(2.0) is True
    assert time.monotonic() - started < 1.5


# ------------------------------------------------------------- the probe ---


def test_an_answer_that_is_too_old_is_not_an_answer():
    """A probe thread stuck on a socket must SHUT its lane, not leave the last
    good answer standing. Routing jobs at a machine that stopped talking is the
    failure the freshness check exists to prevent."""
    client = FakeClient()
    probe = LaneProbe(lambda: client, 0.05, _Log())
    probe.once()
    assert probe.ok() is True
    probe.at = time.monotonic() - 10.0
    assert probe.ok() is False


def test_a_runner_that_is_not_configured_is_an_ordinary_shut_lane():
    probe = LaneProbe(lambda: None, 1.0, _Log())
    probe.once()
    assert probe.ok() is False
    assert "no runner" in probe.why


def test_the_probe_says_a_thing_once_and_not_every_ten_seconds(caplog):
    """A line per round is eight and a half thousand a day saying the same
    thing, and a log nobody reads hides the one line that mattered."""
    client = FakeClient()
    client.state = (False, "not_installed")
    probe = LaneProbe(lambda: client, 1.0, __import__("logging").getLogger("probe-test"))
    with caplog.at_level("INFO"):
        probe.once()
        probe.once()
        probe.once()
    said = [r for r in caplog.records if "probe-test" == r.name]
    assert len(said) == 1, [r.getMessage() for r in said]


# ------------------------------------------------ through the real routes ---


def _wait(client, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


class _Slow(FakeClient):
    """A runner that accepts the connection and then says nothing.

    The expensive case, and not the obvious one: a machine that is SWITCHED OFF
    refuses the connection and is fast. A machine that is asleep, wedged, or
    behind a firewall that drops rather than rejects holds the socket for the
    whole timeout.
    """

    def __init__(self, seconds=30.0, **kw):
        super().__init__(**kw)
        self.seconds = seconds
        self.offers = 0

    def offer(self):
        self.offers += 1
        time.sleep(self.seconds)
        return super().offer()

    def speech_state(self):
        self.offers += 1
        time.sleep(self.seconds)
        return self.state


def test_a_runner_that_answers_slowly_never_delays_a_local_job(speech):
    """INVARIANT L, AND IT IS THE REASON THIS MODULE EXISTS.

    `RunnerConfig.timeout` is thirty seconds and `offer()` used to be called on
    the worker's own thread, once per job, deliberately uncached. So a runner
    that accepts a TCP connection and then says nothing added THIRTY SECONDS OF
    SILENCE to a job that was always going to run on this host -- and the only
    evidence anywhere was a job that took half a minute longer than it should.

    Nothing on the path from POST to the first sample may make a network call.
    """
    import app.main as main

    slow = _Slow(seconds=30.0)
    main.state["runner"] = slow
    # The probe is genuinely stuck on the socket, which is the state being
    # asserted about -- not merely "the probe has not run yet".
    threading.Thread(target=main.dispatch.lanes["runner"].probe.once,
                     daemon=True).start()
    try:
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        done = _wait(speech, created["id"], timeout=15.0)
        assert done["status"] == "done"
        assert done["started_at"] - done["created_at"] < 1.0, (
            "the job waited on a network call before it started: "
            f"{done['started_at'] - done['created_at']:.1f}s")
        assert done["backend"] == "local"
    finally:
        main.state["runner"] = None


def test_every_job_finishes_with_the_runner_refusing_every_call(speech):
    """A DEAD RUNNER MUST NOT TAX THE QUEUE.

    One offer per job per rung meant the cost of a runner that is off scaled
    with the amount of work, which is precisely backwards: the busier this
    service is, the more it should be getting on with. The probe asks on its
    own schedule, so twenty jobs cost at most a handful of calls.
    """
    import app.main as main

    class Refuses(FakeClient):
        def __init__(self):
            super().__init__()
            self.offers = 0

        def speech_state(self):
            self.offers += 1
            return False, "machine_busy"

    dead = Refuses()
    main.state["runner"] = dead
    try:
        ids = [speech.post("/jobs", json={"text": f"Line number {i}.",
                                          "voice": "default"}).json()["id"]
               for i in range(20)]
        for job_id in ids:
            done = _wait(speech, job_id)
            assert done["status"] == "done", done.get("error")
            assert done["backend"] == "local"
        assert dead.offers < 20, (
            f"{dead.offers} offers for 20 jobs: the runner is being asked once "
            "per job again, on the job's own thread")
    finally:
        main.state["runner"] = None


def test_a_second_job_starts_while_the_first_is_on_the_runner(speech):
    """THE WIN, ASSERTED. It is the only number in the design that changes
    what somebody experiences.

    One worker meant one job anywhere, so a ten-minute job handed to the
    runner's card held this host's completely idle CPU for the whole ten
    minutes and the next job started twenty-one minutes late. Measured, at the
    real rates, on 300 seconds of speech.
    """
    holding = threading.Event()

    class Holds(FakeClient):
        def job(self, job_id):
            holding.set()
            time.sleep(0.05)
            return super().job(job_id)

    with runner(Holds(cfg=RunnerConfig(host="h", service="chatterbox", poll=0.0))):
        long = speech.post("/jobs", json={"text": "A long line. " * 400,
                                          "voice": "default"}).json()
        assert holding.wait(10), "the first job never reached the runner"
        second = speech.post("/jobs", json={"text": "One short line.",
                                            "voice": "default"}).json()
        done = _wait(speech, second["id"], timeout=20.0)

    assert done["backend"] == "local", "the second job queued behind the first"
    assert done["started_at"] - done["created_at"] < 2.0, (
        "the second job waited for a lane that was not this one")
    _wait(speech, long["id"], timeout=30.0)


def test_a_runner_that_lies_about_being_ready_still_finishes_the_job(speech):
    """It says it is free and then never starts the work. THE JOB ENDS `done`.

    Marking it `failed` would be the easy reading and the wrong one: nothing
    about the synthesis went wrong, another machine simply did not do it. The
    record says where it had been, which is the only way anybody finds out that
    the runner is lying.
    """
    liar = FakeClient(cfg=RunnerConfig(host="h", service="chatterbox",
                                       max_wait=0.0, poll=0.0), segments=99)
    liar.job = lambda job_id: {"status": "queued", "artefacts": []}

    with runner(liar):
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        done = _wait(speech, created["id"], timeout=20.0)

    assert done["status"] == "done", done.get("error")
    assert done["backend"] == "local"
    assert done["fell_back_from"] == "runner"


def test_a_streamed_job_is_never_offered_to_a_lane_that_can_hand_it_back(speech):
    """A RemoteYield with `delivered > 0` on a stream cannot be re-run: _run
    re-enters with a fresh encoder and a fresh offsets list, so the client gets
    a second file header in the middle of the first file, and the job is failed
    instead. The dispatcher refuses to REACH that state rather than building
    resumable streaming for it."""
    import app.main as main

    client = FakeClient(cfg=RunnerConfig(host="h", service="chatterbox", poll=0.0))
    with runner(client):
        response = speech.post("/v1/audio/speech",
                               json={"input": "One short line.",
                                     "stream_format": "sse",
                                     "response_format": "pcm"})
        assert response.status_code == 200
        assert b"speech.audio.done" in response.content

    assert client.submitted == [], "a streamed job was handed to the runner"
    streamed = [j for j in main.jobs.values() if j.get("chunks")]
    assert all(j.get("backend") == "local" for j in streamed)


def test_the_time_the_runner_spent_not_working_is_not_charged_to_the_rate(speech):
    """THE DEFECT THIS PREVENTS: a healthy runner that reads as a slow one.

    `compute_seconds` fed rate_for(...).observe, and it was wall clock. Every
    second the runner spent handing its GPU back to its owner therefore went
    into the average that decides whether the NEXT request can be answered
    inside one HTTP call. A machine that yields twice reads as permanently slow
    and never gets another job.
    """
    stalls = FakeClient(cfg=RunnerConfig(host="h", service="chatterbox",
                                         max_wait=60.0, poll=0.0),
                        segments=2, yield_after=1, yield_polls=8)
    with runner(stalls):
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        done = _wait(speech, created["id"], timeout=20.0)

    assert done["status"] == "done", done.get("error")
    assert done["backend"] == "runner"
    assert done["lane_seconds"] >= done["compute_seconds"], (
        "occupancy cannot exceed wall clock")
    assert "lane_seconds" in done, "the wall clock was not recorded at all"


# --------------------------------------- an engine no lane here can carry ---
#
# THE MID-RUN HALF OF THE SPRING-OFF RULE. Refusing at submit is the other
# half and it lives in test_engines.py; these two are about a job that was
# ACCEPTED -- the runner was up when it arrived -- and then lost its only
# machine. It had no clock at all: `_pick` returned None and looped, `_sweep`
# skips anything whose `finished_at` is None, so it sat `queued` until the
# process died.


def _runner_only(**kw):
    """A dispatcher where `local` cannot carry the engine named `elsewhere`."""
    return _dispatcher(
        lane_allows=lambda lane, engine: not (lane == "local"
                                              and engine == "elsewhere"),
        **kw)


def test_a_job_no_lane_can_carry_does_not_hold_up_the_queue_behind_it():
    """THE DEFECT THIS PREVENTS IS 429 ON A COMPLETELY IDLE LANE.

    `_assign` reads the HEAD of the deque and returns False when nothing can
    take it, which is exactly right while "nothing can take it" means "every
    lane that could is busy" -- nothing behind it could move either. It became
    wrong the moment a job could name an engine this host has no
    implementation of: that job waits on a machine that has gone home, and the
    Chatterbox job behind it waits with it, on a local lane doing nothing.
    TTS_MAX_QUEUE is 32 and deployment-wide, so thirty-two of those answer 429
    to every caller of an engine that works perfectly.
    """
    jobs = {"stranded": {"id": "stranded", "engine": "elsewhere", "work": 1.0},
            "ordinary": {"id": "ordinary", "engine": "chatterbox", "work": 1.0}}
    d = _runner_only(jobs=jobs, runner_ready=False)
    d.submit("stranded")
    d.submit("ordinary")

    assert d._assign(time.monotonic()) is True, \
        "the job behind the stranded one never moved"
    assert d.lanes["local"].slot == "ordinary"
    assert "stranded" in list(d._pending), \
        "the stranded job was dropped rather than skipped"


def test_a_runner_only_job_with_no_lane_fails_on_a_deadline():
    """A JOB WAITING ON A MACHINE THAT NEVER COMES BACK NEEDS AN ANSWER.

    Terminal, on a clock, with the reason -- never `queued` for ever. On a
    fake clock, because fifteen minutes is the shipped default and a test that
    waited it out would be a test nobody runs.
    """
    expired: list[tuple[str, str]] = []
    jobs = {"stranded": {"id": "stranded", "engine": "elsewhere", "work": 1.0}}
    d = _runner_only(jobs=jobs, runner_ready=False,
                     expire=lambda job_id, waited: expired.append((job_id, waited)),
                     stranded_deadline=900.0)
    d.submit("stranded")

    now = 1000.0
    assert d._assign(now) is False, "a job nothing can run was dispatched"
    assert expired == [], "the deadline fired the instant the lane went away"
    assert round(d.stranded_for("stranded", now)) == 0

    # Fourteen minutes with no lane. Still waiting: a reboot or a game
    # finishing has to be survivable, or the deadline is a hair trigger.
    assert d._assign(now + 840.0) is False
    assert expired == [], "expired inside the deadline"
    assert round(d.stranded_for("stranded", now + 840.0)) == 840

    # Past it.
    d._assign(now + 901.0)
    assert [job_id for job_id, _ in expired] == ["stranded"]
    assert round(expired[0][1]) == 901, \
        "the expiry was not told how long the job had been without a lane"
    assert list(d._pending) == [], "the queue slot was never released"
    assert d.finished_ids == ["stranded"], \
        "nobody waiting on this job was ever woken"


def test_the_deadline_counts_time_with_no_lane_and_never_time_in_the_queue():
    """A LEGITIMATELY LONG RENDER MUST NOT TRIP IT.

    An hour behind another hour is a job that has waited an hour and is
    perfectly well. The clock this deadline reads has to start when the last
    capable lane disappears and reset the moment one comes back -- otherwise
    the first Voxtral job of a busy evening is failed for being second in a
    queue that was always going to serve it.
    """
    jobs = {"stranded": {"id": "stranded", "engine": "elsewhere", "work": 1.0}}
    probe = _Probe(ready=False)
    expired: list[str] = []
    d = _dispatcher(
        jobs=jobs, runner_ready=False,
        lane_allows=lambda lane, engine: not (lane == "local"
                                              and engine == "elsewhere"),
        expire=lambda job_id, waited: expired.append(job_id),
        stranded_deadline=100.0)
    d.lanes["runner"].probe = probe
    d.submit("stranded")

    now = 500.0
    d._assign(now)
    d._assign(now + 90.0)
    assert round(d.stranded_for("stranded", now + 90.0)) == 90

    # The machine comes back. The clock is not paused, it is GONE: a job that
    # has been runnable since is a job with no complaint to make.
    probe.ready = True
    d._assign(now + 91.0)
    assert d.lanes["runner"].slot == "stranded", "the lane came back and was not used"
    assert d.stranded_for("stranded", now + 91.0) == 0.0
    assert expired == []


def test_a_lane_with_no_implementation_of_an_engine_is_never_offered_it():
    """`Lane.free()` CANNOT ASK THIS QUESTION AND MUST NOT BE MADE TO.

    It asks whether the machine will take work at all, and `local` has no
    probe because it is always willing -- so before this filter the local lane
    would accept an engine whose `local_class` is None, import None inside
    somebody's job, and fail it after a job id and a progress bar existed.
    """
    d = _runner_only(runner_ready=True)
    now = time.monotonic()
    lanes = [l.name for l in d._eligible({"engine": "elsewhere"}, "j", now)]
    assert lanes == ["runner"], "the local lane was offered an engine it has no class for"
    lanes = [l.name for l in d._eligible({"engine": "chatterbox"}, "k", now)]
    assert set(lanes) == {"local", "runner"}, "an ordinary engine lost a lane"


def test_no_estimate_is_ever_quoted_from_a_lane_that_cannot_run_the_engine():
    """THE DEFECT THIS PREVENTS GOES OUT IN A 202 AND SIZES A PROGRESS BAR.

    `estimate_for` walked every lane, and `local` has no probe, so it was
    always a candidate -- answering with a rate nobody has ever measured for
    an engine it cannot load. That number is `estimated_seconds` in the 202
    and the total of the page's progress bar.
    """
    d = _runner_only(rates={"local": 0.23, "runner": 0.104})
    lane, seconds = d.estimate_for(60.0, "elsewhere")
    assert lane == "runner"
    # 60 s of speech at 0.104x is 577 s, plus the 8 s handover.
    assert round(seconds) == 585
    here, _ = d.estimate_for(60.0, "chatterbox")
    assert here == "local", "an ordinary engine stopped being quoted from here"
