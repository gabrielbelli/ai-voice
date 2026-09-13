"""A yield that lands DURING a shutdown, which is the one nobody wired up.

THE MACHINE THAT YIELDS IS ALLOWED TO BE ABSENT, ASLEEP OR LYING. That is the
whole premise of the runner lane: spring can come back to its keyboard at any
instant, including the instant orko is stopping, and a yield is normal traffic
rather than an error. Every other yield path is covered by test_dispatch.py --
back to the head of the deque, the same lane never offered twice, the cooldown.
None of them run while `_stopping` is set, and that is the gap.

At HEAD a yield was handled on the worker's own thread by the `_speak` ladder
and ran locally there and then, so shutdown never saw one. Now `_work` puts the
job back on `_pending` (dispatch.py:463) and deliberately does not wake the
chooser -- but by then the chooser has already returned on `_stopping`, so
nothing will ever pick the job up again, and `drain()` looks only at lane slots:

    return not any(l.slot is not None for l in self._lanes.values())

The deque is not in that sentence. So drain reports the shutdown CLEAN, the
cancellation loop in main.py:1159-1165 never runs, and the job that was one
handover away from finishing is left `queued` for ever with nobody to run it and
nothing said about it anywhere. `_choose_loop` says "Jobs left in the deque
never started and are the caller's to mark" -- and the caller is never told.

NOTHING HERE OPENS A SOCKET, LOADS A MODEL OR SLEEPS. The lanes park on the
dispatcher's own Condition until `drain()` sets `_stopping`, which is the same
notify the lane threads already wait on, so the shutdown happens exactly where
the test wants it and not a wall-clock guess later.
"""

from __future__ import annotations

import threading

from app.dispatch import FINISHED, YIELDED, Dispatcher


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def exception(self, *a, **k): pass


class _Probe:
    """A LaneProbe as far as Lane.free can tell, answering yes by hand."""

    def start(self): pass

    def stop(self): pass

    def ok(self): return True

    def snapshot(self): return {"ready": True}


def _dispatcher(execute, jobs):
    """The two lanes the service really has, at the two measured rates.

    0.23x here and 0.70x there with an 8 s handover: 300 seconds of speech is
    1304 s local against 437 s remote, so a 300 s job goes to the runner. Those
    are the numbers the rest of the suite pins, and this file relies on them to
    get a job onto the lane that is able to hand it back.
    """
    rates = {"local": 0.23, "runner": 0.70}

    def finished(job_id):
        # WHAT THE REAL CALLER DOES WITH A FINISHED JOB, so that a fix which
        # runs the yielded job somewhere instead of cancelling it is accepted
        # by these tests rather than failed by them.
        row = jobs.get(job_id)
        if row is not None:
            row["status"] = "finished"

    d = Dispatcher(
        execute=execute,
        job_of=jobs.get,
        work_of=lambda job: job.get("work", 1.0),
        rate_of=lambda name, engine=None: rates[name],
        finished=finished,
        log=_Log())
    d.add_lane("local")
    d.add_lane("runner", hop=8.0, probe=_Probe())
    return d


def _park_until_shutdown(d, running: threading.Event) -> None:
    """Hold a lane inside `execute` until `drain()` says stop.

    A wall-clock sleep here would be a race dressed up as a delay. `drain()`
    sets `_stopping` under this exact Condition and notifies it, so waiting on
    it is what makes "the yield happens during the shutdown" a fact rather than
    a hope.
    """
    running.set()
    with d._cond:
        assert d._cond.wait_for(lambda: d._stopping, timeout=5), \
            "drain() never started, so nothing was tested"


def _mark_what_drain_left(clean: bool, jobs: dict) -> None:
    """main.py:1158-1165, copied, because that is the code under test.

    CANCELLED, NEVER FAILED -- and only when drain says it did not finish. A
    drain that reports clean means this loop does not run at all, which is the
    entire defect.
    """
    if not clean:
        for job in jobs.values():
            if job["status"] in {"queued", "running"}:
                job.update(status="cancelled",
                           error="the service was shutting down")


def test_a_yield_during_shutdown_does_not_leave_the_job_queued_for_ever():
    """THE DEFECT: spring comes back to its keyboard while orko is stopping.

    The job is on the runner lane, the runner hands it back, and the handback
    lands after the chooser has gone. Nothing runs it, `drain()` says the
    shutdown was clean so nothing cancels it, and the operator is left with a
    job sitting `queued` behind a service that is no longer running -- the exact
    "sat queued for ever with no error anywhere" failure the lanes were built to
    remove, reached from the other end.
    """
    jobs = {"a": {"id": "a", "work": 300.0, "status": "queued"}}
    running = threading.Event()

    def execute(job, lane):
        assert lane == "runner", f"the 300 s job should be on the runner, not {lane}"
        job["status"] = "running"
        _park_until_shutdown(d, running)
        # WHAT main.py DOES WITH A RemoteYield: back to `queued`, and the
        # dispatcher is told to requeue it rather than to finish it.
        job["status"] = "queued"
        return YIELDED

    d = _dispatcher(execute, jobs)
    d.start()
    d.submit("a")
    assert running.wait(5), "the job never reached the runner lane"

    clean = d.drain(2.0)
    _mark_what_drain_left(clean, jobs)

    assert jobs["a"]["status"] != "queued", (
        "the runner handed the job back during shutdown and nobody was told: "
        f"drain() reported clean={clean} and the job is still queued, with "
        f"{d.snapshot()['waiting']} in the deque and no thread left to run it")


def test_drain_does_not_report_clean_while_jobs_are_still_queued():
    """`drain()` reads the lanes and calls that the whole service.

    Waiting work is not in `not any(l.slot is not None ...)`, so a shutdown with
    a full deque behind it is reported exactly like a shutdown with nothing left
    to do. The caller uses that boolean to decide whether to mark anything
    cancelled, so a false clean is not cosmetic -- it is the difference between
    five jobs marked cancelled and five jobs left queued.

    Measured shape: one job handed back by the runner at shutdown, one finishing
    locally, four never started. Five waiting, and drain said True.
    """
    jobs = {"a": {"id": "a", "work": 300.0, "status": "queued"},
            "b": {"id": "b", "work": 300.0, "status": "queued"}}
    for i in range(4):
        jobs[f"q{i}"] = {"id": f"q{i}", "work": 1.0, "status": "queued"}
    on_runner, on_local = threading.Event(), threading.Event()

    def execute(job, lane):
        job["status"] = "running"
        if lane == "runner":
            _park_until_shutdown(d, on_runner)
            job["status"] = "queued"
            return YIELDED
        _park_until_shutdown(d, on_local)
        return FINISHED

    d = _dispatcher(execute, jobs)
    d.start()
    d.submit("a")
    assert on_runner.wait(5), "the first job never reached the runner lane"
    d.submit("b")
    assert on_local.wait(5), "the second job never reached the local lane"
    for i in range(4):
        d.submit(f"q{i}")

    clean = d.drain(2.0)

    waiting = d.snapshot()["waiting"]
    assert waiting == 5, f"expected the yielded job plus four unstarted, got {waiting}"
    assert d.position("a") == 0, "the yielded job did not go back to the head"
    assert clean is False, (
        f"drain() reported the shutdown clean with {waiting} jobs still in the "
        "deque, so the caller marked none of them cancelled")
