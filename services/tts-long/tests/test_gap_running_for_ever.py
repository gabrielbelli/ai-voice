"""The runner that is reachable, answers, and lies: for ever "running".

NOTHING HERE OPENS A SOCKET AND NOTHING HERE SLEEPS. The runner is a fake
object with the methods RemoteSynth calls, and `app.remote.time` is replaced by
a clock that only moves when the poller asks it to move -- so a bound measured
in minutes is spent in microseconds, and a poller that never gives up is caught
by a poll cap rather than by a test run that never returns.

`max_wait` is the only bound on speak_segments, and it is charged ONLY for the
time the runner says it is NOT working (remote.py, `if status != "running"`).
That is right for the case it was written for -- a long job on an idle runner
must never be abandoned for taking a long time -- and it leaves the case below
with no bound at all.

spring is a gaming PC that nobody administers while it is being played on. It
may crash mid-job, wedge its model load, lose its worker thread, or simply keep
answering a status field it stopped updating. Every one of those looks
identical from here: `{"status": "running"}`, no artefacts, for ever. THE
SERVER MUST WORK COMPLETELY WITH SPRING SWITCHED OFF, UNPLUGGED, OR LYING ABOUT
ITSELF, and this is the lying case.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


JOB = "the-local-uuid"

# The generation fields as one mapping, which is the shape both speak_segments
# have. Which keys are in it is the engine's business, not this call's.
DIALS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}

# The bound under test, in virtual seconds. Small enough to read, large enough
# that one poll is not the whole of it.
MAX_WAIT = 30.0
POLL = 2.0

# HOW THIS TEST IS ALLOWED TO END, and the only reason it is safe to point a
# real poll loop at a runner that never finishes. At POLL seconds a turn this
# is 400 virtual seconds, thirteen times MAX_WAIT: any bound that applies at
# all has fired long before, so reaching the cap IS the defect rather than a
# tuning accident.
POLL_CAP = 200

# The rate the whole service is built on, and a plain constant rather than
# something re-imported per test: `_decode` asserts the runner answered at this
# rate, so an artefact built at any other one is rejected before it is spliced.
SAMPLE_RATE = 24000


class _RunnerNeverStopped(Exception):
    """Raised by the fake when the poller has gone past every plausible bound."""


class _Clock:
    """The `time` module as far as app.remote is concerned.

    Patched over the module attribute rather than over `time.monotonic` itself,
    so pytest's own timing and everything else in the process keep the real
    clock.

    `sleep` ADVANCES INSTEAD OF SLEEPING, and that is what makes a runner which
    never finishes cost microseconds. Bounding this test with a real sleep
    would mean POLL_CAP polls of POLL seconds each -- minutes of a suite that
    is not allowed to sleep at all.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += seconds


class ForeverRunningClient:
    """A runner that answers `running` for ever and never produces anything.

    Not a hostile fake: this is spring with its worker thread dead, spring
    mid-crash, spring with a wedged model load, and spring left running a build
    that stopped updating its own status field. The controller is up, the
    handshake succeeds, the job document parses. The one thing that never
    happens is an artefact.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.submitted: list[tuple[dict, str]] = []
        self.cancelled: list[str] = []
        self.polls = 0
        self.state = (True, "")
        self.device = "gpu"
        self.cpu_pct = 100

    def speech_state(self):
        return self.state

    def offer(self):
        from app.remote import RunnerOffer

        ready, why = self.speech_state()
        return RunnerOffer(ready, why, device=self.device, cpu_pct=self.cpu_pct)

    def ensure_asset(self, path):  # pragma: no cover - no reference clip here
        raise AssertionError("this test speaks with no reference clip")

    def submit(self, params, idempotency_key):
        self.submitted.append((params, idempotency_key))
        return "remote-job-1"

    def job(self, job_id):
        self.polls += 1
        if self.polls > POLL_CAP:
            raise _RunnerNeverStopped(
                f"speak_segments polled a runner stuck at 'running' "
                f"{self.polls - 1} times ({POLL_CAP * POLL:.0f} virtual seconds, "
                f"{POLL_CAP * POLL / MAX_WAIT:.0f}x the {MAX_WAIT:.0f}s max_wait) "
                "and was still going. NO BOUND APPLIES to a runner that says it "
                "is working and never produces a segment: the job holds its lane "
                "and its queue slot until the process dies.")
        # The whole defect in one dict. `waited` is charged only when the status
        # is NOT "running", so this document accumulates exactly zero of the
        # bound however many times it is served.
        return {"status": "running", "artefacts": [], "record": None}

    def artefact(self, job_id, name):  # pragma: no cover - nothing is produced
        raise AssertionError("this runner never lists an artefact")

    def cancel(self, job_id):
        self.cancelled.append(job_id)


class ProductiveRunningClient(ForeverRunningClient):
    """The honest runner this must not be confused with: slow, but delivering.

    One segment per poll, taking far longer in total than `max_wait`. The
    difference from the fake above is progress, not status: both answer
    "running" every single time they are asked.
    """

    def __init__(self, cfg, segments: int) -> None:
        super().__init__(cfg)
        self._segments = segments

    def job(self, job_id):
        self.polls += 1
        if self.polls > POLL_CAP:  # pragma: no cover - a delivering runner ends
            raise _RunnerNeverStopped("the productive runner never finished")
        done = min(self.polls, self._segments)
        return {
            "status": "done" if done >= self._segments else "running",
            # The real naming, from services/lib: <job id>.<segment>.f32
            "artefacts": [f"remote-job-1.{i}.f32" for i in range(done)],
            "record": {"input_tokens": 7} if done >= self._segments else None,
        }

    def artefact(self, job_id, name):
        index = int(name.split(".")[-2])
        return np.full(SAMPLE_RATE // 10, 0.1 * (index + 1), dtype="<f4").tobytes()


def segs(n):
    return [(f"segment {i}", 0.0) for i in range(n)]


@pytest.fixture
def runner(monkeypatch):
    """`app.remote` with its clock replaced, and the clock, as one pair.

    THE MODULE IS RESOLVED HERE AND EVERY NAME IS TAKEN OUT OF IT, never bound
    at file import. conftest._build deletes every `app.*` from sys.modules and
    imports the package again for each app it builds, so a class bound at file
    import can belong to a module object that is no longer the one in
    sys.modules -- and patching `time` on one module while driving a
    RemoteSynth out of another silently restores the real clock. That failure
    is invisible from inside the test: it still polls, it still asserts, and it
    sleeps POLL seconds per poll for real, so the price of getting this wrong
    is a quarter of an hour of wall clock in a suite that must not sleep.
    """
    module = importlib.import_module("app.remote")
    clock = _Clock()
    monkeypatch.setattr(module, "time", clock)
    assert module.time is clock, "the clock was patched onto the wrong module"
    return module, clock


def _cfg(module):
    return module.RunnerConfig(host="runner.invalid", service="chatterbox",
                               poll=POLL, max_wait=MAX_WAIT)


def test_a_runner_that_says_running_for_ever_is_still_bounded(runner):
    """THE DEFECT THIS PREVENTS: one lying runner wedging the service.

    `waited` is charged only while the runner says it is not working, so a
    runner answering {"status": "running"} and producing nothing accumulates
    zero of `max_wait` and NO BOUND APPLIES AT ALL. speak_segments polls every
    TTS_RUNNER_POLL until the process is killed.

    What the operator sees is worse than an error: the row sits at "running"
    with a progress bar that never moves, and it holds the lane and a slot
    against TTS_MAX_QUEUE while it does so. The local CPU would have finished
    at 0.275x realtime and reported a real failure hours earlier.

    The runner does not have to be malicious to do this. Nobody administers
    spring while it is being played on; a dead worker thread, a wedged model
    load and a crash that leaves the controller answering from a stale document
    all look exactly like this from here.
    """
    module, clock = runner
    client = ForeverRunningClient(_cfg(module))

    with pytest.raises((module.RemoteYield, module.RemoteUnavailable)) as raised:
        module.RemoteSynth(client, JOB).speak_segments(
            segs(3), "en", DIALS, None)

    # The bound has to be roughly the bound. A fix that gives up only after ten
    # times max_wait has moved the wedge rather than removed it.
    assert clock.slept <= MAX_WAIT + 10 * POLL, (
        f"gave up only after {clock.slept:.0f}s of a {MAX_WAIT:.0f}s bound")
    # WITHDRAW THE LEASE ON THE WAY OUT, exactly as the yield path already does.
    # A job abandoned here without a cancel leaves the runner holding a lease
    # under this idempotency key that nobody will ever collect.
    assert client.cancelled == ["remote-job-1"], (
        "the lease was left on a runner that is producing nothing; "
        f"cancelled={client.cancelled!r}")
    assert getattr(raised.value, "delivered", 0) == 0, \
        "nothing was delivered, so nothing may be reported as delivered"


def test_a_slow_runner_that_is_actually_delivering_is_not_bounded(runner):
    """THE REGRESSION THE FIX ABOVE COULD EASILY CAUSE, pinned here.

    The obvious fix -- charge `waited` for running time too -- reinstates the
    deadline-from-submission that was removed for good reason: at the default
    bound and around 0.6x realtime it abandoned every long job on a COMPLETELY
    IDLE runner, and, once anything had been streamed, `_worker` turned a
    healthy job into a FAILED one. tts-long is the long-job service.

    The distinction the fix has to make is progress, not status. This runner
    needs twenty polls, well past max_wait, and must still finish.
    """
    module, clock = runner
    client = ProductiveRunningClient(_cfg(module), segments=20)

    spoken = module.RemoteSynth(client, JOB).speak_segments(
        segs(20), "en", DIALS, None)

    assert clock.slept > MAX_WAIT, \
        "the fake runner did not outlast max_wait, so this pins nothing"
    assert spoken.audio.size == 20 * (SAMPLE_RATE // 10), \
        "a runner that was delivering was cut off for taking a long time"
    assert client.cancelled == [], "the lease was withdrawn from a working runner"
