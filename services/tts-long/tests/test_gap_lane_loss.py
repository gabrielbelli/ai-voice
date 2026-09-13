"""Spring goes away mid-job, and the job must be RE-SPOKEN, not failed.

THE REQUIREMENT THESE FOUR TESTS ARE THE WHOLE OF. This server has to work
completely with the runner switched off, unplugged, asleep, out of memory or
lying about itself. A second machine is an accelerator; it is never a
dependency. Every scenario below is that machine disappearing at a moment the
dispatcher cannot see, and every one of them must end `done` on this host.

WHY THE PROBE CANNOT COVER THIS AND WHY THE WINDOW GOT WIDER. `_backend_for`
used to call `offer()` milliseconds before the submit, so "up when we looked,
down when we sent" was about a zero-width window. `LaneProbe.ok` now accepts an
answer up to three probe intervals old -- thirty seconds at the shipped
TTS_RUNNER_PROBE_S -- so there is now a half-minute in which every dispatched
job is handed to a machine that is already gone. dispatch.py's own docstring
says "which the yield path already recovers". IT DOES NOT RECOVER THIS: only
RemoteYield reaches `_execute_on_lane` as a lane problem, and losing the lane
arrives as an ordinary exception that `_run` writes onto the job as `failed`.

NOTHING HERE OPENS A SOCKET, LOADS A MODEL OR SLEEPS TO PASS TIME. The runner
is test_remote's FakeClient with one method rewritten to go away, exactly as
its neighbours do it, and the probe is primed by hand by `runner()`.

`RemoteUnavailable` IS IMPORTED INSIDE THE FUNCTIONS THAT RAISE IT, NEVER AT
THE TOP OF THIS FILE. conftest deletes every `app.*` from sys.modules and
reimports, so a class bound at module import is a DIFFERENT class object from
the one app.main is holding -- and an `except RemoteUnavailable` added in
app.main would not catch the stale one, which would make these tests
unfixable rather than failing.

LOSING THE LANE IS A CLASS, NOT ONE EXCEPTION TYPE, and `RunnerClient._request`
is where it becomes one: it wraps the connect, the request and the read in
`(OSError, http.client.HTTPException)` and turns every one of them into
RemoteUnavailable. The second half of that tuple is not decoration --
`http.client.HTTPException` does not inherit from OSError, so IncompleteRead
(a desktop suspending mid-read) escaped as itself and failed a job. See
test_gap_lying_runner.py, which is the rest of this story: a runner that ANSWERS
and lies.
"""

from __future__ import annotations

from test_remote import FakeClient, _wait, runner


def _gone(where: str, errno: int, why: str):
    """Raise what the runner client raises when the machine is not there.

    Imported here rather than at module scope for the reason in the docstring
    above: this has to be app.main's OWN RemoteUnavailable, not a stale copy
    from before conftest reimported the package.
    """
    from app.remote import RemoteUnavailable

    raise RemoteUnavailable(f"{where}: [Errno {errno}] {why}")


def _lane(main):
    return main.dispatch.lanes["runner"]


# ------------------------------------------------- 1. unplugged mid-job -----


def test_a_runner_unplugged_mid_job_is_re_spoken_here_not_failed(speech):
    """THE CABLE COMES OUT WHILE THE JOB IS RUNNING ON THAT MACHINE.

    Errno 113 is what the poll loop gets: the lease was accepted, one segment
    was collected, and then the host stopped existing. The audio still has to
    arrive. Nobody asked this service for a second machine and nobody should
    have to notice that it went.
    """
    fake = FakeClient(segments=3)
    polls = {"n": 0}
    healthy = fake.job

    def unplugged(job_id):
        polls["n"] += 1
        if polls["n"] > 1:
            _gone("GET /v1/jobs", 113, "No route to host")
        return healthy(job_id)

    fake.job = unplugged

    with runner(fake) as main:
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])
        cooling = _lane(main).cooldown_until

    assert polls["n"] > 1, "the fake never got as far as going away"
    assert finished["status"] == "done", (
        "the runner was unplugged mid-job and the job was reported "
        + repr(finished["status"]) + " (" + str(finished.get("error"))
        + ") instead of being spoken on this host")
    assert finished["audio_seconds"] > 0, "a done job with no audio in it"
    assert finished.get("backend") == "local", \
        "the re-speak did not happen here, so nothing re-spoke it"
    assert cooling > 0, \
        "the lane that lost the job was never cooled, so the next job goes there too"


# --------------------------------- 2. asleep between probe and submit -------


def test_a_runner_asleep_since_the_probe_answered_is_re_spoken_here(speech):
    """THE WINDOW THE REBUILD WIDENED, AND IT IS NOW THIRTY SECONDS WIDE.

    The probe answered "free" and the machine went to sleep afterwards. Errno
    111 is the whole of what the submit gets back. Nothing was generated, no
    client was sent anything, and re-speaking this on the CPU is free of every
    hazard that makes a mid-stream restart unsafe -- so a `failed` here is a
    job thrown away for no reason at all.

    THE PROBE IS STILL SAYING FREE AFTERWARDS, on purpose. Only the dispatcher
    being TOLD can shut this lane inside the staleness window; waiting for the
    probe to notice is up to thirty seconds of jobs sent at a sleeping machine.
    """
    fake = FakeClient(segments=1)
    attempts: list[str] = []

    def asleep(params, idempotency_key):
        attempts.append(idempotency_key)
        _gone("POST /v1/jobs", 111, "Connection refused")

    fake.submit = asleep

    with runner(fake) as main:
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])
        lane = _lane(main)
        still_ready, cooling = lane.probe.ok(), lane.cooldown_until

    assert attempts == [created["id"]], \
        "the job never reached the sleeping runner, so this tested nothing"
    assert finished["status"] == "done", (
        "the runner was asleep when the job was sent and the job was reported "
        + repr(finished["status"]) + " (" + str(finished.get("error"))
        + ") instead of being spoken on this host")
    assert finished["audio_seconds"] > 0, "a done job with no audio in it"
    assert still_ready is True, \
        "the fake stopped claiming to be free, so the stale-answer window is untested"
    assert cooling > 0, (
        "the probe still says free and the lane was never cooled, so every job "
        "arriving in the next thirty seconds is sent at a sleeping machine")


# ------------------------------------ 3. the runner failing the job itself --


def test_a_runner_that_fails_the_job_itself_is_re_spoken_here(speech):
    """SPRING RUNS OUT OF MEMORY AND SAYS SO, POLITELY.

    The lease comes back `status: failed` with a reason, which
    `RemoteSynth.speak_segments` turns into RemoteUnavailable. That is a
    statement about THAT MACHINE -- 8 GB of card, a game already in it -- and
    not about the text, which this host will speak without complaint. Copying
    the runner's error onto the local job makes the CPU path, the entire reason
    the local Synth is never removed, unreachable at the exact moment it is
    needed.
    """
    fake = FakeClient(segments=2, fail="CUDA error: out of memory")

    with runner(fake) as main:
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])
        cooling = _lane(main).cooldown_until

    assert len(fake.submitted) == 1, "the runner was never given the job"
    assert finished["status"] == "done", (
        "the runner failed the job and this service copied its error onto the "
        "local job (" + str(finished.get("error")) + ") instead of speaking it here")
    assert "out of memory" not in str(finished.get("error") or ""), \
        "somebody else's memory is not this job's error"
    assert finished["audio_seconds"] > 0, "a done job with no audio in it"
    assert cooling > 0, \
        "a runner that cannot start a job was left open for the next one"


# --------------------------------------- 4. five at once, runner asleep -----


def test_a_burst_against_a_sleeping_runner_shuts_the_lane_after_one_attempt(speech):
    """THE ONE THAT PROVES THE LANE IS NEVER SHUT, AND THE WORST OF THE FOUR.

    One job hitting a sleeping machine is a job. Five is the shape of the
    defect: the lane is never told it lost anything, so its cooldown is never
    set, `_refused` never names it, and the dispatcher hands it the next job,
    and the next, for as long as the probe's stale answer stands. Five
    submissions at a machine that is not there, five failed jobs, and a lane
    still advertising itself as free.

    ONE ATTEMPT IS THE WHOLE ASSERTION. The dispatcher already knows how to
    stop -- `_work` cools a lane that hands a job back and records the refusal
    -- so a lane loss that reaches it as a lane loss costs exactly one wasted
    submit and the other four jobs never leave this host.
    """
    fake = FakeClient(segments=1)
    attempts: list[str] = []

    def asleep(params, idempotency_key):
        attempts.append(idempotency_key)
        _gone("POST /v1/jobs", 111, "Connection refused")

    fake.submit = asleep

    with runner(fake) as main:
        created = [speech.post("/jobs", json={"text": f"Line number {n}.",
                                              "voice": "default"}).json()["id"]
                   for n in range(5)]
        finished = [_wait(speech, job_id) for job_id in created]
        cooling = _lane(main).cooldown_until

    assert attempts, "not one of the five reached the runner, so this tested nothing"
    failed = [j for j in finished if j["status"] != "done"]
    assert not failed, (
        str(len(failed)) + " of 5 jobs were failed because a machine this "
        "service does not need was asleep; the first said "
        + repr(failed[0].get("error") if failed else None))
    assert len(attempts) == 1, (
        "the sleeping runner was submitted to " + str(len(attempts))
        + " times: the lane is never told it lost a job, so it is never shut "
        "and every job in the burst is offered to it in turn")
    assert cooling > 0, "the lane was still open after losing five jobs"
