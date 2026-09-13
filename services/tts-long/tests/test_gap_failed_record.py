"""The runs worth debugging are the ones the run log throws away.

D5. `_write_record` is called when a job finishes, when the sweeper expires its
audio, when POST /runs imports a row and when somebody deletes the audio. It is
NOT called when a job FAILS and it is NOT called when a job is cancelled in the
queue, so those two runs exist only in `jobs`, a dict in this process.

THE DESIGN THIS BREAKS IS THE INVERSION `_recover` IS BUILT ON: the record is
the index and the audio is an attachment to it. A failed job has neither. Its
record was never written, so step 2 of `_recover` finds nothing to index, and
it produced no audio, so step 4 finds nothing to adopt. The row is gone on the
next restart -- and "what did we run and where did it go wrong" is the only
question this log exists to answer.

Both tests below drive the real routes, the real dispatcher and the real
record store, and empty `jobs` exactly as a restart empties it. Nothing here
starts a server, opens a socket or waits out a clock: the queued job is held by
an Event that the test itself releases.
"""

from __future__ import annotations

import threading
import time


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    """Poll until the job reaches a terminal status, as the neighbours do."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ------------------------------------------------------- the failure path ---


def test_a_failed_job_survives_a_restart(speech, monkeypatch):
    """A job that went wrong is the one row nobody can afford to lose.

    `_run` marks the job `failed` in the dict and logs a traceback, and that is
    the whole of it: no record is written, so the run is a line in a log file
    that scrolls away and a row that disappears the next time the service is
    restarted. The audio that would otherwise let step 4 of `_recover` adopt it
    does not exist either -- the job failed before anything was encoded.
    """
    from app import main
    from app import synth as synth_module

    def _falls_over(self, text, language, controls, reference):
        raise RuntimeError("the model fell over")

    monkeypatch.setattr(synth_module.Synth, "_speak", _falls_over)

    created = speech.post("/jobs", json={"text": "One short line, spoken once.",
                                         "voice": "default",
                                         "language": "en"}).json()
    job_id = created["id"]
    failed = _wait(speech, job_id)
    assert failed["status"] == "failed", failed
    assert "the model fell over" in (failed.get("error") or "")

    record = main._sidecar(job_id)
    assert record.exists(), (
        f"{record} was never written: the failure path does not call "
        "_write_record, so the only trace of this run is a dict in one "
        "process and a traceback in the log")

    main.jobs.clear()                                  # the restart
    main._recover()

    back = main.jobs.get(job_id)
    assert back is not None, (
        "the restart erased the failed run entirely: no record to index and "
        "no audio to adopt")
    assert back["status"] == "failed", (
        f"the recovered row says {back['status']!r}; a run that went wrong "
        "must not come back reading as a run that went fine")
    assert "the model fell over" in (back.get("error") or ""), (
        "the row came back without the error, which is the only part of a "
        "failed run anybody reads it for")
    assert back["voice"] == "default"
    assert back["text"].startswith("One short line"), "and what was said"


# -------------------------------------------------- the cancellation path ---


def test_a_job_cancelled_in_the_queue_survives_a_restart(speech, monkeypatch):
    """Cancelled BEFORE it started, which is a different path from cancelled
    while running.

    A job stopped mid-run keeps the audio it made and leaves through the
    success path, which does write a record. A job cancelled while it sat in
    the queue returns from the top of `_run` after four lines -- no audio, no
    record, nothing on disk at all -- so the row exists until the next restart
    and then does not.

    The queue here is real: the local lane is width 1, so holding its one
    runner thread on an Event is what keeps the second job genuinely queued
    rather than merely quick. The Event is released by this test, never by a
    clock.
    """
    from app import main
    from app import synth as synth_module

    entered = threading.Event()
    release = threading.Event()
    spoken = synth_module.Synth._speak

    def _held(self, text, language, controls, reference):
        # Only the holder blocks. The cancelled job must never reach this at
        # all, and _speaks below asserts that it did not.
        if "hold the lane" in text.lower():
            entered.set()
            assert release.wait(10), "the held job was never released"
        return spoken(self, text, language, controls, reference)

    monkeypatch.setattr(synth_module.Synth, "_speak", _held)

    try:
        holder = speech.post("/jobs", json={"text": "Hold the lane.",
                                            "voice": "default"}).json()
        assert entered.wait(10), "the holding job never reached the model"

        created = speech.post("/jobs", json={"text": "This is never spoken.",
                                             "voice": "default",
                                             "language": "en"}).json()
        job_id = created["id"]
        assert speech.get(f"/jobs/{job_id}").json()["status"] == "queued", (
            "the second job was not waiting behind the first, so this test is "
            "not exercising the cancelled-in-the-queue path")

        assert speech.delete(f"/jobs/{job_id}").json()["status"] == "cancelling"
    finally:
        # Always, or a failed assertion above leaves the lane thread parked on
        # this Event and the fixture's shutdown waits out the whole drain.
        release.set()

    _wait(speech, holder["id"])
    cancelled = _wait(speech, job_id)
    assert cancelled["status"] == "cancelled", cancelled

    record = main._sidecar(job_id)
    assert record.exists(), (
        f"{record} was never written: a job cancelled in the queue returns "
        "from the top of _run without calling _write_record, and it made no "
        "audio for _recover to adopt either")

    main.jobs.clear()                                  # the restart
    main._recover()

    back = main.jobs.get(job_id)
    assert back is not None, (
        "the restart erased the cancelled run: the record is the index and "
        "nobody wrote a record")
    assert back["status"] == "cancelled", (
        f"the recovered row says {back['status']!r}. `done` would claim audio "
        "that was never made, and `failed` would blame the synthesis for a "
        "person pressing cancel")
    assert back.get("audio_expired") is not True, (
        "audio that was never made has not expired; telling a reader a file "
        "was lost when none ever existed is the lie the split states prevent")
