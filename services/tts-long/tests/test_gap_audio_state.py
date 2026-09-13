"""The three ways a row could say something untrue about its own audio.

`_audio` publishes one state per row and its docstring says why: `never` means
this service never held a file, `deleted` means somebody pressed a button,
`expired` means a clock ran out, and telling a reader one when another is true
is the exact lie the enum exists to prevent. Two routes manufactured that lie
and one field was simply wrong until a restart.

  * DELETE /jobs/{id}/audio answered 200 on an imported speech or transcription
    row and set `audio_deleted`, so a row that never had a file here read
    `deleted` afterwards. The page never issues it -- its button is gated on
    the state -- but the route is proxied and it is in the gateway, so any
    keyed client reaches it.
  * POST /runs accepted `audio_deleted` and `audio_expired` from a sender for
    ANY kind. `SERVER_ONLY_KEYS` cannot carry that rule, because for a clone
    row those two are exactly what a re-imported record must bring back.
  * `audio.bytes` was set in `_recover` and nowhere else, so every job this
    process had actually run published `bytes: 0` and only a restart made the
    number true.

NOTHING HERE OPENS A SOCKET OR LOADS A MODEL. Every test drives the real routes
against the app conftest builds.
"""

from __future__ import annotations

import time


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def _imported(client, kind: str) -> str:
    """One finished run from another service, as that service posts it."""
    created = client.post("/runs", json={
        "kind": kind, "service": "tts-stack", "engine": "kokoro",
        "host": "orko", "status": "done", "text": "spoken somewhere else",
    })
    assert created.status_code == 201, created.text
    return created.json()["id"]


# --------------------------------- deleting audio that was never here -------


def test_deleting_the_audio_of_a_run_that_never_had_any_is_refused(speech):
    """THE LIE THE ROUTE USED TO MANUFACTURE.

    An instant-speech row keeps no audio on this service, which `_audio` says
    as `never`. Deleting it answered 200, reported the state as `deleted`, and
    left the row reading `deleted` for good -- a reader told that somebody
    removed a file that never existed on this machine at all.
    """
    job_id = _imported(speech, "speech")
    assert speech.get(f"/jobs/{job_id}").json()["audio"] == {"state": "never"}

    refused = speech.delete(f"/jobs/{job_id}/audio")
    assert refused.status_code == 409, (
        "a run with no audio here answered "
        + str(refused.status_code) + ": " + refused.text)
    assert speech.get(f"/jobs/{job_id}").json()["audio"] == {"state": "never"}, (
        "the refused delete still changed the row, so the state is a lie that "
        "survives the 409")


def test_deleting_the_audio_of_a_clone_job_still_works(speech):
    """The half that must NOT change, pinned beside the half that did.

    A refusal wide enough to cover a clone job would take the button the page
    actually has -- reclaiming a gigabyte while keeping the row is the whole
    reason DELETE /jobs/{id}/audio exists.
    """
    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    _wait(speech, created["id"])

    gone = speech.delete(f"/jobs/{created['id']}/audio")
    assert gone.status_code == 200, gone.text
    assert gone.json()["audio"] == {"state": "deleted"}


# ------------------------------- a sender claiming somebody deleted it ------


def test_a_sender_cannot_say_a_run_with_no_audio_lost_it(speech):
    """The same lie, reached from a compromised or a merely buggy sender.

    `audio_deleted` and `audio_expired` are in RECORD_KEYS because a clone row
    needs them back after a restart, so the filter that stops a sender setting
    `path` lets these two straight through for every kind.
    """
    for field in ("audio_deleted", "audio_expired"):
        refused = speech.post("/runs", json={
            "kind": "speech", "service": "tts-stack", "engine": "kokoro",
            "host": "orko", "status": "done", "text": "spoken elsewhere",
            field: True})
        assert refused.status_code == 400, (
            f"a sender set {field} on a run that keeps no audio here and the "
            "receiver answered " + str(refused.status_code) + ": " + refused.text)
        assert field in refused.text, "the 400 does not name the field to fix"


def test_a_clone_record_may_still_come_back_saying_its_audio_was_deleted(speech):
    """THE CASE THE REFUSAL ABOVE MUST NOT TAKE WITH IT.

    A clone row whose audio somebody deleted is re-imported -- by a migration,
    or by a restore -- and it has to keep saying so. Refusing the two fields
    outright would quietly turn every one of those back into `expired`, which
    is the same lie with the blame moved to a clock.
    """
    created = speech.post("/runs", json={
        "kind": "clone", "service": "tts-long", "engine": "chatterbox",
        "host": "orko", "status": "done", "voice": "default",
        "audio_deleted": True})
    assert created.status_code == 201, created.text
    row = speech.get(f"/jobs/{created.json()['id']}").json()
    assert row["audio"] == {"state": "deleted"}


# ------------------------------------------- the size of the file, today ----


def test_a_job_reports_the_size_of_its_audio_before_any_restart(speech):
    """`bytes` was set in `_recover` AND NOWHERE ELSE.

    So the published `audio` object carried `bytes: 0` for every job this
    process had run, and the true figure appeared only after the service was
    restarted -- a field that was wrong until the one event that is supposed to
    lose nothing. The page does not draw it yet, which is exactly how a
    published field stays wrong.
    """
    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    done = _wait(speech, created["id"])
    assert done["audio"]["state"] == "present", done

    from app import main

    on_disk = len(main.Path(main.jobs[created["id"]]["path"]).read_bytes())
    assert on_disk > 0, "the fixture wrote no audio, so this pins nothing"
    assert done["audio"]["bytes"] == on_disk, (
        "the job reports " + str(done["audio"]["bytes"]) + " bytes for a file "
        "of " + str(on_disk) + ": the size is set when the audio is recovered "
        "and not when it is written")
