

# ------------------------------------------- surviving a restart --


def test_finished_jobs_are_recovered_from_disk(tmp_path, monkeypatch):
    """`jobs` is a dict in one process, so a restart forgets every job while
    the audio sits in a volume and survives. Three things followed: a finished
    job became unreachable though its file was right there, the page showed it
    as pending for ever, and the sweeper could never expire a file it had no
    job for -- so /output grew across restarts with nothing able to clean it.
    """
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)

    assert main._recover() == 1
    job = main.jobs["abc-123"]
    assert job["status"] == "done"
    assert job["format"] == "wav"
    assert job["recovered"] is True
    assert job["finished_at"] > 0, "the sweeper needs this to expire the file"


def test_recovery_says_what_it_does_not_know(tmp_path, monkeypatch):
    """voice, language and the generation parameters lived only in the dict.
    A recovered record must not invent plausible values for them."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)
    main._recover()

    job = main.jobs["abc-123"]
    for unknown in ("voice", "language", "chunks", "realtime_factor"):
        assert unknown not in job, f"{unknown} was invented"


def test_recovery_ignores_files_it_cannot_serve(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "notes.txt").write_text("not audio")
    (tmp_path / "half.part").write_bytes(b"")
    assert main._recover() == 0


def test_recovery_never_overwrites_a_live_job(tmp_path, monkeypatch):
    """A restart is not the only time this runs in a process's life -- and a
    running job's record must win over a stale file with the same name."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {"abc-123": {"id": "abc-123",
                                                   "status": "running"}})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)
    assert main._recover() == 0
    assert main.jobs["abc-123"]["status"] == "running"


# ------------------------------------------- what a restart used to lose --


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def test_a_recovered_job_still_knows_its_voice(speech, tmp_path):
    """GAB-629, reported with a screenshot: every row after a restart read
    "voice unknown".

    `jobs` is a dict in one process. The audio survived in the volume and the
    record did not, so _recover rebuilt each row from the only thing left --
    the filename -- and the voice, the language, the parameters, the chunk
    count and the realtime factor were gone. A list of jobs made with several
    different cloned voices all read the same way.

    This is the whole round trip: a real job through the real routes, the dict
    emptied exactly as a restart empties it, and _recover run against the files
    that survived.
    """
    from app import main

    job = speech.post("/jobs", json={"text": "One short line, spoken once.",
                                     "voice": "default", "language": "en"}).json()
    finished = _wait(speech, job["id"])
    assert finished["voice"] == "default"

    main.jobs.clear()                                  # the restart
    assert main._recover() == 1
    back = main.jobs[job["id"]]

    assert back["voice"] == "default", "the screenshot's 'voice unknown'"
    assert back["language"] == "en"
    assert back["chunks"] == finished["chunks"]
    assert back["realtime_factor"] == finished["realtime_factor"]
    assert back["audio_seconds"] == finished["audio_seconds"]
    assert back["text"].startswith("One short line"), "and what was said"
    assert back["status"] == "done"
    assert back["recovered"] is True, "it did come back from disk, and says so"


def test_a_recovered_cancelled_job_does_not_claim_to_be_done(speech):
    """The status lived in the dict too, and every recovered row said "done".

    A cancelled job keeps the audio it managed to make -- that is what "stop
    and keep what's done" means -- so its file is on disk and indistinguishable
    from a finished one by name.
    """
    from app import main

    job = speech.post("/jobs", json={"text": "A line to cancel."}).json()
    _wait(speech, job["id"])
    main.jobs[job["id"]]["status"] = "cancelled"
    main._write_sidecar(main.jobs[job["id"]])

    main.jobs.clear()
    main._recover()
    assert main.jobs[job["id"]]["status"] == "cancelled"


def test_the_sidecar_is_removed_with_the_audio(speech):
    """Both, or DELETE and the sweeper leave a {id}.json for every file they
    remove -- the unbounded growth of /output the sweeper exists to stop, one
    directory entry smaller. The sweeper calls the same _discard."""
    from app import main

    job = speech.post("/jobs", json={"text": "A line to discard."}).json()
    _wait(speech, job["id"])
    sidecar = main.OUT_DIR / f"{job['id']}.json"
    assert sidecar.exists(), "nothing was written to remove"

    assert speech.delete(f"/jobs/{job['id']}").json()["status"] == "deleted"
    assert not sidecar.exists()
    assert not (main.OUT_DIR / f"{job['id']}.wav").exists()


def test_a_sidecar_cannot_point_the_audio_route_somewhere_else(tmp_path, monkeypatch):
    """/output is a writable volume and `path` is the argument to open() on
    /jobs/{id}/audio, so the record is filtered through SIDECAR_KEYS on the way
    in. The file describes a job; it does not get to say which file it is."""
    import json

    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)
    (tmp_path / "abc-123.json").write_text(json.dumps({
        "voice": "narrator", "path": "/etc/passwd", "id": "somebody-else",
        "recovered": False, "format": "exe"}))

    main._recover()
    job = main.jobs["abc-123"]
    assert job["voice"] == "narrator", "the part it is allowed to say"
    assert job["path"] == str(tmp_path / "abc-123.wav")
    assert job["id"] == "abc-123"
    assert job["format"] == "wav"
    assert job["recovered"] is True


def test_a_corrupt_sidecar_still_recovers_the_audio(tmp_path, monkeypatch):
    """A torn write must cost the row its voice, not its audio. That is the
    state every job was in before the sidecar existed."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)
    (tmp_path / "abc-123.json").write_text('{"voice": "narrator", trunca')

    assert main._recover() == 1
    assert "voice" not in main.jobs["abc-123"], "half a file is not a fact"


def test_an_orphan_sidecar_is_not_left_on_disk(tmp_path, monkeypatch):
    """Audio deleted from outside leaves a {id}.json describing nothing that
    can be played or listed. Removed for the same reason the sweeper exists."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    orphan = tmp_path / "gone-999.json"
    orphan.write_text('{"voice": "narrator"}')

    assert main._recover() == 0
    assert not orphan.exists()


def test_a_segments_only_job_does_not_break_the_whole_listing(speech):
    """GET /jobs was a 500 whenever any job had been submitted as segments.

    The worker holds segments as (text, pause_after) PAIRS, not as the Segment
    models the request carried, and the preview read them as dicts:
    `'tuple' object has no attribute 'get'`, raised inside the comprehension
    that builds the response, so ONE such job took every other job's row with
    it. The page sends segments whenever the text has paragraph pauses, which
    made the Jobs tab go blank rather than degrade.
    """
    job = speech.post("/jobs", json={
        "segments": [{"text": "First paragraph.", "pause_after": 0.4},
                     {"text": "Second paragraph."}]}).json()

    listing = speech.get("/jobs")
    assert listing.status_code == 200, listing.text
    row = next(j for j in listing.json()["jobs"] if j["id"] == job["id"])
    assert row["text_preview"].startswith("First paragraph.")
    # And the expandable row, which reads `text` and showed "the text was not
    # kept for this job" for every one of them.
    assert "Second paragraph." in speech.get(f"/jobs/{job['id']}").json()["text"]


def test_the_listing_carries_the_estimate_the_202_promised(speech):
    """The page sizes its progress bar from `job.estimated_seconds` in the
    LISTING, and the listing never carried the field.

    It was computed inside the POST handler, put in the 202 body and dropped.
    So `job.estimated_seconds || 0` was 0 for every job read back, and the bar,
    the elapsed-and-remaining line and the "past the estimate" state rendered
    nothing at all -- for any client that polled, reloaded, or opened the page
    on a second device, which is every client after the first render.

    Both halves are asserted: the number is in the listing, and it is the SAME
    number the caller was promised rather than a fresh reading of a drifting
    average.
    """
    created = speech.post("/jobs", json={"text": "A line long enough to cost "
                                                 "a measurable moment.",
                                         "voice": "default",
                                         "language": "en"}).json()
    promised = created["estimated_seconds"]
    assert promised > 0

    listed = {j["id"]: j for j in speech.get("/jobs").json()["jobs"]}[created["id"]]
    assert listed["estimated_seconds"] == promised, "the bar has a total again"
    assert speech.get(f"/jobs/{created['id']}").json()["estimated_seconds"] == promised


def test_the_estimate_survives_a_restart(speech):
    """A recovered row is finished, so nothing reads its estimate -- but a
    restart must not become the one way to lose a number a client was given.
    """
    from app import main

    created = speech.post("/jobs", json={"text": "One short line, spoken once.",
                                         "voice": "default",
                                         "language": "en"}).json()
    _wait(speech, created["id"])
    main.jobs.clear()
    assert main._recover() == 1
    assert main.jobs[created["id"]]["estimated_seconds"] == created["estimated_seconds"]


def test_chatterbox_is_the_slower_talker_and_the_constant_says_so(speech):
    """449 characters of ordinary prose measured 37.4 s of Chatterbox audio on
    the deployed stack: 12.0 chars/s, not the 15 this held.

    15 was the middle of a spread of four samples, chosen when the spread was
    all there was. It under-predicted the audio by a fifth, and the audio is
    then divided by a realtime factor near 0.27 to reach the number a reader
    sees -- so a 20% error arrives as roughly four times that in the wait.
    """
    from app import chunking

    assert chunking.CHARS_PER_SECOND == 12.0
    # The measurement this was taken from, within the rounding of one sample.
    assert abs(chunking.speech_seconds(449) - 37.4) < 2.0


def test_a_job_that_vanished_before_it_ran_does_not_kill_the_worker(speech):
    """THE DEFECT THIS PREVENTS: one lost id silently stopping every later job.

    `jobs` is read from the event loop and from the worker thread while DELETE
    and the sweeper both pop from it. The worker used to read `jobs[job_id]`
    OUTSIDE its try block, so an id whose row had gone raised KeyError straight
    out of the `while True` loop and ended the only thread that runs anything.
    Nothing logged it and nothing restarted it: every job submitted afterwards
    would sit at `queued` for ever, and the service would look alive.

    So the failure is deliberately provoked, and then a real job has to prove
    the queue still works.
    """
    from app import main

    main.queue.put("a-job-id-that-is-not-in-the-dict")

    job = speech.post("/jobs", json={"text": "One short line, spoken once.",
                                     "voice": "default"}).json()
    assert _wait(speech, job["id"])["status"] == "done", \
        "the worker thread died on the missing id and never ran this"
