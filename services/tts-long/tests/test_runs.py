"""The record store: what survives its audio, and what a sender may say.

THE INVERSION THIS FILE IS ABOUT. `_recover` used to walk the AUDIO files and
read a note beside each one, so a record with no file was an orphan and was
deleted. Instant speech and transcriptions keep no audio at all, and a clone
job whose audio was swept is a row people still want, so records with no file
are now the majority. The record is the index; the audio is an attachment.

Every test is named after the mistake it prevents.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


# The contract between this service and the two that post to it. Reachable from
# every service directory under the documented test command.
FIXTURES = Path(__file__).resolve().parents[3] / "packages/common/tests/fixtures/run_records.json"


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ----------------------------------------------------------- the contract ---


def test_the_shared_fixture_round_trips(speech):
    """THE ONE FILE BOTH SIDES OF THIS FEATURE MAY TREAT AS AUTHORITATIVE.

    tts and stt build their bodies against it; this service accepts them. If
    the two ever disagree the fixture is right and the code is wrong, which is
    only true while something actually reads the file -- so this ASSERTS IT
    EXISTS and does not skip. A missing contract is the failure, not a reason
    to pass.
    """
    assert FIXTURES.exists(), (
        f"{FIXTURES} is the contract between this service and the senders in "
        "services/tts and services/stt. It is missing, so nothing here is "
        "pinned to anything.")
    bodies = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert set(bodies) == {"clone", "speech", "transcribe", "preset"}

    for kind, body in bodies.items():
        created = speech.post("/runs", json=body)
        assert created.status_code == 201, created.text
        job_id = created.json()["id"]
        assert job_id, "the receiver must mint an id when the sender omits one"
        if "id" in body:
            assert job_id == body["id"], "an id the sender chose was thrown away"

        row = speech.get(f"/jobs/{job_id}").json()
        # THE KEY IS THE EXAMPLE'S NAME, NOT THE `kind` COLUMN, and the fourth
        # example is why. A preset-voice job is `speech` -- it has no speaker
        # encoder, so it is not cloning anything -- and `KINDS` deliberately
        # did not grow a fourth value for it: three services post to /runs and
        # a fourth value is a fourth table that has to agree.
        assert row["kind"] == body["kind"]
        for field, value in body.items():
            if field in {"text", "id"}:
                continue
            assert row.get(field) == value, f"{kind}.{field} did not survive"
        assert row["text"] == body["text"]

    listing = speech.get("/jobs").json()
    rows = {j["kind"]: j for j in listing["jobs"]}
    # AUDIO THAT WAS NEVER KEPT IS NOT AUDIO THAT WAS LOST. Telling a reader
    # "expired" about a file that never existed is the lie this state prevents.
    assert rows["speech"]["audio"]["state"] == "never"
    assert rows["transcribe"]["audio"]["state"] == "never"


def test_a_sender_cannot_set_path(speech):
    """`path` is the argument to open() on /jobs/{id}/audio.

    The record file was already filtered on the way in because /output is a
    writable volume; this arrives over HTTP from another container, so the
    same rule is now the whole security boundary of the route. A sender that
    tries is a bug in that sender, not a compatibility case: the 400 names the
    field so it gets fixed rather than silently ignored.
    """
    r = speech.post("/runs", json={"kind": "speech", "service": "tts",
                                   "path": "/etc/passwd"})
    assert r.status_code == 400, r.text
    assert "path" in r.text
    assert not any(j.get("path") == "/etc/passwd" for j in
                   speech.get("/jobs").json()["jobs"])


def test_a_sender_cannot_claim_an_id_that_is_already_a_job(speech):
    """`id` is also the audio FILENAME for a clone job, so a collision would
    attach somebody else's recording to this record."""
    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    _wait(speech, created["id"])
    r = speech.post("/runs", json={"id": created["id"], "kind": "speech",
                                   "service": "tts", "engine": "kokoro"})
    assert r.status_code == 201
    assert r.json()["id"] != created["id"]
    assert speech.get(f"/jobs/{created['id']}").json()["kind"] == "clone", \
        "the imported record overwrote a real job"


def test_an_unknown_field_does_not_400(speech):
    """THE WHOLE MID-UPGRADE STORY IN ONE RULE.

    A field added to a newer sender must never refuse an older receiver, or
    every deploy becomes an ordering problem and the two halves of this feature
    cannot ship independently. Unknown keys are DROPPED, silently.
    """
    r = speech.post("/runs", json={"kind": "speech", "service": "tts",
                                   "engine": "kokoro", "host": "orko",
                                   "a_field_from_next_year": {"deep": [1, 2]}})
    assert r.status_code == 201, r.text
    row = speech.get(f"/jobs/{r.json()['id']}").json()
    assert "a_field_from_next_year" not in row


def test_a_record_with_no_kind_reads_as_clone(speech, tmp_path):
    """Every record written before this release has no `kind`, and there is no
    migration of file contents anywhere in this service. The default is what
    makes them all valid."""
    from app import main

    main.jobs.clear()
    (main.OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
    main._sidecar("old-1").write_text(json.dumps(
        {"status": "done", "voice": "narrator", "finished_at": 1.0}))
    main._recover()

    row = speech.get("/jobs/old-1").json()
    assert row["kind"] == "clone"
    assert row["voice"] == "narrator"


# --------------------------------------------------------------- the TTLs ---


def test_the_audio_ttl_keeps_the_record(speech):
    """THE DEFECT THIS PREVENTS: a history that deletes itself every day.

    One TTL took the record with the audio at twenty-four hours, so what was
    said, which voice said it and which machine ran it -- a few hundred bytes
    -- was thrown away with the megabytes. The Jobs tab could only ever show
    today, which is the whole reason there was nothing to filter.
    """
    from app import main

    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    done = _wait(speech, created["id"])
    audio = Path(done["path"])
    assert audio.exists()

    # A day and a second later, with the record TTL a month out.
    main._sweep(done["finished_at"] + main.AUDIO_TTL + 1)

    assert not audio.exists(), "the audio was not reclaimed"
    row = speech.get(f"/jobs/{created['id']}").json()
    assert row["voice"] == "default", "the record went with the audio"
    assert row["audio"]["state"] == "expired"
    # AND NOT `deleted`. Nobody pressed anything; a clock ran out. Telling
    # somebody they deleted a file they did not delete is the lie this split
    # exists to prevent.
    assert not row.get("audio_deleted")
    assert main._sidecar(created["id"]).exists(), "the record file went too"


def test_the_record_ttl_takes_the_record_and_the_row(speech):
    """A month later it does go: "keep everything for ever" is the growth the
    sweeper exists to stop, in smaller files."""
    from app import main

    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    done = _wait(speech, created["id"])
    main._sweep(done["finished_at"] + main.RECORD_TTL + 1)

    assert speech.get(f"/jobs/{created['id']}").status_code == 404
    assert not main._sidecar(created["id"]).exists()


def test_a_live_job_is_never_aged_out_from_under_itself(speech):
    """A queued job has no `finished_at`. Reading `time.time()` as its default
    -- which the old sweeper did -- put every unfinished job permanently one
    tick from the cutoff."""
    from app import main

    main.jobs["live-1"] = {"id": "live-1", "status": "running", "cancelled": False,
                           "created_at": 0.0, "segments": [], "text": ""}
    try:
        main._sweep(time.time() + 10 * main.RECORD_TTL)
        assert "live-1" in main.jobs
    finally:
        main.jobs.pop("live-1", None)


# ------------------------------------------------------------- _recover ----


def test_recover_indexes_records_not_audio(speech, tmp_path, monkeypatch):
    """THE INVERSION, ASSERTED. A record with no audio behind it is a ROW, not
    rubbish -- and a clone record whose file has gone says `expired` rather
    than pretending the file is there."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "kept.json").write_text(json.dumps(
        {"status": "done", "voice": "narrator", "format": "wav"}))
    (runs / "heard.json").write_text(json.dumps(
        {"status": "done", "kind": "transcribe", "engine": "parakeet",
         "text": "what was heard"}))
    (tmp_path / "kept.wav").write_bytes(b"RIFF" + b"\0" * 100)

    assert main._recover() == 2
    assert main.jobs["kept"]["path"] == str(tmp_path / "kept.wav")
    assert main.jobs["kept"]["bytes"] == 104
    assert not main.jobs["kept"].get("audio_expired")
    # The transcription never had a file here and must not be told it lost one.
    assert main.jobs["heard"].get("audio_expired") is None
    assert main._audio(main.jobs["heard"])["state"] == "never"


def test_a_clone_record_whose_audio_is_gone_says_expired_after_a_restart(
        speech, tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "gone.json").write_text(json.dumps(
        {"status": "done", "voice": "narrator", "format": "wav"}))

    assert main._recover() == 1
    assert main.jobs["gone"]["audio_expired"] is True
    assert main._audio(main.jobs["gone"])["state"] == "expired"


def test_a_legacy_record_beside_the_audio_is_moved_once(speech, tmp_path,
                                                        monkeypatch):
    """Two places to look for the same fact is one too many, and _recover's
    audio walk would keep tripping over them."""
    from app import main

    monkeypatch.setattr(main, "OUT_DIR", tmp_path)
    monkeypatch.setattr(main, "jobs", {})
    (tmp_path / "abc-123.wav").write_bytes(b"RIFF" + b"\0" * 100)
    (tmp_path / "abc-123.json").write_text('{"voice": "narrator"}')

    assert main._recover() == 1
    assert not (tmp_path / "abc-123.json").exists()
    assert (tmp_path / "runs" / "abc-123.json").exists()
    assert main.jobs["abc-123"]["voice"] == "narrator"


# ----------------------------------------------------------- the filters ---


def _seed(speech):
    """One record of each shape the listing has to tell apart."""
    bodies = [
        {"kind": "speech", "service": "tts", "engine": "kokoro", "host": "orko",
         "voice": "bm_george", "text": "spoken now"},
        {"kind": "transcribe", "service": "stt", "engine": "parakeet",
         "host": "orko", "text": "heard now"},
        {"kind": "clone", "service": "tts-long", "engine": "chatterbox",
         "host": "orko", "voice": "gabriel", "status": "failed",
         "error": "the model fell over", "text": "never spoken"},
    ]
    return [speech.post("/runs", json=b).json()["id"] for b in bodies]


def test_the_inverse_filter_exists_by_name(speech):
    """"SHOW ME THE JOBS WHOSE AUDIO IS GONE" WAS NOT ASKABLE.

    The page had one filter -- "only jobs whose audio still exists" -- and no
    inverse, so the question somebody actually had (what did I delete, what
    expired) had no answer anywhere in the product.
    """
    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    _wait(speech, created["id"])
    speech.delete(f"/jobs/{created['id']}/audio")

    listing = speech.get("/jobs?audio=deleted").json()
    assert [j["id"] for j in listing["jobs"]] == [created["id"]]
    assert listing["jobs"][0]["audio"] == {"state": "deleted"}
    assert speech.get("/jobs?audio=present").json()["jobs"] == []


def test_a_live_job_survives_every_filter_combination(speech):
    """It has no audio YET, which is not the same as having none. Hiding the
    thing somebody is waiting for is the worst possible reading of a filter."""
    from app import main

    main.jobs["live-1"] = {"id": "live-1", "status": "queued", "cancelled": False,
                           "created_at": time.time(), "segments": [], "text": "",
                           "kind": "clone"}
    try:
        for query in ("", "?audio=deleted", "?audio=expired", "?audio=present",
                      "?status=done", "?status=failed",
                      "?audio=never&status=cancelled"):
            ids = [j["id"] for j in speech.get(f"/jobs{query}").json()["jobs"]]
            assert "live-1" in ids, f"a live job vanished under {query!r}"
    finally:
        main.jobs.pop("live-1", None)


def test_a_failed_job_is_never_hidden_by_an_audio_filter(speech):
    """A clone that fails has no audio. A naive filter makes it disappear at
    the exact moment its owner is watching it, which reads as data loss rather
    than as a failure."""
    _, _, failed = _seed(speech)
    ids = [j["id"] for j in speech.get("/jobs?audio=present").json()["jobs"]]
    assert failed in ids


def test_the_kind_filter_is_what_makes_the_default_shippable(speech):
    """A morning of Kokoro presses would push last night's clone off the end of
    a capped listing before any client-side filter ever saw it. A cap applied
    before a filter is a cap on the wrong set, which is why this is served
    here and not in the browser."""
    spoken, heard, cloned = _seed(speech)
    got = speech.get("/jobs?kind=speech").json()
    assert [j["id"] for j in got["jobs"]] == [spoken]
    got = speech.get("/jobs?kind=speech,transcribe").json()
    assert {j["id"] for j in got["jobs"]} == {spoken, heard}


def test_counts_are_computed_before_the_filter(speech):
    """The counts are what stop a default that hides rows from reading as data
    loss. "Everything (412)" has to be legible without asking a second time."""
    _seed(speech)
    got = speech.get("/jobs?kind=speech").json()
    assert len(got["jobs"]) == 1
    assert got["counts"]["all"] == 3
    assert got["counts"]["speech"] == 1
    assert got["counts"]["transcribe"] == 1
    assert got["counts"]["clone"] == 1
    assert got["counts"]["never"] == 2


def test_an_unknown_filter_value_is_refused_rather_than_matching_nothing(speech):
    """A filter that silently matches nothing looks exactly like an empty
    stack, and somebody will conclude their jobs are gone."""
    r = speech.get("/jobs?audio=vanished")
    assert r.status_code == 400
    assert "vanished" in r.text


def test_no_parameters_answers_what_it_always_did_plus_the_counts(speech):
    created = speech.post("/jobs", json={"text": "One short line.",
                                         "voice": "default"}).json()
    _wait(speech, created["id"])
    got = speech.get("/jobs").json()
    assert [j["id"] for j in got["jobs"]] == [created["id"]]
    assert got["counts"]["all"] == 1
    assert got["truncated"] is False


# ------------------------------------------------------------ the switches --


def test_a_service_that_is_not_accepting_records_says_so_once(speech,
                                                              monkeypatch):
    """404, and the senders drop the record and carry on. Switching this off
    must not make another service noisy or slow."""
    from app import main

    monkeypatch.setattr(main, "RUNLOG_ACCEPT", False)
    r = speech.post("/runs", json={"kind": "speech", "service": "tts"})
    assert r.status_code == 404


def test_a_sender_in_a_loop_is_refused_before_the_disk_fills(speech,
                                                             monkeypatch):
    from app import main

    monkeypatch.setattr(main, "RUNLOG_RATE", 3)
    main._runlog_hits.clear()
    codes = [speech.post("/runs", json={"kind": "speech", "service": "tts"}).status_code
             for _ in range(5)]
    assert codes[:3] == [201, 201, 201]
    assert codes[3:] == [429, 429]


def test_the_ceiling_is_not_a_scandir_per_write(speech, monkeypatch):
    """A count per write is O(n) per write and quadratic over a batch, on the
    same disk the audio is on. The count is cached on a timer."""
    from app import main

    calls = {"n": 0}
    real = Path.glob

    def counted(self, pattern):
        if self.name == "runs":
            calls["n"] += 1
        return real(self, pattern)

    monkeypatch.setattr(Path, "glob", counted)
    for _ in range(10):
        assert speech.post("/runs", json={"kind": "speech",
                                          "service": "tts"}).status_code == 201
    assert calls["n"] <= 1, f"{calls['n']} directory walks for 10 records"


def test_switching_the_text_off_keeps_the_length(speech, monkeypatch):
    """Somebody who does not want a transcript of every voice note on disk
    still wants to know the run happened and how big it was."""
    from app import main

    monkeypatch.setattr(main, "RUNLOG_TEXT", False)
    created = speech.post("/jobs", json={"text": "A line that will not be kept.",
                                         "voice": "default"}).json()
    _wait(speech, created["id"])
    on_disk = json.loads(main._sidecar(created["id"]).read_text())
    assert "text" not in on_disk
    assert on_disk["chars"] == len("A line that will not be kept.")
