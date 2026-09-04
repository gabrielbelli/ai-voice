

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
