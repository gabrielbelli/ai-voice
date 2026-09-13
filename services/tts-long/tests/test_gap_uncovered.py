"""Three lines that carry loud comments and had no test behind them.

An audit mutated each of these and every suite in the repository stayed green,
which means the comment above the line was the only thing defending it.

  * `_estimate` divides by whatever the CHOSEN lane achieves. Reverting it to
    the local rate -- which is what it did before lanes, and what makes the
    number look reasonable -- promises twenty-one minutes for a job that takes
    seven, and the caller who was told twenty-one minutes goes away.
    `test_dispatch.py` pins `Dispatcher.estimate_for`; nothing asked what a
    CALLER is actually told, which is the only number that leaves the process.
  * `segments_from_runner` and `fell_back_from` are both ACCUMULATED and both
    say so in capitals. Assignment passes every existing test, because no test
    ever hands the same job back twice.

NOTHING HERE OPENS A SOCKET OR LOADS A MODEL.
"""

from __future__ import annotations

import time

import pytest

from test_remote import FakeClient, runner


# ------------------------------------------------- what the caller is told ---


def test_the_estimate_in_the_202_is_about_the_lane_the_job_will_go_to(speech):
    """THE DEFECT THIS PREVENTS: a promise about the wrong machine.

    `estimate_for` is asserted in test_dispatch.py, but `_estimate` is the
    function that turns it into the number a client receives -- and its
    predecessor divided by the LOCAL rate whatever the destination. With a
    runner open, a long job is quoted at the runner's rate; with the lane shut,
    the same text is quoted at this host's, which is several times longer.
    Both figures are computed here from the rates themselves, so a `_estimate`
    that quietly went back to the local rate cannot satisfy the first one.
    """
    import app.main as main

    # Long enough that the two rates cannot be confused with each other: at
    # 0.23x here and 0.70x there the same speech is three times the wait.
    text = "One short line, spoken plainly. " * 200
    # THE CHUNKED LENGTH, which is what `_enqueue` charges for: the chunker
    # trims, so len(text) is a few seconds of speech more than the job is.
    work = main.speech_seconds(sum(len(t) for t, _ in main._segments(text, None)))
    here = work / main.rate_for("local").value
    there = work / main.rate_for("runner").value
    assert here > 2 * there, "the seeded rates are too close to tell apart"

    fake = FakeClient(segments=1)
    with runner(fake):            # the fixture opens the lane and zeroes the hop
        quoted = speech.post("/jobs", json={"text": text,
                                            "voice": "default"}).json()
        promised = quoted["estimated_seconds"]
        speech.delete(f"/jobs/{quoted['id']}")

    assert promised == pytest.approx(there, abs=2), (
        "a job bound for the runner was quoted " + str(promised) + "s, and the "
        "runner would take " + str(round(there)) + "s; the local lane would "
        "have taken " + str(round(here)) + "s, which is the number this used "
        "to give whatever machine the job was going to")

    # And with no lane but this one, the honest answer is this host's.
    local_quote = speech.post("/jobs", json={"text": text,
                                             "voice": "default"}).json()
    speech.delete(f"/jobs/{local_quote['id']}")
    assert local_quote["estimated_seconds"] == pytest.approx(here, abs=2), (
        "with the runner shut the caller must be told what this host will "
        "take, not what a machine it cannot reach would have")


# ------------------------------------------------- the accumulation sites ---


class _Yields:
    """A backend that hands the job back, having delivered some of it."""

    def __init__(self, delivered: int) -> None:
        self.delivered = delivered
        self.waited = 0.0

    def speak_segments(self, *a, **k):
        from app.remote import RemoteYield

        raise RemoteYield("the machine was taken back", delivered=self.delivered)


def _job() -> dict:
    """The keys `_run` touches before it reaches the backend, and no others."""
    return {"id": "0123456789abcdef", "status": "queued", "cancelled": False,
            "created_at": time.time(), "segments": [("One short line.", 0.0)],
            "language": "en", "exaggeration": 0.5, "cfg_weight": 0.5,
            "temperature": 0.8, "reference": None, "format": "wav",
            "voice": "default", "stream": None}


def test_two_lanes_handing_the_same_job_back_are_both_recorded(speech, monkeypatch):
    """THE DEFECT THIS PREVENTS: a job that visited two machines reporting one.

    Every one of these three fields is written `prior + new` and every one of
    them says so in capitals, and no test had ever handed the same job back
    twice -- so a plain assignment on any of them passed the whole suite. What
    is lost is the only answer to "why did this take so long": that it started
    on the card, was handed back, was handed back again, and finished here.
    """
    import app.main as main
    from app.dispatch import YIELDED

    job = _job()
    monkeypatch.setattr(main, "_backend_for",
                        lambda job, lane: _Yields(2 if lane == "runner" else 3))

    assert main._execute_on_lane(job, "runner") == YIELDED
    assert main._execute_on_lane(job, "runner_cpu") == YIELDED

    assert job["segments_from_runner"] == 5, (
        "two attempts delivered 2 and 3 segments and the job reports "
        + str(job.get("segments_from_runner")) + ": each attempt's work is "
        "ADDED, and a plain assignment reports only the last one")
    assert job["fell_back_from"] == "runner, runner_cpu", (
        "the job left two lanes and remembers " + repr(job.get("fell_back_from")))
    reason = job.get("fell_back_reason") or ""
    assert reason.count("taken back") == 2, (
        "both refusals are kept -- a record that holds only the last one "
        "cannot say whether the card was busy or missing; got " + repr(reason))
    assert job["status"] == "queued", "a handed-back job is waiting, not running"
