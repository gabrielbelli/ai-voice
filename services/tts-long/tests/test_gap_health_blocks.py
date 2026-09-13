"""SPRING MAY LIE ABOUT ITSELF AND THE SERVER MUST STILL WORK.

`install_health` registers `/health` as `async def` and says so in capitals in
its own docstring: `details` must not block, because it runs on the event loop.
`_health()` breaks that on the last field it added. `state["runner"].snapshot()`
is a cache-missing BLOCKING call that makes up to two `_request`s with no
`timeout=`, so each of them gets `cfg.timeout` -- thirty seconds. The cache is
five seconds and the container healthcheck interval is sixty, so EVERY
healthcheck is a cache miss and a LAN round trip to a desktop that is allowed to
be switched off.

The half that makes this more than a slow probe is the second test below. While
`/health` sits on that socket, the loop is not running anything else, so `/jobs`,
`/v1/audio/speech` and the SSE routes are all held behind a status panel. A
gaming PC that drops packets rather than refusing them therefore stops a service
that is perfectly able to speak.

NOTHING HERE OPENS A SOCKET AND NOTHING HERE SLEEPS. The runner is a fake whose
`snapshot()` blocks on a `threading.Event` -- which is what a dropped packet
looks like from this side -- and every wait in the file has a bound.
"""

from __future__ import annotations

import threading
from time import perf_counter

import pytest
from starlette.testclient import TestClient

from conftest import _build

# How long the fake runner holds the caller for. The real figure is thirty
# seconds per request and the measured cost was three; a second is enough to
# separate "waited on it" from "did not" by a factor of four and cheap enough
# that a failing run is over before anyone looks away.
SLOW = 1.0


class _SlowRunner:
    """A runner that accepts the connection and then says nothing.

    That is the failure mode this has to survive: spring DROPPING packets
    rather than refusing them. A refused connection comes back in a
    millisecond; a dropped one costs the full timeout, and the full timeout is
    what `snapshot()` asks for.

    Only the two methods this path reaches are implemented. `offer()` is here
    because the runner lane's probe reads `state["runner"]` on its own thread
    every round, and it must not be the thing that blocks or the thing that
    routes a job at a machine that does not exist.
    """

    def __init__(self, budget: float, freed: threading.Event | None = None) -> None:
        self.budget = budget
        self.entered = threading.Event()
        self.freed = freed if freed is not None else threading.Event()
        # Did anything else get to run while this call was outstanding? Set
        # from the answer to `wait`, so it is a fact about the loop rather than
        # a reading of a clock.
        self.loop_stayed_free = False

    def snapshot(self, max_age: float = 5.0) -> dict:
        self.entered.set()
        self.loop_stayed_free = self.freed.wait(self.budget)
        return {"reachable": False, "error": "timed out"}

    def offer(self):
        from app.remote import RunnerOffer

        return RunnerOffer(False, "fake runner: never asked over the network", "")


@pytest.fixture
def blocked(tmp_path, monkeypatch):
    """A client on the real app, plus an event that fires when `/jobs` lands.

    The marker is an ASGI wrapper rather than a route, because the question is
    not "did `/jobs` answer" but "did the event loop ever pick the request up".
    A wrapper is the first thing the loop runs for a request and the last thing
    a blocked loop reaches, so it separates a route that is slow from a service
    that is stopped.

    One `TestClient`, therefore one portal, therefore ONE EVENT LOOP shared by
    both requests -- which is the whole point. Two clients would each get their
    own loop and the file would prove nothing.
    """
    inner = _build(tmp_path, monkeypatch)
    reached_jobs = threading.Event()

    async def marker(scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/jobs":
            reached_jobs.set()
        await inner(scope, receive, send)

    with TestClient(marker) as client:
        yield client, reached_jobs


def _get_in_a_thread(client, path: str, out: dict) -> threading.Thread:
    def run() -> None:
        started = perf_counter()
        out["response"] = client.get(path)
        out["seconds"] = perf_counter() - started

    thread = threading.Thread(target=run, daemon=True, name=f"GET {path}")
    thread.start()
    return thread


def test_a_slow_runner_delays_every_other_route(blocked):
    """THE WHOLE SERVICE STOPS, NOT ONLY THE STATUS PANEL.

    Two numbers, measured on one loop. `/health` waiting on the runner is the
    premise and it is asserted first, so a fake that failed to bite cannot be
    mistaken for a service that behaved. `/jobs` is the contract: it shares
    nothing with the runner, it is issued only once `/health` is already inside
    `snapshot()`, and it must come back while `/health` is still waiting.
    """
    import app.main as main

    client, _ = blocked
    runner = _SlowRunner(SLOW)
    main.state["runner"] = runner
    try:
        health: dict = {}
        thread = _get_in_a_thread(client, "/health", health)
        assert runner.entered.wait(10.0), "the fake runner was never asked"

        started = perf_counter()
        jobs = client.get("/jobs")
        jobs_seconds = perf_counter() - started

        thread.join(timeout=10.0)
        assert not thread.is_alive(), "/health never came back"

        assert health["response"].status_code == 200
        assert jobs.status_code == 200
        assert health["seconds"] >= SLOW * 0.8, (
            f"/health answered in {health['seconds']:.2f}s while the runner "
            f"held its caller for {SLOW:.1f}s: the fake never bit, so this run "
            "measured nothing")
        assert jobs_seconds < SLOW * 0.25, (
            f"GET /jobs took {jobs_seconds:.2f}s while /health was waiting "
            f"{health['seconds']:.2f}s on the runner. /jobs does not touch the "
            "runner; it was delayed because _health() calls "
            "RunnerClient.snapshot() on the event loop, so a desktop that "
            "drops packets stops speech, the listing and the SSE routes too")
    finally:
        main.state["runner"] = None


def test_the_event_loop_keeps_running_while_the_runner_is_asked(blocked):
    """THE SAME DEFECT WITHOUT A CLOCK, so a slow machine cannot excuse it.

    `snapshot()` is released by the arrival of the `/jobs` request itself. If
    the loop is free, the wrapper runs, the event fires and `/health` returns
    at once; if the loop is inside `snapshot()`, nothing runs, the request is
    never picked up and the wait ends on its bound instead. The assertion is a
    boolean about what the loop did, not a threshold.
    """
    import app.main as main

    client, reached_jobs = blocked
    runner = _SlowRunner(SLOW, freed=reached_jobs)
    main.state["runner"] = runner
    try:
        health: dict = {}
        thread = _get_in_a_thread(client, "/health", health)
        assert runner.entered.wait(10.0), "the fake runner was never asked"

        client.get("/jobs")
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "/health never came back"

        assert runner.loop_stayed_free, (
            "the GET /jobs request was still unread by the event loop "
            f"{SLOW:.1f}s after /health entered the runner call. /health is "
            "async and _health() blocks on RunnerClient.snapshot(), so the "
            "loop accepted no other work for the length of a LAN timeout")
    finally:
        main.state["runner"] = None
