"""The run record: what leaves this service after the audio has.

Instant speech keeps no file. The audio goes into the response and the run is
gone, so until this existed the only two engines a person actually uses every
day — Kokoro here and Parakeet in stt-stack — were the two with no history at
all, while the slow one behind tts-long had kept a record of every job since
the beginning.

Two properties are worth more than the rest and every test here is arranged
around one of them:

  IT MUST NOT REACH THE CALLER. Not the latency, not the failure, not the
  exception. /speak and /v1/audio/speech are sync handlers on AnyIO's pool, so
  a POST made inline would be added to the reply of somebody sitting in front
  of it, and a tts-long that is merely slow would become a tts-stack that is
  slow to speak.

  THE STREAM MUST BE RECORDED TOO. `samples` and `compute` are locals of the
  SSE generator and starlette has already sent the headers before it is first
  pulled, so no header can carry them: a streamed run was the one shape this
  service could not account for, and a client that hangs up halfway is how it
  usually ends.

No socket is bound anywhere here. urllib.request.urlopen is replaced with a
receiver that records what it was handed, which is both faster and honest about
what is under test: the body, the threading and the failure handling are this
repo's, and HTTP itself is not.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request

import pytest
from starlette.testclient import TestClient
from voice_common.conformance import module_app
from voice_common.runlog import MAX_QUEUED, SERVER_ONLY, RunLog

from test_openai_speech import MULTI_CHUNK, FakeSynth

# THE ONE THING BOTH HALVES MAY TREAT AS AUTHORITATIVE. tts-long codes its
# POST /runs against this file and so does this service; if the two disagree,
# the fixture is right and the code is wrong. Asserted to exist rather than
# skipped over — a missing contract is the failure, not a reason to pass.
FIXTURES = (pathlib.Path(__file__).resolve().parents[3]
            / "packages" / "common" / "tests" / "fixtures" / "run_records.json")


def contract(kind: str) -> dict[str, object]:
    assert FIXTURES.is_file(), (
        f"{FIXTURES} is the shared record contract and it is not there; "
        "without it this service and tts-long are each guessing at the other")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))[kind]


class Receiver:
    """tts-long's POST /runs, with no socket in the way.

    Records every body it is handed BEFORE it sleeps or fails, so a test can
    tell "the record never left" from "the receiver never answered" — the two
    have opposite fixes and look identical from the caller.
    """

    def __init__(self, *, delay: float = 0.0,
                 error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.bodies: list[dict[str, object]] = []
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.entered = threading.Event()

    def __call__(self, request, timeout: float | None = None):  # noqa: ANN001
        self.urls.append(request.full_url)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        self.bodies.append(json.loads(request.data.decode("utf-8")))
        self.entered.set()
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return contextlib.nullcontext()

    def only(self) -> dict[str, object]:
        assert len(self.bodies) == 1, f"expected one record, got {self.bodies}"
        return self.bodies[0]


def drained(log: RunLog, expected: int = 1, timeout: float = 5.0) -> dict:
    """Wait for the sender thread to have finished with `expected` records.

    Polls the counters /health publishes rather than the queue, because those
    are the numbers an operator reads and a test that watched a private
    attribute could pass while /health said nothing happened.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = log.stats()
        if stats["sent"] + stats["dropped"] >= expected:
            return stats
        time.sleep(0.005)
    raise AssertionError(f"no record was sent or dropped in {timeout}s: "
                         f"{log.stats()}")


@pytest.fixture
def receiver(monkeypatch: pytest.MonkeyPatch) -> Receiver:
    fake = Receiver()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def sender(**kwargs: object) -> RunLog:
    return RunLog(url="http://tts-long:8002", host="orko", service="tts",
                  engine="kokoro", **kwargs)  # type: ignore[arg-type]


class Service:
    """One built app, with the module object that app was actually built from.

    module_app drops app.* from sys.modules and imports it again, so a module
    imported at the top of this file is NOT the module the app under test is
    running — its `runlog` and its `state` belong to a discarded copy. Every
    test here reaches the live one through this.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch,
                 log: RunLog | None = None) -> None:
        app = module_app("app.main")()
        import app.main as fresh

        self.main = fresh
        self.log = log if log is not None else sender()
        self.synth = FakeSynth()
        fresh.state["synth"] = self.synth
        monkeypatch.setattr(fresh, "runlog", self.log)
        self.client = TestClient(app)

    def speech(self, **body: object):
        body.setdefault("model", "tts-1")
        body.setdefault("input", "One. Two. Three.")
        body.setdefault("voice", "fable")
        return self.client.post("/v1/audio/speech", json=body)

    def health(self) -> dict:
        return json.loads(self.client.get("/health").text)


@pytest.fixture
def service(receiver: Receiver, monkeypatch: pytest.MonkeyPatch) -> Service:
    """The app with a fake model and a sender pointed at the fake receiver.

    The lifespan is never entered, for the reason every other file here gives:
    a suite that pulled 340 MB of weights into CI would be switched off within
    a week. `runlog` is rebound rather than reconfigured, which is the whole
    reason every call site reads the module global at call time.
    """
    return Service(monkeypatch)


# --- the contract ----------------------------------------------------------

def test_a_run_is_recorded_with_the_fields_the_contract_names(
        service: Service, receiver: Receiver) -> None:
    """The key set is the fixture's, exactly, and the values mean what it says.

    Compared against the shared file rather than a list written out here: a
    list in this repo can be edited to match whatever the emitter grew into,
    which is how a contract stops being one.
    """
    assert service.speech(input="Hello there.").status_code == 200
    drained(service.log)

    body = receiver.only()
    assert receiver.urls == ["http://tts-long:8002/runs"]
    assert receiver.headers[0]["content-type"] == "application/json"
    assert set(body) == set(contract("speech"))

    assert body["kind"] == "speech"
    assert body["service"] == "tts"
    assert body["engine"] == "kokoro"
    assert body["host"] == "orko"
    assert body["route"] == "/v1/audio/speech"
    assert body["client"] == "openai"
    assert body["model_requested"] == "tts-1"
    assert body["status"] == "done"
    assert body["backend"] == "local"
    assert body["text"] == "Hello there."
    assert body["chars"] == len("Hello there.")
    assert body["voice"] in service.synth.voices
    assert body["format"] == "mp3"
    assert body["audio_seconds"] > 0
    assert body["realtime_factor"] > 0
    # An instant run does not queue, so the two are the same instant and the
    # third is that instant plus the compute the record already reports.
    assert body["created_at"] == body["started_at"]
    assert body["finished_at"] >= body["started_at"]


def test_a_sender_never_sets_a_field_tts_long_owns(
        service: Service, receiver: Receiver) -> None:
    """`path` in particular would be a sender choosing the argument to open().

    tts-long answers 400 to any of these rather than dropping them, so a
    service that grew one would lose every record it sent with nothing but a
    rejection count to show for it.
    """
    assert service.speech().status_code == 200
    assert service.client.post(
        "/speak", json={"text": "Hello.", "format": "wav"}).status_code == 200
    drained(service.log, expected=2)
    assert len(receiver.bodies) == 2
    for body in receiver.bodies:
        assert SERVER_ONLY.isdisjoint(body), (
            f"{sorted(SERVER_ONLY & set(body))} is tts-long's to set")


def test_a_field_tts_long_owns_is_dropped_before_it_can_be_a_400(
        service: Service, receiver: Receiver, caplog) -> None:  # noqa: ANN001
    """The guard under the fence above, asserted where it can actually fail.

    The fence asserts the POSTED body, which the guard has already cleaned, so
    it cannot see the guard itself. This can: tts-long answers 400 to any of
    these, and a sender that grew one would lose every record it sent — so the
    field goes, the record still lands, and the key is NAMED in the log rather
    than vanishing quietly.
    """
    with caplog.at_level("WARNING", logger="voice_common.runlog"):
        service.log.record(kind="speech", status="done",
                           path="/models/../etc/passwd", text="Hello.")
    drained(service.log)

    body = receiver.only()
    assert "path" not in body
    assert body["text"] == "Hello."
    assert "path" in caplog.text


def test_the_native_route_is_recorded_too(service: Service,
                                          receiver: Receiver) -> None:
    """/speak is the route the page uses, and it has its own log line.

    The build order names one call site "beside the log line"; this service has
    three log lines and recording only one of them would leave whole routes
    absent from a listing that says it holds everything.
    """
    response = service.client.post("/speak", json={"text": "One. Two.",
                                                   "voice": "bm_george",
                                                   "format": "wav",
                                                   "speed": 1.0})
    assert response.status_code == 200
    drained(service.log)

    body = receiver.only()
    assert body["route"] == "/speak"
    # Null rather than a guess: the page, a script and a shell all send this
    # same body, and a user agent read as a client would be stored as a fact.
    assert "client" not in body
    assert body["voice"] == "bm_george"
    assert body["format"] == "wav"


def test_the_segment_offsets_reach_the_record(service: Service,
                                              receiver: Receiver) -> None:
    """The same list X-Segment-Offsets carries, so a reader can follow the text
    back without the response that has already been thrown away."""
    response = service.client.post("/speak", json={
        "segments": [{"text": "One.", "pause_after": 0.5},
                     {"text": "Two.", "pause_after": 0.0}],
        "format": "wav"})
    assert response.status_code == 200
    drained(service.log)

    body = receiver.only()
    assert body["offsets"] == [
        float(o) for o in response.headers["X-Segment-Offsets"].split(",")]
    assert body["text"] == "One. Two."


# --- it must not reach the caller ------------------------------------------

def test_the_response_does_not_wait_on_the_log(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The receiver takes five seconds. The reply must not.

    This is the whole reason for the thread and the queue. Written as a
    measurement rather than a claim about the code, because the failure it
    guards — somebody making the POST inline "just for now" — reads perfectly
    well and is invisible until tts-long is slow.
    """
    slow = Receiver(delay=5.0)
    monkeypatch.setattr(urllib.request, "urlopen", slow)
    service = Service(monkeypatch)

    started = time.monotonic()
    assert service.speech(input="Hello there.").status_code == 200
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"the reply took {elapsed:.2f}s against a receiver sleeping for 5s; "
        "the record is on the caller's clock")
    # And it really was sent, so this is not passing because nothing was posted.
    assert slow.entered.wait(2.0)


def test_a_receiver_that_refuses_never_reaches_the_caller(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 is what TTS_RUNLOG_ACCEPT=0 or an older tts-long answers.

    The audio is unaffected and the reason is visible in /health, which is the
    only shape of failure a log is allowed to have.
    """
    refusing = Receiver(error=urllib.error.HTTPError(
        "http://tts-long:8002/runs", 404, "Not Found", {}, None))  # type: ignore[arg-type]
    monkeypatch.setattr(urllib.request, "urlopen", refusing)
    service = Service(monkeypatch)

    response = service.speech(input="Hello there.")
    assert response.status_code == 200
    assert response.content

    deadline = time.monotonic() + 5.0
    while service.log.last_error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service.log.stats()["sent"] == 0
    assert "404" in str(service.log.stats()["last_error"])
    assert service.health()["runlog"]["last_error"]


def test_an_unset_url_records_nothing_and_raises_nothing(
        receiver: Receiver, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default, and the deployment rule it exists for.

    This stack has to work completely with tts-long stopped, unplugged or
    lying about itself. Unset means no queue, no thread and no POST — not a
    connection refused once a request, which is the shape this would have if
    the URL had a default.
    """
    service = Service(monkeypatch, log=RunLog(url=None, host="orko",
                                              service="tts", engine="kokoro"))

    assert service.speech().status_code == 200
    assert service.client.post("/speak",
                               json={"text": "Hello."}).status_code == 200
    assert receiver.bodies == []
    # THIS sender's thread, not "a thread called runlog": the senders other
    # tests in this file build are daemons that outlive them, so enumerating
    # every thread in the process would assert about somebody else's.
    assert service.log._thread is None, (
        "a disabled sender started a sender thread; unset has to cost a None "
        "check and nothing else")
    health = service.health()
    assert health["runlog"]["url"] is None
    assert health["runlog"]["sent"] == 0
    assert health["host_label"] == "orko"


def test_a_full_queue_drops_and_says_so_in_health(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped record must be visible as a gap, not invisible as an absence.

    Dropping is the right trade — the alternative is holding memory for a
    receiver that may never come back — but only while somebody can see it
    happening. A row that is simply not there says nothing at all.
    """
    blocked = threading.Event()

    def stuck(request, timeout: float | None = None):  # noqa: ANN001, ARG001
        blocked.wait(10.0)
        return contextlib.nullcontext()

    monkeypatch.setattr(urllib.request, "urlopen", stuck)
    service = Service(monkeypatch)
    try:
        for index in range(MAX_QUEUED + 20):
            service.log.record(kind="speech", status="done", text=f"{index}")

        assert service.log.stats()["dropped"] > 0
        assert service.health()["runlog"]["dropped"] > 0
        # The runs themselves are unaffected; it is their history that is lost.
        assert service.speech().status_code == 200
    finally:
        blocked.set()


def test_the_text_can_be_left_behind_without_losing_the_record(
        monkeypatch: pytest.MonkeyPatch, receiver: Receiver) -> None:
    """RUNLOG_TEXT=0 for a deployment that wants the timings and not the words.

    `chars` still goes, so the length survives the switch: the alternative is
    an operator choosing between sending every utterance to a store and having
    no history at all.
    """
    service = Service(monkeypatch, log=sender(send_text=False))

    assert service.speech(input="Something private.").status_code == 200
    drained(service.log)
    body = receiver.only()
    assert "text" not in body
    assert body["chars"] == len("Something private.")


# --- the stream ------------------------------------------------------------

def test_a_streamed_run_is_recorded_when_the_client_hangs_up(
        service: Service, receiver: Receiver) -> None:
    """The case that left nothing behind at all, and the usual way a stream ends.

    A client that closes the connection never reaches the `done` frame, and the
    close arrives as GeneratorExit at a yield — a BaseException, so the
    generator's `except Exception` never sees it. Without the finally the run
    is simply absent: no header carries `samples` or `compute`, because
    starlette sent http.response.start before this generator was first pulled.

    The partial numbers are the honest ones. That audio was generated and that
    time was spent, and reporting nothing would say the run never happened.
    """
    synth = service.synth
    chunks = synth.plan(MULTI_CHUNK, "en-us")
    assert len(chunks) > 2, "a one-chunk stream cannot be abandoned partway"

    stream = service.main._sse_body(synth, chunks, "bm_george", "en-gb", 1.0,
                                    "pcm", len(MULTI_CHUNK), text=MULTI_CHUNK,
                                    model_requested="tts-1")
    assert next(stream).startswith(b'data: {"type":"speech.audio.delta"')
    stream.close()
    drained(service.log)

    body = receiver.only()
    assert body["status"] == "failed"
    assert "closed the stream" in str(body["error"])
    assert len(synth.calls) < len(chunks), "the stream was not abandoned"
    assert body["audio_seconds"] > 0, (
        "a stream somebody walked away from still generated audio; rounding "
        "it to nothing says the run never happened")
    assert body["route"] == "/v1/audio/speech"
    # No usage: the done frame is where that number is computed and it never
    # left, so claiming one would be inventing it.
    assert "usage" not in body


def test_a_streamed_run_that_finishes_is_recorded_as_done(
        service: Service, receiver: Receiver) -> None:
    """The other ending, and the numbers no header on this route can carry."""
    response = service.speech(input=MULTI_CHUNK, stream_format="sse",
                              response_format="pcm")
    assert response.status_code == 200
    assert b"speech.audio.done" in response.content
    drained(service.log)

    body = receiver.only()
    assert body["status"] == "done"
    assert "error" not in body
    assert body["audio_seconds"] > 0
    assert body["usage"]["total_tokens"] > 0
    assert body["text"] == MULTI_CHUNK


def test_a_stream_that_fails_is_recorded_as_failed(
        service: Service, receiver: Receiver) -> None:
    """The in-band error frame is what the client sees; this is what is kept.

    A synthesis that fails partway through a stream answered 200 with an error
    frame and left no trace anywhere else — the one failure mode of this
    service that a person could not find afterwards.
    """
    service.synth.fail_on = 2
    response = service.speech(input=MULTI_CHUNK, stream_format="sse",
                              response_format="pcm")
    assert response.status_code == 200
    assert b'"error"' in response.content
    drained(service.log)

    body = receiver.only()
    assert body["status"] == "failed"
    assert "synthesis failed" in str(body["error"])
