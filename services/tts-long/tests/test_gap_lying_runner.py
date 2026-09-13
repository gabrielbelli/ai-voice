"""Spring answers, and what it says is not what was agreed.

THE THREE HOLES THESE TESTS CLOSE, all measured end to end before the fix.
`_execute_on_lane` knows two exceptions -- `RemoteYield` and
`RemoteUnavailable` -- and treats both as facts about the LANE: the job goes
back to the head of the queue, the lane is cooled, and this host speaks it.
Anything else falls into the blanket handler below them, which marks the JOB
failed and cools nothing. So the next job is dispatched at the same broken
runner and fails the same way, and the one after that.

  * `job()` ran `json.loads` on a 200 body. A runner answering HTML -- a proxy,
    a captive portal, a different program on the port -- raises JSONDecodeError,
    which is a ValueError. Measured: the row read `failed`, the error was
    "Expecting value: line 1 column 1 (char 0)", and `cooling` was 0.0.
  * `submit()` ran `json.loads(data).get("job_id")`. A runner answering a JSON
    list raises AttributeError one line later, in the caller. Measured.
  * `_request` caught `OSError` and its own comment claimed "there is only one
    name to catch now". THAT WAS FALSE. `http.client.HTTPException` does not
    inherit from OSError, so `IncompleteRead`, `BadStatusLine` and `LineTooLong`
    escaped as themselves -- and IncompleteRead is exactly what a desktop going
    to sleep DURING `resp.read()` produces, which is the commonest way this
    particular machine leaves.

NOTHING HERE OPENS A SOCKET. The transport tests replace `_connect`, one layer
below the seam every other test in test_remote uses, so the real `_request` --
the code being tested -- runs unchanged.

`RemoteUnavailable` AND `RunnerClient` ARE IMPORTED INSIDE EVERY FUNCTION, NEVER
AT THE TOP OF THIS FILE, and test_gap_lane_loss.py's docstring says why: conftest
deletes every `app.*` from sys.modules and reimports, so a class bound at
collection time is a DIFFERENT class object from the one the app is holding.
Binding it here made `pytest.raises(RemoteUnavailable)` match a stale class
against a fresh one -- green on its own, red in the suite, and the failure reads
like the fix does not work.
"""

from __future__ import annotations

import http.client
import json

import pytest
from test_remote import _wait, runner


def _fresh():
    """app.main's OWN remote module, whatever conftest last reimported."""
    import app.remote as remote

    return remote


def _cfg():
    return _fresh().RunnerConfig(host="runner.invalid", service="chatterbox",
                                 poll=0.0)


# ---------------------------------------------------------- the transport ---


def test_an_http_exception_is_not_an_oserror():
    """The premise the false comment got wrong, asserted rather than believed.

    If this ever becomes true upstream, the tuple below is redundant and the
    tests around it are testing nothing -- which is worth being told about.
    """
    assert not issubclass(http.client.HTTPException, OSError)


class _Conn:
    """A connection that dies at a chosen point, the way a sleeping desktop does."""

    def __init__(self, raises):
        self._raises = raises
        self.closed = False

    def request(self, *a, **kw):
        pass

    def getresponse(self):
        return self

    def read(self):
        raise self._raises

    @staticmethod
    def getheaders():
        return []

    status = 200

    def close(self):
        self.closed = True


@pytest.mark.parametrize("gone", [
    # A desktop suspending in the middle of resp.read(). THE COMMONEST WAY
    # THIS MACHINE LEAVES, and the one that used to fail the job.
    http.client.IncompleteRead(b"half a segment"),
    http.client.BadStatusLine("\\x16\\x03\\x01"),
    http.client.LineTooLong("header line"),
    # And the half that was always caught, so the fix is a widening rather
    # than a replacement.
    ConnectionResetError(104, "Connection reset by peer"),
])
def test_losing_the_machine_is_one_class_however_it_is_lost(gone, monkeypatch):
    """Every one of these means the same thing: this job cannot run over there.

    A caller that has to enumerate exception hierarchies to find out whether it
    lost the lane will miss one, and the one it misses fails a job the local
    CPU was going to speak perfectly well.
    """
    client = _fresh().RunnerClient(_cfg())
    conn = _Conn(gone)
    monkeypatch.setattr(client, "_connect", lambda timeout=None: conn)

    with pytest.raises(_fresh().RemoteUnavailable) as lost:
        client.job("remote-job-1")
    # NAMED, not an empty string. Several of these stringify to nothing at all,
    # and "GET /v1/services/chatterbox/jobs/remote-job-1: " on a job row tells
    # nobody anything.
    assert type(gone).__name__ in str(lost.value)
    assert conn.closed, "the connection was leaked on the failure path"


def test_a_close_that_itself_raises_does_not_replace_the_real_reason(monkeypatch):
    """A socket whose peer vanished mid-response can fail BOTH the read and the
    close, and an exception raised from a `finally` replaces the one being
    raised with one nothing handles."""
    client = _fresh().RunnerClient(_cfg())
    conn = _Conn(http.client.IncompleteRead(b""))
    conn.close = lambda: (_ for _ in ()).throw(OSError(107, "Transport endpoint"))
    monkeypatch.setattr(client, "_connect", lambda timeout=None: conn)

    with pytest.raises(_fresh().RemoteUnavailable) as lost:
        client.job("remote-job-1")
    assert "IncompleteRead" in str(lost.value)


# ------------------------------------------------------------ the bodies ----


def _answering(body: bytes, status: int = 200):
    client = _fresh().RunnerClient(_cfg())
    client._request = lambda *a, **kw: (status, {}, body)  # type: ignore[method-assign]
    return client


@pytest.mark.parametrize("body", [
    b"<html><head><title>502 Bad Gateway</title></head></html>",
    b"",
    b"Service Unavailable",
])
def test_a_two_hundred_that_is_not_json_is_a_lost_lane_not_a_failed_job(body):
    """MEASURED: the row read `failed`, the error was "Expecting value: line 1
    column 1 (char 0)", and the lane was never cooled."""
    with pytest.raises(_fresh().RemoteUnavailable) as lost:
        _answering(body).job("remote-job-1")
    assert "not JSON" in str(lost.value)
    assert "chatterbox" in str(lost.value), "the message does not say who lied"


@pytest.mark.parametrize("body", [b"[]", b'["job_id"]', b"7", b"null", b'"x"'])
def test_a_json_value_where_an_object_was_agreed_is_a_lost_lane(body):
    """`.get` on a list is an AttributeError one line later, in the caller,
    where it is even further from anything that knows what a lane is."""
    for call in (lambda c: c.submit({}, "key"), lambda c: c.job("remote-job-1")):
        with pytest.raises(_fresh().RemoteUnavailable) as lost:
            call(_answering(body))
        assert "object was agreed" in str(lost.value)


def test_offer_survives_a_runner_that_is_not_speaking_this_protocol():
    """The probe catches every exception, so this one never failed a job -- but
    it did make the lane's reason read `unreachable: JSONDecodeError`, which
    sends somebody looking for a network fault on a machine that answered."""
    with pytest.raises(_fresh().RemoteUnavailable) as lost:
        _answering(b"<html>").offer()
    assert "not speaking this protocol" in str(lost.value)


def test_a_good_body_is_still_a_good_body():
    """The guard must not have narrowed what a working runner may answer."""
    client = _answering(json.dumps({"job_id": "r1", "extra": [1, 2]}).encode())
    assert client.submit({}, "key") == "r1"
    assert client.job("r1")["extra"] == [1, 2]


# ------------------------------------------------------------ end to end ----


def _lying_runner(job_body: bytes):
    """A REAL RunnerClient whose transport answers, and lies on the poll.

    `_request` is replaced and nothing below it is, so `submit`, `job` and the
    whole of `speak_segments` run exactly as they do against a real machine.
    """
    client = _fresh().RunnerClient(_cfg())

    def transport(method, path, body=None, content_type=None, extra=None,
                  timeout=None):
        if method == "POST" and path.endswith("/jobs"):
            return 200, {}, json.dumps({"job_id": "r1"}).encode()
        if path.endswith("/jobs/r1"):
            return 200, {}, job_body
        if path == "/v1/services":
            return 200, {}, json.dumps({
                "gpu_available": True,
                "services": [{"id": "chatterbox", "installed": True,
                              "enabled": True, "available": True,
                              "device": "gpu"}]}).encode()
        return 200, {}, b"{}"

    client._request = transport  # type: ignore[method-assign]
    return client


@pytest.mark.parametrize("job_body", [
    b"<html><title>502 Bad Gateway</title></html>",
    b"[]",
])
def test_a_lying_runner_costs_the_lane_and_never_the_job(speech, job_body):
    """THE WHOLE POINT, END TO END.

    The audio still has to arrive, on this host, and the lane that lied has to
    be cooled -- because the alternative is measured: the row read `failed`,
    `cooling` was 0.0, and the next job went straight back to the same runner.
    """
    with runner(_lying_runner(job_body)) as main:
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])
        cooling = main.dispatch.lanes["runner"].cooldown_until

    assert finished["status"] == "done", (
        "the runner answered nonsense and the job was reported "
        f"{finished['status']!r} ({finished.get('error')}) instead of being "
        "spoken on this host")
    assert finished["audio_seconds"] > 0, "a done job with no audio in it"
    assert finished["backend"] == "local", "nothing re-spoke it here"
    assert finished.get("fell_back") is True
    assert cooling > 0, (
        "the lane that lied was never cooled, so the next job goes there too")
