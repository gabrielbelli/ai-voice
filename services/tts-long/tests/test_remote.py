"""The GPU runner seam, and the four ways it could quietly ruin this service.

NOTHING HERE OPENS A SOCKET. The runner is a fake object with the six methods
RemoteSynth calls, so these run on any machine, offline, in milliseconds. What is
real is everything on this side of the seam: the chooser, the per-backend rate,
the decode, the yield handling and the asset flow.

Every test is named after the mistake it prevents.
"""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from app.remote import (RemoteSynth, RemoteUnavailable, RemoteYield,
                        RunnerConfig, _pinned_context)
from app.synth import SAMPLE_RATE


# ------------------------------------------------------------------ fakes ---


class FakeClient:
    """The six calls RemoteSynth makes, and a script for how to answer them."""

    def __init__(self, cfg=None, *, segments=1, yield_after=None,
                 yield_polls=None, fail=None, tokens=7):
        # poll=0: these tests assert ordering and state transitions, not timing,
        # and a 2 s default would turn the file into a minute of sleeping.
        self.cfg = cfg or RunnerConfig(host="runner.invalid", service="chatterbox", poll=0.0)
        self.submitted: list[tuple[dict, str]] = []
        self.uploaded: list[bytes] = []
        self.cancelled: list[str] = []
        self._segments = segments
        self._yield_after = yield_after
        # How many polls the runner stays queued once it has handed the GPU
        # back. None means for ever, which is the owner who games all evening.
        self._yield_polls = yield_polls
        self._stalled = 0
        self._fail = fail
        self._tokens = tokens
        self._polls = 0
        self.state = (True, "")
        # WHAT THE RUNNER IS SELLING, not only whether it will sell. The chooser
        # reads the cap for the processor rung, because `available: true` at a
        # hundred per cent and at five per cent are the same boolean and a
        # twenty to one difference in delivered speech.
        self.device = "gpu"
        self.cpu_pct = 100

    def speech_state(self):
        return self.state

    def offer(self):
        # Through speech_state, so a subclass that overrides that one method --
        # the runner that is switched off, below -- still behaves like one.
        from app.remote import RunnerOffer
        ready, why = self.speech_state()
        return RunnerOffer(ready, why, device=self.device, cpu_pct=self.cpu_pct)

    def ensure_asset(self, path):
        raw = Path(path).read_bytes()
        self.uploaded.append(raw)
        return hashlib.sha256(raw).hexdigest()

    def submit(self, params, idempotency_key):
        self.submitted.append((params, idempotency_key))
        return "remote-job-1"

    def job(self, job_id):
        if self._fail:
            return {"status": "failed", "artefacts": [], "record": {"error": self._fail}}
        # THE YIELD, modelled the way the runner really behaves: the segments
        # already produced STAY produced and stay listed, the status goes back
        # to `queued`, and progress freezes until the owner leaves. It does not
        # start again from zero, which is the whole reason the lease carries an
        # idempotency key.
        if (self._yield_after is not None
                and self._polls >= self._yield_after
                and (self._yield_polls is None or self._stalled < self._yield_polls)):
            self._stalled += 1
            return {"status": "queued", "artefacts": self._names(self._yield_after)}
        self._polls += 1
        done = min(self._polls, self._segments)
        return {
            "status": "done" if done >= self._segments else "running",
            "artefacts": self._names(done),
            "record": {"input_tokens": self._tokens} if done >= self._segments else None,
        }

    @staticmethod
    def _names(n):
        # The real naming, from services/lib: <job id>.<segment>.f32
        return [f"remote-job-1.{i}.f32" for i in range(n)]

    def artefact(self, job_id, name):
        index = int(name.split(".")[-2])
        audio = np.full(SAMPLE_RATE // 10, 0.1 * (index + 1), dtype="<f4")
        return audio.tobytes()

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def segs(n):
    return [(f"segment {i}", 0.0) for i in range(n)]


# The local job id, bound at construction because speak_segments has to keep
# exactly the signature Synth.speak_segments has. See RemoteSynth.__init__.
JOB = "the-local-uuid"

# The three fields the multilingual checkpoint reads, as one mapping. It is one
# mapping and not three positional floats because `speak_segments` has to keep
# byte for byte the signature the local Synth has -- `_run` calls it
# positionally on whichever backend it was handed -- and a per-field parameter
# is a signature that has to be widened in four places for every engine after
# the third. Which keys exist is the ENGINE's business now, not this call's.
DIALS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}


# --------------------------------------------------- the unconfigured case ---


def test_with_no_runner_configured_nothing_remote_is_constructed(speech):
    """THE PIN THE WHOLE SUITE RESTS ON.

    conftest fakes `Synth._speak`, one method. If `_backend_for` returned
    anything other than the local Synth when no runner is configured, every one
    of the other 90 tests in this repository would quietly stop exercising the
    code it was written to exercise, and would carry on passing. That is the
    worst possible failure mode for a test suite, so it gets its own assertion.
    """
    import app.main as main

    assert main.state.get("runner") is None
    job = {"id": "abc123", "segments": segs(1)}
    assert main._backend_for(job) is main._local_synth("chatterbox")
    # Same change as the route-level test above: the job record now names its
    # backend on every path, so the proof that nothing remote happened is the
    # returned object being the local Synth, not the absence of a label.
    assert job["backend"] == "local", "an unconfigured run must say local"


def test_an_unset_host_means_local_only_not_a_disabled_client():
    """Returning None rather than a disabled object is what keeps the local path
    free of remote code. A truthy-but-off client is how a 'disabled' feature
    ends up making network calls in production."""
    assert RunnerConfig.from_env({}) is None
    assert RunnerConfig.from_env({"TTS_RUNNER_HOST": "   "}) is None
    cfg = RunnerConfig.from_env({"TTS_RUNNER_HOST": "box", "TTS_RUNNER_PORT": "9999"})
    assert cfg is not None and cfg.host == "box" and cfg.port == 9999


# ------------------------------------------------------ the three states -----


@pytest.mark.parametrize("state,expect_warning", [
    ((False, "not_installed"), True),
    ((False, "not_enabled"), True),
    ((False, "no_such_service"), True),
    ((False, "gpu_busy"), False),
    # THE NEW ONE. A runner that declines because somebody is at the keyboard,
    # or because there is not enough free memory to start a 6.5 GiB model, is in
    # the same class as a busy GPU: temporary, nobody's to fix, and not worth
    # waking anyone up about. It must not be logged like a missing service.
    ((False, "machine_busy: signed in and using the machine"), False),
    ((False, "machine_busy"), False),
])
def test_not_installed_is_not_the_same_answer_as_gpu_busy(speech, caplog, state,
                                                          expect_warning):
    """THE DEFECT THIS PREVENTS: a client that cannot tell 'that machine has no
    speech service' from 'somebody is playing a game'.

    Both fall back to the local CPU, so the audio is fine either way, and that
    is exactly why this is easy to get wrong and never notice. The difference is
    what a human is told. 'not installed' will never clear on its own and is
    fixed by one command on the runner; 'gpu busy' clears in a minute or two and
    is nobody's problem. Logging both at the same level trains people to ignore
    the one that matters.
    """
    import app.main as main

    client = FakeClient()
    client.state = state
    main.state["runner"] = client
    probe = main.dispatch.lanes["runner"].probe
    try:
        with caplog.at_level("INFO"):
            probe.once()
        assert probe.ok() is False, "a runner that cannot help must not be used"
        warned = any(r.levelname == "WARNING" for r in caplog.records)
        assert warned is expect_warning
    finally:
        main.state["runner"] = None


def test_a_ready_runner_is_used_and_says_so_on_the_job(speech):
    import app.main as main
    # From app.main's own module graph, not this file's top-level import.
    # conftest deletes every `app.*` from sys.modules and reimports, so the
    # RemoteSynth bound at the top of this file is a DIFFERENT class object from
    # the one app.main just imported, and isinstance against it is always False.
    from app.remote import RemoteSynth as FreshRemoteSynth

    main.state["runner"] = FakeClient()
    try:
        job = {"id": "abc123"}
        assert isinstance(main._backend_for(job, "runner"), FreshRemoteSynth)
        # The label is what routes the timing to the right EMA below.
        assert job["backend"] == "runner"
    finally:
        main.state["runner"] = None


def test_an_unreachable_runner_falls_back_rather_than_failing_the_job(speech):
    """A runner on somebody's desk is off, asleep or on a different network
    most of the time. That is an ordinary condition, not an error: this service
    has a CPU path that works and it must use it silently.

    ASKED ON THE PROBE'S THREAD, WHICH IS THE WHOLE POINT. The refusal used to
    be discovered by the worker, on the job's own clock, so an unreachable
    runner charged its connect timeout to a job it was never going to run.
    """
    import app.main as main

    class Dead(FakeClient):
        def speech_state(self):
            raise RemoteUnavailable("connection refused")

    main.state["runner"] = Dead()
    probe = main.dispatch.lanes["runner"].probe
    try:
        probe.once()
        assert probe.ok() is False
        assert "unreachable" in probe.why
        job = {"id": "abc123"}
        # And the lane the chooser would then pick is this host.
        assert main._backend_for(job, "local") is main._local_synth("chatterbox")
    finally:
        main.state["runner"] = None


# ------------------------------------------------------------ the rate EMA ---


def test_a_gpu_rate_never_enters_the_cpu_average(speech):
    """THE DEFECT THIS PREVENTS, and it is the expensive one.

    `rate` decides whether an incoming request is answered synchronously or
    handed a 202. Chatterbox on this CPU is 0.275x realtime; on a GPU it is
    tens of times faster. One shared average lands between the two, describing
    no machine that exists, and _sync_budget then accepts a synchronous request
    the CPU cannot possibly finish. It does that at the precise moment the GPU
    disappears, because that is when jobs come back here.
    """
    import app.main as main

    before = main.rate.value
    main.rate_for("runner").observe(audio_seconds=100.0, compute_seconds=5.0)  # 20x
    assert main.rate.value == before, "the local EMA moved because of a remote job"
    assert main.rate_for("runner").value > main.rate.value

    main.rate_for("local").observe(audio_seconds=1.0, compute_seconds=4.0)
    assert main.rate.value != before, "rate_for('local') must BE `rate`, not a copy"
    assert main.rate_for("local") is main.rate


# ----------------------------------------------------------------- speaking --


def test_each_segment_arrives_separately_so_streaming_still_works():
    """THE DEFECT THIS PREVENTS: returning one blob at the end.

    `on_chunk` per segment is what builds `offsets`, which is an exact boundary
    recorded in the sidecar, and what lets an SSE stream start before the job
    finishes. A remote backend that returned everything at once would pass a
    naive 'the audio is right' test and silently remove both.
    """
    client = FakeClient(segments=3)
    seen: list[int] = []
    spoken = RemoteSynth(client, JOB).speak_segments(
        segs(3), "en", DIALS, None,
        on_chunk=lambda piece: seen.append(piece.size))

    assert len(seen) == 3, f"expected one callback per segment, got {len(seen)}"
    assert spoken.audio.size == sum(seen)
    assert spoken.audio.dtype == np.float32
    assert spoken.input_tokens == 7


def test_segments_past_the_ninth_are_not_spoken_out_of_order():
    """THE DEFECT THIS PREVENTS is audible and reports no error.

    Artefacts are named <job>.<n>.f32, so a lexical sort puts segment 10
    between 1 and 2. Any job long enough to chunk into more than ten segments -
    which is most of them, since this service exists for long text - would come
    back with its sentences shuffled.
    """
    client = FakeClient(segments=12)
    order: list[float] = []
    RemoteSynth(client, JOB).speak_segments(
        segs(12), "en", DIALS, None,
        on_chunk=lambda piece: order.append(round(float(piece[0]), 3)))

    # The fake makes segment i a constant of 0.1 * (i + 1), so the amplitudes
    # are the segment numbers and out-of-order is visible rather than inferred.
    assert order == [round(0.1 * (i + 1), 3) for i in range(12)], order


def test_the_runner_is_sent_text_and_never_a_pause():
    """Silence is generated on this host by splice(), because no TTS model
    reliably produces a beat you can act inside. A pause is this service's
    policy and the runner must not learn about it, or two clients with different
    chunking policies would need two runners."""
    client = FakeClient(segments=2)
    RemoteSynth(client, JOB).speak_segments(
        [("first", 0.4), ("second", 1.2)], "en", DIALS, None)
    params, _ = client.submitted[0]
    assert params["segments"] == ["first", "second"]
    assert "0.4" not in repr(params) and "1.2" not in repr(params)


def test_the_local_job_id_is_the_idempotency_key():
    """A yield puts the job back in the runner's queue and this host retries.
    Without the key that retry is a second generation, which for speech means
    the same sentence spoken twice into the same file."""
    client = FakeClient()
    RemoteSynth(client, JOB).speak_segments(segs(1), "en", DIALS, None)
    _, key = client.submitted[0]
    assert key == "the-local-uuid"


def test_a_reference_clip_crosses_as_a_digest_and_never_as_a_path(tmp_path):
    """THE DEFECT THIS PREVENTS is two at once.

    `job["reference"]` is an absolute path inside a volume on this host. A
    machine on somebody's desk has no such directory, so sending the path means
    the runner either fails or, worse, silently speaks in the built-in voice.

    And the path contains the voice's NAME. A cloned voice's name is not
    something to put on a wire when a digest identifies the bytes exactly.
    """
    clip = tmp_path / "someones-cloned-voice.wav"
    clip.write_bytes(b"RIFF....fake wav bytes")

    client = FakeClient()
    RemoteSynth(client, JOB).speak_segments(segs(1), "en", DIALS, str(clip))
    params, _ = client.submitted[0]

    blob = repr(params)
    assert "someones-cloned-voice" not in blob, "the voice name reached the wire"
    assert str(tmp_path) not in blob, "a local path reached the wire"
    assert params["reference_sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert client.uploaded == [clip.read_bytes()]


def test_the_sample_rate_crosses_so_a_mismatch_can_be_caught():
    """Raw PCM has no header, so nothing in the bytes says 24 kHz. If the
    runner ever answers at a different rate, `duration = size / SAMPLE_RATE` is
    wrong and every derived number with it: audio_seconds, realtime_factor, the
    token counts, the rate EMA. Every file ships at the wrong pitch, playable
    and silent about it. Telling the runner the rate is what makes it a contract
    rather than an assumption."""
    client = FakeClient()
    RemoteSynth(client, JOB).speak_segments(segs(1), "en", DIALS, None)
    assert client.submitted[0][0]["sample_rate"] == SAMPLE_RATE


def test_a_truncated_segment_is_refused_rather_than_reinterpreted():
    """float32 is four bytes. A payload that is not a multiple of four is a
    truncated transfer or a different format, and np.frombuffer would either
    raise something opaque or, with a different dtype, cheerfully produce
    noise."""
    client = FakeClient()
    client.artefact = lambda job_id, name: b"\x00\x01\x02"
    with pytest.raises(RemoteUnavailable, match="float32"):
        RemoteSynth(client, JOB).speak_segments(segs(1), "en", DIALS, None)


# ------------------------------------------------------------- the yield -----


def test_a_yield_is_waited_out_rather_than_reported_at_all():
    """THE DEFECT THIS PREVENTS: treating the normal case as an event.

    The runner hands the GPU back the instant its owner touches the machine,
    under six seconds, several times an evening. The lease survives, so the
    right response is to wait: the job finishes complete and nothing above this
    line ever learns it happened. Raising here would have made every game launch
    somebody starts into a failed job or a restart on this host.
    """
    client = FakeClient(segments=4, yield_after=2, yield_polls=3)
    spoken = RemoteSynth(client, JOB).speak_segments(
        segs(4), "en", DIALS, None)

    assert client._stalled == 3, "the fake did not actually yield"
    assert spoken.audio.size == 4 * (SAMPLE_RATE // 10), \
        "the job came back short, so waiting lost the segments it was waiting for"
    assert client.cancelled == [], "a yield is not a reason to withdraw the lease"


def test_the_local_job_reads_queued_again_while_the_owner_is_gaming():
    """A yield is surfaced as `queued`, which is what it is, rather than left
    reading `running` for a quarter of an hour with nothing happening. That is
    the difference between a poller that waits and one whose user concludes the
    service has hung."""
    client = FakeClient(segments=3, yield_after=1, yield_polls=2)
    seen: list[bool] = []
    RemoteSynth(client, JOB, on_wait=seen.append).speak_segments(
        segs(3), "en", DIALS, None)

    assert seen == [True, False], \
        "expected queued-then-running exactly once, got " + repr(seen)


def test_a_segment_made_before_a_yield_is_never_spoken_or_delivered_twice():
    """THE PROOF THE WHOLE REMOTE PATH RESTS ON, and the reason a yield is
    waited out instead of retried.

    A retry is the obvious-looking response to the GPU going away, and it is how
    the same sentence gets spoken twice. Three separate routes to it, all closed
    here by one behaviour:

      * the runner regenerating what it already made -- closed by submitting
        once, under an idempotency key that is the local job id;
      * this client collecting an artefact it already has -- closed by `seen`;
      * `_run` being re-entered, which rebuilds the encoder and replays every
        delta into an open SSE stream -- closed by never returning early.

    So the assertions are: one submission, one delivery per segment, in order,
    and each piece distinct. The fake gives segment i an amplitude of
    0.1 * (i + 1), so a repeat is visible as a duplicate value rather than
    having to be inferred from a length.
    """
    client = FakeClient(segments=5, yield_after=2, yield_polls=4)
    pieces: list[np.ndarray] = []
    spoken = RemoteSynth(client, JOB).speak_segments(
        segs(5), "en", DIALS, None, on_chunk=pieces.append)

    assert len(client.submitted) == 1, \
        "the runner was asked to generate twice: " + repr(client.submitted)
    assert client.submitted[0][1] == JOB, "the lease was not keyed to the local job"

    amplitudes = [round(float(p[0]), 3) for p in pieces]
    assert amplitudes == [0.1, 0.2, 0.3, 0.4, 0.5], \
        "segments were duplicated, dropped or reordered across the yield: " \
        + repr(amplitudes)
    assert spoken.audio.size == 5 * (SAMPLE_RATE // 10)


def test_a_real_failure_is_still_a_failure():
    """The other half. If yielding stopped being reported, this test would keep
    passing on a client that had stopped distinguishing anything at all."""
    client = FakeClient(fail="out of VRAM")
    with pytest.raises(RemoteUnavailable, match="out of VRAM"):
        RemoteSynth(client, JOB).speak_segments(segs(1), "en", DIALS, None)


def test_waiting_forever_is_bounded_and_the_job_is_withdrawn():
    """A runner whose owner games all evening must not hold a job for ever.
    The bound is configuration; what is not optional is cancelling on the way
    out, so the runner is not left with a lease nobody is waiting for."""
    cfg = RunnerConfig(host="h", max_wait=0.0, poll=0.0)
    client = FakeClient(cfg=cfg, segments=99)
    client.job = lambda job_id: {"status": "queued", "artefacts": []}
    # "using it", not "using the machine": the same wait now serves the
    # processor rung, where nothing about a card is involved.
    with pytest.raises(RemoteYield, match="its owner is using it") as caught:
        RemoteSynth(client, JOB).speak_segments(segs(2), "en", DIALS, None)
    assert client.cancelled == ["remote-job-1"]
    # Nothing was produced, so nothing has left this host and speaking it
    # locally is invisible to every client. _worker reads exactly this.
    assert caught.value.delivered == 0


def test_giving_up_says_how_much_had_already_been_streamed():
    """`delivered` is what decides whether a local re-run is safe, so it has to
    be the real count. Reporting zero when segments had gone out would let
    _worker replay a partly-sent SSE stream from the beginning."""
    cfg = RunnerConfig(host="h", max_wait=0.0, poll=0.0)
    client = FakeClient(cfg=cfg, segments=9, yield_after=1, yield_polls=None)
    pieces: list[np.ndarray] = []
    with pytest.raises(RemoteYield) as caught:
        RemoteSynth(client, JOB).speak_segments(
            segs(9), "en", DIALS, None, on_chunk=pieces.append)

    # Counted against what actually went out rather than against a fixed
    # number, because that IS the invariant: `delivered` is a promise about how
    # many pieces a caller has already seen, and a test pinning it to a literal
    # would keep passing if the two drifted apart.
    assert caught.value.delivered == len(pieces) > 0, \
        f"reported {caught.value.delivered} but delivered {len(pieces)}"


def test_cancelling_locally_withdraws_the_job_on_the_runner():
    client = FakeClient(segments=99)
    RemoteSynth(client, JOB).speak_segments(segs(2), "en", DIALS, None,
                                            cancelled=lambda: True)
    assert client.cancelled == ["remote-job-1"]


# ------------------------------------------------------------------- TLS -----


def test_verification_is_never_disabled_anywhere_in_this_module():
    """THE DEFECT THIS PREVENTS is the one that gets added later, in a hurry,
    by somebody whose certificate did not verify.

    Reading the source is the only check that catches a future edit rather than
    the current behaviour. `CERT_NONE` appears exactly once, in the pinned
    branch, where the fingerprint is the stricter check that replaces the chain;
    `_connect` raises if the digest does not match. Nothing else may turn
    verification off, and no environment variable may.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app/remote.py").read_text()
    for forbidden in ("verify=False", "_create_unverified_context", "--insecure",
                      "check_hostname = True  # noqa"):
        assert forbidden not in source, f"{forbidden} appeared in remote.py"

    # Assignments only. Prose in a docstring that MENTIONS CERT_NONE is not a
    # code path, and counting it would make this test fail every time somebody
    # improved the comment explaining why there is only one.
    cert_none = [line for line in source.splitlines()
                 if re.search(r"verify_mode\s*=\s*ssl\.CERT_NONE", line)]
    assert len(cert_none) == 1, f"CERT_NONE is assigned {len(cert_none)} times, expected 1"

    # And the one occurrence is only reachable with a fingerprint to check.
    assert re.search(r"if cfg\.fingerprint:\s*\n(?:\s*#.*\n)*\s*ctx\.check_hostname = False",
                     source), "CERT_NONE is not guarded by `if cfg.fingerprint`"


def test_a_configuration_with_no_pin_and_no_ca_still_verifies():
    """The fallback is the system trust store, not 'trust anything'. Somebody
    who sets only a host must get a connection that fails to verify rather than
    one that succeeds without checking."""
    ctx = _pinned_context(RunnerConfig(host="h"))
    import ssl

    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_a_pinned_configuration_pins_the_digest_rather_than_the_name():
    ctx = _pinned_context(RunnerConfig(host="192.0.2.11", fingerprint="ab" * 32))
    import ssl

    # Hostname checking is off BECAUSE the digest is stricter, not instead of a
    # check. RunnerClient._connect compares it and raises on a mismatch.
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_a_fingerprint_is_accepted_in_the_form_the_runner_prints_it():
    """`idlegpu fingerprint` prints lower-case hex and also a colon-grouped
    form for reading aloud. Somebody will paste the grouped one."""
    grouped = ":".join(["AB"] * 32)
    cfg = RunnerConfig.from_env({"TTS_RUNNER_HOST": "h", "TTS_RUNNER_FINGERPRINT": grouped})
    assert cfg.fingerprint == "ab" * 32


# ------------------------------------- through the real queue, end to end -----
#
# Everything above drives RemoteSynth directly. These drive the actual routes,
# the actual worker thread and the actual _run, because that is where the seam
# is wired up and where the two defects below were living: both of them passed
# every direct-call test in this file while being broken in production.


def _wait(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


@contextmanager
def runner(client, cpu=None):
    """Attach a fake runner to the running app for the length of one test.

    TWO THINGS HAVE TO HAPPEN AND BOTH ARE THE POINT OF LANES.

    The probe is primed by hand. Nothing asks a runner whether it is free on
    the job's own thread any more -- that is what used to add a thirty second
    socket timeout to a job that was always going to run here -- so a fake
    attached after startup is invisible until the probe next runs. Calling it
    once here is the test's stand-in for the ten seconds a real deployment
    would take, and it is deliberately explicit: a test that forgets it gets a
    local job and a very clear reason why.

    The hop is set to zero. Every test below is about the remote PROTOCOL --
    idempotency keys, artefact ordering, what a yield does -- and not about
    routing. With the real 8 s handover charged against it, "One short line" is
    a second of speech and belongs on this host by any honest arithmetic, so
    leaving the hop in would silently turn each of these into a test that the
    runner is never used. Routing is asserted in test_dispatch.py, where it can
    be read.
    """
    import app.main as main

    main.state["runner"] = client
    main.state["runner_cpu"] = cpu
    lane = main.dispatch.lanes["runner"]
    hop = lane.hop
    lane.hop = 0.0
    lane.probe.once()
    try:
        yield main
    finally:
        main.state["runner"] = None
        main.state["runner_cpu"] = None
        lane.hop = hop
        lane.probe.once()


def test_the_runner_is_keyed_to_the_local_job_id_when_a_real_job_runs(speech):
    """THE DEFECT THIS PREVENTS, and it had shipped.

    The idempotency key is the only thing standing between a resumed lease and
    a second generation of the same text. It used to be a `job_id=` keyword on
    speak_segments -- which `_run` cannot pass, because `_run` calls the method
    positionally on whichever backend it was handed and the local Synth has no
    such parameter. So every real job fell through to `anon-<clock>`, a fresh
    key on every attempt, and the guarantee was off in production while the
    direct-call test for it passed by handing over the keyword itself.

    This test submits through the route and reads the key off the wire. It is
    the only shape that could have caught it, which is why it exists.
    """
    fake = FakeClient(segments=1)
    with runner(fake):
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        _wait(speech, created["id"])

    assert fake.submitted, "the runner was never asked to do anything"
    _, key = fake.submitted[0]
    assert key == created["id"], (
        "the lease was keyed to " + repr(key) + " rather than to the local job "
        "id, so a resumed job would be generated a second time")
    assert not key.startswith("anon-"), "the anonymous fallback key came back"


def test_a_mid_job_yield_finishes_the_job_instead_of_failing_it(speech):
    """THE NORMAL CASE, end to end: somebody starts a game half way through.

    The job must come back `done`, with audio, having been generated once. A
    `failed` here would mean this service reports an error every time the owner
    of that machine sits down at it.
    """
    fake = FakeClient(segments=3, yield_after=1, yield_polls=3)
    with runner(fake):
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])

    assert fake._stalled == 3, "the fake never actually yielded"
    assert finished["status"] == "done", finished.get("error")
    assert len(fake.submitted) == 1, "the text was sent for generation twice"
    assert finished["audio_seconds"] > 0


def test_a_runner_that_never_comes_back_is_spoken_once_here_not_twice(speech,
                                                                     monkeypatch):
    """THE OTHER HALF OF "never spoken twice", on the path that DOES re-run.

    When the bounded wait runs out the job is spoken on this host instead --
    that fallback is the whole reason the CPU Synth is never removed. What must
    not happen is the local run being entered more than once, or being entered
    on top of audio a client has already been sent.

    `_speak` is counted rather than inferred: one call per segment, once. If
    _worker ever grew a retry loop around the fallback this is what would fail.
    """
    import app.main as main
    from app import synth as synth_module

    calls: list[str] = []
    original = synth_module.Synth._speak

    def counting(self, text, *a, **kw):
        calls.append(text)
        return original(self, text, *a, **kw)

    monkeypatch.setattr(synth_module.Synth, "_speak", counting)

    # A runner that accepts the job and then never starts it: the owner is at
    # the machine. max_wait 0 makes the bound fire at once instead of in
    # fifteen minutes.
    cfg = RunnerConfig(host="h", max_wait=0.0, poll=0.0)
    fake = FakeClient(cfg=cfg, segments=99)
    fake.job = lambda job_id: {"status": "queued", "artefacts": []}

    with runner(fake):
        created = speech.post("/jobs", json={"text": "One short line.",
                                             "voice": "default"}).json()
        finished = _wait(speech, created["id"])

    assert finished["status"] == "done", finished.get("error")
    assert finished["backend"] == "local", "it did not fall back to this host"
    assert len(calls) == finished["chunks"] == len(set(calls)), (
        "the text was spoken more than once locally: " + repr(calls))
    assert fake.cancelled == ["remote-job-1"], \
        "the lease was left on the runner for a job nobody is waiting for"
    assert main.jobs[created["id"]]["status"] == "done"


def test_the_local_path_is_untouched_when_no_runner_is_configured(speech,
                                                                  monkeypatch):
    """THE ASSERTION THE BRIEF ASKED FOR, made rather than claimed.

    With nothing configured the remote module must not be reached at all: no
    client, no chooser branch that constructs one, no label on the job. The
    strong form is to make every entry point into remote.py explode and then run
    a real job through the real routes; anything that had crept into the local
    path would raise instead of quietly working.
    """
    import app.main as main
    from app import remote as remote_module

    assert main.state.get("runner") is None

    def forbidden(*_a, **_kw):
        raise AssertionError("the local path reached into the remote module")

    monkeypatch.setattr(remote_module, "RemoteSynth", forbidden)
    monkeypatch.setattr(remote_module.RunnerClient, "_request", forbidden)
    monkeypatch.setattr(main, "RemoteSynth", forbidden)

    created = speech.post("/jobs", json={"text": "One short line, spoken once.",
                                         "voice": "default"}).json()
    finished = _wait(speech, created["id"])

    assert finished["status"] == "done"
    # THE PROOF IS THE FORBIDDEN STUBS ABOVE, NOT A MISSING KEY. This used to
    # assert `"backend" not in finished`, which read "the chooser did nothing"
    # -- but absence proved only that nothing had written a label, and the job
    # record now says where it ran on every path, deliberately: an ordinary
    # local job carried no backend at all, so after a restart every row looked
    # identical whether it had run on a GPU across the LAN or on this CPU.
    #
    # The real assertion is unchanged and is stronger: every entry point into
    # remote.py raises, and a job ran to completion anyway.
    assert finished["backend"] == "local", "an unconfigured job must say local"
    assert "runner_host" not in finished, "nothing remote was involved"
    assert "fell_back" not in finished, "there was nothing to fall back from"
    # AND THE SECOND CLIENT IS NOT CONSTRUCTED EITHER. The processor rung is a
    # second RunnerClient over the same host, so a chooser that built one
    # unconditionally would open a socket on a stack that has no runner at all
    # -- and would do it from the module every other test depends on being inert.
    assert main.state.get("runner_cpu") is None


def test_the_two_backends_speak_through_exactly_the_same_signature():
    """THE STRUCTURAL PIN, and the one whose absence let the idempotency key be
    silently unwired for a whole release.

    `_run` calls `speak_segments` positionally on whichever backend it was
    handed and cannot know which it got. The instant the two signatures differ,
    either the remote backend has a parameter production can never pass -- which
    is exactly what happened to `job_id`, and it disabled the no-double-speak
    guarantee while its own test went on passing -- or `_run` has to grow a
    branch on backend type, which is the seam collapsing.

    Compared as text so a defaulted extra parameter, the sort that looks
    harmless, still fails.
    """
    import inspect

    from app.synth import Synth

    local = inspect.signature(Synth.speak_segments)
    remote = inspect.signature(RemoteSynth.speak_segments)
    assert str(local) == str(remote), (
        "the backends have drifted apart:\n  local  " + str(local)
        + "\n  remote " + str(remote))


def test_a_long_job_on_a_busy_runner_is_never_charged_for_working():
    """THE DEFECT THIS PREVENTS: a healthy runner losing every long job.

    `max_wait` bounds how long the GPU's owner may have the machine. It was
    implemented as a deadline from submission, which bounds the whole job
    instead -- so with the 900 s default and a runner speaking at around 0.6x
    realtime, ANY job over roughly eight minutes of audio abandoned the GPU and
    re-spoke on the CPU, on a completely idle machine, reporting that somebody
    was gaming. Streamed, it was worse: _worker treats a RemoteYield with
    `delivered > 0` as an honest failure, so a working runner produced a FAILED
    job. This is the service whose whole purpose is long jobs.

    Here the runner never stalls and needs more polls than the bound would
    allow if running time were charged. It must still finish.
    """
    client = FakeClient(segments=6)
    client.cfg = RunnerConfig(host="runner.invalid", service="chatterbox",
                              poll=0.0, max_wait=0.0)
    spoken = RemoteSynth(client, JOB).speak_segments(
        segs(6), "en", DIALS, None)

    assert spoken.audio.size == 6 * (SAMPLE_RATE // 10), \
        "a running runner was charged for running and the job was abandoned"
    assert client.cancelled == [], "the lease was withdrawn from a working runner"


def test_only_the_time_the_owner_has_the_machine_counts_towards_the_bound():
    """The other half of the same fix: a stall IS charged, and still raises.

    max_wait=0 means the first poll that finds the job not running is already
    over the bound, so a runner that yields and never comes back must give up
    rather than wait for ever.
    """
    client = FakeClient(segments=4, yield_after=2, yield_polls=None)
    client.cfg = RunnerConfig(host="runner.invalid", service="chatterbox",
                              poll=0.0, max_wait=0.0)
    with pytest.raises(RemoteYield) as raised:
        RemoteSynth(client, JOB).speak_segments(
            segs(4), "en", DIALS, None)

    assert raised.value.delivered == 2, "the segments already made were forgotten"
    assert client.cancelled == ["remote-job-1"], \
        "the lease was left on a runner nobody is waiting for"


def test_a_job_queued_before_it_ever_ran_says_so():
    """THE DEFECT THIS PREVENTS: the commonest case reporting nothing.

    The guard was `status == "queued" and parts and not waiting`, so waiting
    was only ever reported once a segment had landed. Submitting while the
    owner is ALREADY at the machine -- which is most of an evening -- produced
    no callback at all, and the local job read `running` for the whole bound
    with nothing happening. The segment count belongs in the message, not in
    the condition.
    """
    client = FakeClient(segments=2, yield_after=0, yield_polls=2)
    seen: list[bool] = []
    spoken = RemoteSynth(client, JOB, on_wait=seen.append).speak_segments(
        segs(2), "en", DIALS, None)

    assert seen == [True, False], \
        "expected queued-then-running with nothing delivered yet, got " + repr(seen)
    assert spoken.audio.size == 2 * (SAMPLE_RATE // 10), \
        "reporting the wait cost the job its segments"


# ----------------------------------------- the runner sells two resources ---
#
# These drive the real RunnerClient.speech_state and RunnerClient.snapshot
# against a canned /v1/services and /v1/status document. Nothing opens a socket:
# _request is replaced, which is the same seam every other test in this file
# uses, one layer lower.


def _client_answering(doc, path="/v1/services", services=None, cfg=None):
    """A real RunnerClient whose HTTP calls return canned documents.

    TWO PATHS, because the runner's answer lives in two documents and neither
    holds all of it. /v1/status has the live load and what is running;
    /v1/services has the per-service device and the resolved `available`. A
    fake that answered only one of them is how the snapshot came to ask
    /v1/status for a `device` field the runner has never published there.

    `services=` supplies the second document. Anything unasked for answers an
    empty object, which is also what an older runner effectively gives.
    """
    import json as _json

    from app.remote import RunnerClient

    docs = {path: doc}
    if services is not None:
        docs["/v1/services"] = services
    c = RunnerClient(cfg or RunnerConfig(host="runner.invalid", service="chatterbox",
                                         fingerprint="ab" * 32))

    def fake(method, p, body=None, headers=None, timeout=None):
        return 200, {}, _json.dumps(docs.get(p, {})).encode()

    c._request = fake            # type: ignore[method-assign]
    return c


def test_a_busy_gpu_does_not_hide_a_free_processor():
    """THE DEFECT THIS PREVENTS: walking away from a runner that would have
    spoken.

    A game takes the card and leaves twelve threads idle. Reading the
    machine-wide `gpu_available` would have made this side fall back to the NAS
    CPU at 0.275x realtime, for a machine that was about to do the work.
    """
    ok, why = _client_answering({
        "gpu_available": False,
        "machine_state": "busy",
        "services": [{"id": "chatterbox", "installed": True, "enabled": True,
                      "device": "cpu", "available": True}],
    }).speech_state()
    assert ok is True, why
    assert why == ""


def test_the_runner_saying_no_is_believed_even_when_the_gpu_looks_free():
    """And the other way round, which is the one that costs somebody their work.

    `gpu_available` can be true while the service still cannot start: not enough
    free memory for a 6.5 GiB model, or a CPU service in a state whose row is
    zero. Trusting the machine-wide flag would submit a job the runner is not
    going to run.
    """
    ok, why = _client_answering({
        "gpu_available": True,
        "machine_state": "lightuse",
        "machine_state_reason": "signed in and using the machine",
        "services": [{"id": "chatterbox", "installed": True, "enabled": True,
                      "device": "cpu", "available": False,
                      "unavailable_reason": "only 3,100 MiB of memory is free"}],
    }).speech_state()
    assert ok is False
    assert why.startswith("machine_busy")
    # The runner's own words, not ours. A caller told "gpu busy" would go looking
    # for a game that is not running.
    assert "3,100 MiB" in why


def test_an_older_runner_without_the_field_is_not_declared_busy_for_ever():
    """A runner that predates the split publishes no `available` at all.

    Treating a missing field as false would make this side refuse a perfectly
    good older runner permanently, which is a worse failure than being slightly
    conservative. The machine-wide flag is the fallback.
    """
    doc = {"gpu_available": True,
           "services": [{"id": "chatterbox", "installed": True, "enabled": True}]}
    assert _client_answering(doc).speech_state() == (True, "")
    doc["gpu_available"] = False
    ok, why = _client_answering(doc).speech_state()
    assert (ok, why) == (False, "gpu_busy")


def test_not_installed_still_wins_over_a_free_machine():
    """Order matters. A service that is not installed is the owner's to fix and
    says so, whatever the machine is doing; reporting it as busy would have
    somebody waiting for a state that never arrives."""
    ok, why = _client_answering({
        "gpu_available": True,
        "services": [{"id": "chatterbox", "installed": False, "enabled": True,
                      "available": True}],
    }).speech_state()
    assert (ok, why) == (False, "not_installed")


def test_the_status_panel_can_say_what_the_runner_is_giving_up():
    """A job that takes four times as long because the owner is at their desk is
    not a fault, and a panel that cannot say so sends people looking for one."""
    snap = _client_answering({
        "state": "available",
        "machine_state": "lightuse",
        "machine_state_reason": "signed in and using the machine",
        "limits": {"gpu": True, "cpu_pct": 10, "priority": "idle",
                   "working_set_mib": 8192, "min_free_mib": 6144},
        "cpu": {"machine_pct": 12.5, "own_pct": 9.8, "foreign_pct": 2.7,
                "logical_processors": 16},
        "memory": {"total_mib": 32670, "available_mib": 24513, "load_pct": 24},
        "services": [{"id": "chatterbox", "running": True, "queued": 0}],
    }, path="/v1/status", services={
        # DEVICE AND AVAILABLE COME FROM HERE, and the test used to put them on
        # /v1/status where the runner has never published them. The snapshot
        # read two nulls and the assertion passed anyway because the fake was
        # answering whatever it was asked. Both documents are now supplied
        # separately, which is what the runner really does.
        "gpu_available": False,
        "cpu_available": True,
        "services": [{"id": "chatterbox", "device": "cpu", "available": True}],
    }).snapshot(max_age=0.0)
    assert snap["machine_state"] == "lightuse"
    assert snap["limits"]["cpu_pct"] == 10
    assert snap["cpu"]["foreign_pct"] == 2.7
    assert snap["memory"]["available_mib"] == 24513
    assert snap["services"][0]["device"] == "cpu"
    assert snap["services"][0]["available"] is True
    # The two machine-wide answers, side by side. A panel drawing one of them
    # over the words "the runner" says the wrong thing half the time.
    assert snap["gpu_available"] is False and snap["cpu_available"] is True


def test_an_older_runner_reports_no_limits_rather_than_zero_ones():
    """Absent is not zero. A panel that renders a missing cap as "0%" tells the
    reader the runner will do nothing, which is the opposite of the truth."""
    snap = _client_answering({
        "state": "available",
        "services": [],
    }, path="/v1/status").snapshot(max_age=0.0)
    assert snap["machine_state"] is None
    assert snap["limits"] is None
    assert snap["cpu"] is None
    assert snap["memory"] is None


# ------------------------------------------- the rung that was never reached --
#
# THE PROCESSOR RUNG IS GONE AND THIS IS THE RECORD OF WHY, because deleting a
# feature quietly is how it gets reinvented.
#
# The runner sold its card and its processor as two service ids over one host,
# and the chooser had a whole arithmetic for the second: read the published
# cap, derate the measured rate by it, compare against the local backlog. Eight
# tests covered it and all eight passed. NOT ONE OF THEM COULD HAVE FAILED IN
# PRODUCTION, because the rung had never run a single job and could not:
#
#   * the server sets TTS_RUNNER_CPU_SERVICE=chatterbox-cpu and the runner
#     registers exactly two services, `echo` and `chatterbox`. Every offer was
#     `no_such_service`;
#   * and the rung sat BELOW `local` in the shipped order, while `local` is a
#     rung that is always willing -- so nothing below it was ever asked.
#
# The arithmetic refuses it anyway at the measured rates: 300 s of speech is
# 1250 s on the runner's processor against 1304 s here, and 1250 x 1.25 is more
# than 1304. It is not a special case any more; it is a lane that would lose.
# The two-lane chooser is asserted in tests/test_dispatch.py instead.


def _spoken(chars=400):
    """A job the size of a real one, because the arithmetic is about size."""
    return {"id": "abc123", "segments": [("x" * chars, 0.0)]}


def test_the_processor_rung_is_not_a_lane_and_naming_it_does_not_make_one(speech):
    """THE DEFECT THIS PREVENTS: a configuration knob that silently does
    nothing.

    TTS_BACKEND_ORDER used to be a ladder and `runner_cpu` was a rung on it.
    Somebody who still has it in their environment must not be left believing
    a third machine is in play.
    """
    import app.main as main

    assert "runner_cpu" not in main.dispatch.lanes
    assert set(main.dispatch.lanes) <= {"local", "runner"}


def test_the_processors_rate_never_enters_the_cards_average_or_this_hosts(speech):
    """Three machines, three averages. Merging any two of them describes a
    machine that does not exist, and `rate` is what decides whether a request is
    answered synchronously or handed a 202."""
    import app.main as main

    local_before = main.rate.value
    gpu_before = main.rate_for("runner").value
    main.rate_for("runner_cpu").observe(audio_seconds=10.0, compute_seconds=50.0)

    assert main.rate.value == local_before
    assert main.rate_for("runner").value == gpu_before
    assert main.rate_for("runner_cpu").value != main.rate_for("local").value


def test_a_seed_is_a_hypothesis_and_health_says_which_numbers_are_measured(speech):
    """THE DEFECT THIS PREVENTS: a documented measurement from another machine
    being read as this stack's own.

    Every backend's rate is seeded, and a seed and an average over four hundred
    jobs look identical in `realtime_factor_by_backend`. The count beside it is
    what makes the difference legible -- to a router, and to whoever is trying
    to work out why a job went where it went.
    """
    import app.main as main

    main.rate_for("runner_cpu")            # seeded, never observed
    doc = speech.get("/health").json()
    assert doc["backend_observations"]["runner_cpu"] == 0
    assert doc["realtime_factor_by_backend"]["runner_cpu"] > 0
    assert doc["backend_order"] == list(main.BACKEND_ORDER)


# ------------------------------------------------- configuring the rung ------


def test_the_processor_rung_shares_the_pin_and_the_key_of_the_card():
    """THE DEFECT THIS PREVENTS: a second client built by hand.

    One host, one certificate, one key, one port. Building the second config by
    `replace` is what stops the two drifting apart the day a field is added --
    and a forgotten fingerprint is not an error anybody would ever see, because
    the only visible symptom is a connection that works.
    """
    cfg = RunnerConfig(host="box", fingerprint="ab" * 32, api_key="secret",
                       port=47601, max_wait=900.0)
    cpu = cfg.for_cpu()
    assert cpu is not None
    assert cpu.service == "chatterbox-cpu" and cfg.service == "chatterbox"
    assert cpu.fingerprint == cfg.fingerprint
    assert cpu.api_key == cfg.api_key and cpu.port == cfg.port
    # Shorter, and the asymmetry is the point: the rung below the card is about
    # as fast as the card's machine, so waiting out a game is worth it. The rung
    # below the processor is no slower than the processor, so it is not.
    assert cpu.max_wait == 300.0 and cpu.max_wait < cfg.max_wait


def test_an_empty_cpu_service_means_the_runner_sells_only_its_card():
    """UNSET AND EMPTY ARE DIFFERENT ANSWERS. Unset is "the default id, which is
    what the runner ships". Empty is "this machine has no processor rung", which
    is how the two-backend behaviour is restored on purpose. `or` would have
    collapsed them into one."""
    assert RunnerConfig.from_env(
        {"TTS_RUNNER_HOST": "box"}).cpu_service == "chatterbox-cpu"
    assert RunnerConfig.from_env(
        {"TTS_RUNNER_HOST": "box", "TTS_RUNNER_CPU_SERVICE": ""}).cpu_service == ""
    assert RunnerConfig.from_env(
        {"TTS_RUNNER_HOST": "box", "TTS_RUNNER_CPU_SERVICE": ""}).for_cpu() is None
    assert RunnerConfig(host="box", service="only-one",
                        cpu_service="only-one").for_cpu() is None, \
        "one service id cannot be two rungs"


def test_the_cap_is_read_from_the_row_in_force_and_not_from_a_service_field():
    """The cap is a property of the MACHINE's current state, not of the service:
    it is the matrix row in force, published once at the top of /v1/services.
    Reading it off the service entry would find nothing and refuse for ever."""
    offer = _client_answering({
        "gpu_available": False,
        "cpu_available": True,
        "machine_state": "nobodyhome",
        "machine_state_reason": "nobody is signed in",
        "limits": {"gpu": True, "cpu_pct": 100, "priority": "normal"},
        "services": [{"id": "chatterbox", "installed": True, "enabled": True,
                      "device": "cpu", "available": True}],
    }).offer()
    assert offer.ready is True
    assert offer.cpu_pct == 100
    assert offer.device == "cpu"
    assert offer.machine_state == "nobodyhome"


def test_a_runner_that_publishes_no_limits_gives_a_cap_of_none_not_zero():
    """Absent is not zero and it is not a hundred. Zero would read as "it will
    do nothing" and a hundred as "help yourself"; the truth is "this runner does
    not say", and the chooser refuses on it."""
    offer = _client_answering({
        "gpu_available": True,
        "services": [{"id": "chatterbox", "installed": True, "enabled": True,
                      "available": True}],
    }).offer()
    assert offer.ready is True
    assert offer.cpu_pct is None



# ------------------------------------------- two engines on one machine -----
#
# The runner sells one card and now carries two speech services on it. "Is the
# runner free" stopped being a question with one answer, and every test below
# is a way that could have gone unnoticed.


TURBO = "chatterbox-turbo"


def _two_services(*, turbo=None, baseline=None):
    """A /v1/services document with both speech services on it.

    Passing None for either leaves it OUT of the document entirely, which is
    what a runner whose worker.ini has no such section really answers -- and is
    exactly the state `chatterbox-cpu` was in for its whole life.
    """
    services = []
    if baseline is not None:
        services.append({"id": "chatterbox", "device": "gpu", **baseline})
    if turbo is not None:
        services.append({"id": TURBO, "device": "gpu", **turbo})
    return {"gpu_available": True, "services": services}


READY = {"installed": True, "enabled": True, "available": True}


def test_offer_reports_every_service_not_only_ours():
    """THE DEFECT THIS PREVENTS IS HOW `chatterbox-cpu` DIED.

    It was configured server-side, never registered on the runner, and because
    the loop `continue`d past every service but its own, `offer()` was never
    called for it and `no_such_service` never fired. THE DETECTOR EXISTED AND
    NOTHING RAN IT. Asking about the second service is the same HTTP request.
    """
    offer = _client_answering(_two_services(
        baseline=READY,
        turbo={"installed": False, "enabled": True})).offer()
    # UNCHANGED for the service this client is pinned to: every existing caller
    # reads `ready` and `why` and must keep getting the same two answers.
    assert (offer.ready, offer.why) == (True, "")
    assert offer.state_for("chatterbox") == (True, "")
    assert offer.state_for(TURBO) == (False, "not_installed")
    # A service the runner does not list at all is the sentence that will never
    # clear on its own, not a shrug.
    assert offer.state_for("nothing-like-this") == (False, "no_such_service")


def test_a_missing_turbo_service_does_not_shut_the_lane_for_baseline(speech):
    """One card, two services, and they fail independently.

    A turbo service that was never installed must not take the card away from
    baseline, which is installed, enabled and idle. Shutting the whole lane
    would be this side inventing an outage for an engine baseline never needed.
    """
    import app.main as main

    client = _client_answering(_two_services(baseline=READY))
    main.state["runner"] = client
    probe = main.dispatch.lanes["runner"].probe
    try:
        probe.once()
        assert probe.ok() is True, "the lane shut for a service it does not need"
        assert probe.ok_for("chatterbox") is True
        assert probe.ok_for(TURBO) is False
        assert probe.why_for(TURBO) == "no_such_service"
    finally:
        main.state["runner"] = None


def test_a_manifest_id_mismatch_shuts_that_engine_only():
    """THE SINGLE MOST LIKELY COPY-PASTE IN THE WHOLE EXERCISE, and the only one
    whose output is WRONG rather than absent.

    `Install.Json` writes the outer id from the INI section and splices the
    controller's own manifest underneath it verbatim. Nothing reconciles the
    two, so a turbo controller copied from the baseline one with `"id":
    "chatterbox"` left in publishes a turbo service that produces BASELINE
    AUDIO FROM A TURBO REQUEST, silently.
    """
    offer = _client_answering(_two_services(
        baseline=READY,
        turbo={**READY, "manifest": {"id": "chatterbox"}})).offer()
    assert offer.state_for("chatterbox") == (True, "")
    assert offer.state_for(TURBO) == (False, "manifest_mismatch")


def test_an_older_agent_publishing_no_manifest_is_not_refused():
    """ABSENT IS SKIPPED, NEVER FAILED. Refusing a runner for not carrying a
    field it predates would be this side manufacturing an outage."""
    assert _client_answering(_two_services(
        turbo={**READY})).offer().state_for(TURBO) == (True, "")
    assert _client_answering(_two_services(
        turbo={**READY, "manifest": {}})).offer().state_for(TURBO) == (True, "")
    assert _client_answering(_two_services(
        turbo={**READY, "manifest": {"id": TURBO}})).offer().state_for(TURBO) \
        == (True, "")


def test_the_runners_own_words_reach_the_row_per_engine():
    """NEVER FLATTENED. "another service holds the card" and "somebody is at the
    keyboard" have to arrive as different sentences, or turbo starving baseline
    off the card is indistinguishable from a week of gaming."""
    offer = _client_answering(_two_services(
        baseline={"installed": True, "enabled": True, "available": False,
                  "unavailable_reason": "chatterbox-turbo holds the card"},
        turbo=READY)).offer()
    ready, why = offer.state_for("chatterbox")
    assert ready is False
    assert "holds the card" in why
    assert offer.state_for(TURBO) == (True, "")


def test_turbo_params_omit_language_exaggeration_and_cfg_weight():
    """ABSENT, NOT NULL, and the difference is the whole house rule on the wire.

    `{"exaggeration": null}` is a VALUE to a controller that reads the key and
    passes it on: turbo's generate() accepts it, logs a warning nobody sees and
    discards it. `language` is worse -- turbo's generate() has no language_id
    at all and raises TypeError, so sending it fails the job on the runner.
    """
    from app.engines import ENGINES

    fake = FakeClient(segments=1)
    RemoteSynth(fake, JOB, spec=ENGINES["chatterbox"]).speak_segments(
        segs(1), "en",
        {"exaggeration": 0.3, "cfg_weight": 0.3, "temperature": 0.6})
    baseline, _ = fake.submitted[-1]
    assert baseline["language"] == "en"
    assert baseline["exaggeration"] == 0.3 and baseline["cfg_weight"] == 0.3

    # And with turbo enabled, the same call sends three fewer keys.
    import os

    os.environ["TTS_ENGINES"] = "chatterbox,chatterbox-turbo"
    try:
        import importlib

        import app.engines as engines_module
        importlib.reload(engines_module)
        turbo_spec = engines_module.ENGINES[TURBO]
    finally:
        del os.environ["TTS_ENGINES"]

    fake = FakeClient(segments=1)
    RemoteSynth(fake, JOB, spec=turbo_spec).speak_segments(
        segs(1), "en",
        {"exaggeration": 0.3, "cfg_weight": 0.3, "temperature": 0.6})
    params, _ = fake.submitted[-1]
    assert "language" not in params, "turbo's generate() raises TypeError on it"
    assert "exaggeration" not in params, "accepted, logged and discarded"
    assert "cfg_weight" not in params, "accepted, logged and discarded"
    assert params["temperature"] == 0.6, "a control turbo DOES read was dropped"
    assert params["segments"] == ["segment 0"]


def test_a_synth_with_no_spec_sends_exactly_what_it_always_did():
    """The engine that existed before there were two, byte for byte.

    Every deployment that has not enabled a second engine, every older caller
    and every other test in this file goes down this path, so a change here
    would be a wire change nobody asked for.
    """
    fake = FakeClient(segments=1)
    RemoteSynth(fake, JOB).speak_segments(segs(1), "fr",
        {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.7})
    params, _ = fake.submitted[-1]
    assert params["language"] == "fr"
    assert params["exaggeration"] == 0.4
    assert params["cfg_weight"] == 0.5
    assert params["temperature"] == 0.7
