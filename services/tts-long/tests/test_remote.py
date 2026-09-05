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
                 fail=None, tokens=7):
        # poll=0: these tests assert ordering and state transitions, not timing,
        # and a 2 s default would turn the file into a minute of sleeping.
        self.cfg = cfg or RunnerConfig(host="runner.invalid", service="chatterbox", poll=0.0)
        self.submitted: list[tuple[dict, str]] = []
        self.uploaded: list[bytes] = []
        self.cancelled: list[str] = []
        self._segments = segments
        self._yield_after = yield_after
        self._fail = fail
        self._tokens = tokens
        self._polls = 0
        self.state = (True, "")

    def speech_state(self):
        return self.state

    def ensure_asset(self, path):
        raw = Path(path).read_bytes()
        self.uploaded.append(raw)
        return hashlib.sha256(raw).hexdigest()

    def submit(self, params, idempotency_key):
        self.submitted.append((params, idempotency_key))
        return "remote-job-1"

    def job(self, job_id):
        self._polls += 1
        done = min(self._polls, self._segments)
        if self._fail:
            return {"status": "failed", "artefacts": [], "record": {"error": self._fail}}
        if self._yield_after is not None and done > self._yield_after:
            # Artefacts already delivered stay delivered; the status goes back
            # to queued, which is what the runner really does.
            return {"status": "queued", "artefacts": self._names(self._yield_after)}
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
    assert main._backend_for(job) is main.state["synth"]
    assert "backend" not in job, "an unconfigured run must not even label itself"


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
    try:
        with caplog.at_level("INFO"):
            chosen = main._backend_for({"id": "abc123"})
        assert chosen is main.state["synth"], "a runner that cannot help must not be used"
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
        assert isinstance(main._backend_for(job), FreshRemoteSynth)
        # The label is what routes the timing to the right EMA below.
        assert job["backend"] == "runner"
    finally:
        main.state["runner"] = None


def test_an_unreachable_runner_falls_back_rather_than_failing_the_job(speech):
    """A runner on somebody's desk is off, asleep or on a different network
    most of the time. That is an ordinary condition, not an error: this service
    has a CPU path that works and it must use it silently."""
    import app.main as main

    class Dead(FakeClient):
        def speech_state(self):
            raise RemoteUnavailable("connection refused")

    main.state["runner"] = Dead()
    try:
        assert main._backend_for({"id": "abc123"}) is main.state["synth"]
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
    spoken = RemoteSynth(client).speak_segments(
        segs(3), "en", 0.5, 0.5, 0.8, None,
        on_chunk=lambda piece: seen.append(piece.size), job_id="job-1")

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
    RemoteSynth(client).speak_segments(
        segs(12), "en", 0.5, 0.5, 0.8, None,
        on_chunk=lambda piece: order.append(round(float(piece[0]), 3)), job_id="j")

    # The fake makes segment i a constant of 0.1 * (i + 1), so the amplitudes
    # are the segment numbers and out-of-order is visible rather than inferred.
    assert order == [round(0.1 * (i + 1), 3) for i in range(12)], order


def test_the_runner_is_sent_text_and_never_a_pause():
    """Silence is generated on this host by splice(), because no TTS model
    reliably produces a beat you can act inside. A pause is this service's
    policy and the runner must not learn about it, or two clients with different
    chunking policies would need two runners."""
    client = FakeClient(segments=2)
    RemoteSynth(client).speak_segments(
        [("first", 0.4), ("second", 1.2)], "en", 0.5, 0.5, 0.8, None, job_id="j")
    params, _ = client.submitted[0]
    assert params["segments"] == ["first", "second"]
    assert "0.4" not in repr(params) and "1.2" not in repr(params)


def test_the_local_job_id_is_the_idempotency_key():
    """A yield puts the job back in the runner's queue and this host retries.
    Without the key that retry is a second generation, which for speech means
    the same sentence spoken twice into the same file."""
    client = FakeClient()
    RemoteSynth(client).speak_segments(segs(1), "en", 0.5, 0.5, 0.8, None,
                                       job_id="the-local-uuid")
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
    RemoteSynth(client).speak_segments(segs(1), "en", 0.5, 0.5, 0.8, str(clip),
                                       job_id="job-1")
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
    RemoteSynth(client).speak_segments(segs(1), "en", 0.5, 0.5, 0.8, None, job_id="j")
    assert client.submitted[0][0]["sample_rate"] == SAMPLE_RATE


def test_a_truncated_segment_is_refused_rather_than_reinterpreted():
    """float32 is four bytes. A payload that is not a multiple of four is a
    truncated transfer or a different format, and np.frombuffer would either
    raise something opaque or, with a different dtype, cheerfully produce
    noise."""
    client = FakeClient()
    client.artefact = lambda job_id, name: b"\x00\x01\x02"
    with pytest.raises(RemoteUnavailable, match="float32"):
        RemoteSynth(client).speak_segments(segs(1), "en", 0.5, 0.5, 0.8, None, job_id="j")


# ------------------------------------------------------------- the yield -----


def test_a_yield_is_not_a_failure():
    """THE DEFECT THIS PREVENTS: reporting `failed` when somebody starts a game.

    The runner hands the GPU back the instant its owner touches the machine.
    That happens several times an evening and is the system working exactly as
    designed. A distinct exception type is what lets the caller wait, retry or
    fall back, instead of a caller that sees RuntimeError and marks the job
    failed for what was a completely normal event.
    """
    client = FakeClient(segments=4, yield_after=2)
    with pytest.raises(RemoteYield) as caught:
        RemoteSynth(client).speak_segments(segs(4), "en", 0.5, 0.5, 0.8, None, job_id="j")

    assert not isinstance(caught.value, RemoteUnavailable), \
        "a yield must be distinguishable from the runner being broken"
    assert "2 of 4" in str(caught.value), "say how far it got: " + str(caught.value)


def test_a_real_failure_is_still_a_failure():
    """The other half. If yielding stopped being reported, this test would keep
    passing on a client that had stopped distinguishing anything at all."""
    client = FakeClient(fail="out of VRAM")
    with pytest.raises(RemoteUnavailable, match="out of VRAM"):
        RemoteSynth(client).speak_segments(segs(1), "en", 0.5, 0.5, 0.8, None, job_id="j")


def test_waiting_forever_is_bounded_and_the_job_is_withdrawn():
    """A runner whose owner games all evening must not hold a job for ever.
    The bound is configuration; what is not optional is cancelling on the way
    out, so the runner is not left with a lease nobody is waiting for."""
    cfg = RunnerConfig(host="h", max_wait=0.0, poll=0.0)
    client = FakeClient(cfg=cfg, segments=99)
    client.job = lambda job_id: {"status": "queued", "artefacts": []}
    with pytest.raises(RemoteYield, match="did not start"):
        RemoteSynth(client).speak_segments(segs(2), "en", 0.5, 0.5, 0.8, None, job_id="j")
    assert client.cancelled == ["remote-job-1"]


def test_cancelling_locally_withdraws_the_job_on_the_runner():
    client = FakeClient(segments=99)
    RemoteSynth(client).speak_segments(segs(2), "en", 0.5, 0.5, 0.8, None,
                                       cancelled=lambda: True, job_id="j")
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
