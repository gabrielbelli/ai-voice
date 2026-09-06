"""Speak on somebody else's GPU, over TLS, and fall back the moment it is gone.

WHY THIS EXISTS. Chatterbox is the only component in this stack slower than
realtime: 0.275x on the NAS's CPU at eight threads, which is about four minutes
of compute per minute of audio. Parakeet does 8.81x and Kokoro 1.83x on the same
CPU and neither wants a GPU. So there is exactly one reason to reach across a
network, and this module is it.

WHAT THIS IS NOT. It is not a client for a cluster, a broker or a scheduler.
There is no central server, and the runner on the other end is an `idlegpu`
agent listening on its own machine with a pinned self-signed certificate. If
several runners ever exist, a chooser goes in front of this and nothing here
changes.

THE SHAPE, AND THE ONE RULE IT KEEPS

`RemoteSynth` implements exactly the signature `Synth.speak_segments` has, and
nothing else, so `_run()` never learns which one it got:

    speak_segments(segments, language, exaggeration, cfg_weight, temperature,
                   reference, on_chunk=..., cancelled=...) -> Spoken

Everything after that call stays here on this host and is untouched: the
encoding, the chunk boundaries, the file write into OUT_DIR named by the local
job id, the sidecar. Those are pinned by byte-exact tests against *this*
machine's ffmpeg, and moving them to a stranger's Windows box would turn them
into facts about a build nobody in this repository controls.

WHAT CROSSES THE WIRE

Per segment, in order: raw float32 PCM at 24000 Hz, plus that segment's input
token count. Not encoded audio, and not one blob at the end. One blob would kill
streaming and would kill `offsets`, which is an exact boundary recorded in the
sidecar and is built from the per-segment callback.

The reference clip crosses as BYTES, content-addressed. `job["reference"]` is an
absolute path inside a volume on the NAS; a machine on somebody's desk has no
such directory and never will. So the clip is hashed, `HEAD /v1/assets/<sha256>`
asks whether the runner already holds it, and the bytes are sent only on a miss.
The digest is the identity and the name is a label, which is also what keeps a
cloned voice's NAME out of anything that crosses a network.

A YIELD IS NOT A FAILURE, AND THIS IS THE PART MOST LIKELY TO BE GOT WRONG

The runner gives the GPU back the instant its owner touches the machine. Under
six seconds for a known game. That is the NORMAL case, several times an evening,
and it is not an error at any level: the job is WAITED OUT here, inside
`speak_segments`, and reported as `queued` on the local job while it waits.

Waiting rather than restarting is what makes "a job is never spoken twice" a
property rather than a hope, and it closes three separate routes to it at once.
The lease on the runner survives under the same idempotency key, so segments it
already produced are not produced again. The `seen` set means segments already
collected are not collected again. And because the call never returns, `_run` is
never re-entered, so no encoder is rebuilt and no SSE client receives a second
file header half way through a stream.

`RemoteYield` is therefore raised for one thing only: the bounded wait running
out, which means the owner has been at the machine long enough that the local
CPU would have finished. It says "speak this here instead", not "this failed".
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import ssl
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from voice_common.audio import check_rate, splice

from .synth import SAMPLE_RATE, Spoken

log = logging.getLogger("tts-long.remote")


class RemoteUnavailable(RuntimeError):
    """The runner could not be reached, or refused. Fall back to local CPU."""


class RemoteYield(RuntimeError):
    """The bounded wait for the GPU ran out. Speak it locally instead.

    NOT A FAILURE, and it must never reach a caller as one. A short yield is
    invisible above this line: it is waited out inside `speak_segments`. This is
    raised only when the owner has been at their machine for longer than
    `max_wait`, at which point the right answer is the local CPU path, which is
    slow but always there.

    `delivered` is how many segments had already been handed to `on_chunk`
    before giving up, and it is on the exception because it decides whether
    speaking the job again locally is safe. Zero means nothing has left this
    host and a local re-run is invisible to everyone. Anything else means an SSE
    client has already been sent audio that a re-run would send again.
    """

    def __init__(self, message: str, delivered: int = 0) -> None:
        super().__init__(message)
        self.delivered = delivered


@dataclass(frozen=True)
class RunnerConfig:
    """Where the runner is and how to trust it.

    Every field is configuration with a documented default. UNSET MEANS
    LOCAL-ONLY: with no URL there is no remote path at all, `_backend_for`
    returns the local `Synth`, and not one line of `_run` behaves differently.
    That is deliberate and load-bearing, because the entire test suite rests on
    monkeypatching `Synth._speak`.
    """

    host: str
    port: int = 47600
    # The SHA-256 of the runner's self-signed certificate, which IS the trust
    # root. There is no public certificate authority for a program somebody runs
    # on their own desktop, and pretending otherwise would mean either shipping
    # a private key or turning verification off.
    fingerprint: str = ""
    # A private CA bundle instead, for anyone who does have an internal PKI.
    ca_file: str = ""
    api_key: str = ""
    service: str = "chatterbox"
    # THE SECOND RUNG ON THE SAME MACHINE. The runner sells its processor as
    # well as its card, as two service ids over one host, port, pin and key --
    # so the CPU client is `dataclasses.replace(cfg, service=cfg.cpu_service)`
    # and nothing else. Set TTS_RUNNER_CPU_SERVICE to "" to switch the rung off
    # and get exactly the two-backend behaviour that shipped before it existed.
    cpu_service: str = "chatterbox-cpu"
    timeout: float = 30.0
    # How long a job may sit queued on the runner while its owner is gaming
    # before this host gives up and speaks it locally. Longer than one game, and
    # shorter than an evening.
    max_wait: float = 900.0
    # SHORTER FOR THE PROCESSOR RUNG, and the asymmetry is the point. A job
    # waiting on the card falls back to a machine that is roughly as fast
    # (0.230x here against 0.70x there, measured), so waiting out a game is
    # usually cheaper than giving up. A job waiting on the runner's processor
    # falls back to a machine that is no slower than it was going to be
    # anyway, so a long wait buys nothing at all.
    cpu_max_wait: float = 300.0
    poll: float = 2.0

    def for_cpu(self) -> "RunnerConfig | None":
        """The same machine, its other service, or None when it has none.

        One host, one port, one pin, one key, one certificate: the processor
        rung differs from the card by a service id and by how long it is worth
        waiting for. Building it by `replace` rather than by hand is what stops
        the two from drifting apart the day a field is added to one of them --
        a second constructor call is a second place to forget the fingerprint,
        and forgetting the fingerprint is not an error anybody would see.
        """
        if not self.cpu_service or self.cpu_service == self.service:
            return None
        return replace(self, service=self.cpu_service, max_wait=self.cpu_max_wait)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RunnerConfig | None":
        """Build from TTS_RUNNER_*, or return None for local-only.

        Returning None rather than a disabled object is the point: the caller
        branches once, at startup, and the local path keeps no remote code in it.
        """
        e = env if env is not None else os.environ
        host = (e.get("TTS_RUNNER_HOST") or "").strip()
        if not host:
            return None
        key = (e.get("TTS_RUNNER_API_KEY") or "").strip()
        key_file = (e.get("TTS_RUNNER_API_KEY_FILE") or "").strip()
        if not key and key_file:
            try:
                key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                # Named, not swallowed. A misconfigured key file means every
                # request is a 401 and the fallback quietly hides it.
                log.warning("cannot read TTS_RUNNER_API_KEY_FILE: %s", exc)
        return cls(
            host=host,
            port=int(e.get("TTS_RUNNER_PORT") or 47600),
            fingerprint=(e.get("TTS_RUNNER_FINGERPRINT") or "").replace(":", "").lower().strip(),
            ca_file=(e.get("TTS_RUNNER_CA_FILE") or "").strip(),
            api_key=key,
            service=(e.get("TTS_RUNNER_SERVICE") or "chatterbox").strip(),
            # Unset means the documented default, which is the id the runner
            # ships. An explicitly EMPTY value is a different answer -- "this
            # runner has no processor rung" -- and `or` would have collapsed
            # the two, so the presence of the variable is what decides.
            cpu_service=(e["TTS_RUNNER_CPU_SERVICE"].strip()
                         if "TTS_RUNNER_CPU_SERVICE" in e else "chatterbox-cpu"),
            timeout=float(e.get("TTS_RUNNER_TIMEOUT") or 30.0),
            max_wait=float(e.get("TTS_RUNNER_MAX_WAIT") or 900.0),
            cpu_max_wait=float(e.get("TTS_RUNNER_CPU_MAX_WAIT") or 300.0),
            poll=float(e.get("TTS_RUNNER_POLL") or 2.0),
        )


def _pinned_context(cfg: RunnerConfig) -> ssl.SSLContext:
    """A TLS context that verifies, always.

    THREE WAYS TO ESTABLISH TRUST AND NONE OF THEM IS "DO NOT CHECK".

    1. A pinned SHA-256 fingerprint. The runner mints a self-signed certificate
       into its own directory on first run and prints the digest; `idlegpu
       fingerprint` prints it again. This is the normal case.
    2. A private CA bundle, for anyone who has an internal PKI.
    3. The system trust store, if the runner somehow has a publicly trusted
       certificate.

    `check_hostname` is off ONLY in case 1, because a self-signed certificate
    minted for a machine's own name will not match an IP address somebody typed,
    and in that case the fingerprint is a STRICTER check than the name would
    have been: it pins one specific key, not any certificate a CA would sign.
    The verification does not go away, it moves.

    There is no code path here that reaches ssl.CERT_NONE, and there is no
    environment variable that can create one.
    """
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if cfg.ca_file:
        ctx.load_verify_locations(cafile=cfg.ca_file)
        return ctx
    if cfg.fingerprint:
        # Verified by digest instead of by chain, below, on the socket's own
        # certificate. Both of these stay meaningless without that check, which
        # is why _connect() raises rather than returning if it does not match.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


@dataclass(frozen=True)
class RunnerOffer:
    """What the runner will do for ONE service, in its own words.

    `speech_state()` reduces this to (ready, why) and is what the chooser used
    while there was one remote rung. There are two now, on one machine, and the
    difference between them is not whether they will run but HOW MUCH OF THE
    MACHINE each is being given: `available: true` at a hundred per cent and
    `available: true` at five per cent are the same boolean and a twenty to one
    difference in delivered speech. The cap is therefore carried out of here
    rather than reduced away, because the router needs it to decide whether a
    network hop is worth taking at all.

    `cpu_pct` is None when the runner does not publish `limits` -- an older
    build, or one answering without an agent behind it. NONE IS NOT A HUNDRED.
    A missing cap means the delivered rate is unknowable from here, and the
    chooser refuses the processor rung rather than guessing, because guessing
    high turns a ten minute job into a two hour one and nothing reports a fault.
    """

    ready: bool
    why: str = ""
    # "gpu" | "cpu" | "", from the runner's own per-service field.
    device: str = ""
    # The share of the WHOLE machine the matrix row in force allows, 0-100.
    cpu_pct: int | None = None
    # The named posture in force, for a person reading the panel.
    machine_state: str = ""
    machine_state_reason: str = ""


def _project(src: dict, names: dict[str, str]) -> dict:
    """{our name: their value} for the keys they actually sent.

    A missing key is LEFT OUT rather than set to None, because a panel that
    tests `if (g.util_gpu !== undefined)` is asking "did the machine say", and
    an invented null answers a question nobody asked.
    """
    return {ours: src[theirs] for ours, theirs in names.items() if theirs in src}


def _services_of(status_doc: dict, services_doc: dict) -> list[dict]:
    """One row per service, from the two documents that each hold half of it.

    /v1/status knows what is running and how deep the queue is. /v1/services
    knows what device a service wants and whether this machine will take it
    right now, already resolved. Neither knows both, and the page needs both on
    one line, so they are joined here on the id rather than on the page.
    """
    resolved = {x.get("id"): x for x in (services_doc.get("services") or [])}
    rows = []
    for x in status_doc.get("services") or []:
        r = resolved.get(x.get("id")) or {}
        rows.append({"id": x.get("id"),
                     "running": x.get("running"),
                     "queued": x.get("queued"),
                     "device": r.get("device"),
                     "available": r.get("available"),
                     "unavailable_reason": r.get("unavailable_reason") or None})
    return rows


class RunnerClient:
    """The HTTP client. Small on purpose: seven calls and no dependencies."""

    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self._ctx = _pinned_context(cfg)
        self._lock = threading.Lock()
        self._assets: set[str] = set()
        # See snapshot(): the page polls health, health must not become a
        # round trip to somebody's desktop on every poll.
        self._snap: dict | None = None
        self._snap_at = 0.0

    def _connect(self) -> http.client.HTTPSConnection:
        conn = http.client.HTTPSConnection(
            self.cfg.host, self.cfg.port, timeout=self.cfg.timeout, context=self._ctx)
        conn.connect()
        if self.cfg.fingerprint:
            der = conn.sock.getpeercert(binary_form=True)
            got = hashlib.sha256(der).hexdigest()
            if got != self.cfg.fingerprint:
                conn.close()
                # The whole point. A mismatch is a different machine or a
                # different key, and there is no flag that makes this a warning.
                raise RemoteUnavailable(
                    f"certificate fingerprint mismatch: expected "
                    f"{self.cfg.fingerprint[:16]}..., got {got[:16]}...")
        return conn

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str = "application/json",
                 extra: dict[str, str] | None = None) -> tuple[int, dict, bytes]:
        headers = {"Accept": "application/json", "Connection": "close"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        if body is not None:
            headers["Content-Type"] = content_type
            headers["Content-Length"] = str(len(body))
        if extra:
            headers.update(extra)
        conn = self._connect()
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, dict(resp.getheaders()), data
        finally:
            conn.close()

    # -- capability ---------------------------------------------------------

    def speech_state(self) -> tuple[bool, str]:
        """(will it speak for us, why not).

        The reduction of `offer()` to the pair the chooser has always taken,
        kept because it is the whole answer for the card: the GPU rung is
        either given or it is not, and there is no dial on it. The processor
        rung has a dial, so anything choosing between the two calls `offer()`.
        """
        o = self.offer()
        return o.ready, o.why

    def offer(self) -> RunnerOffer:
        """What this runner will give this service, right now.

        THE THREE STATES, KEPT APART. The runner answers "known", "installed"
        and "ready" for the service, and separately whether the machine is free,
        and the reason those are separate fields is exactly this function.
        `not_installed` is the owner's to fix and will not change on its own;
        `machine_busy` is nobody's to fix and clears in a minute or two. A caller
        that cannot tell them apart either retries for ever against a runner with
        no speech service, or gives up on one that was about to be free.

        WHY THE PER-SERVICE `available` FIELD IS PREFERRED. The runner sells its
        processor as well as its card now, and one boolean cannot answer "will
        you work for me" any more. A game takes the GPU and leaves twelve threads
        idle; a compile takes every thread and leaves the card at five per cent.
        Reading the machine-wide `gpu_available` would walk away from a runner
        that was about to speak on the CPU, and working it out here from a device
        name and two booleans would be this side reimplementing a decision the
        runner already makes correctly, including the memory headroom check we
        cannot see from here.

        `gpu_available` IS STILL READ, as the fallback, because a runner that
        predates the split does not publish `available` at all and answering
        "busy" for ever against a perfectly good older runner is a worse failure
        than being slightly conservative.

        NOT CACHED, deliberately, while `snapshot()` is. This answer decides
        where one job goes and is asked once per job; that one is drawn on a
        page that polls. Five seconds of staleness costs a page nothing and
        costs a job the difference between the runner it was promised and the
        runner it got.
        """
        status, _, data = self._request("GET", "/v1/services")
        if status != 200:
            raise RemoteUnavailable(f"GET /v1/services returned {status}")
        doc = json.loads(data)
        limits = doc.get("limits") or {}
        pct = limits.get("cpu_pct")
        common = {
            "cpu_pct": int(pct) if isinstance(pct, (int, float)) else None,
            "machine_state": doc.get("machine_state") or "",
            "machine_state_reason": doc.get("machine_state_reason") or "",
        }
        for svc in doc.get("services", []):
            if svc.get("id") != self.cfg.service:
                continue
            device = svc.get("device") or ""
            if not svc.get("installed"):
                return RunnerOffer(False, "not_installed", device, **common)
            if not svc.get("enabled"):
                return RunnerOffer(False, "not_enabled", device, **common)
            available = svc.get("available")
            if available is None:
                # An older runner, or one answering without an agent behind it.
                if not doc.get("gpu_available"):
                    return RunnerOffer(False, "gpu_busy", device, **common)
                return RunnerOffer(True, "", device, **common)
            if not available:
                # The runner's own words when it has them. "somebody is gaming"
                # and "there is not enough memory free to start a 6.5 GiB model"
                # are both temporary and both worth telling a person apart.
                why = svc.get("unavailable_reason") or doc.get("machine_state_reason")
                return RunnerOffer(False, f"machine_busy: {why}" if why else "machine_busy",
                                   device, **common)
            return RunnerOffer(True, "", device, **common)
        return RunnerOffer(False, "no_such_service", "", **common)

    def snapshot(self, max_age: float = 5.0) -> dict:
        """What the runner is doing, for a person to look at.

        CACHED, because /health is polled by the page every few seconds and by
        the container healthcheck on its own timer, and neither should turn into
        a round trip over the LAN to somebody's desktop. Five seconds is well
        under the poll interval and well over the runner's own one-second
        sampling tick, so the page never shows a figure older than the thing
        producing it.

        NEVER RAISES. This feeds a status panel, and a runner that is switched
        off, asleep or being rebooted is the normal case rather than an error.
        The unreachable answer is itself the status.
        """
        now = time.monotonic()
        if self._snap is not None and (now - self._snap_at) < max_age:
            return self._snap
        try:
            status, _, data = self._request("GET", "/v1/status")
            if status != 200:
                snap = {"reachable": False, "error": f"HTTP {status}"}
            else:
                doc = json.loads(data)
                # BEST EFFORT, AND ON ITS OWN. A runner that answers /v1/status
                # and not /v1/services is still a reachable runner with a live
                # load figure worth drawing, so a failure here loses the
                # per-service half of the panel rather than the whole card.
                offered: dict = {}
                try:
                    st2, _, d2 = self._request("GET", "/v1/services")
                    if st2 == 200:
                        offered = json.loads(d2)
                except Exception as exc:  # noqa: BLE001 - see above
                    log.debug("runner /v1/services did not answer: %s", exc)
                gpu = doc.get("gpu") or {}
                cpu = doc.get("cpu") or {}
                mem = doc.get("memory") or {}
                snap = {
                    "reachable": True,
                    "state": doc.get("state"),
                    "can_run": doc.get("can_run"),
                    "mode": doc.get("mode"),
                    "reason": doc.get("reason"),
                    "seconds_until_available": doc.get("seconds_until_available"),
                    "job_running": doc.get("job_running"),
                    "running_service": doc.get("running_service"),
                    "yields": doc.get("yields"),
                    # WHAT THE RUNNER IS CURRENTLY WILLING TO GIVE UP, not only
                    # whether it will run. A job that takes four times as long
                    # because the owner is at their desk is not a fault, and a
                    # status panel that cannot say "capped to 10 per cent while
                    # somebody is using that machine" sends people looking for one.
                    # Absent on a runner that predates the split, and absent is
                    # not the same as zero, so these stay None rather than 0.
                    "machine_state": doc.get("machine_state"),
                    "machine_state_reason": doc.get("machine_state_reason"),
                    "limits": doc.get("limits"),
                    "cpu": {k: cpu.get(k) for k in
                            ("machine_pct", "own_pct", "foreign_pct", "logical_processors")
                            if k in cpu} or None,
                    "memory": {k: mem.get(k) for k in
                               ("total_mib", "available_mib", "load_pct")
                               if k in mem} or None,
                    # THE RUNNER'S OWN FIELD NAMES, TRANSLATED HERE, and this
                    # is a bug fix rather than a rename. The projection asked
                    # for util_gpu, mem_used_mib, power_w and name; the runner
                    # publishes utilisation_pct, memory_used_mib and
                    # power_watts, and publishes no temperature and no card
                    # name at all. The intersection of the two lists was
                    # {"healthy"}, which the page does not draw, so the GPU
                    # detail line has been permanently empty on a working stack
                    # and nothing anywhere said so. The translation belongs on
                    # this side: the page reads names, the runner publishes
                    # names, and one of the two has to speak the other's.
                    #
                    # temperature_c, name and mem_total_mib are simply absent
                    # from the runner and are therefore left out rather than
                    # invented. The page already draws only the bits it has.
                    "gpu": _project(gpu, {"healthy": "healthy",
                                          "util_gpu": "utilisation_pct",
                                          "mem_used_mib": "memory_used_mib",
                                          "power_w": "power_watts",
                                          "pstate": "pstate"}),
                    # WHAT IT WILL TAKE, PER SERVICE, and it does not come from
                    # /v1/status. That document lists services with their pids
                    # and queue depths and says nothing about device or
                    # availability; those live on /v1/services, which is the
                    # document that resolves them. Asking for both is two small
                    # requests every five seconds to a desktop on the LAN, and
                    # it is the difference between a panel that can say "the
                    # card has gone to a game, the processor is still selling"
                    # and one that shows two nulls.
                    "services": _services_of(doc, offered),
                    "gpu_available": offered.get("gpu_available"),
                    "cpu_available": offered.get("cpu_available"),
                    "gpu_contended": offered.get("gpu_contended"),
                    "cpu_contended": offered.get("cpu_contended"),
                    "profile": offered.get("profile"),
                    "profile_label": offered.get("profile_label"),
                }
        except Exception as exc:  # noqa: BLE001 - the failure IS the status
            snap = {"reachable": False, "error": type(exc).__name__}
        snap["host"] = self.cfg.host
        snap["port"] = self.cfg.port
        snap["service"] = self.cfg.service
        # Which of the listed services is the processor rung, so a panel can
        # label the two without matching on a name it made up.
        snap["cpu_service"] = self.cfg.cpu_service
        self._snap, self._snap_at = snap, now
        return snap

    # -- assets -------------------------------------------------------------

    def ensure_asset(self, path: str) -> str:
        """Upload a reference clip once, ever, and return its digest.

        A clip is a few hundred kilobytes to a few megabytes and the same one is
        used for every job that voice ever speaks, so sending it each time would
        be the largest thing on this wire by a wide margin for no reason. The
        runner caches by digest in its own contained directory, and the local set
        here saves even the HEAD after the first time in this process.
        """
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with self._lock:
            if digest in self._assets:
                return digest
        status, _, _ = self._request("HEAD", f"/v1/assets/{digest}")
        if status != 200:
            status, _, data = self._request(
                "POST", "/v1/assets", raw, content_type="application/octet-stream")
            if status not in (200, 201):
                raise RemoteUnavailable(f"POST /v1/assets returned {status}: {data[:200]!r}")
        with self._lock:
            self._assets.add(digest)
        return digest

    # -- jobs ---------------------------------------------------------------

    def submit(self, params: dict, idempotency_key: str) -> str:
        body = json.dumps(params).encode("utf-8")
        status, _, data = self._request(
            "POST", f"/v1/services/{self.cfg.service}/jobs", body,
            # THE KEY IS THE LOCAL JOB ID, which is already a uuid4. It is what
            # turns a retry after a yield into a retry rather than into the same
            # sentence being spoken twice.
            extra={"Idempotency-Key": idempotency_key})
        if status not in (200, 202):
            raise RemoteUnavailable(f"submit returned {status}: {data[:200]!r}")
        job_id = json.loads(data).get("job_id")
        if not job_id:
            raise RemoteUnavailable("the runner accepted the job without returning an id")
        return job_id

    def job(self, job_id: str) -> dict:
        status, _, data = self._request(
            "GET", f"/v1/services/{self.cfg.service}/jobs/{job_id}")
        if status != 200:
            raise RemoteUnavailable(f"job status returned {status}")
        return json.loads(data)

    def artefact(self, job_id: str, name: str) -> bytes:
        status, _, data = self._request(
            "GET", f"/v1/services/{self.cfg.service}/jobs/{job_id}/result?artefact={name}")
        if status != 200:
            raise RemoteUnavailable(f"artefact {name} returned {status}")
        return data

    def cancel(self, job_id: str) -> None:
        try:
            self._request("DELETE", f"/v1/services/{self.cfg.service}/jobs/{job_id}")
        except OSError as exc:
            # Best effort. A cancel that does not arrive costs one wasted job on
            # somebody's idle GPU, which is what the GPU was idle for.
            log.debug("cancel of %s did not arrive: %s", job_id, exc)


class RemoteSynth:
    """A `Synth` as far as `_run()` can tell.

    Deliberately NOT a subclass of Synth. It shares one method signature and
    nothing else: no model, no lock, no reaper, no `loaded`. Inheriting would
    have meant either dragging torch into this module or overriding half of a
    class to raise, and both of those make the local path harder to read for no
    benefit to the remote one.
    """

    def __init__(self, client: RunnerClient, job_id: str,
                 on_wait: Callable[[bool], None] | None = None) -> None:
        """The job id is bound HERE, not passed to speak_segments.

        THE DEFECT THIS PREVENTS, and it is the one that silently disarms the
        whole no-double-speak guarantee. `speak_segments` must have byte for
        byte the signature `Synth.speak_segments` has, because `_run` calls it
        positionally and knows nothing about which backend it got. An extra
        `job_id=` keyword therefore CANNOT be supplied by `_run` without
        breaking the local path, so it was never supplied: every real job fell
        through to an `anon-<clock>` idempotency key that was different on every
        attempt. The tests passed because they called the method directly and
        handed it the keyword that production had no way to pass.

        Binding it to the instance is what makes the key the local job id in the
        one place that has both: `_backend_for`, which is handed the job.
        """
        self._client = client
        self._job_id = job_id
        # Called with True when the runner hands the GPU back and this client
        # starts waiting, and False when it resumes. It is how a local job can
        # read `queued` again while somebody plays a game, rather than sitting
        # at `running` for a quarter of an hour with nothing happening.
        self._on_wait = on_wait

    @property
    def loaded(self) -> bool:
        # Reported separately from the local Synth's, so /health can say which
        # of the two answered rather than implying one model in two places.
        return False

    def close(self) -> None:
        pass

    def speak_segments(self, segments: list[tuple[str, float]], language: str,
                       exaggeration: float, cfg_weight: float,
                       temperature: float,
                       reference: str | None = None,
                       on_chunk: Callable[[np.ndarray], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Spoken:
        """Speak on the runner, delivering each segment as it lands.

        The chunking already happened. `segments` is the output of `chunk_text`,
        which knows about the 40-second `generate()` ceiling measured on this
        stack; the runner carries no text policy at all and must not learn any.

        The signature is exactly `Synth.speak_segments`. Nothing may be added to
        it, including something as harmless-looking as a job id: `_run` calls it
        positionally on whichever backend it was handed, so a parameter the
        local Synth does not have is a parameter production can never pass.
        """
        cfg = self._client.cfg
        key = self._job_id

        params: dict = {
            # TEXT ONLY. The pauses stay here, and that is the whole reason
            # `pauses` is captured below: no TTS model reliably produces a beat
            # you can act inside, so silence is generated locally by splice()
            # and a pause is not something the runner should learn about. It
            # also means the runner's job body is the same shape whatever this
            # client's chunking policy happens to be.
            "segments": [text for text, _pause in segments],
            "language": language,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "sample_rate": SAMPLE_RATE,
        }
        if reference:
            # BYTES BY DIGEST, never the path. `reference` is an absolute path
            # inside a volume on this host; the runner has no such directory. It
            # also means a cloned voice's name never crosses the wire: the
            # digest is the identity and the name stays here.
            params["reference_sha256"] = self._client.ensure_asset(reference)

        remote_id = self._client.submit(params, key)
        seen: set[str] = set()
        parts: list[np.ndarray] = []
        total_tokens = 0
        pauses = [pause for _, pause in segments]
        waiting = False
        # HOW LONG THE OWNER HAS HAD THE MACHINE, ACCUMULATED. Not a deadline
        # from submission, which is what this was and which made the bound mean
        # something nobody intended.
        #
        # `max_wait` defaults to 900 s and the runner speaks at around 0.6x
        # realtime, so a fixed deadline meant every job over roughly eight
        # minutes of audio abandoned the GPU and re-spoke on the CPU -- on a
        # completely idle machine, blaming an owner who was not there. Worse
        # once anything had been delivered: _worker treats a RemoteYield with
        # `delivered > 0` as an honest failure, so a healthy runner produced a
        # FAILED job. tts-long is the long-job service. That was the case it
        # exists for.
        #
        # Only time the runner is NOT working on this job counts. Waiting out a
        # six-second yield is right; waiting out an evening of gaming is not,
        # and this is the difference between those two.
        waited = 0.0
        ticked = time.monotonic()

        while True:
            if cancelled is not None and cancelled():
                self._client.cancel(remote_id)
                break

            doc = self._client.job(remote_id)
            status = doc.get("status")

            # Artefacts as they appear, IN SEGMENT ORDER, so a stream on this
            # host can start before the whole job finishes and `offsets` records
            # a real boundary per segment rather than one entry for everything.
            #
            # Sorted numerically, not lexically. The names are <job>.<n>.f32, so
            # a plain sort puts segment 10 between 1 and 2 and every job over ten
            # segments is spoken in the wrong order - audible, wrong, and with
            # nothing reporting an error.
            fresh = [n for n in (doc.get("artefacts") or [])
                     if n.endswith(".f32") and n not in seen]
            for name in sorted(fresh, key=lambda n: self._segment_index(n, 0)):
                seen.add(name)
                index = self._segment_index(name, len(parts))
                audio = self._decode(self._client.artefact(remote_id, name))
                pause = pauses[index] if index < len(pauses) else 0.0
                piece = splice([(audio, pause)])
                if piece.size:
                    parts.append(piece)
                    if on_chunk is not None:
                        on_chunk(piece)

            if status == "done":
                record = doc.get("record") or {}
                total_tokens = int(record.get("input_tokens") or 0)
                break
            if status == "failed":
                record = doc.get("record") or {}
                raise RemoteUnavailable(
                    f"the runner failed the job: {record.get('error', 'no reason given')}")
            if status == "cancelled":
                break

            # WAS RUNNING, IS QUEUED AGAIN: the owner came back and the runner
            # handed the GPU over inside six seconds. THE NORMAL CASE, several
            # times an evening, and it is waited out rather than raised.
            #
            # Waiting is what makes "a job cannot be spoken twice" true rather
            # than aspirational. The lease on the runner is still there under
            # the same idempotency key, so the segments it already produced are
            # not produced again; `seen` means the ones already collected are
            # not collected again; and because this call never returns, `_run`
            # is never re-entered, so no encoder is rebuilt and no SSE client is
            # sent a second RIFF header half way through a stream. Every one of
            # those would have been a way to speak something twice.
            #
            # Restarting on a yield would have been the natural-looking choice
            # and is wrong in all three of those ways at once.
            # NO `parts` GUARD. It used to require a segment to have landed
            # before this would report waiting, so submitting while the owner
            # was ALREADY gaming -- the commonest case there is -- reported
            # nothing, and the job read "running" for the whole bound. The
            # count belongs in the message, not in the condition.
            if status == "queued" and not waiting:
                waiting = True
                if parts:
                    # "the machine", not "the GPU". The same code drives the
                    # processor rung, where a yield is a throttle rather than a
                    # handover of a card, and a log line naming hardware the
                    # service never asked for sends people looking for a game
                    # that is not running.
                    log.info("the runner paused %s after %d of %d segments; "
                             "waiting for its owner to finish", cfg.service,
                             len(parts), len(segments))
                else:
                    log.info("the runner has not started this job; it is "
                             "queued behind other work or its owner is using "
                             "the machine")
                if self._on_wait is not None:
                    self._on_wait(True)
            elif status == "running" and waiting:
                waiting = False
                if self._on_wait is not None:
                    self._on_wait(False)

            # Charged only while the runner is not working on this job, so a
            # long job on an idle runner is never charged at all.
            now = time.monotonic()
            if status != "running":
                waited += now - ticked
            ticked = now

            if waited > cfg.max_wait:
                # THE BOUND, and the only thing that still raises. Waiting out a
                # six-second yield is right; waiting out an entire evening of
                # gaming is not, because the local CPU would have finished long
                # ago at 0.275x realtime. Withdraw the lease on the way out so
                # the runner is not left holding a job nobody is waiting for.
                self._client.cancel(remote_id)
                raise RemoteYield(
                    f"the runner produced {len(parts)} of {len(segments)} "
                    f"segments on {cfg.service} and then had the machine taken "
                    f"back for {waited:.0f}s of the {cfg.max_wait:.0f}s "
                    "allowed; its owner is using it", delivered=len(parts))
            time.sleep(cfg.poll)

        audio = splice([(p, 0.0) for p in parts]) if parts else np.zeros(0, dtype=np.float32)
        return Spoken(audio=audio, input_tokens=total_tokens)

    @staticmethod
    def _segment_index(name: str, fallback: int) -> int:
        """Recover the segment number from `<job id>.<n>.f32`.

        The controller writes artefacts through the shared library, which names
        every one of them `<job id>.<suffix>` and refuses anything else, so the
        number is the second-to-last dot-separated field. A job id is hex, so it
        contains no dots of its own.
        """
        parts = name.split(".")
        if len(parts) < 3:
            return fallback
        try:
            return int(parts[-2])
        except ValueError:
            return fallback

    @staticmethod
    def _decode(raw: bytes) -> np.ndarray:
        """Raw float32 at 24 kHz, and it is checked rather than trusted.

        THE DEFECT THIS PREVENTS is silent and expensive. Samples arriving at
        any rate but 24000 make `duration = audio.size / SAMPLE_RATE` wrong, and
        with it `audio_seconds`, `realtime_factor`, the usage token counts and
        the rate EMA that decides whether the next request is answered
        synchronously. Every wav would ship at the wrong pitch: playable, wrong,
        and reporting no error anywhere. `check_rate` exists for exactly this.

        The rate is asserted from the parameters we SENT rather than read out of
        the bytes, because raw PCM carries no header to read it from. That is
        the trade for not paying for a container per segment, and it is why the
        runner is told the sample rate in the job body: if it ever answers at a
        different one, the contract was broken on its side and this assertion is
        the only thing that would catch it.
        """
        check_rate(SAMPLE_RATE)
        if len(raw) % 4:
            raise RemoteUnavailable(
                f"a segment was {len(raw)} bytes, which is not a whole number of "
                "float32 samples; the runner is not speaking the agreed format")
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
