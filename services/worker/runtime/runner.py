#!/usr/bin/env python3
"""The GPU child. The only process in this worker that imports torch.

WHY THIS IS A SEPARATE PROCESS AND NOT A THREAD IN THE AGENT. Killing a PID is
the only thing that reliably returns the ~300-500 MB CUDA context plus ~3 GB of
weights to the driver. torch.cuda.empty_cache() frees cached blocks but not the
context. The agent runs this inside a Windows job object with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so the whole tree dies together - including
any child torch spawns - and dies even if the agent itself is killed, because the
kernel closes the handle when the agent's process object is torn down.

THE YIELD CONTRACT, and its hard limit. The agent writes one line, "YIELD", to
this process's stdin when the user wants the GPU back. A watcher thread sets an
Event. The chunk loop checks it BEFORE starting the next generate().

  It cannot check it during. Chatterbox's generate() has no interruption point
  inside it - services/tts-long/app/synth.py says so in its own docstring, and
  DELETE /jobs/{id} documents the same limit for the local CPU worker. So the
  real floor on getting out of the user's way is one generate() call, and
  measuring that number on the 3070 is one of the things this proof of concept
  exists to do. If it comes back above about five seconds, TTS_CHUNK_MAX_CHARS
  has to come down - it is already an environment variable, so no code change.

  What this process can do is refuse to start another chunk and throw away the
  one in flight. It does exactly that. The half-finished chunk is discarded by
  design: audio only becomes real when the whole float32 array is delivered, so
  there is no partial write to corrupt, and the coordinator simply hands the
  chunk to somebody else.

STDIN EOF IS A FREE DEAD MAN'S SWITCH. Once the agent redirects stdin to send
YIELD, this process gets agent-death detection for nothing: if the agent dies,
the pipe closes, the read returns EOF, and this process exits on its own rather
than becoming an orphan holding a CUDA context on an 8 GiB card. That is a second
independent guarantee alongside the job object.

TRANSPORT IS A DIRECTORY, DELIBERATELY. There is no HTTP here. Chunks arrive as
queue/pending/NNNN.json and results leave as queue/done/NNNN.f32 plus a sidecar.
That exercises the entire lease/abandon state machine - including the case that
decides everything, a chunk dropped mid-flight - with no network code and no
listening port on a machine running kernel-mode anti-cheat. See the README for
what is real and what is stubbed.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

YIELD = threading.Event()
LOG_LOCK = threading.Lock()


def log(msg, **kw):
    """One JSON object per line on stderr. The agent captures it into worker.log
    and the timing fields are the deliverable, so they are machine-readable
    rather than prose."""
    rec = {"t": round(time.time(), 3), "msg": msg}
    rec.update(kw)
    with LOG_LOCK:
        sys.stderr.write(json.dumps(rec) + "\n")
        sys.stderr.flush()


def watch_stdin():
    """Block on stdin. A YIELD line means stop; EOF means the agent is gone."""
    try:
        for line in sys.stdin:
            if line.strip().upper() == "YIELD":
                log("yield requested by agent")
                YIELD.set()
                return
    except Exception as exc:                      # pragma: no cover - pipe teardown
        log("stdin watcher error", error=str(exc))
    # Falling out of the loop is EOF: the agent's end of the pipe closed.
    log("stdin closed; agent is gone")
    YIELD.set()


class Runtime:
    """Loads the model once, then answers chunks until told to stop."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.load_seconds = None

    def load(self):
        t0 = time.monotonic()
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if self.device == "cuda" and not torch.cuda.is_available():
            # Loud, not silent. The CPU-only wheel is the most common way this
            # setup fails and everything about it looks fine until the worker
            # turns out slower than the NAS.
            raise SystemExit(
                "FATAL: torch cannot see the GPU (%s). This is almost always the "
                "CPU-only wheel from PyPI instead of the cu126 wheel from "
                "download.pytorch.org." % torch.__version__
            )

        log("loading model", torch=torch.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.load_seconds = time.monotonic() - t0

        vram = None
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            vram = round((total - free) / 2 ** 20, 1)
        # COLD START. Process spawn to model ready. This decides whether
        # abandon-and-reload is viable at all: if it is 60 s and real idle windows
        # are two minutes, a worker that yields correctly never finishes anything.
        log("model ready", load_seconds=round(self.load_seconds, 3), vram_used_mib=vram)

    def speak(self, lease):
        """One chunk. Returns (float32 numpy array, timings)."""
        import numpy as np

        text = lease["text"]
        t0 = time.monotonic()
        wav = self.model.generate(
            text,
            language_id=lease.get("language", "en"),
            audio_prompt_path=lease.get("reference_path") or None,
            exaggeration=lease.get("exaggeration", 0.5),
            cfg_weight=lease.get("cfg_weight", 0.5),
            temperature=lease.get("temperature", 0.8),
        )
        compute = time.monotonic() - t0

        audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        sample_rate = lease.get("sample_rate", 24000)
        audio_seconds = len(audio) / float(sample_rate)
        return audio, {
            # THE number GAB-628 is asking for, against the NAS's measured 0.138x.
            "compute_seconds": round(compute, 3),
            "audio_seconds": round(audio_seconds, 3),
            "realtime_factor": round(audio_seconds / compute, 3) if compute > 0 else None,
            "samples": int(len(audio)),
            "chars": len(text),
        }


class FakeRuntime:
    """The same lifecycle, the same yield behaviour, no torch and no GPU.

    WHY THIS EARNS ITS PLACE. It lets the whole agent-to-child yield path - stdin
    YIELD, the EOF dead man's switch, the job object kill, the abandon-and-restore
    bookkeeping and the latency measurement - be exercised on the real machine
    BEFORE anyone downloads 5.6 GiB, and re-run afterwards whenever that path
    changes. The expensive part of this system is not the part most likely to be
    wrong, and it should not have to be present to test the part that is.

    It burns CPU rather than sleeping, so that "the child was busy and had to be
    killed" is a real state and not a timer that was always going to fire.
    """

    def __init__(self, chunk_seconds=3.0):
        self.device = "fake"
        self.model = object()
        self.load_seconds = None
        self.chunk_seconds = chunk_seconds

    def load(self):
        t0 = time.monotonic()
        time.sleep(1.0)          # stand-in for weights coming off disk
        self.load_seconds = time.monotonic() - t0
        log("model ready", load_seconds=round(self.load_seconds, 3), fake=True)

    def speak(self, lease):
        import array
        import math
        t0 = time.monotonic()
        # Deliberately uninterruptible, exactly like the real generate(). Nothing
        # in here checks YIELD, because the real one cannot either.
        while time.monotonic() - t0 < self.chunk_seconds:
            sum(math.sqrt(i) for i in range(20000))
        compute = time.monotonic() - t0
        sample_rate = lease.get("sample_rate", 24000)
        n = int(sample_rate * self.chunk_seconds)
        audio = array.array("f", (0.0 for _ in range(n)))
        return audio, {
            "compute_seconds": round(compute, 3),
            "audio_seconds": round(n / float(sample_rate), 3),
            "realtime_factor": round((n / float(sample_rate)) / compute, 3),
            "samples": n,
            "chars": len(lease.get("text", "")),
            "fake": True,
        }


def run_queue(root: Path, device: str, once: bool, fake=False, chunk_seconds=3.0):
    pending = root / "pending"
    done = root / "done"
    working = root / "working"
    for d in (pending, done, working):
        d.mkdir(parents=True, exist_ok=True)

    rt = FakeRuntime(chunk_seconds) if fake else Runtime(device)
    rt.load()

    served = abandoned = 0
    first_chunk_at = None
    started = time.monotonic()

    while not YIELD.is_set():
        leases = sorted(pending.glob("*.json"))
        if not leases:
            if once:
                break
            time.sleep(0.25)
            continue

        lease_path = leases[0]
        # Claim it by moving it out of pending. If we are killed after this, the
        # file is in working/ and the sweep at startup puts it back - which is the
        # directory-queue equivalent of a lease expiring.
        claimed = working / lease_path.name
        try:
            lease_path.rename(claimed)
        except OSError:
            continue

        lease = json.loads(claimed.read_text(encoding="utf-8"))

        # The check that matters, in the only place it can happen. Between chunks.
        if YIELD.is_set():
            claimed.rename(pending / lease_path.name)
            abandoned += 1
            break

        try:
            audio, timings = rt.speak(lease)
        except Exception as exc:
            log("chunk failed", chunk=lease_path.stem, error=str(exc))
            claimed.rename(pending / lease_path.name)
            abandoned += 1
            continue

        if YIELD.is_set():
            # Computed, but the user came back while we were inside generate().
            # Throw it away and hand the chunk back. Delivering it would mean
            # another few hundred milliseconds of disk and bookkeeping at exactly
            # the moment the user wants their frames.
            claimed.rename(pending / lease_path.name)
            abandoned += 1
            log("discarded a finished chunk", chunk=lease_path.stem, **timings)
            break

        (done / (lease_path.stem + ".f32")).write_bytes(
            audio.tobytes() if hasattr(audio, "tobytes") else bytes(audio))
        meta = dict(lease)
        meta.pop("text", None)                    # the sidecar is telemetry, not content
        meta.update(timings)
        meta["text_length"] = len(lease.get("text", ""))
        (done / (lease_path.stem + ".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        claimed.unlink()

        served += 1
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - started
            # Spawn to first chunk delivered, including the model load.
            log("first chunk delivered", seconds_from_start=round(first_chunk_at, 3))
        log("chunk done", chunk=lease_path.stem, **timings)

    # Anything still in working/ was interrupted. Put it back.
    for leftover in working.glob("*.json"):
        leftover.rename(pending / leftover.name)
        abandoned += 1

    log("runner exiting", served=served, abandoned=abandoned,
        yielded=YIELD.is_set(), load_seconds=round(rt.load_seconds or 0, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True, help="directory holding pending/ and done/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--once", action="store_true",
                    help="drain the queue and exit rather than waiting for more")
    ap.add_argument("--no-stdin-watch", action="store_true",
                    help="for running by hand, where stdin is a terminal")
    ap.add_argument("--fake-model", action="store_true",
                    help="exercise the yield path with no torch and no GPU")
    ap.add_argument("--chunk-seconds", type=float, default=3.0,
                    help="how long a fake chunk takes (default 3)")
    args = ap.parse_args()

    if not args.no_stdin_watch:
        threading.Thread(target=watch_stdin, daemon=True).start()

    run_queue(Path(args.queue), args.device, args.once,
              fake=args.fake_model, chunk_seconds=args.chunk_seconds)


if __name__ == "__main__":
    main()
