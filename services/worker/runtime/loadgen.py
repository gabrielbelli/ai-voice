#!/usr/bin/env python3
"""Put a known, bounded load on the GPU so the busy thresholds can be measured.

WHY THIS IS NOT OPTIONAL. Every tier-2 threshold in Config.cs - 1500 MHz memory
clock, a pstate outside P5/P8/P12, 70 W, 25 per cent utilisation, 20 per cent on
the 3D engine - is currently a margin reasoned from a measured IDLE baseline.
Nobody has seen what this card reads under load, because read-only probing on
somebody's gaming PC cannot generate any. That makes the busy half of the policy
the one part of it that has never been exercised against reality.

The alternative to this file is asking the user to schedule a match every time
the thresholds change, which means the thresholds never get checked. Twenty lines
of torch turns `--calibrate` into something repeatable, on demand, in a minute.

WHAT IT DOES NOT DO. It is a compute load - large matrix multiplies - so it moves
utilisation, power, clocks and the pstate, and it allocates real VRAM. It does
NOT drive the 3D engine, because that needs an actual renderer; \\GPU Engine(*)
will show this under "compute" rather than "3d". So this calibrates most of tier
2 but not Util3dBusyPct, and the honest way to get that one is a real game or a
D3D sample. The README says so rather than letting the number look measured when
it is not.

IT ALWAYS EXITS. There is a hard deadline and a finally block that frees the
allocation. Nothing here outlives the command, which is a requirement on this
machine and not a nicety.

  python runner-venv\\Scripts\\python.exe loadgen.py --seconds 60 --vram-mib 2000

Run it in one window and `ai-voice-worker.exe --calibrate busy.csv 2` in another.
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="hard deadline; the process exits after this (default 60)")
    ap.add_argument("--vram-mib", type=int, default=1024,
                    help="roughly how much VRAM to hold, in MiB (default 1024)")
    ap.add_argument("--size", type=int, default=4096,
                    help="matrix edge length for the compute load (default 4096)")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="fraction of each second spent computing, 0-1 (default 1)")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible to torch")

    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))

    hold = None
    try:
        # A block of VRAM held for the whole run, so the VRAM veto can be
        # exercised too: this process is not on the allowlist, so above
        # ForeignVramBusyMiB it should read as a game.
        if args.vram_mib > 0:
            hold = torch.empty(args.vram_mib * 1024 * 1024 // 4,
                               dtype=torch.float32, device=dev)
            hold.fill_(1.0)
            print("holding ~%d MiB" % args.vram_mib)

        a = torch.randn(args.size, args.size, device=dev)
        b = torch.randn(args.size, args.size, device=dev)

        deadline = time.monotonic() + args.seconds
        print("loading for %.0f s (duty %.2f); ctrl-c to stop early"
              % (args.seconds, args.duty))
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            for _ in range(20):
                a = (a @ b).clamp_(-3, 3)
            torch.cuda.synchronize()
            busy = time.monotonic() - t0
            if args.duty < 1.0:
                # Idle the rest of the duty window, so a partial load can be
                # produced: this is how you find where the trip points actually
                # sit rather than only confirming that a full load crosses them.
                time.sleep(max(0.0, busy * (1.0 / args.duty - 1.0)))
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        # Give the VRAM back promptly rather than at interpreter teardown, so a
        # calibration trace shows a clean edge on the way down.
        del hold
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("done")


if __name__ == "__main__":
    main()
