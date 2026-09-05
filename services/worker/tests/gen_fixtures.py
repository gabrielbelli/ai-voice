#!/usr/bin/env python3
"""Build the replay fixtures the policy tests assert against.

WHAT IS REAL AND WHAT IS NOT, because this is the file where that line is drawn.

REAL, recorded from spring (RTX 3070, driver 610.47, Windows 11 Pro 10.0.26200)
over SSH on 2026-09-05, desktop up and untouched:

  _recorded_idle_spring.csv   150 samples at ~1 Hz of every nvidia-smi field the
                              policy reads. util_gpu 5-6, clocks.mem 810 MHz with
                              zero variance, pstate P5 for all 150, power
                              34.55-36.02 W, memory.used 388 MiB with zero
                              variance.
  _recorded_vram_spring.csv   the \\GPU Process Memory(*)\\Dedicated Usage counter
                              for every process holding GPU memory at idle.
                              Largest: dwm 169.4 MiB, CamoStudio 112.3,
                              explorer 40.3, steamwebhelper 33.5.
  the launcher state          RunningAppID = 0, vgc Stopped/Manual,
                              vgk Running/System, GPU Engine 3d 1.06,
                              videodecode 0.00.

NOT REAL, and this is the honest part:

  * The SESSION columns in every fixture. Everything above was measured from an
    SSH session, which Windows puts in session 0 while the console user is in
    session 1. From there GetForegroundWindow() returns 0 and GetLastInputInfo()
    reports the SSH session's idle time. So no real reading of the foreground,
    full-screen, lock or input signals exists yet. The console fixtures below set
    own_session = console_session = 1 to get past tier 0 and reach the rest of
    the policy; session0_blocked.csv is the one fixture whose session columns ARE
    real, because session 0 is exactly what was observed.

  * EVERY BUSY FIXTURE. Read-only probing on somebody's gaming PC cannot generate
    GPU load, so the busy side of every tier-2 threshold is reasoned from the idle
    baseline rather than observed. The numbers below (P0, 7000 MHz, 180 W, 95%)
    are what an RTX 3070 under a game is expected to show; they are not what one
    was seen to show. Replacing them is what `--calibrate` and runtime/loadgen.py
    are for, and until that happens the tier-1 vetoes carry the weight - which is
    the design, but it is also a single point of failure worth naming.

Run:  python3 tests/gen_fixtures.py
"""

import csv
import os
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")

COLS = ("iso_time,state,util_gpu,util_mem,enc,dec,mem_clk_mhz,sm_clk_mhz,pstate,power_w,"
        "mem_used_mib,eng_3d,eng_decode,eng_encode,gpu_healthy,counters_fresh,"
        "own_session,console_session,locked,input_idle_s,fullscreen,fg_process,"
        "steam_appid,steam_appname,vgc,game_procs,vram_top_pid,vram_top_name,vram_top_mib,reason"
        ).split(",")

T0 = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)


def real_idle_rows():
    """The 150 measured nvidia-smi samples, in order."""
    path = os.path.join(FIX, "_recorded_idle_spring.csv")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def base(i, r):
    """One row: measured GPU columns, desktop-idle everything else.

    vram_top is dwm at 169.4 MiB - the real measured largest consumer. It must
    NOT trip the veto, because dwm is allowlisted and 169.4 is under the 512 MiB
    trip point. A fixture that used a post-filter figure here could not tell that
    case apart from a clear desktop, which is why the raw name travels.
    """
    return {
        "iso_time": (T0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
        "state": "", "util_gpu": r["util_gpu"], "util_mem": r["util_mem"],
        "enc": r["util_enc"], "dec": r["util_dec"],
        "mem_clk_mhz": r["clk_mem"], "sm_clk_mhz": r["clk_sm"],
        "pstate": r["pstate"], "power_w": r["power_w"], "mem_used_mib": r["mem_used_mib"],
        "eng_3d": "1.06", "eng_decode": "0.00", "eng_encode": "0.00",
        "gpu_healthy": "1", "counters_fresh": "1",
        "own_session": "1", "console_session": "1",
        "locked": "0", "input_idle_s": "4", "fullscreen": "0", "fg_process": "explorer",
        "steam_appid": "0", "steam_appname": "", "vgc": "0", "game_procs": "",
        "vram_top_pid": "1752", "vram_top_name": "dwm", "vram_top_mib": "169.4",
        "reason": "",
    }


def busy_gpu(row):
    """Tier-2 load. SYNTHETIC - see the module docstring.

    An RTX 3070 rendering a game leaves P5/810 MHz immediately: the driver raises
    the memory clock to its rated 7000 MHz (14 Gbps effective) and the pstate to
    P0 before frame rate is reached. Power goes from ~35 W to well over 150 W.
    """
    row = dict(row)
    row.update({"util_gpu": "95", "util_mem": "62", "mem_clk_mhz": "7000",
                "sm_clk_mhz": "1920", "pstate": "P0", "power_w": "182.40",
                "mem_used_mib": "4820", "eng_3d": "94.20"})
    return row


def write(name, rows):
    path = os.path.join(FIX, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("%-28s %4d rows" % (name, len(rows)))


def main():
    real = real_idle_rows()
    idle = [base(i, r) for i, r in enumerate(real)]

    # 1. REAL GPU data, console session. 150 s of a genuinely idle desktop.
    #    Must end Available: the cooldown is 90 s and this is 150.
    write("idle_desktop.csv", idle)

    # 2. REAL, including the session columns. This is precisely what an agent
    #    started over SSH sees, and it must refuse to run for ever.
    s0 = [dict(r, own_session="0", console_session="1",
               input_idle_s="620953", fg_process="") for r in idle]
    write("session0_blocked.csv", s0)

    # 3. Steam sets RunningAppID before the game renders its first frame, so the
    #    GPU columns here stay at the measured idle values throughout. This is
    #    the fixture that proves the veto does not wait for the GPU to move.
    steam = [dict(r) for r in idle[:120]]
    for r in steam[100:]:
        r["steam_appid"] = "730"
        r["steam_appname"] = "Counter-Strike 2"
    write("steam_launch.csv", steam)

    # 4. THE CASE THE NAIVE POLICY GETS WRONG. A game paused at a menu draws
    #    nothing - every GPU column is at the measured idle value - but it still
    #    owns 3.8 GiB of VRAM and resumes in one frame. Steam is not the source
    #    here (an Epic or Game Pass title sets no RunningAppID), so the VRAM veto
    #    is the only thing standing between this and a job taking the card.
    paused = [dict(r) for r in idle[:120]]
    for r in paused[100:]:
        r.update({"vram_top_pid": "22104", "vram_top_name": "TheFinals",
                  "vram_top_mib": "3812.5"})
    write("paused_game_vram.csv", paused)

    # 5. Video playback must NOT yield. The decode engine is busy and utilisation
    #    is raised, but the 3D engine is not and the clocks stay at idle. A
    #    desktop is nearly always playing something; yielding to it would make
    #    the worker useless.
    video = [dict(r) for r in idle[:120]]
    for r in video[40:]:
        r.update({"dec": "38", "eng_decode": "41.30", "util_gpu": "14",
                  "eng_3d": "2.10", "vram_top_pid": "10896",
                  "vram_top_name": "msedgewebview2", "vram_top_mib": "312.0"})
    write("video_playback.csv", video)

    # 6. Tier-2 only: real rendering load from something with no Steam appid, no
    #    known process name and under the VRAM trip point. Must take exactly
    #    BusyConfirmSamples rows to fire, and not one fewer.
    load = [dict(r) for r in idle[:120]]
    for r in load[60:]:
        r.update(busy_gpu(r))
        r["vram_top_mib"] = "169.4"      # keep the VRAM veto out of it on purpose
        r["vram_top_name"] = "dwm"
    write("tier2_load_only.csv", load)

    # 7. Valorant: no Steam, no window yet, vgc starts with the game. vgk is
    #    always running and is therefore not the signal.
    vgc = [dict(r) for r in idle[:120]]
    for r in vgc[80:]:
        r["vgc"] = "1"
    write("valorant_vgc.csv", vgc)

    # 8. The stream died. Fail closed.
    dead = [dict(r, gpu_healthy="0") for r in idle[:60]]
    write("smi_dead.csv", dead)

    # 9. The asymmetry, end to end. Busy for 30 s, then completely clear for
    #    150 s. Available must not return one second before the 90 s cooldown.
    cool = [dict(r) for r in idle[:30]]
    for r in cool:
        r["steam_appid"] = "730"
        r["steam_appname"] = "Counter-Strike 2"
    cool += [dict(r) for r in idle[30:]]
    for i, r in enumerate(cool):
        r["iso_time"] = (T0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
    write("cooldown_90s.csv", cool)

    # 10. Alt-tab out of a running game to a browser. RunningAppID stays set
    #     because the process is still alive, the GPU falls back to near-idle,
    #     and the fullscreen flag drops. Nothing here may return Available.
    alt = [dict(r) for r in idle[:120]]
    for i, r in enumerate(alt):
        r["steam_appid"] = "730"
        r["steam_appname"] = "Counter-Strike 2"
        r["vram_top_pid"] = "22104"
        r["vram_top_name"] = "cs2"
        r["vram_top_mib"] = "3410.0"
        if i < 60:
            r.update(busy_gpu(r))
            r["fullscreen"] = "1"
            r["fg_process"] = "cs2"
            r["vram_top_name"] = "cs2"
            r["vram_top_mib"] = "3410.0"
        else:
            r["fullscreen"] = "0"
            r["fg_process"] = "msedge"
    write("alt_tab_midgame.csv", alt)

    # 11. Our own job holding 3.2 GiB must never read as a user. Without the
    #     own-pid exemption the policy would yield to itself the moment it
    #     allocated, then take the GPU back, for ever.
    own = [dict(r) for r in idle]
    for r in own:
        r.update({"vram_top_pid": "31337", "vram_top_name": "python",
                  "vram_top_mib": "3210.0"})
    write("own_job_vram.csv", own)

    # 12. Locked screen with a full-screen window behind it. The user is
    #     demonstrably not at the machine, so the full-screen veto must not fire.
    lock = [dict(r, locked="1", fullscreen="1", fg_process="LogonUI") for r in idle]
    write("locked_fullscreen.csv", lock)


if __name__ == "__main__":
    main()
