# ai-voice-worker

A Windows machine lends its GPU to the voice stack, but only while its owner is
not using it. One 40 KB executable, a tray icon with **Auto / Always on / Off**,
and a policy that errs towards giving the card back.

The machine this was built and measured against is `spring` — an RTX 3070, 8 GiB,
driver 610.47, Windows 11 Pro 10.0.26200, a Ryzen 7 5700X3D and 32 GB, which is
somebody's gaming PC and not a server. Everything below with a number in it was
measured there on 2026-09-05. The scripts that produced those numbers are in
[`probe/`](probe) and re-run unchanged.

## Why this exists at all

One component in the stack is slower than realtime: Chatterbox voice cloning runs
at **0.138x realtime** on the NAS's CPU, which is about seven minutes of compute
per minute of audio. Nothing else is close — Parakeet transcribes at 47-63x
realtime on CPU and Kokoro beats realtime — so nothing else has any business
asking for a GPU, and neither of them is in scope here. Chatterbox's 0.5B fits an
8 GiB card comfortably. That is the entire case, and it is worth keeping in view,
because it means this worker only ever has to run one kind of job.

## What was actually hard

Not the tray icon. The question nobody could answer in advance was whether
idleness can be detected reliably enough to be trusted with somebody's ranked
match, and the obvious answer is wrong in three separate directions.

**`utilisation.gpu < N` is unreachable on this machine.** A 90-second, 92-sample
baseline of an ordinary idle desktop — dwm, explorer, two Edge WebView2, Raycast,
Steam's helper, CamoStudio and Vanguard's tray all resident:

| | min | max | mean | p95 |
|---|---|---|---|---|
| `utilisation.gpu` | 5 % | 6 % | 5.13 % | 6 % |
| `utilisation.memory` | 3 % | 4 % | 3.73 % | 4 % |
| `clocks.mem` | 810 MHz | 810 MHz | — | **zero variance, 92/92** |
| `pstate` | — | — | — | **P5, 92/92** |
| `power.draw` | 33.68 W | 35.13 W | 34.59 W | 34.98 W |
| `memory.used` | 416 MiB | 416 MiB | — | zero variance |

An idle desktop never reads below 5 %. A threshold under 5 can never fire; a
threshold over 6 is inside the noise. The signal is not merely imprecise, it is
the wrong instrument.

**`nvidia-smi` cannot see per-process memory here.** `--query-compute-apps`
returns `used_memory` as `[N/A]` for all fifteen resident processes, and `pmon`
returns `-` for every per-process column. That is the WDDM driver model, not a
fault, and it removes the obvious way to notice a game that is paused but still
resident — which is the expensive failure, because a paused game draws nothing
and expects to resume in one frame.

**A browser decoding video is not a game.** Utilisation rises for both.

## The signals that do work

The answer was not a better threshold. It was a different class of signal.

| | signal | source | idle reading | why it earns its place |
|---|---|---|---|---|
| 1 | `RunningAppID` | `HKCU\Software\Valve\Steam` | `0` | Steam writes the appid **before the game renders a frame**. Fires during the launcher splash, seconds ahead of any GPU evidence. |
| 2 | `vgc` service | SCM | `Stopped` | Starts with Valorant. Note `vgk`, the kernel driver, is *always* Running (StartType System) and is therefore useless — `vgc` is the one to watch. |
| 3 | game process exists | `Process.GetProcessesByName` | none | Survives pause, alt-tab and minimise. |
| 4 | per-process VRAM | `\GPU Process Memory(*)\Dedicated Usage` | dwm 172.4, CamoStudio 112.3, explorer 40.3 MiB | **The counter that replaces what nvidia-smi will not tell us.** Survives pause. |
| 5 | memory clock / pstate | `nvidia-smi` | 810 MHz, P5, no variance | The driver's own answer to "is this real work". |
| 6 | power draw | `nvidia-smi` | 33.7–35.1 W | A 3070 under load pulls 200 W+. A six-fold separation, against a 1.45 W idle spread. |
| 7 | engine-type split | `\GPU Engine(*)\Utilization Percentage` | 3d 0.95, videodecode 0.00 | Separates "rendering" from "playing a video". |
| 8 | utilisation | `nvidia-smi` | 5–6 % | Last, weakest, advisory only. |

Signal 4 is worth dwelling on. The Windows GPU performance counters give exactly
what NVML refuses to: per-process dedicated memory, and utilisation split by
engine. The two accountings cross-check — the counters reported 423.4 MiB of
adapter dedicated memory against `nvidia-smi`'s 417 MiB for the same instant, a
1.5 % disagreement, which is two ways of counting the same truth rather than two
different truths.

They are also read through PDH, a documented OS API. Nothing is injected, no
process is opened for write, no graphics API is hooked, no shell hook is
registered. On a machine with Riot Vanguard resident in the kernel at all times,
that is a constraint on the design and not a footnote: full-screen detection
compares a window rectangle to a monitor rectangle, which is what every screen
reader does, rather than reaching for `IDXGIOutput`.

## The policy

Three tiers. **Tiers 0 and 1 are vetoes** — any one alone means yield
immediately, with no confirmation window, because each answers "does a game
exist" rather than "is a game currently drawing", and the first question is the
one that matters. **Tier 2 votes**, and needs three consecutive seconds to carry.

**Tier 0, fail closed.** The `nvidia-smi` stream is stale or dead; or the agent is
not in the console session. Blindness must never mean "help yourself".

**Tier 1, a game exists.** Steam has a game open; `vgc` is running; a named game
process exists; a non-allowlisted process holds ≥ 512 MiB of GPU memory; the
foreground window covers a whole monitor.

**Tier 2, something is drawing.** Memory clock above 1500 MHz; pstate outside
`P5/P8/P12`; power above 70 W; utilisation above 25 %; 3D engine above 20 %.

Deliberately **not** a load vote: `utilisation.decoder` and the `videodecode`
engine. Video playback uses a fixed-function block, not the shaders. A Chatterbox
job and a YouTube tab can share this card, and yielding to a video would make the
worker useless on a desktop that is nearly always playing something. This is a
decision about what to ignore, not a threshold.

Also deliberately not a gate: **user input**. The user asked for a GPU worker, and
someone typing in an editor is not using the GPU. `GetLastInputInfo` is collected
and logged, but it does not decide.

### Where the thresholds come from

| setting | default | measured basis |
|---|---|---|
| `MemClockBusyMhz` | 1500 | idle floor 810 MHz, zero variance over 92 samples |
| `PowerBusyWatts` | 70 | idle ceiling 35.13 W; 70 is double the entire idle draw |
| `UtilGpuBusyPct` | 25 | idle p95 6 %; four times the noise floor |
| `ForeignVramBusyMiB` | 512 | largest idle consumer dwm at 172.4 MiB; three times it |
| `BusyConfirmSamples` | 3 | 3 s at the 1 Hz sample rate |
| `ClearCooldownSeconds` | 90 | longer than a level load or a round break |

### The asymmetry, which is the whole design

Detection is fast and release is slow. A false "busy" costs one job. A false
"idle" costs the user their game. So the confirmation window is three seconds and
the cooldown before the GPU is taken back is ninety.

### How long it takes to notice a game starting

This is the number the feature will be judged on.

| what starts | detected by | detection | + yield grace | total |
|---|---|---|---|---|
| a Steam game | `RunningAppID`, before the first frame | ≤ 1 s | 5 s | **≤ 6 s** |
| Valorant | `vgc` service starting | ≤ 1 s | 5 s | **≤ 6 s** |
| a known game process | process enumeration | ≤ 1 s | 5 s | **≤ 6 s** |
| an unknown non-Steam game | VRAM veto at the 5 s counter poll | ≤ 5 s | 5 s | **≤ 10 s** |
| an unknown game, GPU signals only | 3 votes + ~1 s driver averaging | ≤ 4 s | 5 s | **≤ 9 s** |

The launcher signals fire *before* the game touches the GPU, so in the common
case the card is handed back during the splash screen. The worst case is a game
that uses no launcher and is not in `GameProcessNames`, at about ten seconds —
and the fix for that is one line in `worker.ini`.

## What is proven, and what is not

Proven on the real machine, by running the actual binary there:

- it **compiles with the C# compiler that ships inside Windows** — 6 files,
  `csc.exe`, exit 0, **40,448 bytes**, no SDK, no NuGet, no network
- the GPU stream, launcher registry, service query, session detection and
  performance counters all work, and the JSON snapshot is populated
- **the VRAM veto fires and names its evidence.** With the threshold dropped to
  100 MiB and dwm removed from the allowlist, it reported
  `dwm (pid 1752) holds 169 MiB of GPU memory` and refused to run. That is the
  mechanism the paused-game case depends on, exercised against real data
- the game-process veto fires (`game process present: steamwebhelper`)
- the tier-2 load votes fire and name themselves. With every trip point moved
  below the measured idle floor they reported `performance state P5`,
  `power draw 36.3 W` and `utilisation 5%` on the third consecutive sample —
  confirming both the votes and the three-sample confirmation window. They now
  correctly stand down while a veto is active, since the decision is already made
- **the session check fires, which is the fail-closed path working in the field.**
  Run over SSH the agent reports
  `agent is in session 0, console is session 1: cannot observe the user`
  and refuses to run
- cost: **3.94 % of one core, 0.25 % of this 16-thread CPU**, 35.6 MiB for the
  agent plus 30.9 MiB for its `nvidia-smi` child, which itself measured 0.00 %
- reading the counters in-process costs **98 ms**, against 1069–1851 ms for the
  `Get-Counter` cmdlet — an 11-19x difference, and the reason the agent is not a
  PowerShell script
- no process is left behind by any mode

**Not proven, and it must be said plainly: the busy side of every GPU threshold.**
The instruction for this investigation was read-only — install nothing, start
nothing that outlives a command, it is the user's gaming PC — so no load could be
generated to measure what a running game actually looks like. The idle side is
measured to four significant figures. The busy side is inferred from the idle
distribution and the known behaviour of the part.

That gap is closed by measurement rather than argument:

```powershell
.\ai-voice-worker.exe --calibrate game.csv 20
```

Run that, play for twenty minutes, and it writes every signal once a second to a
CSV and exits on its own. The tier-2 thresholds should then be set from the
observed separation instead of from a margin on the idle floor. Until that has
been done once, the tier-1 vetoes are doing the real work — which they are
designed to, and which is why the launcher signals are ranked first.

Also unverified, for the same reason: `GetLastInputInfo`, `GetForegroundWindow`
and the lock detection could only be exercised from session 0, where they are
meaningless by construction. They need one run from the desktop.

## Packaging

The ask was "something contained, like an AppImage". AppImage is Linux; the
Windows options are a portable `.exe`, an embeddable Python distribution shipped
alongside, or MSIX. The answer here is **both of the first two, split along the
line where the size is**.

| | agent | job runtime |
|---|---|---|
| what | detection, tray, policy | Chatterbox on torch |
| size | **40 KB** | ~2.5-3 GB |
| dependencies | **none** | CUDA wheels |
| install | copy the file | `runtime/provision.ps1` |
| removal | delete the file | delete the folder |

The agent is C# against .NET Framework 4.8, which is part of Windows — measured
present on spring at release 533509, with `csc.exe` in
`C:\Windows\Microsoft.NET\Framework64\v4.0.30319`. There is no runtime dependency
to manage because the runtime is the operating system. It also means the build
needs no build host: any Windows machine can produce the binary, and it was in
fact produced on the target machine over SSH.

**What that costs.** C# 5, because the in-box compiler predates Roslyn and no SDK
is installed — no string interpolation, no `?.`, no `nameof`. It is a real
constraint on the source and it is the price of a build with no toolchain.

**Why not the alternatives.** PyInstaller onefile needs a Windows build host and
a CI job, produces 30-60 MB, unpacks to a temp directory on every launch, and is
routinely flagged by antivirus — poor manners on a machine running kernel-mode
anti-cheat. MSIX needs a signing certificate and gives package identity and a
virtualised filesystem, neither of which helps something whose whole job is to
read the machine it sits on. .NET 8 is present here (8.0.29) but is not
guaranteed on any other Windows box, and a self-contained publish is ~70 MB.

The job runtime cannot be made small — torch is torch — so it is provisioned once
by a script into one directory. Embeddable Python is a ZIP: no registry, no PATH,
no administrator, removed by deleting the folder. The Store Python currently on
`PATH` on spring is the alias stub that prints "Python was not found", and `py` is
absent; the embeddable distribution sidesteps both. The cost is that it has no
`ensurepip` and no `venv`, so pip is bootstrapped by hand and `._pth` edited to
re-enable `site` — about fifteen lines, once, in `runtime/provision.ps1`.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1     # -> dist\ai-voice-worker.exe
powershell -ExecutionPolicy Bypass -File install.ps1   # -> %LOCALAPPDATA%, HKCU Run key
```

`install.ps1` uses the **Run key, not a Windows service**, and that is not a
shortcut. A service runs in session 0, and from session 0 the agent cannot see
the user at all — measured, `GetForegroundWindow()` returned 0 and
`GetLastInputInfo()` reported 620953 ms of idleness while the user was at the
machine. The policy detects this and refuses to run, so a service install would
never work rather than working badly.

Then, when the GPU work is wanted:

```powershell
powershell -ExecutionPolicy Bypass -File runtime\provision.ps1
```

and point `JobCommand` in `worker.ini` at the resulting `python.exe`. With
`JobCommand` empty the agent runs in detection-only mode, which is what this
proof of concept ships as.

## Modes

```powershell
.\ai-voice-worker.exe --once                   # one JSON snapshot, exits
.\ai-voice-worker.exe --watch 30               # a line a second for 30s, exits
.\ai-voice-worker.exe --calibrate game.csv 20  # every signal to CSV, exits
.\ai-voice-worker.exe                          # tray icon
```

Every command-line mode exits on its own. That is deliberate: it is what makes the
thing safe to drive over SSH against a machine you are a guest on.

The tray icon is green when the GPU is available, amber during the cooldown, red
when it has been given back, grey when off or blocked. The menu carries the
current state, the reason for it, the three modes and **Copy diagnostics**, which
puts the full JSON snapshot on the clipboard.

## Talking to the rest of the stack

The agent writes its status to
`%LOCALAPPDATA%\ai-voice-worker\state.json` once a second, replaced by rename so
there are no torn reads. **It opens no listening socket**, which is a deliberate
choice on a machine running kernel-mode anti-cheat.

Nothing in the five existing services was changed, and nothing needs to change for
detection to work. What the stack would need before this worker can take real
jobs, none of it done here:

- **`services/tts-long` needs a way to hand a job out and take a result back.**
  Today it owns its queue in-process. A remote worker needs to lease a job, get
  the reference audio, and return the audio — the job model exists, the transport
  does not.
- **The runtime must yield within the grace period.** The agent kills its job
  tree via a Windows job object with `KILL_ON_JOB_CLOSE`, which is reliable but
  blunt: a job killed mid-generation is lost work. A cooperative shutdown
  signal — a sentinel file, or a line on stdin — would let a job checkpoint
  instead, and `JobRunner.Stop` is where that would attach.
- **Whatever leases the work must assume the worker vanishes**, because it will,
  in about six seconds, whenever somebody launches a game.

## Layout

```text
src/Signals.cs   sampling: nvidia-smi, PDH counters, session, launchers
src/Policy.cs    the three tiers, the hysteresis, the state machine
src/Agent.cs     the loop, and the job object that supervises the runtime
src/TrayApp.cs   the icon and its menu
src/Program.cs   --once, --watch, --calibrate, --tray
build.ps1        csc.exe -> one .exe
install.ps1      copy, and the HKCU Run key
runtime/         provision.ps1: embeddable Python, pip, torch, a CUDA check
probe/           the read-only probes behind every number in this file
```

See Linear GAB-627 and GAB-628.
