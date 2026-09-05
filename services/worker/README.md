# ai-voice-worker

A Windows tray agent that lends an idle gaming PC's GPU to ai-voice, and gets out
of the way the instant its owner wants it back.

**This is a proof of concept.** It exists to answer two questions nobody could
answer before: whether GPU idleness can be detected reliably enough to trust with
somebody's gaming session, and whether Chatterbox can run on a machine with no
Python installed. A polished tray application that cannot answer those would not
have been the deliverable.

Read [What is real, what is stubbed](#what-is-real-what-is-stubbed) before
anything else. Quite a lot here is deliberately not built.

---

## Why only Chatterbox, and why only this machine

`ai-voice` has five services. Exactly one of them is slower than realtime:
Chatterbox voice cloning runs at **0.138x** on the NAS's CPU, which is about
seven minutes of compute per minute of audio. Parakeet already does 47-63x
realtime on CPU and Kokoro beats realtime, so neither wants a GPU and neither is
in scope. An RTX 3070 with 8 GiB fits Chatterbox's 0.5B model comfortably.

That single number is the whole justification (Linear GAB-627, GAB-628).

---

## The problem, and why the obvious answer does not work

The obvious answer is "run when `utilisation.gpu` is below some threshold". On
this machine that fails in three separate ways, each measured rather than
assumed. The 150-sample idle baseline in
`tests/fixtures/_recorded_idle_spring.csv` was taken with the desktop up and
untouched:

| signal | idle reading |
|---|---|
| `utilisation.gpu` | **5-6%**, never lower, in all 150 samples |
| `clocks.mem` | **810 MHz**, zero variance |
| `pstate` | **P5**, 150 of 150 |
| `power.draw` | 34.55-36.02 W |
| `memory.used` | 388 MiB, zero variance |

**Wrong low.** A threshold under 5% can never be satisfied here. A threshold over
6% is inside the noise. The naive test is not merely imprecise, it is
unreachable.

**Wrong high.** A browser decoding video raises utilisation without touching the
shaders. Yielding to that would make the worker useless on a desktop that is
nearly always playing something.

**Wrong late, and this is the expensive one.** A game paused at a menu draws
nothing. It still owns its VRAM and resumes in one frame. `utilisation.gpu` calls
that idle and hands the card to a job.

And the obvious fix for the third case is not available:
`nvidia-smi --query-compute-apps` returns `used_memory` as **`[N/A]` for every
process** on this machine, because of the WDDM driver model. `pmon` shows `-` per
process too. Per-process GPU memory simply is not there.

**What is there is the OS.** `\GPU Process Memory(*)\Dedicated Usage` read dwm at
169.4 MiB, CamoStudio at 112.3, explorer at 40.3, and the adapter total within
1.5% of nvidia-smi's own figure for the same instant. `\GPU Engine(*)` splits
`3d` 1.06 from `videodecode` 0.00 — which is exactly what separates a game from a
YouTube tab.

---

## The policy: a ladder, not a threshold

### Tier 0 — blind. Fail closed.

`nvidia-smi` stream dead, or the agent is not in the console session. The state is
**`Blocked`**, which is deliberately a different thing from `Busy`: "I cannot see
you" is not "a game is running", and the tray says so, because one of those means
something needs fixing and the other does not.

The policy starts in `Blocked`. An agent that has seen no samples knows nothing,
and "knows nothing" must never mean "help yourself to the GPU".

### Tier 1 — vetoes. A game **exists**. Instant, no confirmation.

These survive a pause, an alt-tab and a minimise, which `utilisation.gpu` does
not.

| signal | source | why |
|---|---|---|
| `RunningAppID != 0` | `HKCU\Software\Valve\Steam` | Steam writes it **before the first frame is rendered** |
| `vgc` running | service control manager | starts with Valorant, before the window. **`vgk` is always running and is useless** — measured Running/System while `vgc` was Stopped/Manual |
| named game process | `GameProcessNames` | survives pause and minimise |
| ≥512 MiB held by a non-allowlisted process | `GPU Process Memory` | the paused-game backstop; three times the largest thing on an idle desktop (dwm, 169.4 MiB) |
| full-screen foreground window | `GetWindowRect` vs `GetMonitorInfoW` | not proof of a game, but proof the user is doing one thing with their whole screen |

Our own child process is always exempt. Counting our own VRAM as evidence of a
user would make the policy oscillate the moment a job allocated anything.

### Tier 2 — votes. Something is **drawing**. Three consecutive seconds.

Memory clock > 1500 MHz · pstate outside P5/P8/P12 · power > 70 W · utilisation >
25% · 3D engine > 20%.

**Video decode deliberately does not yield.** **Input idleness deliberately does
not gate** — the request was GPU-only. `GetLastInputInfo` is collected into
`state.json` for the trace and never consulted by `Evaluate()`.

### The asymmetry is the safety property

```ini
BusyConfirmSamples   = 3     # ~3 s for a tier-2 vote; tier 1 is instant
ClearCooldownSeconds = 90    # busy -> available
YieldGraceSeconds    = 2
FastPollMs           = 1000
CounterPollMs        = 2000
```

Going busy is instant. Coming back takes **90 seconds of continuous clear**. A
false busy costs one job; a false idle costs the user their game.

---

## The idleness policy is a separate, testable module

This is the part that decides whether the user trusts the thing, so it is
exercisable with no GPU, no game, no driver and no console session.

```powershell
powershell -ExecutionPolicy Bypass -File tests\run-tests.ps1
```

**38 assertions, 0 failures**, about a second, compiled by the C# compiler inside
Windows. It links `Model.cs`, `Config.cs`, `Policy.cs` and `Replay.cs` and
**nothing else** — no `Signals.cs`, so no user32, no PDH, no registry, no service
control manager. That split is why the tests exist at all: before it, compiling
the policy meant compiling the P/Invoke, and the decision could only be exercised
on a Windows desktop with a GPU in it, which is to say only where getting it
wrong is expensive.

There is **no second implementation**. The tests drive the same
`Policy.Evaluate` the tray runs; only the input differs.

Each test is named after the mistake it stops:

| fixture | asserts |
|---|---|
| `idle_desktop.csv` | 150 s of **real measured idle** ends `Available` and is never `Busy` |
| `session0_blocked.csv` | **real**, including session columns: an agent over SSH refuses for ever and names the session, not a game |
| `steam_launch.csv` | Steam vetoes on the row the appid appears, while the GPU is still P5 at 810 MHz — proving it fires before any load signal could |
| `paused_game_vram.csv` | a game holding 3.8 GiB with **no Steam appid and an idle GPU** is still caught |
| `video_playback.csv` | decoder at 41% never yields |
| `tier2_load_only.csv` | a load vote fires on the third consecutive sample, not the second or fourth |
| `valorant_vgc.csv` | caught by `vgc` |
| `smi_dead.csv` | a dead stream is `Blocked`, never `Busy` |
| `cooldown_90s.csv` | `Available` returns at exactly row 119, not 118 |
| `alt_tab_midgame.csv` | alt-tabbing out of a game never frees the GPU, even after the GPU falls back to idle |
| `own_job_vram.csv` | our own 3.2 GiB is not a user — **and the same 3.2 GiB held by anything else is** |
| `locked_fullscreen.csv` | a lock screen is not a full-screen game |
| `_recorded_vram_spring.csv` | none of the **15 real processes** on an idle desktop trips the veto |

**The fixture format is the calibration format.** `Replay.cs` reads exactly what
`--calibrate` writes, so twenty minutes of a real match drops into
`tests/fixtures/` and becomes a regression test with nobody transcribing numbers.

`tests/gen_fixtures.py` regenerates the synthetic ones and its docstring is where
the real/invented line is drawn.

---

## What is proven, and what is not

### Measured, on the real machine

| claim | result |
|---|---|
| Policy decides correctly on recorded samples | **38/38**, no GPU needed |
| Builds with no SDK, no NuGet, no network | **49,664 bytes**, zero warnings at `/warn:4`, built on `spring` itself |
| Fails closed from the wrong session | `--once` returns `"state":"blocked"`, `"can_run":false` |
| Counter read cost, in-process | **98 ms** (vs `Get-Counter`'s measured 1069-1851 ms) |
| **Cooperative yield: YIELD on stdin → child gone** | **20 ms** |
| **Stubborn yield: child ignores stdin → job object kills the tree** | **2002 ms**, bounded exactly by `YieldGraceSeconds` |
| Runner abandons cleanly, chunks restored | verified: 6/6 back in `pending`, none stuck |
| stdin EOF dead-man's switch | verified: agent gone → runner exits without taking work |
| uv fetches its own CPython with **no system Python** | **CPython 3.12.14 in 1.14 s** with `env -i` — empty environment, no PATH |
| Lock pins the **GPU** wheel | `torch 2.6.0+cu126` from `download.pytorch.org` |
| Download size | **5.55 GiB** (measured, see below) |

### Not proven, and it matters

**The busy side of every tier-2 threshold.** 1500 MHz / P0 / 70 W / 25% / 20% are
margins reasoned from an idle baseline. Read-only probing on somebody's gaming PC
cannot generate GPU load, so nothing has ever been observed under a game. Until
`--calibrate` runs, **tier 1 carries all the weight** — by design, but it is a
single point of failure for any game that Steam and the process list do not know.

**Every signal that has never run outside session 0**: foreground, full-screen,
lock, input. All measurements so far came from SSH, where `GetForegroundWindow()`
returns 0. The first run from the console is also the first test of that code.
That the untested half is untested *because the safety mechanism worked* is
correct, and still leaves it untested.

**Chatterbox's realtime factor on the 3070.** Nobody has it. It is the entire
justification for GAB-628.

**Cold start: spawn to first chunk.** If it is 60 s and real idle windows are two
minutes, a worker that yields correctly never finishes anything and the design is
net negative on merit rather than on a bug. `runner.py` logs
`first chunk delivered` for exactly this.

**Killing torch mid-CUDA-kernel** is expected-safe but unobserved on this card.

**False-idle rate over a week.** This is the actual deliverable, not the tray
icon. False busies do not count in any quantity; a single false idle during a
ranked match ends the project.

---

## A finding about this specific machine

`spring` runs `C:\gpu-clocklock.ps1` on a loop, which applies
`nvidia-smi -lgc 1800` and `-pl 270` every ten minutes (GAB-578). **That locks the
core clock**, which is why `clocks.sm` reads exactly 1800 in all 150 baseline
samples. The core clock is a constant here and worthless as a signal — so the
policy uses `clocks.mem`, which `-lgc` does not pin and which measured 810 MHz
with zero variance.

If anyone ever adds `-lmc` to that script, the memory-clock vote dies **silently**:
tier 1 keeps working, so it would be a quiet loss of sensitivity rather than a
visible break.

---

## Packaging

Two pieces, split along the line where the size is:

```
ai-voice-worker.exe   49,664 bytes   no dependencies    does the detection
runtime\              ~8.0 GiB       one script, once   does the work
```

### The agent needs nothing, because its runtime is the operating system

Built by `C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe` — the C#
compiler that ships inside Windows. No SDK, no NuGet, no build host, no network.
It was built and run on `spring` itself over SSH. **The runtime dependency the
question was about does not exist for the agent.**

The cost is **C# 5**: no string interpolation, no `?.`, no `nameof`. The source is
written to it. The tray icon is drawn rather than shipped so the deliverable stays
one file.

### The runtime is `uv` with a locked, on-disk venv

AppImage is Linux. The honest Windows equivalent is **one folder you delete to
uninstall, that touched nothing outside itself** — because nothing produces a
self-contained *file* here: 4.28 GiB of CUDA DLLs and 3 GiB of weights land on
disk as files under every option.

`uv.exe` is a single ~17 MB binary with no prerequisites. It fetches its own
CPython. Everything is confined by `UV_PYTHON_INSTALL_DIR`,
`UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR` and — most importantly — **`HF_HOME`**,
so weights go to `models\` and survive a reinstall.

**Proven, not asserted.** With `env -i` — a completely empty environment, no PATH,
no system Python reachable — uv downloaded CPython 3.12.14 and installed it in
**1.14 seconds**, then created a venv whose `base_prefix` points inside the
isolated directory, and installed packages into it.

That also kills the embeddable-Python option outright: python.org's newest
embeddable build is **3.12.10** and 3.12.11 is a **404** (the branch is
security-fix-only). Choosing it would pin a gaming PC to a permanently unpatched
interpreter, while uv fetches 3.12.14.

### The trap this hit, and you will too

`runtime/pyproject.toml` names `torch` and `torchaudio` as **direct**
dependencies even though `chatterbox-tts` already requires both. That is not
style. `[tool.uv.sources]` only applies to direct dependencies, so with torch left
transitive the lock resolved **`torch 2.6.0` from pypi.org — the CPU-only wheel**.
Verified here, then fixed; `uv lock` then reported:

```
Updated torch v2.6.0 -> v2.6.0+cu126
Updated torchaudio v2.6.0 -> v2.6.0+cu126
```

This is the single most common way this setup fails and **it fails silently**:
everything imports, everything runs, and the worker turns out slower than the NAS
it was meant to relieve. `provision.ps1` asserts `torch.cuda.is_available()` and
`"+cu" in torch.__version__` as a second line of defence.

The previously committed `provision.ps1` had exactly this bug in pip form and has
been rewritten.

### Why the version pin is forced, not chosen

`chatterbox-tts==0.1.7` hard-pins `torch==2.6.0`, which exists on Windows as
cu124 and cu126 only. Python 3.14 would unlock torch ≥2.9, but 3.14 removed
`pkg_resources`, which `resemble-perth` imports — already documented at
`services/tts-long/requirements.txt:30`. **Python 3.12 + cu126 overrides nobody's
pin.** Resolved: 112 packages, `numpy 1.26.4`, `transformers 5.2.0`.
`gradio 6.8.0` and `pre-commit` are hard runtime deps of chatterbox — upstream's
mistake, inherited, ~40 MB out of 5,600, not worth stripping now.

### The honest download size

| | measured |
|---|---|
| `torch 2.6.0+cu126` win_amd64 wheel | **2496.1 MB** |
| `torchaudio` | 4.2 MB |
| the other 109 wheels | 251.2 MB |
| Chatterbox weights (6 files, `allow_patterns`) | **3208.9 MB** |
| **total download, once** | **≈ 5960 MB = 5.55 GiB** |
| on disk afterwards | ≈ 8.0 GiB, against 242 GB free |

**No packaging choice makes this smaller.** Every option downloads the same bytes,
and a frozen exe would still need the network for the weights, so "works offline
out of the box" was never available.

The wheel ships `cudart`, `cublas`, `cudnn_*`, `cufft`, `cusolver`, `cusparse`,
`curand`, `nvrtc` and `nvJitLink` in `torch\lib\`. **No CUDA Toolkit is required
and none should be installed.** The only host dependencies are the display driver
and the MSVC 2015-2022 x64 runtime, both already present.

### What was rejected

| option | why not |
|---|---|
| **PyInstaller `--onefile`** | unpacks the whole payload to temp **on every launch** — 4.3 GiB of DLL writes per start — and AV-flags. Poor manners next to kernel-mode anti-cheat |
| **PyInstaller `--onedir` / Nuitka** | produces the directory uv would have produced, minus reproducibility and minus `pip install`-a-fix, and makes us the redistributor of NVIDIA's DLLs and cuDNN's separate SLA. Nuitka additionally fights `torch.jit` and compiles for hours |
| **MSIX** | needs a cert in the **LocalMachine Trusted People store from an admin PowerShell**. Admin, on a gaming PC, for an 8 GiB package |
| **Embeddable Python zip** | no `ensurepip`, no `venv`, a `._pth` that must be hand-edited — and it is a dead end at 3.12.10 |
| **WSL2 / Docker** | neither installed. `wsl --install` is admin + reboot + hypervisor features on a machine running Vanguard |

**Costs, plainly:** first run needs network and takes minutes; there is a `.venv`
a curious user could break; **the exe is unsigned**, so SmartScreen warns on first
run and Mark-of-the-Web must be cleared. A code-signing certificate is the honest
fix and is out of scope here.

---

## Vanguard safety

Riot Vanguard is kernel-mode and always resident on this machine. Everything here
is passive and ordinary: `Shell_NotifyIcon`, registry **reads**,
`ServiceController.Status`, performance counters, and spawning `nvidia-smi`.

**No `SetWindowsHookEx`, no `ReadProcessMemory`, no injection, no driver, no input
synthesis, no overlay, and no listening socket.** The parts that could have looked
like automation — anything reaching into another process — are exactly the parts
designed out by reading OS counters instead.

The residual risk is social, not technical: an unsigned executable from an unknown
publisher, running at login, spawning a GPU process on a machine with kernel-mode
anti-cheat. **The cheapest mitigation is the recommended first step below.**

---

## Install

Nothing here needs administrator, a reboot, or a PATH edit.

### 1. Build and install the agent

```powershell
git clone <this repo>
cd services\worker
powershell -ExecutionPolicy Bypass -File build.ps1
powershell -ExecutionPolicy Bypass -File install.ps1
```

That writes `%LOCALAPPDATA%\ai-voice-worker\` and one `HKCU\...\Run` value. It
installs as a **Run key, not a service**, and that is load-bearing: a service runs
in session 0, where `GetForegroundWindow()` returns 0 and `GetLastInputInfo()`
reported **620,953 ms** of "idle" with the user sitting at the machine. Tier 0
detects that and refuses to run, so a service install would fail loudly rather
than work badly.

### 2. Run it in **Off** mode for a week first

```ini
StartMode = Off
```

**This is the recommended way to begin and it is nearly free.** Off keeps
sampling and keeps logging; it just never takes the GPU. So a week of
`worker.log` gives the false-idle record — checked against your own account of
when you were playing — before the thing has ever touched your card, and before
any anti-cheat has seen it spawn a GPU process.

Check it with:

```powershell
& "$env:LOCALAPPDATA\ai-voice-worker\ai-voice-worker.exe" --watch 30
```

### 3. Calibrate the thresholds you have not measured

The busy side of tier 2 is guesswork until this runs.

```powershell
# during twenty minutes of an actual game
& "$env:LOCALAPPDATA\ai-voice-worker\ai-voice-worker.exe" --calibrate game.csv 20
```

Or, without waiting for a match, once the runtime exists:

```powershell
.\runtime\.venv\Scripts\python.exe runtime\loadgen.py --seconds 120 --vram-mib 2000
# in another window
& ".\ai-voice-worker.exe" --calibrate busy.csv 2
```

`loadgen.py` moves utilisation, power, clocks, pstate and VRAM. It does **not**
drive the 3D engine — that needs a real renderer — so `Util3dBusyPct` still wants
a game. Drop either CSV into `tests\fixtures\` and it becomes a test.

### 4. Provision the GPU runtime (~5.6 GiB, once)

```powershell
powershell -ExecutionPolicy Bypass -File runtime\provision.ps1
```

It prints the two lines to paste into `worker.ini`. While it runs the tray shows
**"Setting up: ..."** in its own colour, because a grey icon during a 5.6 GiB
download reads as broken and gets killed at 4 GB.

### Uninstall

```powershell
Remove-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Run AiVoiceWorker
Remove-Item -Recurse "$env:LOCALAPPDATA\ai-voice-worker"
```

That is all of it. No registry beyond that one value, no service, no driver.

---

## The tray

| mode | behaviour |
|---|---|
| **Auto** | runs only when the policy says `Available`. The default, and the one everything above exists to make trustworthy |
| **Always on** | ignores the policy — **but still evaluates, logs and publishes the verdict it is overriding**, so you can benchmark on demand while a detector that is not in charge keeps collecting evidence |
| **Off** | never runs, **keeps sampling and logging** |

Mode is persisted to `worker.ini` immediately. A setting that silently resets is a
setting people stop trusting.

```
Yielded to you                     <- headline
Steam is running Counter-Strike 2  <- WHY, one line
Why? >                             <- every reason, in full, untruncated
------
( ) Auto   ( ) Always on   (o) Off
------
3 started, 1 yielded, last 20 ms
Copy diagnostics
Exit
```

**The "why" line is the feature.** The question this proof of concept has to
answer is whether the user trusts the detector, and trust comes from it being able
to say `dwm (pid 1752) holds 169 MiB` or `agent is in session 0, console is
session 1: cannot observe the user` rather than just going red.

| colour | state |
|---|---|
| green | `Available` |
| amber | `Draining` — cooling down, with the seconds left |
| red | `Busy` — yielded to you |
| grey | `Blocked` or Off |
| blue | provisioning |
| **hollow ring** | **Always-on is running against a busy verdict** — so "I am using your GPU while you game" is never invisible |

**No balloon notifications, ever.** A toast over a ranked match is the wrong
thing and it is what makes people uninstall.

`state.json` is written once a second by write-temp-then-rename, so anything on
the box can read the worker's mind **without opening a port** on a machine running
kernel-mode anti-cheat.

---

## How yielding works

**t = 0 ms.** The policy returns a tier-1 veto. State flips to `Busy`; the icon
goes red in the same tick. Three things happen and none of them blocks:

1. The lease is released — the chunk goes back to `pending` before the child has
   noticed.
2. `YIELD` is written to the child's stdin.
3. `Stop()` is queued to the thread pool **so the 1 Hz sampler keeps sampling**.

**t ≈ 20 ms** (measured). If the runner was between chunks, it exits and the
driver gets the CUDA context and the weights back.

**t = 2002 ms** (measured). Grace expires. `CloseHandle(job)` takes the whole tree
down together — including any torch child — and takes it down even if the agent
itself is killed, because the kernel closes the handle when the agent's process
object is torn down.

### The hard floor, stated plainly

**`generate()` has no interruption point inside it.** `services/tts-long`'s
`synth.py` says so in its own docstring, and `DELETE /jobs/{id}` documents the
same limit for the local CPU worker. The runner **cannot** abort mid-chunk. What
it can do is not start another one and drop the current result.

That is why the grace is **2 seconds and the kill is the normal path, not the
exception**. Waiting politely for a chunk to finish is a bounded-but-unmeasured
number of seconds of the user's frame time, and the user has full priority. The
cooperative stage exists only to save a chunk that is already computed.

Killing mid-CUDA-kernel is safe by construction: process teardown returns the
context, and audio only becomes real when the whole array is delivered, so there
is no partial write to corrupt.

**The number this does not know is how long one `generate()` takes on the 3070.**
If it is 30 s, `TTS_CHUNK_MAX_CHARS` has to come down — already an environment
variable, no code change.

---

## What is real, what is stubbed

### Real

- The whole idleness policy, and 38 tests over recorded samples.
- The tray, all three modes, the why-list, mode persistence, `state.json`,
  `worker.log`.
- The agent↔child supervision: job object, stdin YIELD, EOF dead-man's switch,
  and both yield latencies **measured on the real machine**.
- `runner.py`'s queue, lease-claim, abandon-and-restore and timing instrumentation
  — exercised end to end with `--fake-model`.
- The uv bootstrap and the hash-pinned 112-package Windows lock.

### Stubbed, deliberately

- **The entire wire protocol.** No HTTP, no socket, no TLS, no auth, no
  `/workers/*` routes. The transport is a **directory queue**:
  `queue\pending\NNNN.json` carries the lease shape verbatim and `runner.py`
  writes `queue\done\NNNN.f32` plus a JSON sidecar. This exercises the whole
  lease/abandon state machine — including the case that decides everything, a
  chunk dropped mid-flight — with zero network code and zero listening ports.
- **Zero changes to any existing service.** `tts-long`, `gateway`, `ui`, `stt`,
  `tts` and `compose.yaml` are untouched: no new route, no port, no volume, no
  key. That is the design's best evidence — a remote GPU worker that needs no
  change to the deployment topology has not moved the auth boundary.
- **The reference-clip fetch.** No content-addressed cache. Copy one `.wav` in by
  hand and put its path in the lease as `reference_path`.
- **Voice registry, language matching, capability negotiation.** One model, one
  language, one machine.
- **No audio returns to orko.** The f32le blobs stay on disk so the byte-for-byte
  diff can be run by hand later.
- **`GATEWAY_WORKER_KEYS`**, the ~10-line path-scoped credential. Not built,
  because nothing authenticates to anything yet. The current gateway key can
  synthesise, transcribe, delete jobs and read every clip, which is why a worker
  should not carry one.
- **No code signing, no auto-update, no telemetry, no scheduler, no second
  worker.** Exactly one chunk in flight, ever — which is also true of the real
  design, because `on_chunk` feeds a sequential encoder and parallel chunks would
  break `offsets` silently.

### Not yet run at all

Chatterbox itself has **never been run on this GPU**. `provision.ps1` has not been
executed on `spring` — nothing was installed there, as required. Every number in
[What is proven](#what-is-proven-and-what-is-not) about torch, CUDA, realtime
factor and cold start is therefore still open.

> **When the local-vs-remote byte-identity diff is eventually run**, it will fail
> unless it is run with a **fixed seed and `temperature=0`**. Chatterbox samples.
> Without that caveat the difference gets blamed on the transport, which will be
> the one thing that is innocent.

---

## Layout

```
src/Model.cs      plain data. No P/Invoke, so the policy is testable
src/Config.cs     every default, with the measurement behind it
src/Policy.cs     tiers, hysteresis, state machine, the VRAM veto
src/Replay.cs     recorded CSV -> Snapshots
src/Signals.cs    nvidia-smi stream, PDH counters, session, launchers
src/Agent.cs      1 Hz loop + 2 s counter loop + the job object
src/TrayApp.cs    icon, modes, why-list
src/Program.cs    --once  --watch  --calibrate  --tray
build.ps1         csc.exe -> one exe          install.ps1   HKCU Run key
tests/            run-tests.ps1, Tests.cs, gen_fixtures.py, fixtures/
runtime/          pyproject.toml, uv.lock, provision.ps1,
                  runner.py, loadgen.py, fake_job.ps1
probe/            p1..p7, the original evidence
```

`--watch` and `--calibrate` exit on their own. That is not decoration: it is what
makes the thing safe to drive over SSH against somebody's gaming PC without
leaving a process behind.

---

## What the rest of the stack would need, when this stops being a stub

Nothing yet — and that is the point. When the transport is built:
`tts-long` gains five `/workers/*` routes, a `generate=` hook in
`speak_segments`, three keys added to `_public`'s **strip set** (`assigned`,
`pieces`, `attempts` — `_public` is a denylist, so forgetting this 500s `GET
/jobs` exactly like the `_said()` bug fixed in `566f6a2`), and a 5 s lease
sweeper. The gateway gains five proxy entries and, ideally,
`GATEWAY_WORKER_KEYS`. `compose.yaml` gains nothing.

A `tts-long` restart still loses queued jobs. That is unchanged and deliberately
out of scope: this proves a **worker** can vanish safely; the **coordinator**
vanishing is GAB-627's separate question, and conflating the two is how a proof of
concept stops proving anything.

**Not evaluated: Ray Serve, LitServe, vLLM router, llama-swap.** The repo's own
recycle-before-building rule says check first. My read is that they solve provider
pooling but none solves *opportunistic* membership on a machine whose owner
outranks the scheduler — which is the only hard part here — but that read is
unverified and should be checked before this is built beyond a proof of concept.
