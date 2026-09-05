# Provision the GPU job runtime on a Windows machine with no Python.
#
# READ THIS FIRST. NOTHING IN THIS FILE HAS BEEN RUN ON spring. The investigation
# that produced the agent was read-only by instruction: no installs, no settings
# changes, nothing left running. This script is therefore the WRITTEN-DOWN STEP
# the user runs themselves, not something that was executed on their behalf. It
# is deliberately idempotent and deliberately confined to one directory.
#
# WHY THIS IS SEPARATE FROM THE AGENT AT ALL.
#
# The agent is 40 KB and needs nothing installed - measured, it compiles and runs
# on spring with the C# compiler that ships inside Windows. The job runtime is
# the opposite kind of thing: Chatterbox is torch, and torch with CUDA 12 wheels
# is roughly 2.5-3 GB on disk. No packaging format makes that small. Bundling it
# into the agent would turn a 40 KB file that can be copied anywhere into a
# multi-gigabyte download that has to be rebuilt whenever a threshold changes.
#
# So they are split along the line where the size is:
#
#   ai-voice-worker.exe   40 KB, no dependencies, does the detection
#   runtime\              ~3 GB, provisioned once by this script, does the work
#
# and the agent supervises the runtime as a child process in a job object, so it
# can take the GPU back inside the grace period.
#
# WHY EMBEDDABLE PYTHON RATHER THAN AN INSTALLER OR THE STORE.
#
#   * python-3.11-embed-amd64.zip is a ZIP, not an installer. It writes nothing
#     to the registry, adds nothing to PATH, needs no administrator, and is
#     removed by deleting the folder. On a machine that is somebody's gaming PC
#     and runs kernel-mode anti-cheat, "unzip a folder" is a much easier thing to
#     justify than "run an installer that touches system state".
#   * The Microsoft Store python is what is currently on PATH on spring, and it
#     is the alias stub: measured, `python` resolves to
#     C:\Users\htcga\AppData\Local\Microsoft\WindowsApps\python.exe and prints
#     "Python was not found". `py` is absent entirely. Store Python also runs
#     under a package identity with a virtualised filesystem, which is a poor
#     host for CUDA libraries.
#   * winget install would work and is one line, but it changes machine state
#     the user did not ask to change, and it is not removable by deleting a
#     folder.
#
# The cost, stated plainly: the embeddable distribution has no ensurepip, no
# tkinter and no venv, so pip has to be bootstrapped by hand (done below) and the
# ._pth file has to be edited to re-enable site-packages (also done below). That
# is about fifteen lines of ceremony, once, in exchange for a runtime that is a
# directory rather than an installation.

param(
  [string] $Root       = "$env:LOCALAPPDATA\ai-voice-worker\runtime",
  [string] $PyVersion  = "3.11.9",
  # cu124 wheels: the driver on spring is 610.47 with CUDA UMD 13.3 (measured),
  # which is far newer than the 12.4 runtime the wheels carry, and the CUDA
  # minor-version compatibility guarantee runs forwards, so this is safe.
  [string] $TorchIndex = "https://download.pytorch.org/whl/cu124"
)

$ErrorActionPreference = 'Stop'
$py = Join-Path $Root 'python'

Write-Host "Provisioning into $Root"
New-Item -ItemType Directory -Force -Path $Root, $py | Out-Null

# --- 1. embeddable Python -----------------------------------------------------
$zip = Join-Path $env:TEMP "python-$PyVersion-embed-amd64.zip"
if (-not (Test-Path (Join-Path $py 'python.exe'))) {
  $url = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
  Write-Host "  downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath $py -Force
  Remove-Item $zip -Force
}

# --- 2. re-enable site-packages ----------------------------------------------
# The embeddable build ships a python311._pth with "import site" commented out,
# which is what stops pip from working at all. Uncommenting it is the documented
# way to make the distribution usable as an application runtime.
Get-ChildItem (Join-Path $py 'python*._pth') | ForEach-Object {
  $t = Get-Content $_.FullName
  if ($t -match '^\s*#\s*import site') {
    ($t -replace '^\s*#\s*import site', 'import site') | Set-Content $_.FullName -Encoding ASCII
    Write-Host "  enabled site in $($_.Name)"
  }
}

# --- 3. pip -------------------------------------------------------------------
if (-not (Test-Path (Join-Path $py 'Scripts\pip.exe'))) {
  $gp = Join-Path $env:TEMP 'get-pip.py'
  Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $gp
  & (Join-Path $py 'python.exe') $gp --no-warn-script-location
  Remove-Item $gp -Force
}

# --- 4. the payload -----------------------------------------------------------
# torch first and from its own index, so pip does not resolve the CPU-only wheel
# from PyPI and leave you with a worker that cannot use the GPU it exists for.
Write-Host "  installing torch (this is the ~2.5 GB part)"
& (Join-Path $py 'python.exe') -m pip install --no-warn-script-location `
    --index-url $TorchIndex torch torchaudio

Write-Host "  installing chatterbox"
& (Join-Path $py 'python.exe') -m pip install --no-warn-script-location chatterbox-tts

# --- 5. prove the GPU is actually reachable -----------------------------------
# WHY THIS CHECK IS NOT OPTIONAL. Installing the CPU-only torch wheel by accident
# is the single most common way this kind of setup fails, and it fails silently:
# everything imports, everything runs, and the job is slower than the NAS it was
# meant to relieve. Fail loudly here instead.
$probe = @'
import sys, torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FAIL: no CUDA. The CPU-only wheel was probably installed.")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print("vram free/total MiB:", free // 1048576, "/", total // 1048576)
'@
$probeFile = Join-Path $Root 'check_gpu.py'
$probe | Set-Content $probeFile -Encoding UTF8
& (Join-Path $py 'python.exe') $probeFile
if ($LASTEXITCODE -ne 0) { throw "GPU check failed - see above" }

Write-Host ""
Write-Host "Runtime ready at $py"
Write-Host "Point worker.ini at it:"
Write-Host "  JobCommand   = $py\python.exe"
Write-Host "  JobArguments = -m worker.run"
