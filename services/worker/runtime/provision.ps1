# Put a working Chatterbox GPU runtime on a machine with no Python.
#
# WHAT THIS REPLACED, AND WHY. The first version of this script installed
# unpinned "torch torchaudio" from the cu124 index and then chatterbox-tts. Because
# chatterbox-tts pins torch==2.6.0, pip then resolved 2.6.0 FROM PYPI - which on
# Windows is the CPU-ONLY wheel - and silently replaced the GPU one that had just
# been downloaded. Its torch.cuda.is_available() gate caught the result, so it
# failed loudly rather than shipping a worker slower than the NAS it was meant to
# relieve. But it failed, every time.
#
# The same trap exists in uv and it was hit while writing this: a source mapping
# only applies to a DIRECT dependency, so with torch left transitive the lock
# resolved torch 2.6.0 from pypi.org. runtime/pyproject.toml therefore names torch
# and torchaudio explicitly. uv.lock is committed and hash-pinned, so what gets
# installed here is what was resolved and checked, not whatever resolves today.
#
# NOTHING IS INSTALLED ON THE MACHINE. No Python, no CUDA Toolkit, no admin, no
# reboot, no PATH edit, no registry write. Everything lands in one directory:
#
#   %LOCALAPPDATA%\ai-voice-worker\runtime\
#     uv.exe        a single ~17 MB binary with no prerequisites of its own
#     python\       CPython 3.12, fetched by uv from python-build-standalone
#     .venv\        the 112-package locked environment
#     models\       HF_HOME. ~3 GiB of weights, content-addressed
#     cache\        uv's wheel cache
#
# Uninstall is: delete that directory.
#
# WHY NOT THE EMBEDDABLE PYTHON ZIP, which this script used to use. It has no
# ensurepip and no venv, its ._pth disables site so Lib\site-packages is not
# importable until you hand-edit it, and pip has to be bootstrapped from
# get-pip.py. It is also a dead end: python.org ships 3.12.10 as the newest
# embeddable build and 3.12.11 is a 404, because the 3.12 branch is
# security-fix-only. uv fetched CPython 3.12.14 - four patch releases newer -
# in 1.14 seconds, verified with a completely empty environment and no system
# Python reachable at all.
#
#   powershell -ExecutionPolicy Bypass -File provision.ps1

[CmdletBinding()]
param(
    [string]$Root = (Join-Path $env:LOCALAPPDATA 'ai-voice-worker\runtime'),
    # The project definition and lock. Defaults to the directory this script is in,
    # which is how it works from a git checkout.
    [string]$Project = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # a progress bar over SSH is noise

$statusFile = Join-Path $Root 'provision.status'

function Set-Status([string]$text) {
    # The tray polls this file and shows it in its own colour. First run pulls
    # about 5.6 GiB; a grey icon for twenty minutes reads as broken and gets
    # killed at 4 GB, which is the worst possible moment to stop.
    try {
        New-Item -ItemType Directory -Force -Path $Root | Out-Null
        Set-Content -Path $statusFile -Value $text -Encoding UTF8
    } catch { }
    Write-Host "==> $text"
}

function Clear-Status {
    try { Remove-Item -Force $statusFile -ErrorAction SilentlyContinue } catch { }
}

try {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null

    # Everything uv does stays inside $Root. HF_HOME matters most: the weights are
    # content-addressed, are shared across versions, and are the one part of the
    # payload that must survive a reinstall - so they must never land inside .venv
    # where a rebuild would take them with it.
    $env:UV_PYTHON_INSTALL_DIR  = Join-Path $Root 'python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $Root '.venv'
    $env:UV_CACHE_DIR           = Join-Path $Root 'cache'
    $env:HF_HOME                = Join-Path $Root 'models'
    $env:UV_PYTHON_DOWNLOADS    = 'automatic'

    # ---------------------------------------------------------------- uv ---
    $uv = Join-Path $Root 'uv.exe'
    if (-not (Test-Path $uv)) {
        Set-Status 'downloading uv (17 MB)'
        $zip = Join-Path $Root 'uv.zip'
        Invoke-WebRequest -UseBasicParsing `
            -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' `
            -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $Root -Force
        Remove-Item -Force $zip
    }
    if (-not (Test-Path $uv)) { throw "uv.exe missing after download" }
    Set-Status ("uv " + (& $uv --version))

    # ------------------------------------------------------ interpreter ---
    # No system Python is consulted, and none is required. uv fetches a
    # standalone CPython build into $Root\python.
    Set-Status 'fetching CPython 3.12 (24 MB)'
    & $uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed" }

    # ---------------------------------------------------------- packages ---
    # --frozen means: use uv.lock exactly, do not re-resolve. Offline it still
    # starts from whatever is already in .venv. When nothing has changed this is
    # a sub-second no-op, which is what makes "git pull && uv sync" a viable
    # update path instead of reshipping a 4 GB binary.
    Set-Status 'installing torch and CUDA (2.5 GB) - this is the long one'
    & $uv sync --project $Project --frozen --no-dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    $py = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { throw "venv python missing at $py" }

    # ------------------------------------------------------------- gate ---
    # The single most important check in this file. Installing the CPU-only wheel
    # is the most common way this setup fails and it fails SILENTLY: everything
    # imports, everything runs, and the worker is slower than the NAS it was
    # supposed to relieve. Fail here, loudly, rather than at benchmark time.
    Set-Status 'checking CUDA'
    $probe = @'
import sys, torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
print("cuda version   ", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see the GPU. This is almost always the CPU-only "
             "wheel from PyPI rather than the cu126 wheel from download.pytorch.org. "
             "Check that uv.lock pins torch==2.6.0+cu126.")
if "+cu" not in torch.__version__:
    sys.exit("FATAL: %s is not a CUDA build." % torch.__version__)
print("device         ", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print("vram free/total %.0f / %.0f MiB" % (free/2**20, total/2**20))
'@
    $probe | & $py -
    if ($LASTEXITCODE -ne 0) { throw "CUDA check failed" }

    # ----------------------------------------------------------- models ---
    if (-not $SkipModels) {
        # ~3.0 GiB, into $Root\models via HF_HOME. Done here rather than on first
        # job so that the first job is not a twenty-minute download during
        # somebody's idle window.
        Set-Status 'downloading Chatterbox weights (3.0 GB)'
        $fetch = @'
import os
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print("HF_HOME =", os.environ.get("HF_HOME"))
ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("weights present")
'@
        $fetch | & $py -
        if ($LASTEXITCODE -ne 0) { throw "model download failed" }
    }

    Clear-Status
    Write-Host ""
    Write-Host "Runtime ready at $Root"
    Write-Host "Set JobCommand in worker.ini to:"
    Write-Host "  JobCommand = $py"
    Write-Host "  JobArguments = `"$(Join-Path $Project 'runner.py')`" --queue `"$(Join-Path (Split-Path -Parent $Root) 'queue')`""
}
catch {
    Set-Status ("FAILED: " + $_.Exception.Message)
    Write-Error $_
    exit 1
}
