# Install the agent for the logged-in user. No administrator, no service.
#
# THE ONE THING THIS FILE EXISTS TO GET RIGHT: the agent must run in the
# interactive desktop session, not session 0.
#
# Measured on spring (probe p3): a process started over SSH lands in session 0
# while the console user is in session 1. From session 0, GetForegroundWindow()
# returns 0 and GetLastInputInfo() reports session 0's own idle time, which was
# 620953 ms while the user was in fact at the machine. A Windows service is in
# the same position. An agent installed that way is blind to the person it exists
# to yield to, and the policy in Policy.cs refuses to run at all when it detects
# this - which is correct, and also means a service install would simply never
# work rather than working badly.
#
# So: the Run key, which starts the agent inside the user's own session at logon.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = 'Stop'
$src  = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'dist\ai-voice-worker.exe'
if (-not (Test-Path $src)) { throw "build it first: powershell -File build.ps1" }

$dest = "$env:LOCALAPPDATA\ai-voice-worker"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item $src (Join-Path $dest 'ai-voice-worker.exe') -Force

$ini = Join-Path $dest 'worker.ini'
$example = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'worker.ini.example'
if (-not (Test-Path $ini) -and (Test-Path $example)) { Copy-Item $example $ini }

Set-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' `
  -Name 'AiVoiceWorker' -Value ('"' + (Join-Path $dest 'ai-voice-worker.exe') + '"')

Write-Host "installed to $dest"
Write-Host "starts at next logon; to start it now:"
Write-Host "  & '$dest\ai-voice-worker.exe'"
Write-Host "to remove: Remove-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Run AiVoiceWorker; Remove-Item -Recurse '$dest'"
