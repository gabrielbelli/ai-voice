# A stand-in for the GPU job, in PowerShell, so the yield path can be measured on
# a machine with no Python.
#
# WHY IT EXISTS. spring has no Python and installing one is out of scope for a
# read-only investigation, but the thing most worth measuring - how long the agent
# takes to get a child process off the GPU - does not need Python, or torch, or a
# GPU. It needs a child that behaves the two ways a real one can:
#
#   default     watches stdin, exits on YIELD or on EOF. This is runner.py's
#               cooperative path and it should complete in milliseconds.
#   -Stubborn   ignores stdin entirely and loops. This is a runner wedged inside
#               generate(), which cannot be interrupted, and it forces the agent
#               to fall through the grace period and close the job object. It is
#               the path that must be proven to actually work, because on the
#               real runtime it is the NORMAL path, not the exception.
#
# It prints an ISO timestamp on start and on exit. Diff those against worker.log
# to get the yield latency without instrumenting anything else.
#
#   powershell -ExecutionPolicy Bypass -File fake_job.ps1 [-Stubborn]

[CmdletBinding()]
param(
    [switch]$Stubborn,
    # A hard deadline regardless of mode. Nothing started on this machine may
    # outlive the command that started it, including by accident.
    [int]$MaxSeconds = 300
)

$ErrorActionPreference = 'Stop'
$start = Get-Date
Write-Host ("{0} fake job started, pid {1}, stubborn={2}" -f `
    $start.ToUniversalTime().ToString('o'), $PID, [bool]$Stubborn)

$deadline = $start.AddSeconds($MaxSeconds)

if ($Stubborn) {
    # Deliberately never reads stdin. The agent's YIELD line goes nowhere, the
    # grace period elapses, and the job object takes this process down. That is
    # the measurement: the kill path is the one the real runtime relies on.
    while ((Get-Date) -lt $deadline) {
        $null = 1..2000 | ForEach-Object { [Math]::Sqrt($_) }
    }
    Write-Host ("{0} fake job hit its own deadline" -f (Get-Date).ToUniversalTime().ToString('o'))
    exit 0
}

# Cooperative: block on stdin. ReadLine returns $null at EOF, which is the agent
# having gone away - the dead man's switch runner.py relies on.
while ($true) {
    if ((Get-Date) -ge $deadline) {
        Write-Host ("{0} fake job hit its own deadline" -f (Get-Date).ToUniversalTime().ToString('o'))
        exit 0
    }
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) {
        Write-Host ("{0} stdin closed; agent is gone" -f (Get-Date).ToUniversalTime().ToString('o'))
        exit 0
    }
    if ($line.Trim().ToUpperInvariant() -eq 'YIELD') {
        Write-Host ("{0} YIELD received; exiting" -f (Get-Date).ToUniversalTime().ToString('o'))
        exit 0
    }
}
