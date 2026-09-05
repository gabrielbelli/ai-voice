# Build the ai-voice GPU worker.
#
# THIS IS THE WHOLE BUILD. There is no SDK to install, no NuGet restore, no
# network access and no build host: csc.exe ships inside Windows itself, and was
# measured present on the target machine (spring) at
#   C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
# alongside .NET Framework 4.8.09221 (release 533509). Every assembly referenced
# below is in the same directory. That is the argument for this packaging, made
# executable: the "runtime dependency" the user was worried about does not exist,
# because the runtime is part of the operating system.
#
# The output is ONE file. Copy ai-voice-worker.exe anywhere and run it.
#
#   powershell -ExecutionPolicy Bypass -File build.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$csc  = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { throw "csc.exe not found at $csc - is this Windows?" }

$out = Join-Path $root 'dist\ai-voice-worker.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'dist') | Out-Null

$refs = @(
  'System.dll','System.Core.dll','System.Drawing.dll','System.Windows.Forms.dll',
  'System.ServiceProcess.dll','System.Configuration.Install.dll'
) | ForEach-Object { "/r:$_" }

$src = Get-ChildItem (Join-Path $root 'src\*.cs') | ForEach-Object { $_.FullName }

# /target:winexe, not /target:exe. WHY: the tray build must not flash a console
# window at every login. Program.BorrowParentConsole() reattaches stdout for the
# --once/--watch/--calibrate modes, so the command line still works.
& $csc /nologo /target:winexe /platform:x64 /optimize+ /warn:4 `
       "/out:$out" $refs $src
if ($LASTEXITCODE -ne 0) { throw "compile failed with $LASTEXITCODE" }

Write-Output ("built {0} ({1:N0} bytes)" -f $out, (Get-Item $out).Length)
