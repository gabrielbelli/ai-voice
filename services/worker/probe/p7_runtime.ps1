$ErrorActionPreference='Continue'
Write-Output "=== .NET FRAMEWORK (in-box) ==="
$r = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full' -EA SilentlyContinue
Write-Output ("NDP v4 Full Release=" + $r.Release + "  Version=" + $r.Version)
Write-Output "=== csc.exe / MSBuild ==="
foreach($p in @('C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe','C:\Windows\Microsoft.NET\Framework64\v4.0.30319\System.Windows.Forms.dll')){
  Write-Output ("  {0} : {1}" -f $p,(Test-Path $p))
}
Write-Output "=== dotnet SDK/runtimes ==="
$dn = Get-Command dotnet -EA SilentlyContinue
if($dn){ Write-Output ("dotnet: " + $dn.Source); dotnet --list-runtimes 2>&1 | Select-Object -First 12 } else { Write-Output "dotnet NOT on PATH" }
Write-Output "=== python / py (confirming absence) ==="
foreach($c in @('python','py','python3')){
  $g = Get-Command $c -EA SilentlyContinue
  Write-Output ("  {0}: {1}" -f $c, $(if($g){$g.Source}else{'ABSENT'}))
}
Write-Output "=== winget ==="
$w = Get-Command winget -EA SilentlyContinue; Write-Output ("winget: " + $(if($w){$w.Source}else{'ABSENT'}))
Write-Output "=== DISK FREE ==="
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object { "{0} free={1:N1} GB of {2:N1} GB" -f $_.DeviceID, ($_.FreeSpace/1GB), ($_.Size/1GB) }
Write-Output "=== CPU / RAM ==="
$cs = Get-CimInstance Win32_ComputerSystem
"{0} cores logical, {1:N1} GB RAM" -f (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors, ($cs.TotalPhysicalMemory/1GB)
(Get-CimInstance Win32_Processor).Name
Write-Output "=== Get-Counter COST (ms) for the two counter sets ==="
1..3 | ForEach-Object {
  $sw=[Diagnostics.Stopwatch]::StartNew()
  $null=(Get-Counter '\GPU Engine(*)\Utilization Percentage' -EA SilentlyContinue)
  $sw.Stop(); "  GPU Engine sample: {0} ms" -f $sw.ElapsedMilliseconds
}
Write-Output "=== EXECUTION POLICY / DEFENDER (informational only) ==="
Get-ExecutionPolicy -List | Format-Table -Auto | Out-String
