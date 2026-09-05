$ErrorActionPreference='Continue'
Write-Output "=== STEAM REGISTRY: RunningAppID + per-app Running/Updating ==="
$k='HKCU:\Software\Valve\Steam'
if(Test-Path $k){
  $s = Get-ItemProperty $k
  Write-Output ("RunningAppID = " + $s.RunningAppID)
  Write-Output ("SteamExe = " + $s.SteamExe)
  Write-Output ("pid = " + $s.pid)
} else { Write-Output "no HKCU Steam key" }
Write-Output "--- HKCU\Software\Valve\Steam\Apps (running/updating flags) ---"
$apps='HKCU:\Software\Valve\Steam\Apps'
if(Test-Path $apps){
  $n = (Get-ChildItem $apps -EA SilentlyContinue)
  Write-Output ("app subkeys: " + $n.Count)
  $n | Select-Object -First 8 | ForEach-Object {
    $pp = Get-ItemProperty $_.PSPath -EA SilentlyContinue
    "  appid=$($_.PSChildName) Name='$($pp.Name)' Running=$($pp.Running) Updating=$($pp.Updating) Installed=$($pp.Installed)"
  }
  Write-Output "--- any with Running=1 ---"
  $n | ForEach-Object { $pp = Get-ItemProperty $_.PSPath -EA SilentlyContinue; if($pp.Running -eq 1){ "  RUNNING appid=$($_.PSChildName) $($pp.Name)" } }
} else { Write-Output "no Apps key" }
Write-Output "=== SCHEMA OF ActiveProcess ==="
$ap='HKCU:\Software\Valve\Steam\ActiveProcess'
if(Test-Path $ap){ Get-ItemProperty $ap | Format-List | Out-String } else { Write-Output "none" }
Write-Output "=== VANGUARD ==="
Get-Service -Name vgc,vgk -EA SilentlyContinue | Select-Object Name,Status,StartType | Format-Table -Auto | Out-String
Get-Process -Name vgtray,vgc -EA SilentlyContinue | Select-Object Id,ProcessName | Format-Table -Auto | Out-String
Write-Output "=== OTHER LAUNCHERS PRESENT ==="
foreach($p in @('C:\Program Files (x86)\Steam\steam.exe','C:\Program Files\Epic Games','C:\Riot Games','C:\Program Files\Battle.net','C:\Program Files (x86)\Battle.net')){
  Write-Output ("  {0} : {1}" -f $p, (Test-Path $p))
}
Write-Output "=== GAME BAR / FULLSCREEN-ISH HINTS (registry, read-only) ==="
Get-ItemProperty 'HKCU:\System\GameConfigStore' -EA SilentlyContinue | Select-Object GameDVR_* | Format-List | Out-String
Write-Output "=== GPU-SCHEDULING / HAGS ==="
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Name HwSchMode -EA SilentlyContinue | Select-Object HwSchMode | Format-List | Out-String
Write-Output "=== POWER SCHEME ==="
powercfg /getactivescheme
