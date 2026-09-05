$ErrorActionPreference='Continue'
Write-Output "=== GPU ENGINE COUNTER SET EXISTS? ==="
try {
  $s = Get-Counter -ListSet "GPU Engine" -ErrorAction Stop
  Write-Output ("counterset OK, paths: " + $s.Paths.Count)
} catch { Write-Output ("FAIL: " + $_.Exception.Message) }
Write-Output "=== TOP GPU ENGINE UTILISATION (nonzero) ==="
try {
  $c = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples |
       Where-Object { $_.CookedValue -gt 0.05 } | Sort-Object CookedValue -Descending
  Write-Output ("nonzero instances: " + @($c).Count)
  $c | Select-Object -First 15 | ForEach-Object { "{0,8:N2}  {1}" -f $_.CookedValue, $_.InstanceName }
} catch { Write-Output ("FAIL: " + $_.Exception.Message) }
Write-Output "=== SUM BY ENGTYPE ==="
try {
  (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples |
    Group-Object { ($_.InstanceName -split 'engtype_')[-1] } |
    ForEach-Object { "{0,-16} sum={1,8:N2}" -f $_.Name, (($_.Group | Measure-Object CookedValue -Sum).Sum) }
} catch { Write-Output ("FAIL: " + $_.Exception.Message) }
Write-Output "=== GPU PROCESS MEMORY (dedicated, top 12) ==="
try {
  (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
    Where-Object { $_.CookedValue -gt 1MB } | Sort-Object CookedValue -Descending |
    Select-Object -First 12 | ForEach-Object {
      $pidn = ($_.InstanceName -split '_')[1]
      $nm = (Get-Process -Id $pidn -ErrorAction SilentlyContinue).ProcessName
      "{0,9:N1} MiB  pid={1,-7} {2}" -f ($_.CookedValue/1MB), $pidn, $nm
    }
} catch { Write-Output ("FAIL: " + $_.Exception.Message) }
Write-Output "=== GPU LOCAL ADAPTER MEMORY ==="
try {
  (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
    Where-Object { $_.CookedValue -gt 0 } | ForEach-Object { "{0,9:N1} MiB  {1}" -f ($_.CookedValue/1MB), $_.InstanceName }
} catch { Write-Output ("FAIL: " + $_.Exception.Message) }
