$ErrorActionPreference='Continue'
Write-Output "=== A: nvidia-smi -l with --query-gpu (the combination to find) ==="
Write-Output "--- A1: --query-gpu ... -l 1  (loop seconds), killed after 3s ---"
$p = Start-Process -FilePath "nvidia-smi" -ArgumentList '--query-gpu=timestamp,utilization.gpu,clocks.mem,pstate --format=csv,noheader -l 1' -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\a1.txt" -RedirectStandardError "$env:TEMP\a1e.txt"
Start-Sleep -Seconds 4; if(!$p.HasExited){ $p.Kill() }
Get-Content "$env:TEMP\a1.txt" -EA SilentlyContinue | Select-Object -First 6
Write-Output ("stderr: " + ((Get-Content "$env:TEMP\a1e.txt" -EA SilentlyContinue) -join ' | '))
Write-Output "--- A2: -lms 200 ---"
$p = Start-Process -FilePath "nvidia-smi" -ArgumentList '--query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 200' -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\a2.txt" -RedirectStandardError "$env:TEMP\a2e.txt"
Start-Sleep -Seconds 3; if(!$p.HasExited){ $p.Kill() }
Get-Content "$env:TEMP\a2.txt" -EA SilentlyContinue | Select-Object -First 8
Write-Output ("stderr: " + ((Get-Content "$env:TEMP\a2e.txt" -EA SilentlyContinue) -join ' | '))
Write-Output "--- A3: -l 1 -c 3 (the reported failure) ---"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1 -c 3 2>&1 | Select-Object -First 5
Write-Output "=== B: cost of one nvidia-smi query invocation (ms) ==="
1..5 | ForEach-Object {
  $sw=[Diagnostics.Stopwatch]::StartNew()
  $null = nvidia-smi --query-gpu=utilization.gpu,clocks.mem,pstate,memory.used --format=csv,noheader,nounits
  $sw.Stop(); "  invocation {0}: {1} ms" -f $_, $sw.ElapsedMilliseconds
}
Write-Output "=== C: 100ms polling for 12s -- how often do values actually change? ==="
$rows=@()
for($i=0;$i -lt 120;$i++){
  $t=[Diagnostics.Stopwatch]::StartNew()
  $v = (nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.mem,clocks.sm,pstate,power.draw --format=csv,noheader,nounits)
  $rows += ("{0,5} {1}" -f [int]($i*100), $v)
  $t.Stop(); $s = 100 - $t.ElapsedMilliseconds; if($s -gt 0){ Start-Sleep -Milliseconds $s }
}
$rows | Select-Object -First 40
Write-Output "--- distinct utilisation.gpu values over 12s ---"
($rows | ForEach-Object { ($_ -split ',')[0].Trim() -replace '^\s*\d+\s+','' }) | Group-Object | ForEach-Object { "  val=$($_.Name) count=$($_.Count)" }
