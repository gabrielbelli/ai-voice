$ErrorActionPreference='Continue'
Write-Output "=== 90s IDLE BASELINE @1Hz: util.gpu,util.mem,enc,dec,clocks.mem,pstate,power,mem.used ==="
$p = Start-Process -FilePath "nvidia-smi" -ArgumentList '--query-gpu=utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,clocks.mem,pstate,power.draw,memory.used --format=csv,noheader,nounits -l 1' -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\bl.txt"
Start-Sleep -Seconds 92
if(!$p.HasExited){ $p.Kill(); $p.WaitForExit() }
$d = Get-Content "$env:TEMP\bl.txt"
Write-Output ("samples: " + $d.Count)
$cols = @('util.gpu','util.mem','enc','dec','clk.mem','pstate','power','mem.used')
for($c=0;$c -lt 8;$c++){
  $vals = $d | ForEach-Object { ($_ -split ',')[$c].Trim() }
  if($c -eq 5){ Write-Output ("{0,-9} distinct: {1}" -f $cols[$c], (($vals | Group-Object | ForEach-Object {"$($_.Name)x$($_.Count)"}) -join ' ')) }
  else {
    $n = $vals | ForEach-Object { [double]$_ }
    $st = $n | Measure-Object -Min -Max -Average
    $srt = $n | Sort-Object
    $p95 = $srt[[int]([math]::Floor(0.95*($srt.Count-1)))]
    Write-Output ("{0,-9} min={1,-8:N2} max={2,-8:N2} mean={3,-8:N2} p95={4,-8:N2}" -f $cols[$c],$st.Minimum,$st.Maximum,$st.Average,$p95)
  }
}
Write-Output "--- raw util.gpu series ---"
Write-Output (($d | ForEach-Object { ($_ -split ',')[0].Trim() }) -join ' ')
Write-Output "--- raw power series ---"
Write-Output (($d | ForEach-Object { ($_ -split ',')[6].Trim() }) -join ' ')
Remove-Item "$env:TEMP\bl.txt","$env:TEMP\a1.txt","$env:TEMP\a1e.txt","$env:TEMP\a2.txt","$env:TEMP\a2e.txt" -EA SilentlyContinue
Write-Output "cleaned temp files"
