$ErrorActionPreference = 'Continue'
Write-Output "=== VERSION ==="
nvidia-smi --version
Write-Output "=== QUERY-GPU BASIC ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,fan.speed --format=csv,noheader,nounits
Write-Output "=== QUERY-GPU ENCODER/DECODER ==="
nvidia-smi --query-gpu=utilization.encoder,utilization.decoder --format=csv,noheader
Write-Output "=== ENCODER SESSIONS ==="
nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps,encoder.stats.averageLatency --format=csv,noheader
Write-Output "=== DISPLAY/PERSISTENCE/COMPUTE MODE ==="
nvidia-smi --query-gpu=display_active,display_mode,persistence_mode,compute_mode,accounting.mode,driver_model.current,driver_model.pending --format=csv,noheader
Write-Output "=== CLOCK THROTTLE REASONS ==="
nvidia-smi --query-gpu=clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap --format=csv,noheader
Write-Output "=== COMPUTE APPS ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
Write-Output "=== GRAPHICS APPS ==="
nvidia-smi --query-accounted-apps=pid,gpu_utilization,mem_utilization,max_memory_usage --format=csv
Write-Output "=== PMON SUPPORT ==="
nvidia-smi pmon -c 1 2>&1 | Select-Object -First 12
Write-Output "=== DMON 1 SAMPLE ==="
nvidia-smi dmon -c 2 2>&1 | Select-Object -First 12
