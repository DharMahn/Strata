# The "!!!" hunt: does Windows move the engine's GPU memory into system RAM around a failing request?
# Samples the engine process's GPU memory (performance counters "GPU Process Memory": Local = in the adapter's own
# memory, Non Local = in system RAM) for every adapter it uses - the RX 6950 XT and the CPU's integrated GPU, which
# only holds the arena's registration - one CSV line per adapter per sample.
#   powershell -File tuning\gpu_mem_sampler.ps1 -Out tuning\results\x\gpumem.csv
param([string]$Out = "gpumem.csv", [int]$IntervalMs = 300)
"time,pid,luid,local_mib,nonlocal_mib" | Out-File $Out -Encoding ascii
while ($true) {
    $p = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'strata*' -and $_.ProcessName -ne 'strata-vision' } | Select-Object -First 1
    if ($p) {
        $samples = (Get-Counter -Counter "\GPU Process Memory(pid_$($p.Id)_*)\Local Usage", "\GPU Process Memory(pid_$($p.Id)_*)\Non Local Usage" -ErrorAction SilentlyContinue).CounterSamples
        if ($samples) {
            $t = Get-Date -Format 'HH:mm:ss.fff'
            $byLuid = @{}
            foreach ($s in $samples) {
                $luid = (($s.InstanceName -split '_luid_')[1] -split '_phys')[0]
                if (-not $byLuid.ContainsKey($luid)) { $byLuid[$luid] = @{ local = 0.0; nonlocal = 0.0 } }
                if ($s.Path -like '*\non local usage') { $byLuid[$luid].nonlocal += $s.CookedValue } else { $byLuid[$luid].local += $s.CookedValue }
            }
            foreach ($k in $byLuid.Keys) {
                "{0},{1},{2},{3:F0},{4:F0}" -f $t, $p.Id, $k, ($byLuid[$k].local / 1MB), ($byLuid[$k].nonlocal / 1MB) | Out-File $Out -Append -Encoding ascii
            }
        }
    }
    Start-Sleep -Milliseconds $IntervalMs
}
