# The "!!!" hunt: is the PC out of commit or RAM while the engine starts and answers its first request?
# One CSV line per sample: the system's committed bytes against the commit limit, the RAM still available, and the
# GPU's dedicated / shared memory in use (all processes).
#   powershell -File tuning\sys_mem_sampler.ps1 -Out tuning\results\x\sysmem.csv
param([string]$Out = "sysmem.csv", [int]$IntervalMs = 500)
"time,commit_gib,limit_gib,avail_gib,gpu_dedicated_gib,gpu_shared_gib" | Out-File $Out -Encoding ascii
$paths = "\Memory\Committed Bytes", "\Memory\Commit Limit", "\Memory\Available Bytes",
         "\GPU Adapter Memory(*)\Dedicated Usage", "\GPU Adapter Memory(*)\Shared Usage"
while ($true) {
    $s = (Get-Counter -Counter $paths -ErrorAction SilentlyContinue).CounterSamples
    if ($s) {
        $c = ($s | Where-Object { $_.Path -like '*committed bytes' }).CookedValue
        $l = ($s | Where-Object { $_.Path -like '*commit limit' }).CookedValue
        $a = ($s | Where-Object { $_.Path -like '*available bytes' }).CookedValue
        $d = ($s | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Maximum).Maximum
        $h = ($s | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum
        "{0},{1:F2},{2:F2},{3:F2},{4:F2},{5:F2}" -f (Get-Date -Format 'HH:mm:ss.fff'), ($c / 1GB), ($l / 1GB), ($a / 1GB), ($d / 1GB), ($h / 1GB) |
            Out-File $Out -Append -Encoding ascii
    }
    Start-Sleep -Milliseconds $IntervalMs
}
