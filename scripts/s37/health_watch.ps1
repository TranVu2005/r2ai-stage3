# Health watcher for the S37 detached jobs. Every 2 min it looks at the logs and the processes and writes ONE line per new problem to
# out\runs\s37\ALERT.log ("ALERT <time> <job>: <what>"); health.beat holds the time of its last loop (a stale beat = the watcher died).
# It only observes: it never starts, stops or edits a job. Problems it reports:
#   - a log line "exit=<nonzero>" / "STOP" / "failed" from the d300 C-max, pipeline, en-embed, Jina watchdog or C-max supervisor logs
#   - Jina (jina-final2) not complete, a part count that did not grow for 10 min, or no scorer/supervisor process
#   - the d300 C-max score waiter missing while its prerequisites are still pending
$run = 'D:\GitHub\r2ai-stage3\out\runs\s37'; $alert = "$run\ALERT.log"; $beat = "$run\health.beat"
$seen = @{}
$startIso = Get-Date -Format s     # lines older than the watcher are history (earlier, already handled failures)
$lastParts = -1; $lastGrow = Get-Date
function Raise($key, $msg) {
    if ($seen.ContainsKey($key)) { return }
    $seen[$key] = 1
    for ($i = 0; $i -lt 5; $i++) { try { "ALERT $(Get-Date -Format s) ${msg}" | Add-Content -Encoding utf8 $alert -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } }
}
function Scan($name, $path, $pattern) {
    if (-not (Test-Path $path)) { return }
    try { $lines = Get-Content $path -ErrorAction Stop } catch { return }
    foreach ($l in $lines) { if ($l -match $pattern) {
        if ($l -match '^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)' -and $Matches[1] -lt $startIso) { continue }
        Raise "$name|$l" "${name}: $($l.Substring(0, [Math]::Min(200, $l.Length)))" } }
}
Raise 'start' 'health_watch started'
while ($true) {
    Get-Date -Format s | Set-Content $beat
    Scan 'd300_cmax' "$run\d300_cmax.log" 'exit=[1-9]'
    Scan 'pipeline_lo7' "$run\pipeline_lo7.log" 'STOP'
    Scan 'en_embed' "$run\en_embed.log" 'STOP'
    Scan 'jina_watchdog' "$run\jina-final2\watchdog.log" 'STOP|error'
    Scan 'cmax_d300_supervisor' "$run\cmax-d300\supervisor.log" 'STOP|exit=[1-9]'
    $done = Test-Path "$run\jina-final2\scores-all.parquet"
    if (-not $done) {
        $parts = @(Get-ChildItem "$run\jina-final2\parts-b10s0\*.parquet" -ErrorAction SilentlyContinue).Count
        if ($parts -gt $lastParts) { $lastParts = $parts; $lastGrow = Get-Date }
        $paused = (Test-Path "$run\en_embed.log") -and -not (Select-String -Path "$run\en_embed.log" -Pattern 'embed done' -Quiet)
        if (-not $paused -and ((Get-Date) - $lastGrow).TotalMinutes -ge 10) { Raise "jina-stall-$parts" "jina: no new part for $([int]((Get-Date) - $lastGrow).TotalMinutes) min (parts=$parts)" }
        $procs = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'r2ai\.s35\.jina' }).Count
        if (-not $paused -and $procs -eq 0) { Raise "jina-noproc-$parts" "jina: no scorer process (parts=$parts)" }
    }
    $manifest = Test-Path "$run\cmax-d300\pool\manifest.json"
    $waiter = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_d300_cmax\.ps1|scores-all\.parquet' -and $_.Name -match 'powershell' }).Count
    $scoreStarted = (Test-Path "$run\cmax-d300\supervisor.log")
    if (-not $scoreStarted -and $waiter -eq 0) { Raise 'd300-no-waiter' 'cmax_d300: no prepare/score job and the C-max has not started' }
    Start-Sleep -Seconds 120
}
