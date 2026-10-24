# Watchdog for the S37 Jina run: keeps scripts\s37\run_jina.ps1 alive until scores-all.parquet exists.
# Every 2 min it checks that the supervisor+scorer processes exist AND the part count grew; if not it kills what is left and
# restarts run_jina.ps1 (parts are checkpoints, so a restart resumes). At most $MaxRestarts restarts, each one logged to watchdog.log.
# The block size stays b10 (it is part of the layout b10s0: changing it changes the scores), so repeated OOM is not "fixed" by smaller
# blocks: the watchdog waits longer between restarts and finally stops and says so in the log.
param(
    [string]$Run = 'D:\GitHub\r2ai-stage3\out\runs\s37\jina-final',
    [string]$Script = 'D:\GitHub\r2ai-stage3\scripts\s37\run_jina.ps1',
    [int]$MaxRestarts = 5,
    [int]$CheckSeconds = 120
)
$log = "$Run\watchdog.log"
function Note($m) { "$(Get-Date -Format s) watchdog_pid=$PID $m" | Add-Content -Encoding utf8 $log }
function Procs() { @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_jina\.ps1' -or ($_.CommandLine -match 'r2ai\.s35\.jina' -and $_.CommandLine -match [regex]::Escape((Split-Path $Run -Leaf))) }) }
function Parts() { @(Get-ChildItem "$Run\parts-b10s0\*.parquet" -ErrorAction SilentlyContinue).Count }
function StartRun() {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Script -Run $Run"; CurrentDirectory = 'D:\GitHub\r2ai-stage3' }
    Note "started run_jina.ps1 pid=$($r.ProcessId) rv=$($r.ReturnValue) parts=$(Parts)"
}
$restarts = 0
$last = Parts
Note "start run=$Run parts=$last max_restarts=$MaxRestarts check_s=$CheckSeconds"
if (-not (Procs).Count -and -not (Test-Path "$Run\scores-all.parquet")) { StartRun }
$stale = 0
while ($true) {
    Start-Sleep -Seconds $CheckSeconds
    if (Test-Path "$Run\scores-all.parquet") { Note "DONE scores-all.parquet exists parts=$(Parts) restarts=$restarts"; exit 0 }
    $now = Parts
    $alive = (Procs).Count -gt 0
    if ($now -gt $last) { $stale = 0 } else { $stale++ }
    $last = $now
    if ($alive -and $stale -lt 3) { continue }          # 3 checks (6 min) without a new part = hung
    $why = if (-not $alive) { 'process gone' } else { "no new part for $($stale * $CheckSeconds) s" }
    if ($restarts -ge $MaxRestarts) { Note "STOP after $restarts restarts ($why); parts=$now. Not changing the block size (layout b10s0). Tell Vu."; exit 4 }
    $restarts++
    Note "restart $restarts/$MaxRestarts reason=$why parts=$now"
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_jina\.ps1' -or ($_.CommandLine -match 'r2ai\.s35\.jina') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds (30 * $restarts)               # longer pause after each repeated failure
    StartRun
    $stale = 0; $last = Parts
}
