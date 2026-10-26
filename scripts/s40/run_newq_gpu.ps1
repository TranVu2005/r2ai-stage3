# S40: local GPU scoring of a new query set (r2ai.newq), one job on the GPU at a time: C-max (0.6B) then Jina v3.5 (b10s0).
# Each stage uses the S37 supervisor (restarts the scorer after a CUDA crash, resumes from finished parts), then imports its scores.
# A finished stage (its import done) is skipped, so this script can simply be started again.
#   powershell -File scripts\s40\run_newq_gpu.ps1 -Run D:\GitHub\r2ai-stage3\out\runs\s40\rh100 [-Stages cmax,jina]
# Writes <Run>\gpu.log (stage start / end with seconds, exit codes) and <Run>\gpu.pid.
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$Stages = 'cmax,jina',
    [string]$Python = 'D:\GitHub\r2ai-stage3\.venv\Scripts\python.exe'
)
$wt = (Resolve-Path "$PSScriptRoot\..\..").Path
$env:PYTHONPATH = "$wt\src"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$log = "$Run\gpu.log"
$PID | Out-File -Encoding ascii "$Run\gpu.pid"
function Note($m) { for ($i = 0; $i -lt 5; $i++) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } } }
foreach ($stage in $Stages.Split(',')) {
    $done = if ($stage -eq 'cmax') { "$Run\cmax.parquet" } else { "$Run\jina.parquet" }
    if (Test-Path $done) { Note "$stage already imported, skipped"; continue }
    $t0 = Get-Date
    Note "$stage start"
    if ($stage -eq 'cmax') {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$wt\scripts\s37\run_cmax.ps1" -Run "$Run\cmax"
    } else {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$wt\scripts\s37\run_jina.ps1" -Run "$Run\jina" -Repo $wt
    }
    $rc = $LASTEXITCODE
    Note "$stage scorer exit=$rc seconds=$([int]((Get-Date) - $t0).TotalSeconds)"
    if ($rc -ne 0) { Note "STOP $stage failed"; exit $rc }
    & $Python -m r2ai.newq import-scores --run $Run --source $stage *>> "$Run\gpu-import.out"
    if ($LASTEXITCODE -ne 0) { Note "STOP import $stage failed"; exit 5 }
    Note "$stage imported seconds=$([int]((Get-Date) - $t0).TotalSeconds)"
}
Note 'ALL DONE'
Remove-Item "$Run\gpu.pid" -ErrorAction SilentlyContinue
