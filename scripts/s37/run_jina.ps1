# Supervisor for the S37 Jina Reranker v3.5 run (S35 code, layout b10s0), adapted from scripts/s35/run_jina.sh.
# Scores out\runs\s37\<Run>\pool.parquet one query per part (checkpoint = parts-b10s0\part-NNNN.parquet, resumable),
# restarts after a CUDA/driver crash, then collects scores-all.parquet. The S35 outputs are only read, never written.
# Writes: supervisor.log (attempts, parts done, queries/min and ETA), score.pid (current scorer PID), score.stdout/stderr.attemptN.
param(
    [string]$Run = 'D:\GitHub\r2ai-stage3\out\runs\s37\jina',
    [string]$Repo = 'D:\GitHub\r2ai-stage3\.superpowers\worktrees\s35-part-jina',
    [string]$Python = 'D:\GitHub\r2ai-stage3\.venv\Scripts\python.exe',
    [string]$Layout = 'b10s0',
    [int]$MaxAttempts = 30
)
$env:PYTHONPATH = "$Repo\src"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:R2AI_OUT_DIR = 'D:/GitHub/r2ai-stage3/out'; $env:R2AI_DATA_DIR = 'D:/GitHub/r2ai-stage3/data'; $env:R2AI_STATE_DIR = 'D:/GitHub/r2ai-stage3/state'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'   # fewer fragmentation OOMs (a no-op with a warning on Windows builds without support)
$env:R2AI_LOG_DIR ='D:/GitHub/r2ai-stage3/out/logs'; $env:HF_HOME = 'C:/Users/Admin/.cache/huggingface'; $env:HF_HUB_OFFLINE = '1'
$log = "$Run\supervisor.log"
function Note($m) { "$(Get-Date -Format s) supervisor_pid=$PID $m" | Add-Content -Encoding utf8 $log }
$parts = "$Run\parts-$Layout"
$total = (& $Python -c "import pandas as pd,sys; print(pd.read_parquet(sys.argv[1], columns=['query_id']).query_id.nunique())" "$Run\pool.parquet" | Select-Object -Last 1)
$t0 = Get-Date
Note "start run=$Run layout=$Layout queries=$total"
for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $done0 = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
    $p = Start-Process -FilePath $Python -ArgumentList @('-u', '-m', 'r2ai.s35.jina', 'score', '--out', $Run, '--layout', $Layout, '--part-queries', '1') `
        -WorkingDirectory $Repo -WindowStyle Hidden -RedirectStandardOutput "$Run\score.stdout.attempt$attempt" -RedirectStandardError "$Run\score.stderr.attempt$attempt" -PassThru
    $null = $p.Handle
    $p.Id | Out-File -Encoding ascii "$Run\score.pid"
    Note "attempt=$attempt scorer_pid=$($p.Id) parts_before=$done0 of $total"
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 60
        $d = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
        $min = ((Get-Date) - $t0).TotalMinutes
        $rate = if ($d -gt $done0) { [math]::Round(($d - $done0) / $min, 2) } else { 0 }
        $eta = if ($rate -gt 0) { [math]::Round(($total - $d) / $rate / 60, 1) } else { 'n/a' }
        Note "progress parts=$d/$total queries_per_min=$rate eta_hours=$eta"
    }
    $p.WaitForExit()
    $done1 = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
    Note "attempt=$attempt exit=$($p.ExitCode) parts_after=$done1 of $total"
    & $Python -u -m r2ai.s35.jina collect --out $Run --layout $Layout *> "$Run\collect.stdout"
    if ($LASTEXITCODE -eq 0) { Note 'collected scores-all.parquet'; Remove-Item "$Run\score.pid" -ErrorAction SilentlyContinue; exit 0 }
    if ((Select-String -Path "$Run\score.stderr.attempt$attempt" -Pattern 'Non-finite' -Quiet -ErrorAction SilentlyContinue)) { Note 'STOP non-finite score'; exit 3 }
    Start-Sleep -Seconds 30
}
Note "STOP after $MaxAttempts attempts"
exit 4
