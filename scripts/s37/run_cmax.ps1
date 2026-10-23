# Supervisor for the S37 C-max run (Qwen3-Reranker-0.6B on the new 120ask chunk rows), adapted from scripts/s32/run_score.ps1.
# Restarts judge_score after a CUDA failure (the GPU is shared with the desktop; an OOM leaves the CUDA context unusable).
# Checkpoint = finished parts (2,000 rows each); a fresh process resumes from them.
# Writes: supervisor.log (every attempt, pairs done/total, parts/s ETA), score.pid (current scorer PID), score.stdout/stderr.attemptN.
# Stops on: complete, a non-finite score (FloatingPointError), or N attempts without a new part.
param(
    [string]$Run = 'D:\GitHub\r2ai-stage3\out\runs\s37\cmax',
    [string]$Python = 'D:\GitHub\r2ai-stage3\.venv\Scripts\python.exe',
    [string]$Snapshot = 'D:/GitHub/r2ai-stage3/data/hf_model_cache/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473',
    [int]$MaxBatchTokens = 8192,
    [int]$MaxBatchSize = 16,
    [int]$MaxFutile = 3
)
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$log = "$Run\supervisor.log"
function Note($m) { "$(Get-Date -Format s) supervisor_pid=$PID $m" | Add-Content -Encoding utf8 $log }
$judge = "$Run\pool\judge_score.py"
$parts = "$Run\gpu\qwen3-reranker-0.6b\parts"
$poolArgs = @('--pool', "$Run\pool", '--out', "$Run\gpu", '--model', 'qwen3-reranker-0.6b', '--model-path', $Snapshot, '--subset', 'all', '--part-pairs', '2000')
$stArgs = @('--pool', "$Run\pool", '--out', "$Run\gpu", '--model', 'qwen3-reranker-0.6b', '--subset', 'all', '--part-pairs', '2000')
$futile = 0; $attempt = 0; $t0 = Get-Date
Note "start max_batch_tokens=$MaxBatchTokens max_batch_size=$MaxBatchSize"
while ($true) {
    $before = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
    $status = & $Python $judge status @stArgs | Select-Object -Last 1 | ConvertFrom-Json
    $el = ((Get-Date) - $t0).TotalSeconds
    $eta = if ($status.pairs_done -gt 0) { [math]::Round($el * ($status.pairs_total - $status.pairs_done) / $status.pairs_done / 60, 1) } else { 'n/a' }
    Note "pairs=$($status.pairs_done)/$($status.pairs_total) parts=$($status.parts_done)/$($status.parts_total) eta_min=$eta"
    if ($status.complete) { Note 'complete'; break }
    $attempt++
    $err = "$Run\score.stderr.attempt$attempt"
    $p = Start-Process -FilePath $Python -ArgumentList (@($judge, 'score') + $poolArgs + @('--max-batch-tokens', $MaxBatchTokens, '--max-batch-size', $MaxBatchSize)) `
        -WorkingDirectory $Run -WindowStyle Hidden -RedirectStandardOutput "$Run\score.stdout.attempt$attempt" -RedirectStandardError $err -PassThru
    $p.Id | Out-File -Encoding ascii "$Run\score.pid"
    Note "attempt=$attempt scorer_pid=$($p.Id) parts_before=$before"
    $p.WaitForExit()
    $after = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
    Note "attempt=$attempt exit=$($p.ExitCode) parts_after=$after"
    if ((Test-Path $err) -and (Select-String -Path $err -Pattern 'FloatingPointError|Non-finite' -Quiet)) {
        Note 'STOP non-finite score: stopped at the last checkpoint'; exit 3
    }
    if ($after -le $before) { $futile++ } else { $futile = 0 }
    if ($futile -ge $MaxFutile) { Note "STOP $futile attempts without a new part"; exit 4 }
    Start-Sleep -Seconds 20
}
Remove-Item "$Run\score.pid" -ErrorAction SilentlyContinue
& $Python $judge collect @stArgs | Out-File -Encoding utf8 "$Run\collect.stdout"
Note 'collect done'
