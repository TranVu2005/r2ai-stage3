# Supervisor for the S32 chunk scoring run: restarts judge_score after a CUDA failure.
# The GPU is shared with the desktop; a "CUDA error: out of memory" leaves the CUDA context unusable,
# so the process exits and the finished parts (2,000 pairs each) are resumed by a fresh process.
# Stops on: complete, a non-finite score (FloatingPointError), or N attempts without a new part.
param(
    [string]$Run = 'D:\GitHub\r2ai-stage3\out\runs\s32-chunk-judge',
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Python = 'D:\GitHub\r2ai-stage3\.venv\Scripts\python.exe',
    [string]$Snapshot = 'D:/GitHub/r2ai-stage3/data/hf_model_cache/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473',
    [int]$MaxBatchTokens = 8192,
    [int]$MaxBatchSize = 16,
    [int]$MaxFutile = 3
)
$env:PYTHONPATH = "$Repo\src"; $env:PYTHONUNBUFFERED = '1'
$log = "$Run\supervisor.log"
function Note($m) { "$(Get-Date -Format s) supervisor_pid=$PID $m" | Add-Content -Encoding utf8 $log }
$parts = "$Run\gpu\qwen3-reranker-0.6b\parts"
$common = @('-m', 'r2ai.s31.judge_score')
$poolArgs = @('--pool', "$Run\pool", '--out', "$Run\gpu", '--model', 'qwen3-reranker-0.6b', '--model-path', $Snapshot, '--subset', 'all', '--part-pairs', '2000')
$futile = 0; $attempt = 0
Note "start max_batch_tokens=$MaxBatchTokens max_batch_size=$MaxBatchSize"
while ($true) {
    $before = @(Get-ChildItem "$parts\*.parquet" -ErrorAction SilentlyContinue).Count
    $status = & $Python @common status @poolArgs | Select-Object -Last 1 | ConvertFrom-Json
    if ($status.complete) { Note "complete parts=$($status.parts_done)/$($status.parts_total)"; break }
    $attempt++
    $err = "$Run\score.stderr.attempt$attempt"
    $p = Start-Process -FilePath $Python -ArgumentList ($common + @('score') + $poolArgs + @('--max-batch-tokens', $MaxBatchTokens, '--max-batch-size', $MaxBatchSize)) `
        -WorkingDirectory $Repo -WindowStyle Hidden -RedirectStandardOutput "$Run\score.stdout.attempt$attempt" -RedirectStandardError $err -PassThru
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
