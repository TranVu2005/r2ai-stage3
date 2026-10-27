# S40: C-max for the D = 400 pairs (out\runs\s40\d400\cmax_pairs.parquet), built exactly like the d300 C-max of free-v2:
# chunk choice by the S33 / S36 code of the worktree s35-part-jina (same as run_d300_cmax.ps1), scored by scripts\s37\run_cmax.ps1 of the
# main checkout (Qwen3-Reranker-0.6B bf16, 8,192 tokens / 16 rows per batch, parts of 2,000 rows), reduce to out\runs\s37\cmax_scores-d400.parquet.
# Only the pair list, the output folder and the id bases differ (d300_cmax.py of this worktree takes them as arguments).
# Checkpointed: prepare is skipped when pool\manifest.json exists, run_cmax.ps1 resumes from its finished parts, reduce never overwrites.
#   run_d400_cmax.ps1                  all stages; log out\runs\s37\d400_cmax.log, PID in out\runs\s37\cmax-d400\driver.pid
param([ValidateSet('all', 'prepare', 'score')][string]$Stage = 'all')
$repo = 'D:\GitHub\r2ai-stage3'; $wt = "$repo\.superpowers\worktrees\s35-part-jina"; $me = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = "$repo\.venv\Scripts\python.exe"; $run = "$repo\out\runs\s37"; $out = "$run\cmax-d400"
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:R2AI_OUT_DIR = "$repo/out"; $env:R2AI_DATA_DIR = "$repo/data"; $env:R2AI_STATE_DIR = "$repo/state"; $env:R2AI_LOG_DIR = "$repo/out/logs"
$env:HF_HOME = 'C:/Users/Admin/.cache/huggingface'; $env:HF_HUB_OFFLINE = '1'
$log = "$run\d400_cmax.log"
$cmaxArgs = @('--src', "$repo\out\runs\s40\d400\cmax_pairs.parquet", '--out', $out, '--dest', "$run\cmax_scores-d400.parquet",
              '--src-id-base', '51000000', '--row-id-base', '61000000')
function Note($m) { for ($i = 0; $i -lt 5; $i++) { try { "$(Get-Date -Format s) pid=$PID $m" | Add-Content -Encoding utf8 $log -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } } }
New-Item -ItemType Directory -Force $out | Out-Null
$PID | Set-Content "$out\driver.pid"
if ($Stage -in 'all', 'prepare') {
    if (Test-Path "$out\pool\manifest.json") { Note 'prepare skipped (pool\manifest.json exists)' } else {
        Set-Location $wt
        $env:PYTHONPATH = "$wt\src"
        Note 'prepare start'
        & $py "$me\scripts\s37\d300_cmax.py" prepare @cmaxArgs *>> "$run\d400_cmax.prepare.out"
        Note "prepare exit=$LASTEXITCODE"; if ($LASTEXITCODE -ne 0) { Note 'FAILED prepare'; exit $LASTEXITCODE }
    }
}
if ($Stage -in 'all', 'score') {
    Set-Location $repo
    $env:PYTHONPATH = ''
    Note 'score start'
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\s37\run_cmax.ps1" -Run $out
    Note "run_cmax exit=$LASTEXITCODE"
    $env:PYTHONPATH = "$wt\src"
    if (Test-Path "$run\cmax_scores-d400.parquet") { Note 'reduce skipped (output exists)' } else {
        & $py "$me\scripts\s37\d300_cmax.py" reduce @cmaxArgs *>> "$run\d400_cmax.reduce.out"
        Note "reduce exit=$LASTEXITCODE"; if ($LASTEXITCODE -ne 0) { Note 'FAILED reduce'; exit $LASTEXITCODE }
    }
    Note 'DONE'
}
