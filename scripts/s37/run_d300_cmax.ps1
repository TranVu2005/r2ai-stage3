# C-max for the S36 d300 pairs, in two detached stages (CPU prepare, then the usual run_cmax.ps1 scorer on the GPU, then reduce).
#   run_d300_cmax.ps1 -Stage prepare   CPU only (S33/S36 code of the worktree s35-part-jina), writes out\runs\s37\cmax-d300\pool
#   run_d300_cmax.ps1 -Stage score     run_cmax.ps1 -Run cmax-d300 (GPU), then reduce, then the hand-off manifest
param([ValidateSet('prepare', 'score')][string]$Stage = 'prepare', [int]$Limit = 0)
$repo = 'D:\GitHub\r2ai-stage3'; $wt = "$repo\.superpowers\worktrees\s35-part-jina"; $py = "$repo\.venv\Scripts\python.exe"; $run = "$repo\out\runs\s37"
$env:PYTHONPATH = "$wt\src"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:R2AI_OUT_DIR = "$repo/out"; $env:R2AI_DATA_DIR = "$repo/data"; $env:R2AI_STATE_DIR = "$repo/state"; $env:R2AI_LOG_DIR = "$repo/out/logs"
$env:HF_HOME = 'C:/Users/Admin/.cache/huggingface'; $env:HF_HUB_OFFLINE = '1'
$log = "$run\d300_cmax.log"
function Note($m) { for ($i = 0; $i -lt 5; $i++) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } } }
if ($Stage -eq 'prepare') {
    Set-Location $wt
    Note "prepare start limit=$Limit"
    $a = @("$repo\scripts\s37\d300_cmax.py", 'prepare'); if ($Limit -gt 0) { $a += @('--limit', $Limit) }
    & $py @a *>> "$run\d300_cmax.prepare.out"
    Note "prepare exit=$LASTEXITCODE"; exit $LASTEXITCODE
}
Set-Location $repo
$env:PYTHONPATH = ''                      # the S37 modules live in the main checkout
Note 'score start'
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\s37\run_cmax.ps1" -Run "$run\cmax-d300"
Note "run_cmax exit=$LASTEXITCODE"
$env:PYTHONPATH = "$wt\src"
& $py "$repo\scripts\s37\d300_cmax.py" reduce *>> "$run\d300_cmax.reduce.out"
Note "reduce exit=$LASTEXITCODE"
$env:PYTHONPATH = ''
& $py -m r2ai.s37.handoff manifest *>> "$run\d300_cmax.reduce.out"
Note 'DONE'
