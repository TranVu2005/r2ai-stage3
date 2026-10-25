# Detached pipeline for the last crawl batches (b0008, b0009), so the GPU never sits idle between steps:
#   wait for the running Jina (jina-final) -> embed (cuda, --max-util 70, Vu approved) -> C-max per batch -> pairs_new_final_v2
#   -> Jina top-up pool (only queries whose pool changed) -> Jina top-up run under the watchdog -> hand-off manifest.
# No 4B export here (S35 packs these pairs together with the d300 pack). Log: out\runs\s37\pipeline_lo7.log. Stops at the first failing step.
param([string[]]$Batches = @('b0008', 'b0009'))
$repo = 'D:\GitHub\r2ai-stage3'; $py = "$repo\.venv\Scripts\python.exe"; $run = "$repo\out\runs\s37"; $log = "$run\pipeline_lo7.log"
Set-Location $repo
function Note($m) {
    for ($i = 0; $i -lt 5; $i++) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } }
}
function Step($name, [scriptblock]$cmd) {
    Note "BEGIN $name"
    $out = & $cmd 2>&1 | Out-String
    $rc = $LASTEXITCODE
    Add-Content -Encoding utf8 "$run\pipeline_lo7.$($name -replace '\W','_').out" $out -ErrorAction SilentlyContinue
    if ($rc -ne 0 -and $null -ne $rc) { Note "STOP $name rc=$rc"; exit 1 }
    Note "END $name"
}
Note "start batches=$($Batches -join ',')"
while (-not (Test-Path "$run\jina-final\scores-all.parquet")) { Start-Sleep -Seconds 60 }
Note 'jina-final complete'
# embed: the GPU guard may refuse while a desktop app is busy, so retry until every segment is embedded
for ($try = 1; $try -le 12; $try++) {
    Note "embed try $try"
    (& $py -m r2ai.s37.ingest embed --device cuda --max-util 70 2>&1 | Out-String) | Add-Content -Encoding utf8 "$run\pipeline_lo7.embed.out" -ErrorAction SilentlyContinue   # a refusal by the GPU guard is not fatal: retried below
    $st = & $py -m r2ai.s37.ingest status 2>$null | Out-String
    if (($Batches | ForEach-Object { $b = $_; [regex]::IsMatch($st, "$b-120ask\.com"": \{\s*""state"": ""embedded""") }) -notcontains $false) { break }
    Note "embed incomplete after try $try, waiting 5 min"; Start-Sleep -Seconds 300
}
foreach ($b in $Batches) {
    Step "cmax-export-$b" { & $py -m r2ai.s37.export cmax --tag $b --batches $b }
    Step "cmax-score-$b" { & powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\s37\run_cmax.ps1" -Run "$run\cmax-$b" }
    Step "cmax-reduce-$b" { & $py -m r2ai.s37.export reduce --tag $b }
}
Step 'pairs-v2' { & $py -m r2ai.s37.handoff pairsv2 }
Step 'jina-final2-pool' { & $py -m r2ai.s37.jinapool final2 }
Step 'jina-final2-run' { & powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\s37\watch_jina.ps1" -Run "$run\jina-final2" -MaxRestarts 8 }
Step 'manifest' { & $py -m r2ai.s37.handoff manifest }
Note 'ALL DONE'
