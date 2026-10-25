# Detached chain: embed the en PubMed segment (resumable parts of 10,000 docs), then resume the paused Jina top-up under its watchdog,
# then refresh the hand-off manifest. Log: out\runs\s37\en_embed.log. The embed retries when the GPU guard refuses (a desktop app is busy).
$repo = 'D:\GitHub\r2ai-stage3'; $py = "$repo\.venv\Scripts\python.exe"; $run = "$repo\out\runs\s37"; $log = "$run\en_embed.log"
Set-Location $repo
function Note($m) { for ($i = 0; $i -lt 5; $i++) { try { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 300 } } }
Note 'start en embed'
for ($try = 1; $try -le 20; $try++) {
    Note "embed try $try"
    & $py -m r2ai.s37.en embed --device cuda --max-util 70 *>> "$run\en_embed.out"
    if (Test-Path "$repo\data\s37_en\seg37900-en-pubmed\vectors.npz") { break }
    Note "embed incomplete (rc=$LASTEXITCODE), waiting 2 min"; Start-Sleep -Seconds 120
}
if (-not (Test-Path "$repo\data\s37_en\seg37900-en-pubmed\vectors.npz")) { Note 'STOP embed did not finish; Jina NOT resumed, resume by hand'; exit 1 }
Note 'embed done; resuming Jina top-up'
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\s37\watch_jina.ps1" -Run "$run\jina-final2" -MaxRestarts 8
Note "jina watchdog exit=$LASTEXITCODE"
& $py -m r2ai.s37.handoff manifest *>> "$run\en_embed.out"
Note 'ALL DONE'
