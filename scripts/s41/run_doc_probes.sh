#!/usr/bin/env bash
# S41 (27/10): doc-axis probes cut from the validated FREE-cut150 bundle (r2ai.s41.doc_probe), sequential (RAM).
#   FREE-cut110-sub: docsel of the lambda = 1 EF2 cut (S40 apply FREE-lam1-check, (a, b, s) of the 18-number refit)
#   FREE-cut90-sub:  apply configs/s41-cut90.json (lambda solved for 90 docs / query), then the cut
# Checkpointed: a probe whose validated probe.json exists is skipped. Log out/runs/s41/probe/probe.log.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
S40="D:/GitHub/r2ai-stage3/out/runs/s40/cut"
S41="D:/GitHub/r2ai-stage3/out/runs/s41"
BASE="$S40/compose/FREE-cut150-full+w768-t1536.zip"
BASE_SHA=183364075f8b8bcd10b20fc788027313303f9393f7ce29dfa61e4e9737e448b9
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$S41/probe" "$S41/cut/apply"
LOG="$S41/probe/probe.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$S41/probe/probe.pid"
done_ok() { [ -f "$S41/probe/$1.probe.json" ] && grep -q '"ok": true' "$S41/probe/$1.probe.json"; }

if [ ! -f "$S41/cut/apply/docsel-FREE-cut90.parquet" ]; then
  say "apply FREE-cut90 start"
  "$PY" -u -m r2ai.s36.s37 apply --config configs/s41-cut90.json > "$S41/cut/apply-FREE-cut90.log" 2>&1 || { say "apply FREE-cut90 FAILED"; exit 1; }
  say "apply FREE-cut90 done"
fi

probe() {  # name docsel apply-json
  if done_ok "$1"; then say "probe $1 skipped (validated)"; return 0; fi
  say "probe $1 start"
  "$PY" -u -m r2ai.s41.doc_probe --name "$1" --base-zip "$BASE" --base-sha "$BASE_SHA" --docsel "$2" --apply-json "$3" \
    --out "$S41/probe" >> "$S41/probe/$1.log" 2>&1 || { say "probe $1 FAILED"; exit 1; }
  say "probe $1 done"
}
probe FREE-cut110-sub "$S40/apply/docsel-FREE-lam1-check.parquet" "$S40/apply/FREE-lam1-check.apply.json"
probe FREE-cut90-sub "$S41/cut/apply/docsel-FREE-cut90.parquet" "$S41/cut/apply/FREE-cut90.apply.json"
say "all done"
