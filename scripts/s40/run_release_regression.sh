#!/usr/bin/env bash
# S40 regression (26/10): after the S40 edits of apply / compose / registry / validator, both hand-off builds must still be rebuilt byte for
# byte (python -m r2ai.release build compares every output with the manifest). Waits for the FREE-noJina-canon compose (RAM), then runs
# free-v2 and p4b-v3 one after the other into out/runs/s40/release-regression/<name>.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="D:/GitHub/r2ai-stage3/out/runs/s40/release-regression"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$OUT"; LOG="$OUT/run.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
until grep -q "canon rc=" D:/GitHub/r2ai-stage3/out/runs/s40/canon/run.log 2>/dev/null; do sleep 60; done
for N in 120A-free-v2 120A-p4b-v3; do
  if [ -f "$OUT/$N/release-report.json" ] && grep -q '"ok": true' "$OUT/$N/release-report.json" && grep -q '"compare"' "$OUT/$N/release-report.json"; then continue; fi
  say "$N start"; "$PY" -u -m r2ai.release build --config "configs/s36-handoff-$N.json" --work "$OUT/$N" >> "$OUT/$N.out" 2>&1; say "$N rc=$?"
done
say "ALL DONE"
