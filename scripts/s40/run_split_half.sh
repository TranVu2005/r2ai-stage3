#!/usr/bin/env bash
# S40 (26/10): FREE-split and FREE-half at T = 1536 (Vu), after the free-v2 ablations (one compose-sized job at a time: RAM).
# Checkpointed: a ZIP whose compose.json has a validator result is skipped, so the watchdog can rerun this script.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="D:/GitHub/r2ai-stage3/out/runs/s40/split"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
until grep -q "ALL DONE" D:/GitHub/r2ai-stage3/out/runs/s40/abl/abl.log 2>/dev/null; do sleep 60; done
if [ ! -f "$OUT/split.cands.parquet" ]; then
  say "prepare start"; "$PY" -u scripts/s40/split_compose.py prepare >> "$OUT/prepare.out" 2>&1 || { say "prepare FAILED"; exit 1; }; say "prepare done"
fi
for MODE in split half; do
  NAME="FREE-$MODE-t1536"
  if [ -f "$OUT/$NAME.compose.json" ] && grep -q '"validator"' "$OUT/$NAME.compose.json"; then continue; fi
  say "$NAME start"; "$PY" -u scripts/s40/split_compose.py write --t 1536 --mode "$MODE" >> "$OUT/$NAME.out" 2>&1 || { say "$NAME FAILED"; exit 1; }; say "$NAME done"
done
say "ALL DONE"
