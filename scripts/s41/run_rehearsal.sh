#!/usr/bin/env bash
# S41 (27/10): rehearsal of the frozen private config (configs/s41-private-final.json, newq default) on the 100 rehearsal queries
# (out/runs/s40/rehearsal-input, ids 5001..5100): prepare -> export-scoring -> import the 4B / C-max scores of out/runs/s40/rh100 (same
# pairs: the export must reproduce rh100's pair / chunk-row ids, checked) -> build (full 4B) -> build --impute-4b -> build --impute-4b
# --test-drop-4b 0.3 (seed 40) -> validator (inside build). Every step timed in <run>/rehearsal-times.tsv. Run out/runs/s41/rh100-final.
# Waits for the public regression compose (RAM) when REGRESS_PID is set.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/rh100-final"
OLD="D:/GitHub/r2ai-stage3/out/runs/s40/rh100"
QUERIES="D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/queries-rh100.parquet"
QSHA=181819a3291aed7f12b6b9a1859d27fb54462f25b24a6d393938fb0aa9c2dfda
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN"
LOG="$RUN/rehearsal.log"
TIMES="$RUN/rehearsal-times.tsv"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/rehearsal.pid"
if [ -n "${REGRESS_PID:-}" ]; then while kill -0 "$REGRESS_PID" 2>/dev/null; do sleep 30; done; fi
step() {   # step <label> <command...>
  local label="$1"; shift
  say "$label start"
  local t0=$(date +%s)
  "$@" >> "$RUN/$label.out" 2>&1 || { say "$label FAILED (see $RUN/$label.out)"; exit 1; }
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\n' "$label" "$dt" >> "$TIMES"
  say "$label done (${dt}s)"
}
[ "$(sha256sum "$QUERIES" | cut -d' ' -f1)" = "$QSHA" ] || { say "queries SHA256 differs from $QSHA"; exit 1; }
step prepare "$PY" -u -m r2ai.newq prepare --queries "$QUERIES" --run "$RUN"
step export "$PY" -u -m r2ai.newq export-scoring --run "$RUN"
step same-pairs "$PY" -u "$WT/scripts/s41/rehearsal_same_pairs.py" "$RUN" "$OLD"
TAG="$(basename "$RUN")-4b-1"
step import-4b "$PY" -u -m r2ai.newq import-scores --run "$RUN" --source 4b --tag "$TAG" \
  --scores "$OLD/4b/rh100-4b-1/scores-extra.parquet" --report "$OLD/4b/rh100-4b-1/collect-extra-report.json"
step import-cmax "$PY" -u -m r2ai.newq import-scores --run "$RUN" --source cmax --scores "$OLD/cmax/gpu/qwen3-reranker-0.6b/scores-all.parquet"
step build "$PY" -u -m r2ai.newq build --run "$RUN"
step build-impute "$PY" -u -m r2ai.newq build --run "$RUN" --impute-4b
step build-drop30-impute "$PY" -u -m r2ai.newq build --run "$RUN" --impute-4b --test-drop-4b 0.3 --test-drop-seed 40
say "all done"
