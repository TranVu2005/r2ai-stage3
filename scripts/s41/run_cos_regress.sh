#!/usr/bin/env bash
# S41 (Vu 27/10): regressions of the COS fallback (`newq build --cos-only`, configs/s41-private-fallback-cos.json), one after the other (RAM):
#   rh100-cos  rehearsal on the 100 rehearsal queries with NO 4B imported: prepare -> export-scoring -> same pairs as out/runs/s40/rh100 ->
#              import C-max -> build --cos-only (validator inside build); every step timed in <run>/rehearsal-times.tsv
#   CH-rrf     scripts/s41/regress_public.py --variant CH-rrf --compose: ZIP must be b12b8591...08c8 byte for byte
#   COS        scripts/s41/regress_public.py --variant COS --compose: ZIP must be 1c5600bb...0fc1 byte for byte
# Log out/runs/s41/cos-regress/cos-regress.log, PID cos-regress.pid. STEPS="rh100 CH-rrf COS" (default) picks the parts.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
OUT="D:/GitHub/r2ai-stage3/out/runs/s41/cos-regress"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/rh100-cos"
OLD="D:/GitHub/r2ai-stage3/out/runs/s40/rh100"
QUERIES="D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/queries-rh100.parquet"
QSHA=181819a3291aed7f12b6b9a1859d27fb54462f25b24a6d393938fb0aa9c2dfda
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$OUT" "$RUN"
LOG="$OUT/cos-regress.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$OUT/cos-regress.pid"
step() {   # step <dir> <label> <command...>
  local dir="$1" label="$2"; shift 2
  say "$label start"
  local t0=$(date +%s)
  "$@" >> "$dir/$label.out" 2>&1 || { say "$label FAILED (see $dir/$label.out)"; exit 1; }
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\n' "$label" "$dt" >> "$dir/rehearsal-times.tsv"
  say "$label done (${dt}s)"
}
for S in ${STEPS:-rh100 CH-rrf COS}; do
  case "$S" in
    rh100)
      [ "$(sha256sum "$QUERIES" | cut -d' ' -f1)" = "$QSHA" ] || { say "queries SHA256 differs from $QSHA"; exit 1; }
      step "$RUN" prepare "$PY" -u -m r2ai.newq prepare --queries "$QUERIES" --run "$RUN"
      step "$RUN" export "$PY" -u -m r2ai.newq export-scoring --run "$RUN"
      step "$RUN" same-pairs "$PY" -u "$WT/scripts/s41/rehearsal_same_pairs.py" "$RUN" "$OLD"
      step "$RUN" import-cmax "$PY" -u -m r2ai.newq import-scores --run "$RUN" --source cmax --scores "$OLD/cmax/gpu/qwen3-reranker-0.6b/scores-all.parquet"
      step "$RUN" build-cos-only "$PY" -u -m r2ai.newq build --run "$RUN" --cos-only ;;
    CH-rrf|COS)
      step "$OUT" "regress-$S" "$PY" -u "$WT/scripts/s41/regress_public.py" --variant "$S" --out "$OUT/$S" --compose ;;
  esac
done
say "all done"
