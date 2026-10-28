#!/usr/bin/env bash
# S41 (Vu 28/10): regressions before tag v-private-2026-10-28c, on main after merging --cos-k (feat/s41) and --en-hedge (exp/s43),
# one after the other (RAM):
#   CH-rrf     scripts/s41/regress_public.py --variant CH-rrf --compose: ZIP must be b12b8591...08c8 byte for byte
#   EN-hedge   scripts/s41/regress_enhedge.py on the CH-rrf ZIP just rebuilt: ZIP must be ce0fabc0...8742 (S43, delivered) byte for byte
#   COSK150    scripts/s41/regress_public.py --variant COSK150 --compose: ZIP must be fd331252...c2fb byte for byte
#   rh100      rehearsal on the 100 rehearsal queries with NO 4B imported, fresh run: prepare -> export-scoring -> same pairs as
#              out/runs/s40/rh100 -> import C-max -> build --cos-k 150; every step timed in <run>/rehearsal-times.tsv
# Log out/runs/s41/regress-28c/regress-28c.log, PID regress-28c.pid. STEPS="CH-rrf EN-hedge COSK150 rh100" (default) picks the parts.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
OUT="D:/GitHub/r2ai-stage3/out/runs/s41/regress-28c"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/rh100-cosk150"
OLD="D:/GitHub/r2ai-stage3/out/runs/s40/rh100"
QUERIES="D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/queries-rh100.parquet"
QSHA=181819a3291aed7f12b6b9a1859d27fb54462f25b24a6d393938fb0aa9c2dfda
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$OUT" "$RUN"
LOG="$OUT/regress-28c.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$OUT/regress-28c.pid"
step() {   # step <dir> <label> <command...>
  local dir="$1" label="$2"; shift 2
  say "$label start"
  local t0=$(date +%s)
  "$@" >> "$dir/$label.out" 2>&1 || { say "$label FAILED (see $dir/$label.out)"; exit 1; }
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\n' "$label" "$dt" >> "$dir/rehearsal-times.tsv"
  say "$label done (${dt}s)"
}
for S in ${STEPS:-CH-rrf EN-hedge COSK150 rh100}; do
  case "$S" in
    CH-rrf|COSK150)
      step "$OUT" "regress-$S" "$PY" -u "$WT/scripts/s41/regress_public.py" --variant "$S" --out "$OUT/$S" --compose ;;
    EN-hedge)
      step "$OUT" "regress-$S" "$PY" -u "$WT/scripts/s41/regress_enhedge.py" --base "$OUT/CH-rrf/compose/CH-rrf-full+w768-t1536.zip" --out "$OUT/$S" ;;
    rh100)
      [ "$(sha256sum "$QUERIES" | cut -d' ' -f1)" = "$QSHA" ] || { say "queries SHA256 differs from $QSHA"; exit 1; }
      step "$RUN" prepare "$PY" -u -m r2ai.newq prepare --queries "$QUERIES" --run "$RUN"
      step "$RUN" export "$PY" -u -m r2ai.newq export-scoring --run "$RUN"
      step "$RUN" same-pairs "$PY" -u "$WT/scripts/s41/rehearsal_same_pairs.py" "$RUN" "$OLD"
      step "$RUN" import-cmax "$PY" -u -m r2ai.newq import-scores --run "$RUN" --source cmax --scores "$OLD/cmax/gpu/qwen3-reranker-0.6b/scores-all.parquet"
      step "$RUN" build-cos-k150 "$PY" -u -m r2ai.newq build --run "$RUN" --cos-k 150 ;;
  esac
done
say "all done"
