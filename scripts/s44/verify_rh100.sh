#!/usr/bin/env bash
# S44: rebuild the two recorded rh100 rehearsals from tag v-private-2026-10-28c and compare with the recorded SHA256.
#   A rh100-final   : prepare, export, same-pairs, import 4B + C-max (old s40/rh100 scores), build         -> ZIP 57dca179...
#                     build --no-cmax (B0), build --en-hedge 3 (no recorded SHA: validator only)
#   B rh100-cosk150 : prepare, export, same-pairs, import C-max, build --cos-k 150                         -> ZIP 23d1d3ae...
#                     build --cos-k 150 --en-hedge 3 --name rh100-cosk150-free-cosk150-en3                  -> en3 ZIP f13ee54c...
# Usage: verify_rh100.sh <out dir below out/runs, e.g. D:/GitHub/r2ai-stage3/out/runs/s44-verify-1>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTROOT="$1"
WT="D:/GitHub/r2ai-stage3/.superpowers/worktrees/s44-verify-28c"
PY="D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe"
OLD="D:/GitHub/r2ai-stage3/out/runs/s40/rh100"
QUERIES="D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/queries-rh100.parquet"
QSHA=181819a3291aed7f12b6b9a1859d27fb54462f25b24a6d393938fb0aa9c2dfda
HFV="D:/_r2ai_s44_hf/hf_home"            # HF_HOME holding only BAAI/bge-m3 (junction)
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8 HF_HOME="$HFV" HF_HUB_OFFLINE=1 \
  R2AI_LEGACY_OUT_DIR="D:/_r2ai_s44_absent/out" R2AI_LEGACY_STATE_DIR="D:/_r2ai_s44_absent/state"
[ "$(git rev-parse HEAD)" = 7e8894d34184716b1a668bb5c8f2c00e184439a2 ] || { echo "worktree not at tag 28c"; exit 1; }
mkdir -p "$OUTROOT"
LOG="$OUTROOT/verify.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
step() {   # step <run> <label> <command...>
  local run="$1" label="$2"; shift 2
  say "$label start"; local t0=$(date +%s)
  "$@" >> "$run/$label.out" 2>&1 || { say "$label FAILED (see $run/$label.out)"; exit 1; }
  say "$label done ($(( $(date +%s) - t0 ))s)"
}
[ "$(sha256sum "$QUERIES" | cut -d' ' -f1)" = "$QSHA" ] || { say "queries SHA256 differs"; exit 1; }
A="$OUTROOT/rh100-final"; B="$OUTROOT/rh100-cosk150"
mkdir -p "$A" "$B"
step "$A" prepare "$PY" -u -m r2ai.newq prepare --queries "$QUERIES" --run "$A"
step "$A" export "$PY" -u -m r2ai.newq export-scoring --run "$A"
step "$A" same-pairs "$PY" -u "$WT/scripts/s41/rehearsal_same_pairs.py" "$A" "$OLD"
step "$A" import-4b "$PY" -u -m r2ai.newq import-scores --run "$A" --source 4b --tag rh100-final-4b-1 \
  --scores "$OLD/4b/rh100-4b-1/scores-extra.parquet" --report "$OLD/4b/rh100-4b-1/collect-extra-report.json"
step "$A" import-cmax "$PY" -u -m r2ai.newq import-scores --run "$A" --source cmax --scores "$OLD/cmax/gpu/qwen3-reranker-0.6b/scores-all.parquet"
step "$A" build "$PY" -u -m r2ai.newq build --run "$A"
step "$A" build-noCmax "$PY" -u -m r2ai.newq build --run "$A" --no-cmax
step "$A" build-en3 "$PY" -u -m r2ai.newq build --run "$A" --en-hedge 3 --name rh100-final-free-en3
step "$B" prepare "$PY" -u -m r2ai.newq prepare --queries "$QUERIES" --run "$B"
step "$B" export "$PY" -u -m r2ai.newq export-scoring --run "$B"
step "$B" same-pairs "$PY" -u "$WT/scripts/s41/rehearsal_same_pairs.py" "$B" "$OLD"
step "$B" import-cmax "$PY" -u -m r2ai.newq import-scores --run "$B" --source cmax --scores "$OLD/cmax/gpu/qwen3-reranker-0.6b/scores-all.parquet"
step "$B" build-cos-k150 "$PY" -u -m r2ai.newq build --run "$B" --cos-k 150
step "$B" build-cos-k150-en3 "$PY" -u -m r2ai.newq build --run "$B" --cos-k 150 --en-hedge 3 --name rh100-cosk150-free-cosk150-en3
"$PY" -I "$HERE/compare_rh100.py" "$OUTROOT" | tee -a "$LOG"
say "all done"
