#!/usr/bin/env bash
# Score the S36 C-max supplement (Qwen3-Reranker-0.6B on the chunks / doc_text rows of the free-frame pairs S32 did not score)
# with the S32 settings; restart after a CUDA failure (finished 2,000-row parts are kept). Stops after 3 attempts without a new part.
# Needs out/runs/s36-fuse4/extra/cmax (python -m r2ai.s36.extra cmax-prepare).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
ROOT="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/extra"
SNAP="D:/GitHub/r2ai-stage3/data/hf_model_cache/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
LOG="$ROOT/cmax-score.log"
ARGS=(--pool "$ROOT/cmax" --out "$ROOT/cmax-gpu" --model qwen3-reranker-0.6b --model-path "$SNAP" --subset all --part-pairs 2000)
futile=0
for attempt in $(seq 1 60); do
  before=$(ls "$ROOT/cmax-gpu/qwen3-reranker-0.6b/parts" 2>/dev/null | wc -l)
  status=$("$PY" -m r2ai.s31.judge_score status "${ARGS[@]}" 2>>"$LOG" | tail -1)
  if echo "$status" | grep -q '"complete": true'; then
    echo "$(date -Is) complete $status" >> "$LOG"
    "$PY" -m r2ai.s31.judge_score collect "${ARGS[@]}" >> "$LOG" 2>&1 && echo "$(date -Is) collected" >> "$LOG"
    exit 0
  fi
  echo "$(date -Is) attempt=$attempt parts_before=$before" >> "$LOG"
  "$PY" -m r2ai.s31.judge_score score "${ARGS[@]}" --max-batch-tokens 8192 --max-batch-size 16 >> "$LOG" 2>&1
  after=$(ls "$ROOT/cmax-gpu/qwen3-reranker-0.6b/parts" 2>/dev/null | wc -l)
  echo "$(date -Is) attempt=$attempt exit parts_after=$after" >> "$LOG"
  if grep -q -E "FloatingPointError|Non-finite" "$LOG"; then echo "$(date -Is) STOP non-finite" >> "$LOG"; exit 3; fi
  if [ "$after" -le "$before" ]; then futile=$((futile+1)); else futile=0; fi
  [ "$futile" -ge 3 ] && { echo "$(date -Is) STOP futile" >> "$LOG"; exit 4; }
  sleep 20
done
