#!/usr/bin/env bash
# Score the S35 Jina pool until every part exists; restart after a CUDA/driver crash (parts are atomic, resumable).
# Usage: scripts/s35/run_jina.sh [layout]   (run from the S35 worktree; needs out/runs/s35-part-jina/jina/pool.parquet)
set -u
LAYOUT="${1:-b50s0}"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)/src"
LOG="${R2AI_OUT_DIR:-D:/GitHub/r2ai-stage3/out}/runs/s35-part-jina/jina/score.log"
for attempt in $(seq 1 30); do
  "$PY" -u -m r2ai.s35.jina score --layout "$LAYOUT" >> "$LOG" 2>&1
  if "$PY" -u -m r2ai.s35.jina collect >> "$LOG" 2>&1; then echo "collected after attempt $attempt" >> "$LOG"; exit 0; fi
  echo "attempt $attempt incomplete, restarting in 30 s" >> "$LOG"; sleep 30
done
exit 1
