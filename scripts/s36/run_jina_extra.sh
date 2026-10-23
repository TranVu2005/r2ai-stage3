#!/usr/bin/env bash
# Score the S36 Jina supplement (free-frame pairs S35 did not score) until every part exists; restart after a crash.
# Needs out/runs/s36-fuse4/extra/jina/pool.parquet (python -m r2ai.s36.extra jina-prepare).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
ROOT="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/extra/jina"
LOG="$ROOT/score.log"
for attempt in $(seq 1 30); do
  "$PY" -u -m r2ai.s35.jina score --out "$ROOT" --layout b10s0 >> "$LOG" 2>&1
  if "$PY" -u -m r2ai.s35.jina collect --out "$ROOT" --layout b10s0 >> "$LOG" 2>&1; then
    "$PY" -u -m r2ai.s36.extra jina-reduce >> "$LOG" 2>&1
    echo "collected after attempt $attempt" >> "$LOG"; exit 0
  fi
  echo "attempt $attempt incomplete, restarting in 30 s" >> "$LOG"; sleep 30
done
exit 1
