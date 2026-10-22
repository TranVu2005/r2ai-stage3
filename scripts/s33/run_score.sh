#!/usr/bin/env bash
# Supervisor for the S33 bge-reranker-v2-m3 runs: scores each pool in turn and restarts the scorer
# after a crash (a CUDA OOM leaves the context unusable; finished parts of 2,000 pairs are kept).
# Stops on: every pool complete, a non-finite score, or MAX_FUTILE attempts without a new part.
# Usage: bash scripts/s33/run_score.sh pool-rdoc pool-rchk
set -u
RUN=${RUN:-D:/GitHub/r2ai-stage3/out/runs/s33-btc-baseline}
PY=${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}
SNAP=${SNAP:-C:/Users/Admin/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e}
MAX_FUTILE=${MAX_FUTILE:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
LOG="$RUN/supervisor.log"
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor_pid=$$ $*" >> "$LOG"; }
echo $$ > "$RUN/supervisor.pid"
for name in "$@"; do
  POOL="$RUN/$name"; OUT="$RUN/gpu-${name#pool-}"
  parts="$OUT/bge-reranker-v2-m3/parts"
  futile=0; attempt=0
  note "start pool=$name"
  while true; do
    before=$(ls "$parts"/*.parquet 2>/dev/null | wc -l)
    status=$("$PY" -m r2ai.s33.bge_score status --pool "$POOL" --out "$OUT" | tail -n 1)
    if echo "$status" | grep -q '"complete": true'; then note "complete pool=$name $status"; break; fi
    attempt=$((attempt + 1))
    err="$OUT/score.stderr.attempt$attempt"; mkdir -p "$OUT"
    "$PY" -m r2ai.s33.bge_score score --pool "$POOL" --out "$OUT" --model-path "$SNAP" \
      > "$OUT/score.stdout.attempt$attempt" 2> "$err" &
    pid=$!; echo $pid > "$OUT/score.pid"
    note "pool=$name attempt=$attempt scorer_pid=$pid parts_before=$before"
    wait $pid; code=$?
    after=$(ls "$parts"/*.parquet 2>/dev/null | wc -l)
    note "pool=$name attempt=$attempt exit=$code parts_after=$after"
    if grep -q -E 'FloatingPointError|non-finite' "$err"; then note "STOP non-finite score"; exit 3; fi
    if [ "$after" -le "$before" ]; then futile=$((futile + 1)); else futile=0; fi
    if [ "$futile" -ge "$MAX_FUTILE" ]; then note "STOP $futile attempts without a new part"; exit 4; fi
    sleep 20
  done
done
note "all done"
