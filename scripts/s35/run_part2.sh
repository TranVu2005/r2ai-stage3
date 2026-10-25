#!/usr/bin/env bash
# S35 part 2 end to end: Jina scoring (resumable) -> JINA-K40 + Check-J4-K40 -> J3-thrC -> S26 validator on every ZIP.
# Usage (from the S35 worktree): scripts/s35/run_part2.sh [layout]     log: out/runs/s35-part-jina/part2.log
set -u
LAYOUT="${1:-b10s0}"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$HERE/src"
RUN="${R2AI_OUT_DIR:-D:/GitHub/r2ai-stage3/out}/runs/s35-part-jina"
LOG="$RUN/part2.log"
PHARM="${R2AI_OUT_DIR:-D:/GitHub/r2ai-stage3/out}/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start layout=$LAYOUT"
bash "$HERE/scripts/s35/run_jina.sh" "$LAYOUT" || { say "jina scoring failed"; exit 1; }
say "jina scores collected"
"$PY" -u -m r2ai.s35.topk --variant JINA-K40 >> "$RUN/build-topk.stdout" 2>> "$RUN/build-topk.stderr"; say "topk build exit $?"
"$PY" -u -m r2ai.s35.build --variant J3-thrC --jina-scores "$RUN/jina/scores-all.parquet" --out "$RUN/build-j3" >> "$RUN/build-j3.stdout" 2>> "$RUN/build-j3.stderr"; say "j3 build exit $?"
for zip in "$RUN"/build-topk/*.zip "$RUN"/build-j3/*.zip; do
  [ -f "$zip" ] || continue
  "$PY" -u "$HERE/scripts/s26/validate_pa3.py" "$zip" "$PHARM" > "${zip%.zip}.validate.json" 2> "${zip%.zip}.validate.err"; say "validate $(basename "$zip") exit $?"
done
say "ALLDONE"
