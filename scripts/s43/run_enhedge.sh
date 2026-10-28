#!/usr/bin/env bash
# S43 (27/10, Vu): EN-hedge probes on the CH-rrf ZIP, one after the other (RAM): n=3 at the P99 of the en cosine over every
# (query, en doc) pair (0,5226) and at the P99 of the 24,000 top-20 lane pairs (0,6991) (out/runs/s43/probes/en/threshold.json).
# Log out/runs/s43/probes/enhedge.log, PID out/runs/s43/probes/enhedge.pid; per probe run-<name>.out. Restart-safe: a probe whose
# <name>/*.enhedge.json reports validator ok is skipped.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s43/probes"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
LOG="$RUN/enhedge.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/enhedge.pid"
for P in ${PROBES:-EN-hedge3-c5226:0.5226 EN-hedge3-c6991:0.6991168761253355}; do
  N="${P%%:*}"; C="${P##*:}"
  if ls "$RUN/$N"/*.enhedge.json >/dev/null 2>&1 && grep -q '"ok": true' "$RUN/$N"/*.enhedge.json; then say "$N skipped (validated)"; continue; fi
  say "$N start (n 3, min_cos $C)"
  "$PY" -u -m r2ai.s43.enhedge public --n 3 --min-cos "$C" --name "$N" --out "$RUN" >> "$RUN/run-$N.out" 2>&1 || { say "$N FAILED"; exit 1; }
  say "$N done"
done
say "all done"
