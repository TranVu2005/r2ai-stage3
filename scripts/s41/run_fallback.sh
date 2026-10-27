#!/usr/bin/env bash
# S41 (27/10, Vu): compose the private fallback probes on the public queries (scripts/s41/fallback_public.py --compose), one after the other
# (RAM). A variant whose fallback.json already has a validated compose is skipped, so the driver can simply be started again after a reboot.
#   VARIANTS="B0" bash scripts/s41/run_fallback.sh          (default: B0 IMP30 COS)
# Log out/runs/s41/fallback/fallback.log, PID out/runs/s41/fallback/fallback.pid; per variant <variant>/compose.log and run-<variant>.out.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/fallback"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN"
LOG="$RUN/fallback.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/fallback.pid"
for V in ${VARIANTS:-B0 IMP30 COS}; do
  if [ -f "$RUN/$V/fallback.json" ] && grep -q '"validator_ok": true' "$RUN/$V/fallback.json"; then say "$V skipped (validated compose)"; continue; fi
  say "$V start"
  "$PY" -u "$WT/scripts/s41/fallback_public.py" --variant "$V" --out "$RUN" --compose >> "$RUN/run-$V.out" 2>&1 || { say "$V FAILED"; exit 1; }
  say "$V done"
done
say "all done"
