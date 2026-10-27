#!/usr/bin/env bash
# S43 (27/10, Vu): compose the carrier probes on the public queries, one after the other (RAM), after the S41 fallback job
# (scripts/s41/run_fallback.sh) has finished: wait while any python process runs fallback_public.py or r2ai.s38.compose.
#   PROBES="P1-carrier-w2:2 P2-carrier-w3:3" bash scripts/s43/run_probes.sh
# A probe whose check.json already has a validated compose is skipped (restart-safe). Log out/runs/s43/probes/probes.log,
# PID out/runs/s43/probes/probes.pid; per probe <name>/compose.log and run-<name>.out.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s43/probes"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN"
LOG="$RUN/probes.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/probes.pid"
busy() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'fallback_public|r2ai\.s38\.compose' }).Count" | tr -d '\r'
}
say "waiting for other compose jobs"
while [ "$(busy)" != "0" ]; do sleep 120; done
say "no other compose job"
for P in ${PROBES:-P1-carrier-w2:2 P2-carrier-w3:3}; do
  N="${P%%:*}"; W="${P##*:}"
  if [ -f "$RUN/$N/check.json" ] && grep -q '"validator_ok": true' "$RUN/$N/check.json"; then say "$N skipped (validated compose)"; continue; fi
  say "$N start (w_cos_carrier $W)"
  "$PY" -u "$WT/scripts/s43/build_public.py" --name "$N" --w-cos-carrier "$W" --out "$RUN" --compose >> "$RUN/run-$N.out" 2>&1 || { say "$N FAILED"; exit 1; }
  say "$N done"
done
say "all done"
