#!/usr/bin/env bash
# CR-fair builds (S35, 24/10, default chunk rule): carriers = top ceil(rho |D_q|) docs by P4B per query (rho global, r2ai.s36.carriers),
# S38 compose full+w768 on the -docread base of the docsel, thin by content, T searched.
# Usage: scripts/s36/run_crfair.sh EF2-cut-p4b:p4b-CRfair EF2-free-d211:d211-CRfair     (<docsel>:<name>; one at a time)
set -u
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-dev}"
S35="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
busy() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'r2ai[.]s38[.]compose' }).Count" | tr -d '\r'; }
while [ "$(busy)" != "0" ]; do sleep 30; done
for pair in "$@"; do
  D="${pair%%:*}"; N="${pair##*:}"
  echo "$(date -Is) $N start" >> "$RUN/crfair.log"
  (cd "$S35" && PYTHONPATH=src "$PY" -m r2ai.s36.carriers --docsel "$RUN/docsel/docsel-$D.parquet" --rule fair --name "$N" > "$RUN/carriers-$N.stdout") || { echo "$(date -Is) $N carriers failed" >> "$RUN/crfair.log"; continue; }
  (cd "$S38" && bash scripts/s38/compose.sh "$D" full+w768 --name "$N" --carriers "$RUN/carriers/carriers-$N.parquet" --rounds "${ROUNDS:-10}" >> "$RUN/compose-$N.log" 2>&1)
  echo "$(date -Is) $N compose rc=$?" >> "$RUN/crfair.log"
done
echo "$(date -Is) crfair done" >> "$RUN/crfair.log"
