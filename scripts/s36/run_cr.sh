#!/usr/bin/env bash
# CR-fair then CR-swap on the EF2-free docsel (S35, 24/10): carrier sets by r2ai.s36.carriers, then S38 compose full+w768.
# One at a time (a compose build needs several GB of RAM); a compose run already in flight finishes first.
# Usage: scripts/s36/run_cr.sh fair swap   (S38=<worktree whose compose has --carriers>, ROUNDS=<n> default 10: rounds of the T search,
# measured candidates are reused between runs)
set -u
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-dev}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
cd "$S38"
busy() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'r2ai[.]s38[.]compose' }).Count" | tr -d '\r'; }
while [ "$(busy)" != "0" ]; do sleep 30; done
for r in "$@"; do
  case "$r" in fair) N=EF2-free-CRfair;; swap) N=EF2-free-CRswap;; esac
  echo "$(date -Is) compose $N start" >> "$RUN/cr.log"
  bash scripts/s38/compose.sh EF2-free full+w768 --name "$N" --carriers "$RUN/carriers/carriers-$N.parquet" --rounds "${ROUNDS:-10}" >> "$RUN/compose-$N.log" 2>&1
  echo "$(date -Is) compose $N rc=$?" >> "$RUN/cr.log"
done
echo "$(date -Is) cr done" >> "$RUN/cr.log"
