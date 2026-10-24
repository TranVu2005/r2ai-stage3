#!/usr/bin/env bash
# CR-fair then CR-swap on the EF2-free docsel (S35, 24/10): carriers sets by r2ai.s36.carriers, then S38 compose full+w768.
# One at a time (a compose build needs several GB of RAM). Usage: scripts/s36/run_cr.sh [fair] [swap]
set -u
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-chunk-metric-v2}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
cd "$S38"
for r in "${@:-fair swap}"; do
  case "$r" in fair) N=EF2-free-CRfair;; swap) N=EF2-free-CRswap;; esac
  echo "$(date -Is) compose $N start" >> "$RUN/cr.log"
  bash scripts/s38/compose.sh EF2-free full+w768 --name "$N" --carriers "$RUN/carriers/carriers-$N.parquet" > "$RUN/compose-$N.log" 2>&1
  echo "$(date -Is) compose $N rc=$?" >> "$RUN/cr.log"
done
echo "$(date -Is) cr done" >> "$RUN/cr.log"
