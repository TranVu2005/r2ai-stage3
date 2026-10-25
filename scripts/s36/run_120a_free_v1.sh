#!/usr/bin/env bash
# 120A-free-v1 (S35, 25/10): 120A-free-v0 with the full S37 Jina scores (intermediate, not submitted) (docsel from `python -m r2ai.s36.s37 apply --config configs/s36-s37-120A-free-v1.json`),
# S38 compose full+w768 without a docread ZIP; carriers that do not fit are dropped by the global lowest-P4B rule (--fit-carriers).
set -u
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-dev}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$S38" && export PYTHONPATH="$S38/src" PYTHONIOENCODING=utf-8
echo "$(date -Is) compose 120A-free-v1 start" >> "$RUN/120a-free-v1.log"
"$PY" -u -m r2ai.s38.compose run --name 120A-free-v1-full+w768 --docsel "$RUN/s37/docsel-120A-free-v1.parquet" --rule full+w768 \
  --carriers "$RUN/s37/carriers-120A-free-v1.parquet" --fit-carriers --rounds "${ROUNDS:-10}" >> "$RUN/compose-120A-free-v1.log" 2>&1
echo "$(date -Is) compose 120A-free-v1 rc=$?" >> "$RUN/120a-free-v1.log"
