#!/usr/bin/env bash
# 120A-p4b-v0 (S35, 25/10): EF2-cut-p4b frame (pool cosine13) on the S37 data (docsel from `python -m r2ai.s36.s37 apply --config configs/s36-s37-120A-p4b-v0.json`),
# S38 compose full+w768 without a docread ZIP; carriers that do not fit are dropped by the global lowest-P4B rule (--fit-carriers).
set -u
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-dev}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$S38" && export PYTHONPATH="$S38/src" PYTHONIOENCODING=utf-8
echo "$(date -Is) compose 120A-p4b-v0 start" >> "$RUN/120a-p4b-v0.log"
"$PY" -u -m r2ai.s38.compose run --name 120A-p4b-v0-full+w768 --docsel "$RUN/s37/docsel-120A-p4b-v0.parquet" --rule full+w768 \
  --carriers "$RUN/s37/carriers-120A-p4b-v0.parquet" --fit-carriers --rounds "${ROUNDS:-10}" >> "$RUN/compose-120A-p4b-v0.log" 2>&1
echo "$(date -Is) compose 120A-p4b-v0 rc=$?" >> "$RUN/120a-p4b-v0.log"
