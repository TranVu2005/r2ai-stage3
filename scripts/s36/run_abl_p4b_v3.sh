#!/usr/bin/env bash
# Ablation of 120A-p4b-v3 (S35, 26/10): usage `run_abl_p4b_v3.sh noJina|noCmax` after
# `python -m r2ai.s36.s37 apply --config configs/s36-abl-p4b-v3-<tag>.json`. Same compose as run_120a_p4b_v3.sh (full+w768, --fit-carriers, T searched).
set -u
TAG="$1"; NAME="ABL-${TAG}-p4b-v3"
S38="${S38:-D:/GitHub/r2ai-stage3/.superpowers/worktrees/s38-dev}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$S38" && export PYTHONPATH="$S38/src" PYTHONIOENCODING=utf-8
echo "$(date -Is) compose $NAME start" >> "$RUN/abl-p4b-v3.log"
"$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$RUN/s37/docsel-$NAME.parquet" --rule full+w768 \
  --carriers "$RUN/s37/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" >> "$RUN/compose-$NAME.log" 2>&1
echo "$(date -Is) compose $NAME rc=$?" >> "$RUN/abl-p4b-v3.log"
