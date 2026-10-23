#!/usr/bin/env bash
# Regression gate of the free frame: RRF(cosine, 4B) through the FUSE4-free path with the pre-fix chunk rule must give the
# JSON of K4-free (S31). Waits for the thrC script to finish (one build at a time: ~8 GB RAM each).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
until grep -q ALLDONE "$RUN/thrc.log"; do sleep 30; done
echo "$(date -Is) free check start" >> "$RUN/free.log"
"$PY" -m r2ai.s36.build --variant Check-FUSE4-free-id --out "$RUN/build-check" > "$RUN/check-free.stdout" 2> "$RUN/check-free.stderr"; echo "$(date -Is) check-free rc=$?" >> "$RUN/free.log"
