#!/usr/bin/env bash
# K4-free regression gate (RRF(cosine, 4B) through the FUSE4-free path, pre-fix chunk rule -> JSON of S31 K4-free); slow (body-cut loop).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
until grep -q "free chain done" "$RUN/free.log" 2>/dev/null; do sleep 30; done
echo "$(date -Is) free check 2 start" >> "$RUN/free.log"
"$PY" -m r2ai.s36.build --variant Check-FUSE4-free-id --out "$RUN/build-check" > "$RUN/check-free.stdout" 2> "$RUN/check-free.stderr"; echo "$(date -Is) check-free rc=$?" >> "$RUN/free.log"
