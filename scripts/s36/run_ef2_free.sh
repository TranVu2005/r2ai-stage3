#!/usr/bin/env bash
# EF2-free -docread bundle + docsel (S35, 24/10), after `python -m r2ai.s36.cal ef2-free`; then the S26 validator.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
note() { echo "$(date -Is) $*" >> "$RUN/ef2free.log"; }
note "build EF2-free start"
"$PY" -m r2ai.s36.build --variant EF2-free --out "$RUN/build-docread" > "$RUN/EF2-free.stdout" 2> "$RUN/EF2-free.stderr"; note "build rc=$?"
z="$RUN/build-docread/EF2-free-docread.zip"
[ -f "$z" ] && { "$PY" scripts/s26/validate_pa3.py "$z" "$HANDOFF" > "${z%.zip}.validate.json" 2> "${z%.zip}.validate.err"; note "validate rc=$?"; }
note "done"
