#!/usr/bin/env bash
# EF2-free / EF2-free-d211 -docread bundle + docsel (S35, 24/10), after `python -m r2ai.s36.cal ef2-free[-d211]`; then the S26 validator.
# Usage: scripts/s36/run_ef2_free.sh [EF2-free|EF2-free-d211]
set -u
V="${1:-EF2-free}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
note() { echo "$(date -Is) $*" >> "$RUN/$V.log"; }
note "build $V start"
"$PY" -m r2ai.s36.build --variant $V --out "$RUN/build-docread" > "$RUN/$V.stdout" 2> "$RUN/$V.stderr"; note "build rc=$?"
z="$RUN/build-docread/$V-docread.zip"
[ -f "$z" ] && { "$PY" scripts/s26/validate_pa3.py "$z" "$HANDOFF" > "${z%.zip}.validate.json" 2> "${z%.zip}.validate.err"; note "validate rc=$?"; }
note "done"
