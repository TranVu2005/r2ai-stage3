#!/usr/bin/env bash
# S36 thrC frame: regression gates, then FUSE4-thrC and FUSE4w-thrC; validates every ZIP. Sequential (each build ~8 GB RAM).
# Steps already done can be skipped: SKIP_ID=1 skips the fx-path gate, SKIP_PRE=1 the pre-fix gate.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
mkdir -p "$RUN"
step() { echo "$(date -Is) $*" >> "$RUN/thrc.log"; }
validate() { "$PY" scripts/s26/validate_pa3.py "$1" "$HANDOFF" > "${1%.zip}.validate.json" 2> "${1%.zip}.validate.err"; step "validate $1 rc=$?"; }
step start
[ -z "${SKIP_ID:-}" ] && { "$PY" -m r2ai.s36.build --variant Check-FUSE4-thrC-id --out "$RUN/build-check" > "$RUN/check-id.stdout" 2> "$RUN/check-id.stderr"; step "check-id rc=$?"; }
"$PY" -m r2ai.s36.build --variant FUSE4-thrC --variant FUSE4w-thrC --variant Check-FUSE4-thrC-th --out "$RUN/build" > "$RUN/thrc.stdout" 2> "$RUN/thrc.stderr"; step "thrC rc=$?"
for n in FUSE4-thrC FUSE4w-thrC Check-FUSE4-thrC-th; do [ -f "$RUN/build/$n.zip" ] && validate "$RUN/build/$n.zip"; done
[ -z "${SKIP_PRE:-}" ] && { "$PY" -m r2ai.s36.build --variant Check-FUSE4-thrC-pre --out "$RUN/build-check" > "$RUN/check-pre.stdout" 2> "$RUN/check-pre.stderr"; step "check-pre rc=$?"; }
step ALLDONE
