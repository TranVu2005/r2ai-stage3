#!/usr/bin/env bash
# CPU chain of the -docread bundles (one build at a time, ~8 GB RAM each): EF2-cut / LANG-thr (needs `python -m r2ai.s36.cal run`),
# then FUSE4-thrC / FUSE4w-thrC. Waits for the free-frame regression gate. Each ZIP goes through the S26 validator.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
until grep -q "check-free rc" "$RUN/free.log" 2>/dev/null; do sleep 30; done
note() { echo "$(date -Is) $*" >> "$RUN/docread.log"; }
build() {  # build <outdir> <variants...>
  local out="$1"; shift; local args=(); for v in "$@"; do args+=(--variant "$v"); done
  note "build $* start"
  "$PY" -m r2ai.s36.build "${args[@]}" --out "$RUN/$out" > "$RUN/$out.stdout" 2> "$RUN/$out.stderr"; note "build $* rc=$?"
  for v in "$@"; do
    z="$RUN/$out/$v-docread.zip"
    [ -f "$z" ] && { "$PY" scripts/s26/validate_pa3.py "$z" "$HANDOFF" > "${z%.zip}.validate.json" 2> "${z%.zip}.validate.err"; note "validate $v rc=$?"; }
  done
}
build build-docread EF2-cut-p4b LANG-thr-avg EF2-cut-avg
build build-docread FUSE4-thrC FUSE4w-thrC
note "docread chain done"
