#!/usr/bin/env bash
# FUSE4-free: calibrate t_c once (chunks = FUSE4-thrC = 175,548), build FUSE4-free-docread with it, validate; then the K4-free
# regression gate. Needs the Jina and C-max supplements (extra/jina-scores.parquet, extra/cmax-scores.parquet) and the docread chain done.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s36-fuse4"
HANDOFF="D:/GitHub/r2ai-stage3/out/runs/s27-pharmacity/handoff/cc-s27-pharmacity"
cd "$REPO" && export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1
until grep -q "docread chain done" "$RUN/docread.log" 2>/dev/null; do sleep 30; done
note() { echo "$(date -Is) $*" >> "$RUN/free.log"; }
note "calibrate start"
"$PY" -m r2ai.s36.build --variant Cal-FUSE4-free --chunk-target 175548 --out "$RUN/build-free" > "$RUN/cal-free.stdout" 2> "$RUN/cal-free.stderr"; note "calibrate rc=$?"
TC=$("$PY" - <<PYEOF
import json,re
text=open("$RUN/cal-free.stdout",encoding="utf-8").read()
for line in text.splitlines():
    if line.startswith("{") and "calibrated_not_built" in line:
        print(repr(json.loads(line)["t_c"])); break
PYEOF
)
note "t_c=$TC"
[ -n "$TC" ] && {
  "$PY" -m r2ai.s36.build --variant FUSE4-free --t-c "$TC" --out "$RUN/build-docread" > "$RUN/free-docread.stdout" 2> "$RUN/free-docread.stderr"; note "FUSE4-free rc=$?"
  z="$RUN/build-docread/FUSE4-free-docread.zip"
  [ -f "$z" ] && { "$PY" scripts/s26/validate_pa3.py "$z" "$HANDOFF" > "${z%.zip}.validate.json" 2> "${z%.zip}.validate.err"; note "validate FUSE4-free rc=$?"; }
}
note "free chain done"
