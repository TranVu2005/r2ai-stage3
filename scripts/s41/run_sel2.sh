#!/usr/bin/env bash
# S41 (27/10): CH-sel2 (r2ai.s41.sel2) after the carrier probes (waits for "all done" / "FAILED" in out/runs/s41/ch/ch.log; Vu: lower
# priority than CH-rrf / CH-cmax / CH-ef2). Base FREE-noJina by default; when a carrier probe wins on the LB, run again with
#   NAME=CH-sel2-<probe> BASE_ZIP=out/runs/s41/ch/compose/<probe>-full+w768-t<T>.zip BASE_SHA=<sha> DOCSEL=<same docsel>
# Log out/runs/s41/sel2/sel2.log.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/sel2"
NAME="${NAME:-CH-sel2}"
BASE_ZIP="${BASE_ZIP:-D:/GitHub/r2ai-stage3/out/runs/s40/abl/compose/ABL-noJina-free-v2-full+w768-t1536.zip}"
BASE_SHA="${BASE_SHA:-153aba8b61c14e74b0ab2d26bc344111a8bdf1ef56d0e6d8850b03c97e3c918d}"
DOCSEL="${DOCSEL:-D:/GitHub/r2ai-stage3/out/runs/s40/abl/apply/docsel-ABL-noJina-free-v2.parquet}"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN"
LOG="$RUN/sel2.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/sel2.pid"
CHLOG="D:/GitHub/r2ai-stage3/out/runs/s41/ch/ch.log"
until [ ! -f "$CHLOG" ] || tail -1 "$CHLOG" | grep -qE "all done|FAILED"; do sleep 60; done
if [ -f "$RUN/$NAME.sel2.json" ] && grep -q '"ok": true' "$RUN/$NAME.sel2.json"; then say "$NAME skipped (validated)"; exit 0; fi
say "$NAME start (base $BASE_ZIP)"
"$PY" -u -m r2ai.s41.sel2 --name "$NAME" --base-zip "$BASE_ZIP" --base-sha "$BASE_SHA" --docsel "$DOCSEL" --out "$RUN" >> "$RUN/$NAME.log" 2>&1 \
  || { say "$NAME FAILED"; exit 1; }
say "$NAME done"
