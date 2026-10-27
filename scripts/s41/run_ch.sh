#!/usr/bin/env bash
# S41 (27/10): chunk-carrier probes on the FREE-noJina docsel (docs unchanged), carriers from r2ai.s41.carriers (out/runs/s41/carriers),
# same S38 compose as FREE-noJina (full+w768, --fit-carriers, T searched), validator S26, one after the other (RAM). Starts after the D400
# compose (waits for "all done" / "FAILED" in out/runs/s41/d400/d400.log). Checkpointed per build. Log out/runs/s41/ch/ch.log.
#   CH-rrf   CR-fair (rho of FREE-noJina) in docsel RRF order instead of P4B
#   CH-cmax  CR-fair in C-max order
#   CH-ef2   per-query carrier count maximising expected chunk F2 (RRF order), same total
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/ch"
DOCSEL="D:/GitHub/r2ai-stage3/out/runs/s40/abl/apply/docsel-ABL-noJina-free-v2.parquet"
CARR="D:/GitHub/r2ai-stage3/out/runs/s41/carriers"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN/compose"
LOG="$RUN/ch.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/ch.pid"
D400LOG="D:/GitHub/r2ai-stage3/out/runs/s41/d400/d400.log"
until [ ! -f "$D400LOG" ] || tail -1 "$D400LOG" | grep -qE "all done|FAILED"; do sleep 60; done
say "start (D400 compose finished)"
for NAME in ${NAMES:-CH-rrf CH-cmax CH-ef2}; do
  if ls "$RUN/compose/$NAME-full+w768"-t*.compose.json >/dev/null 2>&1 && grep -q '"ok": true' "$RUN/compose/$NAME-full+w768"-t*.compose.json; then
    say "compose $NAME skipped (validated compose exists)"; continue
  fi
  say "compose $NAME start"
  "$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$DOCSEL" --rule full+w768 \
    --carriers "$CARR/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose-$NAME.log" 2>&1 \
    || { say "compose $NAME FAILED"; exit 1; }
  say "compose $NAME done"
done
say "all done"
