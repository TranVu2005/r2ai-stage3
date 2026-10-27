#!/usr/bin/env bash
# S40 (26/10): three ablations of 120A-free-v2, each drops exactly one component; everything else as configs/s36-handoff-120A-free-v2.json.
#   noJina / noCmax: apply with rrf_keys (3-source RRF), then S38 compose full+w768 --fit-carriers (T searched), validator S26.
#   nowin: the free-v2 docsel and carriers unchanged, compose write --threshold 1000000000 (full body only, no 768 window), validator S26.
# Steps are checkpointed (a step whose output exists is skipped), so the watchdog can simply rerun this script.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="D:/GitHub/r2ai-stage3/out/runs/s40/abl"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
REL="D:/GitHub/r2ai-stage3/out/runs/release/120A-free-v2"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN/apply" "$RUN/compose" "$RUN/nowin"
LOG="$RUN/abl.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
for TAG in noJina noCmax; do
  NAME="ABL-$TAG-free-v2"
  if [ ! -f "$RUN/apply/$NAME.apply.json" ]; then
    say "apply $NAME start"
    "$PY" -u -m r2ai.s36.s37 apply --config "configs/s40-abl-free-v2-$TAG.json" > "$RUN/apply-$NAME.log" 2>&1 || { say "apply $NAME FAILED"; exit 1; }
    say "apply $NAME done"
  fi
done
for TAG in noJina noCmax; do
  NAME="ABL-$TAG-free-v2"
  if ! ls "$RUN/compose/$NAME-full+w768"-t*.compose.json >/dev/null 2>&1 || ! grep -q '"validator"' "$RUN/compose/$NAME-full+w768"-t*.compose.json; then
    say "compose $NAME start"
    "$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$RUN/apply/docsel-$NAME.parquet" --rule full+w768 \
      --carriers "$RUN/apply/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose-$NAME.log" 2>&1 \
      || { say "compose $NAME FAILED"; exit 1; }
    say "compose $NAME done"
  fi
done
NAME="120A-free-v2-full+w768"
if [ ! -f "$RUN/nowin/$NAME-t1000000000.compose.json" ] || ! grep -q '"validator"' "$RUN/nowin/$NAME-t1000000000.compose.json"; then
  say "nowin start"
  for f in "$NAME.prepare.json" "$NAME.cache.parquet" "$NAME.base.parquet" "$NAME.swap.json"; do
    [ -f "$RUN/nowin/$f" ] || cp "$REL/compose/$f" "$RUN/nowin/$f"
  done
  "$PY" -u -m r2ai.s38.compose write --name "$NAME" --threshold 1000000000 --out "$RUN/nowin" >> "$RUN/compose-nowin.log" 2>&1 || { say "nowin FAILED"; exit 1; }
  say "nowin done"
fi
say "ALL DONE"
