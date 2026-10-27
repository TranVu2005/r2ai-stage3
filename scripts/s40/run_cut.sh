#!/usr/bin/env bash
# S40 (27/10): compose the EF2-cut probes FREE-cut150 / FREE-cut130 (apply: configs/s40-cut{150,130}.json, (a, b, s) of the 18-number LB
# refit, EF2 threshold x a global lambda solved for the mean docs per query), same S38 compose as the ablations (full+w768, --fit-carriers,
# T searched), validator S26. Checkpointed per probe: skipped when its validated compose exists. Log out/runs/s40/cut/cut.log.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="D:/GitHub/r2ai-stage3/out/runs/s40/cut"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
LOG="$RUN/cut.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/compose.pid"
mkdir -p "$RUN/compose"
for NAME in FREE-cut150 FREE-cut130; do
  if ls "$RUN/compose/$NAME-full+w768"-t*.compose.json >/dev/null 2>&1 && grep -q '"validator"' "$RUN/compose/$NAME-full+w768"-t*.compose.json; then
    say "compose $NAME skipped (validated compose exists)"; continue
  fi
  say "compose $NAME start"
  "$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$RUN/apply/docsel-$NAME.parquet" --rule full+w768 \
    --carriers "$RUN/apply/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose-$NAME.log" 2>&1 \
    || { say "compose $NAME FAILED"; exit 1; }
  say "compose $NAME done"
done
say "all done"
