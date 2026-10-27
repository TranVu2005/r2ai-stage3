#!/usr/bin/env bash
# S40 (26/10): compose FREE-noJina-recal (apply: configs/s40-recal-noJina.json, (a, b, s_free) refitted by scripts/s40/refit_pcal.py),
# same S38 compose as the ablations (full+w768, --fit-carriers, T searched), validator S26. Checkpointed: skipped when the validated compose exists.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="D:/GitHub/r2ai-stage3/out/runs/s40/recal"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
NAME="FREE-noJina-recal"
LOG="$RUN/recal.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/compose.pid"
if ! ls "$RUN/compose/$NAME-full+w768"-t*.compose.json >/dev/null 2>&1 || ! grep -q '"validator"' "$RUN/compose/$NAME-full+w768"-t*.compose.json; then
  say "compose $NAME start"
  "$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$RUN/apply/docsel-$NAME.parquet" --rule full+w768 \
    --carriers "$RUN/apply/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose.log" 2>&1 \
    || { say "compose $NAME FAILED"; exit 1; }
  say "compose $NAME done"
fi
