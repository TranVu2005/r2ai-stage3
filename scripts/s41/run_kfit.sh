#!/usr/bin/env bash
# S41 (27/10, Vu, S42 hypothesis): FREE-Kfit = CH-rrf with the docs per query from a learned K_q (r2ai.s41.kfit). fit (CV report) ->
# build (refused when the best CV R2 < 0.2) -> S38 compose full+w768 as scripts/s41/run_ch.sh (--fit-carriers, T searched, validator S26).
# Waits for the rehearsal (RAM: "all done" / "FAILED" in out/runs/s41/rh100-final/rehearsal.log) unless NO_WAIT=1. Log out/runs/s41/kfit/kfit.log.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/kfit"
NAME=FREE-Kfit
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN/compose"
LOG="$RUN/kfit.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/kfit.pid"
RHLOG="D:/GitHub/r2ai-stage3/out/runs/s41/rh100-final/rehearsal.log"
until [ "${NO_WAIT:-0}" = 1 ] || { [ -f "$RHLOG" ] && grep -qE "all done|FAILED" "$RHLOG"; }; do sleep 60; done
say "fit start"
"$PY" -u -m r2ai.s41.kfit fit --out "$RUN" >> "$RUN/fit.out" 2>&1 || { say "fit FAILED"; exit 1; }
say "fit done: $(grep -o '"go": [a-z]*' "$RUN/fit.json")"
grep -q '"go": true' "$RUN/fit.json" || { say "CV R2 < 0.2: no build (Vu 27/10)"; say "all done"; exit 0; }
say "build start"
"$PY" -u -m r2ai.s41.kfit build --out "$RUN" --name "$NAME" >> "$RUN/build.out" 2>&1 || { say "build FAILED"; exit 1; }
say "compose $NAME start"
"$PY" -u -m r2ai.s38.compose run --name "$NAME-full+w768" --docsel "$RUN/apply/docsel-$NAME.parquet" --rule full+w768 \
  --carriers "$RUN/apply/carriers-$NAME.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose-$NAME.log" 2>&1 \
  || { say "compose $NAME FAILED"; exit 1; }
say "compose $NAME done"
say "all done"
