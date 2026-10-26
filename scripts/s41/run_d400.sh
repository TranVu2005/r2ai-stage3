#!/usr/bin/env bash
# S41 (27/10): compose the D = 400 builds (4B D400 imported from Kaggle, C-max D400 of S40), same S38 compose as the S40 ablations
# (full+w768, --fit-carriers, T searched), validator S26, one after the other (RAM).
#   FREE-D400-cut150lam  configs/s41-d400-cut150lam.json: (a, b, s) of the 18-number refit, lambda of FREE-cut150 (depth effect at a fixed rule)
#   FREE-noJina-D400     configs/s40-d400-noJina.json: FREE-noJina at depth 400 (old a, b, s)
# Checkpointed: a build whose validated compose exists is skipped. Log out/runs/s41/d400/d400.log.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
RUN="D:/GitHub/r2ai-stage3/out/runs/s41/d400"
cd "$WT" && export PYTHONPATH="$WT/src" PYTHONIOENCODING=utf-8
mkdir -p "$RUN/compose"
LOG="$RUN/d400.log"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
echo $$ > "$RUN/compose.pid"
build() {  # name apply-dir
  if ls "$RUN/compose/$1-full+w768"-t*.compose.json >/dev/null 2>&1 && grep -q '"ok": true' "$RUN/compose/$1-full+w768"-t*.compose.json; then
    say "compose $1 skipped (validated compose exists)"; return 0
  fi
  say "compose $1 start"
  "$PY" -u -m r2ai.s38.compose run --name "$1-full+w768" --docsel "$2/docsel-$1.parquet" --rule full+w768 \
    --carriers "$2/carriers-$1.parquet" --fit-carriers --rounds "${ROUNDS:-10}" --out "$RUN/compose" >> "$RUN/compose-$1.log" 2>&1 \
    || { say "compose $1 FAILED"; exit 1; }
  say "compose $1 done"
}
[ "${WITH_CUT150LAM:-0}" = 1 ] && build FREE-D400-cut150lam "$RUN/apply"   # off: Vu 27/10 (cut series stopped, private keeps the old calibration)
build FREE-noJina-D400 "D:/GitHub/r2ai-stage3/out/runs/s40/d400/apply"
say "all done"
