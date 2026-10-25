#!/usr/bin/env bash
# S38 composed bundle: docsel <NAME> (S36) + chunk rule -> ZIP + validator S26.
# Usage (from the S38 worktree): scripts/s38/compose.sh <DOCSEL> <RULE> [extra compose args]
#   DOCSEL: a docsel of out/runs/s36-fuse4/docsel (e.g. EF2-cut-avg); its S36 "-docread" ZIP is the whole-body base.
#   RULE:   clean | cap2048 | full+w768 | clean+cap2048   (full+w768 picks the lowest T under 102,600,000 B)
# Thin placeholder+base64 baophutho carriers pass their chunk on (global P4B order); --no-thin keeps them.
# Outputs in out/runs/s38-chunk-metric-v2/compose/<DOCSEL>-<RULE>*; a prepared cache is reused.
set -eu
DOCSEL="$1"; RULE="$2"; shift 2
PY="${PY:-D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="${R2AI_OUT_DIR:-D:/GitHub/r2ai-stage3/out}"
S36="$OUT_ROOT/runs/s36-fuse4"
export PYTHONPATH="$HERE/src" PYTHONIOENCODING=utf-8
"$PY" -u -m r2ai.s38.compose run --name "$DOCSEL-$RULE" \
  --base-zip "$S36/build-docread/$DOCSEL-docread.zip" --docsel "$S36/docsel/docsel-$DOCSEL.parquet" --rule "$RULE" "$@"
