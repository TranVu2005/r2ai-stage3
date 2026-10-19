#!/usr/bin/env bash
# S26 P-A4: build + validate one variant over the full canonical doc vectors (waits for full_embed.py to finish).
# Usage: pa4_run.sh <name> [pa3_build options...]   e.g. pa4_run.sh P-A4-K90 nc k=0.9
set -u
S=${S26_SCRATCH:?set S26_SCRATCH to the dir holding q_dense.npy / q_ids.npy}
O=${R2AI_ROOT:-D:/GitHub/r2ai-stage3}/out/runs/s26-gold-structure
name=$1; shift
until [ -f "$O/full/full_doc_dense.npz" ] && grep -q "^done " "$O/full/embed.log"; do
  grep -q "Traceback" "$O/full/embed.err" && { echo "embed job failed"; tail -5 "$O/full/embed.err"; exit 1; }
  sleep 60
done
mkdir -p "$O/pa4"
python -u "$(dirname "$0")/pa3_build.py" "$O/pa4" "$S" "$O/full/full_doc_dense.npz" "$name" pa3 "$@" > "$O/pa4/$name.log" 2>&1
tail -45 "$O/pa4/$name.log"
[ -f "$O/pa4/$name.zip" ] && python -u "$(dirname "$0")/validate_pa3.py" "$O/pa4/$name.zip" > "$O/pa4/$name.validate.json" 2>&1
grep -v '": 0' "$O/pa4/$name.validate.json"
