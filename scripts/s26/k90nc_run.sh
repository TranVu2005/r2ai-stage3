#!/usr/bin/env bash
# S26: K90 doc set with more chunks per query (n_c x <mult>, at most K_q). Usage: k90nc_run.sh <name> <ncx> [extra opts]
set -u
S=${S26_SCRATCH:?set S26_SCRATCH to the dir holding q_dense.npy / q_ids.npy}
O=${R2AI_ROOT:-D:/GitHub/r2ai-stage3}/out/runs/s26-gold-structure
name=$1; ncx=$2; shift 2
mkdir -p "$O/pa6"
python -u "$(dirname "$0")/pa3_build.py" "$O/pa6" "$S" "$O/full/full_doc_dense.npz" "$name" pa3 nc new=body k=0.9 ncx=$ncx \
  ref=$O/pa4/P-A4-K90.zip "$@" > "$O/pa6/$name.log" 2>&1
grep "N=" "$O/pa6/$name.log"
[ -f "$O/pa6/$name.zip" ] && python -u "$(dirname "$0")/validate_pa3.py" "$O/pa6/$name.zip" > "$O/pa6/$name.validate.json" 2>&1
grep '"ok"' "$O/pa6/$name.validate.json"
