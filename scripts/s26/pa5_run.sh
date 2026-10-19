#!/usr/bin/env bash
# S26 P-A5: build + validate the K90-based selection variants (Ku / cos / rel), sequentially.
set -u
S=${S26_SCRATCH:?set S26_SCRATCH to the dir holding q_dense.npy / q_ids.npy}
O=${R2AI_ROOT:-D:/GitHub/r2ai-stage3}/out/runs/s26-gold-structure
T=162.6225                                   # mean docs/query of P-A4-K90
REF=$O/pa4/P-A4-K90.zip
mkdir -p "$O/pa5"
for v in "P-A5-Ku ku" "P-A5-cos cos" "P-A5-rel rel"; do
  set -- $v
  python -u "$(dirname "$0")/pa3_build.py" "$O/pa5" "$S" "$O/full/full_doc_dense.npz" "$1" pa3 nc new=body sel=$2 tmean=$T ref=$REF > "$O/pa5/$1.log" 2>&1
  [ -f "$O/pa5/$1.zip" ] && python -u "$(dirname "$0")/validate_pa3.py" "$O/pa5/$1.zip" > "$O/pa5/$1.validate.json" 2>&1
  echo "== $1"; grep "^sel " "$O/pa5/$1.log"; grep '"ok"' "$O/pa5/$1.validate.json"
done
