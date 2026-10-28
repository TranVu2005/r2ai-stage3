#!/usr/bin/env bash
# S44: hold non-KEEP paths, rebuild rh100 from tag 28c, always restore. Usage: run_check.sh <verify out dir>
set -u
S="$(cd "$(dirname "$0")" && pwd)"
PYH=python
echo "$(date -Is) hold start" >> "$S/check.log"
$PYH -I -X utf8 "$S/hold.py" hold >> "$S/check.log" 2>&1 || { echo "$(date -Is) hold FAILED, restoring" >> "$S/check.log"; $PYH -I -X utf8 "$S/hold.py" restore >> "$S/check.log" 2>&1; exit 1; }
trap '$PYH -I -X utf8 "$S/hold.py" restore >> "$S/check.log" 2>&1; echo "$(date -Is) restore rc=$?" >> "$S/check.log"' EXIT
bash "$S/verify_rh100.sh" "$1" >> "$S/check.log" 2>&1
echo "$(date -Is) verify rc=$?" >> "$S/check.log"
