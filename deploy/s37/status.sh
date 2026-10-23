#!/usr/bin/env bash
# Progress: done / ok / suspect / failed / ETA.   ./status.sh [data|pilot]
cd "$(dirname "$0")"
OUT="${1:-data}"
tmux has-session -t s37 2>/dev/null && echo "tmux s37: running" || echo "tmux s37: not running"
[ -e "$OUT/STOP.json" ] && { echo "!! STOPPED:"; cat "$OUT/STOP.json"; }
.venv/bin/python crawler.py status --urls urls.tsv --out "$OUT"
[ -e "$OUT.run.log" ] && { echo "--- last log lines"; tail -n 3 "$OUT.run.log"; }
[ -d "$OUT/batches" ] && echo "batches packed: $(ls "$OUT"/batches/*.tar 2>/dev/null | wc -l)"
exit 0
