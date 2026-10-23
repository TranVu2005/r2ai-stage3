#!/usr/bin/env bash
# Start the crawler inside tmux session "s37".
#   ./run.sh pilot        50 URLs spread over the queue -> pilot/   (read the numbers with ./status.sh pilot)
#   ./run.sh full         whole queue, round 1 -> data/             (resumes where it stopped)
#   ./run.sh retry 2      round 2: only URLs that ended as `failed` in data/log.jsonl
# Attach: tmux attach -t s37   (detach with Ctrl-b d).  Stop politely: tmux send-keys -t s37 C-c
# A 403/429/401/451 or challenge page writes <out>/STOP.json and the loop ends; read it, then delete it by hand to continue.
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-}"
case "$MODE" in
  pilot) OUT=pilot; ARGS=(--sample 50 --round 1) ;;
  full)  OUT=data;  ARGS=(--round 1) ;;
  retry) OUT=data;  ARGS=(--round "${2:?usage: run.sh retry <round>=2..}") ;;
  *) echo "usage: $0 pilot | full | retry <round>"; exit 2 ;;
esac
[ -s urls.tsv ] || { echo "urls.tsv missing"; exit 1; }
[ ! -e "$OUT/STOP.json" ] || { echo "$OUT/STOP.json exists:"; cat "$OUT/STOP.json"; exit 1; }
tmux has-session -t s37 2>/dev/null && { echo "tmux session s37 already running (tmux attach -t s37)"; exit 1; }
tmux new-session -d -s s37 "cd '$PWD' && for i in \$(seq 1 20); do
  .venv/bin/python crawler.py run --urls urls.tsv --out $OUT ${ARGS[*]} >> $OUT.run.log 2>&1; rc=\$?
  echo \"\$(date -u +%FT%TZ) crawler rc=\$rc\" >> $OUT.run.log
  [ \$rc -eq 0 ] || [ \$rc -eq 3 ] || [ \$rc -eq 4 ] && break
  sleep 60
done; echo finished; sleep 3600"
echo "started in tmux session s37 -> $OUT (log: $OUT.run.log)"
