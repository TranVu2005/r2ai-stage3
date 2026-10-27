#!/usr/bin/env bash
# S40 watchdog: rerun a checkpointed driver until its log says ALL DONE (at most MAX tries). Usage: watchdog.sh <driver.sh> <log> [MAX]
set -u
DRV="$1"; LOG="$2"; MAX="${3:-3}"
echo $$ > "$LOG.watchdog.pid"
for i in $(seq 1 "$MAX"); do
  echo "$(date -Is) watchdog try $i/$MAX: $DRV" >> "$LOG"
  bash "$DRV" & echo $! > "$LOG.driver.pid"; wait $!
  rc=$?
  tail -1 "$LOG" | grep -q "ALL DONE" && { echo "$(date -Is) watchdog: done" >> "$LOG"; exit 0; }
  echo "$(date -Is) watchdog: driver rc=$rc, DIED (try $i)" >> "$LOG"
done
echo "$(date -Is) watchdog: GAVE UP" >> "$LOG"; exit 1
