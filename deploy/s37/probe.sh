#!/usr/bin/env bash
# Light availability probe for www.120ask.com: ONE request per 30 min, append "timestamp http_code latency_s" to probe.log.
# Never starts the crawler. Stops itself (and writes probe.UP) the first time the site answers 200, so a human decides what next.
#   nohup ./probe.sh > /dev/null 2>&1 &        stop early: pkill -f probe.sh
cd "$(dirname "$0")"
URL="https://www.120ask.com/question/60917294.htm"
UA="r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)"
while true; do
  out=$(curl -s -o /dev/null -m 20 -A "$UA" -w '%{http_code} %{time_total}' "$URL" || true)
  echo "$(date -Is) ${out:-000 0}" >> probe.log
  case "$out" in 200\ *) date -Is > probe.UP; exit 0 ;; esac
  sleep 1800
done
