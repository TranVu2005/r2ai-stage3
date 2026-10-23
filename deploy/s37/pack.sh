#!/usr/bin/env bash
# Pack finished shards into batches of 5 shards (~10k URLs) -> data/batches/batch-NNNN.tar (+ .sha256).
#   ./pack.sh          only full batches (safe while the crawler runs)
#   ./pack.sh --final  also the leftover partial batch (after the crawl ends)
# Shards are already gzip: tar only bundles them with log.jsonl and MANIFEST.json (no recompression).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python crawler.py pack --urls urls.tsv --out data --batch-shards 5 "$@"
ls -lh data/batches/ 2>/dev/null || true
