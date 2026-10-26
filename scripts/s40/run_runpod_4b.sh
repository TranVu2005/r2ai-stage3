#!/usr/bin/env bash
# S40: Qwen3-Reranker-4B on a RunPod Linux pod (one RTX 4090, 24 GB) for a subset exported by r2ai.newq, with the SAME scorer and settings
# as the Kaggle notebook notebooks/s36-kaggle-4b-final.ipynb: judge_score.py SHA 688948a7..., model commit 22e683669bc0f0bd69640a1354a6d0aebcfeede5,
# backend hf, dtype float16, max 16,384 padded tokens per batch, documents already cut to 512 Qwen3 tokens in the shard (export-scoring).
#
# Resumable: rerun after a disconnect; finished parts under $OUT/<tag>/qwen3-reranker-4b/parts are kept (judge_score.py skips them).
# No token or other secret is needed or read (the model is public). Nothing outside $WORK is written.
#
# Usage on the pod (upload the dataset folder first, e.g. runpodctl send / scp):
#   tmux new -s s40
#   DATASET=/workspace/kaggle-dataset-private-4b-2 bash run_runpod_4b.sh 2>&1 | tee -a /workspace/s40-runpod.log
# DATASET holds judge_score.py and one directory per subset (manifest.json, extra_pairs.parquet, shards/pool-*.parquet); TAGS defaults to
# every such directory, regression subset (-reg) first.
#
# Result: /workspace/s40-<tag>-runpod.tar.gz per subset with <tag>/qwen3-reranker-4b/{scores-extra.parquet, collect-extra-report.json, parts/},
# the same files the Kaggle notebook writes, for `python -m r2ai.newq import-scores --source 4b --tag <tag> --scores ... --report ...`
# (a -reg or sample subset is compared with scripts/s40/gate_4b.py instead of imported).
set -euo pipefail

DATASET=${DATASET:?set DATASET to the uploaded kaggle-dataset-<tag> or sample-<suffix> folder}
WORK=${WORK:-/workspace}
OUT=${OUT:-$WORK/s40-out}
VENV=${VENV:-$WORK/venv-s40}
MODEL=qwen3-reranker-4b
HF_ID=Qwen/Qwen3-Reranker-4B
REVISION=22e683669bc0f0bd69640a1354a6d0aebcfeede5
JUDGE_SHA=688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186
DTYPE=float16
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-16384}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-}   # empty: latest; set it to the version the Kaggle notebook printed (cell 2) to match Kaggle
export HF_HOME=${HF_HOME:-$WORK/hf}
export TOKENIZERS_PARALLELISM=false
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

SCRIPT="$DATASET/judge_score.py"
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT"; exit 1; }
[ "$(sha256sum "$SCRIPT" | cut -d' ' -f1)" = "$JUDGE_SHA" ] || { echo "judge_score.py is not the pinned scorer $JUDGE_SHA"; exit 1; }
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"

# 1. environment (versions recorded with the results)
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch "transformers${TRANSFORMERS_VERSION:+==$TRANSFORMERS_VERSION}" pandas pyarrow huggingface_hub packaging
fi
PY="$VENV/bin/python"
"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
"$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.version.cuda)" | tee "$OUT/versions.txt"

# 2. model at the pinned commit (public, no token)
"$PY" - "$HF_ID" "$REVISION" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
assert Path(path).name == sys.argv[2], (path, sys.argv[2])
print("snapshot", path)
PYEOF

# 3. subsets: verify the manifest SHA256s exactly as the notebook does, then score, collect, pack
TAGS=${TAGS:-$(cd "$DATASET" && ls -d */ | tr -d / | awk '{print (/-reg$/ ? 0 : 1) "\t" $0}' | sort | cut -f2)}
for TAG in $TAGS; do
  D="$DATASET/$TAG"
  "$PY" - "$D" "$REVISION" <<'PYEOF'
import glob, hashlib, json, sys
d, rev = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = json.load(open(f"{d}/manifest.json"))
assert man["revision"] == rev, man["revision"]
assert sha(f"{d}/extra_pairs.parquet") == man["extra_pairs_sha256"], "extra_pairs.parquet SHA mismatch"
shard = glob.glob(f"{d}/shards/pool-*.parquet")
assert len(shard) == 1 and sha(shard[0]) == man["shard_sha256"], "shard SHA mismatch"
print(man["tag"], man["pairs"], "pairs ok")
PYEOF
  ARGS=(--pool "$D" --out "$OUT/$TAG" --pairs-file "$D/extra_pairs.parquet" --subset extra --model "$MODEL")
  mkdir -p "$OUT/$TAG"
  T0=$(date +%s)
  log "scoring $TAG"
  "$PY" "$SCRIPT" score "${ARGS[@]}" --backend hf --dtype "$DTYPE" --max-batch-tokens "$MAX_BATCH_TOKENS" --revision "$REVISION" \
    --worker 0 --workers 1 2>&1 | tee -a "$OUT/$TAG/worker-0.stdout"
  "$PY" "$SCRIPT" status "${ARGS[@]}" | tail -1 | tee "$OUT/$TAG/status.json"
  grep -q '"complete": true' "$OUT/$TAG/status.json" || { log "$TAG INCOMPLETE: rerun this script to resume"; exit 2; }
  "$PY" "$SCRIPT" collect "${ARGS[@]}"
  echo "{\"tag\": \"$TAG\", \"wall_seconds\": $(( $(date +%s) - T0 )), \"gpu\": \"$(head -2 "$OUT/gpu.txt" | tail -1)\"}" > "$OUT/$TAG/runpod-timing.json"
  cp "$OUT/versions.txt" "$OUT/pip-freeze.txt" "$OUT/gpu.txt" "$OUT/$TAG/"
  tar -czf "$WORK/s40-$TAG-runpod.tar.gz" -C "$OUT" "$TAG"
  log "done $TAG: $WORK/s40-$TAG-runpod.tar.gz"
done
log "all subsets done; download the tar files, then stop the pod"
