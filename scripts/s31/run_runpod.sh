#!/usr/bin/env bash
# S31: score the strong-judge pool with Qwen3-Reranker-8B on a RunPod Linux pod with one 24 GB GPU.
#
# Every stage is resumable: rerun the script after a disconnect and it skips what is done
# (finished parts are kept under $OUT/$MODEL/parts). No token or other secret is needed or read:
# the models are public on Hugging Face.
#
# Stages
#   1. venv with vLLM (pinned, override with VLLM_VERSION), pip freeze recorded
#   2. download the 0.6B model (regression gate) and $MODEL; the resolved commit is pinned in $OUT
#   3. gate: 0.6B hf on 1,000 pool13 pairs vs S29 scores, Spearman >= 0.99 or stop
#   4. pilot $MODEL on the same 1,000 pairs with hf and with vllm;
#      Spearman(hf, vllm) >= 0.99 -> score with vllm, else with hf
#   5. score --subset pool13, collect pool13 (enough for J8-rrf / J8-rrf3)
#   6. only with SUBSETS="pool13 tail": score the 1.3x-2.0x tail, collect tail and all
#      (default skips it: the wider pool did not help on LB, decision 21/10)
#   7. tar the results for download
#
# Usage on the pod (see docs/s31-strong-judge.md for the full steps):
#   tmux new -s s31
#   bash /workspace/pool/run_runpod.sh 2>&1 | tee -a /workspace/s31-run.log
set -euo pipefail

POOL=${POOL:-/workspace/pool}
OUT=${OUT:-/workspace/s31-out}
MODEL=${MODEL:-qwen3-reranker-8b}
HF_ID=${HF_ID:-Qwen/Qwen3-Reranker-8B}
REVISION=${REVISION:-}                 # empty: resolve once, then reuse $OUT/$MODEL.revision
VLLM_VERSION=${VLLM_VERSION:-0.11.0}
VENV=${VENV:-/workspace/venv-s31}
PILOT_PAIRS=${PILOT_PAIRS:-1000}
HF_BATCH_TOKENS=${HF_BATCH_TOKENS:-8192}
SUBSETS=${SUBSETS:-pool13}
export HF_HOME=${HF_HOME:-/workspace/hf}
export TOKENIZERS_PARALLELISM=false

SCRIPT="$POOL/judge_score.py"
REF="$POOL/s29_qwen06_scores.parquet"
GATE_REVISION=e61197ed45024b0ed8a2d74b80b4d909f1255473
mkdir -p "$OUT"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
for f in "$SCRIPT" "$REF" "$POOL/manifest.json"; do
  [ -f "$f" ] || { echo "missing $f (upload the pool directory to $POOL first)"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"

# 1. environment
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv $VENV with vllm==$VLLM_VERSION"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "vllm==$VLLM_VERSION" pandas pyarrow
fi
PY="$VENV/bin/python"
"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
"$PY" -c "import torch, transformers, vllm; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)" | tee "$OUT/versions.txt"

# 2. models (public, no token)
if [ -z "$REVISION" ] && [ -f "$OUT/$MODEL.revision" ]; then REVISION=$(cat "$OUT/$MODEL.revision"); fi
REVISION=$("$PY" - "$HF_ID" "$REVISION" "$GATE_REVISION" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
hf_id, revision, gate = sys.argv[1], sys.argv[2] or None, sys.argv[3]
snapshot_download(repo_id="Qwen/Qwen3-Reranker-0.6B", revision=gate)
print(Path(snapshot_download(repo_id=hf_id, revision=revision)).name)
PYEOF
)
echo "$REVISION" > "$OUT/$MODEL.revision"
log "$HF_ID pinned at $REVISION"

# 3. regression gate: 0.6B vs S29
GATE="$OUT/qwen3-reranker-0.6b/pilot-$PILOT_PAIRS-hf-bfloat16.parquet"
if [ ! -f "$GATE" ]; then
  "$PY" "$SCRIPT" pilot --pool "$POOL" --out "$OUT" --model qwen3-reranker-0.6b --pairs "$PILOT_PAIRS" \
    --dtype bfloat16 --max-batch-tokens "$HF_BATCH_TOKENS"
fi
"$PY" - "$GATE" "$REF" "$OUT/gate-0.6b.json" <<'PYEOF'
import json, sys
import pandas as pd
ours, ref = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
j = ours.merge(ref[["pair_id", "qwen_score"]], on="pair_id")
rho = float(j.score.corr(j.qwen_score, method="spearman"))
res = {"pairs": len(j), "spearman_vs_s29": rho, "pass": len(j) == len(ours) and rho >= 0.99}
json.dump(res, open(sys.argv[3], "w"), indent=2)
print(res)
sys.exit(0 if res["pass"] else 1)
PYEOF

# 4. pilot hf vs vllm on the same pairs
HF_PILOT="$OUT/$MODEL/pilot-$PILOT_PAIRS-hf-bfloat16"
VLLM_PILOT="$OUT/$MODEL/pilot-$PILOT_PAIRS-vllm-bfloat16"
if [ ! -f "$HF_PILOT.parquet" ]; then
  "$PY" "$SCRIPT" pilot --pool "$POOL" --out "$OUT" --model "$MODEL" --revision "$REVISION" --pairs "$PILOT_PAIRS" \
    --backend hf --dtype bfloat16 --max-batch-tokens "$HF_BATCH_TOKENS"
fi
if [ ! -f "$VLLM_PILOT.parquet" ]; then
  "$PY" "$SCRIPT" pilot --pool "$POOL" --out "$OUT" --model "$MODEL" --revision "$REVISION" --pairs "$PILOT_PAIRS" \
    --backend vllm --dtype bfloat16 --max-batch-size 2048 --max-batch-tokens 100000000 || log "vllm pilot failed; hf only"
fi
BACKEND=$("$PY" - "$HF_PILOT" "$VLLM_PILOT" "$OUT/backend-choice.json" <<'PYEOF'
import json, os, sys
import pandas as pd
hf, vl, target = sys.argv[1:4]
res = {"hf": json.load(open(hf + ".json"))["pairs_per_second"], "backend": "hf"}
if os.path.exists(vl + ".parquet"):
    j = pd.read_parquet(hf + ".parquet").merge(pd.read_parquet(vl + ".parquet"), on="pair_id", suffixes=("_hf", "_vllm"))
    res["vllm"] = json.load(open(vl + ".json"))["pairs_per_second"]
    res["spearman_hf_vllm"] = float(j.score_hf.corr(j.score_vllm, method="spearman"))
    res["pairs"] = len(j)
    if res["spearman_hf_vllm"] >= 0.99:
        res["backend"] = "vllm"
json.dump(res, open(target, "w"), indent=2)
print(json.dumps(res), file=sys.stderr)
print(res["backend"])
PYEOF
)
log "scoring backend: $BACKEND"
if [ "$BACKEND" = vllm ]; then
  BATCH=(--backend vllm --dtype bfloat16 --max-batch-size 2048 --max-batch-tokens 100000000)
else
  BATCH=(--backend hf --dtype bfloat16 --max-batch-tokens "$HF_BATCH_TOKENS")
fi

# 5-6. pool13 (and the tail only if SUBSETS lists it)
for SUBSET in $SUBSETS; do
  log "scoring subset $SUBSET"
  "$PY" "$SCRIPT" score --pool "$POOL" --out "$OUT" --model "$MODEL" --revision "$REVISION" --subset "$SUBSET" "${BATCH[@]}"
  "$PY" "$SCRIPT" collect --pool "$POOL" --out "$OUT" --model "$MODEL" --subset "$SUBSET"
done
if "$PY" "$SCRIPT" status --pool "$POOL" --out "$OUT" --model "$MODEL" --subset tail | grep -q '"complete": true'; then
  "$PY" "$SCRIPT" collect --pool "$POOL" --out "$OUT" --model "$MODEL" --subset all
fi

# 7. results for download (scores, parts, reports, logs; no model weights)
TAR="/workspace/s31-$MODEL-results.tar.gz"
tar -czf "$TAR" -C "$OUT" .
ls -l "$TAR"
log "done; download $TAR (see docs/s31-strong-judge.md), then stop the pod"
