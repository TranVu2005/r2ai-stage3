"""S40 C-max determinism probe (GPU, reads the S32 pool and scores ON PURPOSE; never an input of r2ai.newq).

Rescore rows of ONE finished S32 0.6B part with the pinned scorer (``judge_score.py``, Qwen3-Reranker-0.6B e61197ed):
  A  same part, same rows, same batching as S32 (bf16, 8192 tokens / 16 rows)  -> equals the stored S32 score? (same batches => same numbers?)
  B  same rows in a shuffled order (other batch neighbours), bf16                -> batch-composition effect
  C  bf16, batch size 1 (no padding)
  D  fp32, 8192 / 16
  E  fp32, batch size 1
Reports max / p99 |diff| against the stored S32 score and between the variants, and rows/s of each.
Usage: cmax_determinism.py [part_index] [rows]"""
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import paths  # noqa: E402

S32 = paths.RUNS_DIR / "s32-chunk-judge"
SNAP = "D:/GitHub/r2ai-stage3/data/hf_model_cache/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"
spec = importlib.util.spec_from_file_location("judge_score", Path(__file__).resolve().parents[2] / "src" / "r2ai" / "s31" / "judge_score.py")
js = importlib.util.module_from_spec(spec)
spec.loader.exec_module(js)

part_index = int(sys.argv[1]) if len(sys.argv) > 1 else 3
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
part = sorted(glob.glob(str(S32 / "gpu" / "qwen3-reranker-0.6b" / "parts" / "*.parquet")))[part_index]
stored = pd.read_parquet(part)
shard_name = Path(part).name.split(".part-")[0] + ".parquet"
rows = pd.read_parquet(S32 / "pool" / "shards" / shard_name)
frame = rows.set_index("pair_id").loc[stored.pair_id].reset_index()          # the part's rows in the part's order
want = stored.set_index("pair_id").score
log = js.Logger(Path(paths.RUNS_DIR / "s40" / "cmax-determinism.log"))


def run(dtype: str, tokens: int, size: int, order=None) -> tuple[pd.Series, float]:
    sc = SCORERS.setdefault(dtype, js.HFScorer(SNAP, None, dtype, log))
    f = frame if order is None else frame.iloc[order].reset_index(drop=True)
    f = f.head(limit) if order is None else f
    t0 = time.monotonic()
    res, _ = js.score_frame(sc, f, tokens, size, log)
    return res.set_index("pair_id").score, time.monotonic() - t0


SCORERS: dict = {}
out = {"part": Path(part).name, "rows": int(min(limit, len(frame)))}
a, ta = run("bfloat16", 8192, 16)
keep = a.index
perm = np.random.default_rng(40).permutation(len(frame))
b, tb = run("bfloat16", 8192, 16, perm)
c, tc = run("bfloat16", 512 * 4, 1)
d, td = run("float32", 8192, 16)
e, te = run("float32", 8192 * 4, 1)
variants = {"A_bf16_same_batches": a, "B_bf16_shuffled": b, "C_bf16_bs1": c, "D_fp32_b16": d, "E_fp32_bs1": e}
for name, s in variants.items():
    diff = (s.reindex(keep) - want.reindex(keep)).abs()
    out[name] = {"vs_s32_max": float(diff.max()), "vs_s32_p99": float(diff.quantile(.99)), "vs_s32_exact": float((diff == 0).mean())}
out["seconds"] = {"A": ta, "B": tb, "C": tc, "D": td, "E": te}
out["B_vs_A_max"] = float((b.reindex(keep) - a).abs().max())
out["C_vs_E_max"] = float((c.reindex(keep) - e.reindex(keep)).abs().max())
out["D_vs_E_max"] = float((d.reindex(keep) - e.reindex(keep)).abs().max())
print(json.dumps(out, indent=1))
(paths.RUNS_DIR / "s40" / "cmax-determinism.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
