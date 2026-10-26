"""S40 Kaggle / RunPod 4B regression gate (Vu 26/10): the same pairs scored on both machines (same scorer 688948a7, revision 22e68366,
fp16, 512-token Qwen3 truncation). Pass: Spearman >= 0.99 over the common pair_ids (and every pair of the smaller file is common).
Reports mean / p99 / max |delta|. Fail -> score the RunPod half on Kaggle as well (Kaggle-only plan).

Usage: gate_4b.py <kaggle scores-extra.parquet> <runpod scores-extra.parquet> [out.json]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

a, b = pd.read_parquet(sys.argv[1], columns=["pair_id", "score"]), pd.read_parquet(sys.argv[2], columns=["pair_id", "score"])
j = a.merge(b, on="pair_id", suffixes=("_kaggle", "_runpod"))
d = (j.score_kaggle - j.score_runpod).abs()
res = {"pairs_kaggle": len(a), "pairs_runpod": len(b), "common": len(j), "spearman": float(j.score_kaggle.corr(j.score_runpod, method="spearman")),
       "pearson": float(j.score_kaggle.corr(j.score_runpod)), "abs_delta_mean": float(d.mean()), "abs_delta_p99": float(d.quantile(.99)),
       "abs_delta_max": float(d.max()), "exact_share": float((d == 0).mean()), "abs_delta_gt_0.01": int((d > .01).sum())}
res["pass"] = bool(res["common"] == min(len(a), len(b)) and res["common"] > 0 and res["spearman"] >= 0.99 and np.isfinite(d).all())
print(json.dumps(res, indent=1))
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
sys.exit(0 if res["pass"] else 1)
