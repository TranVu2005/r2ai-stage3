"""Regression of the 4B scorer with the ``en`` instruction line (S35, 25/10).

Re-scores 200 vi/zh pairs that already have a 4B score (S31 pool-extra) with BOTH the pinned scorer (SHA 688948a7...) and the new one
(``r2ai.s36.enlane scorer``), on the same machine with the same settings, and compares: the vi/zh path must not change (|diff| <= 1e-3, in
practice 0). The score of the old Kaggle run is reported too (it differs by hardware numerics: S31 measured local-vs-Kaggle max 0.0054).

    PYTHONPATH=src python scripts/s36/scorer_regression.py prepare     # 200-pair package (seed 20261025: 140 vi + 60 zh)
    PYTHONPATH=src python scripts/s36/scorer_regression.py run         # old then new scorer (GPU, ~2 x 2 min with 4 GiB offload)
    PYTHONPATH=src python scripts/s36/scorer_regression.py compare     # writes regression.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json
from r2ai.s36 import enlane, fuse

REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
S31 = paths.RUNS_DIR / "s31-strong-judge"
ROOT = fuse.run_root() / "en" / "regression"
N_VI, N_ZH, SEED = 140, 60, 20261025
TOL = 1e-3


def prepare() -> dict:
    shard = pd.read_parquet(S31 / "pool-extra" / "kaggle" / "shards" / "pool-91000.parquet")
    ref = pd.read_parquet(S31 / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet", columns=["pair_id", "score"])
    shard = shard.merge(ref, on="pair_id", how="inner")
    pick = pd.concat([shard[shard.lang == "vi"].sample(N_VI, random_state=SEED), shard[shard.lang == "zh"].sample(N_ZH, random_state=SEED)])
    pick = pick.sort_values("pair_id").reset_index(drop=True)
    pkg = ROOT / "pkg"
    paths.assert_writable(pkg)
    (pkg / "shards").mkdir(parents=True, exist_ok=True)
    pick.drop(columns="score").to_parquet(pkg / "shards" / "pool-91000.parquet", index=False)
    pick[["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score"]].to_parquet(pkg / "extra_pairs.parquet", index=False)
    pick[["pair_id", "score"]].to_parquet(ROOT / "reference.parquet", index=False)
    return {"pairs": int(len(pick)), "lang": pick.lang.value_counts().to_dict(), "pkg": str(pkg)}


def run() -> dict:
    new = Path(json.loads((enlane.out_dir() / "scorer" / "scorer.json").read_text(encoding="utf-8"))["path"])
    done = {}
    for tag, script in (("old", enlane.JUDGE_SCORE), ("new", new)):
        out = ROOT / tag
        paths.assert_writable(out)
        common = ["--pool", str(ROOT / "pkg"), "--out", str(out), "--pairs-file", str(ROOT / "pkg" / "extra_pairs.parquet"), "--subset", "extra",
                  "--model", "qwen3-reranker-4b"]
        score = [sys.executable, str(script), "score", *common, "--backend", "hf", "--dtype", "float16", "--max-batch-tokens", "4096", "--revision", REVISION,
                 "--device-map", "auto", "--max-gpu-memory", "4GiB", "--worker", "0", "--workers", "1"]
        subprocess.run(score, check=True)
        subprocess.run([sys.executable, str(script), "collect", *common], check=True)
        done[tag] = str(out / "qwen3-reranker-4b" / "scores-extra.parquet")
    return done


def compare() -> dict:
    ref = pd.read_parquet(ROOT / "reference.parquet").rename(columns={"score": "kaggle"})
    old = pd.read_parquet(ROOT / "old" / "qwen3-reranker-4b" / "scores-extra.parquet", columns=["pair_id", "score"]).rename(columns={"score": "old"})
    new = pd.read_parquet(ROOT / "new" / "qwen3-reranker-4b" / "scores-extra.parquet", columns=["pair_id", "score"]).rename(columns={"score": "new"})
    both = ref.merge(old, on="pair_id").merge(new, on="pair_id")
    d_new, d_kaggle = (both.new - both.old).abs(), (both.new - both.kaggle).abs()
    result = {"pairs": int(len(both)), "new_vs_old_abs_max": float(d_new.max()), "new_vs_old_abs_mean": float(d_new.mean()), "tolerance": TOL,
              "ok": bool(len(both) == N_VI + N_ZH and d_new.max() <= TOL), "new_vs_kaggle_abs_max": float(d_kaggle.max()),
              "new_vs_kaggle_abs_mean": float(d_kaggle.mean()), "spearman_new_vs_kaggle": float(both.new.corr(both.kaggle, method="spearman")),
              "scorer_new_sha256": json.loads((enlane.out_dir() / "scorer" / "scorer.json").read_text(encoding="utf-8"))["sha256"]}
    _atomic_json(ROOT / "regression.json", result)
    return result


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("prepare", "run", "compare"):
        raise SystemExit(__doc__)
    print(json.dumps({"prepare": prepare, "run": run, "compare": compare}[cmd](), indent=1, ensure_ascii=False))
