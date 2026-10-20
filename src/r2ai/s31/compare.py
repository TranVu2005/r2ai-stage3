"""S31 checks: S29 reference scores keyed by pool pair_id, and the 0.6B regression gate.

``python -m r2ai.s31.compare ref`` writes ``pool/s29_qwen06_scores.parquet`` (S29
Qwen3-Reranker-0.6B scores for every 1.3x pair, keyed by S31 pair_id) so a GPU host
can run the regression gate without the SQLite checkpoint.
``python -m r2ai.s31.compare regress --scores PILOT.parquet`` compares a pilot run
with that reference (gate: Spearman >= 0.99).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256

REGRESSION_MIN_SPEARMAN = 0.99


def regression_stats(ours: pd.DataFrame, ref: pd.DataFrame) -> dict[str, Any]:
    """Join on pair_id; Spearman/Pearson of ``score`` vs reference ``qwen_score``."""
    joined = ours[["pair_id", "score"]].merge(ref[["pair_id", "qwen_score"]], on="pair_id", how="inner")
    if len(joined) != len(ours):
        raise ValueError(f"{len(ours) - len(joined)} pilot pairs have no reference score")
    diff = (joined.score - joined.qwen_score).abs()
    spearman = float(joined.score.corr(joined.qwen_score, method="spearman"))
    return {
        "pairs": len(joined),
        "spearman": spearman,
        "pearson": float(joined.score.corr(joined.qwen_score)),
        "abs_diff_mean": float(diff.mean()),
        "abs_diff_max": float(diff.max()),
        "gate_min_spearman": REGRESSION_MIN_SPEARMAN,
        "gate_pass": spearman >= REGRESSION_MIN_SPEARMAN,
    }


def export_ref(pool_dir: Path) -> dict[str, Any]:
    s29_db = paths.RUNS_DIR / "s29-qwen-judge" / "qwen_scores.sqlite"
    index_path = pool_dir / "pool_index.parquet"
    paths.require_inputs(s29_db)
    paths.require_inputs(index_path)
    target = pool_dir / "s29_qwen06_scores.parquet"
    temp = target.with_name(target.name + ".tmp")
    for path in (target, temp, pool_dir / "s29_qwen06_scores.json", pool_dir / "s29_qwen06_scores.json.tmp"):
        paths.assert_writable(path)
    uri = s29_db.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        s29 = pd.read_sql_query("SELECT query_id, doc_id, cosine_rank, cosine_score, qwen_score FROM scores", conn)
    index = pd.read_parquet(index_path)
    pool13 = index[index.in_pool13]
    joined = pool13.merge(s29, on=["query_id", "doc_id"], how="left", suffixes=("", "_s29"))
    if joined.qwen_score.isna().any() or len(s29) != len(pool13):
        raise ValueError("S29 scores do not cover the S31 1.3x pairs one-to-one")
    if not (joined.cosine_rank == joined.cosine_rank_s29).all() or not np.array_equal(
        joined.cosine_score.to_numpy(), joined.cosine_score_s29.to_numpy()
    ):
        raise ValueError("S29 cosine rank/score differ from the S31 pool")
    out = joined[["pair_id", "query_id", "doc_id", "qwen_score"]].sort_values("pair_id")
    out.to_parquet(temp, index=False)
    temp.replace(target)
    report = {"pairs": len(out), "file": str(target), "sha256": _sha256(target), "source": str(s29_db)}
    _atomic_json(pool_dir / "s29_qwen06_scores.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=None, help="S31 pool dir (default RUNS_DIR/s31-strong-judge/pool)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ref", help="export S29 0.6B scores keyed by S31 pair_id")
    reg = sub.add_parser("regress", help="compare a pilot score parquet with the S29 reference")
    reg.add_argument("--scores", type=Path, required=True)
    reg.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    pool_dir = paths.resolve_path(args.pool or paths.RUNS_DIR / "s31-strong-judge" / "pool")
    if args.command == "ref":
        print(json.dumps(export_ref(pool_dir), ensure_ascii=False, indent=2))
    else:
        stats = regression_stats(pd.read_parquet(args.scores), pd.read_parquet(pool_dir / "s29_qwen06_scores.parquet"))
        stats["scores"] = str(args.scores)
        if args.report:
            paths.assert_writable(args.report)
            _atomic_json(paths.resolve_path(args.report), stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0 if stats["gate_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
