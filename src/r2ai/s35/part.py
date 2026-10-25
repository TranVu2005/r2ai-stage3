"""S35 part 1: encode-by-part cosine on the S31 cosine top-211 pool (J4-K40 pool, 253,200 pairs).

``python -m r2ai.s35.part``

* ``cos_doc512`` = the pool's BGE-M3 doc-level cosine (``cosine_score`` of the S31 scored union).
* ``cos_part``   = max cosine(query, t256 chunk vector) over the chunks of the doc. It is the
  ``chunk_cosine`` of ``chunk_rank == 1`` in the S33 R-chk pool: S32/S33 scored *every* chunk slot of the
  doc owner with the S29 query vector and kept the top 3, so rank 1 is the max. Stored canonical vectors
  are used as they are; chunks without a stored vector were embedded earlier by S32/S33 (same
  ``M3Encoder``, ``vector_verify`` gate), nothing is embedded here.
* ``cos_doc2``   = max(cos_doc512, cos_part); a doc without any chunk keeps cos_doc512.

Outputs below ``out/runs/s35-part-jina/part/``: ``part_table.parquet`` and ``part_report.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import RRF_K, doc_set_diff, judge_correlations, select_rrf

S35_NAME = "s35-part-jina"
K_COMPARE = 40


def run_root() -> Path:
    return paths.run_dir(S35_NAME)


def s31_root() -> Path:
    return paths.RUNS_DIR / "s31-strong-judge"


def s33_root() -> Path:
    return paths.RUNS_DIR / "s33-btc-baseline"


def load_pools() -> tuple[dict[int, list[dict[str, Any]]], dict[str, Path]]:
    """The J4-K40 pool: cosine top-211 of the S31 scored union, 4B scores and ranks inside it."""
    from r2ai.s33.build import load_union

    union, files = load_union(s31_root())
    pools, _ = s31_build.topk_pools(union, s31_build.P_FREE)
    return pools, files


def best_chunk_cosine(rchk_pools: Sequence[Path]) -> pd.Series:
    """``src_pair_id -> max chunk cosine`` (chunk_rank 1 of the R-chk pools)."""
    frames = [pd.read_parquet(p / "chunk_pairs.parquet", columns=["src_pair_id", "chunk_rank", "chunk_cosine"])
              for p in rchk_pools]
    frame = pd.concat(frames, ignore_index=True)
    top = frame[frame.chunk_rank == 1]
    if top.src_pair_id.duplicated().any():
        raise ValueError("Two rank-1 chunks for one pool pair")
    return top.set_index("src_pair_id").chunk_cosine


def add_part_cosine(pools: dict[int, list[dict[str, Any]]], best: pd.Series) -> None:
    """Set ``cos_part`` (NaN without chunk), ``cos_doc2`` and ``cos2_rank`` on every pool row.

    ``cos2_rank``: 1-based rank by cos_doc2 descending inside the query pool, ties by the old
    ``cosine_rank`` (so cos_doc2 == cos_doc512 everywhere gives back ``cosine_rank``).
    """
    lookup = best.to_dict()
    for rows in pools.values():
        for r in rows:
            part = lookup.get(int(r["pair_id"]))
            r["cos_part"] = float("nan") if part is None else float(part)
            r["cos_doc2"] = float(r["cosine_score"]) if part is None else max(float(r["cosine_score"]), float(part))
        for rank, r in enumerate(sorted(rows, key=lambda r: (-r["cos_doc2"], int(r["cosine_rank"]))), 1):
            r["cos2_rank"] = rank


def table(pools: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame([{"query_id": q, "pair_id": r["pair_id"], "doc_id": r["doc_id"], "cosine_rank": r["cosine_rank"],
                          "cos2_rank": r["cos2_rank"], "judge_rank": r["judge_rank"], "cos_doc512": r["cosine_score"],
                          "cos_part": r["cos_part"], "cos_doc2": r["cos_doc2"], "p4b": r["judge_score"],
                          "origin": r["origin"]} for q, rows in pools.items() for r in rows])


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                        np.quantile(values, [0, .1, .25, .5, .75, .9, 1]))}


def report(pools: dict[int, list[dict[str, Any]]], frame: pd.DataFrame) -> dict[str, Any]:
    has = frame.cos_part.notna()
    gain = frame.cos_part - frame.cos_doc512
    higher = has & (gain > 0)
    per_query_higher = frame.assign(h=higher).groupby("query_id").h.mean()
    corr_in = frame.assign(delta=frame.cos_doc2 - frame.cos_doc512)
    # top-K overlap and RRF doc sets with cos_doc2 in place of the cosine rank
    cos_ids = {q: [int(r["doc_id"]) for r in sorted(rows, key=lambda r: r["cosine_rank"])[:K_COMPARE]] for q, rows in pools.items()}
    cos2_ids = {q: [int(r["doc_id"]) for r in sorted(rows, key=lambda r: r["cos2_rank"])[:K_COMPARE]] for q, rows in pools.items()}
    pools2 = {q: [{**r, "cosine_rank_old": r["cosine_rank"], "cosine_rank": r["cos2_rank"]} for r in rows] for q, rows in pools.items()}
    j4_ids = {q: [int(r["doc_id"]) for r in select_rrf(rows, K_COMPARE, ("cosine_rank", "judge_rank"), RRF_K)] for q, rows in pools.items()}
    p2_ids = {q: [int(r["doc_id"]) for r in select_rrf(rows, K_COMPARE, ("cosine_rank", "judge_rank"), RRF_K)] for q, rows in pools2.items()}
    # docs gained by the part cosine: in the new top-40 cosine but not in the old one
    win = frame[higher]
    return {
        "pairs": int(len(frame)), "queries": int(frame.query_id.nunique()),
        "pairs_with_chunk": int(has.sum()), "pairs_without_chunk": int((~has).sum()),
        "share_cos_part_gt_cos_doc512": {
            "of_pairs_with_chunk": float(higher.sum() / has.sum()), "of_all_pairs": float(higher.mean()),
            "pairs": int(higher.sum()), "per_query_share": quantiles(per_query_higher.to_numpy())},
        "cos_part_minus_cos_doc512": quantiles(gain[has].to_numpy()),
        "cos_part_minus_cos_doc512_where_part_wins": quantiles(gain[higher].to_numpy()) if higher.any() else None,
        "mean": {"cos_doc512": float(frame.cos_doc512.mean()), "cos_part": float(frame.cos_part.mean()),
                 "cos_doc2": float(frame.cos_doc2.mean())},
        "p4b_mean": {"all": float(frame.p4b.mean()), "part_wins": float(win.p4b.mean()), "doc_wins_or_tie": float(frame[has & ~higher].p4b.mean())},
        "spearman_with_p4b_and_cosine": judge_correlations(
            corr_in.assign(cosine=corr_in.cos_doc512), ["cos_doc2", "cosine", "cos_part", "p4b"]),
        "top40_cosine_vs_cos_doc2": doc_set_diff(cos2_ids, cos_ids),
        "top40_rrf_cos_doc2_4b_vs_j4_k40": doc_set_diff(p2_ids, j4_ids),
        "rank_shift_cos2_vs_cosine": quantiles((frame.cosine_rank - frame.cos2_rank).abs().to_numpy().astype(np.float64)),
        "origin": frame.origin.value_counts().to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root33 = s33_root()
    parser.add_argument("--rchk-pool", type=Path, action="append", default=None, help="default: S33 pool-rchk, pool-rchk-tie")
    parser.add_argument("--out", type=Path, default=run_root() / "part")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out)
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S35 output must stay below RUNS_DIR: {out}")
    for name in ("part_table.parquet", "part_table.parquet.tmp", "part_report.json", "part_report.json.tmp"):
        paths.assert_writable(out / name)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rchk_pools = [paths.resolve_path(p) for p in (args.rchk_pool or [root33 / "pool-rchk", root33 / "pool-rchk-tie"])]
    pools, _ = load_pools()
    add_part_cosine(pools, best_chunk_cosine(rchk_pools))
    frame = table(pools)
    temp = out / "part_table.parquet.tmp"
    frame.to_parquet(temp, index=False)
    temp.replace(out / "part_table.parquet")
    result = report(pools, frame)
    result["table_sha256"] = _sha256(out / "part_table.parquet")
    result["rchk_pools"] = [str(p) for p in rchk_pools]
    result["embedded_here"] = 0
    result["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / "part_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
