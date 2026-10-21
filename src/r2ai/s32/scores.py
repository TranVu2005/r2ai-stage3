"""Reduce S32 chunk scores to document scores and compare them with the S29/S31 judges (CPU).

``python -m r2ai.s32.scores reduce`` reads the ``scores-<subset>.parquet`` written by
``r2ai.s31.judge_score collect`` for the chunk pool and writes, below ``--out``:

* ``doc_scores.parquet``: one row per (query, doc) pair of the S31 pool13 plus the Pharmacity lot 3
  pairs, with ``cmax`` (max P(yes) over the scored chunks), ``ctop2`` (mean of the two best),
  ``cmix`` (max(cmax, S29 0.6B doc_text score)), ``q06`` (S29 / S31-ext 0.6B doc_text score);
* ``cmaxnt`` (C-max-nt): max over the scored chunks whose field is not ``title``; a doc whose scored
  chunks are all titles keeps its title score. ``cmax_is_title`` flags pairs whose best chunk is a title;
* ``fallback`` pairs (doc without chunk) take ``q06`` for all three and are flagged;
* ``score-<kind>-pool13.parquet`` / ``score-<kind>-ph979.parquet`` for kind in cmax/ctop2/cmix:
  ``pair_id, query_id, doc_id, qwen_score`` aligned to the S31 pool13 index and to the
  Pharmacity-extended index, the format ``r2ai.s31.build.load_candidates`` reads as 0.6B scores;
* ``correlations.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256

KINDS = ("cmax", "ctop2", "cmix", "cmaxnt")


def reduce_chunk_scores(
    chunk_pairs: pd.DataFrame, chunk_scores: pd.DataFrame, fallback: pd.DataFrame, q06: pd.DataFrame
) -> pd.DataFrame:
    """One row per source pair: cmax / ctop2 / cmix / q06 and the fallback flag.

    ``chunk_pairs``: pair_id, src_pair_id, query_id, doc_id, source, field. ``chunk_scores``: pair_id, score.
    ``fallback``: src_pair_id, query_id, doc_id, source (documents scored by ``q06``).
    ``q06``: src_pair_id, q06 for every pair.
    """
    scored = chunk_pairs.merge(chunk_scores[["pair_id", "score"]], on="pair_id", how="inner")
    if scored.score.isna().any() or not np.isfinite(scored.score).all():
        raise ValueError("Non-finite chunk scores")
    if scored.pair_id.duplicated().any():
        raise ValueError("A chunk pair has two scores")
    expected = chunk_pairs.groupby("src_pair_id").size().rename("n_selected")
    ordered = scored.sort_values(["src_pair_id", "score"], ascending=[True, False], kind="stable")
    grouped = ordered.groupby("src_pair_id")
    nt = scored[scored.field != "title"].groupby("src_pair_id").score.max().rename("cmaxnt")
    best = grouped.head(1).set_index("src_pair_id").field.eq("title").rename("cmax_is_title")
    agg = pd.DataFrame({
        "cmax": grouped.score.max(),
        "ctop2": grouped.score.apply(lambda s: float(s.iloc[:2].mean())),
        "n_scored": grouped.size(),
    })
    agg = agg.join(expected).join(nt).join(best)
    agg["cmaxnt"] = agg.cmaxnt.fillna(agg.cmax)
    meta = chunk_pairs.drop_duplicates("src_pair_id").set_index("src_pair_id")[["query_id", "doc_id", "source"]]
    agg = agg.join(meta)
    agg["fallback"] = False
    fb = fallback.set_index("src_pair_id")[["query_id", "doc_id", "source"]].copy()
    fb["n_scored"] = 0
    fb["n_selected"] = 0
    fb["cmax_is_title"] = False
    fb["fallback"] = True
    out = pd.concat([agg.reset_index(), fb.reset_index()], ignore_index=True)
    if out.src_pair_id.duplicated().any():
        raise ValueError("A pair is both chunk-scored and fallback")
    out = out.merge(q06[["src_pair_id", "q06"]], on="src_pair_id", how="left")
    if out.q06.isna().any():
        raise ValueError("Missing 0.6B doc_text score for some pairs")
    for kind in ("cmax", "ctop2", "cmaxnt"):
        out[kind] = np.where(out.fallback, out.q06, out[kind])
    out["cmix"] = np.maximum(out.cmax, out.q06)
    return out.sort_values("src_pair_id").reset_index(drop=True)


def spearman_table(frame: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Global Spearman and the mean/p10 of per-query Spearman for each (left, right) column pair."""
    result = {}
    for left, right in pairs:
        per_query = frame.groupby("query_id").apply(
            lambda g: g[left].corr(g[right], method="spearman"), include_groups=False
        )
        result[f"{left}~{right}"] = {
            "n": int(len(frame)),
            "spearman_global": float(frame[left].corr(frame[right], method="spearman")),
            "spearman_per_query_mean": float(per_query.mean()),
            "spearman_per_query_p10": float(per_query.quantile(0.10)),
        }
    return result


def reduce_command(pool: Path, scores_path: Path, out: Path) -> dict[str, Any]:
    s31 = paths.RUNS_DIR / "s31-strong-judge"
    ext_dir = s31 / "pharmacity-ext"
    for target in [out, out / "doc_scores.parquet", out / "correlations.json", out / "reduce.json"] + [
        out / f"score-{k}-{i}.parquet" for k in KINDS for i in ("pool13", "ph979")
    ]:
        paths.assert_writable(target)
    out.mkdir(parents=True, exist_ok=True)
    chunk_pairs = pd.read_parquet(pool / "chunk_pairs.parquet",
                                  columns=["pair_id", "src_pair_id", "query_id", "doc_id", "source", "chunk_rank", "field"])
    fallback = pd.read_parquet(pool / "doc_fallback.parquet")
    chunk_scores = pd.read_parquet(scores_path)
    q06_main = pd.read_parquet(s31 / "pool" / "s29_qwen06_scores.parquet")[["pair_id", "qwen_score"]]
    q06_ext = pd.read_parquet(ext_dir / "q06_scores.parquet")[["pair_id", "qwen_score"]]
    q06 = pd.concat([q06_main, q06_ext[q06_ext.pair_id >= 10_000_000]], ignore_index=True).rename(
        columns={"pair_id": "src_pair_id", "qwen_score": "q06"})
    unexpected = set(chunk_scores.pair_id) - set(chunk_pairs.pair_id)
    if unexpected:
        raise ValueError(f"{len(unexpected)} scored pair_ids are not in chunk_pairs")
    docs = reduce_chunk_scores(chunk_pairs, chunk_scores, fallback, q06)
    docs.to_parquet(out / "doc_scores.parquet", index=False)

    index = pd.read_parquet(s31 / "pool" / "pool_index.parquet")
    index = index[index.in_pool13][["pair_id", "query_id", "doc_id", "cosine_score"]]
    ext = pd.read_parquet(ext_dir / "pool13_index.parquet")[["pair_id", "query_id", "doc_id", "cosine_score"]]
    written = {}
    for label, idx in (("pool13", index), ("ph979", ext)):
        joined = idx.merge(docs.rename(columns={"query_id": "q_s", "doc_id": "d_s"}), left_on="pair_id",
                           right_on="src_pair_id", how="left")
        if joined.src_pair_id.isna().any() or not ((joined.query_id == joined.q_s).all() and (joined.doc_id == joined.d_s).all()):
            raise ValueError(f"doc_scores do not cover the {label} index one-to-one")
        for kind in KINDS:
            target = out / f"score-{kind}-{label}.parquet"
            joined[["pair_id", "query_id", "doc_id"]].assign(qwen_score=joined[kind].to_numpy()).to_parquet(target, index=False)
            written[f"{kind}-{label}"] = {"file": str(target), "sha256": _sha256(target), "pairs": len(joined)}

    main = index.merge(docs, left_on="pair_id", right_on="src_pair_id", suffixes=("", "_d"))
    j4 = pd.read_parquet(s31 / "gpu" / "qwen3-reranker-4b" / "scores-pool13.parquet")[["pair_id", "score"]].rename(
        columns={"score": "j4"})
    main = main.merge(j4, on="pair_id", how="left").rename(columns={"cosine_score": "cosine"})
    pairs = [("cmax", "q06"), ("cmax", "j4"), ("cmax", "cosine"), ("ctop2", "q06"), ("ctop2", "j4"), ("ctop2", "cosine"),
             ("cmix", "q06"), ("cmix", "j4"), ("cmix", "cosine"), ("cmax", "ctop2"), ("cmax", "cmaxnt"), ("cmaxnt", "q06"), ("cmaxnt", "j4"), ("cmaxnt", "cosine"), ("q06", "j4"), ("q06", "cosine"),
             ("j4", "cosine")]
    correlations = {"all_pool13_pairs": spearman_table(main, pairs),
                    "chunk_scored_pairs_only": spearman_table(main[~main.fallback], pairs)}
    _atomic_json(out / "correlations.json", correlations)
    report = {
        "status": "reduced",
        "chunk_scores": str(scores_path),
        "chunk_scores_sha256": _sha256(scores_path),
        "chunk_pairs_sha256": _sha256(pool / "chunk_pairs.parquet"),
        "pairs": int(len(docs)),
        "pairs_pool13": int((docs.source == "pool13").sum()),
        "fallback_pairs": int(docs.fallback.sum()),
        "n_scored_distribution": {str(k): int(v) for k, v in docs.n_scored.value_counts().sort_index().items()},
        "doc_scores": str(out / "doc_scores.parquet"),
        "doc_scores_sha256": _sha256(out / "doc_scores.parquet"),
        "score_files": written,
        "cmax_quantiles": {f"p{int(q * 100)}": float(docs.cmax.quantile(q)) for q in (0.1, 0.5, 0.9)},
        "q06_quantiles": {f"p{int(q * 100)}": float(docs.q06.quantile(q)) for q in (0.1, 0.5, 0.9)},
        "share_cmax_gt_q06": float((docs.cmax > docs.q06).mean()),
        "share_cmax_from_title_chunk_scored_pairs": float(docs.loc[~docs.fallback, "cmax_is_title"].mean()),
        "share_cmax_from_title_pool13_chunk_scored": float(docs.loc[~docs.fallback & (docs.source == "pool13"), "cmax_is_title"].mean()),
        "share_cmaxnt_ne_cmax": float((docs.loc[~docs.fallback, "cmaxnt"] != docs.loc[~docs.fallback, "cmax"]).mean()),
    }
    _atomic_json(out / "reduce.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    red = sub.add_parser("reduce")
    red.add_argument("--pool", type=Path, default=None, help="S32 pool dir (default s32-chunk-judge/pool)")
    red.add_argument("--scores", type=Path, required=True, help="scores-<subset>.parquet from judge_score collect")
    red.add_argument("--out", type=Path, default=None, help="default s32-chunk-judge/scores")
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s32-chunk-judge"
    out = paths.resolve_path(args.out or root / "scores")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S32 output must stay below RUNS_DIR: {out}")
    report = reduce_command(paths.resolve_path(args.pool or root / "pool"), paths.resolve_path(args.scores), out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
