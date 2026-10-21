"""Extend the S31 pool13 with Pharmacity lots added after the pool export (612 -> 979 docs).

``prepare``: cosine of every query against the new lots' doc-level vectors (same fp32
matmul as S29), merged into each query's 1.3x pool; the pool size per query stays
``round(1.3 * K_q)``, so each new doc that enters evicts the lowest cosine pair.
Writes ``ext/pool13_index.parquet`` (the full extended 1.3x pool), the new pairs
with Qwen3-truncated doc_text as ``ext/shards/pool-90000.parquet`` for
``judge_score.py``, and ``ext/manifest.json``.

``merge-scores``: combine kept old scores (S29 0.6B, Kaggle 4B) with the new-pair
scores into ``ext/q06_scores.parquet`` and ``ext/<model>-scores-pool13.parquet``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _model_path, _sha256, truncate_document

NEW_PAIR_ID_BASE = 10_000_000
NEW_SHARD = "pool-90000.parquet"


def merge_pool(existing: pd.DataFrame, new_ids: np.ndarray, new_scores: np.ndarray) -> tuple[pd.DataFrame, int]:
    """One query: existing pool rows (sorted by cosine_rank) + new docs, keep the same count.

    Order is descending cosine; on an exact tie an existing doc stays ahead of a new one.
    Returns the merged rows (cosine_rank renumbered 1..n, ``is_new``) and the evicted count.
    """
    n = len(existing)
    old = pd.DataFrame({"pair_id": existing.pair_id.to_numpy(), "doc_id": existing.doc_id.to_numpy(),
                        "cosine_score": existing.cosine_score.to_numpy(), "is_new": False,
                        "order": np.arange(n)})
    new = pd.DataFrame({"pair_id": -1, "doc_id": new_ids, "cosine_score": new_scores.astype(np.float64),
                        "is_new": True, "order": n + np.arange(len(new_ids))})
    both = pd.concat([old, new], ignore_index=True)
    if both.doc_id.duplicated().any():
        raise ValueError("A new Pharmacity doc is already in the pool")
    both = both.sort_values(["cosine_score", "is_new", "order"], ascending=[False, True, True], kind="stable")
    kept = both.head(n).reset_index(drop=True)
    if not kept[~kept.is_new].cosine_score.is_monotonic_decreasing:
        raise ValueError("Existing pool order is not descending cosine")
    kept["cosine_rank"] = np.arange(1, n + 1)
    return kept.drop(columns="order"), int((~both.tail(len(both) - n).is_new).sum())


def _new_lots(inputs: dict[str, Path], manifest: dict[str, Any]) -> list[str]:
    used = {Path(f).parent.name for f in manifest["extra_vector_files"]}
    lots = sorted(p.name for p in (inputs["extra_docs"] / "lots").iterdir() if p.is_dir())
    return [lot for lot in lots if lot not in used]


def prepare(pool_dir: Path, out: Path) -> dict[str, Any]:
    started = time.monotonic()
    inputs = _inputs()
    manifest = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    lots = _new_lots(inputs, manifest)
    if not lots:
        raise ValueError("No Pharmacity lot newer than the pool export")
    for target in (out, out / "shards", out / "shards" / NEW_SHARD, out / "pool13_index.parquet", out / "manifest.json",
                   out / "manifest.json.tmp"):
        paths.assert_writable(target)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    s29_qvec = paths.RUNS_DIR / "s29-qwen-judge" / "query_vectors.npz"
    if _sha256(s29_qvec) != manifest["s29_query_vectors_sha256"]:
        raise ValueError("Query vectors differ from the pool export")
    with np.load(s29_qvec) as qv:
        q_ids = qv["q_ids"].astype(np.int64)
        q32 = qv["q_dense"].astype(np.float32)
    ids, dense, texts, lot_info = [], [], {}, []
    for lot in lots:
        vec_path = inputs["extra_vectors"] / lot / "dense_doc_level.npz"
        with np.load(vec_path) as v:
            lot_ids = v["doc_id"].astype(np.int64)
            ids.append(lot_ids)
            dense.append(v["dense"])
        docs = pd.read_parquet(inputs["extra_docs"] / "lots" / lot / "chunks" / "docs.parquet", columns=["doc_id", "doc_text"])
        if sorted(docs.doc_id.astype(int)) != sorted(lot_ids.tolist()):
            raise ValueError(f"{lot}: doc and vector IDs differ")
        texts.update({int(d): t or "" for d, t in zip(docs.doc_id, docs.doc_text)})
        lot_info.append({"lot": lot, "docs": len(lot_ids), "vector_sha256": _sha256(vec_path)})
    new_ids = np.concatenate(ids)
    scores = q32 @ np.concatenate(dense).astype(np.float32).T

    index = pd.read_parquet(pool_dir / "pool_index.parquet")
    pool13 = index[index.in_pool13].sort_values(["query_id", "cosine_rank"])
    if set(new_ids.tolist()) & set(index.doc_id.astype(int)):
        raise ValueError("New lot doc IDs already in the pool")
    groups = dict(tuple(pool13.groupby("query_id")))
    merged, evicted_total, next_id = [], 0, NEW_PAIR_ID_BASE
    for qpos, qid in enumerate(q_ids.tolist()):
        cut = float(groups[qid].cosine_score.min())
        enter = scores[qpos] >= cut
        rows, evicted = merge_pool(groups[qid], new_ids[enter], scores[qpos][enter])
        new_mask = rows.is_new.to_numpy()
        rows.loc[new_mask, "pair_id"] = np.arange(next_id, next_id + int(new_mask.sum()))
        next_id += int(new_mask.sum())
        evicted_total += evicted
        rows.insert(0, "query_id", qid)
        merged.append(rows)
    ext = pd.concat(merged, ignore_index=True)
    ext["pair_id"] = ext.pair_id.astype(np.int64)
    new_pairs = ext[ext.is_new]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    passages = {d: truncate_document(tokenizer, texts[d]) for d in sorted(set(new_pairs.doc_id.astype(int)))}
    qtable = pq.read_table(inputs["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    table = pa.table({
        "pair_id": pa.array(new_pairs.pair_id, pa.int64()),
        "query_id": pa.array(new_pairs.query_id, pa.int64()),
        "doc_id": pa.array(new_pairs.doc_id, pa.int64()),
        "cosine_rank": pa.array(new_pairs.cosine_rank, pa.int32()),
        "cosine_score": pa.array(new_pairs.cosine_score, pa.float64()),
        "in_pool13": pa.array([True] * len(new_pairs), pa.bool_()),
        "lang": pa.array(["vi"] * len(new_pairs), pa.string()),
        "query_text": pa.array([queries[int(q)] for q in new_pairs.query_id], pa.string()),
        "doc_text": pa.array([passages[int(d)][0] for d in new_pairs.doc_id], pa.string()),
        "doc_tokens": pa.array([passages[int(d)][1] for d in new_pairs.doc_id], pa.int32()),
    })
    pq.write_table(table, out / "shards" / NEW_SHARD, compression="zstd")
    ext.to_parquet(out / "pool13_index.parquet", index=False)
    report = {
        "status": "prepared",
        "new_lots": lot_info,
        "new_docs_in_lots": int(len(new_ids)),
        "new_pairs_in_pool13": int(len(new_pairs)),
        "new_docs_in_pool13": int(new_pairs.doc_id.nunique()),
        "queries_changed": int(new_pairs.query_id.nunique()),
        "evicted_pairs": evicted_total,
        "pool13_pairs": int(len(ext)),
        "pool13_pairs_before": int(len(pool13)),
        "pool13_index_sha256": _sha256(out / "pool13_index.parquet"),
        "new_shard_sha256": _sha256(out / "shards" / NEW_SHARD),
        "rule": "per query keep round(1.3*K_q) pairs by descending cosine; ties keep the existing doc first",
        "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", report)
    return report


def merge_scores(pool_dir: Path, out: Path, new_scores: dict[str, Path], judge_old: dict[str, Path]) -> dict[str, Any]:
    """Old kept scores + new-pair scores for each model; checks exact coverage of the extended pool."""
    ext = pd.read_parquet(out / "pool13_index.parquet")
    report: dict[str, Any] = {}
    for model, new_path in new_scores.items():
        old_path = judge_old[model]
        old = pd.read_parquet(old_path)
        score_col = "qwen_score" if "qwen_score" in old else "score"
        old = old.rename(columns={score_col: "score"})[["pair_id", "query_id", "doc_id", "score"]]
        new = pd.read_parquet(new_path)[["pair_id", "query_id", "doc_id", "score"]]
        both = pd.concat([old[old.pair_id.isin(ext.pair_id)], new[new.pair_id.isin(ext.pair_id)]], ignore_index=True)
        check = ext[["pair_id", "query_id", "doc_id"]].merge(both, on="pair_id", how="left", suffixes=("", "_s"))
        if check.score.isna().any() or both.pair_id.duplicated().any():
            raise ValueError(f"{model}: scores do not cover the extended pool13 one-to-one")
        if not ((check.query_id == check.query_id_s).all() and (check.doc_id == check.doc_id_s).all()):
            raise ValueError(f"{model}: pair_id/query_id/doc_id mismatch")
        frame = check[["pair_id", "query_id", "doc_id", "score"]].sort_values("pair_id")
        name = "q06_scores.parquet" if model == "qwen3-reranker-0.6b" else f"{model}-scores-pool13.parquet"
        if model == "qwen3-reranker-0.6b":
            frame = frame.rename(columns={"score": "qwen_score"})
        target = out / name
        paths.assert_writable(target)
        frame.to_parquet(target, index=False)
        report[model] = {"file": str(target), "sha256": _sha256(target), "pairs": len(frame),
                         "new_pairs": int(new.pair_id.isin(ext.pair_id).sum()), "old_source": str(old_path)}
    _atomic_json(out / "merge-scores.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    ms = sub.add_parser("merge-scores")
    ms.add_argument("--new-q06", type=Path, required=True)
    ms.add_argument("--new-judge", type=Path, required=True)
    ms.add_argument("--judge", default="qwen3-reranker-4b")
    ms.add_argument("--old-judge", type=Path, required=True)
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s31-strong-judge"
    pool_dir = paths.resolve_path(args.pool or root / "pool")
    out = paths.resolve_path(args.out or root / "pharmacity-ext")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"Output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    if args.command == "prepare":
        print(json.dumps(prepare(pool_dir, out), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(merge_scores(
            pool_dir, out,
            {"qwen3-reranker-0.6b": paths.resolve_path(args.new_q06), args.judge: paths.resolve_path(args.new_judge)},
            {"qwen3-reranker-0.6b": pool_dir / "s29_qwen06_scores.parquet", args.judge: paths.resolve_path(args.old_judge)},
        ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
