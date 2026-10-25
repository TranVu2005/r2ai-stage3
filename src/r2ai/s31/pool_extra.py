"""S31 pool-extra: the 4B pairs still needed for K4-free (top-211) and K4-thr-ext (capped queries).

``python -m r2ai.s31.pool_extra`` lists, deduplicated by (query_id, doc_id):

* **free**: for every query, the cosine top-``P_FREE`` (211, no K_q) over canonical + every
  Pharmacity lot, minus pairs already scored by 4B (extended pool13 and the evicted
  original pool13 pairs);
* **capped**: for the queries at the 1.3x cap in K4-thr, the S31 1.3x-2.0x tail pairs plus
  any doc (e.g. Pharmacity lot 3) of the fresh cosine top-round(2.0 * len(P1)) not yet
  in the S31 pool.

Pairs already in the S31 2.0x shards keep their pair_id; the others get pair_ids from
``NEW_PAIR_ID_BASE`` and their Qwen3-truncated text in ``shards/pool-91000.parquet``.
Output ``out/runs/s31-strong-judge/pool-extra/``:

* ``extra_pairs.parquet``: pair_id, query_id, doc_id, cosine_rank (fresh), cosine_score,
  source (``s31-tail`` | ``new``), free, capped;
* ``pool211_index.parquet``: the fresh cosine top-211 of every query with its pair_id
  and where its 4B score comes from (``pool13`` | ``evicted`` | ``extra``);
* ``capped_queries.parquet``; ``manifest.json``;
* ``kaggle/``: ``judge_score.py``, ``extra_pairs.parquet``, ``shards/pool-91000.parquet``
  (dataset "pool-extra" for the notebook, ``SUBSET = "extra"``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_doc_vectors, _load_submission, _model_path, _sha256
from r2ai.s31.build import P_FREE, load_candidates, rank_candidates, variable_keep
from r2ai.s31.pool import _top_cosine, _truncate_all

REVISION_4B = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
NEW_PAIR_ID_BASE = 20_000_000
NEW_SHARD = "pool-91000.parquet"
PAIRS_PER_SECOND_2XT4 = 5.8


def needed_pairs(
    fresh: dict[int, list[tuple[int, float]]],
    targets: dict[int, int],
    scored: set[tuple[int, int]],
    tail: dict[int, set[int]],
    capped: set[int],
) -> list[tuple[int, int, int, float, bool, bool]]:
    """(query_id, doc_id, fresh rank, cosine, free, capped) of the pairs still to score.

    ``fresh[q]`` is the fresh cosine ranking (doc, score) of query q (at least ``targets[q]``
    long); ``tail[q]`` the S31 tail docs added for capped queries even if the fresh top cut
    them (exact cosine ties or Pharmacity docs pushing them down).
    """
    rows: dict[tuple[int, int], list[Any]] = {}
    for qid, ranking in fresh.items():
        rank_of = {doc: (rank, score) for rank, (doc, score) in enumerate(ranking, 1)}
        free_n = min(P_FREE, len(ranking))
        for rank, (doc, score) in enumerate(ranking[: targets[qid]], 1):
            if (qid, doc) in scored:
                continue
            rows[(qid, doc)] = [qid, doc, rank, score, rank <= free_n, qid in capped]
        for doc in tail.get(qid, set()) if qid in capped else ():
            if (qid, doc) in scored:
                continue
            if (qid, doc) in rows:
                rows[(qid, doc)][5] = True
            else:
                rank, score = rank_of.get(doc, (0, float("nan")))
                rows[(qid, doc)] = [qid, doc, rank, score, False, True]
    return [tuple(r) for r in rows.values()]


def run(pool_dir: Path, ext_dir: Path, out: Path, judge_score_py: Path) -> dict[str, Any]:
    started = time.monotonic()
    shard_path = out / "shards" / NEW_SHARD
    kaggle = out / "kaggle"
    targets_out = [out, out / "shards", shard_path, kaggle, kaggle / "shards", kaggle / "shards" / NEW_SHARD,
                   kaggle / "judge_score.py", kaggle / "extra_pairs.parquet", out / "extra_pairs.parquet",
                   out / "pool211_index.parquet", out / "capped_queries.parquet", out / "manifest.json",
                   out / "manifest.json.tmp"]
    for target in targets_out:
        paths.assert_writable(target)
    (kaggle / "shards").mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        print(f"[{time.monotonic() - started:7.0f}s] {message}", flush=True)

    inputs = _inputs()
    judge = ext_dir / "qwen3-reranker-4b-scores-pool13.parquet"
    frame = load_candidates(pool_dir, judge, ext_dir / "pool13_index.parquet", ext_dir / "q06_scores.parquet")
    ranked = rank_candidates(frame)
    base_k = {int(r["id"]): len(r["relevant_docs"]) for r in _load_submission(inputs["baseline"])}
    if round(float(np.mean([1.3 * k for k in base_k.values()]))) != P_FREE:
        raise ValueError("P_FREE no longer equals round(mean(1.3 * K_q))")
    keep, _, thr = variable_keep(ranked, base_k, "thr")
    capped = {q for q in ranked if keep[q] == int(np.floor(1.3 * base_k[q] + 1e-9))}

    pool_index = pd.read_parquet(pool_dir / "pool_index.parquet")
    ext = pd.read_parquet(ext_dir / "pool13_index.parquet")
    n13 = ext.groupby("query_id").size().to_dict()
    n20 = pool_index.groupby("query_id").size().to_dict()
    free = {q for q in ranked if n13[q] < P_FREE}
    # Every query: fresh top-211 (pairs outside pool13 only for pool13 < 211, exact cosine ties
    # at the S29 cut, or Pharmacity docs); capped queries: also the top-round(2.0 * len(P1)).
    targets = {q: max(P_FREE, n20[q] if q in capped else 0) for q in ranked}
    orig13 = pool_index[pool_index.in_pool13]
    scored = set(zip(ext.query_id.astype(int), ext.doc_id.astype(int))) | set(
        zip(orig13.query_id.astype(int), orig13.doc_id.astype(int)))
    tail_rows = pool_index[~pool_index.in_pool13]
    tail = {q: set(g.doc_id.astype(int)) for q, g in tail_rows[tail_rows.query_id.isin(capped)].groupby("query_id")}
    report(f"capped={len(capped)} free(pool13<{P_FREE})={len(free)} both={len(capped & free)}")

    with np.load(paths.RUNS_DIR / "s29-qwen-judge" / "query_vectors.npz") as qv:
        q_ids = qv["q_ids"].astype(np.int64)
        q_dense = np.array(qv["q_dense"], copy=True)
    doc_ids, doc_dense, base_count, extra_files = _load_doc_vectors(inputs)
    report(f"doc vectors rows={len(doc_ids)} base={base_count} lots={len(extra_files)}")
    pharmacity_ids = set(doc_ids[base_count:].astype(int).tolist())
    all_q = [int(q) for q in q_ids]
    q_pos = {q: i for i, q in enumerate(all_q)}
    want = sorted(set(all_q))
    top_n = max(max(targets.values()), P_FREE)
    top_i, top_s = _top_cosine(q_dense[[q_pos[q] for q in want]], doc_dense, top_n, report)
    del doc_dense
    fresh = {q: list(zip(doc_ids[top_i[i]].astype(int).tolist(), top_s[i].astype(float).tolist()))
             for i, q in enumerate(want)}

    # Consistency: the fresh top-n13 of each query should be the extended pool13 (up to cosine ties).
    ext_sets = {q: set(g.doc_id.astype(int)) for q, g in ext.groupby("query_id")}
    mismatch = {q: len(ext_sets[q] - {d for d, _ in fresh[q][: n13[q]]}) for q in want}
    report(f"fresh top-n13 vs extended pool13: queries differing={sum(1 for v in mismatch.values() if v)} docs={sum(mismatch.values())}")

    rows = needed_pairs(fresh, targets, scored, tail, capped)
    need = pd.DataFrame(rows, columns=["query_id", "doc_id", "cosine_rank", "cosine_score", "free", "capped"])
    s31_pair = {(int(q), int(d)): int(p) for q, d, p in zip(pool_index.query_id, pool_index.doc_id, pool_index.pair_id)}
    need["pair_id"] = [s31_pair.get((q, d), -1) for q, d in zip(need.query_id, need.doc_id)]
    need["source"] = np.where(need.pair_id >= 0, "s31-tail", "new")
    new_mask = need.pair_id < 0
    need.loc[new_mask, "pair_id"] = NEW_PAIR_ID_BASE + np.arange(int(new_mask.sum()))
    need = need.sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)
    if need.duplicated(["query_id", "doc_id"]).any() or need.pair_id.duplicated().any():
        raise ValueError("Duplicate pairs in pool-extra")
    if need[need.source == "s31-tail"].pair_id.isin(orig13.pair_id).any():
        raise ValueError("A pool13 pair was listed as unscored")
    report(f"pairs={len(need)} s31-tail={int((~new_mask).sum())} new={int(new_mask.sum())}")

    new_pairs = need[need.source == "new"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    passages, langs, need_full = _truncate_all(inputs, set(new_pairs.doc_id.astype(int)), tokenizer, report)
    qtable = pq.read_table(inputs["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    table = pa.table({
        "pair_id": pa.array(new_pairs.pair_id, pa.int64()),
        "query_id": pa.array(new_pairs.query_id, pa.int64()),
        "doc_id": pa.array(new_pairs.doc_id, pa.int64()),
        "cosine_rank": pa.array(new_pairs.cosine_rank, pa.int32()),
        "cosine_score": pa.array(new_pairs.cosine_score, pa.float64()),
        "in_pool13": pa.array([False] * len(new_pairs), pa.bool_()),
        "lang": pa.array([langs[int(d)] for d in new_pairs.doc_id], pa.string()),
        "query_text": pa.array([queries[int(q)] for q in new_pairs.query_id], pa.string()),
        "doc_text": pa.array([passages[int(d)][0] for d in new_pairs.doc_id], pa.string()),
        "doc_tokens": pa.array([passages[int(d)][1] for d in new_pairs.doc_id], pa.int32()),
    })
    pq.write_table(table, shard_path, compression="zstd", row_group_size=5_000)
    pairs_cols = ["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "source", "free", "capped"]
    need[pairs_cols].to_parquet(out / "extra_pairs.parquet", index=False)

    # Fresh top-211 of every query with the origin of its 4B score.
    ext_pair = {(int(q), int(d)): int(p) for q, d, p in zip(ext.query_id, ext.doc_id, ext.pair_id)}
    orig_pair = {(int(q), int(d)): int(p) for q, d, p in zip(orig13.query_id, orig13.doc_id, orig13.pair_id)}
    extra_pair = {(int(q), int(d)): int(p) for q, d, p in zip(need.query_id, need.doc_id, need.pair_id)}
    rows211 = []
    for q in want:
        for rank, (doc, score) in enumerate(fresh[q][:P_FREE], 1):
            key = (q, doc)
            if key in ext_pair:
                origin, pid = "pool13", ext_pair[key]
            elif key in orig_pair:
                origin, pid = "evicted", orig_pair[key]
            elif key in extra_pair:
                origin, pid = "extra", extra_pair[key]
            else:
                raise ValueError(f"Top-{P_FREE} pair {key} has no score source")
            rows211.append((q, doc, rank, score, pid, origin))
    pool211 = pd.DataFrame(rows211, columns=["query_id", "doc_id", "cosine_rank", "cosine_score", "pair_id", "score_origin"])
    pool211.to_parquet(out / "pool211_index.parquet", index=False)
    capped_frame = pd.DataFrame({"query_id": sorted(capped)})
    capped_frame["k_q"] = capped_frame.query_id.map(base_k)
    capped_frame["k_q_prime"] = capped_frame.query_id.map(keep)
    capped_frame["n_q_at_t"] = capped_frame.query_id.map(dict(zip(sorted(ranked), thr["n"].tolist())))
    capped_frame["extra_pairs"] = capped_frame.query_id.map(need[need.capped].groupby("query_id").size()).fillna(0).astype(int)
    capped_frame.to_parquet(out / "capped_queries.parquet", index=False)

    shutil.copyfile(shard_path, kaggle / "shards" / NEW_SHARD)
    shutil.copyfile(out / "extra_pairs.parquet", kaggle / "extra_pairs.parquet")
    shutil.copyfile(judge_score_py, kaggle / "judge_score.py")
    total = len(need)
    result = {
        "status": "prepared",
        "p_free": P_FREE,
        "k4_thr_t": thr["t"],
        "capped_queries": len(capped),
        "free_queries_pool13_below_p": len(free),
        "queries_both": len(capped & free),
        "pairs_total": total,
        "pairs_free": int(need.free.sum()),
        "pairs_capped": int(need.capped.sum()),
        "pairs_free_and_capped": int((need.free & need.capped).sum()),
        "pairs_s31_tail": int((need.source == "s31-tail").sum()),
        "pairs_new": int((need.source == "new").sum()),
        "new_pairs_pharmacity": int(new_pairs.doc_id.isin(pharmacity_ids).sum()),
        "new_docs_pharmacity": int(new_pairs.loc[new_pairs.doc_id.isin(pharmacity_ids), "doc_id"].nunique()),
        "new_docs": int(new_pairs.doc_id.nunique()),
        "new_full_text_fallback": len(need_full),
        "fresh_vs_ext_pool13_docs_differing": int(sum(mismatch.values())),
        "pool211_pairs": len(pool211),
        "pool211_score_origin": pool211.score_origin.value_counts().to_dict(),
        "eta_hours_kaggle_2xT4": round(total / PAIRS_PER_SECOND_2XT4 / 3600, 2),
        "eta_rule": f"{PAIRS_PER_SECOND_2XT4} pairs/s on 2x T4 (4B fp16, measured on pool13)",
        "model": "qwen3-reranker-4b",
        "revision": REVISION_4B,
        "extra_pairs_sha256": _sha256(out / "extra_pairs.parquet"),
        "new_shard_sha256": _sha256(shard_path),
        "judge_score_py_sha256": _sha256(kaggle / "judge_score.py"),
        "pool211_index_sha256": _sha256(out / "pool211_index.parquet"),
        "pool_index_sha256": _sha256(pool_dir / "pool_index.parquet"),
        "ext_pool13_index_sha256": _sha256(ext_dir / "pool13_index.parquet"),
        "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--ext", type=Path, default=None, help="pharmacity_ext output dir")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s31-strong-judge"
    out = paths.resolve_path(args.out or root / "pool-extra")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"Output must stay below RUNS_DIR: {out}")
    result = run(paths.resolve_path(args.pool or root / "pool"), paths.resolve_path(args.ext or root / "pharmacity-ext"),
                 out, Path(__file__).with_name("judge_score.py"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
