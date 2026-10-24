"""S36 fixed-depth pools (S35, 25/10): the pairs a cosine top-D pool for EVERY query needs scored, on the corpus WITH the S37 120ask docs.

``python -m r2ai.s36.d300 count --depth 300 400``   counts only (no export, no scoring)
``python -m r2ai.s36.d300 export --depth 300``      writes the Kaggle 4B subset ``d300`` and the C-max pair list; scoring is NOT started

Merged ranking: per query, the canonical + Pharmacity cosine top-400 of ``out/runs/s37/pool.npz`` (S26 doc vectors, the pool the S29/S31
pools were cut from) merged with the cosine of every 120ask doc (segment ``vectors.npz``), top ``depth`` by cosine (canonical first on a tie).
A pair is *scored* when it has a 4B score (S36 union + S37 Kaggle scores), a C-max (S32 + S36 supplement + S37) or a Jina score
(S35 + S36 supplement + the S37 rescoring when it exists). Jina is only counted: the extra part is not scored, the pairs rank last in
the Jina source (``missing: last``).

Export layout (S31 pool-extra / S37 ``kaggle-4b`` layout) in ``out/runs/s36-fuse4/d300/``:
``kaggle/{judge_score.py, extra_pairs.parquet, shards/pool-93002.parquet, manifest.json}``, ``pairs_d300.parquet`` (the new pairs with
their ids, languages and segments, for the later ``apply``), ``cmax_pairs.parquet`` (pairs without a C-max), ``counts.json``.
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
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _model_path, _sha256, truncate_document
from r2ai.s36 import fuse, s37
from r2ai.s31.pool import _truncate_all

RUN37 = paths.RUNS_DIR / "s37"
STORE = paths.DATA_DIR / "s37_120ask"
QUERY_VECTORS = paths.RUNS_DIR / "s29-qwen-judge" / "query_vectors.npz"
JUDGE_SCORE = paths.RUNS_DIR / "s31-strong-judge" / "pool-extra" / "kaggle" / "judge_score.py"
JUDGE_SHA = "688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186"
REVISION_4B = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
PAIR_ID_BASE = 5_000_000          # S31 pairs 0..433,731 and 20,000,000+; S37 pairs 1,000,000+; S37 C-max chunk rows 3,000,000+
SHARD = "pool-93002.parquet"
TAG = "d300"
KAGGLE_PAIRS_PER_S = 5.8


def out_dir() -> Path:
    return fuse.run_root() / "d300"


def load_pool() -> dict[str, np.ndarray]:
    with np.load(RUN37 / "pool.npz") as z:
        return {k: z[k] for k in z.files}


def ask_cosines(q_dense: np.ndarray, thr: np.ndarray, store: Path | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per segment, ``(doc_ids, query_index, cosine)`` of every 120ask doc scoring above the query's cut ``thr``."""
    store = store or STORE
    q32 = q_dense.astype(np.float32)
    out = []
    for seg in sorted(p for p in store.iterdir() if (p / "vectors.npz").exists()):
        with np.load(seg / "vectors.npz") as z:
            ids, dense = z["doc_id"].astype(np.int64), z["dense"].astype(np.float32)
        cos = dense @ q32.T                                                   # [docs, queries]
        di, qi = np.nonzero(cos > thr[None, :])
        out.append((ids[di], qi, cos[di, qi].astype(np.float64)))
    return out


def merged_top(depth: int, pool: dict[str, np.ndarray] | None = None, q_dense: np.ndarray | None = None,
               asks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """``query_id, cosine_rank, doc_id, cosine_score, is_ask`` of the top ``depth`` docs of every query over canonical + Pharmacity + 120ask."""
    pool = pool or load_pool()
    if depth > pool["scores"].shape[1]:
        raise ValueError(f"depth {depth} > the {pool['scores'].shape[1]} cosines kept in pool.npz")
    if q_dense is None:
        with np.load(QUERY_VECTORS) as z:
            if not np.array_equal(z["q_ids"].astype(np.int64), pool["q_ids"]):
                raise ValueError("query order of query_vectors.npz differs from pool.npz")
            q_dense = np.array(z["q_dense"], copy=True)
    if asks is None:
        asks = ask_cosines(q_dense, pool["scores"][:, depth - 1])
    by_query: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ids, qi, cos in asks:
        for i in np.unique(qi):
            sel = qi == i
            by_query.setdefault(int(i), []).append((ids[sel], cos[sel]))
    frames = []
    for i, q in enumerate(pool["q_ids"].tolist()):
        d, s, a = [pool["doc_ids"][i, :depth]], [pool["scores"][i, :depth].astype(np.float64)], [np.zeros(depth, bool)]
        for ids, cos in by_query.get(i, []):
            d.append(ids), s.append(cos), a.append(np.ones(len(ids), bool))
        d, s, a = np.concatenate(d), np.concatenate(s), np.concatenate(a)
        order = np.argsort(-s, kind="stable")[:depth]                          # canonical first on a tie
        frames.append(pd.DataFrame({"query_id": q, "cosine_rank": np.arange(1, len(order) + 1), "doc_id": d[order],
                                    "cosine_score": s[order], "is_ask": a[order]}))
    return pd.concat(frames, ignore_index=True)


def scored_sets() -> dict[str, set[tuple[int, int]]]:
    """(query, doc) pairs that already have a 4B / C-max / Jina score."""
    _, union, _, _, _, _ = fuse.load_frames()
    spec = s37.default_inputs()
    pairs = pd.read_parquet(spec["pairs_new"], columns=["pair_id", "query_id", "doc_id"])
    key_of = dict(zip(pairs.pair_id.astype(int), zip(pairs.query_id.astype(int), pairs.doc_id.astype(int))))
    four_b = {(q, int(r["doc_id"])) for q, rows in union.items() for r in rows}
    four_b |= {key_of[p] for p in s37.load_scores(spec["scores_4b"], "pair_id", "score")}
    jina, _ = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
    cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
    cmax_keys = set(cmax) | {key_of[p] for p in s37.load_scores(spec["cmax"], "pair_id", "cmax")}
    jina_keys = set(jina)
    if Path(spec["jina"]).exists():
        frame = pd.read_parquet(spec["jina"])
        if {"query_id", "doc_id"} <= set(frame.columns):
            jina_keys |= set(zip(frame.query_id.astype(int), frame.doc_id.astype(int)))
        elif "pair_id" in frame.columns:
            jina_keys |= {key_of[p] for p in frame.pair_id.astype(int) if p in key_of}
    return {"4b": four_b, "cmax": cmax_keys, "jina": jina_keys}


def need(top: pd.DataFrame, scored: dict[str, set[tuple[int, int]]]) -> pd.DataFrame:
    keys = list(zip(top.query_id.astype(int), top.doc_id.astype(int)))
    out = top.copy()
    for source, have in scored.items():
        out[f"has_{source}"] = [k in have for k in keys]
    return out


def summarize(top: pd.DataFrame, depth: int) -> dict[str, Any]:
    new = top[~top.has_4b]
    per_query = new.groupby("query_id").size().reindex(range(1, 1201), fill_value=0)
    return {"depth": depth, "pairs": int(len(top)), "pairs_ask": int(top.is_ask.sum()),
            "pairs_4b_needed": int(len(new)), "pairs_4b_needed_ask": int(new.is_ask.sum()), "pairs_4b_needed_canonical": int((~new.is_ask).sum()),
            "queries_needing_4b": int((per_query > 0).sum()), "per_query_p50_p90_max": [float(np.quantile(per_query, .5)), float(np.quantile(per_query, .9)),
                                                                                         int(per_query.max())],
            "pairs_cmax_needed": int((~top.has_cmax).sum()), "pairs_cmax_needed_with_4b": int((top.has_4b & ~top.has_cmax).sum()),
            "pairs_jina_missing": int((~top.has_jina).sum()), "pairs_jina_missing_ask": int((top[~top.has_jina].is_ask).sum()),
            "eta_hours_kaggle_2xT4": round(len(new) / KAGGLE_PAIRS_PER_S / 3600, 2)}


def count(depths: Sequence[int]) -> dict[str, Any]:
    pool, scored = load_pool(), scored_sets()
    result = {"scored": {k: len(v) for k, v in scored.items()}, "depths": {}}
    for depth in depths:
        result["depths"][str(depth)] = summarize(need(merged_top(depth, pool), scored), depth)
    out = out_dir()
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "counts.json", result)
    return result


def export(depth: int) -> dict[str, Any]:
    started = time.monotonic()
    out = out_dir()
    kaggle = out / "kaggle"
    for target in (out, kaggle, kaggle / "shards" / SHARD, kaggle / "extra_pairs.parquet", kaggle / "judge_score.py", kaggle / "manifest.json",
                   out / "pairs_d300.parquet", out / "cmax_pairs.parquet", out / "counts.json"):
        paths.assert_writable(target)
    if _sha256(JUDGE_SCORE) != JUDGE_SHA:
        raise SystemExit(f"judge_score.py SHA differs from {JUDGE_SHA}: refusing to package an unknown scorer")
    pool, scored = load_pool(), scored_sets()
    top = need(merged_top(depth, pool), scored)
    summary = summarize(top, depth)
    todo = top[~top.has_4b].sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)
    todo["pair_id"] = PAIR_ID_BASE + np.arange(len(todo))
    ask_docs = sorted(set(todo.loc[todo.is_ask, "doc_id"].astype(int)))
    seg_of: dict[int, str] = {}
    texts: dict[int, str] = {}
    for seg in sorted(p for p in STORE.iterdir() if (p / "docs.parquet").exists()):
        docs = pq.read_table(seg / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", ask_docs)]).to_pandas()
        for d, t in zip(docs.doc_id.astype(int), docs.doc_text):
            seg_of[d], texts[d] = seg.name, t or ""
    if len(seg_of) != len(ask_docs):
        raise ValueError(f"{len(ask_docs) - len(seg_of)} 120ask docs of the new pairs are in no segment")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    log = lambda m: print(f"[{time.monotonic() - started:7.0f}s] {m}", flush=True)  # noqa: E731
    passages, langs, need_full = _truncate_all(_inputs(), set(todo.loc[~todo.is_ask, "doc_id"].astype(int)), tokenizer, log)
    if need_full:
        raise ValueError(f"{len(need_full)} canonical docs need the full text (not handled here)")
    for d in ask_docs:
        passages[d], langs[d] = truncate_document(tokenizer, texts[d]), "zh"
    qtable = pq.read_table(_inputs()["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    shard = pa.table({
        "pair_id": pa.array(todo.pair_id, pa.int64()), "query_id": pa.array(todo.query_id, pa.int64()), "doc_id": pa.array(todo.doc_id, pa.int64()),
        "cosine_rank": pa.array(todo.cosine_rank, pa.int32()), "cosine_score": pa.array(todo.cosine_score, pa.float64()),
        "in_pool13": pa.array([False] * len(todo), pa.bool_()), "lang": pa.array([langs[int(d)] for d in todo.doc_id], pa.string()),
        "query_text": pa.array([queries[int(q)] for q in todo.query_id], pa.string()),
        "doc_text": pa.array([passages[int(d)][0] for d in todo.doc_id], pa.string()),
        "doc_tokens": pa.array([passages[int(d)][1] for d in todo.doc_id], pa.int32())})
    (kaggle / "shards").mkdir(parents=True, exist_ok=True)
    pq.write_table(shard, kaggle / "shards" / SHARD, compression="zstd", row_group_size=5_000)
    extra = pd.DataFrame({"pair_id": todo.pair_id, "query_id": todo.query_id, "doc_id": todo.doc_id, "cosine_rank": todo.cosine_rank,
                          "cosine_score": todo.cosine_score, "source": f"s36-{TAG}", "free": todo.cosine_rank <= 211, "capped": False})
    extra.to_parquet(kaggle / "extra_pairs.parquet", index=False)
    todo.assign(lang=[langs[int(d)] for d in todo.doc_id], segment=[seg_of.get(int(d), "") for d in todo.doc_id], in_top211=todo.cosine_rank <= 211,
                in_pool13=False, batch=TAG)[["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "in_pool13", "lang", "segment", "in_top211",
                                             "is_ask", "batch"]].to_parquet(out / "pairs_d300.parquet", index=False)
    top[~top.has_cmax][["query_id", "doc_id", "cosine_rank", "cosine_score", "is_ask", "has_4b"]].to_parquet(out / "cmax_pairs.parquet", index=False)
    shutil.copyfile(JUDGE_SCORE, kaggle / "judge_score.py")
    manifest = {"status": "prepared", "tag": TAG, "depth": depth, **summary, "model": "qwen3-reranker-4b", "revision": REVISION_4B,
                "pair_id_range": [int(todo.pair_id.min()), int(todo.pair_id.max())] if len(todo) else None,
                "lang_counts": pd.Series([langs[int(d)] for d in todo.doc_id]).value_counts().to_dict(),
                "doc_tokens_mean": float(np.mean([passages[int(d)][1] for d in todo.doc_id])) if len(todo) else 0.0,
                "extra_pairs_sha256": _sha256(kaggle / "extra_pairs.parquet"), "shard_sha256": _sha256(kaggle / "shards" / SHARD),
                "judge_score_py_sha256": JUDGE_SHA, "pairs_d300_sha256": _sha256(out / "pairs_d300.parquet"),
                "cmax_pairs_sha256": _sha256(out / "cmax_pairs.parquet"), "seconds": round(time.monotonic() - started, 1),
                "note": "nothing is scored here; Jina is not requested for these pairs (rank last in the Jina source)"}
    _atomic_json(kaggle / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--depth", type=int, nargs="+", default=[300, 400])
    e = sub.add_parser("export")
    e.add_argument("--depth", type=int, default=300)
    args = parser.parse_args(argv)
    result = count(args.depth) if args.cmd == "count" else export(args.depth)
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
