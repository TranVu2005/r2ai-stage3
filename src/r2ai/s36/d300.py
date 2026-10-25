"""S36 fixed-depth pools (S35, 25/10): the pairs a cosine top-D pool for EVERY query needs scored, on the corpus WITH the S37 120ask docs.

``python -m r2ai.s36.d300 count --depth 300 400``   counts only (no export, no scoring)
``python -m r2ai.s36.d300 export --depth 300``      writes the Kaggle 4B subset ``d300`` and the C-max pair list; scoring is NOT started
``python -m r2ai.s36.d300 export-final --depth 300`` one subset ``final-4b``: the top-D pairs and the unscored ``pairs_new_final_v2`` pairs, deduplicated
``python -m r2ai.s36.d300 import4b --tag final-4b --scores scores-extra.parquet --report collect-extra-report.json``

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
from r2ai.s36 import enlane, fuse, s37
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


def registry_files() -> list[Path]:
    """The pair registries that exist: S37 ``pairs_new`` and ``pairs_new_final_v2`` and the pairs exported here (``pair_keys`` checks that a
    ``pair_id`` means one pair across them)."""
    return [p for p in (RUN37 / "pairs_new.parquet", RUN37 / "pairs_new_final_v2.parquet", out_dir() / "pairs_d300.parquet",
                        out_dir() / "pairs_final.parquet") if p.exists()]


def score_files() -> list[str]:
    """4B score files: the S37 Kaggle subsets and the S36 subsets (``d300/<tag>/gpu/qwen3-reranker-4b/scores-extra.parquet``)."""
    mine = sorted(str(p) for p in out_dir().glob("*/gpu/qwen3-reranker-4b/scores-extra.parquet"))
    return list(s37.default_inputs()["scores_4b"]) + mine


def pair_keys(files: Sequence[Path]) -> dict[int, tuple[int, int]]:
    """``pair_id -> (query_id, doc_id)`` over the registries; one pair_id must mean one pair."""
    out: dict[int, tuple[int, int]] = {}
    for f in files:
        frame = pd.read_parquet(f, columns=["pair_id", "query_id", "doc_id"])
        for pid, q, d in zip(frame.pair_id.astype(int), frame.query_id.astype(int), frame.doc_id.astype(int)):
            if out.setdefault(pid, (q, d)) != (q, d):
                raise ValueError(f"pair_id {pid} is two different pairs ({out[pid]} and {(q, d)}) in {f}")
    return out


def scored_sets() -> dict[str, set[tuple[int, int]]]:
    """(query, doc) pairs that already have a 4B / C-max / Jina score."""
    _, union, _, _, _, _ = fuse.load_frames()
    spec = s37.default_inputs()
    key_of = pair_keys(registry_files())
    four_b = {(q, int(r["doc_id"])) for q, rows in union.items() for r in rows}
    four_b |= {key_of[p] for p in s37.load_scores(score_files(), "pair_id", "score")}
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
    result = {"registries": [str(p) for p in registry_files()], "scored": {k: len(v) for k, v in scored.items()}, "depths": {}}
    for depth in depths:
        result["depths"][str(depth)] = summarize(need(merged_top(depth, pool), scored), depth)
    out = out_dir()
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "counts.json", result)
    return result


def combine_needs(depth_pairs: pd.DataFrame, registry_pairs: pd.DataFrame, known_ids: dict[tuple[int, int], int], next_id: int) -> pd.DataFrame:
    """One row per (query, doc) of the pairs that need a 4B score: the cosine top-D pairs ``depth_pairs`` (``query_id, doc_id, cosine_rank,
    cosine_score, is_ask``) and the unscored registry pairs ``registry_pairs`` (``pair_id, query_id, doc_id, cosine_rank, cosine_score``;
    their ``pair_id`` wins), duplicates removed. A pair without a registry id keeps the id it got in an earlier export (``known_ids``) or takes
    the next free one from ``next_id``. ``origin``: ``both`` / ``depth`` / ``registry``."""
    reg = registry_pairs.assign(origin="registry")
    dep = depth_pairs.assign(origin="depth", pair_id=-1)
    merged = pd.concat([reg, dep], ignore_index=True, sort=False)
    merged["is_ask"] = merged["is_ask"].fillna(False).astype(bool) if "is_ask" in merged else False
    merged["key"] = list(zip(merged.query_id.astype(int), merged.doc_id.astype(int)))
    both = set(merged.key[merged.origin == "registry"]) & set(merged.key[merged.origin == "depth"])
    merged = merged.drop_duplicates("key", keep="first")                       # registry rows come first: their id wins
    merged["origin"] = ["both" if k in both else o for k, o in zip(merged.key, merged.origin)]
    ids, taken = [], max([next_id - 1, *known_ids.values()])
    for k, pid in zip(merged.key, merged.pair_id.astype(int)):
        if pid >= 0:
            ids.append(pid)
        elif k in known_ids:
            ids.append(known_ids[k])
        else:
            taken += 1
            ids.append(taken)
    merged["pair_id"] = ids
    if merged.pair_id.duplicated().any():
        raise ValueError("pair_id collision between the registry and the D=300 ids")
    return merged.sort_values(["query_id", "cosine_rank"]).drop(columns="key").reset_index(drop=True)


def passages_for(todo: pd.DataFrame, started: float) -> tuple[dict[int, tuple[str, int]], dict[int, str], dict[int, str]]:
    """Qwen3-truncated text, language and segment of every doc of ``todo`` (120ask docs from their segment, the others from the canonical bundle)."""
    ask_docs = sorted(set(todo.loc[todo.is_ask, "doc_id"].astype(int)))
    is_en = todo["is_en"] if "is_en" in todo else pd.Series(False, index=todo.index)
    en_docs = sorted(set(todo.loc[is_en, "doc_id"].astype(int)))
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
    passages, langs, need_full = _truncate_all(_inputs(), set(todo.loc[~todo.is_ask & ~is_en, "doc_id"].astype(int)), tokenizer, log)
    if need_full:
        raise ValueError(f"{len(need_full)} canonical docs need the full text (not handled here)")
    for d in ask_docs:
        passages[d], langs[d] = truncate_document(tokenizer, texts[d]), "zh"
    if en_docs:                                                                  # English PubMed lane: text from the en segment
        docs = pq.read_table(enlane.SEG / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", en_docs)]).to_pandas()
        if len(docs) != len(en_docs):
            raise ValueError(f"{len(en_docs) - len(docs)} en docs of the new pairs are not in {enlane.SEG.name}")
        clash = set(en_docs) & (set(passages) | set(ask_docs))
        if clash:
            raise ValueError(f"{len(clash)} en doc_ids are also vi/zh doc_ids, e.g. {sorted(clash)[:3]}")
        for d, t in zip(docs.doc_id.astype(int), docs.doc_text):
            passages[d], langs[d], seg_of[d] = truncate_document(tokenizer, t or ""), "en", enlane.SEG.name
    return passages, langs, seg_of


def write_subset(todo: pd.DataFrame, out: Path, tag: str, shard_name: str, summary: dict[str, Any], started: float,
                 top: pd.DataFrame | None = None, scorer: tuple[Path, str] | None = None) -> dict[str, Any]:
    """Kaggle 4B subset (S31 pool-extra layout) of ``todo`` under ``out/kaggle`` or ``out/<tag>``, plus ``pairs_<tag>.parquet`` and the C-max list."""
    sub = out / ("kaggle" if tag == TAG else tag)
    pairs_name = "pairs_d300.parquet" if tag == TAG else "pairs_final.parquet"
    for target in (out, sub, sub / "shards" / shard_name, sub / "extra_pairs.parquet", sub / "judge_score.py", sub / "manifest.json",
                   out / pairs_name, out / "cmax_pairs.parquet", out / "counts.json"):
        paths.assert_writable(target)
    scorer_path, scorer_sha = scorer or (JUDGE_SCORE, JUDGE_SHA)
    if _sha256(scorer_path) != scorer_sha:
        raise SystemExit(f"judge_score.py SHA differs from {scorer_sha}: refusing to package an unknown scorer")
    passages, langs, seg_of = passages_for(todo, started)
    qtable = pq.read_table(_inputs()["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    shard = pa.table({
        "pair_id": pa.array(todo.pair_id, pa.int64()), "query_id": pa.array(todo.query_id, pa.int64()), "doc_id": pa.array(todo.doc_id, pa.int64()),
        "cosine_rank": pa.array(todo.cosine_rank, pa.int32()), "cosine_score": pa.array(todo.cosine_score, pa.float64()),
        "in_pool13": pa.array([False] * len(todo), pa.bool_()), "lang": pa.array([langs[int(d)] for d in todo.doc_id], pa.string()),
        "query_text": pa.array([queries[int(q)] for q in todo.query_id], pa.string()),
        "doc_text": pa.array([passages[int(d)][0] for d in todo.doc_id], pa.string()),
        "doc_tokens": pa.array([passages[int(d)][1] for d in todo.doc_id], pa.int32())})
    (sub / "shards").mkdir(parents=True, exist_ok=True)
    pq.write_table(shard, sub / "shards" / shard_name, compression="zstd", row_group_size=5_000)
    pd.DataFrame({"pair_id": todo.pair_id, "query_id": todo.query_id, "doc_id": todo.doc_id, "cosine_rank": todo.cosine_rank,
                  "cosine_score": todo.cosine_score, "source": f"s36-{tag}", "free": todo.cosine_rank <= 211, "capped": False}
                 ).to_parquet(sub / "extra_pairs.parquet", index=False)
    is_en = todo["is_en"] if "is_en" in todo else pd.Series(False, index=todo.index)
    todo.assign(lang=[langs[int(d)] for d in todo.doc_id], segment=[seg_of.get(int(d), "") for d in todo.doc_id], in_top211=todo.cosine_rank <= 211,
                in_pool13=False, batch=tag, lane=np.where(is_en, "en", "main"))[
        ["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "in_pool13", "lang", "segment", "in_top211", "is_ask", "batch", "lane"]
    ].to_parquet(out / pairs_name, index=False)
    if top is not None:
        need_cmax = top[~top.has_cmax][["query_id", "doc_id", "cosine_rank", "cosine_score", "is_ask", "has_4b"]].assign(lane="main")
        if is_en.any():                                                       # en abstracts are short: one chunk each, no C-max score exists yet
            en = todo[is_en][["query_id", "doc_id", "cosine_rank", "cosine_score", "is_ask"]].assign(has_4b=False, lane="en")
            need_cmax = pd.concat([need_cmax, en], ignore_index=True)
        need_cmax.to_parquet(out / "cmax_pairs.parquet", index=False)
    shutil.copyfile(scorer_path, sub / "judge_score.py")
    manifest = {"status": "prepared", "tag": tag, **summary, "model": "qwen3-reranker-4b", "revision": REVISION_4B, "pairs": int(len(todo)),
                "pair_id_range": [int(todo.pair_id.min()), int(todo.pair_id.max())] if len(todo) else None,
                "lang_counts": pd.Series([langs[int(d)] for d in todo.doc_id]).value_counts().to_dict(),
                "doc_tokens_mean": float(np.mean([passages[int(d)][1] for d in todo.doc_id])) if len(todo) else 0.0,
                "eta_hours_kaggle_2xT4": round(len(todo) / KAGGLE_PAIRS_PER_S / 3600, 2),
                "extra_pairs_sha256": _sha256(sub / "extra_pairs.parquet"), "shard_sha256": _sha256(sub / "shards" / shard_name),
                "judge_score_py_sha256": scorer_sha, "judge_score_base_sha256": JUDGE_SHA, f"{pairs_name.split('.')[0]}_sha256": _sha256(out / pairs_name),
                "seconds": round(time.monotonic() - started, 1),
                "note": "nothing is scored here; Jina is not requested for these pairs (rank last in the Jina source)"}
    if top is not None:
        manifest["cmax_pairs_sha256"] = _sha256(out / "cmax_pairs.parquet")
    _atomic_json(sub / "manifest.json", manifest)
    return manifest


def export(depth: int) -> dict[str, Any]:
    """Kaggle subset ``d300``: the pairs of the cosine top-``depth`` pool without a 4B score."""
    started = time.monotonic()
    pool, scored = load_pool(), scored_sets()
    top = need(merged_top(depth, pool), scored)
    todo = top[~top.has_4b].sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)
    todo["pair_id"] = PAIR_ID_BASE + np.arange(len(todo))
    return write_subset(todo, out_dir(), TAG, SHARD, {"depth": depth, **summarize(top, depth)}, started, top)


def export_final(depth: int, registry: Path | None = None, en: bool = False) -> dict[str, Any]:
    """Kaggle subset ``final-4b`` (S35, 25/10): ONE subset holding (a) the cosine top-``depth`` pairs of the current corpus without a 4B score and
    (b) the unscored pairs of the registry (``pairs_new_final_v2``: the newly crawled 120ask docs), duplicates removed. Ids: the registry's,
    else those of an earlier ``pairs_d300``/``pairs_final`` export, else new ones above the highest."""
    started = time.monotonic()
    registry = registry or (RUN37 / "pairs_new_final_v2.parquet")
    if not registry.exists():
        raise FileNotFoundError(f"{registry} is not there yet")
    pool, scored = load_pool(), scored_sets()
    # the EF2-cut-p4b frame (pool13) takes the cosine top-n_q of the corpus, n_q = round(1.3 K_q) (up to 339 > 300): cover max(depth, n_q) per query
    size13 = {int(q): len(rows) for q, rows in fuse.load_frames()[0].items()}
    cover = {q: max(depth, n) for q, n in size13.items()}
    top = need(merged_top(max(cover.values()), pool), scored)
    top = top[top.cosine_rank <= top.query_id.astype(int).map(cover)]
    reg = pd.read_parquet(registry, columns=["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score"])
    reg = reg[[k not in scored["4b"] for k in zip(reg.query_id.astype(int), reg.doc_id.astype(int))]]
    depth_pairs = top[~top.has_4b][["query_id", "doc_id", "cosine_rank", "cosine_score", "is_ask"]]
    known = {}
    for name in ("pairs_final.parquet", "pairs_d300.parquet"):
        p = out_dir() / name
        if p.exists():
            old = pd.read_parquet(p, columns=["pair_id", "query_id", "doc_id"])
            known.update({(int(q), int(d)): int(i) for i, q, d in zip(old.pair_id, old.query_id, old.doc_id)})
    # registry docs are 120ask docs: flag them so the text comes from their segment
    todo = combine_needs(depth_pairs, reg, known, PAIR_ID_BASE)
    todo["is_en"] = False
    ask = set(top.loc[top.is_ask, "doc_id"].astype(int)) | set(pd.read_parquet(registry, columns=["doc_id"]).doc_id.astype(int))
    todo["is_ask"] = todo.doc_id.astype(int).isin(ask)
    free_keys = set(zip(*[top[top.cosine_rank <= depth][c].astype(int) for c in ("query_id", "doc_id")])) - scored["4b"]
    p4b_keys = set(zip(*[top[top.cosine_rank <= top.query_id.astype(int).map(size13)][c].astype(int) for c in ("query_id", "doc_id")])) - scored["4b"]
    todo_keys = set(zip(todo.query_id.astype(int), todo.doc_id.astype(int)))
    if not (free_keys | p4b_keys) <= todo_keys:
        raise ValueError(f"{len((free_keys | p4b_keys) - todo_keys)} pairs of the free top-{depth} / pool13 are missing from the subset")
    summary = {"depth": depth, "registry": str(registry), "registry_sha256": _sha256(registry), **summarize(top[top.cosine_rank <= depth], depth),
               "registry_pairs_unscored": int(len(reg)), "pairs_from_depth_only": int((todo.origin == "depth").sum()),
               "pairs_from_registry_only": int((todo.origin == "registry").sum()), "pairs_in_both": int((todo.origin == "both").sum())}
    # per frame: the free frame (fixed pool D) needs the unscored pairs of the cosine top-D, the EF2-cut-p4b frame the unscored pairs of the cosine
    # top-n_q (pool13 size); the rest of the subset is the registry pairs below both cuts (new 120ask pairs that the S37 crawl found)
    both_frames = free_keys | p4b_keys
    summary.update({"pairs_needed_free_frame": len(free_keys), "pairs_needed_p4b_frame": len(p4b_keys), "pairs_needed_either_frame": len(both_frames),
                    "pairs_registry_below_both_cuts": len(todo_keys - both_frames),
                    "eta_hours_free_frame": round(len(free_keys) / KAGGLE_PAIRS_PER_S / 3600, 2),
                    "eta_hours_p4b_frame": round(len(p4b_keys) / KAGGLE_PAIRS_PER_S / 3600, 2),
                    "eta_hours_either_frame": round(len(both_frames) / KAGGLE_PAIRS_PER_S / 3600, 2),
                    "pool13_size_max": int(max(size13.values())), "coverage_checked": True})
    scorer = None
    if en:                                                                    # English lane: the top-20 en docs per query, appended to the same subset
        en_pairs = enlane.pairs()
        if int(en_pairs.pair_id.max()) >= 20_000_000 or set(en_pairs.pair_id) & set(todo.pair_id):
            raise ValueError("en pair_id range collides with the vi/zh pairs")
        en_rows = en_pairs.assign(origin="en", is_ask=False, is_en=True)[
            ["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "origin", "is_ask", "is_en"]]
        n_vi_zh = len(todo)
        todo = pd.concat([todo, en_rows], ignore_index=True)
        made = enlane.make_scorer(out_dir() / "final-4b-scorer" / "judge_score.py")
        scorer = (Path(made["path"]), made["sha256"])
        summary.update({"en_lane": True, "pairs_en": int(len(en_pairs)), "pairs_vi_zh": int(n_vi_zh), "en_docs": int(en_pairs.doc_id.nunique()),
                        "eta_hours_en": round(len(en_pairs) / KAGGLE_PAIRS_PER_S / 3600, 2), "en_scorer_added_line": made["added_line"]})
    return write_subset(todo, out_dir(), "final-4b", "pool-93003.parquet", summary, started, top, scorer)


def import4b(tag: str, scores: Path, report: Path) -> dict[str, Any]:
    """Validate Kaggle 4B scores of a subset exported here and store them as ``<tag>/gpu/qwen3-reranker-4b/scores-extra.parquet``
    (picked up by ``score_files`` and the S37 ``apply`` defaults); same checks as ``r2ai.s37.export.import4b``."""
    d = out_dir() / ("kaggle" if tag == TAG else tag)
    paths.assert_writable(d)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads(report.read_text(encoding="utf-8"))
    sc, ex = pd.read_parquet(scores), pd.read_parquet(d / "extra_pairs.parquet")
    sha, errors = _sha256(scores), []
    if sha != rep.get("scores_sha256"):
        errors.append(f"scores SHA256 {sha} != report {rep.get('scores_sha256')}")
    if rep.get("revisions") != [REVISION_4B] or rep.get("model") != "qwen3-reranker-4b":
        errors.append(f"report model/revision {rep.get('model')} {rep.get('revisions')} != qwen3-reranker-4b {REVISION_4B}")
    if len(sc) != man["pairs"] or sc.pair_id.nunique() != len(sc):
        errors.append(f"{len(sc)} rows / {sc.pair_id.nunique()} unique pair_id, expected {man['pairs']}")
    elif set(sc.pair_id) != set(ex.pair_id):
        errors.append("pair_id set differs from extra_pairs.parquet")
    elif sc.score.isna().any():
        errors.append("NaN scores")
    if errors:
        raise SystemExit("import4b refused: " + "; ".join(errors))
    dest = d / "gpu" / "qwen3-reranker-4b"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scores, dest / "scores-extra.parquet")
    shutil.copyfile(report, dest / "collect-extra-report.json")
    man.update({"status": "scored", "scores_sha256": sha})
    _atomic_json(d / "manifest.json", man)
    return {"tag": tag, "pairs": len(sc), "scores_sha256": sha, "dir": str(dest)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--depth", type=int, nargs="+", default=[300, 400])
    e = sub.add_parser("export")
    e.add_argument("--depth", type=int, default=300)
    f = sub.add_parser("export-final")
    f.add_argument("--depth", type=int, default=300)
    f.add_argument("--registry", type=Path, default=None, help="default out/runs/s37/pairs_new_final_v2.parquet")
    f.add_argument("--en", action="store_true", help="append the English lane: the top-20 en docs per query (needs the S37 en vectors; en scorer)")
    i = sub.add_parser("import4b")
    i.add_argument("--tag", required=True)
    i.add_argument("--scores", type=Path, required=True)
    i.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "count":
        result = count(args.depth)
    elif args.cmd == "export":
        result = export(args.depth)
    elif args.cmd == "export-final":
        result = export_final(args.depth, paths.resolve_path(args.registry) if args.registry else None, args.en)
    else:
        result = import4b(args.tag, args.scores, args.report)
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
