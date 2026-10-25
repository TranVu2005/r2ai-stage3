"""Build S31 judge bundles on CPU with the S29 body/chunk builder (pool 1.3x, nc140, Pharmacity).

``python -m r2ai.s31.build --variant J4-rrf --variant J4-rrf3`` reads the S31 pool
index, the S29 Qwen3-Reranker-0.6B scores (``pool/s29_qwen06_scores.parquet``) and
the strong judge's ``scores-pool13.parquet``, selects ``len(relevant_docs)`` of
K90-ph per query by multi-source RRF (k=60), and writes each bundle with
``r2ai.s29.build._build_variant`` (n_c x1.4, whole bodies, compact JSON).

``Jcheck-q06-rrf`` = RRF(cosine, 0.6B) through this path; its JSON must equal the
S29 ``QRRF-nc140-ph`` JSON (regression gate for the S31 path, needs no GPU scores).

``K4-scale`` / ``K4-thr`` / ``K4-mix``: per-query doc count K_q' set from the 4B scores
(K_q = len(relevant_docs) of K90-ph), same total as J4-rrf, docs = top-K_q' of
RRF(cosine, 4B):
  scale: K_q * r_q / mean(r), r_q = mean P4B of the RRF top-K_q; clip [0.6, 1.3] K_q.
  thr:   number of pool docs with P4B >= t; clip [0.5, 1.3] K_q; t sets the total.
  mix:   sqrt(K_q * n_q(t)) with the t of thr; clip [0.5, 1.3] K_q.
scale and mix reach the total with one factor c on the raw counts (clip after c).
``K4-tfree-t<t>`` (e.g. ``K4-tfree-t0.06``): K_q' = clip(number of pool docs with
P4B >= t, [0.5, 1.3] K_q) with a fixed t and no total constraint.
``K4-thr-C``: docs of K4-thr; chunks by threshold instead of n_c x1.4: in RRF order,
one chunk (whole body) per selected doc with P4B >= t_c, t_c set so the total is
``THR_C_CHUNKS`` (the chunk total of J4-rrf-ph979). If the ZIP is over the ceiling,
bodies are cut with the S29 fallback rule.
``K4-free-p13``: fixed constants instead of K_q / n_c: pool = cosine top-min(P_FREE,
pool13) (still bounded by the K_q-based pool13; the full top-211 needs the pool-extra
scores), docs with P4B >= T_FREE in RRF(cosine, 4B) order, floor F_FREE, cap = pool;
chunks = selected docs with P4B >= t_c, t_c set so the chunk total is K4_FREE_CHUNKS
(the chunk total of K4-thr). Temporary: the full K4-free (top-211 with the pool-extra
scores) calibrates t_c once to K4_FREE_CHUNKS with ``--calibrate-tc`` and then fixes it
as the constant K4_FREE_T_C.

With the pool-extra 4B scores (``--extra-scores``):
``K4-thr-C-ext``: K4-thr-C, but the queries at the 1.3x cap in K4-thr get the whole
scored pool (S31 2.0x + extra) and the cap floor(2.0 K_q); chunks P4B >= T_C_FIXED.
``K4-free``: adaptive depth: pool = cosine top-P_FREE; if its last doc has P4B >= T_FREE,
the pool grows to every scored pair of the query; docs P4B >= T_FREE, floor F_FREE,
cap = pool; chunks P4B >= T_C_FIXED (no recalibration).

``K4-thr-C-fx``: docs of K4-thr-C; one whole-body chunk per selected doc with P4B >=
T_C_FIXED, never cut (fails over the ceiling). Built after the S29 builder fix that gives
Pharmacity docs their body (before it, selected Pharmacity docs never carried a chunk).
``--no-extra-chunks`` offers Pharmacity docs as without text (the pre-fix behaviour), for
the regression check of chunk-selector variants against bundles built before the fix.

Fixed top-K, no K_q and no n_c (``--extra-scores``): pool = cosine top-P_FREE of the
scored union (judge ranks recomputed inside it), K docs per query, one whole-body chunk
for every selected doc with text (no threshold, never cut: fails over the ceiling).
``COS-K<K>``: cosine order alone (BGE-M3 doc-level). ``J4-K<K>``: RRF(cosine, 4B).

``--thin-chunks`` (off by default): thin docs (quality ``thin``) with at least
``THIN_MIN_CHARS`` characters of text and no chunk get their whole text verbatim as
the chunk; they count toward the n_c x1.4 cap like any other doc.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission, _sha256
from r2ai.s31.select import (
    RRF_K,
    add_judge_rank,
    allocate_total,
    doc_set_diff,
    judge_correlations,
    select_rrf,
    threshold_counts,
)

CHUNK_MULTIPLIER = 1.4
WHOLE_BODY = True
POOL_MULTIPLIER = 1.3
# Built before the S29 builder fix (Pharmacity docs without chunk): QRRF-nc140-ph selects
# 211 Pharmacity docs, so Jcheck-q06-rrf no longer reproduces it with the fixed builder.
QRRF_NC140_JSON_SHA256 = "da6cf34bbb71b25a1662c904f9db4e7a0e33408301137ca5c73c335595d63fa7"
# Regression gates after the fix: K4-thr-C with --no-extra-chunks must give the pre-fix
# K4-thr-C JSON (LB 0.4183); K4-thr-C-fx is the new reference.
K4_THR_C_JSON_SHA256 = "d771c4f9b0b5f7b8c68f934b28aeb207179a047a2f7a83f29c7b62628939c9b0"
K4_THR_C_FX_JSON_SHA256 = "c105b78da25c7dbfa141a48187484161fc858dc92fffd0a0311011e00453499c"
JUDGE_TAGS = {"qwen3-reranker-4b": "J4", "qwen3-reranker-8b": "J8"}
K_BOUNDS = {"scale": (0.6, 1.3), "thr": (0.5, 1.3), "mix": (0.5, 1.3), "tfree": (0.5, 1.3)}
THIN_MIN_CHARS = 20
THR_C_CHUNKS = 175_148  # chunk total of J4-rrf-ph979 (LB 0.3997)
# K4-free constants (no K_q / n_c): P = round(mean(1.3 * len(K90-ph))), F = median of
# ceil(0.5 * len(K90-ph)), T = the K4-thr threshold (LB 0.4122), T_C calibrated once.
P_FREE = 211
F_FREE = 86
T_FREE = 0.07751765647688268
K4_FREE_CHUNKS = 159_744  # chunk total of K4-thr (LB 0.4122), used by K4-free-p13 only
T_C_FIXED = 0.07055959618729296  # t_c of K4-thr-C (LB 0.4183), fixed for K4-thr-C-ext and K4-free
K4_FREE_T_C = T_C_FIXED
CAP_EXT = 2.0  # cap of the capped queries in K4-thr-C-ext: floor(2.0 * K_q)
RANK_BANDS = ((1, 10), (11, 20), (21, 40), (41, 60))


class _Calibrated(Exception):
    """Raised by the calibrating chunk selector to stop before writing the bundle."""


def variant_k_mode(name: str, judge_tag: str | None) -> str | None:
    """K_q' mode of a variable-count variant: ``scale``/``thr``/``mix``/``tfree:<t>``, else None."""
    match = re.fullmatch(r"COS-K(\d+)", name) or (judge_tag and re.fullmatch(re.escape(judge_tag) + r"-K(\d+)", name))
    if match:
        return f"topk:{int(match.group(1))}"
    if not judge_tag:
        return None
    prefix = f"K{judge_tag[1:]}-"
    for mode in ("scale", "thr", "mix"):
        if name == f"{prefix}{mode}":
            return mode
    if name in (f"{prefix}thr-C", f"{prefix}thr-C-fx"):
        return "thr"
    if name == f"{prefix}free-p13":
        return "free-p13"
    if name == f"{prefix}thr-C-ext":
        return "thr-ext"
    if name == f"{prefix}free":
        return "free"
    match = re.fullmatch(re.escape(prefix) + r"tfree-t(\d+(?:\.\d+)?)", name)
    if match:
        return f"tfree:{float(match.group(1))}"
    return None


def variant_chunk_mode(name: str, judge_tag: str | None) -> str | None:
    """``threshold`` for K4-thr-C (t_c set so chunks = THR_C_CHUNKS), ``threshold-free``
    for K4-free-p13 (chunks = K4_FREE_CHUNKS), else None (n_c x1.4)."""
    if (variant_k_mode(name, judge_tag) or "").startswith("topk:"):
        return "all"
    if judge_tag and name == f"K{judge_tag[1:]}-thr-C":
        return "threshold"
    if judge_tag and name == f"K{judge_tag[1:]}-thr-C-fx":
        return "fixed-whole"
    if judge_tag and name == f"K{judge_tag[1:]}-free-p13":
        return "threshold-free"
    if judge_tag and name in (f"K{judge_tag[1:]}-thr-C-ext", f"K{judge_tag[1:]}-free"):
        return "fixed"
    return None


def fixed_chunk_selector(score_of: dict[int, dict[int, float]], t_c: float | None, info: dict[str, Any]):
    """``chunk_selector``: every offered carrier with text and P4B >= t_c (all if t_c is None), in RRF order."""

    def select(offered: dict[int, list[tuple[int, int, bool]]]) -> dict[int, list[tuple[int, int]]]:
        result = {qid: [(d, c) for d, c, has_text in offered_q
                        if has_text and (t_c is None or score_of[qid][d] >= t_c)]
                  for qid, offered_q in offered.items()}
        counts = np.array([len(result[q]) for q in sorted(result)])
        info["chunk_doc_ids"] = {qid: [d for d, _ in pairs] for qid, pairs in result.items()}
        info.update({"t_c": None if t_c is None else float(t_c), "chunks": int(counts.sum()),
                     "chunks_per_query": _quantiles(counts),
                     "queries_without_chunk": int((counts == 0).sum()),
                     "carriers_without_text": int(sum(1 for o in offered.values() for _, _, h in o if not h))})
        return result

    return select


def scored_union(
    ranked: dict[int, list[dict[str, Any]]],
    pool_index: pd.DataFrame,
    orig_scores: pd.DataFrame,
    extra_pairs: pd.DataFrame,
    extra_scores: pd.DataFrame,
) -> dict[int, list[dict[str, Any]]]:
    """Every pair with a 4B score per query, in cosine order (cosine_rank renumbered 1..N).

    Sources, deduplicated by doc in this priority: the extended pool13 (``ranked``), the
    original pool13 pairs evicted by the Pharmacity merge (Kaggle pool13 scores), the
    pool-extra pairs (``scores-extra.parquet``). S31 tail pairs keep their S31 cosine.
    """
    extra = extra_pairs.merge(extra_scores[["pair_id", "query_id", "doc_id", "score"]], on="pair_id",
                              how="left", suffixes=("", "_s"))
    if extra.score.isna().any():
        raise ValueError(f"scores-extra misses {int(extra.score.isna().sum())} pool-extra pairs")
    if not ((extra.query_id == extra.query_id_s).all() and (extra.doc_id == extra.doc_id_s).all()):
        raise ValueError("scores-extra pair_id/query_id/doc_id differ from extra_pairs")
    s31_cos = pool_index.set_index("pair_id").cosine_score
    tail = extra.source == "s31-tail"
    extra.loc[tail, "cosine_score"] = s31_cos.reindex(extra.loc[tail, "pair_id"]).to_numpy()
    if extra.cosine_score.isna().any():
        raise ValueError("pool-extra pair without cosine")
    ext_ids = {int(r["pair_id"]) for rows in ranked.values() for r in rows}
    evicted = pool_index[pool_index.in_pool13 & ~pool_index.pair_id.isin(ext_ids)].merge(
        orig_scores[["pair_id", "score"]], on="pair_id", how="left")
    if evicted.score.isna().any():
        raise ValueError("evicted pool13 pairs without Kaggle 4B score")
    out: dict[int, list[dict[str, Any]]] = {}
    seen: dict[int, set[int]] = {}
    for qid, rows in ranked.items():
        out[qid] = [{"pair_id": int(r["pair_id"]), "doc_id": int(r["doc_id"]), "cosine_score": float(r["cosine_score"]),
                     "judge_score": float(r["judge_score"]), "origin": "pool13", "order": (0, int(r["cosine_rank"]))}
                    for r in rows]
        seen[qid] = {r["doc_id"] for r in out[qid]}
    for origin, prio, frame in (("evicted", 1, evicted), ("extra", 2, extra)):
        for pid, qid, did, cos, score in zip(frame.pair_id, frame.query_id, frame.doc_id, frame.cosine_score, frame.score):
            qid, did = int(qid), int(did)
            if did in seen[qid]:
                continue
            seen[qid].add(did)
            out[qid].append({"pair_id": int(pid), "doc_id": did, "cosine_score": float(cos), "judge_score": float(score),
                             "origin": origin, "order": (prio, int(pid))})
    for qid, rows in out.items():
        rows.sort(key=lambda r: (-r["cosine_score"], r["order"]))
        for rank, row in enumerate(rows, 1):
            row["cosine_rank"] = rank
    return out


def _with_judge_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(r) for r in rows]
    add_judge_rank(rows, "judge_score", "judge_rank")
    return rows


def thr_ext_pools(
    ranked: dict[int, list[dict[str, Any]]], union: dict[int, list[dict[str, Any]]], base_k: dict[int, int],
    thr_keep: dict[int, int], t: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int], dict[str, Any]]:
    """K4-thr-C-ext: capped queries of K4-thr use the whole scored pool and cap floor(2.0 K_q)."""
    pools, keep, capped = {}, {}, []
    for qid, rows in ranked.items():
        k = base_k[qid]
        if thr_keep[qid] == int(np.floor(1.3 * k + 1e-9)):
            capped.append(qid)
            pool = _with_judge_rank(union[qid])
            lo, hi = int(np.ceil(0.5 * k - 1e-9)), int(np.floor(CAP_EXT * k + 1e-9))
            if hi > len(pool):
                raise ValueError(f"Query {qid}: scored pool {len(pool)} < cap {hi}")
            keep[qid] = int(np.clip(sum(1 for r in pool if r["judge_score"] >= t), lo, hi))
            pools[qid] = pool
        else:
            pools[qid] = rows
            keep[qid] = thr_keep[qid]
    k_c = np.array([keep[q] for q in capped])
    base_c = np.array([base_k[q] for q in capped])
    hi_c = np.floor(CAP_EXT * base_c + 1e-9).astype(int)
    report = {"mode": "thr-ext", "t": t, "cap_ext_x_k": CAP_EXT, "capped_queries": len(capped),
              "total": int(sum(keep.values())), "added_docs": int(k_c.sum() - np.floor(1.3 * base_c + 1e-9).astype(int).sum()),
              "capped_k_ratio": _quantiles(k_c / base_c), "capped_still_at_new_cap": int((k_c == hi_c).sum()),
              "capped_pool_size": _quantiles(np.array([len(pools[q]) for q in capped], dtype=np.float64)),
              "docs_per_query": _quantiles(np.array([keep[q] for q in sorted(keep)], dtype=np.float64))}
    return pools, keep, report


def free_adaptive(union: dict[int, list[dict[str, Any]]], deepen: bool = True) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int], dict[str, Any]]:
    """K4-free: top-P_FREE; deepen to every scored pair when the 211th doc has P4B >= T_FREE (``deepen`` False: always the top-P_FREE)."""
    pools, keep, deepened, still = {}, {}, [], []
    for qid, rows in union.items():
        if len(rows) < P_FREE:
            raise ValueError(f"Query {qid}: only {len(rows)} scored pairs < {P_FREE}")
        pool = rows[:P_FREE]
        if deepen and pool[-1]["judge_score"] >= T_FREE and len(rows) > P_FREE:
            pool = rows
            deepened.append(qid)
        if pool[-1]["judge_score"] >= T_FREE:  # bottom of the scored pool still at or above t
            still.append(qid)
        pool = _with_judge_rank(pool)
        pools[qid] = pool
        keep[qid] = int(np.clip(sum(1 for r in pool if r["judge_score"] >= T_FREE), F_FREE, len(pool)))
    qids = sorted(pools)
    new = np.array([keep[q] for q in qids])
    size = np.array([len(pools[q]) for q in qids])
    at_211 = [q for q in qids if union[q][P_FREE - 1]["judge_score"] >= T_FREE]
    report = {"mode": "free", "p": P_FREE, "floor": F_FREE, "t": T_FREE, "total": int(new.sum()),
              "queries_211th_doc_at_or_above_t": len(at_211),
              "queries_deepened": len(deepened),
              "queries_last_scored_doc_still_at_or_above_t": len(still),
              "queries_last_scored_doc_still_at_or_above_t_ids": still,
              "pool_size": _quantiles(size.astype(np.float64)),
              "docs_per_query": _quantiles(new.astype(np.float64)),
              "queries_at_floor": int((new == F_FREE).sum()), "queries_at_cap": int((new == size).sum()),
              "share_at_floor": float((new == F_FREE).mean()), "share_at_cap": float((new == size).mean())}
    return pools, keep, report


def topk_pools(union: dict[int, list[dict[str, Any]]], k: int) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    """Cosine top-P_FREE of the scored union with judge ranks inside it; K docs per query."""
    if k > P_FREE:
        raise ValueError(f"K={k} exceeds the pool of {P_FREE}")
    pools = {}
    for qid, rows in union.items():
        if len(rows) < P_FREE:
            raise ValueError(f"Query {qid}: only {len(rows)} scored pairs < {P_FREE}")
        pools[qid] = _with_judge_rank(rows[:P_FREE])
    return pools, {qid: k for qid in pools}


def selection_profile(chosen: dict[int, list[dict[str, Any]]], doc_lang: dict[int, str],
                      chunk_doc_ids: dict[int, list[int]] | None) -> dict[str, Any]:
    """vi/zh shares of the selected docs (and chunks) and mean P4B per rank band."""
    def langs(ids_by_q):
        counts: dict[str, int] = {}
        for ids in ids_by_q.values():
            for d in ids:
                counts[doc_lang[d]] = counts.get(doc_lang[d], 0) + 1
        total = sum(counts.values())
        return {"counts": counts, "share": {lang: n / total for lang, n in sorted(counts.items())}}

    bands = {}
    for lo, hi in RANK_BANDS:
        scores = [float(r["judge_score"]) for rows in chosen.values() for r in rows[lo - 1:hi]]
        if scores:
            bands[f"{lo}-{hi}"] = {"mean_p4b": float(np.mean(scores)), "pairs": len(scores)}
    return {"docs_lang": langs({q: [int(r["doc_id"]) for r in rows] for q, rows in chosen.items()}),
            "chunks_lang": langs(chunk_doc_ids) if chunk_doc_ids is not None else None,
            "mean_p4b_by_rank": bands}


def free_pool(ranked: dict[int, list[dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    """Cosine top-min(P_FREE, pool) of each query, judge ranks recomputed inside it."""
    out: dict[int, list[dict[str, Any]]] = {}
    for qid, rows in ranked.items():
        head = [dict(r) for r in sorted(rows, key=lambda r: int(r["cosine_rank"]))[:P_FREE]]
        add_judge_rank(head, "judge_score", "judge_rank")
        out[qid] = head
    return out


def free_keep(pool: dict[int, list[dict[str, Any]]]) -> tuple[dict[int, int], dict[str, Any]]:
    """K' = clip(#(P4B >= T_FREE), F_FREE, pool size) with fixed constants."""
    qids = sorted(pool)
    n = np.array([sum(1 for r in pool[q] if r["judge_score"] >= T_FREE) for q in qids])
    size = np.array([len(pool[q]) for q in qids])
    new = np.clip(n, F_FREE, size)
    report = {"mode": "free", "p": P_FREE, "floor": F_FREE, "t": T_FREE, "total": int(new.sum()),
              "pool_size": _quantiles(size.astype(np.float64)), "queries_pool_below_p": int((size < P_FREE).sum()),
              "docs_per_query": _quantiles(new.astype(np.float64)),
              "queries_at_floor": int((new == F_FREE).sum()), "queries_at_cap": int((new == size).sum()),
              "share_at_floor": float((new == F_FREE).mean()), "share_at_cap": float((new == size).mean())}
    return {q: int(k) for q, k in zip(qids, new)}, report


def threshold_chunk_selector(score_of: dict[int, dict[int, float]], target: int, info: dict[str, Any]):
    """``chunk_selector`` for ``r2ai.s29.build._build_variant``: global P4B threshold t_c.

    Offered carriers with text are ordered by P4B descending, then query ID, then RRF
    position; the first ``target`` get a chunk (written in RRF order). ``info`` receives t_c
    and the per-query chunk distribution.
    """

    def select(offered: dict[int, list[tuple[int, int, bool]]]) -> dict[int, list[tuple[int, int]]]:
        items = [(score_of[qid][doc_id], qid, pos, doc_id, corpus_id)
                 for qid, offered_q in offered.items()
                 for pos, (doc_id, corpus_id, has_text) in enumerate(offered_q) if has_text]
        if len(items) < target:
            raise ValueError(f"Only {len(items)} carriers with text for {target} chunks")
        items.sort(key=lambda item: (-item[0], item[1], item[2]))
        taken = items[:target]
        t_c = taken[-1][0]
        keep = {(qid, pos) for _, qid, pos, _, _ in taken}
        result = {qid: [(d, c) for pos, (d, c, has_text) in enumerate(offered_q) if (qid, pos) in keep]
                  for qid, offered_q in offered.items()}
        counts = np.array([len(result[q]) for q in sorted(result)])
        info["chunk_doc_ids"] = {qid: [d for d, _ in pairs] for qid, pairs in result.items()}
        info.update({
            "t_c": float(t_c),
            "chunks": int(counts.sum()),
            "carriers_with_text_offered": len(items),
            "carriers_without_text": int(sum(1 for o in offered.values() for _, _, h in o if not h)),
            "pairs_at_t_c": int(sum(1 for item in items if item[0] == t_c)),
            "pairs_at_t_c_taken": int(sum(1 for item in taken if item[0] == t_c)),
            "chunks_per_query": _quantiles(counts),
            "queries_without_chunk": int((counts == 0).sum()),
        })
        return result

    return select


def variant_rank_keys(name: str, judge_tag: str | None) -> tuple[str, ...]:
    """Rank sources of a variant; ``q06_rank`` = S29 0.6B, ``judge_rank`` = the strong judge."""
    if name == "Jcheck-q06-rrf":
        return ("cosine_rank", "q06_rank")
    if judge_tag and name == f"{judge_tag}-rrf":
        return ("cosine_rank", "judge_rank")
    if judge_tag and name == f"{judge_tag}-rrf3":
        return ("cosine_rank", "q06_rank", "judge_rank")
    if re.fullmatch(r"COS-K\d+", name):
        return ("cosine_rank",)
    if variant_k_mode(name, judge_tag):
        return ("cosine_rank", "judge_rank")
    raise ValueError(f"Unsupported S31 variant {name!r} for judge tag {judge_tag!r}")


def load_candidates(
    pool_dir: Path, judge_scores: Path | None, pool13_index: Path | None = None, q06_scores: Path | None = None
) -> pd.DataFrame:
    """Pool13 rows with q06_score (S29) and, if given, judge_score; all joined on pair_id.

    ``pool13_index``/``q06_scores`` replace the pool's 1.3x rows and the S29 scores (the
    Pharmacity-extended pool from ``r2ai.s31.pharmacity_ext``).
    """
    if pool13_index is None:
        index = pd.read_parquet(pool_dir / "pool_index.parquet")
        pool13 = index[index.in_pool13].copy()
    else:
        pool13 = pd.read_parquet(pool13_index)
    ref = pd.read_parquet(q06_scores or pool_dir / "s29_qwen06_scores.parquet")
    frame = pool13.merge(ref[["pair_id", "query_id", "doc_id", "qwen_score"]], on="pair_id", suffixes=("", "_ref"))
    if len(frame) != len(pool13) or not (
        (frame.query_id == frame.query_id_ref).all() and (frame.doc_id == frame.doc_id_ref).all()
    ):
        raise ValueError("S29 0.6B reference scores do not align with the pool13 pairs")
    frame = frame.drop(columns=["query_id_ref", "doc_id_ref"]).rename(columns={"qwen_score": "q06_score"})
    if judge_scores is not None:
        judge = pd.read_parquet(judge_scores)
        if judge.pair_id.duplicated().any():
            raise ValueError(f"{judge_scores}: duplicate pair_id")
        judge = judge[judge.pair_id.isin(frame.pair_id)]
        merged = frame.merge(judge[["pair_id", "query_id", "doc_id", "score"]], on="pair_id", suffixes=("", "_j"))
        if len(merged) != len(frame):
            raise ValueError(f"{judge_scores}: covers {len(merged)} of {len(frame)} pool13 pairs")
        if not ((merged.query_id == merged.query_id_j).all() and (merged.doc_id == merged.doc_id_j).all()):
            raise ValueError(f"{judge_scores}: query_id/doc_id differ from the pool")
        frame = merged.drop(columns=["query_id_j", "doc_id_j"]).rename(columns={"score": "judge_score"})
        if not np.isfinite(frame.judge_score).all():
            raise ValueError("Non-finite judge scores")
    return frame.sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)


def rank_candidates(frame: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for qid, group in frame.groupby("query_id", sort=True):
        rows = group.to_dict("records")
        add_judge_rank(rows, "q06_score", "q06_rank")
        if "judge_score" in group:
            add_judge_rank(rows, "judge_score", "judge_rank")
        grouped[int(qid)] = rows
    return grouped


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {"p10": float(np.quantile(values, 0.10)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)), "min": float(values.min()), "max": float(values.max())}


def variable_keep(
    ranked: dict[int, list[dict[str, Any]]], base_k: dict[int, int], mode: str, thr: dict[str, Any] | None = None
) -> tuple[dict[int, int], dict[str, Any], dict[str, Any] | None]:
    """K_q' per query for a K4-* mode; returns (keep, report, threshold result for reuse)."""
    qids = sorted(ranked)
    k = np.array([base_k[q] for q in qids], dtype=np.int64)
    target = int(k.sum())
    lo_f, hi_f = K_BOUNDS[mode.split(":")[0]]
    lo = np.ceil(lo_f * k - 1e-9).astype(np.int64)
    hi = np.floor(hi_f * k + 1e-9).astype(np.int64)
    if any(hi[i] > len(ranked[q]) for i, q in enumerate(qids)):
        raise ValueError("Upper bound exceeds the pool size")
    report: dict[str, Any] = {"mode": mode, "bounds_x_k": [lo_f, hi_f], "total": target,
                              "bounds_rule": "ceil(lo*K_q) .. floor(hi*K_q)"}
    if mode == "scale":
        keys = ("cosine_rank", "judge_rank")
        r = np.array([np.mean([row["judge_score"] for row in select_rrf(ranked[q], base_k[q], keys)]) for q in qids])
        raw = k * r / r.mean()
        new, c = allocate_total(raw, lo, hi, target)
        report.update({"r_q": _quantiles(r), "r_mean": float(r.mean()), "c": c})
    elif mode.startswith("tfree:"):
        t = float(mode.split(":")[1])
        n = np.array([sum(1 for row in ranked[q] if row["judge_score"] >= t) for q in qids], dtype=np.int64)
        new = np.clip(n, lo, hi)
        report.update({"t": t, "total": int(new.sum()), "total_rule": "free (no total constraint)",
                       "n_q_over_k": _quantiles(n / k), "docs_per_query": _quantiles(new.astype(np.float64))})
    else:
        if thr is None:
            qidx = np.concatenate([np.full(len(ranked[q]), i) for i, q in enumerate(qids)])
            scores = np.concatenate([[row["judge_score"] for row in ranked[q]] for q in qids]).astype(np.float64)
            tie = np.concatenate([[row["judge_rank"] for row in ranked[q]] for q in qids])
            thr_lo = np.ceil(K_BOUNDS["thr"][0] * k - 1e-9).astype(np.int64)
            thr_hi = np.floor(K_BOUNDS["thr"][1] * k + 1e-9).astype(np.int64)
            thr = threshold_counts(qidx, scores, tie, thr_lo, thr_hi, target)
        report.update({"t": thr["t"], "pairs_at_t": thr["pairs_at_t"],
                       "pairs_at_t_above_cut": thr["pairs_at_t_above_cut"],
                       "n_q_over_k": _quantiles(thr["n"] / k)})
        if mode == "thr":
            new = thr["k"]
        else:
            raw = np.sqrt(k * thr["n"].astype(np.float64))
            new, c = allocate_total(raw, lo, hi, target)
            report["c"] = c
    ratio = new / k
    report.update({
        "k_ratio": _quantiles(ratio),
        "queries_at_floor": int((new == lo).sum()),
        "queries_at_cap": int((new == hi).sum()),
        "share_at_floor": float((new == lo).mean()),
        "share_at_cap": float((new == hi).mean()),
        "total_check": int(new.sum()),
        "docs_per_query": _quantiles(new.astype(np.float64)),
    })
    return {q: int(n) for q, n in zip(qids, new)}, report, thr


def _scores_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in frame[["query_id", "doc_id", "cosine_rank", *columns]].itertuples(index=False):
        digest.update(json.dumps([int(row[0]), int(row[1]), int(row[2]), *[float(v) for v in row[3:]]],
                                 separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(
    out: Path,
    variants: Sequence[str],
    judge: str | None,
    judge_scores: Path | None,
    pool_dir: Path,
    pool13_index: Path | None = None,
    q06_scores: Path | None = None,
    name_suffix: str = "",
    thin_chunks: bool = False,
    calibrate_tc: bool = False,
    extra_scores: Path | None = None,
    extra_dir: Path | None = None,
    orig_scores: Path | None = None,
    rank_hook: Any = None,
    rank_keys_override: dict[str, tuple[str, ...]] | None = None,
    variant_alias: dict[str, str] | None = None,
    no_extra_chunks: bool = False,
    pool_hook: Any = None,
    chunk_t_c: float | None = None,
    chunk_mode_override: dict[str, str] | None = None,
    rank_weights: dict[str, tuple[float, ...]] | None = None,
    chunk_target: int | None = None,
    docsel_hook: Any = None,
    keep_override: dict[str, dict[int, int]] | None = None,
    free_deepen: bool = True,
) -> list[dict[str, Any]]:
    """``rank_hook(ranked)`` may add rank keys to the pool13 rows and returns a sidecar dict;
    ``rank_keys_override`` replaces the RRF keys of a variant; ``variant_alias`` maps an output
    name to the S31 variant whose keep/chunk rule it follows (all three used by ``r2ai.s35``).
    S36: ``pool_hook(name, pool)`` adds rank keys to the rows of a K4-free / top-K pool (returns a sidecar dict);
    ``chunk_t_c`` replaces ``T_C_FIXED`` in the fixed chunk selectors; ``chunk_mode_override`` maps an output name
    to its chunk mode (e.g. ``fixed-whole``); ``rank_weights`` gives weighted RRF (one weight per rank key);
    ``chunk_target`` replaces ``THR_C_CHUNKS`` for the ``threshold`` chunk mode (calibration of a new t_c);
    ``keep_override`` maps an output name to its docs-per-query (the pool rows are taken in RRF order);
    ``free_deepen`` False cuts every K4-free pool at the cosine top-P_FREE (EF2-free-d211);
    ``docsel_hook(name, chosen, chunk_doc_ids, chunk_info)`` receives the selected rows (RRF order) of each built bundle."""
    started = time.monotonic()
    judge_tag = JUDGE_TAGS.get(judge) if judge else None
    variant_alias = variant_alias or {}
    keys_by_base = {name: (rank_keys_override or {}).get(name) or variant_rank_keys(variant_alias.get(name, name), judge_tag)
                    for name in variants}
    keys_by_variant = {f"{name}{name_suffix}": keys for name, keys in keys_by_base.items()}
    variants = list(keys_by_variant)
    q06_path = q06_scores or pool_dir / "s29_qwen06_scores.parquet"
    candidate_file = pool13_index or pool_dir / "pool_index.parquet"
    if any("judge_rank" in keys for keys in keys_by_variant.values()) and judge_scores is None:
        raise ValueError("Judge variants need --scores (scores-pool13.parquet of the judge)")
    inputs = _inputs()
    for key in ("p1", "baseline", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    for path in [candidate_file, q06_path] + ([judge_scores] if judge_scores else []):
        paths.require_inputs(path)
    s29_build._preflight_outputs(out, tuple(variants))
    for name in variants:
        paths.assert_writable(out / f"{name}.s31.json")
        paths.assert_writable(out / f"{name}.s31.json.tmp")
    out.mkdir(parents=True, exist_ok=True)

    frame = load_candidates(pool_dir, judge_scores, pool13_index, q06_scores)
    ranked = rank_candidates(frame)
    hook_info = rank_hook(ranked) if rank_hook else None

    # Same context preparation as r2ai.s29.build.build.
    p1_rows = _load_submission(inputs["p1"], with_chunks=True)
    baseline = {int(row["id"]): row for row in _load_submission(inputs["baseline"])}
    if {int(row["id"]) for row in p1_rows} != set(baseline) or set(ranked) != set(baseline):
        raise ValueError("P1, K90-ph and pool13 query IDs differ")
    first_chunk_by_id: dict[int, str] = {}
    for row in p1_rows:
        for chunk in row["relevant_chunks"]:
            first_chunk_by_id.setdefault(int(chunk["doc_id"]), str(chunk["chunk_text"]))
    chunks_path = inputs["canonical"] / "chunks_t256.parquet"
    docs_path = inputs["canonical"] / "docs.parquet"
    groups, cid_to_doc = s29_build._load_aliases(chunks_path, docs_path)
    docs = pq.read_table(docs_path, columns=["doc_id", "lang", "quality"]).to_pandas()
    doc_lang = dict(zip(docs.doc_id.astype(int).tolist(), docs.lang.tolist()))
    thin = set(docs.loc[docs.quality == "thin", "doc_id"].astype(int).tolist())
    del docs
    thin_texts = None
    if thin_chunks:
        table = pq.read_table(docs_path, columns=["doc_id", "doc_text"],
                              filters=[("doc_id", "in", sorted(thin))]).to_pandas()
        thin_texts = {int(d): t for d, t in zip(table.doc_id, table.doc_text)
                      if t is not None and len(t.strip()) >= THIN_MIN_CHARS}
        del table
    extra_texts, extra_spans = s29_build._load_extra_docs(inputs)
    for doc_id in extra_texts:
        groups[doc_id] = [doc_id]
        cid_to_doc[doc_id] = doc_id
        doc_lang[doc_id] = "vi"
    p1_any, row_ids = s29_build._row_id_maps(p1_rows, cid_to_doc)

    reference = {qid: select_rrf(rows, len(baseline[qid]["relevant_docs"]), ("cosine_rank", "q06_rank"), RRF_K)
                 for qid, rows in ranked.items()}
    reference_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in reference.items()}
    candidate_digest = _sha256(candidate_file)
    score_columns = ["q06_score"] + (["judge_score"] if "judge_score" in frame else [])
    score_digest = _scores_digest(frame, score_columns)
    correlations = judge_correlations(frame.assign(cosine=frame.cosine_score), ["cosine", *score_columns])
    base_k = {qid: len(baseline[qid]["relevant_docs"]) for qid in ranked}
    modes = {f"{name}{name_suffix}": variant_k_mode(variant_alias.get(name, name), judge_tag) for name in keys_by_base}
    chunk_modes = {f"{name}{name_suffix}": variant_chunk_mode(variant_alias.get(name, name), judge_tag) for name in keys_by_base}
    chunk_modes.update(chunk_mode_override or {})
    keep_by_variant: dict[str, dict[int, int]] = {}
    keep_reports: dict[str, dict[str, Any]] = {}
    thr_cache = None
    pools: dict[str, dict[int, list[dict[str, Any]]]] = {}
    union = None
    if any(mode and (mode in ("thr-ext", "free") or mode.startswith("topk:")) for mode in modes.values()):
        if extra_scores is None or extra_dir is None or orig_scores is None:
            raise ValueError("K4-thr-C-ext / K4-free / *-K<K> need --extra-scores, --extra-dir and --orig-scores")
        union = scored_union(ranked, pd.read_parquet(pool_dir / "pool_index.parquet"), pd.read_parquet(orig_scores),
                             pd.read_parquet(extra_dir / "extra_pairs.parquet"), pd.read_parquet(extra_scores))
    for name, mode in modes.items():
        if mode == "thr-ext":
            thr_keep, _, thr_cache = variable_keep(ranked, base_k, "thr", thr_cache)
            pools[name], keep_by_variant[name], keep_reports[name] = thr_ext_pools(
                ranked, union, base_k, thr_keep, thr_cache["t"])
        elif mode == "free":
            pools[name], keep_by_variant[name], keep_reports[name] = free_adaptive(union, free_deepen)
        elif mode and mode.startswith("topk:"):
            k = int(mode.split(":")[1])
            pools[name], keep_by_variant[name] = topk_pools(union, k)
            keep_reports[name] = {"mode": "topk", "k": k, "pool": P_FREE, "total": k * len(pools[name])}
        elif mode == "free-p13":
            pools[name] = free_pool(ranked)
            keep_by_variant[name], keep_reports[name] = free_keep(pools[name])
        elif mode:
            keep_by_variant[name], keep_reports[name], thr_cache = variable_keep(ranked, base_k, mode, thr_cache)
    for name, override in (keep_override or {}).items():
        sizes = {q: len(pools.get(name, ranked)[q]) for q in override}
        if set(override) != set(ranked) or any(not 0 <= k <= sizes[q] for q, k in override.items()):
            raise ValueError(f"keep_override of {name} does not fit the pool")
        keep_by_variant[name] = {int(q): int(k) for q, k in override.items()}
        keep_reports[name] = {"mode": "custom", "total": int(sum(override.values())),
                              "docs_per_query": _quantiles(np.array([override[q] for q in sorted(override)], dtype=np.float64))}
    pool_infos = {name: pool_hook(name, pool) for name, pool in pools.items()} if pool_hook else {}
    j4_ids = None
    if keep_by_variant:
        j4_ids = {qid: [int(r["doc_id"]) for r in select_rrf(rows, base_k[qid], ("cosine_rank", "judge_rank"), RRF_K)]
                  for qid, rows in ranked.items()}
        thr_keep, _, thr_cache = variable_keep(ranked, base_k, "thr", thr_cache)
        k4thr_ids = {qid: [int(r["doc_id"]) for r in select_rrf(rows, thr_keep[qid], ("cosine_rank", "judge_rank"), RRF_K)]
                     for qid, rows in ranked.items()}

    summaries = []
    for name in variants:
        keys = keys_by_variant[name]
        keep = keep_by_variant.get(name, base_k)
        source = pools.get(name, ranked)
        weights = (rank_weights or {}).get(name)
        chosen = {qid: select_rrf(rows, keep[qid], keys, RRF_K, weights) for qid, rows in source.items()}
        chosen_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in chosen.items()}
        diff = doc_set_diff(chosen_ids, reference_ids)
        chunk_info: dict[str, Any] = {}
        selector = None
        tc_target = {"threshold": chunk_target or THR_C_CHUNKS, "threshold-free": K4_FREE_CHUNKS}.get(chunk_modes[name])
        if tc_target is not None and calibrate_tc:
            score_of = {qid: {int(r["doc_id"]): float(r["judge_score"]) for r in rows} for qid, rows in source.items()}
            calibrating = threshold_chunk_selector(score_of, tc_target, chunk_info)

            def selector(offered, _select=calibrating):
                _select(offered)
                raise _Calibrated()
        elif tc_target is not None:
            score_of = {qid: {int(r["doc_id"]): float(r["judge_score"]) for r in rows} for qid, rows in source.items()}
            selector = threshold_chunk_selector(score_of, tc_target, chunk_info)
        elif chunk_modes[name] in ("fixed", "fixed-whole", "all"):
            score_of = {qid: {int(r["doc_id"]): float(r["judge_score"]) for r in rows} for qid, rows in source.items()}
            selector = fixed_chunk_selector(score_of, None if chunk_modes[name] == "all" else (T_C_FIXED if chunk_t_c is None else chunk_t_c), chunk_info)
        if no_extra_chunks:
            if selector is None:
                raise ValueError(f"--no-extra-chunks needs a chunk-selector variant, not {name}")
            unmasked = selector

            def selector(offered, _select=unmasked):
                return _select({q: [(d, c, h and d not in extra_texts) for d, c, h in items] for q, items in offered.items()})
        try:
            summary = s29_build._build_variant(
                out, name, p1_rows, baseline, p1_any, first_chunk_by_id, row_ids, groups, cid_to_doc, doc_lang, thin,
                inputs["canonical"], extra_texts, extra_spans, chosen, candidate_digest, POOL_MULTIPLIER, score_digest,
                RRF_K, None, CHUNK_MULTIPLIER,
                WHOLE_BODY and (selector is None or chunk_modes[name] in ("all", "fixed-whole")),
                diff["mean_symmetric_difference"],
                fixed_k=name not in keep_by_variant, thin_texts=thin_texts, chunk_selector=selector,
            )
            chunk_doc_ids = chunk_info.pop("chunk_doc_ids", None)
            if docsel_hook is not None:
                docsel_hook(name, chosen, chunk_doc_ids, chunk_info)
        except _Calibrated:
            chunk_info.pop("chunk_doc_ids", None)
            result = {"name": name, "status": "calibrated_not_built", "target_chunks": tc_target,
                      "keep": keep_reports.get(name), **chunk_info}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            summaries.append(result)
            continue
        sidecar = {
            "name": name,
            "status": "built_cpu_not_submitted",
            "rank_sources": list(keys),
            "rrf_k": RRF_K,
            "judge": judge if "judge_rank" in keys else None,
            "judge_scores": str(judge_scores) if judge_scores and "judge_rank" in keys else None,
            "judge_scores_sha256": _sha256(judge_scores) if judge_scores and "judge_rank" in keys else None,
            "q06_scores": str(q06_path),
            "q06_scores_sha256": _sha256(q06_path),
            "pool_index": str(candidate_file),
            "pool_index_sha256": candidate_digest,
            "pool": "in_pool13 (round(1.3 * K_q), identical to the S29 pool)" if pool13_index is None
            else "round(1.3 * K_q) per query after merging the new Pharmacity lots (r2ai.s31.pharmacity_ext)",
            "keep_rule": "len(relevant_docs) of each K90-ph row" if name not in keep_by_variant
            else f"P4B >= {T_FREE} in cosine top-min({P_FREE}, pool13), floor {F_FREE}, cap = pool (no K_q, no n_c)"
            if modes[name] == "free-p13"
            else f"K4-thr; capped queries: whole scored pool, cap floor({CAP_EXT} K_q)" if modes[name] == "thr-ext"
            else f"top-{keep_reports[name]['k']} of cosine top-{P_FREE} by {' + '.join(keys)} (no K_q, no n_c)"
            if modes[name].startswith("topk:")
            else f"P4B >= {T_FREE}; pool top-{P_FREE}, deepened to all scored pairs if the {P_FREE}th has P4B >= t; "
                 f"floor {F_FREE}, cap = pool" if modes[name] == "free"
            else f"variable K_q' ({keep_reports[name]['mode']}), total = sum len(relevant_docs) of K90-ph",
            "keep": keep_reports.get(name),
            "profile": selection_profile(chosen, {**doc_lang, **{d: "vi" for d in extra_texts}}, chunk_doc_ids)
            if (modes[name] or "").startswith("topk:") else None,
            "vs_j4_rrf_doc_sets": doc_set_diff(chosen_ids, j4_ids) if name in keep_by_variant else None,
            "vs_k4_thr_doc_sets": doc_set_diff(chosen_ids, k4thr_ids) if name in keep_by_variant else None,
            "thin_chunks": None if thin_texts is None else {
                "min_chars": THIN_MIN_CHARS, "thin_docs_eligible": len(thin_texts),
                "thin_docs_in_corpus": len(thin), "chunks_added": summary["chunk_source"].get("thin"),
                "rule": "thin doc without chunk: whole doc_text verbatim as its chunk, counted in the n_c x1.4 cap",
            },
            "chunk_multiplier": CHUNK_MULTIPLIER if selector is None else None,
            "whole_body": WHOLE_BODY if selector is None or chunk_modes[name] in ("all", "fixed-whole")
            else "whole body unless over the ZIP ceiling (S29 cut rule)",
            "chunk_threshold": None if selector is None else {
                **chunk_info, "target_chunks": tc_target,
                "rule": "one whole-body chunk for every selected doc with text, never cut"
                if chunk_modes[name] == "all" else "one whole-body chunk per selected doc with P4B >= t_c, never cut"
                if chunk_modes[name] == "fixed-whole" else "selected docs in RRF order, one whole-body chunk per doc with P4B >= t_c",
                "body_cut_chunks": summary.get("body_cut_chunks"), "body_cut_chars": summary.get("body_cut_chars"),
            },
            "extra_chunks": "masked (pre-fix behaviour, regression only)" if no_extra_chunks
            else "Pharmacity docs carry their body (S29 builder fix)",
            "extra_scores": str(extra_scores) if name in pools and union is not None else None,
            "extra_scores_sha256": _sha256(extra_scores) if name in pools and union is not None else None,
            "vs_qrrf_doc_sets": diff,
            "reference": "RRF(cosine, 0.6B) k=60 = doc set of Q-rrf / QRRF-nc140-ph",
            "judge_correlations_pool13": correlations,
            "json_sha256": summary.get("json_sha256"),
            "zip_sha256": summary.get("zip_sha256"),
            "zip_bytes": summary.get("zip_bytes"),
            "note": "the S29 builder's build.json names the 0.6B model; this sidecar records the S31 rank sources",
        }
        if hook_info is not None:
            sidecar["rank_hook"] = hook_info
        if name in pool_infos:
            sidecar["pool_hook"] = pool_infos[name]
        if weights is not None:
            sidecar["rank_weights"] = dict(zip(keys, weights))
        if chunk_t_c is not None:
            sidecar["chunk_t_c_override"] = chunk_t_c
        if name == "Jcheck-q06-rrf" and pool13_index is None:
            sidecar["regression_target_json_sha256"] = QRRF_NC140_JSON_SHA256
            sidecar["regression_pass"] = summary.get("json_sha256") == QRRF_NC140_JSON_SHA256
        target = {("K4-thr-C", True): K4_THR_C_JSON_SHA256,
                  ("K4-thr-C-fx", False): K4_THR_C_FX_JSON_SHA256}.get((name, no_extra_chunks))
        if target and pool13_index is not None:
            sidecar["regression_target_json_sha256"] = target
            sidecar["regression_pass"] = summary.get("json_sha256") == target
        _atomic_json(out / f"{name}.s31.json", sidecar)
        print(json.dumps(sidecar, ensure_ascii=False), flush=True)
        summaries.append(sidecar)
    print(json.dumps({"seconds": round(time.monotonic() - started, 1)}), flush=True)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True,
                        help="Jcheck-q06-rrf, J4-rrf, J4-rrf3 (J8-* with --judge qwen3-reranker-8b)")
    parser.add_argument("--judge", default="qwen3-reranker-4b", choices=sorted(JUDGE_TAGS))
    parser.add_argument("--scores", type=Path, default=None, help="judge scores-pool13.parquet")
    parser.add_argument("--pool", type=Path, default=None, help="S31 pool dir")
    parser.add_argument("--out", type=Path, default=None, help="output dir below RUNS_DIR")
    parser.add_argument("--pool13-index", type=Path, default=None, help="extended 1.3x pool (pharmacity_ext)")
    parser.add_argument("--q06-scores", type=Path, default=None, help="0.6B scores matching --pool13-index")
    parser.add_argument("--name-suffix", default="", help="appended to the variant names, e.g. -ph979")
    parser.add_argument("--thin-chunks", action="store_true",
                        help="thin docs (>= 20 chars) without chunk get their whole text as chunk")
    parser.add_argument("--extra-scores", type=Path, default=None, help="4B scores-extra.parquet (pool-extra)")
    parser.add_argument("--extra-dir", type=Path, default=None, help="pool-extra dir (extra_pairs.parquet)")
    parser.add_argument("--orig-scores", type=Path, default=None,
                        help="Kaggle 4B scores-pool13.parquet (scores of the evicted pool13 pairs)")
    parser.add_argument("--no-extra-chunks", action="store_true",
                        help="regression only: offer Pharmacity docs as without text (pre-fix builder)")
    parser.add_argument("--calibrate-tc", action="store_true",
                        help="threshold-chunk variants: print the t_c reaching the chunk target, write nothing")
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s31-strong-judge"
    out = paths.resolve_path(args.out or root / "build")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S31 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    pool_dir = paths.resolve_path(args.pool or root / "pool")
    scores = paths.resolve_path(args.scores) if args.scores else None
    results = build(
        out, args.variant, args.judge, scores, pool_dir,
        paths.resolve_path(args.pool13_index) if args.pool13_index else None,
        paths.resolve_path(args.q06_scores) if args.q06_scores else None,
        args.name_suffix, args.thin_chunks, args.calibrate_tc,
        paths.resolve_path(args.extra_scores) if args.extra_scores else None,
        paths.resolve_path(args.extra_dir or root / "pool-extra"),
        paths.resolve_path(args.orig_scores or root / "gpu" / "qwen3-reranker-4b" / "scores-pool13.parquet"),
        args.no_extra_chunks,
    )
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
