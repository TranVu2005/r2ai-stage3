"""S31 selection rules (multi-judge RRF) and between-variant/between-judge statistics."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

RRF_K = 60


def add_judge_rank(rows: Sequence[dict[str, Any]], score_key: str, rank_key: str) -> None:
    """1-based rank by descending score; ties by cosine rank then doc_id (S29 ``qwen_rank`` rule)."""
    ordered = sorted(rows, key=lambda r: (-float(r[score_key]), int(r["cosine_rank"]), int(r["doc_id"])))
    for rank, row in enumerate(ordered, 1):
        row[rank_key] = rank


def select_rrf(
    rows: Sequence[dict[str, Any]], keep: int, rank_keys: Sequence[str], rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    """Keep the top ``keep`` by sum of 1/(k + rank) over ``rank_keys`` (first key must be cosine_rank).

    With ``("cosine_rank", "qwen_rank")`` this is exactly ``r2ai.s29.qwen_judge.select_qrrf``:
    same summation order and tie order (cosine rank, judge rank(s), doc_id).
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if not rank_keys or rank_keys[0] != "cosine_rank":
        raise ValueError("rank_keys must start with cosine_rank")
    if not 0 <= keep <= len(rows):
        raise ValueError(f"keep={keep} is outside candidate count {len(rows)}")
    ranked = []
    for row in rows:
        score = 0.0
        for key in rank_keys:
            score = score + 1.0 / (rrf_k + int(row[key]))
        ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], *[int(item[1][key]) for key in rank_keys], int(item[1]["doc_id"])))
    return [{**row, "rrf_score": score} for score, row in ranked[:keep]]


def allocate_total(
    raw: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: int, iters: int = 200
) -> tuple[np.ndarray, float]:
    """Integer counts in [lo, hi] summing to ``target``, proportional to ``c * raw`` clipped.

    ``c`` is found by bisection on the continuous clipped sum; the integer remainder goes
    to the largest fractional parts (ties by position). Returns the counts and ``c``.
    """
    raw, lo, hi = (np.asarray(x, dtype=np.float64) for x in (raw, lo, hi))
    if (raw < 0).any() or (lo > hi).any():
        raise ValueError("raw must be nonnegative and lo <= hi")
    if not lo.sum() <= target <= hi.sum():
        raise ValueError(f"target {target} outside [{lo.sum()}, {hi.sum()}]")

    def total(c: float) -> float:
        return float(np.clip(c * raw, lo, hi).sum())

    a, b = 0.0, 1.0
    while total(b) < target:
        b *= 2.0
    for _ in range(iters):
        mid = (a + b) / 2.0
        if total(mid) < target:
            a = mid
        else:
            b = mid
    x = np.clip(b * raw, lo, hi)
    counts = np.floor(x + 1e-9).astype(np.int64)
    counts = np.minimum(counts, hi.astype(np.int64))
    remainder = int(target - counts.sum())
    if remainder < 0:
        raise ValueError("allocation overshoot")
    open_ = np.flatnonzero(counts < hi)
    frac = (x - counts)[open_]
    order = open_[np.lexsort((open_, -frac))]
    if remainder > len(order):
        raise ValueError("allocation cannot reach target")
    counts[order[:remainder]] += 1
    if counts.sum() != target or (counts < lo).any() or (counts > hi).any():
        raise ValueError("allocation failed")
    return counts, b


def threshold_counts(
    qidx: np.ndarray, scores: np.ndarray, tiebreak: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: int
) -> dict[str, Any]:
    """Global judge threshold t with clip(n_q(t), lo, hi) summing exactly to ``target``.

    Pairs are ordered by score descending, then query position, then ``tiebreak``; the
    threshold is a cut position in that order, so equal scores at t are split by the
    tie order. ``n`` = pairs per query above the cut (unclipped), ``k`` = clipped counts.
    """
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    order = np.lexsort((tiebreak, qidx, -scores))
    q_sorted = qidx[order]

    def counts(i: int) -> np.ndarray:
        return np.bincount(q_sorted[:i], minlength=len(lo))

    def total(i: int) -> int:
        return int(np.clip(counts(i), lo, hi).sum())

    if not total(0) <= target <= total(len(order)):
        raise ValueError(f"target {target} outside [{total(0)}, {total(len(order))}]")
    a, b = 0, len(order)
    while a < b:
        mid = (a + b) // 2
        if total(mid) < target:
            a = mid + 1
        else:
            b = mid
    n = counts(a)
    k = np.clip(n, lo, hi)
    if int(k.sum()) != target:
        raise ValueError("threshold search failed")
    t = float(scores[order[a - 1]]) if a else float("inf")
    return {
        "k": k,
        "n": n,
        "cut": a,
        "t": t,
        "pairs_at_t": int((scores == t).sum()),
        "pairs_at_t_above_cut": int((scores[order[:a]] == t).sum()),
    }


def doc_set_diff(a: dict[int, Sequence[int]], b: dict[int, Sequence[int]]) -> dict[str, float]:
    """Per-query doc-ID differences between two selections with the same queries."""
    if set(a) != set(b):
        raise ValueError("Selections cover different queries")
    only_a, only_b = [], []
    for qid in a:
        sa, sb = set(a[qid]), set(b[qid])
        only_a.append(len(sa - sb))
        only_b.append(len(sb - sa))
    return {
        "mean_docs_only_in_first": float(np.mean(only_a)),
        "mean_docs_only_in_second": float(np.mean(only_b)),
        "mean_symmetric_difference": float(np.mean(np.add(only_a, only_b))),
        "queries_identical": int(sum(1 for x, y in zip(only_a, only_b) if x == 0 and y == 0)),
    }


def judge_correlations(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Global and mean per-query Spearman between score columns (higher = more relevant)."""
    result: dict[str, Any] = {}
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            per_query = frame.groupby("query_id").apply(
                lambda g: g[left].corr(g[right], method="spearman"), include_groups=False
            )
            result[f"{left}~{right}"] = {
                "spearman_global": float(frame[left].corr(frame[right], method="spearman")),
                "spearman_per_query_mean": float(per_query.mean()),
                "spearman_per_query_p10": float(per_query.quantile(0.10)),
            }
    return result
