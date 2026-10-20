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
