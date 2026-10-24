"""S36 rank sources for the FUSE4 bundles and their score coverage.

Four sources, all on the pool rows of a frame (rank 1 = best inside the query's pool):

* ``cosine_rank``  BGE-M3 doc-level cosine (the S31 rank of the frame),
* ``judge_rank``   Qwen3-Reranker-4B P(yes) (the S31 rank of the frame),
* ``jina_rank``    Jina Reranker v3.5 score (``r2ai.s35.jina``, blocks of 10, one pass b10s0),
* ``cmax_rank``    S32 C-max = max Qwen3-Reranker-0.6B P(yes) over the doc's top-3 t256 chunks (doc_text score
                   of the 0.6B for a doc without chunk).

A pair without a score in a source takes the **last rank of the pool** in that source (all such pairs tied at
``len(pool)``); the number of such pairs is returned by the hook and written to the bundle's sidecar.

Frames: ``thrC`` = pool13 of K4-thr-C (round(1.3 K_q) after Pharmacity 979); ``free`` = K4-free pool (cosine top-211 of the
scored union, deepened to every scored pair when the 211th doc has P4B >= t).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import add_judge_rank
from r2ai.s33.build import load_union

S36_NAME = "s36-fuse4"
FOUR_KEYS = ("cosine_rank", "judge_rank", "jina_rank", "cmax_rank")


def run_root() -> Path:
    return paths.run_dir(S36_NAME)


def s31_root() -> Path:
    return paths.RUNS_DIR / "s31-strong-judge"


def s32_doc_scores() -> Path:
    return paths.RUNS_DIR / "s32-chunk-judge" / "scores" / "doc_scores.parquet"


def s35_jina_scores() -> Path:
    return paths.RUNS_DIR / "s35-part-jina" / "jina" / "scores-all.parquet"


def load_frames(deepen: bool = True) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]], dict[int, int], dict[str, Any], dict[str, Path]]:
    """``(ranked, union, free_pools, free_keep, free_report, files)``: the pool13 rows of K4-thr-C, the scored union
    (cosine order, 4B scores), and the K4-free pools with their kept-doc counts."""
    root = s31_root()
    union, files = load_union(root)
    frame = s31_build.load_candidates(root / "pool", files["judge"], files["pool13_index"], files["q06"])
    ranked = s31_build.rank_candidates(frame)
    free, keep, report = s31_build.free_adaptive(union, deepen)
    return ranked, union, free, keep, report, files


def _lookup(frame: pd.DataFrame, column: str) -> dict[tuple[int, int], float]:
    if frame.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("duplicate (query_id, doc_id) in a score file")
    return dict(zip(zip(frame.query_id.astype(int), frame.doc_id.astype(int)), frame[column].astype(float)))


def load_jina(extra: Path | None = None) -> tuple[dict[tuple[int, int], float], dict[str, str]]:
    """Jina scores of S35 (302,086 pairs) plus the S36 supplement ``extra`` for the pairs the S35 pool lacks."""
    base_path = s35_jina_scores()
    lookup = _lookup(pd.read_parquet(base_path, columns=["query_id", "doc_id", "score"]), "score")
    shas = {str(base_path): _sha256(base_path)}
    if extra is not None and extra.exists():
        more = _lookup(pd.read_parquet(extra, columns=["query_id", "doc_id", "score"]), "score")
        clash = set(more) & set(lookup)
        if clash:
            raise ValueError(f"{len(clash)} supplement Jina pairs already scored in S35")
        lookup.update(more)
        shas[str(extra)] = _sha256(extra)
    return lookup, shas


def load_cmax(extra: Path | None = None) -> tuple[dict[tuple[int, int], float], dict[str, str]]:
    """S32 C-max (``doc_scores.parquet``: pool13 + Pharmacity lot 3) plus the S36 supplement."""
    base_path = s32_doc_scores()
    lookup = _lookup(pd.read_parquet(base_path, columns=["query_id", "doc_id", "cmax"]), "cmax")
    shas = {str(base_path): _sha256(base_path)}
    if extra is not None and extra.exists():
        more = _lookup(pd.read_parquet(extra, columns=["query_id", "doc_id", "cmax"]), "cmax")
        clash = set(more) & set(lookup)
        if clash:
            raise ValueError(f"{len(clash)} supplement C-max pairs already scored in S32")
        lookup.update(more)
        shas[str(extra)] = _sha256(extra)
    return lookup, shas


def rank_missing_last(rows: Sequence[dict[str, Any]], qid: int, lookup: dict[tuple[int, int], float],
                      score_key: str, rank_key: str) -> int:
    """Set ``rows[*][score_key]`` (None if absent) and ``rank_key``: scored rows 1..m by descending score (ties by cosine
    rank, doc_id as ``add_judge_rank``), rows without a score all at rank ``len(rows)``. Returns the number of unscored rows."""
    scored, missing = [], []
    for r in rows:
        value = lookup.get((qid, int(r["doc_id"])))
        r[score_key] = value
        (missing if value is None else scored).append(r)
    add_judge_rank(scored, score_key, rank_key)
    for r in missing:
        r[rank_key] = len(rows)
    return len(missing)


def source_hook(jina: dict[tuple[int, int], float], cmax: dict[tuple[int, int], float], strict: bool = False) -> Callable[[dict[int, list[dict[str, Any]]]], dict[str, Any]]:
    """Hook for ``rank_hook(ranked)`` / ``pool_hook(name, pools)``: adds ``jina_rank`` and ``cmax_rank`` to every row."""

    def apply(rows_by_query: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        pairs = miss_j = miss_c = 0
        queries_j: set[int] = set()
        queries_c: set[int] = set()
        for qid, rows in rows_by_query.items():
            mj = rank_missing_last(rows, qid, jina, "jina_score", "jina_rank")
            mc = rank_missing_last(rows, qid, cmax, "cmax_score", "cmax_rank")
            pairs += len(rows)
            miss_j += mj
            miss_c += mc
            if mj:
                queries_j.add(qid)
            if mc:
                queries_c.add(qid)
        if strict and (miss_j or miss_c):
            raise ValueError(f"{miss_j} pairs without Jina score, {miss_c} without C-max")
        return {"pairs": pairs, "missing_jina": miss_j, "missing_cmax": miss_c,
                "queries_missing_jina": len(queries_j), "queries_missing_cmax": len(queries_c),
                "missing_rule": "unscored pairs take the last rank of the pool in that source (all tied at len(pool))"}

    return apply


def coverage(ranked: dict[int, list[dict[str, Any]]], union: dict[int, list[dict[str, Any]]], free: dict[int, list[dict[str, Any]]],
             jina: dict[tuple[int, int], float], cmax: dict[tuple[int, int], float]) -> tuple[dict[str, Any], dict[str, set[tuple[int, int]]]]:
    """Pair counts per frame and how many lack a Jina / C-max score; also returns the missing pair sets."""
    sets = {
        "thrC pool13": {(q, int(r["doc_id"])) for q, rows in ranked.items() for r in rows},
        "free pool": {(q, int(r["doc_id"])) for q, rows in free.items() for r in rows},
        "pool211": {(q, int(r["doc_id"])) for q, rows in union.items() for r in rows[:s31_build.P_FREE]},
    }
    sets["free deepening (beyond top-211)"] = sets["free pool"] - sets["pool211"]
    report: dict[str, Any] = {}
    for name, pairs in sets.items():
        mj, mc = {p for p in pairs if p not in jina}, {p for p in pairs if p not in cmax}
        report[name] = {"pairs": len(pairs), "missing_jina": len(mj), "missing_cmax": len(mc), "missing_both": len(mj & mc),
                        "queries_missing_jina": len({q for q, _ in mj}), "queries_missing_cmax": len({q for q, _ in mc})}
    return report, {"jina": {p for p in sets["free pool"] if p not in jina}, "cmax": {p for p in sets["free pool"] if p not in cmax}}
