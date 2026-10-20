"""Build S31 judge bundles on CPU with the S29 body/chunk builder (pool 1.3x, nc140, Pharmacity).

``python -m r2ai.s31.build --variant J4-rrf --variant J4-rrf3`` reads the S31 pool
index, the S29 Qwen3-Reranker-0.6B scores (``pool/s29_qwen06_scores.parquet``) and
the strong judge's ``scores-pool13.parquet``, selects ``len(relevant_docs)`` of
K90-ph per query by multi-source RRF (k=60), and writes each bundle with
``r2ai.s29.build._build_variant`` (n_c x1.4, whole bodies, compact JSON).

``Jcheck-q06-rrf`` = RRF(cosine, 0.6B) through this path; its JSON must equal the
S29 ``QRRF-nc140-ph`` JSON (regression gate for the S31 path, needs no GPU scores).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission, _sha256
from r2ai.s31.select import RRF_K, add_judge_rank, doc_set_diff, judge_correlations, select_rrf

CHUNK_MULTIPLIER = 1.4
WHOLE_BODY = True
POOL_MULTIPLIER = 1.3
QRRF_NC140_JSON_SHA256 = "da6cf34bbb71b25a1662c904f9db4e7a0e33408301137ca5c73c335595d63fa7"
JUDGE_TAGS = {"qwen3-reranker-4b": "J4", "qwen3-reranker-8b": "J8"}


def variant_rank_keys(name: str, judge_tag: str | None) -> tuple[str, ...]:
    """Rank sources of a variant; ``q06_rank`` = S29 0.6B, ``judge_rank`` = the strong judge."""
    if name == "Jcheck-q06-rrf":
        return ("cosine_rank", "q06_rank")
    if judge_tag and name == f"{judge_tag}-rrf":
        return ("cosine_rank", "judge_rank")
    if judge_tag and name == f"{judge_tag}-rrf3":
        return ("cosine_rank", "q06_rank", "judge_rank")
    raise ValueError(f"Unsupported S31 variant {name!r} for judge tag {judge_tag!r}")


def load_candidates(pool_dir: Path, judge_scores: Path | None) -> pd.DataFrame:
    """Pool13 rows with q06_score (S29) and, if given, judge_score; all joined on pair_id."""
    index = pd.read_parquet(pool_dir / "pool_index.parquet")
    pool13 = index[index.in_pool13].copy()
    ref = pd.read_parquet(pool_dir / "s29_qwen06_scores.parquet")
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


def _scores_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in frame[["query_id", "doc_id", "cosine_rank", *columns]].itertuples(index=False):
        digest.update(json.dumps([int(row[0]), int(row[1]), int(row[2]), *[float(v) for v in row[3:]]],
                                 separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(out: Path, variants: Sequence[str], judge: str | None, judge_scores: Path | None, pool_dir: Path) -> list[dict[str, Any]]:
    started = time.monotonic()
    judge_tag = JUDGE_TAGS.get(judge) if judge else None
    keys_by_variant = {name: variant_rank_keys(name, judge_tag) for name in variants}
    if any("judge_rank" in keys for keys in keys_by_variant.values()) and judge_scores is None:
        raise ValueError("Judge variants need --scores (scores-pool13.parquet of the judge)")
    inputs = _inputs()
    for key in ("p1", "baseline", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    for path in [pool_dir / "pool_index.parquet", pool_dir / "s29_qwen06_scores.parquet"] + ([judge_scores] if judge_scores else []):
        paths.require_inputs(path)
    s29_build._preflight_outputs(out, tuple(variants))
    for name in variants:
        paths.assert_writable(out / f"{name}.s31.json")
        paths.assert_writable(out / f"{name}.s31.json.tmp")
    out.mkdir(parents=True, exist_ok=True)

    frame = load_candidates(pool_dir, judge_scores)
    ranked = rank_candidates(frame)

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
    extra_texts, extra_spans = s29_build._load_extra_docs(inputs)
    for doc_id in extra_texts:
        groups[doc_id] = [doc_id]
        cid_to_doc[doc_id] = doc_id
        doc_lang[doc_id] = "vi"
    p1_any, row_ids = s29_build._row_id_maps(p1_rows, cid_to_doc)

    reference = {qid: select_rrf(rows, len(baseline[qid]["relevant_docs"]), ("cosine_rank", "q06_rank"), RRF_K)
                 for qid, rows in ranked.items()}
    reference_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in reference.items()}
    candidate_digest = _sha256(pool_dir / "pool_index.parquet")
    score_columns = ["q06_score"] + (["judge_score"] if "judge_score" in frame else [])
    score_digest = _scores_digest(frame, score_columns)
    correlations = judge_correlations(frame.assign(cosine=frame.cosine_score), ["cosine", *score_columns])

    summaries = []
    for name in variants:
        keys = keys_by_variant[name]
        chosen = {qid: select_rrf(rows, len(baseline[qid]["relevant_docs"]), keys, RRF_K) for qid, rows in ranked.items()}
        chosen_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in chosen.items()}
        diff = doc_set_diff(chosen_ids, reference_ids)
        summary = s29_build._build_variant(
            out, name, p1_rows, baseline, p1_any, first_chunk_by_id, row_ids, groups, cid_to_doc, doc_lang, thin,
            inputs["canonical"], extra_texts, extra_spans, chosen, candidate_digest, POOL_MULTIPLIER, score_digest,
            RRF_K, None, CHUNK_MULTIPLIER, WHOLE_BODY, diff["mean_symmetric_difference"],
        )
        sidecar = {
            "name": name,
            "status": "built_cpu_not_submitted",
            "rank_sources": list(keys),
            "rrf_k": RRF_K,
            "judge": judge if "judge_rank" in keys else None,
            "judge_scores": str(judge_scores) if judge_scores and "judge_rank" in keys else None,
            "judge_scores_sha256": _sha256(judge_scores) if judge_scores and "judge_rank" in keys else None,
            "q06_scores_sha256": _sha256(pool_dir / "s29_qwen06_scores.parquet"),
            "pool_index_sha256": candidate_digest,
            "pool": "in_pool13 (round(1.3 * K_q), identical to the S29 pool)",
            "keep_rule": "len(relevant_docs) of each K90-ph row",
            "chunk_multiplier": CHUNK_MULTIPLIER,
            "whole_body": WHOLE_BODY,
            "vs_qrrf_doc_sets": diff,
            "reference": "RRF(cosine, 0.6B) k=60 = doc set of Q-rrf / QRRF-nc140-ph",
            "judge_correlations_pool13": correlations,
            "json_sha256": summary.get("json_sha256"),
            "zip_sha256": summary.get("zip_sha256"),
            "zip_bytes": summary.get("zip_bytes"),
            "note": "the S29 builder's build.json names the 0.6B model; this sidecar records the S31 rank sources",
        }
        if name == "Jcheck-q06-rrf":
            sidecar["regression_target_json_sha256"] = QRRF_NC140_JSON_SHA256
            sidecar["regression_pass"] = summary.get("json_sha256") == QRRF_NC140_JSON_SHA256
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
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s31-strong-judge"
    out = paths.resolve_path(args.out or root / "build")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S31 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    pool_dir = paths.resolve_path(args.pool or root / "pool")
    scores = paths.resolve_path(args.scores) if args.scores else None
    results = build(out, args.variant, args.judge, scores, pool_dir)
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
