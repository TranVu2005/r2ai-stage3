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

``--thin-chunks`` (off by default): thin docs (quality ``thin``) with at least
``THIN_MIN_CHARS`` characters of text and no chunk get their whole text verbatim as
the chunk; they count toward the n_c x1.4 cap like any other doc.
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
QRRF_NC140_JSON_SHA256 = "da6cf34bbb71b25a1662c904f9db4e7a0e33408301137ca5c73c335595d63fa7"
JUDGE_TAGS = {"qwen3-reranker-4b": "J4", "qwen3-reranker-8b": "J8"}
K_BOUNDS = {"scale": (0.6, 1.3), "thr": (0.5, 1.3), "mix": (0.5, 1.3)}
THIN_MIN_CHARS = 20


def variant_k_mode(name: str, judge_tag: str | None) -> str | None:
    """``scale``/``thr``/``mix`` for the variable-count variants (K4-*), else None."""
    if judge_tag:
        for mode in K_BOUNDS:
            if name == f"K{judge_tag[1:]}-{mode}":
                return mode
    return None


def variant_rank_keys(name: str, judge_tag: str | None) -> tuple[str, ...]:
    """Rank sources of a variant; ``q06_rank`` = S29 0.6B, ``judge_rank`` = the strong judge."""
    if name == "Jcheck-q06-rrf":
        return ("cosine_rank", "q06_rank")
    if judge_tag and name == f"{judge_tag}-rrf":
        return ("cosine_rank", "judge_rank")
    if judge_tag and name == f"{judge_tag}-rrf3":
        return ("cosine_rank", "q06_rank", "judge_rank")
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
    lo_f, hi_f = K_BOUNDS[mode]
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
) -> list[dict[str, Any]]:
    started = time.monotonic()
    judge_tag = JUDGE_TAGS.get(judge) if judge else None
    keys_by_base = {name: variant_rank_keys(name, judge_tag) for name in variants}
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
    modes = {f"{name}{name_suffix}": variant_k_mode(name, judge_tag) for name in keys_by_base}
    keep_by_variant: dict[str, dict[int, int]] = {}
    keep_reports: dict[str, dict[str, Any]] = {}
    thr_cache = None
    for name, mode in modes.items():
        if mode:
            keep_by_variant[name], keep_reports[name], thr_cache = variable_keep(ranked, base_k, mode, thr_cache)
    j4_ids = None
    if keep_by_variant:
        j4_ids = {qid: [int(r["doc_id"]) for r in select_rrf(rows, base_k[qid], ("cosine_rank", "judge_rank"), RRF_K)]
                  for qid, rows in ranked.items()}

    summaries = []
    for name in variants:
        keys = keys_by_variant[name]
        keep = keep_by_variant.get(name, base_k)
        chosen = {qid: select_rrf(rows, keep[qid], keys, RRF_K) for qid, rows in ranked.items()}
        chosen_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in chosen.items()}
        diff = doc_set_diff(chosen_ids, reference_ids)
        summary = s29_build._build_variant(
            out, name, p1_rows, baseline, p1_any, first_chunk_by_id, row_ids, groups, cid_to_doc, doc_lang, thin,
            inputs["canonical"], extra_texts, extra_spans, chosen, candidate_digest, POOL_MULTIPLIER, score_digest,
            RRF_K, None, CHUNK_MULTIPLIER, WHOLE_BODY, diff["mean_symmetric_difference"],
            fixed_k=name not in keep_by_variant, thin_texts=thin_texts,
        )
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
            else f"variable K_q' ({keep_reports[name]['mode']}), total = sum len(relevant_docs) of K90-ph",
            "keep": keep_reports.get(name),
            "vs_j4_rrf_doc_sets": doc_set_diff(chosen_ids, j4_ids) if name in keep_by_variant else None,
            "thin_chunks": None if thin_texts is None else {
                "min_chars": THIN_MIN_CHARS, "thin_docs_eligible": len(thin_texts),
                "thin_docs_in_corpus": len(thin), "chunks_added": summary["chunk_source"].get("thin"),
                "rule": "thin doc without chunk: whole doc_text verbatim as its chunk, counted in the n_c x1.4 cap",
            },
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
        if name == "Jcheck-q06-rrf" and pool13_index is None:
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
    parser.add_argument("--pool13-index", type=Path, default=None, help="extended 1.3x pool (pharmacity_ext)")
    parser.add_argument("--q06-scores", type=Path, default=None, help="0.6B scores matching --pool13-index")
    parser.add_argument("--name-suffix", default="", help="appended to the variant names, e.g. -ph979")
    parser.add_argument("--thin-chunks", action="store_true",
                        help="thin docs (>= 20 chars) without chunk get their whole text as chunk")
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
        args.name_suffix, args.thin_chunks,
    )
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
