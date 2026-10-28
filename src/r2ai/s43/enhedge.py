"""S43 (27/10, Vu): EN-hedge, en docs appended at the END of ``relevant_docs`` of a finished ZIP, one full-body chunk each.

    python -m r2ai.s43.enhedge threshold --out out/runs/s43/probes/en
    python -m r2ai.s43.enhedge public --n 3 --min-cos 0.5226 --name EN-hedge3 --out out/runs/s43/probes

The public queries have no en gold (EN-only-Q Doc P = 0, S26), so the LB loss of a public hedge = the price of the same hedge on the private
queries. Per query up to ``n`` en docs of the en lane (``r2ai.s36.enlane``: BGE-M3 doc cosine over ``seg37900-en-pubmed``), in cosine order,
only those with cosine >= ``min_cos``. The vi/zh part of the ZIP is not touched: the en ids follow the base ``relevant_docs``, their chunks
follow the base ``relevant_chunks``. Chunk = the body span of ``r2ai.s38.window._body_span`` (en has no t256 spans: the whole doc_text,
title + abstract), no window (body tokens > T counted only).

``threshold``: the percentiles of the en cosine over every (query, en doc) pair (histogram, step 1e-4) and over the 24,000 top-20 lane pairs.
``public``: the CH-rrf ZIP (``s41/ch``, T 1536) + the hedge from ``s36-fuse4/en/pairs_en.parquet`` (top-20 lane, so any n <= 20).
``r2ai.newq build --en-hedge N [--en-min-cos X]`` applies the same step to the best ZIP of a new query set (X default: P99 over all pairs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s36 import enlane

CH_RRF_ZIP = paths.RUNS_DIR / "s41" / "ch" / "compose" / "CH-rrf-full+w768-t1536.zip"
CH_RRF_SHA = "b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8"
LANE = paths.RUNS_DIR / "s36-fuse4" / "en" / "pairs_en.parquet"
EDGES = np.round(np.linspace(-0.2, 1.0, 12_001), 4)
T_FULL = 1536


def all_pairs_percentiles(q_dense: np.ndarray, pcts: Sequence[float] = (50, 90, 99, 99.9), block: int = 20_000) -> dict[str, Any]:
    """Percentiles of the cosine over every (query, en doc) pair (upper bin edge of a 1e-4 histogram)."""
    _, vectors = enlane.load_vectors()
    q32 = q_dense.astype(np.float32)
    hist = np.zeros(len(EDGES) - 1, np.int64)
    for s in range(0, len(vectors), block):
        hist += np.histogram((vectors[s:s + block].astype(np.float32) @ q32.T).ravel(), EDGES)[0]
    cum = np.cumsum(hist) / hist.sum()
    return {"pairs": int(hist.sum()), **{f"p{p:g}": float(EDGES[1:][np.searchsorted(cum, p / 100)]) for p in pcts}}


def lane_pairs(q_ids: np.ndarray, q_dense: np.ndarray, n: int) -> pd.DataFrame:
    ids, vectors = enlane.load_vectors()
    return enlane.top_pairs(q_ids, q_dense, ids, vectors, n)


def choose(lane: pd.DataFrame, n: int, min_cos: float) -> pd.DataFrame:
    """``query_id, doc_id, en_rank, cosine_score``: per query the first ``n`` lane docs in en_rank order with cosine >= ``min_cos``."""
    frame = lane[lane.cosine_score >= min_cos].sort_values(["query_id", "en_rank"], kind="stable")
    frame = frame[frame.groupby("query_id").cumcount() < n]
    return frame[["query_id", "doc_id", "en_rank", "cosine_score"]].reset_index(drop=True)


def en_texts(ids) -> dict[int, str]:
    table = pq.read_table(enlane.SEG / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", sorted({int(d) for d in ids}))])
    return dict(zip(table["doc_id"].to_pylist(), table["doc_text"].to_pylist()))


def chunk_of(text: str) -> str:
    from r2ai.s38.window import _body_span

    s, e = _body_span(text, "en", pd.DataFrame({"field": [], "char_start": [], "char_end": []}))
    return text[s:e]


def append(rows: list[dict[str, Any]], hedge: pd.DataFrame, text: dict[int, str]) -> dict[str, Any]:
    """Append the hedge docs (and one chunk each) to ``rows`` in place; a doc already in a row is skipped."""
    by_q = {int(q): g.doc_id.astype(int).tolist() for q, g in hedge.groupby("query_id", sort=False)}
    added, present = {}, 0
    for r in rows:
        have = set(r["relevant_docs"])
        new = [d for d in by_q.get(int(r["id"]), []) if d not in have]
        present += len(by_q.get(int(r["id"]), [])) - len(new)
        r["relevant_docs"] = r["relevant_docs"] + new
        r["relevant_chunks"] = r["relevant_chunks"] + [{"doc_id": d, "chunk_text": chunk_of(text[d])} for d in new]
        added[int(r["id"])] = len(new)
    counts = pd.Series(added)
    return {"added": int(counts.sum()), "already_present": present, "per_query_mean": round(float(counts.mean()), 3),
            "queries_ge1": int((counts > 0).sum()), "per_query_dist": {int(k): int(v) for k, v in counts.value_counts().sort_index().items()}}


def write_extra(target: Path, text: dict[int, str]) -> Path:
    """Validator extra source (S15 lot layout) holding the en docs of the hedge."""
    dest = target / "lots" / enlane.SEG.name / "chunks" / "docs.parquet"
    paths.assert_writable(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"doc_id": list(text), "doc_text": list(text.values())}).to_parquet(dest, index=False)
    return target


def run(base_zip: Path, hedge: pd.DataFrame, out: Path, label: str, base_sha: str | None = None) -> dict[str, Any]:
    """Base ZIP + hedge -> ``<out>/<label>.zip`` (validated), report ``<out>/<label>.enhedge.json``."""
    from r2ai.gold_check.gold_check_common import load_tokenizer
    from r2ai.s29.qwen_judge import _atomic_json, _sha256
    from r2ai.s38 import compose
    from r2ai.s38.build import write_bundle
    from r2ai.s38.sources import read_bundle

    report: dict[str, Any] = {"label": label, "base_zip": str(base_zip), "base_zip_sha256": _sha256(base_zip)}
    if base_sha is not None and report["base_zip_sha256"] != base_sha:
        raise ValueError(f"{base_zip} is not {base_sha}")
    out.mkdir(parents=True, exist_ok=True)
    text = en_texts(hedge.doc_id)
    missing = set(hedge.doc_id.astype(int)) - set(text)
    if missing:
        raise ValueError(f"{len(missing)} hedge docs without en text")
    tok = load_tokenizer()
    bt = np.array([len(x) for x in tok([chunk_of(text[d]) for d in text], add_special_tokens=False)["input_ids"]]) if text else np.zeros(0)
    report["hedge"] = {"pairs": len(hedge), "docs": len(text), "cosine_min": float(hedge.cosine_score.min()) if len(hedge) else None,
                       "body_tokens_median": float(np.median(bt)) if len(bt) else None, f"body_tokens_gt{T_FULL}": int((bt > T_FULL).sum())}
    rows = read_bundle(base_zip)
    base_docs = {int(r["id"]): list(r["relevant_docs"]) for r in rows}
    report["added"] = append(rows, hedge, text)
    report["base_docs_prefix_kept"] = all(r["relevant_docs"][:len(base_docs[int(r["id"])])] == base_docs[int(r["id"])] for r in rows)
    target = out / f"{label}.zip"
    paths.assert_writable(target)
    report.update(write_bundle(rows, target, f"{label}.json"))
    report.update({"zip": str(target), "zip_bytes": target.stat().st_size, "zip_sha256": _sha256(target),
                   "limit_bytes": compose.BTC_CEILING - compose.RESERVE, "under_ceiling": target.stat().st_size < compose.BTC_CEILING - compose.RESERVE,
                   "docs": sum(len(r["relevant_docs"]) for r in rows), "chunks": sum(len(r["relevant_chunks"]) for r in rows)})
    ids = {int(d) for r in rows for d in r["relevant_docs"]}
    del rows
    extra = write_extra(out / f"{label}.validator-extra-en", text)
    report["validator"] = compose.run_validator(out, label, "full+w768", target, ids, [extra])
    _atomic_json(out / f"{label}.enhedge.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("threshold")
    t.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("public")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--min-cos", type=float, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "threshold":
        lane = pd.read_parquet(LANE)
        res = {"all_pairs": all_pairs_percentiles(enlane.load_queries()[1]),
               "top20_lane": {"pairs": len(lane), **{f"p{q}": float(np.percentile(lane.cosine_score, q)) for q in (50, 90, 99)}}}
        for key, thr in (("all_pairs_p99", res["all_pairs"]["p99"]), ("top20_lane_p99", res["top20_lane"]["p99"])):
            c = choose(lane, 3, thr).groupby("query_id").size().reindex(sorted(lane.query_id.unique()), fill_value=0)
            res[f"n3_at_{key}"] = {"min_cos": thr, "docs": int(c.sum()), "per_query_mean": round(float(c.mean()), 3), "queries_ge1": int((c > 0).sum()),
                                   "per_query_dist": {int(k): int(v) for k, v in c.value_counts().sort_index().items()}}
        out = args.out.resolve()
        paths.assert_writable(out / "threshold.json")
        out.mkdir(parents=True, exist_ok=True)
        (out / "threshold.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    else:
        work = args.out.resolve() / args.name
        paths.assert_writable(work)
        hedge = choose(pd.read_parquet(LANE), args.n, args.min_cos)
        work.mkdir(parents=True, exist_ok=True)
        hedge.to_parquet(work / "hedge.parquet", index=False)
        res = {"n": args.n, "min_cos": args.min_cos, **run(CH_RRF_ZIP, hedge, work, f"{args.name}-full+w768-t1536", CH_RRF_SHA)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
