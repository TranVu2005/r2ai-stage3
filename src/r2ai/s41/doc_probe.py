"""S41 (27/10): doc-axis LB probes cut from an already validated bundle, without a new compose.

    python -m r2ai.s41.doc_probe --name FREE-cut110-sub --base-zip <cut150 ZIP> --base-sha <ZIP SHA256> \
        --docsel out/runs/s40/cut/apply/docsel-FREE-lam1-check.parquet [--apply-json <apply.json>] --out out/runs/s41/probe

Doc F2 depends only on ``relevant_docs``, so a doc probe keeps, per query, the base docs that are in the probe docsel (base order)
and the chunks of those docs, byte for byte. The probe docsel must lie inside the base docs of every query (an EF2 cut with a larger
lambda on the same RRF order does); otherwise the run stops. Chunk F2 of such a probe is a side number: carriers are not reallocated
and T is not searched again. Only the Doc P / R / F2 of the LB are meant for the p_cal refit; the winner gets a full compose.

Output: ``<out>/<name>.zip`` (same writer as S38 compose: one JSON, LZMA dict 256), the S26 validator (``--max-chunks-per-doc 2``,
120ask segments as extra sources) and ``<out>/<name>.probe.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s38.build import write_bundle
from r2ai.s38.compose import run_validator
from r2ai.s38.sources import _atomic_json, _sha256, read_bundle


def subset_rows(rows: Sequence[dict[str, Any]], keep: dict[int, set[int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows with only the docs of ``keep[id]`` (base order) and their chunks. Every probe doc must be a base doc of its query and every
    base query must have a probe entry."""
    out, missing, dropped_docs, dropped_chunks = [], {}, 0, 0
    if set(keep) != {int(r["id"]) for r in rows}:
        raise ValueError("probe docsel and base bundle have different query ids")
    for r in rows:
        q = int(r["id"])
        base_docs = [int(d) for d in r["relevant_docs"]]
        lack = keep[q] - set(base_docs)
        if lack:
            missing[q] = sorted(lack)
            continue
        docs = [d for d in base_docs if d in keep[q]]
        chunks = [c for c in r["relevant_chunks"] if int(c["doc_id"]) in keep[q]]
        dropped_docs += len(base_docs) - len(docs)
        dropped_chunks += len(r["relevant_chunks"]) - len(chunks)
        out.append({**r, "relevant_docs": docs, "relevant_chunks": chunks})
    if missing:
        q = next(iter(missing))
        raise ValueError(f"probe docs outside the base docs in {len(missing)} queries, e.g. query {q}: {missing[q][:5]}")
    return out, {"dropped_docs": dropped_docs, "dropped_chunks": dropped_chunks}


def per_query(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    n = np.array([len(r[key]) for r in rows])
    return {"total": int(n.sum()), "mean": float(n.mean()), "p10": float(np.percentile(n, 10)), "p50": float(np.percentile(n, 50)),
            "p90": float(np.percentile(n, 90)), "min": int(n.min()), "max": int(n.max())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--base-sha", required=True, help="SHA256 of the base ZIP (pinned; mismatch stops)")
    parser.add_argument("--docsel", type=Path, required=True, help="apply docsel parquet of the probe (query_id, doc_id)")
    parser.add_argument("--apply-json", type=Path, default=None, help="apply report of the probe docsel (lambda, predicted)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    out = args.out.resolve()
    target = out / f"{args.name}.zip"
    paths.assert_writable(target)
    paths.assert_writable(target.with_name(target.name + ".tmp"))
    out.mkdir(parents=True, exist_ok=True)
    if _sha256(args.base_zip) != args.base_sha:
        raise ValueError(f"{args.base_zip} SHA256 differs from the pinned {args.base_sha}")
    docsel = pd.read_parquet(args.docsel, columns=["query_id", "doc_id"])
    keep = {int(q): set(g.doc_id.astype(int)) for q, g in docsel.groupby("query_id")}
    rows, info = subset_rows(read_bundle(args.base_zip), keep)
    res = write_bundle(rows, target, f"{args.name}.json")
    report: dict[str, Any] = {
        "name": args.name, "base_zip": str(args.base_zip), "base_zip_sha256": args.base_sha,
        "docsel": str(args.docsel), "docsel_sha256": _sha256(args.docsel),
        "zip": str(target), "zip_bytes": target.stat().st_size, "zip_sha256": _sha256(target), **res, **info,
        "docs_per_query": per_query(rows, "relevant_docs"), "chunks_per_query": per_query(rows, "relevant_chunks")}
    if args.apply_json is not None:
        a = json.loads(args.apply_json.read_text(encoding="utf-8"))
        report["apply"] = {"json": str(args.apply_json), "ef2_lambda": a["global"].get("ef2_lambda"), "a": a["global"]["a"],
                           "b": a["global"]["b"], "s_free": a["global"]["s_free"], "predicted": a["predicted"]}
    report["validator"] = run_validator(out, args.name, "full+w768", target, {int(d) for r in rows for d in r["relevant_docs"]})
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / f"{args.name}.probe.json", report)
    print(json.dumps({k: report[k] for k in ("name", "zip_bytes", "zip_sha256", "docs_per_query", "validator")}, ensure_ascii=False))
    return 0 if report["validator"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
