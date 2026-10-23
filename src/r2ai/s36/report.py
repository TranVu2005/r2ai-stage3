"""S36 numbers for the report: bundle statistics and differences, and the agreement of the four rank sources.

``python -m r2ai.s36.report bundles --bundle FUSE4-thrC=<zip> ... --ref K4-thr-C-fx=<zip>``
    docs / chunks / language shares per bundle and doc-set / chunk-set differences with every ``--ref`` bundle
``python -m r2ai.s36.report sources``
    Spearman (global and mean per query) between cosine, 4B, Jina and C-max on the pairs of the thrC and free frames
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission
from r2ai.s31.select import doc_set_diff
from r2ai.s36 import fuse


def _doc_lang() -> dict[int, str]:
    docs = pq.read_table(_inputs()["canonical"] / "docs.parquet", columns=["doc_id", "lang"]).to_pandas()
    return dict(zip(docs.doc_id.astype(int), docs.lang))


def load_bundle(path: Path) -> dict[str, dict[int, list[int]]]:
    rows = _load_submission(path, with_chunks=True)
    return {"docs": {r["id"]: r["relevant_docs"] for r in rows},
            "chunks": {r["id"]: [int(c["doc_id"]) for c in r["relevant_chunks"]] for r in rows}}


def bundle_stats(bundle: dict[str, dict[int, list[int]]], lang: dict[int, str]) -> dict[str, Any]:
    docs, chunks = bundle["docs"], bundle["chunks"]
    counts = np.array([len(v) for v in docs.values()])
    chunk_counts = np.array([len(v) for v in chunks.values()])
    langs: dict[str, int] = {}
    for ids in docs.values():
        for d in ids:
            key = lang.get(d, "vi")  # Pharmacity docs are not in docs.parquet
            langs[key] = langs.get(key, 0) + 1
    total = sum(langs.values())
    return {"docs": int(counts.sum()), "chunks": int(chunk_counts.sum()),
            "docs_per_query": {"p10": float(np.quantile(counts, .1)), "median": float(np.median(counts)), "p90": float(np.quantile(counts, .9))},
            "chunks_per_query": {"p10": float(np.quantile(chunk_counts, .1)), "median": float(np.median(chunk_counts)), "p90": float(np.quantile(chunk_counts, .9))},
            "docs_lang_share": {k: v / total for k, v in sorted(langs.items())}}


def chunk_diff(a: dict[int, list[int]], b: dict[int, list[int]]) -> dict[str, float]:
    """Chunk carriers only in a / only in b, per query (chunks are one whole body per doc)."""
    only_a = [len(set(a[q]) - set(b[q])) for q in a]
    only_b = [len(set(b[q]) - set(a[q])) for q in a]
    return {"only_in_a_per_query": float(np.mean(only_a)), "only_in_b_per_query": float(np.mean(only_b)),
            "only_in_a_total": int(sum(only_a)), "only_in_b_total": int(sum(only_b)),
            "queries_identical": int(sum(1 for q in a if set(a[q]) == set(b[q])))}


def bundles_command(named: dict[str, Path], refs: dict[str, Path]) -> dict[str, Any]:
    lang = _doc_lang()
    loaded = {name: load_bundle(path) for name, path in {**refs, **named}.items()}
    result: dict[str, Any] = {}
    for name in named:
        entry = {"path": str(named[name]), **bundle_stats(loaded[name], lang), "vs": {}}
        for ref in refs:
            entry["vs"][ref] = {"docs": doc_set_diff({q: v for q, v in loaded[name]["docs"].items()}, loaded[ref]["docs"]),
                                "chunks": chunk_diff(loaded[name]["chunks"], loaded[ref]["chunks"])}
        result[name] = entry
    for name in refs:
        result.setdefault(name, {"path": str(refs[name]), **bundle_stats(loaded[name], lang)})
    return result


def spearman_table(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    out = {}
    for left, right in combinations(columns, 2):
        per_query = frame.groupby("query_id")[[left, right]].apply(lambda g: g[left].corr(g[right], method="spearman"), include_groups=False)
        out[f"{left}~{right}"] = {"n": int(len(frame)), "global": float(frame[left].corr(frame[right], method="spearman")),
                                  "per_query_mean": float(per_query.mean())}
    return out


def sources_command() -> dict[str, Any]:
    ranked, union, free, keep, free_report, _ = fuse.load_frames()
    jina, _ = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
    cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")

    def frame_of(rows_by_q: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
        rows = [{"query_id": q, "cosine": float(r["cosine_score"]), "p4b": float(r["judge_score"]),
                 "jina": jina.get((q, int(r["doc_id"]))), "cmax": cmax.get((q, int(r["doc_id"])))}
                for q, rs in rows_by_q.items() for r in rs]
        return pd.DataFrame(rows)

    out = {}
    for name, rows_by_q in (("thrC pool13", ranked), ("free pool", free)):
        frame = frame_of(rows_by_q)
        complete = frame.dropna()
        out[name] = {"pairs": len(frame), "pairs_with_all_four": len(complete),
                     "spearman": spearman_table(complete, ["cosine", "p4b", "jina", "cmax"])}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("bundles", "sources"))
    parser.add_argument("--bundle", action="append", default=[], help="name=zip")
    parser.add_argument("--ref", action="append", default=[], help="name=zip")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "bundles":
        split = lambda items: {k: Path(v) for k, v in (item.split("=", 1) for item in items)}  # noqa: E731
        result = bundles_command(split(args.bundle), split(args.ref))
        default = "bundles.json"
    else:
        result = sources_command()
        default = "sources.json"
    target = paths.resolve_path(args.out) if args.out else fuse.run_root() / default
    _atomic_json(target, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
