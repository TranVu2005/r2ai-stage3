"""Build ``JINA-K<K>``: top-K of the S31 cosine top-211 pool ranked by Jina Reranker v3.5 alone.

``python -m r2ai.s35.topk --variant JINA-K40 --jina-scores out/runs/s35-part-jina/jina/scores-all.parquet``

Same rule as ``COS-K<K>`` / ``J4-K<K>`` / ``BGR-K<K>-doc`` (S31/S33): pool = cosine top-211 of the S31 scored union
(``topk_pools``), K docs per query, one whole-body chunk for every selected doc with text, never cut. Order: Jina score
descending, ties by cosine rank then doc_id (``add_judge_rank``). ``Check-J4-K40`` runs RRF(cosine, 4B) through this
path and must give the S31 J4-K40 JSON.

Sidecar ``<name>.s35.json``: score file SHA, doc-set differences with J4-K40, Spearman of Jina with 4B, cosine and the
S33 bge-reranker-v2-m3 scores (R-doc, R-chk) on the pool pairs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission, _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import RRF_K, add_judge_rank, doc_set_diff, judge_correlations, select_rrf
from r2ai.s33.build import J4_K40_JSON_SHA256, MODEL as BGE_MODEL, load_union, reduce_chunks
from r2ai.s35 import part


def bge_scores() -> pd.DataFrame:
    """S33 bge-reranker-v2-m3 R-doc (pool211 + tie pools) and R-chk (max over top-3 chunks, doc fallback), by (query, doc)."""
    root = part.s33_root()
    rdoc = pd.concat([pd.read_parquet(root / name / BGE_MODEL / "scores-all.parquet", columns=["pair_id", "query_id", "doc_id", "score"])
                      for name in ("gpu-rdoc", "gpu-rdoc-tie")], ignore_index=True).rename(columns={"score": "r_doc"})
    chunk_scores = pd.read_parquet(root / "gpu-rchk" / BGE_MODEL / "scores-all.parquet")
    chunk_pairs = pd.concat([pd.read_parquet(root / name / "chunk_pairs.parquet") for name in ("pool-rchk", "pool-rchk-tie")], ignore_index=True)
    best = reduce_chunks(chunk_scores, chunk_pairs)[["pair_id", "r_chk"]]
    merged = rdoc.merge(best, on="pair_id", how="left")
    merged["r_chk"] = merged.r_chk.fillna(merged.r_doc)
    return merged.drop(columns="pair_id")


def build(out: Path, variants: Sequence[str], jina_scores: Path, check: bool = True) -> list[dict[str, Any]]:
    started = time.monotonic()
    root31 = paths.RUNS_DIR / "s31-strong-judge"
    names = list(variants) + (["Check-J4-K40"] if check else [])
    parsed: dict[str, tuple[int, str]] = {}
    for name in names:
        match = re.fullmatch(r"JINA-K(\d+)", name)
        if match:
            parsed[name] = (int(match.group(1)), "jina")
        elif name == "Check-J4-K40":
            parsed[name] = (40, "rrf_cos_4b")
        else:
            raise ValueError(f"Unsupported S35 top-K variant {name!r}")
    inputs = _inputs()
    for key in ("p1", "baseline", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    s29_build._preflight_outputs(out, tuple(names))
    for name in names:
        paths.assert_writable(out / f"{name}.s35.json")
        paths.assert_writable(out / f"{name}.s35.json.tmp")
    out.mkdir(parents=True, exist_ok=True)

    union, files = load_union(root31)
    pools, _ = s31_build.topk_pools(union, s31_build.P_FREE)  # the J4-K40 pool
    jina = pd.read_parquet(jina_scores, columns=["query_id", "doc_id", "score"])
    if jina.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Jina scores: duplicate (query_id, doc_id)")
    lookup = dict(zip(zip(jina.query_id.astype(int), jina.doc_id.astype(int)), jina.score.astype(float)))
    bge = bge_scores()
    bge_lookup = dict(zip(zip(bge.query_id.astype(int), bge.doc_id.astype(int)), zip(bge.r_doc.astype(float), bge.r_chk.astype(float))))
    missing = 0
    for q, rows in pools.items():
        for r in rows:
            value = lookup.get((q, int(r["doc_id"])))
            if value is None:
                missing += 1
                continue
            r["jina"] = value
            r["r_doc"], r["r_chk"] = bge_lookup[(q, int(r["doc_id"]))]
    if missing:
        raise ValueError(f"{missing} pool pairs have no Jina score")
    flat = pd.DataFrame([{"query_id": q, "cosine": r["cosine_score"], "p4b": r["judge_score"], "jina": r["jina"],
                          "r_doc": r["r_doc"], "r_chk": r["r_chk"]} for q, rows in pools.items() for r in rows])
    correlations = judge_correlations(flat, ["jina", "p4b", "cosine", "r_doc", "r_chk"])

    p1_rows = _load_submission(inputs["p1"], with_chunks=True)
    baseline = {int(row["id"]): row for row in _load_submission(inputs["baseline"])}
    first_chunk_by_id: dict[int, str] = {}
    for row in p1_rows:
        for chunk in row["relevant_chunks"]:
            first_chunk_by_id.setdefault(int(chunk["doc_id"]), str(chunk["chunk_text"]))
    groups, cid_to_doc = s29_build._load_aliases(inputs["canonical"] / "chunks_t256.parquet", inputs["canonical"] / "docs.parquet")
    docs = pq.read_table(inputs["canonical"] / "docs.parquet", columns=["doc_id", "lang", "quality"]).to_pandas()
    doc_lang = dict(zip(docs.doc_id.astype(int).tolist(), docs.lang.tolist()))
    thin = set(docs.loc[docs.quality == "thin", "doc_id"].astype(int).tolist())
    del docs
    extra_texts, extra_spans = s29_build._load_extra_docs(inputs)
    for doc_id in extra_texts:
        groups[doc_id] = [doc_id]
        cid_to_doc[doc_id] = doc_id
        doc_lang[doc_id] = "vi"
    p1_any, row_ids = s29_build._row_id_maps(p1_rows, cid_to_doc)
    pool211_path = root31 / "pool-extra" / "pool211_index.parquet"
    candidate_digest = _sha256(pool211_path)
    score_digest = s31_build._scores_digest(
        pd.DataFrame([{"query_id": q, "doc_id": r["doc_id"], "cosine_rank": r["cosine_rank"], "judge_score": r["judge_score"], "jina": r["jina"]}
                      for q, rows in sorted(pools.items()) for r in rows]), ["judge_score", "jina"])

    def choose(source: str, k: int) -> dict[int, list[dict[str, Any]]]:
        if source == "rrf_cos_4b":
            return {q: select_rrf(rows, k, ("cosine_rank", "judge_rank"), RRF_K) for q, rows in pools.items()}
        picked = {}
        for q, rows in pools.items():
            rows = [dict(r) for r in rows]
            add_judge_rank(rows, "jina", "jina_rank")
            picked[q] = sorted(rows, key=lambda r: r["jina_rank"])[:k]
        return picked

    j4_ids = {q: [int(r["doc_id"]) for r in rows] for q, rows in choose("rrf_cos_4b", 40).items()}
    doc_lang_all = {**doc_lang, **{d: "vi" for d in extra_texts}}
    summaries = []
    for name in names:
        k, source = parsed[name]
        chosen = choose(source, k)
        chosen_ids = {q: [int(r["doc_id"]) for r in rows] for q, rows in chosen.items()}
        diff = doc_set_diff(chosen_ids, j4_ids)
        chunk_info: dict[str, Any] = {}
        selector = s31_build.fixed_chunk_selector({}, None, chunk_info)
        summary = s29_build._build_variant(
            out, name, p1_rows, baseline, p1_any, first_chunk_by_id, row_ids, groups, cid_to_doc, doc_lang, thin,
            inputs["canonical"], extra_texts, extra_spans, chosen, candidate_digest, s31_build.POOL_MULTIPLIER,
            score_digest, RRF_K if source == "rrf_cos_4b" else None, None, s31_build.CHUNK_MULTIPLIER, True,
            diff["mean_symmetric_difference"], fixed_k=False, chunk_selector=selector)
        chunk_info.pop("chunk_doc_ids", None)
        langs: dict[str, int] = {}
        for ids in chosen_ids.values():
            for d in ids:
                langs[doc_lang_all[d]] = langs.get(doc_lang_all[d], 0) + 1
        sidecar = {
            "name": name, "status": "built_cpu_not_submitted",
            "rank_source": {"jina": f"Jina Reranker v3.5 score (revision pinned in r2ai.s35.jina), pool cosine top-{s31_build.P_FREE}",
                            "rrf_cos_4b": "RRF(cosine, Qwen3-Reranker-4B), k=60 (= J4-K40)"}[source],
            "keep_rule": f"top-{k} of the S31 scored-union cosine top-{s31_build.P_FREE} (no K_q, no n_c, no threshold)",
            "jina_scores": str(jina_scores), "jina_scores_sha256": _sha256(jina_scores),
            "vs_j4_k40_doc_sets": diff,
            "docs_lang": {"counts": langs, "share": {lang: n / sum(langs.values()) for lang, n in sorted(langs.items())}},
            "chunks": {**chunk_info, "rule": "one whole-body chunk for every selected doc with text, never cut",
                       "body_cut_chunks": summary.get("body_cut_chunks")},
            "correlations_pool211": correlations, "s31_inputs": {k2: str(v) for k2, v in files.items()},
            "json_sha256": summary.get("json_sha256"), "zip_sha256": summary.get("zip_sha256"), "zip_bytes": summary.get("zip_bytes"),
        }
        if name == "Check-J4-K40":
            sidecar["regression_target_json_sha256"] = J4_K40_JSON_SHA256
            sidecar["regression_pass"] = summary.get("json_sha256") == J4_K40_JSON_SHA256
        _atomic_json(out / f"{name}.s35.json", sidecar)
        print(json.dumps({k2: sidecar[k2] for k2 in ("name", "vs_j4_k40_doc_sets", "json_sha256", "zip_sha256", "zip_bytes")}
                         | ({"regression_pass": sidecar["regression_pass"]} if "regression_pass" in sidecar else {}), ensure_ascii=False), flush=True)
        summaries.append(sidecar)
    print(json.dumps({"correlations_pool211": correlations, "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, help="JINA-K<K>")
    parser.add_argument("--jina-scores", type=Path, default=part.run_root() / "jina" / "scores-all.parquet")
    parser.add_argument("--out", type=Path, default=part.run_root() / "build-topk")
    parser.add_argument("--no-check", action="store_true", help="skip the Check-J4-K40 regression bundle")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out)
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S35 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    results = build(out, args.variant, paths.resolve_path(args.jina_scores), check=not args.no_check)
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
