"""Build S33 bundles: top-K of the S31 cosine top-211 pool ranked by bge-reranker-v2-m3 alone.

``python -m r2ai.s33.build --variant BGR-K40-doc --variant BGR-K40-chk``

* Pool: the scored union of S31 (extended pool13 + evicted + pool-extra), cosine top-211 per query,
  exactly as ``COS-K<K>`` / ``J4-K<K>`` (``r2ai.s31.build.topk_pools``). It differs from
  ``pool211_index.parquet`` by 186 pairs (exact cosine ties at rank 211); those are scored from the
  ``pool-*-tie`` pools.
* ``BGR-K<K>-doc``: top-K by R-doc = bge logit of (query, doc_text 512). ``BGR-K<K>-chk``: top-K by
  R-chk = max bge logit over the top-3 t256 chunks (S32 choice); docs without a chunk take R-doc.
  Order: descending score, ties by cosine rank then doc_id (``add_judge_rank``). No cosine fusion.
* Chunks: one whole-body chunk for every selected doc with text (as J4-K40), never cut.
* ``Check-J4-K40``: RRF(cosine, 4B) top-40 through this path; its JSON must equal the S31 J4-K40
  JSON (``J4_K40_JSON_SHA256``), so the bundles differ from J4-K40 only by the doc choice.

Sidecar ``<name>.s33.json``: rank source, score file SHA, doc-set difference with J4-K40 and
Spearman between R-doc, R-chk, 4B and cosine on the 253,200 pool211 pairs.
"""
from __future__ import annotations

import argparse
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
from r2ai.s31 import build as s31_build
from r2ai.s31.select import RRF_K, add_judge_rank, doc_set_diff, judge_correlations, select_rrf

J4_K40_JSON_SHA256 = "e464bbd19436167c7074aaee0f6bfa487fe1e27efe4984c1a244caf0f9d4cafc"
MODEL = "bge-reranker-v2-m3"


def run_root() -> Path:
    return paths.run_dir("s33-btc-baseline")


def reduce_chunks(chunk_scores: pd.DataFrame, chunk_pairs: pd.DataFrame) -> pd.DataFrame:
    """Max chunk logit per pool211 pair (``src_pair_id``) and the rank of the best chunk."""
    frame = chunk_pairs[["pair_id", "src_pair_id", "chunk_rank", "field"]].merge(
        chunk_scores[["pair_id", "score"]], on="pair_id", how="left", validate="one_to_one")
    if frame.score.isna().any():
        raise ValueError(f"{int(frame.score.isna().sum())} chunk rows have no bge score")
    frame = frame.sort_values(["src_pair_id", "score", "chunk_rank"], ascending=[True, False, True], kind="stable")
    best = frame.drop_duplicates("src_pair_id").drop(columns=["pair_id"])
    return best.rename(columns={"src_pair_id": "pair_id", "score": "r_chk", "chunk_rank": "best_chunk_rank",
                                "field": "best_field"})[["pair_id", "r_chk", "best_chunk_rank", "best_field"]]


def load_union(root31: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Path]]:
    """The S31 scored union in cosine order with 4B scores (as the J4-K40 build)."""
    ext = root31 / "pharmacity-ext"
    files = {"pool13_index": ext / "pool13_index.parquet", "q06": ext / "q06_scores.parquet",
             "judge": ext / "qwen3-reranker-4b-scores-pool13.parquet", "pool_index": root31 / "pool" / "pool_index.parquet",
             "orig": root31 / "gpu" / "qwen3-reranker-4b" / "scores-pool13.parquet",
             "extra_pairs": root31 / "pool-extra" / "extra_pairs.parquet",
             "extra_scores": root31 / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet"}
    for path in files.values():
        paths.require_inputs(path)
    frame = s31_build.load_candidates(root31 / "pool", files["judge"], files["pool13_index"], files["q06"])
    ranked = s31_build.rank_candidates(frame)
    union = s31_build.scored_union(ranked, pd.read_parquet(files["pool_index"]), pd.read_parquet(files["orig"]),
                                   pd.read_parquet(files["extra_pairs"]), pd.read_parquet(files["extra_scores"]))
    return union, files


def build(out: Path, variants: Sequence[str], rdoc_scores: Sequence[Path], rchk_scores: Sequence[Path] = (),
          rchk_pool: Sequence[Path] = (),
          check: bool = True) -> list[dict[str, Any]]:
    started = time.monotonic()
    root31 = paths.RUNS_DIR / "s31-strong-judge"
    pool211_path = root31 / "pool-extra" / "pool211_index.parquet"
    names = list(variants) + (["Check-J4-K40"] if check else [])
    parsed = {}
    for name in names:
        match = re.fullmatch(r"BGR-K(\d+)-(doc|chk)", name)
        if match:
            parsed[name] = (int(match.group(1)), f"r_{match.group(2)}")
        elif name == "Check-J4-K40":
            parsed[name] = (40, "rrf_cos_4b")
        else:
            raise ValueError(f"Unsupported S33 variant {name!r}")
    if any(src == "r_chk" for _, src in parsed.values()) and (not rchk_scores or not rchk_pool):
        raise ValueError("BGR-K*-chk needs --rchk-scores and --rchk-pool")
    inputs = _inputs()
    for key in ("p1", "baseline", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    s29_build._preflight_outputs(out, tuple(names))
    for name in names:
        paths.assert_writable(out / f"{name}.s33.json")
        paths.assert_writable(out / f"{name}.s33.json.tmp")
    out.mkdir(parents=True, exist_ok=True)

    union, files = load_union(root31)
    pools, _ = s31_build.topk_pools(union, s31_build.P_FREE)  # the J4-K40 pool
    pool = pd.DataFrame([{"pair_id": int(r["pair_id"]), "query_id": q, "doc_id": int(r["doc_id"])}
                         for q, rows in pools.items() for r in rows])
    pool211 = pd.read_parquet(pool211_path)
    in_index = pool.merge(pool211[["query_id", "doc_id"]], on=["query_id", "doc_id"], how="left", indicator=True)._merge == "both"
    pool_vs_index = {"pairs_not_in_pool211_index": int((~in_index).sum()),
                     "queries": int(pool.query_id[~in_index.to_numpy()].nunique()),
                     "reason": "exact cosine ties at rank 211 cut in a different order; scored from the tie pools"}

    # ---- bge scores per pool pair (main pools + the tie pools)
    rdoc = pd.concat([pd.read_parquet(p) for p in rdoc_scores], ignore_index=True)
    if rdoc.pair_id.duplicated().any():
        raise ValueError("duplicate pair_id across the R-doc score files")
    scores = pool.merge(rdoc[["pair_id", "query_id", "doc_id", "score"]], on="pair_id", how="left", suffixes=("", "_s"),
                        validate="one_to_one")
    if scores.score.isna().any() or not ((scores.query_id == scores.query_id_s).all() and (scores.doc_id == scores.doc_id_s).all()):
        raise ValueError(f"R-doc scores do not cover/align with the pool ({int(scores.score.isna().sum())} missing)")
    scores = scores[["pair_id", "score"]].rename(columns={"score": "r_doc"})
    chk_info: dict[str, Any] = {}
    if rchk_scores and rchk_pool:
        chunk_scores = pd.concat([pd.read_parquet(p) for p in rchk_scores], ignore_index=True)
        chunk_pairs = pd.concat([pd.read_parquet(p / "chunk_pairs.parquet") for p in rchk_pool], ignore_index=True)
        if chunk_scores.pair_id.duplicated().any() or chunk_pairs.pair_id.duplicated().any():
            raise ValueError("duplicate chunk pair_id across the R-chk pools")
        best = reduce_chunks(chunk_scores, chunk_pairs)
        scores = scores.merge(best, on="pair_id", how="left", validate="one_to_one")
        fallback = scores.r_chk.isna()
        fb_ids = set().union(*[set(pd.read_parquet(p / "doc_fallback.parquet").src_pair_id.astype(int)) for p in rchk_pool])
        if not set(scores.pair_id[fallback].astype(int)) <= fb_ids:
            raise ValueError("pool pairs without chunk score that are not in doc_fallback.parquet")
        scores["r_chk_fallback"] = fallback
        scores.loc[fallback, "r_chk"] = scores.loc[fallback, "r_doc"]
        chk_info = {"fallback_pairs": int(fallback.sum()),
                    "best_chunk_rank_share": {str(k): float(v) for k, v in scores.loc[~fallback, "best_chunk_rank"].value_counts(normalize=True).sort_index().items()},
                    "best_field_share": {str(k): float(v) for k, v in scores.loc[~fallback, "best_field"].value_counts(normalize=True).items()}}
    by_pair = scores.set_index("pair_id")
    for rows in pools.values():
        for r in rows:
            r["r_doc"] = float(by_pair.at[int(r["pair_id"]), "r_doc"])
            if "r_chk" in by_pair:
                r["r_chk"] = float(by_pair.at[int(r["pair_id"]), "r_chk"])

    # ---- correlations on pool211
    flat = pd.DataFrame([{"query_id": q, "cosine": r["cosine_score"], "p4b": r["judge_score"], "r_doc": r["r_doc"],
                          **({"r_chk": r["r_chk"]} if "r_chk" in r else {})} for q, rows in pools.items() for r in rows])
    correlations = judge_correlations(flat, [c for c in ("r_doc", "r_chk", "p4b", "cosine") if c in flat])

    # ---- S29 builder context (as r2ai.s31.build.build)
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
    candidate_digest = _sha256(pool211_path)
    score_digest = s31_build._scores_digest(
        pd.DataFrame([{"query_id": q, "doc_id": r["doc_id"], "cosine_rank": r["cosine_rank"], "judge_score": r["judge_score"],
                       "r_doc": r["r_doc"], **({"r_chk": r["r_chk"]} if "r_chk" in r else {})}
                      for q, rows in sorted(pools.items()) for r in rows]),
        [c for c in ("judge_score", "r_doc", "r_chk") if c in flat or c == "judge_score"])

    def choose(source: str, k: int) -> dict[int, list[dict[str, Any]]]:
        if source == "rrf_cos_4b":
            return {q: select_rrf(rows, k, ("cosine_rank", "judge_rank"), RRF_K) for q, rows in pools.items()}
        out_q = {}
        for q, rows in pools.items():
            rows = [dict(r) for r in rows]
            add_judge_rank(rows, source, "bge_rank")
            out_q[q] = sorted(rows, key=lambda r: r["bge_rank"])[:k]
        return out_q

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
            diff["mean_symmetric_difference"], fixed_k=False, chunk_selector=selector,
        )
        chunk_doc_ids = chunk_info.pop("chunk_doc_ids", None)
        langs: dict[str, int] = {}
        for ids in chosen_ids.values():
            for d in ids:
                langs[doc_lang_all[d]] = langs.get(doc_lang_all[d], 0) + 1
        sidecar = {
            "name": name, "status": "built_cpu_not_submitted",
            "rank_source": {"r_doc": "bge-reranker-v2-m3 logit (query, doc_text 512 Qwen3 tokens), pair cut to 512 bge tokens",
                            "r_chk": "max bge-reranker-v2-m3 logit over the top-3 t256 chunks (S32 choice); no chunk: r_doc",
                            "rrf_cos_4b": "RRF(cosine, Qwen3-Reranker-4B), k=60 (= J4-K40)"}[source],
            "keep_rule": f"top-{k} of the S31 scored-union cosine top-{s31_build.P_FREE} (no K_q, no n_c, no threshold)",
            "rdoc_scores": {str(p): _sha256(p) for p in rdoc_scores},
            "rchk_scores": {str(p): _sha256(p) for p in rchk_scores} or None,
            "pool": f"cosine top-{s31_build.P_FREE} of the S31 scored union (= J4-K40 pool)",
            "pool_vs_pool211_index": pool_vs_index, "pool211_index_sha256": candidate_digest,
            "s31_inputs": {k2: str(v) for k2, v in files.items()},
            "vs_j4_k40_doc_sets": diff,
            "docs_lang": {"counts": langs, "share": {l: n / sum(langs.values()) for l, n in sorted(langs.items())}},
            "chunks": {**chunk_info, "rule": "one whole-body chunk for every selected doc with text, never cut",
                       "body_cut_chunks": summary.get("body_cut_chunks")},
            "r_chk_reduce": chk_info or None,
            "correlations_pool211": correlations,
            "json_sha256": summary.get("json_sha256"), "zip_sha256": summary.get("zip_sha256"),
            "zip_bytes": summary.get("zip_bytes"),
        }
        if name == "Check-J4-K40":
            sidecar["regression_target_json_sha256"] = J4_K40_JSON_SHA256
            sidecar["regression_pass"] = summary.get("json_sha256") == J4_K40_JSON_SHA256
        _atomic_json(out / f"{name}.s33.json", sidecar)
        print(json.dumps({k2: sidecar[k2] for k2 in ("name", "vs_j4_k40_doc_sets", "json_sha256", "zip_sha256", "zip_bytes")}
                         | ({"regression_pass": sidecar["regression_pass"]} if "regression_pass" in sidecar else {}),
                         ensure_ascii=False), flush=True)
        summaries.append(sidecar)
    print(json.dumps({"correlations_pool211": correlations, "r_chk": chk_info, "seconds": round(time.monotonic() - started, 1)},
                     ensure_ascii=False), flush=True)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, help="BGR-K<K>-doc / BGR-K<K>-chk")
    root = run_root()
    parser.add_argument("--rdoc-scores", type=Path, action="append", default=None,
                        help="default: gpu-rdoc and gpu-rdoc-tie scores-all.parquet")
    parser.add_argument("--rchk-scores", type=Path, action="append", default=None)
    parser.add_argument("--rchk-pool", type=Path, action="append", default=None, help="default: pool-rchk, pool-rchk-tie")
    parser.add_argument("--out", type=Path, default=root / "build")
    parser.add_argument("--no-check", action="store_true", help="skip the Check-J4-K40 regression bundle")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out)
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S33 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    need_chk = any(v.endswith("-chk") for v in args.variant)
    scores = lambda name: root / name / MODEL / "scores-all.parquet"  # noqa: E731
    rdoc = args.rdoc_scores or [scores("gpu-rdoc"), scores("gpu-rdoc-tie")]
    rchk = args.rchk_scores or [scores("gpu-rchk")]  # the 186 tie pairs have no chunk (S32 fallback)
    rchk_pool = args.rchk_pool or [root / "pool-rchk", root / "pool-rchk-tie"]
    results = build(out, args.variant, [paths.resolve_path(p) for p in rdoc],
                    [paths.resolve_path(p) for p in rchk] if need_chk else (),
                    [paths.resolve_path(p) for p in rchk_pool] if need_chk else (), check=not args.no_check)
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
