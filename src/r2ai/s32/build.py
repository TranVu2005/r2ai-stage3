"""Build S32 chunk-judge bundles on CPU with the S29/S31 builder (nc140, whole bodies, Pharmacity).

``python -m r2ai.s32.build --variant Check-q06-rrf --variant C06-rrf --variant Check-j4-rrf --variant C06-rrf4``

* ``C06-rrf``   = RRF(k=60) of cosine rank and the chunk judge rank ("C" = ``--c-kind``, default
  C-max) over the S31 pool13 (Pharmacity 612 docs); compare with ``QRRF-nc140-ph``.
* ``C06-rrf4``  = RRF(k=60) of cosine, C and the 4B doc_text judge over the Pharmacity-extended
  pool13 (979 docs); compare with ``J4-rrf-ph979``.
* ``C06-rrf-nt`` = ``C06-rrf`` with C-max-nt (best non-title chunk; title-only docs keep the title score).
* ``Check-q06-rrf`` swaps C for the S29 0.6B doc_text score: its JSON must equal ``QRRF-nc140-ph``.
* ``Check-j4-rrf`` is the S31 ``J4-rrf-ph979`` selection through this code path: its JSON must equal
  the JSON SHA256 recorded in that bundle's sidecar.

Selection, chunk count (n_c x1.4), whole bodies and compact JSON come from
``r2ai.s29.build._build_variant`` unchanged; only the rank sources differ.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission, _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import RRF_K, doc_set_diff, judge_correlations, select_rrf

S31_ROOT = "s31-strong-judge"
CHUNK_MULTIPLIER = s31_build.CHUNK_MULTIPLIER
WHOLE_BODY = s31_build.WHOLE_BODY
POOL_MULTIPLIER = s31_build.POOL_MULTIPLIER
J4_SIDECAR = "build-j4-ph979/J4-rrf-ph979.s31.json"

# name -> (index, c source, rank keys, reference selection)
#   index: pool13 = S31 pool13 (Pharmacity 612); ph979 = pool13 extended by Pharmacity lot 3
#   c source: q06 = S29 0.6B doc_text score, chunk = the S32 chunk score of --c-kind
#   rank keys: q06_rank is the "C" rank, judge_rank is the 4B doc_text rank
VARIANTS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "Check-q06-rrf": ("pool13", "q06", ("cosine_rank", "q06_rank"), "q06"),
    "C06-rrf": ("pool13", "chunk", ("cosine_rank", "q06_rank"), "q06"),
    "Check-j4-rrf": ("ph979", "q06", ("cosine_rank", "judge_rank"), "j4"),
    "C06-rrf4": ("ph979", "chunk", ("cosine_rank", "q06_rank", "judge_rank"), "j4"),
    "C06-rrf-nt": ("pool13", "chunk", ("cosine_rank", "q06_rank"), "q06"),
}
# variants that fix the chunk score kind regardless of --c-kind
VARIANT_KIND = {"C06-rrf-nt": "cmaxnt"}


def _frame(index: str, c_source: str, c_kind: str, score_dir: Path, s31: Path, need_judge: bool) -> tuple[pd.DataFrame, dict[str, Path | None]]:
    pool_dir = s31 / "pool"
    ext = s31 / "pharmacity-ext"
    if index == "pool13":
        pool13_index = None
        q06 = pool_dir / "s29_qwen06_scores.parquet" if c_source == "q06" else score_dir / f"score-{c_kind}-pool13.parquet"
    else:
        pool13_index = ext / "pool13_index.parquet"
        q06 = ext / "q06_scores.parquet" if c_source == "q06" else score_dir / f"score-{c_kind}-ph979.parquet"
    judge = ext / "qwen3-reranker-4b-scores-pool13.parquet" if need_judge else None
    frame = s31_build.load_candidates(pool_dir, judge, pool13_index, q06)
    return frame, {"index": pool13_index or pool_dir / "pool_index.parquet", "c_scores": q06, "judge_scores": judge}


def build(out: Path, variants: Sequence[str], score_dir: Path, c_kind: str = "cmax", name_suffix: str = "") -> list[dict[str, Any]]:
    started = time.monotonic()
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported S32 variants {unknown}; choose from {sorted(VARIANTS)}")
    s31 = paths.RUNS_DIR / S31_ROOT
    inputs = _inputs()
    for key in ("p1", "baseline", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    names = {v: f"{v}{name_suffix}" for v in variants}
    s29_build._preflight_outputs(out, tuple(names.values()))
    for name in names.values():
        paths.assert_writable(out / f"{name}.s32.json")
        paths.assert_writable(out / f"{name}.s32.json.tmp")
    out.mkdir(parents=True, exist_ok=True)

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
    new_docs = set(pd.read_parquet(s31 / "pharmacity-ext" / "pool13_index.parquet").query("is_new").doc_id.astype(int))

    summaries = []
    for base in variants:
        name = names[base]
        index, c_source, keys, ref_kind = VARIANTS[base]
        kind = VARIANT_KIND.get(base, c_kind)
        need_judge = "judge_rank" in keys or ref_kind == "j4"
        frame, files = _frame(index, c_source, kind, score_dir, s31, need_judge)
        ranked = s31_build.rank_candidates(frame)
        if set(ranked) != set(baseline):
            raise ValueError("pool13 and K90-ph query IDs differ")
        base_k = {qid: len(baseline[qid]["relevant_docs"]) for qid in ranked}
        ref_keys = ("cosine_rank", "q06_rank") if ref_kind == "q06" else ("cosine_rank", "judge_rank")
        if ref_kind == "q06":  # QRRF-nc140-ph: always the S29 0.6B doc_text score on the 612-doc pool13
            ref_frame, _ = _frame("pool13", "q06", kind, score_dir, s31, False)
            ref_ranked = s31_build.rank_candidates(ref_frame)
        else:
            ref_ranked = ranked
        reference_ids = {qid: [int(r["doc_id"]) for r in select_rrf(rows, base_k[qid], ref_keys, RRF_K)]
                         for qid, rows in ref_ranked.items()}
        chosen = {qid: select_rrf(rows, base_k[qid], keys, RRF_K) for qid, rows in ranked.items()}
        chosen_ids = {qid: [int(r["doc_id"]) for r in rows] for qid, rows in chosen.items()}
        diff = doc_set_diff(chosen_ids, reference_ids)
        score_cols = ["q06_score"] + (["judge_score"] if "judge_score" in frame else [])
        digest = s31_build._scores_digest(frame, score_cols)
        corr = judge_correlations(frame.assign(cosine=frame.cosine_score), ["cosine", *score_cols])
        candidate_file = files["index"]
        summary = s29_build._build_variant(
            out, name, p1_rows, baseline, p1_any, first_chunk_by_id, row_ids, groups, cid_to_doc, doc_lang, thin,
            inputs["canonical"], extra_texts, extra_spans, chosen, _sha256(candidate_file), POOL_MULTIPLIER, digest,
            RRF_K, None, CHUNK_MULTIPLIER, WHOLE_BODY, diff["mean_symmetric_difference"],
        )
        selected_new = sum(1 for rows in chosen_ids.values() for d in rows if d in new_docs)
        sidecar = {
            "name": name,
            "status": "built_cpu_not_submitted",
            "variant": base,
            "pool": "S31 pool13, Pharmacity 612 docs" if index == "pool13"
            else "S31 pool13 extended by Pharmacity lot 3 (979 docs), r2ai.s31.pharmacity_ext",
            "rank_sources": list(keys),
            "rank_source_meaning": {"cosine_rank": "BGE-M3 doc-level cosine",
                                    "q06_rank": "S29 0.6B doc_text score" if c_source == "q06" else f"S32 chunk judge {kind}",
                                    "judge_rank": "S31 Qwen3-Reranker-4B doc_text score"},
            "c_kind": None if c_source == "q06" else kind,
            "rrf_k": RRF_K,
            "keep_rule": "len(relevant_docs) of each K90-ph row",
            "chunk_multiplier": CHUNK_MULTIPLIER,
            "whole_body": WHOLE_BODY,
            "c_scores": str(files["c_scores"]),
            "c_scores_sha256": _sha256(files["c_scores"]),
            "judge_scores": str(files["judge_scores"]) if files["judge_scores"] else None,
            "judge_scores_sha256": _sha256(files["judge_scores"]) if files["judge_scores"] else None,
            "pool_index": str(candidate_file),
            "pool_index_sha256": _sha256(candidate_file),
            "reference": {"q06": "QRRF-nc140-ph doc sets: RRF(cosine, 0.6B doc_text) on pool13 (612)",
                          "j4": "J4-rrf-ph979 doc sets: RRF(cosine, 4B doc_text) on the 979 pool"}[ref_kind],
            "vs_reference_doc_sets": diff,
            "docs_per_query_mean": sum(len(v) for v in chosen_ids.values()) / len(chosen_ids),
            "pharmacity_lot3_docs_selected": selected_new,
            "correlations_pool": corr,
            "json_sha256": summary.get("json_sha256"),
            "zip_sha256": summary.get("zip_sha256"),
            "zip_bytes": summary.get("zip_bytes"),
            "note": "the S29 builder's build.json names the 0.6B model; this sidecar records the S32 rank sources",
        }
        if base == "Check-q06-rrf":
            sidecar["regression_target_json_sha256"] = s31_build.QRRF_NC140_JSON_SHA256
            sidecar["regression_pass"] = summary.get("json_sha256") == s31_build.QRRF_NC140_JSON_SHA256
        if base == "Check-j4-rrf":
            target = json.loads((s31 / J4_SIDECAR).read_text(encoding="utf-8"))["json_sha256"]
            sidecar["regression_target_json_sha256"] = target
            sidecar["regression_pass"] = summary.get("json_sha256") == target
        _atomic_json(out / f"{name}.s32.json", sidecar)
        print(json.dumps({k: sidecar[k] for k in ("name", "docs_per_query_mean", "vs_reference_doc_sets", "json_sha256",
                                                  "zip_sha256", "zip_bytes", "regression_pass") if k in sidecar},
                         ensure_ascii=False), flush=True)
        summaries.append(sidecar)
    print(json.dumps({"seconds": round(time.monotonic() - started, 1)}), flush=True)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, help=", ".join(sorted(VARIANTS)))
    parser.add_argument("--scores", type=Path, default=None, help="S32 scores dir from `r2ai.s32.scores reduce`")
    parser.add_argument("--c-kind", default="cmax", choices=("cmax", "ctop2", "cmix", "cmaxnt"))
    parser.add_argument("--out", type=Path, default=None, help="output dir below RUNS_DIR")
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args(argv)
    root = paths.RUNS_DIR / "s32-chunk-judge"
    out = paths.resolve_path(args.out or root / "build")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S32 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    results = build(out, args.variant, paths.resolve_path(args.scores or root / "scores"), args.c_kind, args.name_suffix)
    failed = [r["name"] for r in results if r.get("regression_pass") is False]
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
