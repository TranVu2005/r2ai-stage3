"""Build S35 bundles through the S31 K4-thr-C path (``r2ai.s31.build.build``), swapping rank sources.

``python -m r2ai.s35.build --variant PART-thrC --variant Check-PART-id``
``python -m r2ai.s35.build --variant J3-thrC --jina-scores .../scores-all.parquet``

* Same pool13 (round(1.3 K_q) after Pharmacity 979), same 4B scores, K_q' clipped to [0.5, 1.3] K_q with
  the global t of K4-thr-C, same chunk rule (one whole-body chunk per selected doc with P4B >= t_c, t_c
  recalibrated so chunks = 175,148 as K4-thr-C), same S29 writer. Only the RRF rank sources change:
  * ``PART-thrC``      RRF(cos_doc2, 4B), cos_doc2 = max(cos_doc512, best t256 chunk cosine).
  * ``Check-PART-id``  the same path with cos_doc2 := cos_doc512; its JSON must equal K4-thr-C.
  * ``J3-thrC``        RRF(cosine, 4B, Jina) on the pool13 docs.
* K_q' (and so t) come from P4B only; judge ranks keep the old cosine rank as tie-break, so the doc
  count per query is identical to K4-thr-C in every variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import add_judge_rank
from r2ai.s35 import part

K4_THR_C_JSON_SHA256 = "d771c4f9b0b5f7b8c68f934b28aeb207179a047a2f7a83f29c7b62628939c9b0"
K4_THR_C_ZIP_SHA256 = "31980453608212846a916b84e306482bc57e3de2f7c040304b8a38a204815a87"
VARIANTS = {
    "PART-thrC": ("cos2_rank", "judge_rank"),
    "Check-PART-id": ("cos2id_rank", "judge_rank"),
    "J3-thrC": ("cosine_rank", "judge_rank", "jina_rank"),
}


def _fallback_ids(pools: Sequence[Path]) -> set[int]:
    ids: set[int] = set()
    for p in pools:
        ids |= set(pd.read_parquet(p / "doc_fallback.parquet", columns=["src_pair_id"]).src_pair_id.astype(int))
    return ids


def chunk_pools() -> list[Path]:
    """S32 pool13 (+ ph979 new) and S33 pool211 R-chk pools; the S33 rows of shared pairs are S32 rows verbatim."""
    return [paths.RUNS_DIR / "s32-chunk-judge" / "pool", part.s33_root() / "pool-rchk", part.s33_root() / "pool-rchk-tie"]


def part_hook(pools: Sequence[Path]):
    """Add ``cos_part``, ``cos_doc2``, ``cos2_rank`` and ``cos2id_rank`` (cos_doc2 := cos_doc512) to the pool13 rows."""

    def hook(ranked: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        frames = [pd.read_parquet(p / "chunk_pairs.parquet", columns=["src_pair_id", "chunk_rank", "chunk_cosine"]) for p in pools]
        top = pd.concat(frames, ignore_index=True)
        top = top[top.chunk_rank == 1]
        spread = top.groupby("src_pair_id").chunk_cosine.agg(["min", "max"])
        if (spread["max"] - spread["min"]).max() > 1e-9:
            raise ValueError("S32 and S33 disagree on the best chunk cosine of a shared pair")
        best = top.drop_duplicates("src_pair_id").set_index("src_pair_id").chunk_cosine.to_dict()
        fallback = _fallback_ids(pools)
        missing = [int(r["pair_id"]) for rows in ranked.values() for r in rows
                   if int(r["pair_id"]) not in best and int(r["pair_id"]) not in fallback]
        if missing:
            raise ValueError(f"{len(missing)} pool13 pairs are in neither the chunk pools nor the no-chunk lists, e.g. {missing[:5]}")
        gains, with_chunk, higher = [], 0, 0
        for rows in ranked.values():
            for r in rows:
                cos = float(r["cosine_score"])
                part_cos = best.get(int(r["pair_id"]))
                r["cos_part"] = float("nan") if part_cos is None else float(part_cos)
                r["cos_doc2"] = cos if part_cos is None else max(cos, float(part_cos))
                if part_cos is not None:
                    with_chunk += 1
                    higher += part_cos > cos
                    gains.append(part_cos - cos)
            for key, score in (("cos2_rank", "cos_doc2"), ("cos2id_rank", "cosine_score")):
                for rank, r in enumerate(sorted(rows, key=lambda r: (-float(r[score]), int(r["cosine_rank"]))), 1):
                    r[key] = rank
            if any(r["cos2id_rank"] != r["cosine_rank"] for r in rows):
                raise ValueError("cos2id_rank differs from cosine_rank: the cosine rank is not (-cosine, rank) sorted")
        shifted = [abs(r["cos2_rank"] - r["cosine_rank"]) for rows in ranked.values() for r in rows]
        return {"hook": "cos_doc2 = max(cos_doc512, max t256 chunk cosine)", "pairs": sum(len(r) for r in ranked.values()),
                "pairs_with_chunk": with_chunk, "pairs_without_chunk": sum(len(r) for r in ranked.values()) - with_chunk,
                "share_part_gt_doc_of_chunked": higher / with_chunk, "mean_gain": float(np.mean(gains)),
                "rank_shift_mean_abs": float(np.mean(shifted)), "chunk_pools": [str(p) for p in pools],
                "chunk_pools_sha256": {str(p): _sha256(p / "chunk_pairs.parquet") for p in pools}}

    return hook


def jina_hook(scores_path: Path):
    """Add ``jina_rank`` (rank by Jina score inside the pool13 rows, ties as the 4B rank)."""

    def hook(ranked: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        scores = pd.read_parquet(scores_path)
        if scores.duplicated(["query_id", "doc_id"]).any():
            raise ValueError("Jina scores: duplicate (query_id, doc_id)")
        lookup = dict(zip(zip(scores.query_id.astype(int), scores.doc_id.astype(int)), scores.score.astype(float)))
        missing = 0
        for qid, rows in ranked.items():
            for r in rows:
                value = lookup.get((qid, int(r["doc_id"])))
                if value is None:
                    missing += 1
                    continue
                r["jina_score"] = value
        if missing:
            raise ValueError(f"{missing} pool13 pairs have no Jina score")
        for rows in ranked.values():
            add_judge_rank(rows, "jina_score", "jina_rank")
        return {"hook": "jina_rank = rank of the Jina Reranker v3.5 score inside the pool13 rows",
                "jina_scores": str(scores_path), "jina_scores_sha256": _sha256(scores_path), "pairs": len(scores)}

    return hook


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, choices=sorted(VARIANTS))
    root = paths.RUNS_DIR / "s31-strong-judge"
    ext = root / "pharmacity-ext"
    parser.add_argument("--out", type=Path, default=part.run_root() / "build-part")
    parser.add_argument("--jina-scores", type=Path, default=None, help="pool13 Jina scores (query_id, doc_id, score) for J3-thrC")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out)
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S35 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    variants = list(dict.fromkeys(args.variant))
    need_part = any(v in ("PART-thrC", "Check-PART-id") for v in variants)
    need_jina = "J3-thrC" in variants
    if need_part and need_jina:
        raise ValueError("Build the cos_doc2 and the Jina variants in separate runs")
    if need_jina and args.jina_scores is None:
        raise ValueError("J3-thrC needs --jina-scores")
    hook = part_hook(chunk_pools()) if need_part else jina_hook(paths.resolve_path(args.jina_scores))
    results = s31_build.build(
        out, variants, "qwen3-reranker-4b", ext / "qwen3-reranker-4b-scores-pool13.parquet", root / "pool",
        pool13_index=ext / "pool13_index.parquet", q06_scores=ext / "q06_scores.parquet",
        rank_hook=hook, rank_keys_override={v: VARIANTS[v] for v in variants}, variant_alias={v: "K4-thr-C" for v in variants})
    failed = []
    for sidecar in results:
        path = out / f"{sidecar['name']}.s31.json"
        if sidecar["name"] == "Check-PART-id":
            sidecar["regression_target_json_sha256"] = K4_THR_C_JSON_SHA256
            sidecar["regression_target_zip_sha256"] = K4_THR_C_ZIP_SHA256
            sidecar["regression_pass"] = sidecar.get("json_sha256") == K4_THR_C_JSON_SHA256
            if not sidecar["regression_pass"]:
                failed.append(sidecar["name"])
        _atomic_json(path, sidecar)
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
