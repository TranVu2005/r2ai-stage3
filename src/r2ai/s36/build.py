"""Build the S36 FUSE4 bundles through the S31 builder (``r2ai.s31.build.build``), changing only the rank sources.

``python -m r2ai.s36.build --variant FUSE4-thrC --variant FUSE4w-thrC``
``python -m r2ai.s36.build --variant Check-FUSE4-thrC-id``          (RRF(cosine, 4B) -> JSON of K4-thr-C-fx)
``python -m r2ai.s36.build --variant Check-FUSE4-thrC-pre``         (same with the pre-fix builder -> JSON of K4-thr-C)
``python -m r2ai.s36.build --variant Cal-FUSE4-free``               (prints t_c, writes no bundle)
``python -m r2ai.s36.build --variant FUSE4-free``
``python -m r2ai.s36.build --variant Check-FUSE4-free-id``          (RRF(cosine, 4B) pre-fix -> JSON of K4-free)

Frames (everything but the rank sources is the S31 builder, unchanged):

* ``thrC``  pool13 (round(1.3 K_q) after Pharmacity 979), K_q' = clip(#P4B >= t, ceil(0.5 K_q), floor(1.3 K_q)), t =
  0.07751765647688268; one whole-body chunk for each of the 175,548 selected docs (with text) of highest P4B, i.e. t_c
  recalibrated per bundle to the chunk total of K4-thr-C-fx (K4-thr-C after the Pharmacity builder fix; its t_c is
  0.07056). With cosine + 4B only, that rule gives the JSON of K4-thr-C-fx (``Check-FUSE4-thrC-th``). A fused ranking
  selects docs with more P4B >= 0.07056, which at that fixed t_c overran the ZIP ceiling (102,725,779 B).
Bundles of the three FUSE4 variants carry the suffix ``-docread`` (whole-body chunks as before: only the doc side is to be read
since the BTC metric change of 23/10) and write ``docsel/docsel-<name>.parquet`` (selected docs, ranks, scores, chunk flag).

* ``free``  K4-free: pool = cosine top-211 of the scored union, deepened to every scored pair when the 211th doc has
  P4B >= t; keep = clip(#P4B >= t, 86, pool); no K_q. Chunks: P4B >= ``FUSE4_FREE_T_C``, whole body, never cut.

Rank sources of the FUSE4 variants: RRF k=60 over cosine, 4B, Jina, C-max (``r2ai.s36.fuse``); ``FUSE4w-thrC`` weights the
4B rank by 2. Docs are cut by P4B only, so the doc count per query does not depend on the fused ranks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from typing import Any, Sequence

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json
from r2ai.s31 import build as s31_build
from r2ai.s36 import fuse

# t_c of FUSE4-free, calibrated once (Cal-FUSE4-free) so that its chunks equal those of FUSE4-thrC; fixed afterwards.
FUSE4_FREE_T_C: float | None = 0.2030746281894702  # Cal-FUSE4-free, 23/10: chunks = 175,548 (FUSE4-thrC) on the FUSE4-free docs
K4_THR_C_FX_JSON_SHA256 = s31_build.K4_THR_C_FX_JSON_SHA256
K4_THR_C_JSON_SHA256 = s31_build.K4_THR_C_JSON_SHA256
K4_FREE_JSON_SHA256 = "896ffcbb6b6aaa3c3dcd319b7f2dd56d94186c6857df0b3bce0688df51f6bbba"  # build-ext/K4-free.s31.json (pre-fix builder)
TWO_KEYS = ("cosine_rank", "judge_rank")
J3_KEYS = ("cosine_rank", "judge_rank", "jina_rank")
THRC_CHUNKS = 175_548  # chunks of K4-thr-C-fx (K4-thr-C after the Pharmacity builder fix); FUSE4 thrC bundles take the same total

# name -> (rank keys, S31 variant followed for keep/chunk rule, frame, weights, regression target JSON SHA256)
VARIANTS: dict[str, dict[str, Any]] = {
    "FUSE4-thrC": {"keys": fuse.FOUR_KEYS, "alias": "K4-thr-C", "frame": "thrC", "chunk_target": THRC_CHUNKS, "docread": True},
    "FUSE4w-thrC": {"keys": fuse.FOUR_KEYS, "alias": "K4-thr-C", "frame": "thrC", "weights": (1.0, 2.0, 1.0, 1.0), "chunk_target": THRC_CHUNKS, "docread": True},
    # S36 doc side (r2ai.s36.cal): J3-thrC order (cosine, 4B, Jina) in pool13, docs per query from cal/keep-<name>.json
    "EF2-cut-p4b": {"keys": J3_KEYS, "alias": "K4-thr-C", "frame": "thrC", "chunk_target": THRC_CHUNKS, "docread": True, "cal": "p4b"},
    "EF2-cut-avg": {"keys": J3_KEYS, "alias": "K4-thr-C", "frame": "thrC", "chunk_target": THRC_CHUNKS, "docread": True, "cal": "avg"},
    "LANG-thr-avg": {"keys": J3_KEYS, "alias": "K4-thr-C", "frame": "thrC", "chunk_target": THRC_CHUNKS, "docread": True, "cal": "avg"},
    "Check-FUSE4-thrC-th": {"keys": TWO_KEYS, "alias": "K4-thr-C", "frame": "thrC", "target": K4_THR_C_FX_JSON_SHA256, "chunk_target": THRC_CHUNKS},
    "Check-FUSE4-thrC-id": {"keys": TWO_KEYS, "alias": "K4-thr-C-fx", "frame": "thrC", "target": K4_THR_C_FX_JSON_SHA256},
    "Check-FUSE4-thrC-pre": {"keys": TWO_KEYS, "alias": "K4-thr-C", "frame": "thrC", "target": K4_THR_C_JSON_SHA256, "no_extra_chunks": True},
    "Cal-FUSE4-free": {"keys": fuse.FOUR_KEYS, "alias": "K4-free", "frame": "free", "chunk_mode": "threshold", "calibrate": True},
    "FUSE4-free": {"keys": fuse.FOUR_KEYS, "alias": "K4-free", "frame": "free", "chunk_mode": "fixed-whole", "needs_t_c": True, "docread": True},
    "Check-FUSE4-free-id": {"keys": TWO_KEYS, "alias": "K4-free", "frame": "free", "target": K4_FREE_JSON_SHA256, "no_extra_chunks": True},
}


def run_settings(names: Sequence[str]) -> dict[str, Any]:
    specs = [VARIANTS[n] for n in names]
    for key in ("no_extra_chunks", "chunk_mode", "calibrate", "needs_t_c", "chunk_target", "docread", "cal"):
        if len({bool(s.get(key)) if key not in ("chunk_mode", "chunk_target") else s.get(key) for s in specs}) > 1:
            raise ValueError(f"Variants with different {key} cannot share a build run: {list(names)}")
    return specs[0]


DOCREAD = "-docread"  # ZIP name suffix: whole-body chunks kept as before, only the doc side of the bundle is to be read


def docsel_writer(out: Path, suffix: str, chunk_t_c: float | None, pcal: dict[str, Any] | None = None):
    """``docsel-<name>.parquet`` (the S35 product for S38): the selected docs of each bundle in RRF order with every score,
    every rank and the flag of the chunk that bundle gives the doc (P4B >= t_c rule, ties as the bundle took them)."""
    import pandas as pd

    def nan(value):
        return float("nan") if value is None else float(value)

    def write(name: str, chosen, chunk_doc_ids, chunk_info) -> None:
        base = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
        t_c = chunk_info.get("t_c", chunk_t_c)
        rows = []
        for qid in sorted(chosen):
            carrying = set(chunk_doc_ids.get(qid, [])) if chunk_doc_ids is not None else set()
            for rank, r in enumerate(chosen[qid], 1):
                rows.append({"query_id": int(qid), "doc_id": int(r["doc_id"]), "rank": rank, "rrf_score": float(r["rrf_score"]),
                             "cosine": float(r["cosine_score"]), "p4b": float(r["judge_score"]),
                             "jina": nan(r.get("jina_score")), "cmax": nan(r.get("cmax_score")),
                             "cosine_rank": int(r["cosine_rank"]), "judge_rank": int(r["judge_rank"]),
                             "jina_rank": int(r.get("jina_rank", 0)), "cmax_rank": int(r.get("cmax_rank", 0)),
                             "chunk": int(r["doc_id"]) in carrying, "p4b_ge_tc": bool(t_c is not None and r["judge_score"] >= t_c),
                             "t_c": nan(t_c)})
        frame = pd.DataFrame(rows)
        if pcal and base in pcal:  # calibrated probability of the pool pair (r2ai.s36.cal)
            frame = frame.merge(pcal[base], on=["query_id", "doc_id"], how="left", validate="one_to_one")
        target = out / f"docsel-{base}.parquet"
        paths.assert_writable(target)
        temp = target.with_name(target.name + ".tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(target)

    return write


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--jina-extra", type=Path, default=None)
    parser.add_argument("--cmax-extra", type=Path, default=None)
    parser.add_argument("--chunk-target", type=int, default=None, help="Cal-FUSE4-free: chunks of FUSE4-thrC")
    parser.add_argument("--t-c", type=float, default=None, help="FUSE4-free: t_c (default FUSE4_FREE_T_C)")
    parser.add_argument("--strict", action="store_true", help="stop when a pair has no Jina / C-max score")
    args = parser.parse_args(argv)
    base_names = list(dict.fromkeys(args.variant))
    settings = run_settings(base_names)
    suffix = DOCREAD if settings.get("docread") else ""
    names = base_names
    root = fuse.run_root()
    out = paths.resolve_path(args.out or root / "build")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S36 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    t_c = args.t_c if args.t_c is not None else FUSE4_FREE_T_C
    if settings.get("needs_t_c") and t_c is None:
        raise ValueError("FUSE4-free needs a calibrated t_c: run Cal-FUSE4-free, then set FUSE4_FREE_T_C or pass --t-c")
    if settings.get("calibrate") and args.chunk_target is None:
        raise ValueError("Cal-FUSE4-free needs --chunk-target")
    needs_sources = any(len(VARIANTS[n]["keys"]) > 2 for n in names)
    hooks: dict[str, Any] = {}
    source_files: dict[str, str] = {}
    if needs_sources:
        jina_extra = paths.resolve_path(args.jina_extra) if args.jina_extra else root / "extra" / "jina-scores.parquet"
        cmax_extra = paths.resolve_path(args.cmax_extra) if args.cmax_extra else root / "extra" / "cmax-scores.parquet"
        jina, jina_sha = fuse.load_jina(jina_extra)
        cmax, cmax_sha = fuse.load_cmax(cmax_extra)
        source_files = {**{f"jina:{k}": v for k, v in jina_sha.items()}, **{f"cmax:{k}": v for k, v in cmax_sha.items()}}
        hooks = {"rank_hook": fuse.source_hook(jina, cmax, args.strict)}
        hooks["pool_hook"] = lambda name, pool, _h=hooks["rank_hook"]: _h(pool)
    frames = {VARIANTS[n]["frame"] for n in names}
    if len(frames) != 1:
        raise ValueError("Build one frame (thrC or free) per run")
    frame = frames.pop()
    ext = fuse.s31_root() / "pharmacity-ext"
    kwargs: dict[str, Any] = dict(
        pool13_index=ext / "pool13_index.parquet", q06_scores=ext / "q06_scores.parquet",
        rank_keys_override={n: VARIANTS[n]["keys"] for n in names}, variant_alias={n: VARIANTS[n]["alias"] for n in names},
        rank_weights={f"{n}{suffix}": VARIANTS[n]["weights"] for n in names if VARIANTS[n].get("weights")}, name_suffix=suffix,
        no_extra_chunks=bool(settings.get("no_extra_chunks")), calibrate_tc=bool(settings.get("calibrate")),
        chunk_target=settings.get("chunk_target") or args.chunk_target)
    if settings.get("chunk_mode"):
        kwargs["chunk_mode_override"] = {f"{n}{suffix}": settings["chunk_mode"] for n in names}
    if settings.get("needs_t_c"):
        kwargs["chunk_t_c"] = t_c
    if frame == "free":
        kwargs.update(extra_scores=fuse.s31_root() / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet",
                      extra_dir=fuse.s31_root() / "pool-extra",
                      orig_scores=fuse.s31_root() / "gpu" / "qwen3-reranker-4b" / "scores-pool13.parquet")
        if "pool_hook" in hooks:
            kwargs["pool_hook"] = hooks["pool_hook"]
    elif "rank_hook" in hooks:
        kwargs["rank_hook"] = hooks["rank_hook"]
    if suffix:
        docsel_dir = fuse.run_root() / "docsel"
        paths.assert_writable(docsel_dir)
        docsel_dir.mkdir(parents=True, exist_ok=True)
        pcal = {n: pd.read_parquet(fuse.run_root() / "cal" / f"pcal-{VARIANTS[n]['cal']}.parquet") for n in names if VARIANTS[n].get("cal")}
        kwargs["docsel_hook"] = docsel_writer(docsel_dir, suffix, t_c if settings.get("needs_t_c") else None, pcal)
    keeps = {f"{n}{suffix}": {int(q): int(k) for q, k in json.loads((fuse.run_root() / "cal" / f"keep-{n}.json").read_text(encoding="utf-8")).items()}
             for n in names if VARIANTS[n].get("cal")}
    if keeps:
        kwargs["keep_override"] = keeps
    results = s31_build.build(out, names, "qwen3-reranker-4b", ext / "qwen3-reranker-4b-scores-pool13.parquet", fuse.s31_root() / "pool", **kwargs)
    failed = []
    for sidecar in results:
        name = sidecar["name"][: -len(suffix)] if suffix and sidecar["name"].endswith(suffix) else sidecar["name"]
        if sidecar.get("status") == "calibrated_not_built":
            continue
        sidecar["s36"] = {"variant": VARIANTS[name]["alias"], "frame": frame, "source_files": source_files,
                          "chunk_t_c": t_c if settings.get("needs_t_c") else None}
        target = VARIANTS[name].get("target")
        if target:
            sidecar["regression_target_json_sha256"] = target
            sidecar["regression_pass"] = sidecar.get("json_sha256") == target
            if not sidecar["regression_pass"]:
                failed.append(name)
        _atomic_json(out / f"{sidecar['name']}.s31.json", sidecar)
    if failed:
        print(json.dumps({"regression_failed": failed}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
