"""S35 part 2: Jina Reranker v3.5 (listwise, 0.6B) on the S31 cosine pools.

``python -m r2ai.s35.jina prepare``   pool of (query, doc_text) pairs: J4-K40 pool (cosine top-211) U pool13
``python -m r2ai.s35.jina pilot``     20 queries: speed, VRAM, block-layout stability, correlations
``python -m r2ai.s35.jina score``     all queries, resumable per part of ``--part-queries`` queries
``python -m r2ai.s35.jina collect``   merge the parts into ``scores-all.parquet`` (query_id, doc_id, score)

Model ``jinaai/jina-reranker-v3.5`` pinned to REVISION (HF commit 2026-07-30, CC-BY-NC-4.0, 0.6B, Qwen3-0.6B
base, ``trust_remote_code`` = the repo's ``modeling.py``, read before use). One forward pass puts every doc of a
block in a single context (``<passage>`` + embed token each) and the query at the end; a doc embedding is
projected to 512-d and scored by cosine with the block's query embedding.

Blocks and normalisation (pilot 20 queries, docs/s35-part-jina.md): a query pool is cut into ``ceil(n / 10)`` balanced
blocks of <= 10 docs in a seeded shuffled order (layout ``b10s0``; the cap is 50). A doc embedding sees the passages
before it, so the pool order matters (cosine order made the scores nearly uncorrelated with cosine, Spearman 0.12), a
shuffle removes that bias and blocks of 10 were as good as 50 against the 4B scores and twice as fast. The final score is
the model's own multi-block merge, as in ``JinaForRanking.rerank``: the block query embeddings are averaged with weight
``max((1 + score) / 2)`` and every doc is scored by cosine against that single query embedding, so all docs of a query
share one reference vector (a per-block z-score was less stable: 0.77 vs 0.87 between two shuffles). The per-block
cosine and its z-score are kept as ``score_block`` / ``z_block``. Several shuffles (``b10s1`` ...) can be averaged in
``collect --layout b10s0,b10s1``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

MODEL_ID = "jinaai/jina-reranker-v3.5"
REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
MAX_BLOCK = 50  # user cap; the run uses blocks of 10 (see docs/s35-part-jina.md)
DEFAULT_LAYOUT = "b10s0"
LAYOUT_SEED = 35
PILOT_QUERIES = 20
PART_QUERIES = 10
SCORE_COLUMNS = ["query_id", "doc_id", "score", "score_block", "z_block", "block", "n_blocks"]


def run_root() -> Path:
    from r2ai import paths

    return paths.run_dir("s35-part-jina")


def _guard(*targets: Path) -> None:
    try:
        from r2ai import paths
    except Exception:  # pragma: no cover - outside the repo
        return
    for target in targets:
        paths.assert_writable(target)
        paths.assert_writable(target.with_name(target.name + ".tmp"))


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


# ------------------------------------------------------------------------------------------ pool

def prepare(out: Path) -> dict[str, Any]:
    """``pool.parquet``: one row per (query, doc) of the J4-K40 pool U pool13, with the S31 doc_text (512 Qwen3 tokens)."""
    import pyarrow.parquet as pq

    from r2ai import paths
    from r2ai.s31 import build as s31_build
    from r2ai.s33.pool import s31_text_shards
    from r2ai.s35 import part

    started = time.monotonic()
    _guard(out / "pool.parquet", out / "pool_manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    pools, _ = part.load_pools()
    root = part.s31_root()
    ext = root / "pharmacity-ext"
    frame = s31_build.load_candidates(root / "pool", ext / "qwen3-reranker-4b-scores-pool13.parquet",
                                      ext / "pool13_index.parquet", ext / "q06_scores.parquet")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for qid, prows in pools.items():
        for r in prows:
            rows[(qid, int(r["doc_id"]))] = {"query_id": qid, "doc_id": int(r["doc_id"]), "cosine_rank": int(r["cosine_rank"]),
                                             "cosine_score": float(r["cosine_score"]), "p4b": float(r["judge_score"]),
                                             "in_pool211": True, "in_pool13": False}
    for qid, doc_id, cosine_rank, cos, p4b in zip(frame.query_id.astype(int), frame.doc_id.astype(int), frame.cosine_rank.astype(int),
                                                  frame.cosine_score.astype(float), frame.judge_score.astype(float)):
        row = rows.setdefault((qid, doc_id), {"query_id": qid, "doc_id": doc_id, "cosine_rank": 10_000 + cosine_rank,
                                              "cosine_score": cos, "p4b": p4b, "in_pool211": False, "in_pool13": False})
        row["in_pool13"] = True
    pool = pd.DataFrame(list(rows.values()))
    texts = []
    for shard in s31_text_shards():
        paths.require_inputs(shard)
        part_frame = pq.read_table(shard, columns=["query_id", "doc_id", "lang", "query_text", "doc_text", "doc_tokens"]).to_pandas()
        key = pd.MultiIndex.from_frame(part_frame[["query_id", "doc_id"]])
        texts.append(part_frame[key.isin(pd.MultiIndex.from_frame(pool[["query_id", "doc_id"]]))])
    texts = pd.concat(texts, ignore_index=True).drop_duplicates(["query_id", "doc_id"])
    merged = pool.merge(texts, on=["query_id", "doc_id"], how="left", validate="one_to_one")
    if merged.doc_text.isna().any():
        raise ValueError(f"{int(merged.doc_text.isna().sum())} pool pairs have no S31 doc text")
    merged = merged.sort_values(["query_id", "cosine_rank"], kind="stable").reset_index(drop=True)
    _atomic_frame(out / "pool.parquet", merged)
    per_query = merged.groupby("query_id").size()
    manifest = {"pairs": len(merged), "queries": int(merged.query_id.nunique()), "pairs_pool211": int(merged.in_pool211.sum()),
                "pairs_pool13": int(merged.in_pool13.sum()), "pairs_pool13_only": int((~merged.in_pool211).sum()),
                "pairs_per_query": {"mean": float(per_query.mean()), "min": int(per_query.min()), "max": int(per_query.max())},
                "doc_tokens_qwen_mean": float(merged.doc_tokens.mean()), "lang": merged.lang.value_counts().to_dict(),
                "seconds": round(time.monotonic() - started, 1)}
    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------------------------- model

def _causal_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Causal attention of one unpadded sequence with torch's memory-efficient kernel.

    HF's ``sdpa`` path passes a bool n x n mask and, with ``enable_gqa``, torch rejects the efficient kernel and falls
    back to the math kernel (42 GiB at 37k tokens). Repeating the KV heads and using ``is_causal`` needs 0.55 GiB at
    24k tokens; the cuDNN kernel is 1.5x faster than the efficient one on the RTX 3050 (16 vs 11 TFLOPS). Only valid for batch size 1 without padding, which is how every block is run.
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import repeat_kv

    key, value = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=scaling, is_causal=True)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    return None


def load_model(device: str = "cuda"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AttentionInterface, AutoModel
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register("jina_causal", _causal_sdpa)
    AttentionMaskInterface.register("jina_causal", _no_mask)
    path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="jina_causal")
    model = model.to(device).eval()
    model._ensure_tokenizer()
    return model, path


def layout_blocks(n: int, query_id: int, layout: str) -> list[np.ndarray]:
    """Balanced blocks of <= B doc positions. ``layout`` = ``b<B>cos`` (pool order) or ``b<B>s<seed>`` (seeded shuffle)."""
    match = re.fullmatch(r"b(\d+)(cos|s\d+)", layout)
    if not match or not 1 <= int(match.group(1)) <= MAX_BLOCK:
        raise ValueError(f"Unknown layout {layout!r}; use b<B>cos or b<B>s<seed> with B <= {MAX_BLOCK}")
    size, kind = int(match.group(1)), match.group(2)
    order = np.arange(n) if kind == "cos" else np.random.default_rng([LAYOUT_SEED + int(kind[1:]), query_id]).permutation(n)
    return [b for b in np.array_split(order, max(1, math.ceil(n / size))) if len(b)]


def embed_block(model, query: str, docs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Doc embeddings (n, 512) and the query embedding (512,) of one listwise context, float32 on CPU.

    Same computation as ``JinaForRanking._compute_single_batch`` (checked bit-identical on a small example)
    but with ``_causal_sdpa`` (one unpadded sequence): the stock HF mask path builds the full n x n score matrix and
    runs out of 6 GB at ~25k tokens.
    """
    import sys

    import torch

    with torch.no_grad():
        return _embed_block(model, query, docs, sys, torch)


def _embed_block(model, query, docs, sys, torch):

    module = sys.modules[type(model).__module__]
    prompt = module.format_docs_prompts_func(query, list(docs), instruction=None, special_tokens=model.special_tokens, no_thinking=True)
    ids = model._tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    is_doc, is_query = ids[0] == model.doc_embed_token_id, ids[0] == model.query_embed_token_id
    if int(is_doc.sum()) != len(docs) or int(is_query.sum()) != 1:
        raise ValueError(f"{int(is_doc.sum())} doc markers for {len(docs)} docs, {int(is_query.sum())} query markers")
    doc_emb = model.projector(hidden[is_doc]).float().cpu().numpy()
    query_emb = model.projector(hidden[is_query]).float().cpu().numpy()[0]
    return doc_emb, query_emb


def score_query(model, query: str, docs: Sequence[str], query_id: int, layout: str = DEFAULT_LAYOUT, embed=embed_block) -> dict[str, np.ndarray]:
    """Listwise scores of ``docs`` for ``query`` (see module docstring); arrays are in the order of ``docs``."""
    import torch

    n = len(docs)
    blocks = layout_blocks(n, query_id, layout)
    doc_emb = np.zeros((n, 512), dtype=np.float32)
    block_of = np.zeros(n, dtype=np.int32)
    score_block = np.zeros(n, dtype=np.float32)
    z_block = np.zeros(n, dtype=np.float32)
    q_embs, weights = [], []
    with torch.no_grad():
        for b, idx in enumerate(blocks):
            d, q = embed(model, query, [docs[i] for i in idx])
            s = d @ q / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))
            doc_emb[idx], block_of[idx], score_block[idx] = d, b, s
            z_block[idx] = (s - s.mean()) / (s.std() + 1e-9)
            q_embs.append(q)
            weights.append(float(((1.0 + s) / 2.0).max()))
    q = np.average(np.array(q_embs), axis=0, weights=weights)
    merged = doc_emb @ q / (np.linalg.norm(q) * np.linalg.norm(doc_emb, axis=1))
    return {"score": merged.astype(np.float32), "score_block": score_block, "z_block": z_block, "block": block_of,
            "n_blocks": np.full(n, len(blocks), dtype=np.int32)}


def _score_frame(group: pd.DataFrame, result: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({"query_id": group.query_id.to_numpy(), "doc_id": group.doc_id.to_numpy(), **result})[SCORE_COLUMNS]


# ------------------------------------------------------------------------------------------ pilot

PILOT_LAYOUTS = ("b50s0", "b50s1", "b25s0", "b10s0", "b50cos")


def _per_query_spearman(frame: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    from scipy.stats import spearmanr

    vals = frame.groupby("query_id").apply(lambda g: spearmanr(g[left], g[right]).statistic, include_groups=False)
    return {"mean": float(vals.mean()), "p10": float(vals.quantile(0.1)), "min": float(vals.min())}


def pilot(pool_path: Path, out: Path, queries: int = PILOT_QUERIES, layouts: Sequence[str] = PILOT_LAYOUTS) -> dict[str, Any]:
    """Score ``queries`` evenly spaced queries (their cosine top-211) under several block layouts; the first is the reference.

    Reports speed, VRAM, per-query Spearman between layouts and with 4B / cosine, the stability of the merged score
    against the per-block cosine and its z-score, and, when several shuffles of one block size ran, how much averaging
    them raises the agreement with a held-out shuffle.
    """
    import torch

    layouts = tuple(layouts)
    ref = layouts[0]
    pool = pd.read_parquet(pool_path)
    ids = np.array(sorted(pool.query_id.unique()))
    pick = ids[np.linspace(0, len(ids) - 1, queries).round().astype(int)]
    model, path = load_model()
    torch.cuda.reset_peak_memory_stats()
    scores: dict[str, list[pd.DataFrame]] = {name: [] for name in layouts}
    timing: dict[str, list[dict[str, float]]] = {name: [] for name in layouts}
    for qid in pick:
        group = pool[(pool.query_id == qid) & pool.in_pool211].reset_index(drop=True)
        for name in layouts:
            torch.cuda.synchronize()
            t0 = time.monotonic()
            res = score_query(model, group.query_text.iat[0], group.doc_text.tolist(), int(qid), name)
            torch.cuda.synchronize()
            timing[name].append({"docs": len(group), "tokens": int(group.doc_tokens.sum()), "seconds": time.monotonic() - t0})
            scores[name].append(_score_frame(group, res))
        print(json.dumps({"query_id": int(qid), "docs": len(group), **{n: round(timing[n][-1]["seconds"], 1) for n in layouts}}), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    base = pool[pool.in_pool211][["query_id", "doc_id", "cosine_score", "p4b"]]
    frames = {name: pd.concat(parts, ignore_index=True).merge(base, on=["query_id", "doc_id"]) for name, parts in scores.items()}
    wide = frames[ref][["query_id", "doc_id", "cosine_score", "p4b"]].copy()
    for name, frame in frames.items():
        wide[name] = frame.score.to_numpy()
        wide[name + ":z"] = frame.z_block.to_numpy()
        wide[name + ":block"] = frame.score_block.to_numpy()
    k = 40
    top = lambda col: {q: set(g.nlargest(k, col).doc_id) for q, g in wide.groupby("query_id")}  # noqa: E731
    top_ref = top(ref)
    per_layout = {}
    for name in layouts:
        sec = np.array([t["seconds"] for t in timing[name]])
        per_layout[name] = {
            "seconds_per_query": float(sec.mean()),
            "docs_per_second": float(sum(t["docs"] for t in timing[name]) / sec.sum()),
            "tokens_per_second": float(sum(t["tokens"] for t in timing[name]) / sec.sum()),
            f"spearman_vs_{ref}": _per_query_spearman(wide, ref, name) if name != ref else None,
            "spearman_vs_p4b": _per_query_spearman(wide, name, "p4b"),
            "spearman_vs_cosine": _per_query_spearman(wide, name, "cosine_score"),
            f"top40_overlap_vs_{ref}": float(np.mean([len(top_ref[q] & s) / k for q, s in top(name).items()])) if name != ref else None,
        }
    norm = {f"merge~z_block ({ref})": _per_query_spearman(wide, ref, f"{ref}:z"),
            f"merge~score_block ({ref})": _per_query_spearman(wide, ref, f"{ref}:block")}
    if len(layouts) > 1:
        other = layouts[1]
        norm.update({f"{ref}~{other} merge": _per_query_spearman(wide, ref, other),
                     f"{ref}~{other} z_block": _per_query_spearman(wide, f"{ref}:z", f"{other}:z"),
                     f"{ref}~{other} score_block": _per_query_spearman(wide, f"{ref}:block", f"{other}:block")})
    ensemble = {}
    shuffled = [n for n in layouts if re.fullmatch(r"b\d+s\d+", n)]
    for size in sorted({n.split("s")[0] for n in shuffled}):
        group_names = [n for n in shuffled if n.startswith(size + "s")]
        if len(group_names) >= 3:
            held, rest = group_names[-1], group_names[:-1]
            wide["_mean"] = wide[rest].mean(axis=1)
            ensemble[size] = {"shuffles": group_names, "pair_agreement": _per_query_spearman(wide, rest[0], rest[1]),
                              "mean_of_others_vs_held_out": _per_query_spearman(wide, "_mean", held),
                              "single_vs_held_out": _per_query_spearman(wide, rest[0], held),
                              "mean_all_vs_p4b": _per_query_spearman(wide.assign(_all=wide[group_names].mean(axis=1)), "_all", "p4b"),
                              "single_vs_p4b": _per_query_spearman(wide, rest[0], "p4b")}
    manifest = pool_manifest(pool_path)
    mean_docs = float(np.mean([t["docs"] for t in timing[ref]]))
    report = {
        "model": MODEL_ID, "revision": REVISION, "snapshot": path, "dtype": "bfloat16", "attention": "cuDNN causal SDPA (repeat KV)",
        "queries": int(queries), "pool": "cosine top-211 (in_pool211 rows)", "mean_docs_per_query": mean_docs,
        "peak_vram_gib": peak, "layouts": per_layout, "normalisation_stability": norm, "shuffle_ensemble": ensemble,
        "eta_full_hours_by_layout": {n: float(np.mean([t["seconds"] for t in timing[n]]) * manifest["pairs_per_query"]["mean"] / mean_docs
                                              * manifest["queries"] / 3600) for n in layouts},
        "full_pool_pairs_per_query_mean": manifest["pairs_per_query"]["mean"],
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    out.mkdir(parents=True, exist_ok=True)
    _guard(out / "pilot.json", out / "pilot_scores.parquet")
    (out / "pilot.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _atomic_frame(out / "pilot_scores.parquet", wide.drop(columns=["_mean"], errors="ignore"))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def pool_manifest(pool_path: Path) -> dict[str, Any]:
    return json.loads((pool_path.parent / "pool_manifest.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- score

def score(pool_path: Path, out: Path, part_queries: int = PART_QUERIES, max_seconds: float | None = None,
          layout: str = DEFAULT_LAYOUT) -> dict[str, Any]:
    import torch

    pool = pd.read_parquet(pool_path)
    ids = sorted(pool.query_id.unique())
    parts = [ids[i : i + part_queries] for i in range(0, len(ids), part_queries)]
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / f"parts-{layout}"
    parts_dir.mkdir(exist_ok=True)
    model, path = load_model()
    started = time.monotonic()
    done = skipped = 0
    for index, part_ids in enumerate(parts):
        target = parts_dir / f"part-{index:04d}.parquet"
        if target.exists():
            skipped += 1
            continue
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            break
        _guard(target)
        t0 = time.monotonic()
        frames = []
        for qid in part_ids:
            group = pool[pool.query_id == qid].reset_index(drop=True)
            frames.append(_score_frame(group, score_query(model, group.query_text.iat[0], group.doc_text.tolist(), int(qid), layout)))
        frame = pd.concat(frames, ignore_index=True)
        if not np.isfinite(frame[["score", "score_block", "z_block"]].to_numpy()).all():
            raise ValueError(f"Non-finite Jina scores in part {index}")
        _atomic_frame(target, frame)
        done += 1
        print(json.dumps({"part": index, "of": len(parts), "queries": len(part_ids), "pairs": len(frame),
                          "seconds": round(time.monotonic() - t0, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    return {"parts": len(parts), "scored_now": done, "already_done": skipped, "snapshot": path, "layout": layout}


def collect(pool_path: Path, out: Path, layouts: Sequence[str] = (DEFAULT_LAYOUT,)) -> dict[str, Any]:
    """``scores-all.parquet``: ``score`` = mean over the layouts (one pass = that pass's merged score), plus one column per layout."""
    pool = pd.read_parquet(pool_path, columns=["query_id", "doc_id"])
    merged = pool
    counts = {}
    for layout in layouts:
        files = sorted((out / f"parts-{layout}").glob("part-*.parquet"))
        scores = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if scores.duplicated(["query_id", "doc_id"]).any():
            raise ValueError(f"Duplicate (query_id, doc_id) in the Jina parts of {layout}")
        merged = merged.merge(scores[["query_id", "doc_id", "score"]].rename(columns={"score": f"score_{layout}"}),
                              on=["query_id", "doc_id"], how="left", validate="one_to_one")
        if merged[f"score_{layout}"].isna().any():
            raise ValueError(f"Jina {layout} misses {int(merged[f'score_{layout}'].isna().sum())} of {len(pool)} pool pairs ({len(files)} parts)")
        counts[layout] = len(files)
    merged["score"] = merged[[f"score_{layout}" for layout in layouts]].mean(axis=1)
    _guard(out / "scores-all.parquet")
    _atomic_frame(out / "scores-all.parquet", merged)
    return {"pairs": len(merged), "layouts": list(layouts), "parts": counts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "pilot", "score", "collect"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--part-queries", type=int, default=PART_QUERIES)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--queries", type=int, default=PILOT_QUERIES)
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help="score: one layout; collect: comma list averaged")
    parser.add_argument("--layouts", default=None, help="pilot: comma list of layouts, first = reference")
    parser.add_argument("--pilot-dir", default="pilot")
    args = parser.parse_args(argv)
    from r2ai import paths

    root = paths.resolve_path(args.out) if args.out else run_root() / "jina"
    if not root.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S35 output must stay below RUNS_DIR: {root}")
    pool_path = root / "pool.parquet"
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "pilot":
        result = pilot(pool_path, root / args.pilot_dir, args.queries, tuple(args.layouts.split(",")) if args.layouts else PILOT_LAYOUTS)
    elif args.command == "score":
        result = score(pool_path, root, args.part_queries, args.max_seconds, args.layout)
    else:
        result = collect(pool_path, root, tuple(args.layout.split(",")))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
