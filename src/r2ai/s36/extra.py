"""S36 score supplements for the free frame: Jina and C-max pairs that S35 / S32 did not score.

``python -m r2ai.s36.extra coverage``       count the pairs of each frame lacking a Jina / C-max score, write the missing lists
``python -m r2ai.s36.extra jina-prepare``   Jina pool (query, doc_text) of the missing pairs -> ``extra/jina/pool.parquet``
``python -m r2ai.s36.extra jina-check``     does a separately scored subset agree with the S35 scores of the same docs?
``python -m r2ai.s36.extra cmax-prepare``   Qwen3-Reranker-0.6B pool (chunks of the missing pairs, doc_text for docs without chunk)
``python -m r2ai.s36.extra cmax-reduce``    chunk scores -> ``extra/cmax-scores.parquet`` (query_id, doc_id, cmax)
``python -m r2ai.s36.extra jina-reduce``    Jina parts -> ``extra/jina-scores.parquet`` (query_id, doc_id, score)

Chunks are chosen exactly as S32 (``r2ai.s33.pool.select_like_s32``: cosine of the S29 query vector and the BGE-M3 chunk vector,
top 3, ties by file order) and a gate re-selects ``GATE_PAIRS`` S32 pairs through the same call. The 0.6B model, instruction,
template and 512-token cut are those of S32 (``r2ai.s31.judge_score``). C-max of a pair is scored only when all of its rows are.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s33 import pool as s33_pool
from r2ai.s36 import fuse

NEW_PAIR_ID_BASE = 40_000_000
ANCHOR_MIN = 30
ANCHOR_SEED = 36
GATE_PAIRS = 1_000
GATE_SEED = 36
TEXT_COLUMNS = ["query_id", "doc_id", "lang", "query_text", "doc_text", "doc_tokens"]


def extra_root() -> Path:
    return fuse.run_root() / "extra"


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    paths.assert_writable(path)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def load_texts(pairs: set[tuple[int, int]]) -> pd.DataFrame:
    """S31 query / doc text (512 Qwen3 tokens) and language of the given (query, doc) pairs."""
    wanted = pd.MultiIndex.from_tuples(sorted(pairs), names=["query_id", "doc_id"])
    frames = []
    for shard in s33_pool.s31_text_shards():
        paths.require_inputs(shard)
        part = pq.read_table(shard, columns=TEXT_COLUMNS).to_pandas()
        frames.append(part[pd.MultiIndex.from_frame(part[["query_id", "doc_id"]]).isin(wanted)])
    texts = pd.concat(frames, ignore_index=True).drop_duplicates(["query_id", "doc_id"])
    if len(texts) != len(pairs):
        raise ValueError(f"{len(pairs) - len(texts)} pairs have no S31 text")
    return texts


def pair_table(free: dict[int, list[dict[str, Any]]], pairs: set[tuple[int, int]]) -> pd.DataFrame:
    rows = [{"query_id": q, "doc_id": int(r["doc_id"]), "pair_id": int(r["pair_id"]), "cosine_rank": int(r["cosine_rank"]),
             "cosine_score": float(r["cosine_score"]), "origin": r["origin"]}
            for q, rs in free.items() for r in rs if (q, int(r["doc_id"])) in pairs]
    return pd.DataFrame(rows).sort_values(["query_id", "cosine_rank"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------- coverage

def coverage_command() -> dict[str, Any]:
    ranked, union, free, keep, free_report, _ = fuse.load_frames()
    jina, jina_sha = fuse.load_jina(None)
    cmax, cmax_sha = fuse.load_cmax(None)
    report, missing = fuse.coverage(ranked, union, free, jina, cmax)
    out = extra_root()
    out.mkdir(parents=True, exist_ok=True)
    for name, pairs in missing.items():
        _atomic_frame(out / f"missing_{name}.parquet", pair_table(free, pairs))
    result = {"frames": report, "free_adaptive": {k: free_report[k] for k in ("queries_deepened", "pool_size", "total")},
              "jina_scores": jina_sha, "cmax_scores": cmax_sha}
    _atomic_json(out / "coverage.json", result)
    return result


# ------------------------------------------------------------------------------------- Jina

def jina_prepare() -> dict[str, Any]:
    """``extra/jina/pool.parquet`` in the layout ``r2ai.s35.jina.score`` reads (query_text, doc_text, cosine_rank)."""
    out = extra_root() / "jina"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(extra_root() / "missing_jina.parquet")
    texts = load_texts(set(zip(table.query_id.astype(int), table.doc_id.astype(int))))
    merged = table.merge(texts, on=["query_id", "doc_id"], how="left", validate="one_to_one")
    merged["anchor"] = False
    # a query with < ANCHOR_MIN missing docs gets already scored docs (seeded pick) as anchors, so that its supplement has
    # >= 3 blocks and one reference vector; the anchors' new scores are dropped by jina_reduce
    s35_pool = pd.read_parquet(paths.RUNS_DIR / "s35-part-jina" / "jina" / "pool.parquet")
    anchors = []
    for qid, count in merged.groupby("query_id").size().items():
        if count < ANCHOR_MIN:
            have = s35_pool[s35_pool.query_id == qid]
            pick = np.random.default_rng([ANCHOR_SEED, int(qid)]).choice(len(have), size=min(ANCHOR_MIN - count, len(have)), replace=False)
            anchors.append(have.iloc[np.sort(pick)][["query_id", "doc_id", "cosine_rank", "cosine_score", "lang", "query_text", "doc_text", "doc_tokens"]]
                           .assign(anchor=True, pair_id=-1, origin="anchor"))
    if anchors:
        merged = pd.concat([merged, *anchors], ignore_index=True)
    merged = merged.sort_values(["query_id", "cosine_rank"], kind="stable").reset_index(drop=True)
    _atomic_frame(out / "pool.parquet", merged)
    per_query = merged[~merged.anchor].groupby("query_id").size()
    manifest = {"pairs": int((~merged.anchor).sum()), "anchor_rows": int(merged.anchor.sum()), "queries": int(merged.query_id.nunique()),
                "pairs_per_query": {"mean": float(per_query.mean()), "min": int(per_query.min()), "max": int(per_query.max())},
                "blocks_of_10_per_query_mean": float(np.ceil(per_query / 10).mean()), "lang": merged.lang.value_counts().to_dict()}
    _atomic_json(out / "pool_manifest.json", manifest)
    return manifest


def jina_check(queries: int = 8, sizes: Sequence[int] = (120, 40, 10), seed: int = 36) -> dict[str, Any]:
    """Score a random subset of already scored S35 docs as a separate set (own blocks, own query vector) and compare with
    their S35 scores: this is what a supplement does to the docs it adds to a query."""
    from scipy.stats import spearmanr

    from r2ai.s35 import jina

    pool = pd.read_parquet(paths.RUNS_DIR / "s35-part-jina" / "jina" / "pool.parquet")
    scores = pd.read_parquet(fuse.s35_jina_scores(), columns=["query_id", "doc_id", "score"])
    rng = np.random.default_rng(seed)
    qids = sorted(rng.choice(sorted(pool.query_id.unique()), size=queries, replace=False).tolist())
    model, _ = jina.load_model()
    rows, started = [], time.monotonic()
    for qid in qids:
        group = pool[pool.query_id == qid].merge(scores, on=["query_id", "doc_id"]).reset_index(drop=True)
        for size in sizes:
            pick = np.sort(rng.choice(len(group), size=min(size, len(group)), replace=False))
            sub = group.iloc[pick].reset_index(drop=True)
            res = jina.score_query(model, sub.query_text.iat[0], sub.doc_text.tolist(), int(qid), jina.DEFAULT_LAYOUT)
            old, new = sub.score.to_numpy(), res["score"]
            rows.append({"query_id": int(qid), "subset_docs": len(sub), "blocks": int(res["n_blocks"][0]), "spearman": float(spearmanr(old, new)[0]),
                         "mean_shift": float(np.mean(new - old)), "std_shift": float(np.std(new - old)),
                         "old_std": float(np.std(old)), "new_std": float(np.std(new))})
    by_size = {}
    for size in sizes:
        part = [r for r in rows if r["subset_docs"] == size]
        by_size[str(size)] = {"queries": len(part), "mean_spearman": float(np.mean([r["spearman"] for r in part])),
                              "min_spearman": float(np.min([r["spearman"] for r in part])),
                              "mean_shift": float(np.mean([r["mean_shift"] for r in part])),
                              "mean_abs_shift": float(np.mean([abs(r["mean_shift"]) for r in part])),
                              "mean_std_ratio": float(np.mean([r["new_std"] / r["old_std"] for r in part]))}
    result = {"rows": rows, "by_subset_size": by_size, "seconds": round(time.monotonic() - started, 1),
              "reference": "two full shuffles of the same pool agree with Spearman ~0.87 (docs/s35-part-jina.md)"}
    _atomic_json(extra_root() / "jina" / "consistency.json", result)
    return result


def jina_reduce() -> dict[str, Any]:
    out = extra_root() / "jina"
    pool = pd.read_parquet(out / "pool.parquet", columns=["query_id", "doc_id", "anchor"])
    anchor_rows = pool[pool.anchor]
    pool = pool[~pool.anchor][["query_id", "doc_id"]]
    files = sorted((out / "parts-b10s0").glob("part-*.parquet"))
    scores = pd.concat([pd.read_parquet(f, columns=["query_id", "doc_id", "score"]) for f in files], ignore_index=True) if files else pool.iloc[:0].assign(score=np.float32(0))
    if scores.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Duplicate supplement Jina scores")
    scores = scores.merge(pool, on=["query_id", "doc_id"], how="inner")
    _atomic_frame(extra_root() / "jina-scores.parquet", scores)
    anchor_scores = None
    if files and len(anchor_rows):
        all_scores = pd.concat([pd.read_parquet(f, columns=["query_id", "doc_id", "score"]) for f in files], ignore_index=True)
        old = pd.read_parquet(fuse.s35_jina_scores(), columns=["query_id", "doc_id", "score"]).rename(columns={"score": "old"})
        cmp = anchor_rows[["query_id", "doc_id"]].merge(all_scores, on=["query_id", "doc_id"]).merge(old, on=["query_id", "doc_id"])
        anchor_scores = {"anchors_scored": len(cmp), "spearman_new_vs_s35": float(cmp.score.corr(cmp.old, method="spearman")) if len(cmp) > 2 else None}
    return {"pairs_pool": len(pool), "pairs_scored": len(scores), "parts": len(files), "anchors": anchor_scores}


# ------------------------------------------------------------------------------------- C-max

RCHK_TYPES = s33_pool.RCHK_COLUMNS


def cmax_prepare() -> dict[str, Any]:
    started = time.monotonic()
    out = extra_root() / "cmax"
    s33_pool._prepare_out(out)
    log = s33_pool._logger(out / "prepare.log")
    table = pd.read_parquet(extra_root() / "missing_cmax.parquet")
    keys = set(zip(table.query_id.astype(int), table.doc_id.astype(int)))
    log(f"missing C-max pairs: {len(table)} in {table.query_id.nunique()} queries")
    texts = load_texts(keys)
    table = table.merge(texts[["query_id", "doc_id", "lang"]], on=["query_id", "doc_id"], how="left", validate="one_to_one")

    root33 = fuse.run_root().parent / "s33-btc-baseline"
    rchk_index = pd.concat([pd.read_parquet(root33 / n / "chunk_pairs.parquet") for n in ("pool-rchk", "pool-rchk-tie")], ignore_index=True)
    rchk_fb = pd.concat([pd.read_parquet(root33 / n / "doc_fallback.parquet") for n in ("pool-rchk", "pool-rchk-tie")], ignore_index=True)
    table["key"] = list(zip(table.query_id.astype(int), table.doc_id.astype(int)))
    in_chunks = set(zip(rchk_index.query_id.astype(int), rchk_index.doc_id.astype(int)))
    in_fb = set(zip(rchk_fb.query_id.astype(int), rchk_fb.doc_id.astype(int)))
    have = table[table.key.isin(in_chunks)]
    fb_old = table[table.key.isin(in_fb) & ~table.key.isin(in_chunks)]
    new = table[~table.key.isin(in_chunks) & ~table.key.isin(in_fb)]
    log(f"S33 chunk rows exist for {len(have)} pairs, S33 no-chunk for {len(fb_old)}, new selection for {len(new)}")

    # ---- S33 rows verbatim (shards carry the texts)
    want = set(have.key)
    parts = []
    for name in ("pool-rchk", "pool-rchk-tie"):
        for shard in sorted((root33 / name / "shards").glob("pool-*.parquet")):
            frame = pq.read_table(shard).to_pandas()
            sel = pd.MultiIndex.from_frame(frame[["query_id", "doc_id"]]).isin(pd.MultiIndex.from_tuples(sorted(want)))
            parts.append(frame[sel])
    old_rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(RCHK_TYPES))
    if set(zip(old_rows.query_id.astype(int), old_rows.doc_id.astype(int))) != want:
        raise ValueError("S33 R-chk shards miss chunk rows of pairs the index lists")
    old_rows["chunk_source"] = "s33"

    # ---- new selection + gate on S32 pairs, through the S33 code path
    pairs = new[["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang"]].rename(columns={"pair_id": "src_pair_id"}).copy()
    pairs["source"] = "s36_deep"
    s32_dir = s33_pool.s32_pool()
    s32_index = pd.read_parquet(s32_dir / "chunk_pairs.parquet")
    covered = s32_index.drop_duplicates("src_pair_id")[["src_pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang"]]
    gate_pairs = covered.sample(n=GATE_PAIRS, random_state=GATE_SEED).copy()
    gate_pairs["source"] = "gate"
    sel, fallback, info = s33_pool.select_like_s32(pd.concat([pairs, gate_pairs], ignore_index=True), log, s32_dir / "supplement_vectors.npz", out)
    gsel = sel[sel.source == "gate"]
    ref = s32_index[s32_index.src_pair_id.isin(gate_pairs.src_pair_id)]
    got = gsel.sort_values(["src_pair_id", "chunk_rank"]).groupby("src_pair_id").chunk_id.apply(list)
    exp = ref.sort_values(["src_pair_id", "chunk_rank"]).groupby("src_pair_id").chunk_id.apply(list)
    mismatch = [int(k) for k in exp.index if got.get(k) != exp[k]]
    gate = {"pairs": GATE_PAIRS, "seed": GATE_SEED, "mismatches": len(mismatch), "examples": mismatch[:5],
            "gate_fallback": int((fallback.source == "gate").sum())}
    _atomic_json(out / "gate.json", gate)
    log(f"gate {json.dumps(gate)}")
    if mismatch:
        raise ValueError("chunk selection differs from S32 on the gate sample; see gate.json")
    sel = s33_pool.attach_texts(sel[sel.source == "s36_deep"], log)
    queries = dict(zip(texts.query_id.astype(int), texts.query_text))
    sel["query_text"] = sel.query_id.map(queries)
    sel["chunk_source"] = "s36"
    sel["in_pool13"] = True
    sel["pair_id"] = -1  # renumbered below

    # ---- docs without any chunk: their S31 doc_text is scored (S32 fallback = the 0.6B doc_text score)
    fb_keys = set(fb_old.key) | set(zip(fallback[fallback.source == "s36_deep"].query_id.astype(int),
                                        fallback[fallback.source == "s36_deep"].doc_id.astype(int)))
    fb_table = table[table.key.isin(fb_keys)]
    fb_rows = fb_table.merge(texts, on=["query_id", "doc_id", "lang"], how="left", validate="one_to_one").rename(columns={"pair_id": "src_pair_id"})
    fb_rows["chunk_id"], fb_rows["chunk_rank"], fb_rows["chunk_cosine"] = -1, 0, np.nan
    fb_rows["field"], fb_rows["origin"], fb_rows["chunk_source"], fb_rows["in_pool13"] = "doc_text", "d", "fallback", True
    fb_rows["pair_id"] = -1  # renumbered below

    cols = list(RCHK_TYPES)
    rows = pd.concat([old_rows.assign(src_pair_id=old_rows.src_pair_id.astype(np.int64))[cols],
                      sel.assign(src_pair_id=sel.src_pair_id.astype(np.int64))[cols], fb_rows[cols]], ignore_index=True)
    rows = rows.sort_values(["query_id", "cosine_rank", "doc_id", "chunk_rank"], kind="stable").reset_index(drop=True)
    rows["pair_id"] = NEW_PAIR_ID_BASE + np.arange(len(rows), dtype=np.int64)
    if rows.duplicated(["query_id", "doc_id", "chunk_rank", "chunk_id"]).any():
        raise ValueError("Duplicate rows in the C-max supplement")
    covered_keys = set(zip(rows.query_id.astype(int), rows.doc_id.astype(int)))
    if covered_keys != keys:
        raise ValueError(f"{len(keys - covered_keys)} missing pairs have no row, {len(covered_keys - keys)} extra")
    shards = s33_pool.write_shards(out, rows, RCHK_TYPES, log)
    s33_pool._atomic_frame(out / "chunk_pairs.parquet", rows[[c for c in cols if c not in ("query_text", "doc_text")]])
    per_pair = rows.groupby(["query_id", "doc_id"]).size()
    manifest = {"status": "exported", "pairs": len(keys), "rows": len(rows), "rows_s33": int((rows.chunk_source == "s33").sum()),
                "rows_new": int((rows.chunk_source == "s36").sum()), "rows_doc_text_fallback": int((rows.chunk_source == "fallback").sum()),
                "pairs_s33_chunks": len(have), "pairs_new_selection": len(new), "pairs_doc_text": len(fb_table),
                "rows_per_pair": {str(k): int(v) for k, v in per_pair.value_counts().sort_index().items()},
                "doc_tokens_mean": float(rows.doc_tokens.mean()), "gate": gate, **info, "shards": shards,
                "seconds": round(time.monotonic() - started, 1)}
    _atomic_json(out / "manifest.json", manifest)
    return manifest


def cmax_reduce(scores_path: Path) -> dict[str, Any]:
    out = extra_root() / "cmax"
    index = pd.read_parquet(out / "chunk_pairs.parquet", columns=["pair_id", "query_id", "doc_id", "chunk_source"])
    scores = pd.read_parquet(scores_path, columns=["pair_id", "score"])
    if scores.pair_id.duplicated().any() or not np.isfinite(scores.score).all():
        raise ValueError("Duplicate or non-finite supplement chunk scores")
    merged = index.merge(scores, on="pair_id", how="left")
    grouped = merged.groupby(["query_id", "doc_id"])
    done = grouped.score.apply(lambda s: s.notna().all())
    best = grouped.score.max()
    result = best[done[done].index].rename("cmax").reset_index()
    result["fallback"] = result.set_index(["query_id", "doc_id"]).index.isin(
        merged[merged.chunk_source == "fallback"].set_index(["query_id", "doc_id"]).index)
    _atomic_frame(extra_root() / "cmax-scores.parquet", result)
    return {"pairs": int(len(done)), "pairs_scored": int(done.sum()), "pairs_incomplete": int((~done).sum()),
            "rows": len(index), "rows_scored": int(merged.score.notna().sum())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("coverage", "jina-prepare", "jina-check", "jina-reduce", "cmax-prepare", "cmax-reduce"))
    parser.add_argument("--scores", type=Path, default=None, help="cmax-reduce: scores-all.parquet of judge_score collect")
    args = parser.parse_args(argv)
    if args.command == "coverage":
        result = coverage_command()
    elif args.command == "jina-prepare":
        result = jina_prepare()
    elif args.command == "jina-check":
        result = jina_check()
    elif args.command == "jina-reduce":
        result = jina_reduce()
    elif args.command == "cmax-prepare":
        result = cmax_prepare()
    else:
        if args.scores is None:
            raise ValueError("cmax-reduce needs --scores")
        result = cmax_reduce(paths.resolve_path(args.scores))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
