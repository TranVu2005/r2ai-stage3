"""S38 chunk scores and per-doc chunk choice for the J3-thrC chunk carriers.

``python -m r2ai.s38.score plan``    pieces without a stored vector -> ``scores/embed_todo.parquet`` (+ count)
``python -m r2ai.s38.score embed``   BGE-M3 (``r2ai.index.bge_m3.M3Encoder``, max_len 512) for those pieces, parts of
                                     ``--part-size`` texts under ``scores/embed/`` (a rerun skips finished parts);
                                     ``--wait-gpu`` first waits until no other process computes on the GPU
``python -m r2ai.s38.score select``  ``scores/selection.parquet`` for every variant

Score of a piece (Vu, 23/10):
* t256: the S32 chunk-level judge (Qwen3-Reranker-0.6B, top-3 cosine chunks per pool13 pair) on the answer/body
  chunks it scored; a carrier without such a chunk ranks every t256 answer/body chunk by cosine.
* t128 and sec: cosine(S29 query vector, BGE-M3 vector of the piece); one scale per doc. A doc without structure
  takes its t256 chunks as sec pieces and the t256 choice (judge first).
Pieces of one carrier equal after the metric normalisation count once (the best one).
Vectors: canonical ``dense.npy`` for chunks with a vector, the S32 supplement vectors for the others, the S38
embeddings for pieces that are not a canonical chunk. Ties: lower piece index first.

Variants (chunks per carrier doc, in score order):
  CH-t256-1  best t256          CH-t256-2  best t256 + the 2nd when its score >= 0.5 x the 1st (same scale)
  CH-sec-1   best sec           CH-t128-1  best t128
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s38.sources import _atomic_json, _atomic_parquet, describe, run_root

S32_ROOT = Path("runs/s32-chunk-judge")
S32_SCORES = S32_ROOT / "gpu" / "qwen3-reranker-0.6b" / "scores-all.parquet"
S32_SCORES_SHA256 = "cc1177522694f90887b3b5da34a8522106093d5a81839ec238a0bf69a9231a17"
SECOND_RATIO = 0.5
VARIANTS = {"CH-t256-1": ("t256", 1), "CH-t256-2": ("t256", 2), "CH-sec-1": ("sec", 1), "CH-t128-1": ("t128", 1)}
DIM = 1024
PART_SIZE = 20_000
EMBED_BATCH = 64


def scores_dir() -> Path:
    return run_root() / "scores"


def _log_to(path: Path) -> Callable[[str], None]:
    def log(message: str) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                           "message": message}, ensure_ascii=False)
        print(line, flush=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    return log


# --------------------------------------------------------------------------------------- choice

def rank_pieces(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Order of pieces: score descending, lower piece index first on ties."""
    return np.lexsort((idx, -scores))


def choose(scores: Sequence[float], idx: Sequence[int], n: int, ratio: float = SECOND_RATIO) -> list[int]:
    """Positions of the chosen pieces: the best, then (n = 2) the next one if its score >= ratio x the best."""
    s, i = np.asarray(scores, dtype=np.float64), np.asarray(idx)
    if not len(s):
        return []
    order = rank_pieces(s, i)
    out = [int(order[0])]
    if n >= 2 and len(order) > 1 and s[order[1]] >= ratio * s[order[0]]:
        out.append(int(order[1]))
    return out


def select_variant(cands: pd.DataFrame, n: int, ratio: float = SECOND_RATIO) -> pd.DataFrame:
    """``cands``: query_id, doc_id, idx, score (one scale per (query, doc)); the rows ``choose`` keeps, with rank."""
    keys = ["query_id", "doc_id"]
    c = cands.sort_values([*keys, "score", "idx"], ascending=[True, True, False, True], kind="stable").copy()
    if "norm_sha" in c:   # pieces equal after the metric normalisation count once (the best one is kept)
        c = c.drop_duplicates([*keys, "norm_sha"], keep="first")
    c["rank"] = c.groupby(keys, sort=False).cumcount() + 1
    best = c.groupby(keys, sort=False).score.transform("first")
    keep = (c["rank"] == 1) | ((n >= 2) & (c["rank"] == 2) & (c.score >= ratio * best))
    return c[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------- vectors

def load_query_vectors() -> dict[int, np.ndarray]:
    from r2ai.s32.chunks import load_query_vectors as s32_qv

    manifest31 = json.loads((paths.RUNS_DIR / "s31-strong-judge" / "pool" / "manifest.json").read_text(encoding="utf-8"))
    return s32_qv(manifest31)


def piece_vectors(pieces: pd.DataFrame, canonical: Path, embed_dir: Path | None) -> np.ndarray:
    """fp16 [len(pieces), 1024]; raises when a piece has no vector."""
    out = np.zeros((len(pieces), DIM), dtype=np.float16)
    have = np.zeros(len(pieces), dtype=bool)
    dense = np.load(canonical / "vectors" / "dense.npy", mmap_mode="r")
    rows = pieces.vec_row.to_numpy()
    where = np.flatnonzero(rows >= 0)
    order = where[np.argsort(rows[where], kind="stable")]
    for begin in range(0, len(order), 200_000):
        take = order[begin:begin + 200_000]
        out[take] = dense[rows[take]]
    have[where] = True
    with np.load(paths.RUNS_DIR / "s32-chunk-judge" / "pool" / "supplement_vectors.npz") as sup:
        keep = sup["origin"] == "c"
        sup_row = {int(c): k for k, c in enumerate(sup["chunk_id"]) if keep[k]}
        need = np.flatnonzero(~have & (pieces.chunk_id.to_numpy() >= 0))
        hit = [(p, sup_row[int(c)]) for p, c in zip(need, pieces.chunk_id.to_numpy()[need]) if int(c) in sup_row]
        if hit:
            pos, src = map(np.array, zip(*hit))
            out[pos] = sup["dense"][src]
            have[pos] = True
    if embed_dir is not None and (embed_dir / "index.parquet").exists():
        index = pd.read_parquet(embed_dir / "index.parquet")
        key = dict(zip(index.text_sha, index.row))
        vecs = np.concatenate([np.load(p)["dense"] for p in sorted(embed_dir.glob("part-*.npz"))])
        need = np.flatnonzero(~have)
        shas = pieces.text_sha.to_numpy()[need]
        rows_e = np.array([key.get(s, -1) for s in shas])
        ok = rows_e >= 0
        out[need[ok]] = vecs[rows_e[ok]]
        have[need[ok]] = True
    if not have.all():
        raise ValueError(f"{int((~have).sum())} pieces without a vector (run plan + embed)")
    return out


def _text_sha(texts: Sequence[str]) -> list[str]:
    import hashlib

    return [hashlib.sha256(t.encode("utf-8")).hexdigest()[:32] for t in texts]


def plan(src: Path, out: Path, kinds: Sequence[str]) -> dict[str, Any]:
    pieces = pd.read_parquet(src / "pieces.parquet", columns=["kind", "chunk_id", "vec_row", "text", "n_tokens"])
    pieces = pieces[pieces.kind.isin(kinds)]
    with np.load(paths.RUNS_DIR / "s32-chunk-judge" / "pool" / "supplement_vectors.npz") as sup:
        sup_ids = set(sup["chunk_id"][sup["origin"] == "c"].astype(int))
    stored = (pieces.vec_row >= 0) | pieces.chunk_id.isin(sup_ids)
    todo = pieces[~stored].copy()
    todo["text_sha"] = _text_sha(todo.text.tolist())
    todo = todo.drop_duplicates("text_sha").sort_values("text_sha").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(todo[["text_sha", "text", "n_tokens", "kind"]], out / "embed_todo.parquet")
    t256_missing = int(((pieces.kind == "t256") & ~stored).sum())
    report = {"kinds": list(kinds), "pieces": len(pieces), "with_stored_vector": int(stored.sum()), "texts_to_embed": len(todo),
              "texts_to_embed_by_kind": todo.kind.value_counts().to_dict(), "t256_without_vector": t256_missing,
              "tokens_to_embed": describe(todo.n_tokens), "tokens_to_embed_total": int(todo.n_tokens.sum())}
    _atomic_json(out / "plan.json", report)
    return report


def gpu_busy() -> bool:
    """True when nvidia-smi lists a compute process other than this one."""
    try:
        res = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True
    pids = {int(x) for x in res.stdout.split() if x.strip().isdigit()}
    return bool(pids - {os.getpid()})


def embed(out: Path, device: str, part_size: int, wait_gpu: bool, poll_s: int = 120) -> dict[str, Any]:
    log = _log_to(out / "embed.log")
    todo = pd.read_parquet(out / "embed_todo.parquet")
    edir = out / "embed"
    edir.mkdir(exist_ok=True)
    if device == "cuda" and wait_gpu:
        idle = 0
        while idle < 2:
            idle = 0 if gpu_busy() else idle + 1
            if idle < 2:
                log(f"waiting for the GPU (other compute processes), poll {poll_s}s")
                time.sleep(poll_s)
    from r2ai.index.bge_m3 import M3Encoder
    from r2ai.s32.chunks import embed_texts

    encoder = M3Encoder(device=device, fp16=device == "cuda", max_len=512)
    started = time.monotonic()
    parts = range(0, len(todo), part_size)
    for k, begin in enumerate(parts):
        target = edir / f"part-{k:05d}.npz"
        if target.exists():
            continue
        chunk = todo.iloc[begin:begin + part_size]
        dense = embed_texts(encoder, chunk.text.tolist(), log, batch=EMBED_BATCH)
        temp = target.with_name(target.name + ".tmp")
        with temp.open("wb") as fh:
            np.savez(fh, text_sha=chunk.text_sha.to_numpy().astype("U32"), dense=dense)
        temp.replace(target)
        done = min(begin + part_size, len(todo))
        log(f"part {k} done texts={done}/{len(todo)} elapsed_s={time.monotonic() - started:.0f}")
    index = pd.DataFrame({"text_sha": np.concatenate([np.load(p)["text_sha"] for p in sorted(edir.glob("part-*.npz"))])})
    index["row"] = np.arange(len(index))
    if len(index) != len(todo) or set(index.text_sha) != set(todo.text_sha):
        raise ValueError("embedded parts do not cover embed_todo")
    _atomic_parquet(index, edir / "index.parquet")
    report = {"texts": len(todo), "device": device, "seconds": round(time.monotonic() - started, 1)}
    _atomic_json(out / "embed.json", report)
    return report


# --------------------------------------------------------------------------------------- select

def judge_scores(carriers: pd.DataFrame) -> pd.DataFrame:
    """S32 chunk-level judge on answer/body chunks of the carriers: query_id, doc_id, chunk_id, judge."""
    from r2ai.s38.sources import _sha256

    spath = paths.OUT_DIR / S32_SCORES
    if _sha256(spath) != S32_SCORES_SHA256:
        raise ValueError(f"{spath}: SHA256 differs from the S32 reduce report")
    cp = pd.read_parquet(paths.OUT_DIR / S32_ROOT / "pool" / "chunk_pairs.parquet",
                         columns=["pair_id", "query_id", "doc_id", "chunk_id", "field", "origin"])
    cp = cp.merge(carriers[["query_id", "doc_id"]], on=["query_id", "doc_id"])
    sc = pd.read_parquet(spath, columns=["pair_id", "query_id", "doc_id", "score"])
    cp = cp.merge(sc, on="pair_id", suffixes=("", "_s"))
    if not ((cp.query_id == cp.query_id_s) & (cp.doc_id == cp.doc_id_s)).all():
        raise ValueError("S32 scores do not align with chunk_pairs")
    cp = cp[(cp.origin == "c") & cp.field.isin(["answer", "body"])]
    return cp[["query_id", "doc_id", "chunk_id", "score"]].rename(columns={"score": "judge"})


def select(src: Path, out: Path, variants: Sequence[str]) -> dict[str, Any]:
    """Choose the pieces of ``variants``; rows of other variants already in selection.parquet are kept."""
    log = _log_to(out / "select.log")
    canonical = paths.DATA_DIR / "canonical-20261018"
    kinds = {"t256"} | {VARIANTS[v][0] for v in variants}        # sec falls back to the t256 choice
    pieces = pd.read_parquet(src / "pieces.parquet", columns=["doc_id", "kind", "idx", "chunk_id", "vec_row", "text"])
    pieces = pieces[pieces.kind.isin(kinds)]
    from r2ai.eval.scorer import normalize_metric

    pieces["text_sha"] = _text_sha(pieces.text.tolist())
    pieces["norm_sha"] = _text_sha([normalize_metric(t) for t in pieces.text])
    pieces = pieces.drop(columns=["text"])
    carriers = pd.read_parquet(src / "carriers.parquet")
    docs = pd.read_parquet(src / "docs.parquet", columns=["doc_id", "sec_from_t256", "spans_ok"])
    vec = piece_vectors(pieces.reset_index(drop=True), canonical, out / "embed")
    qv = load_query_vectors()
    log(f"variants={list(variants)} pieces={len(pieces)} carriers={len(carriers)} vectors ready")

    row_of = pieces.index.to_numpy()                              # position in pieces.parquet
    by_doc = {d: g for d, g in pd.Series(np.arange(len(pieces))).groupby(pieces.doc_id.to_numpy(), sort=False).groups.items()}
    rows = []
    for qid, d in zip(carriers.query_id.astype(int), carriers.doc_id.astype(int)):
        pos = by_doc.get(d)
        if pos is None:
            continue
        pos = np.asarray(pos)
        cos = vec[pos].astype(np.float32) @ qv[qid].astype(np.float32)
        rows.append(pd.DataFrame({"query_id": qid, "doc_id": d, "piece_row": row_of[pos], "cosine": cos}))
    cand = pd.concat(rows, ignore_index=True)
    cand = cand.join(pieces[["kind", "idx", "chunk_id", "text_sha", "norm_sha"]], on="piece_row")
    judge = judge_scores(carriers)
    cand = cand.merge(judge, on=["query_id", "doc_id", "chunk_id"], how="left")
    cand.loc[cand.kind != "t256", "judge"] = np.nan
    log(f"candidates={len(cand)} judged t256={int(cand.judge.notna().sum())}")

    t256 = cand[cand.kind == "t256"].copy()
    judged = t256.groupby(["query_id", "doc_id"]).judge.transform(lambda x: x.notna().any())
    t256_j = t256[judged & t256.judge.notna()].assign(score=lambda f: f.judge, src="judge")
    t256_c = t256[~judged].assign(score=lambda f: f.cosine, src="cosine")
    lists = {"t256": pd.concat([t256_j, t256_c], ignore_index=True)}
    lists["t128"] = cand[cand.kind == "t128"].assign(score=lambda f: f.cosine, src="cosine")
    fallback = set(docs.doc_id[docs.sec_from_t256])
    lists["sec"] = cand[(cand.kind == "sec") & ~cand.doc_id.isin(fallback)].assign(score=lambda f: f.cosine, src="cosine")
    chosen = []
    for name in variants:
        kind, n = VARIANTS[name]
        pick = select_variant(lists[kind], n)
        if kind == "sec":   # docs without structure: the t256 choice
            pick = pd.concat([pick, select_variant(lists["t256"][lists["t256"].doc_id.isin(fallback)], 1)], ignore_index=True)
        pick["variant"] = name
        chosen.append(pick)
        log(f"{name}: chunks={len(pick)} src={pick.src.value_counts().to_dict()}")
    cols = ["variant", "query_id", "doc_id", "rank", "kind", "idx", "piece_row", "text_sha", "score", "src", "cosine", "judge"]
    sel = pd.concat(chosen, ignore_index=True)[cols]
    target = out / "selection.parquet"
    if target.exists():
        old = pd.read_parquet(target)
        sel = pd.concat([old[~old.variant.isin(variants)], sel], ignore_index=True)
    sel = sel.sort_values(["variant", "query_id", "doc_id", "rank"]).reset_index(drop=True)
    _atomic_parquet(sel, target)
    report = json.loads((out / "select.json").read_text(encoding="utf-8")) if (out / "select.json").exists() else {}
    report.update({"carriers": len(carriers),
                   "carriers_judged_t256": int(t256_j.groupby(["query_id", "doc_id"]).ngroups),
                   "carriers_cosine_t256": int(t256_c.groupby(["query_id", "doc_id"]).ngroups)})
    report.setdefault("variants", {})
    first = sel[sel.variant == "CH-t256-1"].set_index(["query_id", "doc_id"]).text_sha
    for name in variants:
        pick = sel[sel.variant == name]
        entry = {"chunks": len(pick), "carriers_with_chunk": int(pick.groupby(["query_id", "doc_id"]).ngroups),
                 "second_chunks": int((pick["rank"] == 2).sum()), "src": pick.src.value_counts().to_dict()}
        if name in ("CH-sec-1", "CH-t128-1") and len(first):
            p1 = pick.set_index(["query_id", "doc_id"]).text_sha
            entry["same_text_as_t256_1"] = float((p1.to_numpy() == first.reindex(p1.index).to_numpy()).mean())
        report["variants"][name] = entry
    _atomic_json(out / "select.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "embed", "select"])
    parser.add_argument("--sources", type=Path, default=None, help="default RUNS_DIR/s38-chunk-metric-v2/sources")
    parser.add_argument("--out", type=Path, default=None, help="default RUNS_DIR/s38-chunk-metric-v2/scores")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    parser.add_argument("--wait-gpu", action="store_true", help="embed: wait until no other GPU compute process")
    parser.add_argument("--kinds", default="sec", help="plan: piece kinds to embed, comma list (default sec)")
    parser.add_argument("--variant", action="append", choices=list(VARIANTS), default=None, help="select: variants")
    args = parser.parse_args(argv)
    src = paths.resolve_path(args.sources or run_root() / "sources")
    out = paths.resolve_path(args.out or scores_dir())
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    for name in ("", "embed_todo.parquet", "plan.json", "embed", "embed.log", "embed.json", "selection.parquet",
                 "select.json", "select.log"):
        paths.assert_writable(out / name if name else out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        report = plan(src, out, args.kinds.split(","))
    elif args.command == "embed":
        report = embed(out, args.device, args.part_size, args.wait_gpu)
    else:
        report = select(src, out, args.variant or list(VARIANTS))
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
