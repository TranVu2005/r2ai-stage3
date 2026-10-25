"""S32 pool: for every S31 pool13 (query, doc) pair keep the top-n cosine chunks as judge inputs.

``python -m r2ai.s32.chunks prepare`` (CPU + a few minutes of GPU for the BGE-M3 checks):

* pairs = the S31 pool13 (281,924 pairs, untouched) plus the pairs that the Pharmacity lot 3
  extension (``pharmacity-ext/pool13_index.parquet``, ``is_new``) adds for the ph979 builds;
* chunks = ``chunks_t256`` of ``data/canonical-20261018`` verbatim (every field, aliases resolved
  through the S29 owner map) and the Pharmacity lots' own ``chunks_t256``;
* chunk vectors = the stored BGE-M3 dense vectors (``vectors/dense.npy``). Chunks that never had a
  vector (VOV s24/s25 chunks, Pharmacity lots) are embedded with the same ``M3Encoder``; the same
  session re-embeds 1,000 stored chunks and compares them with the stored vectors;
* per pair the top ``--top-n`` chunks by cosine(S29 query vector, chunk vector), ties by file order;
  pairs whose document has no chunk keep ``doc_text`` (listed in ``doc_fallback.parquet``).

The shards are ``judge_score.py`` input: ``doc_text`` holds the chunk text and ``in_pool13`` means
``chunk_rank <= 2`` here, so ``--subset pool13`` scores the rank-1/2 chunks, ``--subset tail`` the
rank-3 chunks and ``--subset all`` everything.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import (
    DOC_TOKEN_LIMIT,
    INSTRUCTIONS,
    PREFIX,
    SUFFIX,
    _atomic_json,
    _inputs,
    _model_path,
    _sha256,
    truncate_document,
)
from r2ai.s31.pool import _shard_groups

S32_NAME = "s32-chunk-judge"
S31_NAME = "s31-strong-judge"
TOP_N = 3
RANK_SPLIT = 2  # chunk_rank <= RANK_SPLIT is "in_pool13" for judge_score's --subset
VERIFY_SEED = 32
VERIFY_SAMPLES = 1_000
VERIFY_MIN_MEAN = 0.999
VERIFY_MIN_ONE = 0.99
EXT_SHARD = "pool-90000.parquet"
NEW_PAIR_ID_BASE = 10_000_000
DEFAULT_SHARD_PAIRS = 20_000
EMBED_BATCH = 32
DIM = 1024
PHARMACITY_UID_BASE = 1_000_000_000_000_000


def run_root() -> Path:
    return paths.run_dir(S32_NAME)


# ----------------------------------------------------------------------------- selection

def segment_top(scores: np.ndarray, lens: np.ndarray, top_n: int) -> list[np.ndarray]:
    """Per segment of ``scores`` (consecutive, ``lens`` long) the offsets of the best ``top_n``.

    Descending score; exact ties keep the lower offset (file order of the chunks).
    """
    out: list[np.ndarray] = []
    begin = 0
    for n in lens.tolist():
        seg = scores[begin : begin + n]
        out.append(np.argsort(-seg, kind="stable")[:top_n])
        begin += n
    return out


def gather_index(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(starts[i], starts[i] + lens[i])`` without a Python loop."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.cumsum(lens) - lens
    return np.repeat(starts - offsets, lens) + np.arange(total, dtype=np.int64)


def select_chunks(
    pairs: pd.DataFrame,
    doc_start: np.ndarray,
    doc_count: np.ndarray,
    vectors: np.ndarray,
    query_vectors: dict[int, np.ndarray],
    top_n: int,
) -> pd.DataFrame:
    """Top-n chunk slots per pair. ``pairs`` carries ``slot_doc`` (row of the doc table or -1).

    Returns one row per kept chunk: pair position, chunk_rank (1-based), slot, cosine, n_chunks.
    """
    positions, ranks, slots, cosines, counts = [], [], [], [], []
    for qid, group in pairs.groupby("query_id", sort=True):
        group = group[group.slot_doc.to_numpy() >= 0]
        if group.empty:
            continue
        q32 = query_vectors[int(qid)].astype(np.float32)
        di = group.slot_doc.to_numpy()
        lens = doc_count[di]
        idx = gather_index(doc_start[di], lens)
        scores = vectors[idx].astype(np.float32) @ q32
        offsets = np.cumsum(lens) - lens
        for pos, off, n, take in zip(group.index.to_numpy(), offsets, lens, segment_top(scores, lens, top_n)):
            for rank, k in enumerate(take.tolist(), 1):
                positions.append(pos)
                ranks.append(rank)
                slots.append(int(idx[off + k]))
                cosines.append(float(scores[off + k]))
                counts.append(int(n))
    return pd.DataFrame({"pair_pos": positions, "chunk_rank": ranks, "slot": slots, "chunk_cosine": cosines,
                         "n_chunks": counts})


# --------------------------------------------------------------------------------- loaders

def load_pairs(pool_dir: Path, ext_dir: Path) -> pd.DataFrame:
    """S31 pool13 pairs + the Pharmacity-lot-3 pairs (``source`` tells them apart)."""
    index = pd.read_parquet(pool_dir / "pool_index.parquet")
    main = index[index.in_pool13][["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang"]].copy()
    main["source"] = "pool13"
    ext = pd.read_parquet(ext_dir / "pool13_index.parquet")
    new = ext[ext.is_new][["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score"]].copy()
    new["lang"] = "vi"
    new["source"] = "ph979_new"
    pairs = pd.concat([main, new], ignore_index=True).rename(columns={"pair_id": "src_pair_id"})
    if pairs.src_pair_id.duplicated().any() or pairs.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Duplicate pairs in the S32 pair list")
    for col in ("src_pair_id", "query_id", "doc_id", "cosine_rank"):
        pairs[col] = pairs[col].astype(np.int64)
    return pairs.sort_values(["query_id", "cosine_rank", "src_pair_id"], kind="stable").reset_index(drop=True)


def load_canonical_meta(canonical: Path) -> pd.DataFrame:
    meta = pq.read_table(
        canonical / "chunks_t256.parquet", columns=["chunk_id", "doc_id", "field", "n_tokens", "has_vector"]
    ).to_pandas()
    has = meta.has_vector.to_numpy()
    meta["vec_row"] = np.where(has, np.cumsum(has) - 1, -1).astype(np.int64)
    meta["file_row"] = np.arange(len(meta), dtype=np.int64)
    return meta


def load_pharmacity_chunks(inputs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for lot in sorted(p for p in (inputs["extra_docs"] / "lots").iterdir() if p.is_dir()):
        frame = pd.read_parquet(lot / "chunks" / "chunks_t256.parquet",
                                columns=["chunk_id", "doc_id", "field", "n_tokens", "text"])
        # chunk_id restarts at 0 in every lot: a unique id above every canonical chunk_id (< 1e15)
        frame["chunk_id"] = PHARMACITY_UID_BASE + int(lot.name.rsplit("-", 1)[1]) * 1_000_000 + frame.chunk_id.astype(np.int64)
        frame["lot"] = lot.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["chunk_id"] = out.chunk_id.astype(np.int64)
    out["doc_id"] = out.doc_id.astype(np.int64)
    if out.chunk_id.duplicated().any():
        raise ValueError("Duplicate chunk uid inside the Pharmacity lots")
    return out


def fetch_canonical_texts(canonical: Path, chunk_ids: np.ndarray, log: Callable[[str], None]) -> dict[int, str]:
    """``chunk_id -> text`` for the wanted chunks of the canonical chunk table (one pass)."""
    if len(chunk_ids) == 0:
        return {}
    wanted = pa.array(np.unique(chunk_ids.astype(np.int64)))
    out: dict[int, str] = {}
    started = time.monotonic()
    parquet = pq.ParquetFile(canonical / "chunks_t256.parquet")
    for batch in parquet.iter_batches(batch_size=200_000, columns=["chunk_id", "text"]):
        mask = pc.is_in(batch.column("chunk_id"), value_set=wanted)
        if pc.any(mask).as_py():
            hit = batch.filter(mask)
            out.update(zip(hit.column("chunk_id").to_pylist(), hit.column("text").to_pylist()))
    if len(out) != len(wanted):
        raise ValueError(f"Fetched {len(out)} of {len(wanted)} canonical chunk texts")
    log(f"fetched {len(out)} canonical chunk texts in {time.monotonic() - started:.0f}s")
    return out


def load_query_vectors(manifest: dict[str, Any]) -> dict[int, np.ndarray]:
    path = paths.RUNS_DIR / "s29-qwen-judge" / "query_vectors.npz"
    if _sha256(path) != manifest["s29_query_vectors_sha256"]:
        raise ValueError("S29 query vectors differ from the ones the S31 pool was built with")
    with np.load(path) as qv:
        ids = qv["q_ids"].astype(np.int64)
        dense = np.array(qv["q_dense"], copy=True)
    return {int(q): dense[i] for i, q in enumerate(ids)}


# --------------------------------------------------------------------------------- BGE-M3

def embed_texts(encoder: Any, texts: Sequence[str], log: Callable[[str], None], batch: int = EMBED_BATCH) -> np.ndarray:
    """fp16 L2-normalised CLS vectors in input order, length-sorted batches."""
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), DIM), dtype=np.float16)
    started = time.monotonic()
    for begin in range(0, len(texts), batch):
        take = order[begin : begin + batch]
        dense, _ = encoder.encode_batch([texts[i] for i in take])
        out[take] = dense
        if (begin // batch) % 50 == 0:
            log(f"embedded {min(begin + batch, len(texts))}/{len(texts)} elapsed_s={time.monotonic() - started:.0f}")
    return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32) * b.astype(np.float32)).sum(axis=1)


def summarize_cosines(cos: np.ndarray) -> dict[str, float]:
    return {"n": int(len(cos)), "min": float(cos.min()), "p01": float(np.quantile(cos, 0.01)),
            "median": float(np.median(cos)), "mean": float(cos.mean()), "max": float(cos.max()),
            "below_0.999": int((cos < VERIFY_MIN_MEAN).sum())}


# ---------------------------------------------------------------------------------- prepare

def _guard(out: Path) -> None:
    targets = [out, out / "shards", out / "manifest.json", out / "manifest.json.tmp", out / "prepare.log",
               out / "chunk_pairs.parquet", out / "chunk_pairs.parquet.tmp", out / "doc_fallback.parquet",
               out / "doc_fallback.parquet.tmp", out / "supplement_vectors.npz", out / "supplement_vectors.npz.tmp",
               out / "vector_verify.json", out / "vector_verify.json.tmp"]
    for index in range(60):
        shard = out / "shards" / f"pool-{index:05d}.parquet"
        targets += [shard, shard.with_name(shard.name + ".tmp")]
    targets += [out / "shards" / EXT_SHARD, out / "shards" / (EXT_SHARD + ".tmp")]
    for target in targets:
        paths.assert_writable(target)


def prepare(out: Path, top_n: int = TOP_N, shard_pairs: int = DEFAULT_SHARD_PAIRS, verify_only: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    inputs = _inputs()
    s31_root = paths.RUNS_DIR / S31_NAME
    pool_dir, ext_dir = s31_root / "pool", s31_root / "pharmacity-ext"
    canonical = inputs["canonical"]
    vec_dir = canonical / "vectors"
    for path in (pool_dir / "pool_index.parquet", pool_dir / "manifest.json", ext_dir / "pool13_index.parquet",
                 canonical / "chunks_t256.parquet", vec_dir / "dense.npy", inputs["query"], inputs["extra_docs"]):
        paths.require_inputs(path)
    _guard(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    if sorted((out / "shards").glob("pool-*.parquet")):
        raise FileExistsError(f"{out / 'shards'} already holds shards; use a new --out")

    def log(message: str) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                           "message": message}, ensure_ascii=False)
        print(line, flush=True)
        with (out / "prepare.log").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    manifest31 = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    qvec = load_query_vectors(manifest31)
    pairs = load_pairs(pool_dir, ext_dir)
    log(f"pairs total={len(pairs)} pool13={int((pairs.source == 'pool13').sum())} ph979_new={int((pairs.source != 'pool13').sum())}")
    if int((pairs.source == "pool13").sum()) != manifest31["pool13_pairs"]:
        raise ValueError("pool13 pair count differs from the S31 manifest")
    if set(pairs.query_id) - set(qvec):
        raise ValueError("Pairs reference queries without an S29 query vector")

    # ---- owners and chunk slots
    groups, cid_to_doc = s29_build._load_aliases(canonical / "chunks_t256.parquet", canonical / "docs.parquet")
    ph_chunks_all = load_pharmacity_chunks(inputs)
    ph_docs = set(ph_chunks_all.doc_id.astype(int))
    lot_docs = set()
    for lot in sorted(p for p in (inputs["extra_docs"] / "lots").iterdir() if p.is_dir()):
        lot_docs |= set(pd.read_parquet(lot / "chunks" / "docs.parquet", columns=["doc_id"]).doc_id.astype(int))
    if set(groups) & lot_docs:
        raise ValueError("A Pharmacity lot doc_id is also a canonical chunk owner")
    pair_docs = sorted(set(pairs.doc_id.astype(int)))
    owner_of: dict[int, int] = {}
    origin_of: dict[int, str] = {}
    for doc in pair_docs:
        if doc in ph_docs:
            owner_of[doc], origin_of[doc] = doc, "p"
        elif doc in cid_to_doc and cid_to_doc[doc] in groups:
            owner_of[doc], origin_of[doc] = int(cid_to_doc[doc]), "c"
    owners_c = np.array(sorted({o for d, o in owner_of.items() if origin_of[d] == "c"}), dtype=np.int64)
    log(f"pair docs={len(pair_docs)} with_chunks={len(owner_of)} (canonical owners={len(owners_c)}, pharmacity={sum(1 for v in origin_of.values() if v == 'p')}) no_chunk={len(pair_docs) - len(owner_of)}")

    meta = load_canonical_meta(canonical)
    sub = meta[meta.doc_id.isin(owners_c)]
    slots_c = pd.DataFrame({"origin": "c", "doc_id": sub.doc_id.to_numpy(), "chunk_id": sub.chunk_id.to_numpy(),
                            "field": sub.field.to_numpy(), "n_tokens": sub.n_tokens.to_numpy(),
                            "vec_row": sub.vec_row.to_numpy(), "file_row": sub.file_row.to_numpy()})
    ph_need = ph_chunks_all[ph_chunks_all.doc_id.isin([d for d, v in origin_of.items() if v == "p"])]
    slots_p = pd.DataFrame({"origin": "p", "doc_id": ph_need.doc_id.to_numpy(), "chunk_id": ph_need.chunk_id.to_numpy(),
                            "field": ph_need.field.to_numpy(), "n_tokens": ph_need.n_tokens.to_numpy(),
                            "vec_row": -1, "file_row": np.arange(len(ph_need), dtype=np.int64)})
    slots = pd.concat([slots_c, slots_p], ignore_index=True).sort_values("doc_id", kind="stable").reset_index(drop=True)
    del sub, slots_c, slots_p
    doc_ids_u, doc_start, doc_count = np.unique(slots.doc_id.to_numpy(), return_index=True, return_counts=True)
    log(f"slots={len(slots)} docs={len(doc_ids_u)} canonical_without_vector={int(((slots.origin == 'c') & (slots.vec_row < 0)).sum())} pharmacity={int((slots.origin == 'p').sum())}")

    # ---- vectors: stored, then supplement; verification sample
    dense = np.load(vec_dir / "dense.npy", mmap_mode="r")
    if dense.shape != (int(meta.has_vector.sum()), DIM) or dense.dtype != np.float16:
        raise ValueError(f"Unexpected canonical dense vectors {dense.shape} {dense.dtype}")
    vectors = np.zeros((len(slots), DIM), dtype=np.float16)
    has = (slots.vec_row >= 0).to_numpy()
    where, rows = np.flatnonzero(has), slots.vec_row.to_numpy()[has]
    order = np.argsort(rows, kind="stable")
    for begin in range(0, len(order), 200_000):
        take = order[begin : begin + 200_000]
        vectors[where[take]] = dense[rows[take]]
    log(f"stored vectors loaded rows={int(has.sum())}")
    del meta

    rng = np.random.default_rng(VERIFY_SEED)
    verify_slots = np.sort(rng.choice(np.flatnonzero(has), size=min(VERIFY_SAMPLES, int(has.sum())), replace=False))
    missing_slots = np.flatnonzero(~has)
    need_c = np.concatenate([slots.chunk_id.to_numpy()[verify_slots],
                             slots.chunk_id.to_numpy()[missing_slots][slots.origin.to_numpy()[missing_slots] == "c"]])
    texts_c = fetch_canonical_texts(canonical, need_c, log)
    ph_text = dict(zip(ph_chunks_all.chunk_id.astype(int), ph_chunks_all.text))
    sup_path = out / "supplement_vectors.npz"

    def text_of(slot: int) -> str:
        cid = int(slots.chunk_id.iat[slot])
        return ph_text[cid] if slots.origin.iat[slot] == "p" else texts_c[cid]

    from r2ai.index.bge_m3 import M3Encoder

    encoder = M3Encoder(max_len=512)
    verify_texts = [text_of(int(s)) for s in verify_slots]
    fresh = embed_texts(encoder, verify_texts, log)
    cos = cosine_rows(fresh, vectors[verify_slots])
    verify = {**summarize_cosines(cos), "seed": VERIFY_SEED, "encoder": "r2ai.index.bge_m3.M3Encoder(max_len=512, fp16)",
              "fields": pd.Series(slots.field.to_numpy()[verify_slots]).value_counts().to_dict(),
              "gate": f"mean >= {VERIFY_MIN_MEAN} and min >= {VERIFY_MIN_ONE}"}
    verify["pass"] = bool(verify["mean"] >= VERIFY_MIN_MEAN and verify["min"] >= VERIFY_MIN_ONE)
    _atomic_json(out / "vector_verify.json", verify)
    log(f"vector verify {json.dumps(verify, ensure_ascii=False)}")
    if not verify["pass"]:
        raise ValueError("Stored chunk vectors do not match their chunk text; see vector_verify.json")
    if len(missing_slots):
        sup = embed_texts(encoder, [text_of(int(s)) for s in missing_slots], log)
        vectors[missing_slots] = sup
        temp = sup_path.with_name(sup_path.name + ".tmp")
        with temp.open("wb") as fh:
            np.savez(fh, origin=slots.origin.to_numpy()[missing_slots].astype("U1"),
                     chunk_id=slots.chunk_id.to_numpy()[missing_slots], dense=sup)
        temp.replace(sup_path)
        log(f"supplement vectors embedded={len(missing_slots)}")
    del encoder
    try:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    if verify_only:
        return verify

    # ---- selection
    pairs["owner"] = pairs.doc_id.map(owner_of)
    pairs["slot_doc"] = np.where(pairs.owner.notna(), np.searchsorted(doc_ids_u, pairs.owner.fillna(-1).astype(np.int64)), -1)
    pairs.loc[pairs.owner.isna(), "slot_doc"] = -1
    selection = select_chunks(pairs, doc_start, doc_count, vectors, qvec, top_n)
    log(f"selected chunk rows={len(selection)}")
    sel = selection.merge(pairs[["src_pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang", "source"]],
                          left_on="pair_pos", right_index=True)
    sel["chunk_id"] = slots.chunk_id.to_numpy()[sel.slot]
    sel["field"] = slots.field.to_numpy()[sel.slot]
    sel["origin"] = slots.origin.to_numpy()[sel.slot]
    sel["chunk_n_tokens_bge"] = slots.n_tokens.to_numpy()[sel.slot]
    sel = sel.drop(columns=["pair_pos"])
    covered = set(sel.src_pair_id)
    fallback = pairs[~pairs.src_pair_id.isin(covered)][["src_pair_id", "query_id", "doc_id", "cosine_rank", "source"]].copy()
    fallback["reason"] = "no_chunk"
    log(f"pairs with chunks={len(covered)} fallback={len(fallback)}")

    # ---- chunk text, Qwen tokens, prompt length
    need_final = sel.loc[sel.origin == "c", "chunk_id"].to_numpy()
    texts_c.update(fetch_canonical_texts(canonical, np.array([c for c in np.unique(need_final) if c not in texts_c], dtype=np.int64), log))
    text_by_key = {("c", int(c)): texts_c[int(c)] for c in np.unique(need_final)}
    text_by_key.update({("p", int(c)): ph_text[int(c)] for c in sel.loc[sel.origin == "p", "chunk_id"].unique()})

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    keys = list(text_by_key)
    counts = {}
    for begin in range(0, len(keys), 20_000):
        part = keys[begin : begin + 20_000]
        enc = tokenizer([text_by_key[k] for k in part], add_special_tokens=False)["input_ids"]
        counts.update({k: len(ids) for k, ids in zip(part, enc)})
    truncated = 0
    for key in keys:
        if counts[key] > DOC_TOKEN_LIMIT:
            text_by_key[key], counts[key] = truncate_document(tokenizer, text_by_key[key])
            truncated += 1
    log(f"chunk texts={len(keys)} over_{DOC_TOKEN_LIMIT}_tokens_truncated={truncated}")
    sel["doc_text"] = [text_by_key[(o, int(c))] for o, c in zip(sel.origin, sel.chunk_id)]
    sel["doc_tokens"] = np.array([counts[(o, int(c))] for o, c in zip(sel.origin, sel.chunk_id)], dtype=np.int32)

    qtable = pq.read_table(inputs["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    sel["query_text"] = sel.query_id.map(queries)
    prefix_len = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    suffix_len = len(tokenizer.encode(SUFFIX, add_special_tokens=False))
    bodies = [f"<Instruct>: {INSTRUCTIONS[lang]}\n<Query>: {q}\n<Document>: {d}"
              for lang, q, d in zip(sel.lang, sel.query_text, sel.doc_text)]
    lens = []
    for begin in range(0, len(bodies), 20_000):
        lens.extend(len(ids) for ids in tokenizer(bodies[begin : begin + 20_000], add_special_tokens=False)["input_ids"])
    sel["prompt_tokens"] = np.array(lens, dtype=np.int32) + prefix_len + suffix_len
    del bodies

    # ---- shards
    sel["in_pool13"] = sel.chunk_rank <= RANK_SPLIT
    sel = sel.sort_values(["query_id", "cosine_rank", "chunk_rank"], kind="stable").reset_index(drop=True)
    main_mask = (sel.source == "pool13").to_numpy()
    sel["pair_id"] = np.int64(0)
    sel.loc[main_mask, "pair_id"] = np.arange(int(main_mask.sum()), dtype=np.int64)
    sel.loc[~main_mask, "pair_id"] = NEW_PAIR_ID_BASE + np.arange(int((~main_mask).sum()), dtype=np.int64)
    columns = ["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "in_pool13", "lang", "query_text",
               "doc_text", "doc_tokens", "chunk_id", "chunk_rank", "chunk_cosine", "field", "src_pair_id",
               "prompt_tokens", "source"]
    types = {"pair_id": pa.int64(), "query_id": pa.int64(), "doc_id": pa.int64(), "cosine_rank": pa.int32(),
             "cosine_score": pa.float64(), "in_pool13": pa.bool_(), "lang": pa.string(), "query_text": pa.string(),
             "doc_text": pa.string(), "doc_tokens": pa.int32(), "chunk_id": pa.int64(), "chunk_rank": pa.int32(),
             "chunk_cosine": pa.float64(), "field": pa.string(), "src_pair_id": pa.int64(),
             "prompt_tokens": pa.int32(), "source": pa.string()}

    def write_shard(frame: pd.DataFrame, name: str) -> dict[str, Any]:
        table = pa.table({c: pa.array(frame[c].to_numpy() if types[c] not in (pa.string(),) else frame[c].tolist(),
                                      types[c]) for c in columns})
        target = out / "shards" / name
        temp = target.with_name(target.name + ".tmp")
        pq.write_table(table, temp, compression="zstd", row_group_size=5_000)
        temp.replace(target)
        return {"file": name, "pairs": len(frame), "bytes": target.stat().st_size, "sha256": _sha256(target)}

    main = sel[main_mask]
    per_query = main.groupby("query_id").size().to_dict()
    shard_reports = []
    for shard, group in enumerate(_shard_groups(per_query, shard_pairs)):
        shard_reports.append(write_shard(main[main.query_id.isin(group)], f"pool-{shard:05d}.parquet"))
    ext = sel[~main_mask]
    if len(ext):
        shard_reports.append(write_shard(ext, EXT_SHARD))
    index_cols = [c for c in columns if c not in ("query_text", "doc_text")] + ["n_chunks", "chunk_n_tokens_bge", "origin"]
    index_path = out / "chunk_pairs.parquet"
    temp = index_path.with_name(index_path.name + ".tmp")
    sel[index_cols].to_parquet(temp, index=False)
    temp.replace(index_path)
    fb_path = out / "doc_fallback.parquet"
    temp = fb_path.with_name(fb_path.name + ".tmp")
    fallback.to_parquet(temp, index=False)
    temp.replace(fb_path)

    per_pair = main.drop_duplicates("src_pair_id")
    manifest = {
        "status": "exported",
        "top_n": top_n,
        "rank_split": f"in_pool13 := chunk_rank <= {RANK_SPLIT}",
        "pairs_pool13": int((pairs.source == "pool13").sum()),
        "pairs_pool13_with_chunks": int(per_pair.shape[0]),
        "pairs_pool13_fallback_doc_text": int(((fallback.source == "pool13")).sum()),
        "pairs_ph979_new": int((pairs.source != "pool13").sum()),
        "pairs_ph979_new_with_chunks": int(sel[~main_mask].src_pair_id.nunique()),
        "pairs_ph979_new_fallback_doc_text": int((fallback.source != "pool13").sum()),
        "query_chunk_pairs_main": int(main_mask.sum()),
        "query_chunk_pairs_main_rank_le_2": int((main.chunk_rank <= 2).sum()),
        "query_chunk_pairs_ph979_new": int((~main_mask).sum()),
        "chunks_selected_per_pair": {str(k): int(v) for k, v in main.groupby("src_pair_id").size().value_counts().sort_index().items()},
        "n_chunks_per_doc_quantiles": {f"p{int(q * 100)}": float(per_pair.n_chunks.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "n_chunks_per_doc_mean": float(per_pair.n_chunks.mean()),
        "n_chunks_per_doc_le_top_n": int((per_pair.n_chunks <= top_n).sum()),
        "prompt_tokens_mean": float(main.prompt_tokens.mean()),
        "prompt_tokens_mean_rank_le_2": float(main.loc[main.chunk_rank <= 2, "prompt_tokens"].mean()),
        "prompt_tokens_max": int(main.prompt_tokens.max()),
        "prompt_tokens_doc_text_s31_reference": 561,
        "chunk_tokens_qwen_mean": float(main.doc_tokens.mean()),
        "chunks_truncated_to_512": truncated,
        "selected_field_share": {k: float(v) for k, v in main.field.value_counts(normalize=True).items()},
        "selected_origin": sel.origin.value_counts().to_dict(),
        "unique_chunks_main": int(main.drop_duplicates(["origin", "chunk_id"]).shape[0]),
        "supplement_chunks_embedded": int(len(missing_slots)),
        "supplement_docs": int(slots.iloc[missing_slots].doc_id.nunique()) if len(missing_slots) else 0,
        "vector_verify": verify,
        "fallback_reasons": fallback.reason.value_counts().to_dict(),
        "shards": shard_reports,
        "chunk_pairs": str(index_path),
        "chunk_pairs_sha256": _sha256(index_path),
        "doc_fallback_sha256": _sha256(fb_path),
        "s31_pool_index_sha256": _sha256(pool_dir / "pool_index.parquet"),
        "s31_ext_pool13_index_sha256": _sha256(ext_dir / "pool13_index.parquet"),
        "s29_query_vectors_sha256": manifest31["s29_query_vectors_sha256"],
        "canonical_chunks_sha256": _sha256(canonical / "chunks_t256.parquet"),
        "canonical_dense_bytes": (vec_dir / "dense.npy").stat().st_size,
        "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", manifest)
    log(f"done main_pairs={manifest['query_chunk_pairs_main']} seconds={manifest['seconds']}")
    return manifest


def audit(pool: Path, pairs_n: int = 300, seed: int = VERIFY_SEED) -> dict[str, Any]:
    """Re-derive the chunk choice of sampled pairs with a different code path.

    The alias test is "the doc_id is in the chunk's ``doc_ids`` list" (``prepare`` goes through the
    S29 owner map); stored vectors are read straight from ``dense.npy`` by running row count.
    Pharmacity and supplement-vector pairs are skipped. Also checks the shard text against the table.
    """
    inputs = _inputs()
    canonical = inputs["canonical"]
    index = pd.read_parquet(pool / "chunk_pairs.parquet")
    main = index[(index.source == "pool13") & (index.origin == "c")]
    picks = main.drop_duplicates("src_pair_id").sample(n=pairs_n, random_state=seed)
    targets = set(picks.doc_id.astype(int))
    manifest31 = json.loads((paths.RUNS_DIR / S31_NAME / "pool" / "manifest.json").read_text(encoding="utf-8"))
    qvec = load_query_vectors(manifest31)
    rows: dict[int, list[tuple[int, int, int, str]]] = {d: [] for d in targets}  # chunk_id, vec_row, file_row
    texts: dict[int, str] = {}
    wanted = pa.array(sorted(targets))
    vec_count = 0
    file_row = 0
    parquet = pq.ParquetFile(canonical / "chunks_t256.parquet")
    for batch in parquet.iter_batches(batch_size=200_000, columns=["chunk_id", "doc_ids", "has_vector", "text"]):
        lists = batch.column("doc_ids")
        flat, parents = pc.list_flatten(lists), pc.list_parent_indices(lists)
        hit = pc.is_in(flat, value_set=wanted)
        flat_hit = flat.filter(hit).to_numpy(zero_copy_only=False)
        parent_hit = parents.filter(hit).to_numpy(zero_copy_only=False)
        has = batch.column("has_vector").to_numpy(zero_copy_only=False)
        before = vec_count + np.cumsum(has) - has  # vector row of each chunk in this batch
        cids = batch.column("chunk_id").to_numpy(zero_copy_only=False)
        for doc, parent in zip(flat_hit.tolist(), parent_hit.tolist()):
            rows[int(doc)].append((int(cids[parent]), int(before[parent]) if has[parent] else -1, file_row + parent))
            texts[int(cids[parent])] = batch.column("text")[parent].as_py()
        vec_count += int(has.sum())
        file_row += len(batch)
    dense = np.load(canonical / "vectors" / "dense.npy", mmap_mode="r")
    mismatches, skipped, text_mismatch = [], 0, 0
    for _, pair in picks.iterrows():
        cand = sorted(rows[int(pair.doc_id)], key=lambda r: r[2])
        if not cand or any(r[1] < 0 for r in cand):
            skipped += 1
            continue
        q32 = qvec[int(pair.query_id)].astype(np.float32)
        cos = np.array([dense[r[1]].astype(np.float32) @ q32 for r in cand])
        best = np.argsort(-cos, kind="stable")[:TOP_N]
        expect = [cand[i][0] for i in best]
        got = index[index.src_pair_id == pair.src_pair_id].sort_values("chunk_rank").chunk_id.tolist()
        if expect != got:
            mismatches.append({"src_pair_id": int(pair.src_pair_id), "expect": expect, "got": got})
    shard_check = 0
    for shard in sorted((pool / "shards").glob("pool-0000[0-2].parquet")):
        frame = pd.read_parquet(shard, columns=["chunk_id", "doc_text", "source"]).head(2_000)
        for cid, text in zip(frame.chunk_id, frame.doc_text):
            shard_check += 1
            if int(cid) in texts and texts[int(cid)] != text:
                text_mismatch += 1
    report = {"pairs_sampled": len(picks), "pairs_skipped": skipped, "pairs_compared": len(picks) - skipped,
              "selection_mismatches": mismatches[:5], "selection_mismatch_count": len(mismatches),
              "shard_rows_checked": shard_check, "shard_text_mismatches": text_mismatch}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    aud = sub.add_parser("audit", help="re-derive the chunk choice of sampled pairs")
    aud.add_argument("--pool", type=Path, default=None)
    aud.add_argument("--pairs", type=int, default=300)
    prep = sub.add_parser("prepare", help="select chunks, write judge shards")
    prep.add_argument("--out", type=Path, default=None, help="output dir below RUNS_DIR (default s32-chunk-judge/pool)")
    prep.add_argument("--top-n", type=int, default=TOP_N)
    prep.add_argument("--shard-pairs", type=int, default=DEFAULT_SHARD_PAIRS)
    prep.add_argument("--verify-only", action="store_true", help="only the 1,000-chunk vector check")
    args = parser.parse_args(argv)
    if args.command == "audit":
        pool = paths.resolve_path(args.pool or run_root() / "pool")
        report = audit(pool, args.pairs)
        _atomic_json(pool / "audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    out = paths.resolve_path(args.out or run_root() / "pool")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S32 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    result = prepare(out, args.top_n, args.shard_pairs, args.verify_only)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
