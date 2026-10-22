"""S33 pools for bge-reranker-v2-m3 over the S31 cosine top-211 pool (253,200 pairs).

``python -m r2ai.s33.pool rdoc``: one row per pool211 pair, ``doc_text`` = the S31 doc text (512
Qwen3 tokens, ``truncate_document`` of S29), taken verbatim from the S31 pool shards, the pool-extra
shard ``pool-91000`` and the Pharmacity lot 3 shard. ``pair_id`` = the pool211 pair_id.

``python -m r2ai.s33.pool rchk``: the top-3 t256 chunks of every pool211 pair, chosen exactly as S32
(cosine of the S29 query vector and the BGE-M3 chunk vector, ties by file order). Pairs already in
the S32 pool keep their S32 rows (and S32 ``pair_id``); the pool-extra pairs (not in pool13) are
selected here with the S32 helpers, new ``pair_id`` from ``NEW_PAIR_ID_BASE``. Chunks without a stored
vector take the S32 supplement vector, or are embedded with the same ``M3Encoder``. A gate re-selects
``GATE_PAIRS`` S32 pairs through this code path and requires the same chunk ids in the same order.
Pairs whose doc has no chunk are listed in ``doc_fallback.parquet`` (R-chk falls back to R-doc).

Both pools write ``in_pool13 = True`` on every row so ``judge_score``'s pilot/collect read them.
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
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29 import build as s29_build
from r2ai.s29.qwen_judge import DOC_TOKEN_LIMIT, _atomic_json, _inputs, _model_path, _sha256, truncate_document
from r2ai.s31.pool import _shard_groups
from r2ai.s32 import chunks as s32

S33_NAME = "s33-btc-baseline"
TOP_N = s32.TOP_N
NEW_PAIR_ID_BASE = 30_000_000
SHARD_PAIRS = 20_000
GATE_PAIRS = 2_000
GATE_SEED = 33

RDOC_COLUMNS = {"pair_id": pa.int64(), "query_id": pa.int64(), "doc_id": pa.int64(), "cosine_rank": pa.int32(),
                "cosine_score": pa.float64(), "in_pool13": pa.bool_(), "lang": pa.string(), "query_text": pa.string(),
                "doc_text": pa.string(), "doc_tokens": pa.int32(), "score_origin": pa.string()}
RCHK_COLUMNS = {"pair_id": pa.int64(), "query_id": pa.int64(), "doc_id": pa.int64(), "cosine_rank": pa.int32(),
                "cosine_score": pa.float64(), "in_pool13": pa.bool_(), "lang": pa.string(), "query_text": pa.string(),
                "doc_text": pa.string(), "doc_tokens": pa.int32(), "chunk_id": pa.int64(), "chunk_rank": pa.int32(),
                "chunk_cosine": pa.float64(), "field": pa.string(), "src_pair_id": pa.int64(), "origin": pa.string(),
                "chunk_source": pa.string()}


def run_root() -> Path:
    return paths.run_dir(S33_NAME)


def s31_root() -> Path:
    return paths.RUNS_DIR / s32.S31_NAME


def s32_pool() -> Path:
    return paths.RUNS_DIR / s32.S32_NAME / "pool"


def _logger(path: Path) -> Callable[[str], None]:
    def log(message: str) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                           "message": message}, ensure_ascii=False)
        print(line, flush=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    return log


def _prepare_out(out: Path) -> None:
    paths.assert_writable(out)
    for name in ("shards", "manifest.json", "prepare.log", "doc_fallback.parquet", "chunk_pairs.parquet",
                 "gate.json", "supplement_vectors.npz"):
        paths.assert_writable(out / name)
        paths.assert_writable(out / (name + ".tmp"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    if sorted((out / "shards").glob("pool-*.parquet")):
        raise FileExistsError(f"{out / 'shards'} already holds shards; use a new --out")


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def write_shards(out: Path, frame: pd.DataFrame, types: dict[str, pa.DataType], log: Callable[[str], None]) -> list[dict[str, Any]]:
    reports = []
    per_query = frame.groupby("query_id").size().to_dict()
    for index, group in enumerate(_shard_groups(per_query, SHARD_PAIRS)):
        part = frame[frame.query_id.isin(group)]
        table = pa.table({c: pa.array(part[c].tolist() if types[c] == pa.string() else part[c].to_numpy(), types[c])
                          for c in types})
        target = out / "shards" / f"pool-{index:05d}.parquet"
        temp = target.with_name(target.name + ".tmp")
        pq.write_table(table, temp, compression="zstd", row_group_size=5_000)
        temp.replace(target)
        reports.append({"file": target.name, "pairs": len(part), "bytes": target.stat().st_size, "sha256": _sha256(target)})
    log(f"wrote {len(reports)} shards rows={len(frame)}")
    return reports


def load_pool211() -> pd.DataFrame:
    path = s31_root() / "pool-extra" / "pool211_index.parquet"
    paths.require_inputs(path)
    frame = pd.read_parquet(path)
    if len(frame) != 253_200 or frame.pair_id.duplicated().any() or frame.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("pool211_index is not 1,200 x 211 unique pairs")
    return frame


def s31_text_shards() -> list[Path]:
    root = s31_root()
    return (sorted((root / "pool" / "shards").glob("pool-*.parquet")) + [root / "pool-extra" / "shards" / "pool-91000.parquet"]
            + [root / "pharmacity-ext" / "shards" / "pool-90000.parquet"])


# ------------------------------------------------------------------------------------- R-doc

def prepare_rdoc(out: Path) -> dict[str, Any]:
    started = time.monotonic()
    _prepare_out(out)
    log = _logger(out / "prepare.log")
    pool = load_pool211()
    wanted = set(pool.pair_id.tolist())
    frames = []
    for shard in s31_text_shards():
        paths.require_inputs(shard)
        table = pq.read_table(shard, columns=["pair_id", "query_id", "doc_id", "lang", "query_text", "doc_text", "doc_tokens"])
        part = table.to_pandas()
        frames.append(part[part.pair_id.isin(wanted)])
    texts = pd.concat(frames, ignore_index=True)
    if texts.pair_id.duplicated().any():
        raise ValueError("A pool211 pair_id appears in two S31 text shards")
    merged = pool.merge(texts, on="pair_id", how="left", suffixes=("", "_t"), validate="one_to_one")
    if merged.doc_text.isna().any():
        raise ValueError(f"{int(merged.doc_text.isna().sum())} pool211 pairs have no S31 doc text")
    if not ((merged.query_id == merged.query_id_t).all() and (merged.doc_id == merged.doc_id_t).all()):
        raise ValueError("pool211 query/doc differ from the S31 shard rows")
    merged["in_pool13"] = True
    merged = merged.sort_values(["query_id", "cosine_rank"], kind="stable").reset_index(drop=True)
    shards = write_shards(out, merged, RDOC_COLUMNS, log)
    index_cols = [c for c in RDOC_COLUMNS if c not in ("query_text", "doc_text")]
    _atomic_frame(out / "pool_index.parquet", merged[index_cols])
    manifest = {
        "status": "exported", "pool": "S31 cosine top-211 (pool211_index.parquet)", "pairs": len(merged),
        "queries": int(merged.query_id.nunique()), "lang": merged.lang.value_counts().to_dict(),
        "score_origin": merged.score_origin.value_counts().to_dict(),
        "doc_tokens_qwen_mean": float(merged.doc_tokens.mean()), "doc_tokens_qwen_at_512": int((merged.doc_tokens >= DOC_TOKEN_LIMIT).sum()),
        "pool211_index_sha256": _sha256(s31_root() / "pool-extra" / "pool211_index.parquet"),
        "text_shards": [str(s) for s in s31_text_shards()], "shards": shards, "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", manifest)
    return manifest


# ------------------------------------------------------------------------------------- R-chk

def select_like_s32(pairs: pd.DataFrame, log: Callable[[str], None], supplement: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """S32 chunk choice for ``pairs`` (src_pair_id, query_id, doc_id, cosine_rank, cosine_score, lang, source).

    Same owner map, slots, stored vectors, top-n and ties as ``r2ai.s32.chunks.prepare``; chunk
    texts over 512 Qwen3 tokens are cut with ``truncate_document`` as there.
    """
    inputs = _inputs()
    canonical = inputs["canonical"]
    manifest31 = json.loads((s31_root() / "pool" / "manifest.json").read_text(encoding="utf-8"))
    qvec = s32.load_query_vectors(manifest31)
    groups, cid_to_doc = s29_build._load_aliases(canonical / "chunks_t256.parquet", canonical / "docs.parquet")
    ph_chunks_all = s32.load_pharmacity_chunks(inputs)
    ph_docs = set(ph_chunks_all.doc_id.astype(int))
    owner_of: dict[int, int] = {}
    origin_of: dict[int, str] = {}
    for doc in sorted(set(pairs.doc_id.astype(int))):
        if doc in ph_docs:
            owner_of[doc], origin_of[doc] = doc, "p"
        elif doc in cid_to_doc and cid_to_doc[doc] in groups:
            owner_of[doc], origin_of[doc] = int(cid_to_doc[doc]), "c"
    owners_c = np.array(sorted({o for d, o in owner_of.items() if origin_of[d] == "c"}), dtype=np.int64)
    meta = s32.load_canonical_meta(canonical)
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
    log(f"pair docs={pairs.doc_id.nunique()} with_chunks={len(owner_of)} slots={len(slots)}")

    dense = np.load(canonical / "vectors" / "dense.npy", mmap_mode="r")
    vectors = np.zeros((len(slots), s32.DIM), dtype=np.float16)
    has = (slots.vec_row >= 0).to_numpy()
    where, rows = np.flatnonzero(has), slots.vec_row.to_numpy()[has]
    order = np.argsort(rows, kind="stable")
    for begin in range(0, len(order), 200_000):
        take = order[begin : begin + 200_000]
        vectors[where[take]] = dense[rows[take]]
    with np.load(supplement) as sup:
        sup_key = {(str(o), int(c)): i for i, (o, c) in enumerate(zip(sup["origin"], sup["chunk_id"]))}
        sup_dense = np.array(sup["dense"], copy=True)
    missing = np.flatnonzero(~has)
    from_s32, to_embed = [], []
    for slot in missing.tolist():
        key = (str(slots.origin.iat[slot]), int(slots.chunk_id.iat[slot]))
        if key in sup_key:
            vectors[slot] = sup_dense[sup_key[key]]
            from_s32.append(slot)
        else:
            to_embed.append(slot)
    log(f"vectors stored={int(has.sum())} s32_supplement={len(from_s32)} to_embed={len(to_embed)}")
    ph_text = dict(zip(ph_chunks_all.chunk_id.astype(int), ph_chunks_all.text))
    texts_c: dict[int, str] = {}
    if to_embed:
        need = slots.chunk_id.to_numpy()[to_embed][slots.origin.to_numpy()[to_embed] == "c"]
        texts_c.update(s32.fetch_canonical_texts(canonical, need, log))
        from r2ai.index.bge_m3 import M3Encoder

        encoder = M3Encoder(max_len=512)
        emb = s32.embed_texts(encoder, [ph_text[int(slots.chunk_id.iat[s])] if slots.origin.iat[s] == "p"
                                        else texts_c[int(slots.chunk_id.iat[s])] for s in to_embed], log)
        vectors[to_embed] = emb
        sup_path = out / "supplement_vectors.npz"
        temp = sup_path.with_name(sup_path.name + ".tmp")
        with temp.open("wb") as fh:
            np.savez(fh, origin=slots.origin.to_numpy()[to_embed].astype("U1"), chunk_id=slots.chunk_id.to_numpy()[to_embed], dense=emb)
        temp.replace(sup_path)
        del encoder

    pairs = pairs.reset_index(drop=True).copy()
    pairs["owner"] = pairs.doc_id.map(owner_of)
    pairs["slot_doc"] = np.where(pairs.owner.notna(), np.searchsorted(doc_ids_u, pairs.owner.fillna(-1).astype(np.int64)), -1)
    pairs.loc[pairs.owner.isna(), "slot_doc"] = -1
    selection = s32.select_chunks(pairs, doc_start, doc_count, vectors, qvec, TOP_N)
    sel = selection.merge(pairs[["src_pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang", "source"]],
                          left_on="pair_pos", right_index=True).drop(columns=["pair_pos"])
    sel["chunk_id"] = slots.chunk_id.to_numpy()[sel.slot]
    sel["field"] = slots.field.to_numpy()[sel.slot]
    sel["origin"] = slots.origin.to_numpy()[sel.slot]
    fallback = pairs[~pairs.src_pair_id.isin(set(sel.src_pair_id))][["src_pair_id", "query_id", "doc_id", "cosine_rank", "source"]].copy()
    fallback["reason"] = "no_chunk"
    info = {"supplement_from_s32": len(from_s32), "embedded_here": len(to_embed)}
    return sel.drop(columns=["slot"]), fallback, info


def attach_texts(sel: pd.DataFrame, log: Callable[[str], None]) -> pd.DataFrame:
    inputs = _inputs()
    need = sel.loc[sel.origin == "c", "chunk_id"].to_numpy()
    texts_c = s32.fetch_canonical_texts(inputs["canonical"], need, log)
    ph = s32.load_pharmacity_chunks(inputs)
    ph_text = dict(zip(ph.chunk_id.astype(int), ph.text))
    keys = sorted({(o, int(c)) for o, c in zip(sel.origin, sel.chunk_id)})
    text_by_key = {k: (ph_text[k[1]] if k[0] == "p" else texts_c[k[1]]) for k in keys}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    counts = {}
    for begin in range(0, len(keys), 20_000):
        part = keys[begin : begin + 20_000]
        counts.update({k: len(ids) for k, ids in zip(part, tokenizer([text_by_key[k] for k in part], add_special_tokens=False)["input_ids"])})
    truncated = 0
    for key in keys:
        if counts[key] > DOC_TOKEN_LIMIT:
            text_by_key[key], counts[key] = truncate_document(tokenizer, text_by_key[key])
            truncated += 1
    log(f"chunk texts={len(keys)} truncated_to_{DOC_TOKEN_LIMIT}={truncated}")
    sel = sel.copy()
    sel["doc_text"] = [text_by_key[(o, int(c))] for o, c in zip(sel.origin, sel.chunk_id)]
    sel["doc_tokens"] = np.array([counts[(o, int(c))] for o, c in zip(sel.origin, sel.chunk_id)], dtype=np.int32)
    return sel


def prepare_rchk(out: Path, rdoc: Path) -> dict[str, Any]:
    started = time.monotonic()
    _prepare_out(out)
    log = _logger(out / "prepare.log")
    pool = pd.read_parquet(rdoc / "pool_index.parquet")
    queries = {}
    for shard in sorted((rdoc / "shards").glob("pool-*.parquet")):
        frame = pd.read_parquet(shard, columns=["query_id", "query_text"]).drop_duplicates("query_id")
        queries.update(zip(frame.query_id.astype(int), frame.query_text))
    s32_dir = s32_pool()
    s32_index = pd.read_parquet(s32_dir / "chunk_pairs.parquet")
    s32_fallback = pd.read_parquet(s32_dir / "doc_fallback.parquet")
    in_s32 = pool.pair_id.isin(s32_index.src_pair_id) | pool.pair_id.isin(s32_fallback.src_pair_id)
    log(f"pool211={len(pool)} in_s32={int(in_s32.sum())} new={int((~in_s32).sum())}")

    # ---- new pairs + a gate sample of S32 pairs through the same code path
    new = pool[~in_s32][["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang"]].rename(columns={"pair_id": "src_pair_id"})
    new["source"] = "s33_extra"
    covered = pool[pool.pair_id.isin(s32_index.src_pair_id)]
    gate = covered.sample(n=GATE_PAIRS, random_state=GATE_SEED)[["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "lang"]]
    gate = gate.rename(columns={"pair_id": "src_pair_id"})
    gate["source"] = "gate"
    sel, fallback, info = select_like_s32(pd.concat([new, gate], ignore_index=True), log, s32_dir / "supplement_vectors.npz", out)
    gsel = sel[sel.source == "gate"]
    ref = s32_index[s32_index.src_pair_id.isin(gate.src_pair_id)]
    got = gsel.sort_values(["src_pair_id", "chunk_rank"]).groupby("src_pair_id").chunk_id.apply(list)
    exp = ref.sort_values(["src_pair_id", "chunk_rank"]).groupby("src_pair_id").chunk_id.apply(list)
    mismatch = [int(k) for k in exp.index if got.get(k) != exp[k]]
    gate_report = {"pairs": GATE_PAIRS, "seed": GATE_SEED, "mismatches": len(mismatch), "examples": mismatch[:5],
                   "gate_fallback": int((fallback.source == "gate").sum())}
    _atomic_json(out / "gate.json", gate_report)
    log(f"gate {json.dumps(gate_report)}")
    if mismatch or gate_report["gate_fallback"]:
        raise ValueError("S33 chunk selection differs from S32 on the gate sample; see gate.json")
    sel = attach_texts(sel[sel.source == "s33_extra"], log)
    fallback = fallback[fallback.source == "s33_extra"]

    # ---- S32 rows of the covered pairs (verbatim) + new rows
    old_rows = []
    for shard in sorted((s32_dir / "shards").glob("pool-*.parquet")):
        frame = pd.read_parquet(shard)
        old_rows.append(frame[frame.src_pair_id.isin(covered.pair_id)])
    old = pd.concat(old_rows, ignore_index=True)
    old["origin"] = old.chunk_id.map(dict(zip(s32_index.chunk_id, s32_index.origin)))
    old["chunk_source"] = "s32"
    sel["pair_id"] = NEW_PAIR_ID_BASE + np.arange(len(sel), dtype=np.int64)
    sel["query_text"] = sel.query_id.map(queries)
    sel["chunk_source"] = "s33"
    rows = pd.concat([old, sel], ignore_index=True)
    rows["in_pool13"] = True
    origin_of = dict(zip(pool.pair_id, pool.cosine_rank))
    rows["cosine_rank"] = rows.src_pair_id.map(origin_of).astype(np.int64)  # rank inside pool211
    rows = rows.sort_values(["query_id", "cosine_rank", "chunk_rank"], kind="stable").reset_index(drop=True)
    if rows.pair_id.duplicated().any() or rows.duplicated(["src_pair_id", "chunk_rank"]).any():
        raise ValueError("Duplicate chunk rows")
    shards = write_shards(out, rows, RCHK_COLUMNS, log)
    _atomic_frame(out / "chunk_pairs.parquet", rows[[c for c in RCHK_COLUMNS if c not in ("query_text", "doc_text")]])
    all_fallback = pd.concat([s32_fallback[s32_fallback.src_pair_id.isin(pool.pair_id)].assign(source="s32"), fallback], ignore_index=True)
    _atomic_frame(out / "doc_fallback.parquet", all_fallback)
    per_pair = rows.groupby("src_pair_id").size()
    manifest = {
        "status": "exported", "top_n": TOP_N, "pool211_pairs": len(pool), "pairs_with_chunks": int(per_pair.shape[0]),
        "pairs_fallback_doc_text": len(all_fallback), "chunk_rows": len(rows),
        "chunk_rows_from_s32": int((rows.chunk_source == "s32").sum()), "chunk_rows_new": int((rows.chunk_source == "s33").sum()),
        "chunks_per_pair": {str(k): int(v) for k, v in per_pair.value_counts().sort_index().items()},
        "field_share": {k: float(v) for k, v in rows.field.value_counts(normalize=True).items()},
        "gate": gate_report, **info, "shards": shards, "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", manifest)
    return manifest


def prepare_tie(out_rdoc: Path, out_rchk: Path) -> dict[str, Any]:
    """Pairs of the J4-K40 pool (cosine top-211 of the S31 scored union) missing from pool211_index.

    Both lists cut a block of exactly tied cosines (duplicate vectors) at rank 211 in a different
    order, so a few queries differ. The missing pairs are pool13 pairs: R-doc rows come verbatim
    from the S31 shards, R-chk rows verbatim from the S32 shards (or the S32 fallback list).
    """
    from r2ai.s31 import build as s31_build
    from r2ai.s33.build import load_union

    union, _ = load_union(s31_root())
    pools, _ = s31_build.topk_pools(union, s31_build.P_FREE)
    pool211 = load_pool211()
    known = set(zip(pool211.query_id.astype(int), pool211.doc_id.astype(int)))
    rows = [{"pair_id": int(r["pair_id"]), "query_id": q, "doc_id": int(r["doc_id"]), "cosine_rank": int(r["cosine_rank"]),
             "cosine_score": float(r["cosine_score"]), "score_origin": r["origin"]}
            for q, rs in pools.items() for r in rs if (q, int(r["doc_id"])) not in known]
    tie = pd.DataFrame(rows)
    report: dict[str, Any] = {"pairs": len(tie), "queries": int(tie.query_id.nunique()) if len(tie) else 0,
                              "origin": tie.score_origin.value_counts().to_dict() if len(tie) else {}}
    for out in (out_rdoc, out_rchk):
        _prepare_out(out)
    log = _logger(out_rdoc / "prepare.log")
    ids = set(tie.pair_id)
    texts = pd.concat([pd.read_parquet(s, columns=["pair_id", "query_id", "doc_id", "lang", "query_text", "doc_text", "doc_tokens"])
                       .pipe(lambda f: f[f.pair_id.isin(ids)]) for s in s31_text_shards()], ignore_index=True)
    rdoc = tie.merge(texts, on=["pair_id", "query_id", "doc_id"], how="left", validate="one_to_one")
    if rdoc.doc_text.isna().any():
        raise ValueError("tie pair without S31 doc text")
    rdoc["in_pool13"] = True
    report["rdoc_shards"] = write_shards(out_rdoc, rdoc, RDOC_COLUMNS, log)
    _atomic_frame(out_rdoc / "pool_index.parquet", rdoc[[c for c in RDOC_COLUMNS if c not in ("query_text", "doc_text")]])
    s32_dir = s32_pool()
    s32_index = pd.read_parquet(s32_dir / "chunk_pairs.parquet")
    chk = pd.concat([pd.read_parquet(s).pipe(lambda f: f[f.src_pair_id.isin(ids)]) for s in sorted((s32_dir / "shards").glob("pool-*.parquet"))],
                    ignore_index=True)
    chk["origin"] = chk.chunk_id.map(dict(zip(s32_index.chunk_id, s32_index.origin)))
    chk["chunk_source"] = "s32"
    chk["in_pool13"] = True
    chk["cosine_rank"] = chk.src_pair_id.map(dict(zip(tie.pair_id, tie.cosine_rank))).astype(np.int64)
    fallback = pd.read_parquet(s32_dir / "doc_fallback.parquet")
    fallback = fallback[fallback.src_pair_id.isin(ids)].assign(source="s32")
    if set(chk.src_pair_id) | set(fallback.src_pair_id) != ids:
        raise ValueError("tie pairs missing from both the S32 chunk pool and its fallback list")
    log_c = _logger(out_rchk / "prepare.log")
    report["rchk_rows"] = len(chk)
    report["rchk_fallback"] = len(fallback)
    report["rchk_shards"] = write_shards(out_rchk, chk, RCHK_COLUMNS, log_c)
    _atomic_frame(out_rchk / "chunk_pairs.parquet", chk[[c for c in RCHK_COLUMNS if c not in ("query_text", "doc_text")]])
    _atomic_frame(out_rchk / "doc_fallback.parquet", fallback)
    for out in (out_rdoc, out_rchk):
        _atomic_json(out / "manifest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rdoc = sub.add_parser("rdoc", help="pool211 doc_text shards")
    rdoc.add_argument("--out", type=Path, default=None)
    rchk = sub.add_parser("rchk", help="pool211 top-3 chunk shards (S32 selection)")
    rchk.add_argument("--out", type=Path, default=None)
    rchk.add_argument("--rdoc", type=Path, default=None)
    sub.add_parser("tie", help="J4-K40 pool pairs missing from pool211_index (cosine ties at rank 211)")
    args = parser.parse_args(argv)
    root = run_root()
    if args.command == "tie":
        result = prepare_tie(root / "pool-rdoc-tie", root / "pool-rchk-tie")
    elif args.command == "rdoc":
        result = prepare_rdoc(paths.resolve_path(args.out or root / "pool-rdoc"))
    else:
        result = prepare_rchk(paths.resolve_path(args.out or root / "pool-rchk"), paths.resolve_path(args.rdoc or root / "pool-rdoc"))
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
