"""S38 chunk sources for the chunk-carrying docs of J3-thrC (verbatim spans of canonical doc_text).

``python -m r2ai.s38.sources`` (CPU) -> ``out/runs/s38-chunk-metric-v2/sources/``:

* carriers = every (query, doc_id) of J3-thrC ``relevant_chunks`` (the selected docs with P4B >= t_c);
  J3-thrC is read from its ZIP, pinned by SHA256.
* text of a carrier doc = its own canonical ``doc_text`` row, else the row of its alias-group owner (the
  validator S26 rule); the owner's ``chunks_t256`` spans must index that text verbatim.
* ``t256``: the canonical ``chunks_t256`` of the owner, fields answer/body (title/question are never submitted).
* ``t128``: the same chunker (``r2ai.index.chunk.chunk_paragraphs``, 1-paragraph overlap, blob drop) with
  target 128 on paragraphs rebuilt from the doc_text: the body region (from the first non-title chunk) split at
  ``"\\n\\n"`` (the extractor writes body = "\\n\\n".join(paragraphs)), each paragraph labelled with the field of
  the canonical chunk holding its start. Gate: the same rebuild with target 256 must give the canonical spans.
* ``sec``: the S12 chunker (``r2ai.vi_quality.chunker.structured_chunks``, target 256, extractor headings from
  docs_vi when the doc comes from it). A doc without structure (no heading / item / row paragraph and no
  question field in the body) takes its t256 chunks (``sec_from_t256``).

Outputs: ``pieces.parquet`` (one row per submittable piece: doc_id, owner, kind, idx, field, char_start, char_end,
text, n_tokens raw BGE-M3, chunk_id of the identical canonical t256 chunk or -1, vec_row), ``carriers.parquet``,
``docs.parquet`` (per carrier doc: lang, host, segment, structure flag, gate result), ``sources.json``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.index.chunk import SUBMITTABLE, _token_counts, chunk_paragraphs
from r2ai.vi_quality.chunker import _norm, classify, structured_chunks

RUN_NAME = "s38-chunk-metric-v2"
J3_ZIP = Path("runs/s35-part-jina/build-j3/J3-thrC.zip")
J3_ZIP_SHA256 = "546b6307175fbed663c5e7f6ec9e8791d4f17d24a7d1bcaedc396fbbadf83033"
J3_JSON_SHA256 = "e11e51e6bd0e3793f749915eca99386b48e67e9e590b7b713abd75037855e500"
CANONICAL = "canonical-20261018"
T_SMALL = 128
T_SEC = 256
T_BASE = 256
KINDS = ("t256", "t128", "sec")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def run_root() -> Path:
    return paths.run_dir(RUN_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, data: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    temp.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def describe(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()),
            **{f"p{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES}, "max": float(v.max())}


# ------------------------------------------------------------------------------------------ J3-thrC

def iter_rows(zip_path: Path):
    """Rows of a compose/S29 bundle (one JSON row per line), parsed lazily; also the raw bytes for the hash."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: ZIP must hold one file, found {len(names)}")
        with archive.open(names[0]) as fh:
            for line in fh:
                yield line
                body = line.strip().rstrip(b",")
                if body.startswith(b"{"):
                    yield json.loads(body)


def read_bundle(zip_path: Path, json_sha256: str | None = None) -> list[dict[str, Any]]:
    digest = hashlib.sha256()
    rows = []
    for item in iter_rows(zip_path):
        if isinstance(item, bytes):
            digest.update(item)
        else:
            rows.append(item)
    if json_sha256 is not None and digest.hexdigest() != json_sha256:
        raise ValueError(f"{zip_path}: JSON SHA256 {digest.hexdigest()} != {json_sha256}")
    return rows


def carriers_of(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = [(int(r["id"]), int(c["doc_id"]), j) for r in rows for j, c in enumerate(r["relevant_chunks"])]
    frame = pd.DataFrame(out, columns=["query_id", "doc_id", "chunk_pos"])
    if frame.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("J3-thrC has two chunks for one (query, doc)")
    return frame


# ------------------------------------------------------------------------------------ paragraphs

def rebuild_paragraphs(doc_text: str, chunks: Sequence[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Paragraph spans (field, start, end) of the body region of ``doc_text``.

    ``chunks``: the canonical t256 (field, start, end) of the doc. The region starts at the first non-title chunk;
    it is split at ``"\\n\\n"`` (empty pieces skipped). A paragraph takes the field of the non-title chunk that
    holds its start, else ``body`` (e.g. a dropped blob).
    """
    inner = sorted((s, e, f) for f, s, e in chunks if f != "title")
    if not inner:
        return []
    start = inner[0][0]
    starts = np.array([s for s, _, _ in inner])
    paras = []
    pos = start
    for piece in doc_text[start:].split("\n\n"):
        s, e = pos, pos + len(piece)
        pos = e + 2
        if not piece:
            continue
        k = int(np.searchsorted(starts, s, side="right")) - 1   # chunks k, k-1, ... start at or before s
        field = "body"
        for j in range(k, max(k - 4, -1), -1):                 # overlap is one paragraph: a few steps back suffice
            if s < inner[j][1]:
                field = inner[j][2]
                break
        paras.append((field, s, e))
    return paras


def drop_blobs(doc_text: str, pieces: list[tuple[str, int, int]], ntoks: list[int], target: int):
    """The chunker's post filter: whitespace-free pieces over 1.5 x target tokens are dropped."""
    keep = [not (n > 1.5 * target and not any(ch.isspace() for ch in doc_text[s:e])) for (_, s, e), n in zip(pieces, ntoks)]
    return [p for p, k in zip(pieces, keep) if k], [n for n, k in zip(ntoks, keep) if k]


def has_structure(doc_text: str, paras: Sequence[tuple[str, int, int]], headings: Sequence[str] = ()) -> bool:
    """Vu (23/10): structure = a heading / item / row paragraph, or a question field in the body."""
    hs = {_norm(h) for h in headings if h and h.strip()}
    for f, s, e in paras:
        if f == "question" or classify(doc_text[s:e], hs) in ("heading", "item", "row"):
            return True
    return False


# ------------------------------------------------------------------------------------- loaders

def load_owner_map(canonical: Path) -> tuple[set[int], dict[int, int]]:
    from r2ai.s29 import build as s29_build

    groups, cid_to_doc = s29_build._load_aliases(canonical / "chunks_t256.parquet", canonical / "docs.parquet")
    return set(groups), cid_to_doc


def owner_of(doc_id: int, owners: set[int], cid_to_doc: dict[int, int]) -> int:
    return doc_id if doc_id in owners else int(cid_to_doc.get(doc_id, doc_id))


def load_chunks(canonical: Path, owners: Sequence[int]) -> pd.DataFrame:
    """Canonical t256 rows of ``owners`` with their dense row (running count of has_vector in file order)."""
    path = canonical / "chunks_t256.parquet"
    has = pq.read_table(path, columns=["has_vector"]).column("has_vector").to_numpy(zero_copy_only=False)
    vec_row = np.where(has, np.cumsum(has) - 1, -1).astype(np.int64)
    wanted = pa.array(sorted(set(int(o) for o in owners)), pa.int64())
    frames, offset = [], 0
    parquet = pq.ParquetFile(path)
    cols = ["chunk_id", "doc_id", "field", "char_start", "char_end", "text", "n_tokens", "has_vector"]
    for batch in parquet.iter_batches(batch_size=200_000, columns=cols):
        mask = pc.is_in(batch.column("doc_id"), value_set=wanted)
        idx = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        if len(idx):
            frame = batch.take(pa.array(idx)).to_pandas()
            frame["vec_row"] = vec_row[offset + idx]
            frame["file_row"] = offset + idx
            frames.append(frame)
        offset += len(batch)
    return pd.concat(frames, ignore_index=True)


def load_doc_rows(canonical: Path, ids: Sequence[int]) -> pd.DataFrame:
    return pq.read_table(canonical / "docs.parquet", columns=["doc_id", "lang", "host", "segment", "doc_text"],
                         filters=[("doc_id", "in", sorted(set(int(i) for i in ids)))]).to_pandas()


def load_headings(docs_dir: Path, ids: set[int], log: Callable[[str], None]) -> dict[int, list[str]]:
    """Extractor headings of docs_vi rows (status ok) whose first doc id is wanted."""
    out: dict[int, list[str]] = {}
    files = sorted(glob.glob(str(docs_dir / "*.parquet")))
    wanted = pa.array(sorted(ids), pa.int64())
    started = time.monotonic()
    for k, f in enumerate(files):
        table = pq.read_table(f, columns=["doc_ids", "headings", "status"])
        first = pc.list_element(table["doc_ids"], 0)
        mask = pc.and_(pc.is_in(first, value_set=wanted), pc.equal(table["status"], "ok"))
        if pc.any(mask).as_py():
            hit = table.filter(mask)
            for d, h in zip(pc.list_element(hit["doc_ids"], 0).to_pylist(), hit["headings"].to_pylist()):
                out[int(d)] = [x for x in (h or []) if x]
        if k % 2000 == 0:
            log(f"headings files {k}/{len(files)} hits={len(out)} elapsed_s={time.monotonic() - started:.0f}")
    return out


# ------------------------------------------------------------------------------------ chunking

def doc_pieces(doc_text: str, chunks: list[tuple[int, str, int, int]], ptoks_of: Callable[[list[str]], list[int]],
               ntok: Callable[[str], int], headings: Sequence[str] | None) -> dict[str, Any]:
    """t256 / t128 / sec pieces of one doc. ``chunks``: canonical (chunk_id, field, start, end) in file order."""
    spans = [(f, s, e) for _, f, s, e in chunks]
    paras = rebuild_paragraphs(doc_text, spans)
    ptoks = ptoks_of([doc_text[s:e] for _, s, e in paras]) if paras else []
    by_span = {(s, e): cid for cid, f, s, e in chunks if f in SUBMITTABLE}
    canon_inner = sorted((f, s, e) for f, s, e in spans if f != "title")

    def chunked(target: int) -> list[tuple[str, int, int]]:
        pieces = chunk_paragraphs(doc_text, paras, ptoks, target, ntok)
        return drop_blobs(doc_text, pieces, [ntok(doc_text[s:e]) for _, s, e in pieces], target)[0]

    gate = sorted(chunked(T_BASE)) == canon_inner
    t128 = [p for p in chunked(T_SMALL) if p[0] in SUBMITTABLE]
    structured = has_structure(doc_text, paras, headings or ())
    if structured:
        sec = structured_chunks(doc_text, paras, ptoks, T_SEC, ntok, headings=headings or ())
        sec = drop_blobs(doc_text, sec, [ntok(doc_text[s:e]) for _, s, e in sec], T_SEC)[0]
        sec = [p for p in sec if p[0] in SUBMITTABLE]
    else:
        sec = [(f, s, e) for _, f, s, e in chunks if f in SUBMITTABLE]
    return {"gate": gate, "structured": structured, "paragraphs": len(paras),
            "t256": [(cid, f, s, e) for cid, f, s, e in chunks if f in SUBMITTABLE],
            "t128": [(by_span.get((s, e), -1), f, s, e) for f, s, e in t128],
            "sec": [(by_span.get((s, e), -1), f, s, e) for f, s, e in sec]}


# --------------------------------------------------------------------------------------- prepare

def _guard(out: Path) -> None:
    for name in ("", "pieces.parquet", "carriers.parquet", "docs.parquet", "sources.json", "sources.log"):
        target = out / name if name else out
        paths.assert_writable(target)
        if name:
            paths.assert_writable(target.with_name(target.name + ".tmp"))


def prepare(out: Path, limit_docs: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    _guard(out)
    out.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                           "message": message}, ensure_ascii=False)
        print(line, flush=True)
        with (out / "sources.log").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    zip_path = paths.OUT_DIR / J3_ZIP
    canonical = paths.DATA_DIR / CANONICAL
    paths.require_inputs(zip_path, canonical / "chunks_t256.parquet", canonical / "docs.parquet")
    if _sha256(zip_path) != J3_ZIP_SHA256:
        raise ValueError(f"{zip_path}: ZIP SHA256 differs from J3-thrC")
    rows = read_bundle(zip_path, J3_JSON_SHA256)
    carriers = carriers_of(rows)
    del rows
    log(f"J3-thrC carriers={len(carriers)} docs={carriers.doc_id.nunique()}")

    owners, cid_to_doc = load_owner_map(canonical)
    doc_ids = sorted(set(carriers.doc_id.astype(int)))
    if limit_docs:
        doc_ids = doc_ids[:limit_docs]
    owner = {d: owner_of(d, owners, cid_to_doc) for d in doc_ids}
    docs = load_doc_rows(canonical, set(doc_ids) | set(owner.values()))
    text = dict(zip(docs.doc_id.astype(int), docs.doc_text))
    meta = docs.set_index("doc_id")[["lang", "host", "segment"]]
    text_key = {d: d if d in text else owner[d] for d in doc_ids}
    missing = [d for d in doc_ids if text_key[d] not in text]
    if missing:
        raise ValueError(f"{len(missing)} carrier docs without canonical doc_text, e.g. {missing[:5]}")
    chunks = load_chunks(canonical, set(owner.values()))
    by_owner = {int(o): g.sort_values("file_row") for o, g in chunks.groupby("doc_id", sort=False)}
    log(f"owners={len(set(owner.values()))} t256 rows={len(chunks)}")

    base_ids = {d for d in doc_ids if str(meta.loc[text_key[d], "segment"]) == "base-t256"}
    headings = load_headings(paths.DOCS_DIR, {text_key[d] for d in base_ids}, log)
    log(f"headings found={len(headings)} of base-t256 docs={len(base_ids)}")

    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    cache: dict[str, int] = {}

    def ntok(s: str) -> int:
        v = cache.get(s)
        if v is None:
            v = cache[s] = len(tok(s, add_special_tokens=False)["input_ids"])
        return v

    def ptoks_of(texts: list[str]) -> list[int]:
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        for t, n in zip(todo, _token_counts(tok, todo)):
            cache[t] = n
        return [cache[t] for t in texts]

    piece_rows: list[tuple] = []
    doc_rows: list[dict[str, Any]] = []
    bad_span = 0
    for k, d in enumerate(doc_ids):
        o, key = owner[d], text_key[d]
        t = text[key] or ""
        g = by_owner.get(o)
        ch = [] if g is None else list(zip(g.chunk_id.astype(int), g.field, g.char_start.astype(int), g.char_end.astype(int)))
        texts_ok = g is not None and all(t[s:e] == x for (_, _, s, e), x in zip(ch, g.text))
        if not texts_ok:
            bad_span += 1
            doc_rows.append({"doc_id": d, "owner": o, "text_doc": key, "spans_ok": False, "gate": False,
                             "structured": False, "paragraphs": 0, "headings": key in headings})
            continue
        res = doc_pieces(t, ch, ptoks_of, ntok, headings.get(key))
        vec = dict(zip(g.chunk_id.astype(int), g.vec_row.astype(int)))
        for kind in KINDS:
            for idx, (cid, f, s, e) in enumerate(res[kind]):
                piece_rows.append((d, o, kind, idx, f, s, e, t[s:e], int(cid), vec.get(int(cid), -1) if cid >= 0 else -1))
        doc_rows.append({"doc_id": d, "owner": o, "text_doc": key, "spans_ok": True, "gate": res["gate"],
                         "structured": res["structured"], "paragraphs": res["paragraphs"], "headings": key in headings})
        if k % 5000 == 0:
            log(f"docs {k}/{len(doc_ids)} pieces={len(piece_rows)} elapsed_s={time.monotonic() - started:.0f}")
        if len(cache) > 2_000_000:
            cache.clear()

    pieces = pd.DataFrame(piece_rows, columns=["doc_id", "owner", "kind", "idx", "field", "char_start", "char_end",
                                               "text", "chunk_id", "vec_row"])
    pieces["n_tokens"] = np.array(ptoks_of(pieces.text.tolist()), dtype=np.int32)
    cache.clear()
    dframe = pd.DataFrame(doc_rows)
    dframe = dframe.join(meta, on="text_doc")
    dframe["sec_from_t256"] = dframe.spans_ok & ~dframe.structured
    _atomic_parquet(pieces, out / "pieces.parquet")
    _atomic_parquet(carriers[carriers.doc_id.isin(doc_ids)], out / "carriers.parquet")
    _atomic_parquet(dframe, out / "docs.parquet")

    report: dict[str, Any] = {
        "j3_zip": str(zip_path), "j3_zip_sha256": J3_ZIP_SHA256, "j3_json_sha256": J3_JSON_SHA256,
        "canonical": str(canonical), "carriers": int(carriers.doc_id.isin(doc_ids).sum()), "docs": len(doc_ids),
        "owners": len(set(owner.values())), "docs_spans_not_verbatim": bad_span,
        "gate_t256_rebuild": {"docs": int(dframe.spans_ok.sum()), "equal": int(dframe.gate.sum()),
                              "share": float(dframe.gate[dframe.spans_ok].mean()),
                              "by_lang": dframe[dframe.spans_ok].groupby("lang").gate.mean().to_dict()},
        "structured_docs": int(dframe.structured.sum()), "sec_from_t256_docs": int(dframe.sec_from_t256.sum()),
        "structured_share_by_lang": dframe[dframe.spans_ok].groupby("lang").structured.mean().to_dict(),
        "headings_docs": int(dframe.headings.sum()),
        "pieces": {},
        "seconds": None,
    }
    for kind in KINDS:
        sub = pieces[pieces.kind == kind]
        per_doc = sub.groupby("doc_id").size().reindex(dframe.doc_id[dframe.spans_ok], fill_value=0)
        report["pieces"][kind] = {"pieces": int(len(sub)), "tokens_raw": describe(sub.n_tokens),
                                  "per_doc": describe(per_doc), "docs_without_piece": int((per_doc == 0).sum()),
                                  "identical_to_t256": int((sub.chunk_id >= 0).sum()),
                                  "by_field": sub.field.value_counts().to_dict()}
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / "sources.json", report)
    log(f"done seconds={report['seconds']}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="default RUNS_DIR/s38-chunk-metric-v2/sources")
    parser.add_argument("--limit-docs", type=int, default=None, help="first N carrier docs only (smoke)")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or run_root() / "sources")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    report = prepare(out, args.limit_docs)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
