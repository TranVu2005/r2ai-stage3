"""S38 compose: any doc selection + one S38 chunk rule -> ZIP + validator S26.

``python -m r2ai.s38.compose run --name NAME --base-zip DOCREAD.zip --docsel docsel-X.parquet --rule RULE``

* base = the whole-body bundle of the doc selection (S36 ``-docread`` ZIP, or J3-thrC): its ``relevant_docs`` and
  chunk carriers must equal the docsel (``rank`` order, ``chunk`` == True) unless ``--skip-docsel-check``.
* rules (``r2ai.s38.window`` / ``r2ai.s38.clean``), applied to the base chunk of every carrier:
  ``clean``      cut the tail before the first host marker (``scores/clean_markers.json``, counted on J3-thrC) and,
                 for baophutho.vn, at the first base64 line (``--no-base64`` turns that off; CH-clean had it off);
  ``cap2048``    body > 2,048 tokens -> 2,048-token window around the best t256 chunk;
  ``full+w768``  body > T tokens -> base chunk + 768-token window around the best t256 chunk (2 chunks);
  ``clean+cap2048``  clean first (same markers), then cap2048 on the cleaned body.
  Carriers the rule leaves alone keep the base chunk byte for byte; docs never change.
* thin (Vu, 24/10; on unless ``--no-thin``, needs the docsel): a carrier whose base chunk is junk by content, any host
  (``is_thin``: at least ``THIN_JUNK_SHARE`` = 50% of its characters are base64 lines and ``{placeholder}`` tokens;
  baophutho.vn extraction failures are ``{title}`` ... + a base64 blob = 79-96% junk) loses its chunk, and the chunk
  goes to the next doc of the global S31 order (P4B descending, then query, then rank) that has no chunk and has a
  chunk source as in the S29 builder (P1 chunk of the query, else first P1 chunk of the doc, else the body span of a
  doc with canonical chunks or of a Pharmacity lot doc), not empty and not itself thin. The new chunk sits at its
  rank among the query's chunks and then goes through the rule like any carrier. Kept in ``<name>.swap.json``.
* carriers override (Vu, 24/10, ``--carriers carriers.parquet`` with ``query_id, doc_id``): the chunk carriers become exactly that
  set (it must lie inside the docsel docs). Carriers of the base bundle that are not in the set lose their chunk, new carriers
  get the whole-body chunk of the S29 builder (``s29_chunk_texts``; a pair without a chunk source is skipped and counted).
  Thin is then computed on the modified carriers and everything is folded into ``<name>.swap.json`` relative to the base
  bundle, so ``write`` is unchanged. The docsel and its docs stay as they are. The T estimate takes the carrier change into
  account (``RATE_CARRIER``).
* no base ZIP (``--base-zip`` omitted, S37 data): the base bundle is built from the docsel itself: ``relevant_docs`` in docsel ``rank``
  order, one whole-body chunk (S29 rule, ``s29_chunk_texts``) on every carrier (the docsel ``chunk`` flags, or ``--carriers``), kept in
  ``<name>.base.parquet``. No S36 -docread ZIP is needed. Docs of the S37 registry (``r2ai.s38.registry``, 120ask segments) take
  their body from their segment; a doc without a body source is skipped and counted.
* ``--fit-carriers`` (full+w768 ``run``, S37 data): when the carriers alone do not fit the ceiling, the lowest-P4B carriers are
  dropped (one global threshold, no K_q): the smallest drop N whose ZIP without windows is under ``BTC_CEILING - reserve``, measured
  by parallel builds (resolution ``--fit-step`` carriers) and bisected as T is; then T is searched for that N. A dropped carrier loses
  its chunks, the docs stay.
* best t256 chunk (CH-t256-1 rule): the S32 chunk judge on the scored answer/body chunks of the canonical owner
  (Pharmacity lots: their own chunks), else cosine(S29 query vector, chunk vector); ties: file order.
* ``full+w768`` picks T itself: the ZIP grows by about ``RATE_FULL`` bytes per second chunk (S38, J3-thrC), so the
  first candidates sit around the estimate; every round builds its candidates in parallel processes and the next
  round bisects between the highest T over the ceiling and the lowest under it, until they are 1 token apart or
  ``--rounds`` is used up. Result: the lowest measured T with ZIP < ``BTC_CEILING - reserve`` (``--reserve``, default
  ``RESERVE`` = 2,257,600 B -> 102,600,000 B). No hand tuning per bundle.

Steps (``run`` = ``prepare`` + ``write``): ``prepare`` writes ``<out>/<name>.cache.parquet`` (chunks per carrier,
body tokens) and ``<name>.prepare.json``; ``write --threshold T`` writes ``<name>[-t<T>].zip``, runs the validator
and ``<name>[-t<T>].compose.json``. Validator: ``scripts/s26/validate_pa3.py`` with the S27 Pharmacity handoff and, for a bundle
holding S37 registry docs, their 120ask segments (``<label>.validator-extra/``, ``registry.validator_extra``) as extra verbatim sources,
``--max-chunks-per-doc 2`` for full+w768. ``validate --name N [--threshold T]`` re-runs it on an existing ZIP and records the result in
``<label>.compose.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.index.chunk import SUBMITTABLE
from r2ai.s38 import clean as cl
from r2ai.s38 import registry as reg
from r2ai.s38.build import replace_chunks, write_bundle
from r2ai.s38.score import S32_ROOT, S32_SCORES, choose, load_query_vectors, scores_dir
from r2ai.s38.sources import (CANONICAL, _atomic_json, _atomic_parquet, _sha256, describe, load_doc_rows,
                              load_owner_map, owner_of, read_bundle, run_root)
from r2ai.s38.window import _body_span, pieces_for

RULES = {"clean": None, "cap2048": ("window", 2048, 2048), "full+w768": ("full+window", 768, 1536),
         "clean+cap2048": ("window", 2048, 2048)}
PLACEHOLDER = re.compile(r"\{\w+\}")
THIN_JUNK_SHARE = 0.5   # thin carrier: >= 50% of the chunk is base64 lines + {placeholders}. EF2-cut-p4b has a clean gap
                        # (non-thin carriers: junk share <= 0.059; thin ones: 0.79-0.96), so 0.5 catches exactly the 24
BTC_CEILING = 104_857_600   # BTC upload limit for new files: 100 MiB (docs/PROGRESS.md, "Trần dùng cho file mới")
RESERVE = 2_257_600   # default margin below BTC_CEILING: ZIP limit 102,600,000 B (the S29 internal ceiling); the only knob
NO_P1 = False           # S40 --no-p1: no P1 chunk text as carrier body (every body from canonical / 120ask / Pharmacity); validator --no-p1
RATE_FULL = 94.3        # B per 768 window: (102,599,152 - 101,181,680) / 15,026 (CH-full+w768-t1849 on J3-thrC)
FULL_MIN_T = 1536
NO_WINDOWS = 10 ** 9     # threshold of a probe build without any second chunk
RATE_DOC = 2.9          # B per relevant_docs entry in the ZIP: 641,652 / 220,825 (EF2-free)
RATE_CARRIER = 580.0    # B per whole-body carrier chunk in the ZIP: (101,717,111 - 641,652) / 174,535 (EF2-free base)
PHARM_HANDOFF = Path("runs/s27-pharmacity/handoff/cc-s27-pharmacity")
VALIDATOR = Path("scripts/s26/validate_pa3.py")


def _log(msg: str) -> None:
    print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": msg}, ensure_ascii=False),
          flush=True)


# ------------------------------------------------------------------------------------------- inputs

def check_docsel(rows: list[dict[str, Any]], docsel: pd.DataFrame) -> dict[str, Any]:
    """Docs (in rank order) and chunk carriers of the base bundle against the docsel."""
    sel = docsel.sort_values(["query_id", "rank"])
    docs = sel.groupby("query_id").doc_id.apply(list).to_dict()
    carriers = set(zip(sel.query_id[sel.chunk], sel.doc_id[sel.chunk]))
    bad_docs = sum(int(r["id"]) not in docs or [int(d) for d in r["relevant_docs"]] != docs[int(r["id"])] for r in rows)
    base_car = {(int(r["id"]), int(c["doc_id"])) for r in rows for c in r["relevant_chunks"]}
    return {"queries": len(rows), "queries_docs_differ": bad_docs, "docsel_queries": len(docs),
            "carriers_base": len(base_car), "carriers_docsel": len(carriers),
            "carriers_only_base": len(base_car - carriers), "carriers_only_docsel": len(carriers - base_car)}


def junk_share(text: str) -> float:
    """Share of the characters of ``text`` that sit in base64 lines (``clean.BASE64_LINE``) or ``{placeholder}`` tokens."""
    text = text.strip()
    if not text:
        return 0.0
    lines = text.split("\n")
    base64 = sum(len(x) for x in lines if cl.BASE64_LINE.match(x.strip()))
    holders = sum(len(m.group()) for x in lines if not cl.BASE64_LINE.match(x.strip()) for m in PLACEHOLDER.finditer(x))
    return (base64 + holders) / len(text)


def is_thin(text: str) -> bool:
    """Junk by content, any host: base64 lines and ``{placeholder}`` tokens make up at least ``THIN_JUNK_SHARE``."""
    return junk_share(text) >= THIN_JUNK_SHARE


def host_map(keys: set[int]) -> dict[int, str]:
    hosts = pq.read_table(paths.DATA_DIR / CANONICAL / "docs.parquet", columns=["doc_id", "host"],
                          filters=[("doc_id", "in", sorted(keys))]).to_pandas()
    return dict(zip(hosts.doc_id.astype(int), hosts.host))


def load_markers() -> dict[str, set[str]]:
    return {h: {m["key"] for m in v["markers"]} for h, v in json.loads(
        (scores_dir() / "clean_markers.json").read_text(encoding="utf-8"))["hosts"].items()}


def apply_swap(rows: list[dict[str, Any]], swap: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows with the thin carriers' chunks removed and the replacement chunks inserted at their rank."""
    removed = {(int(q), int(d)) for q, d in swap.get("removed", [])}
    added: dict[int, list[dict[str, Any]]] = {}
    for a in swap.get("added", []):
        added.setdefault(int(a["query_id"]), []).append(a)
    out = []
    for r in rows:
        qid = int(r["id"])
        chunks = [c for c in r["relevant_chunks"] if (qid, int(c["doc_id"])) not in removed]
        if qid in added:
            pos = {int(d): k for k, d in enumerate(r["relevant_docs"])}
            chunks += [{"doc_id": int(a["doc_id"]), "chunk_text": a["body"]} for a in added[qid]]
            chunks.sort(key=lambda c: pos[int(c["doc_id"])])
        out.append({"id": r["id"], "relevant_docs": r["relevant_docs"], "relevant_chunks": chunks})
    return out


def s29_chunk_texts(pairs: pd.DataFrame) -> dict[tuple[int, int], tuple[str | None, str]]:
    """Whole-body chunk of each (query, doc) as the S29 builder makes it: the P1 chunk of the same query, else the
    first P1 chunk of the doc, else (doc with canonical chunks, or a Pharmacity lot doc) its body span; a doc with
    none of these has no chunk source (``(None, "")``)."""
    from r2ai.s29.build import _load_extra_docs
    from r2ai.s29.qwen_judge import _inputs, _load_submission

    inputs = _inputs()
    p1_rows = [] if reg.QUERY_SET is not None or NO_P1 else _load_submission(inputs["p1"], with_chunks=True)   # S40 new query set / --no-p1
    own = {(int(r["id"]), int(c["doc_id"])): c["chunk_text"] for r in p1_rows for c in r["relevant_chunks"]}
    first: dict[int, str] = {}
    for r in p1_rows:
        for c in r["relevant_chunks"]:
            first.setdefault(int(c["doc_id"]), str(c["chunk_text"]))
    canonical = paths.DATA_DIR / CANONICAL
    owners, cid_to_doc = load_owner_map(canonical)
    extra, extra_spans = _load_extra_docs(inputs)
    docs = set(pairs.doc_id.astype(int))
    index = reg.registry_index()
    reg_docs = {d for d in docs if d in index}
    reg_text = reg.load_docs(index, reg_docs)
    reg_lang = dict(zip(reg_text.doc_id.astype(int), reg_text.lang))
    reg_text = dict(zip(reg_text.doc_id.astype(int), reg_text.doc_text))
    reg_spans = reg.body_chunk_spans(index, reg_docs)
    owner = {d: owner_of(d, owners, cid_to_doc) for d in docs}
    need = {d for d in docs if d not in extra and d not in reg_docs and owner[d] in owners}
    rows = load_doc_rows(canonical, need | {owner[d] for d in need})
    text = dict(zip(rows.doc_id.astype(int), rows.doc_text))
    lang = dict(zip(rows.doc_id.astype(int), rows.lang))
    spans = pq.read_table(canonical / "chunks_t256.parquet", columns=["doc_id", "field", "char_start", "char_end"],
                          filters=[("doc_id", "in", sorted({owner[d] for d in need}))]).to_pandas() if need else None
    by_owner = {int(o): g for o, g in spans.groupby("doc_id")} if spans is not None else {}
    empty = pd.DataFrame({"field": [], "char_start": [], "char_end": []})
    out: dict[tuple[int, int], tuple[str | None, str]] = {}
    for q, d in zip(pairs.query_id.astype(int), pairs.doc_id.astype(int)):
        if (q, d) in own:
            out[(q, d)] = ("p1_row", own[(q, d)])
        elif d in first:
            out[(q, d)] = ("p1_any", first[d])
        elif d in reg_docs:
            t = reg_text.get(d) or ""
            a, b = _body_span(t, str(reg_lang.get(d, "zh")), reg_spans.get(d, empty))
            out[(q, d)] = ("registry", t[a:b]) if t else (None, "")
        elif d in extra:
            sp = extra_spans.get(d, {})
            a, b = sp.get("answer") or sp.get("body") or (0, len(extra[d] or ""))
            out[(q, d)] = ("body", (extra[d] or "")[a:b])
        elif owner[d] in owners:
            k = d if d in text else owner[d]
            t = text.get(k) or ""
            g = by_owner.get(owner[d], empty)
            a, b = _body_span(t, str(lang.get(k, "vi")), g[g.field.isin(SUBMITTABLE)] if len(g) else empty)
            out[(q, d)] = ("body", t[a:b])
        else:
            out[(q, d)] = (None, "")
    return out


def thin_swap(rows: list[dict[str, Any]], docsel: pd.DataFrame) -> dict[str, Any]:
    """Thin carriers (``is_thin``, any host) and their replacements in the global S31 order."""
    car = [(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]]
    thin = [(q, d) for q, d, body in car if is_thin(body)]
    thin_docs = {d for _, d in thin}
    if not thin:
        return {"removed": [], "added": [], "thin_docs": []}
    cand = docsel[~docsel.chunk & ~docsel.doc_id.isin(thin_docs)].copy()
    cand["pos"] = cand["rank"] - 1
    cand = cand.sort_values(["p4b", "query_id", "pos"], ascending=[False, True, True], kind="stable").head(200 * len(thin))
    texts = s29_chunk_texts(cand[["query_id", "doc_id"]])
    added = []
    for q, d, p4b, rank in zip(cand.query_id.astype(int), cand.doc_id.astype(int), cand.p4b, cand["rank"].astype(int)):
        src, body = texts.get((q, d), (None, ""))
        if src is None or not body.strip() or is_thin(body):
            continue
        added.append({"query_id": q, "doc_id": d, "rank": rank, "p4b": float(p4b), "source": src, "body": body})
        if len(added) == len(thin):
            break
    if len(added) < len(thin):
        raise ValueError(f"only {len(added)} replacement carriers for {len(thin)} thin ones")
    thin_p4b = docsel.set_index(["query_id", "doc_id"]).p4b
    return {"removed": [[q, d] for q, d in thin], "thin_docs": sorted(thin_docs),
            "removed_p4b": [float(thin_p4b.loc[(q, d)]) for q, d in thin], "added": added,
            "t_c_before": float(docsel[docsel.chunk].p4b.min()),
            "t_c_after": float(min(docsel[docsel.chunk & ~docsel.doc_id.isin(thin_docs)].p4b.min(), min(a["p4b"] for a in added)))}


def compose_swaps(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """One swap relative to the base bundle from ``first`` (applied to the base) and ``second`` (computed on the result of
    ``first``): a chunk added by ``first`` and removed by ``second`` simply disappears."""
    added_keys = {(int(a["query_id"]), int(a["doc_id"])) for a in first.get("added", [])}
    removed2 = {(int(q), int(d)) for q, d in second.get("removed", [])}
    removed = [[int(q), int(d)] for q, d in first.get("removed", [])]
    removed += [[int(q), int(d)] for q, d in second.get("removed", []) if (int(q), int(d)) not in added_keys]
    added = [a for a in first.get("added", []) if (int(a["query_id"]), int(a["doc_id"])) not in removed2] + list(second.get("added", []))
    return {"removed": removed, "added": added}


def carrier_swap(rows: list[dict[str, Any]], docsel: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    """Swap that turns the base carriers into the ``target`` (query_id, doc_id) set; docs are untouched."""
    base = {(int(r["id"]), int(c["doc_id"])) for r in rows for c in r["relevant_chunks"]}
    want = set(zip(target.query_id.astype(int), target.doc_id.astype(int)))
    sel = docsel.set_index(["query_id", "doc_id"])
    outside = [k for k in want if k not in sel.index]
    if outside:
        raise ValueError(f"{len(outside)} target carriers are not docs of the docsel, e.g. {outside[:3]}")
    todo = sorted(want - base)
    texts = s29_chunk_texts(pd.DataFrame(todo, columns=["query_id", "doc_id"])) if todo else {}
    added, skipped = [], []
    for q, d in todo:
        src, body = texts.get((q, d), (None, ""))
        if src is None or not body.strip():
            skipped.append([q, d])
            continue
        added.append({"query_id": q, "doc_id": d, "rank": int(sel.loc[(q, d), "rank"]), "p4b": float(sel.loc[(q, d), "p4b"]),
                      "source": src, "body": body})
    return {"removed": [[q, d] for q, d in sorted(base - want)], "added": added, "skipped_no_source": skipped}


def base_rows_from_docsel(docsel: pd.DataFrame, bodies: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    """Base bundle rows from a docsel: ``relevant_docs`` in ``rank`` order, one chunk on every pair of ``bodies``."""
    rows = []
    for qid, g in docsel.sort_values(["query_id", "rank"]).groupby("query_id", sort=True):
        docs = [int(d) for d in g.doc_id]
        rows.append({"id": int(qid), "relevant_docs": docs,
                     "relevant_chunks": [{"doc_id": d, "chunk_text": bodies[(int(qid), d)]} for d in docs if (int(qid), d) in bodies]})
    return rows


def virtual_base(out: Path, name: str, docsel: pd.DataFrame, carriers: pd.DataFrame | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """S37 mode (no S36 -docread ZIP): whole-body chunks of the carriers from the S29 sources, saved to ``<name>.base.parquet``."""
    target = docsel[docsel.chunk][["query_id", "doc_id"]] if carriers is None else carriers[["query_id", "doc_id"]]
    texts = s29_chunk_texts(target)
    bodies, skipped = {}, []
    for q, d in zip(target.query_id.astype(int), target.doc_id.astype(int)):
        src, body = texts.get((q, d), (None, ""))
        if src is None or not body.strip():
            skipped.append([q, d])
        else:
            bodies[(q, d)] = body
    frame = pd.DataFrame([(q, d, b) for (q, d), b in bodies.items()], columns=["query_id", "doc_id", "body"])
    path = out / f"{name}.base.parquet"
    _atomic_parquet(frame, path)
    return base_rows_from_docsel(docsel, bodies), {
        "base_parquet": str(path), "base_parquet_sha256": _sha256(path), "carriers": len(bodies), "skipped_no_source": len(skipped),
        "skipped_examples": skipped[:5]}


def load_base(prep: dict[str, Any]) -> list[dict[str, Any]]:
    """The base bundle of a prepared run: the ZIP, or the docs of its docsel with the chunks of ``<name>.base.parquet``."""
    if prep.get("base_zip"):
        if _sha256(Path(prep["base_zip"])) != prep["base_zip_sha256"]:
            raise ValueError("base ZIP changed since prepare")
        return read_bundle(Path(prep["base_zip"]))
    base, docsel_path = Path(prep["base_parquet"]), Path(prep["docsel"])
    if _sha256(base) != prep["base_parquet_sha256"] or _sha256(docsel_path) != prep["docsel_sha256"]:
        raise ValueError("base parquet or docsel changed since prepare")
    frame = pd.read_parquet(base)
    docsel = pd.read_parquet(docsel_path, columns=["query_id", "doc_id", "rank"])
    return base_rows_from_docsel(docsel, dict(zip(zip(frame.query_id.astype(int), frame.doc_id.astype(int)), frame.body)))


def doc_texts(doc_ids: set[int]) -> tuple[dict[int, str], dict[int, str], dict[int, int], set[int], dict[int, int]]:
    """(text by carrier doc, lang, owner, Pharmacity ids, text key): own canonical row, else the alias owner's
    (validator S26 rule); Pharmacity lot docs take their lot doc_text."""
    from r2ai.s29.build import _load_extra_docs
    from r2ai.s29.qwen_judge import _inputs

    canonical = paths.DATA_DIR / CANONICAL
    owners, cid_to_doc = load_owner_map(canonical)
    extra, _ = _load_extra_docs(_inputs())
    index = reg.registry_index()
    reg_docs = {d for d in doc_ids if d in index}
    reg_rows = reg.load_docs(index, reg_docs)
    reg_text = dict(zip(reg_rows.doc_id.astype(int), reg_rows.doc_text))
    reg_lang = dict(zip(reg_rows.doc_id.astype(int), reg_rows.lang))
    owner = {d: (d if d in reg_docs else owner_of(d, owners, cid_to_doc)) for d in doc_ids}
    rows = load_doc_rows(canonical, ((doc_ids - reg_docs) | set(owner.values())) - set(extra) - reg_docs)
    text = dict(zip(rows.doc_id.astype(int), rows.doc_text))
    lang = dict(zip(rows.doc_id.astype(int), rows.lang))
    out_text, out_lang, key = {}, {}, {}
    for d in doc_ids:
        if d in reg_docs:
            out_text[d], out_lang[d], key[d] = reg_text.get(d) or "", str(reg_lang.get(d, "zh")), d
            continue
        if d in extra:
            out_text[d], out_lang[d], key[d] = extra[d] or "", "vi", d
            continue
        k = d if d in text else owner[d]
        out_text[d], out_lang[d], key[d] = text.get(k) or "", str(lang.get(k, "vi")), k
    return out_text, out_lang, owner, set(extra) & doc_ids, key


def t256_chunks(owners: set[int], pharm: set[int]) -> pd.DataFrame:
    """Answer/body t256 chunks: canonical owners and Pharmacity lot docs; ``idx`` = file order among them."""
    from r2ai.s29.qwen_judge import _inputs
    from r2ai.s32.chunks import load_pharmacity_chunks
    from r2ai.s38.sources import load_chunks

    frames = []
    if owners:
        c = load_chunks(paths.DATA_DIR / CANONICAL, owners)
        c["origin"] = "c"
        frames.append(c)
    if pharm:
        p = load_pharmacity_chunks(_inputs())
        p = p[p.doc_id.isin(pharm)].copy()
        lots = {}
        for lot in sorted(x for x in (_inputs()["extra_docs"] / "lots").iterdir() if x.is_dir()):
            t = pq.read_table(lot / "chunks" / "chunks_t256.parquet", columns=["chunk_id", "char_start", "char_end"]).to_pandas()
            t["chunk_id"] = (1_000_000_000_000_000 + int(lot.name.rsplit("-", 1)[1]) * 1_000_000 + t.chunk_id.astype(np.int64))
            lots.update(dict(zip(t.chunk_id, zip(t.char_start, t.char_end))))
        p["char_start"] = [lots[c][0] for c in p.chunk_id]
        p["char_end"] = [lots[c][1] for c in p.chunk_id]
        p["vec_row"], p["file_row"], p["origin"] = -1, np.arange(len(p)), "p"
        frames.append(p)
    if not frames:
        return pd.DataFrame({"chunk_id": [], "doc_id": [], "field": [], "char_start": [], "char_end": [], "text": [], "file_row": []})
    return pd.concat(frames, ignore_index=True)


def best_t256(carriers: pd.DataFrame, text: dict[int, str], owner: dict[int, int], pharm: set[int]
              ) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[int, pd.DataFrame]]:
    """((query, doc) -> char span of the best answer/body t256 chunk in the carrier's text (CH-t256-1 rule),
    doc -> its answer/body t256 chunks when their spans index the carrier's text)."""
    docs = set(carriers.doc_id.astype(int))
    index = reg.registry_index()
    reg_docs = {d for d in docs if d in index}
    docs -= reg_docs
    own = {d: (d if d in pharm else owner[d]) for d in docs}
    chunks = t256_chunks({o for d, o in own.items() if d not in pharm}, pharm & docs)
    by_owner = {int(o): g.sort_values("file_row") for o, g in chunks.groupby("doc_id", sort=False)} if len(chunks) else {}
    pieces: dict[int, pd.DataFrame] = {}
    for d in docs:
        g = by_owner.get(own[d])
        if g is None:
            continue
        t = text[d]
        if not all(t[s:e] == x for s, e, x in zip(g.char_start.astype(int), g.char_end.astype(int), g.text)):
            continue                                            # spans do not index this text (sources rule)
        sub = g[g.field.isin(SUBMITTABLE)].reset_index(drop=True)
        if len(sub):
            pieces[d] = sub.assign(idx=np.arange(len(sub)))
    if not docs:                                              # only registry docs
        return reg.best_spans(carriers, text, index, choose)
    if reg.QUERY_SET is not None:                             # S40 new query set: the 0.6B scores of its own C-max chunk rows, its query vectors
        judge, qv = reg.QUERY_SET["chunk_scores"], reg.QUERY_SET["qvec"]
    else:
        cp = pd.read_parquet(paths.OUT_DIR / S32_ROOT / "pool" / "chunk_pairs.parquet",
                             columns=["pair_id", "query_id", "doc_id", "chunk_id", "field"])
        cp = cp.merge(carriers[["query_id", "doc_id"]], on=["query_id", "doc_id"])
        sc = pd.read_parquet(paths.OUT_DIR / S32_SCORES, columns=["pair_id", "score"])
        judge = cp[cp.field.isin(SUBMITTABLE)].merge(sc, on="pair_id")
        judge = {(int(q), int(d), int(c)): float(s) for q, d, c, s in zip(judge.query_id, judge.doc_id, judge.chunk_id, judge.score)}
        qv = load_query_vectors()
    dense = np.load(paths.DATA_DIR / CANONICAL / "vectors" / "dense.npy", mmap_mode="r")
    with np.load(paths.RUNS_DIR / "s32-chunk-judge" / "pool" / "supplement_vectors.npz") as sup:
        sup_row = {(str(o), int(c)): k for k, (o, c) in enumerate(zip(sup["origin"], sup["chunk_id"]))}
        sup_dense = np.array(sup["dense"])
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for q, d in zip(carriers.query_id.astype(int), carriers.doc_id.astype(int)):
        if d in reg_docs:
            continue
        g = pieces.get(d)
        if g is None:
            continue
        cids = g.chunk_id.astype(int).tolist()
        scores = [judge.get((q, d, c)) for c in cids]
        keep = [k for k, s in enumerate(scores) if s is not None]
        if keep:
            vals = [scores[k] for k in keep]
        else:
            vecs, keep = [], []
            for k, (c, vr, o) in enumerate(zip(cids, g.vec_row.astype(int), g.origin)):
                if vr >= 0:
                    vecs.append(dense[vr])
                elif (o, c) in sup_row:
                    vecs.append(sup_dense[sup_row[(o, c)]])
                else:
                    continue
                keep.append(k)
            if not keep:
                continue
            vals = (np.stack(vecs).astype(np.float32) @ qv[q].astype(np.float32)).tolist()
        pick = keep[choose(vals, [int(g.idx.iat[k]) for k in keep], 1)[0]]
        out[(q, d)] = (int(g.char_start.iat[pick]), int(g.char_end.iat[pick]))
    if reg_docs:
        reg_out, reg_pieces = reg.best_spans(carriers[carriers.doc_id.astype(int).isin(reg_docs)], text, index, choose)
        out.update(reg_out)
        pieces.update(reg_pieces)
    return out, pieces


# ------------------------------------------------------------------------------------------ prepare

def prepare(out: Path, name: str, base_zip: Path | None, docsel_path: Path | None, rule: str, base64: bool,
            skip_check: bool, thin: bool = True, carriers: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {"name": name, "rule": rule, "docsel": str(docsel_path) if docsel_path else None,
                              "base64_baophutho": base64 if rule.startswith("clean") else None, "thin": thin}
    if reg.QUERY_SET is not None:
        report["query_set"] = reg.QUERY_SET["run"]
    if NO_P1:
        report["no_p1"] = True
    if base_zip is not None:
        rows = read_bundle(base_zip)
        report.update({"base_zip": str(base_zip), "base_zip_bytes": base_zip.stat().st_size, "base_zip_sha256": _sha256(base_zip)})
    else:
        if docsel_path is None:
            raise ValueError("without --base-zip the base bundle is built from the docsel (--docsel)")
        docsel0 = pd.read_parquet(docsel_path, columns=["query_id", "doc_id", "rank", "chunk", "p4b"])
        target0 = pd.read_parquet(carriers, columns=["query_id", "doc_id"]) if carriers is not None else None
        rows, info = virtual_base(out, name, docsel0, target0)
        estimate = int(RATE_CARRIER * info["carriers"] + RATE_DOC * len(docsel0))
        report.update({"base_zip": None, "base_zip_bytes": estimate, "base_zip_sha256": None, **info,
                       "docsel_sha256": _sha256(docsel_path), "carriers": str(carriers) if carriers else None,
                       "carriers_sha256": _sha256(carriers) if carriers else None})
        carriers = None                              # already folded into the base bundle
        kept = {(int(r["id"]), int(c["doc_id"])) for r in rows for c in r["relevant_chunks"]}
        skip_check = True
    swap: dict[str, Any] = {"removed": [], "added": []}
    if carriers is not None and docsel_path is None:
        raise ValueError("--carriers needs the docsel")
    if docsel_path is not None:
        report["docsel_sha256"] = _sha256(docsel_path)
        docsel = pd.read_parquet(docsel_path, columns=["query_id", "doc_id", "rank", "chunk", "p4b"])
        if base_zip is None:
            docsel = docsel.assign(chunk=[(int(q), int(d)) in kept for q, d in zip(docsel.query_id, docsel.doc_id)])
        check = check_docsel(rows, docsel)
        report["docsel_check"] = check
        if not skip_check and (check["queries_docs_differ"] or check["carriers_only_base"] or check["carriers_only_docsel"]):
            raise ValueError(f"base bundle differs from the docsel: {check}")
        first: dict[str, Any] = {"removed": [], "added": []}
        carrier_info = None
        if carriers is not None:
            target = pd.read_parquet(carriers, columns=["query_id", "doc_id"])
            first = carrier_swap(rows, docsel, target)
            report["carriers"] = str(carriers)
            report["carriers_sha256"] = _sha256(carriers)
            carrier_info = {"target": len(target), "removed": len(first["removed"]), "added": len(first["added"]),
                            "skipped_no_source": len(first["skipped_no_source"])}
            rows = apply_swap(rows, first)
            kept = set(zip(target.query_id.astype(int), target.doc_id.astype(int))) - {tuple(x) for x in first["skipped_no_source"]}
            docsel = docsel.assign(chunk=[(int(q), int(d)) in kept for q, d in zip(docsel.query_id, docsel.doc_id)])
        second = thin_swap(rows, docsel) if thin else {"removed": [], "added": []}
        if carriers is None:
            swap = second
        else:
            swap = {**second, **compose_swaps(first, second), "thin_removed": second["removed"], "thin_added": second["added"],
                    "carriers": carrier_info}
            rows = read_bundle(base_zip)       # the swap is relative to the base bundle
            report["base_zip_bytes_effective"] = int(report["base_zip_bytes"] + RATE_CARRIER * (len(swap["added"]) - len(swap["removed"])))
            report["carrier_swap"] = {**carrier_info, "removed_total": len(swap["removed"]), "added_total": len(swap["added"])}
    elif thin:
        raise ValueError("thin needs the docsel (or pass --no-thin)")
    _atomic_json(out / f"{name}.swap.json", swap)
    rows = apply_swap(rows, swap)
    report["thin"] = {"removed": len(swap["removed"]), "added": len(swap["added"]),
                      "thin_docs": len(swap.get("thin_docs", [])), "t_c_before": swap.get("t_c_before"),
                      "t_c_after": swap.get("t_c_after")} if thin else None
    car = pd.DataFrame([(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]],
                       columns=["query_id", "doc_id", "body"])
    del rows
    text, lang, owner, pharm, key = doc_texts(set(car.doc_id.astype(int)))
    _log(f"carriers={len(car)} docs={car.doc_id.nunique()} pharmacity={len(pharm)}")
    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    out_rows: list[dict[str, Any]] = []
    stats = {"located": 0, "fallback_body": 0, "no_t256_choice": 0, "chunk_outside_body": 0, "not_verbatim_skipped": 0}
    markers = load_markers() if rule.startswith("clean") else {}
    host_of = host_map(set(key.values())) if rule.startswith("clean") else {}
    cut_by_host: dict[str, int] = {}

    def cleaned(d: int, body: str) -> str | None:
        host = host_of.get(key[d])
        b64 = base64 and host in cl.BASE64_HOSTS
        if host not in markers and not b64:
            return None
        return cl.clean_cut(body, markers.get(host, set()), b64)

    if rule == "clean":
        for q, d, body in zip(car.query_id.astype(int), car.doc_id.astype(int), car.body):
            t = cleaned(d, body)
            if t is not None and t not in text[d]:
                stats["not_verbatim_skipped"] += 1
                t = None
            if t is not None:
                host = host_of.get(key[d])
                cut_by_host[host] = cut_by_host.get(host, 0) + 1
            out_rows.append({"query_id": q, "doc_id": d, "rank": 1, "text": t, "mode": "clean" if t is not None else "base",
                             "body_tokens": -1})
        report["carriers_cut_by_host"] = cut_by_host
    else:
        spec = RULES[rule]
        best, pieces = best_t256(car[["query_id", "doc_id"]], text, owner, pharm)
        empty = pd.DataFrame({"field": [], "char_start": [], "char_end": []})
        _log(f"best t256 chunks={len(best)}")
        for begin in range(0, len(car), 5_000):
            part = car.iloc[begin:begin + 5_000]
            bodies, base, cut = [], [], []
            for d, body in zip(part.doc_id.astype(int), part.body):
                t = text[d]
                at = t.find(body)
                c = cleaned(d, body) if rule == "clean+cap2048" else None
                if c is not None:                                    # clean first: the cut body is the body
                    at_c = at if at >= 0 else t.find(c)
                    if at_c >= 0:
                        stats["located"] += 1
                        bodies.append(c)
                        base.append((at_c, True))
                        cut.append(True)
                        host = host_of.get(key[d])
                        cut_by_host[host] = cut_by_host.get(host, 0) + 1
                        continue
                    stats["not_verbatim_skipped"] += 1
                cut.append(False)
                if at >= 0:
                    stats["located"] += 1
                    bodies.append(body)
                    base.append((at, True))
                else:
                    stats["fallback_body"] += 1
                    s, e = _body_span(t, lang[d], pieces.get(d, empty))       # same rule as r2ai.s38.window
                    bodies.append(t[s:e])
                    base.append((s, False))
            enc = tok(bodies, add_special_tokens=False, return_offsets_mapping=True)
            for k, (q, d, body, (b0, located), was_cut) in enumerate(zip(part.query_id.astype(int), part.doc_id.astype(int),
                                                                          bodies, base, cut)):
                offsets = enc["offset_mapping"][k]
                chunk = None
                span = best.get((q, d))
                if span is None:
                    stats["no_t256_choice"] += 1
                else:
                    cs, ce = span[0] - b0, span[1] - b0
                    if ce <= 0 or cs >= len(body):
                        stats["chunk_outside_body"] += 1
                    else:
                        chunk = (max(cs, 0), min(ce, len(body)))
                for rank, piece in enumerate(pieces_for(*spec, body, offsets, chunk), 1):
                    if was_cut and (piece is None or piece[0] == "whole"):
                        out_rows.append({"query_id": q, "doc_id": d, "rank": rank, "text": body, "mode": "clean",
                                         "body_tokens": len(offsets)})
                    elif piece is None or (piece[0] == "whole" and located):
                        out_rows.append({"query_id": q, "doc_id": d, "rank": rank, "text": None, "mode": "base",
                                         "body_tokens": len(offsets)})
                    else:
                        out_rows.append({"query_id": q, "doc_id": d, "rank": rank, "text": body[piece[1]:piece[2]],
                                         "mode": piece[0], "body_tokens": len(offsets)})
            _log(f"carriers {begin + len(part)}/{len(car)}")
        if rule == "clean+cap2048":
            report["carriers_cut_by_host"] = cut_by_host
    frame = pd.DataFrame(out_rows)
    target = out / f"{name}.cache.parquet"
    _atomic_parquet(frame, target)
    changed = frame[frame["mode"] != "base"]
    report.update({"carriers": len(car), "chunks": len(frame), "carriers_changed": int(changed[["query_id", "doc_id"]].drop_duplicates().shape[0]),
                   **stats, "cache": str(target), "seconds": round(time.monotonic() - started, 1)})
    if rule != "clean":
        bt = frame.drop_duplicates(["query_id", "doc_id"]).body_tokens.to_numpy()
        report["body_tokens"] = describe(bt)
    _atomic_json(out / f"{name}.prepare.json", report)
    return report


# -------------------------------------------------------------------------------------------- write

def label_of(name: str, drop: int, threshold: int | None) -> str:
    return f"{name}{f'-d{drop}' if drop else ''}{f'-t{threshold}' if threshold is not None else ''}"


def lowest_p_carriers(prep: dict[str, Any], frame: pd.DataFrame, drop: int) -> set[tuple[int, int]]:
    """The ``drop`` carriers with the lowest P4B (ties: later query, later rank first), from the docsel of the prepared run."""
    if not drop:
        return set()
    docsel = pd.read_parquet(prep["docsel"], columns=["query_id", "doc_id", "rank", "p4b"])
    car = frame[frame["rank"] == 1][["query_id", "doc_id"]].drop_duplicates().merge(docsel, on=["query_id", "doc_id"], how="left")
    if car.p4b.isna().any():
        raise ValueError("carriers outside the docsel")
    worst = car.sort_values(["p4b", "query_id", "rank"], ascending=[True, False, False], kind="stable").head(drop)
    return set(zip(worst.query_id.astype(int), worst.doc_id.astype(int)))


def without(frame: pd.DataFrame, dropped: set[tuple[int, int]]) -> pd.DataFrame:
    if not dropped:
        return frame
    keys = pd.MultiIndex.from_arrays([frame.query_id.astype(int), frame.doc_id.astype(int)])
    return frame[~keys.isin(list(dropped))]


def run_validator(out: Path, label: str, rule: str, target: Path, doc_ids) -> dict[str, Any]:
    """S26 validator on ``target`` (report in ``<label>.validate.json``); the 120ask segments of the registry docs among ``doc_ids`` are extra sources."""
    cmd = [sys.executable, "-u", str(paths.ROOT / VALIDATOR)]
    if rule == "full+w768":
        cmd += ["--max-chunks-per-doc", "2"]
    if reg.QUERY_SET is not None:                             # S40 new query set: its own query ids, P1 is no verbatim source
        cmd += ["--queries", reg.QUERY_SET["queries"], "--no-p1"]
    elif NO_P1:
        cmd += ["--no-p1"]
    cmd += [str(target), str(paths.OUT_DIR / PHARM_HANDOFF)]
    extra = reg.validator_extra(out / f"{label}.validator-extra", reg.registry_index(), doc_ids)
    if extra is not None:
        cmd.append(str(extra))
    res_v = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    (out / f"{label}.validate.json").write_text(res_v.stdout, encoding="utf-8")
    try:
        v = json.loads(res_v.stdout)
        return {"ok": v["ok"], "errors": {k: n for k, n in v["errors"].items() if n}, "chunk_check": v["chunk_check"],
                "extra_sources": sorted(p.parent.parent.name for p in extra.glob("lots/*/chunks/docs.parquet")) if extra else []}
    except (ValueError, KeyError):
        return {"ok": False, "stderr_tail": res_v.stderr[-2000:]}


def validate(out: Path, name: str, threshold: int | None = None, drop: int = 0) -> dict[str, Any]:
    """Re-run the validator on the ZIP already written for ``name`` (same label rule as ``write``) and record it in its compose.json."""
    prep = json.loads((out / f"{name}.prepare.json").read_text(encoding="utf-8"))
    rule = prep["rule"]
    if rule == "full+w768":
        threshold = threshold if threshold is not None else FULL_MIN_T
    label = label_of(name, drop, threshold if rule == "full+w768" else None)
    target, entry_path = out / f"{label}.zip", out / f"{label}.compose.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    if entry.get("zip_sha256") != _sha256(target):
        raise ValueError(f"{target.name} differs from the ZIP its compose.json describes")
    docsel = prep.get("docsel")
    ids = set(pd.read_parquet(docsel, columns=["doc_id"]).doc_id.astype(int)) if docsel and Path(docsel).exists() else set(reg.registry_index())
    entry["validator"] = run_validator(out, label, rule, target, ids)
    _atomic_json(entry_path, entry)
    return entry["validator"]


def write(out: Path, name: str, threshold: int | None, validate: bool = True, reserve: int = RESERVE, drop: int = 0) -> dict[str, Any]:
    started = time.monotonic()
    prep = json.loads((out / f"{name}.prepare.json").read_text(encoding="utf-8"))
    rule = prep["rule"]
    frame = pd.read_parquet(out / f"{name}.cache.parquet")
    dropped = lowest_p_carriers(prep, frame, drop)
    frame = without(frame, dropped)
    label = label_of(name, drop, None)
    if rule == "full+w768":
        threshold = threshold if threshold is not None else FULL_MIN_T
        if threshold < FULL_MIN_T:
            raise ValueError(f"threshold below {FULL_MIN_T}: windows not prepared")
        frame = frame[(frame["rank"] == 1) | (frame.body_tokens > threshold)]
        label = label_of(name, drop, threshold)
    chosen: dict[tuple[int, int], list] = {}
    for q, d, t in zip(frame.query_id.astype(int), frame.doc_id.astype(int), frame.text):
        chosen.setdefault((q, d), []).append(None if t is None or t != t else t)
    rows = load_base(prep)
    swap_path = out / f"{name}.swap.json"
    rows = apply_swap(rows, json.loads(swap_path.read_text(encoding="utf-8"))) if swap_path.exists() else rows
    new_rows = [replace_chunks(r, chosen) for r in rows]
    if dropped:
        for r in new_rows:
            r["relevant_chunks"] = [c for c in r["relevant_chunks"] if (int(r["id"]), int(c["doc_id"])) not in dropped]
    target = out / f"{label}.zip"
    paths.assert_writable(target)
    paths.assert_writable(target.with_name(target.name + ".tmp"))
    res = write_bundle(new_rows, target, f"{label}.json")
    entry = {"name": label, "rule": rule, "threshold": threshold, "dropped_carriers": drop, "zip": str(target), "zip_bytes": target.stat().st_size,
             "zip_sha256": _sha256(target), **res, "limit_bytes": BTC_CEILING - reserve, "under_ceiling": target.stat().st_size < BTC_CEILING - reserve,
             "docs": sum(len(r["relevant_docs"]) for r in new_rows),
             "chunks": sum(len(r["relevant_chunks"]) for r in new_rows),
             "second_chunks": int((frame["rank"] == 2).sum()),
             "carriers_changed": sum(any(t is not None for t in v) for v in chosen.values()),
             "docs_equal_base": all(a["relevant_docs"] == b["relevant_docs"] for a, b in zip(rows, new_rows))}
    if validate:
        entry["validator"] = run_validator(out, label, rule, target, {int(d) for r in new_rows for d in r["relevant_docs"]})
    entry["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / f"{label}.compose.json", entry)
    return entry


def second_chunk_tokens(frame: pd.DataFrame) -> np.ndarray:
    """Ascending body tokens of the carriers that have a window (rank 2): second chunks at T = #(bt > T)."""
    return np.sort(frame[frame["rank"] == 2].body_tokens.to_numpy())


def t_for_count(bt: np.ndarray, count: int) -> int:
    """Lowest T (>= FULL_MIN_T) that leaves about ``count`` windows."""
    if count >= len(bt):
        return FULL_MIN_T
    return max(int(bt[len(bt) - max(count, 0) - 1]), FULL_MIN_T)


def estimate_thresholds(frame: pd.DataFrame, base_zip_bytes: int, reserve: int = RESERVE) -> dict[str, Any]:
    """T so that (number of second chunks) x RATE_FULL fits under ``BTC_CEILING - reserve``, and two neighbours."""
    bt = second_chunk_tokens(frame)
    budget = BTC_CEILING - reserve - base_zip_bytes
    k = int(max(budget, 0) / RATE_FULL)
    return {"second_chunks_at_min_t": int(len(bt)), "budget_bytes": int(budget), "k_estimate": k,
            "t_estimate": t_for_count(bt, k),
            "candidates": sorted({t_for_count(bt, int(k * 1.03)), t_for_count(bt, k), t_for_count(bt, int(k * 0.97))})}


def reusable(out: Path, name: str, t: int, drop: int = 0, validated: bool = True) -> dict[str, Any] | None:
    """The measurement of an earlier ``write`` at T when its ZIP is still there, unchanged, and newer than the prepared cache."""
    label = label_of(name, drop, t)
    meta, zip_path, prepared = out / f"{label}.compose.json", out / f"{label}.zip", out / f"{name}.prepare.json"
    if not (meta.exists() and zip_path.exists() and prepared.exists()) or meta.stat().st_mtime < prepared.stat().st_mtime:
        return None
    entry = json.loads(meta.read_text(encoding="utf-8"))
    return entry if entry.get("zip_sha256") == _sha256(zip_path) and (entry.get("validator") or not validated) else None


def fit_carriers(out: Path, name: str, base_bytes: int, reserve: int, rounds: int, n_carriers: int, step: int = 25) -> dict[str, Any]:
    """Smallest number of lowest-P4B carriers to drop so that the ZIP without any window is under ``BTC_CEILING - reserve``. Probe builds
    run in parallel (no validator); the next round interpolates between the largest drop still over and the smallest under."""
    limit = BTC_CEILING - reserve
    tried: dict[int, int] = {}                                    # drop -> ZIP bytes without windows
    n0 = max(0, math.ceil((base_bytes - limit) / RATE_CARRIER))
    clip = lambda n: int(min(max(n, 0), n_carriers - 1))           # noqa: E731
    candidates = sorted({clip(n0 * 0.97), clip(n0), clip(n0 * 1.03 + 1)})
    for r in range(rounds):
        todo = []
        for n in candidates:
            old = reusable(out, name, NO_WINDOWS, n, validated=False)
            if n in tried:
                continue
            if old is not None:
                tried[n] = old["zip_bytes"]
                _log(f"fit drop={n} reused zip={old['zip_bytes']}")
            else:
                todo.append(n)
        _log(f"fit round {r + 1}: drops {todo}")
        procs = [(n, subprocess.Popen([sys.executable, "-m", "r2ai.s38.compose", "write", "--name", name, "--out", str(out), "--threshold", str(NO_WINDOWS),
                                       "--drop", str(n), "--reserve", str(reserve), "--no-validate"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
                 for n in todo]
        for n, p in procs:
            _, err = p.communicate()
            if p.returncode:
                raise RuntimeError(f"fit write drop={n} failed: {err.decode('utf-8', 'replace')[-2000:]}")
            tried[n] = json.loads((out / f"{label_of(name, n, NO_WINDOWS)}.compose.json").read_text(encoding="utf-8"))["zip_bytes"]
            _log(f"fit drop={n} zip={tried[n]} under={tried[n] < limit}")
        under = sorted(n for n, z in tried.items() if z < limit)
        over = sorted(n for n, z in tried.items() if z >= limit)
        if not under:
            hi = max(over)
            candidates = [clip(hi + (tried[hi] - limit) / RATE_CARRIER * f + step) for f in (1.0, 1.03, 1.1)]
            continue
        lo_ok = under[0]
        below = [n for n in over if n < lo_ok]
        if not below:
            if lo_ok == 0:
                break
            candidates = [clip(lo_ok - (limit - tried[lo_ok]) / RATE_CARRIER * f) for f in (0.9, 1.0, 1.05)]
            candidates = sorted({n for n in candidates if n < lo_ok})
            if not candidates:
                break
            continue
        hi_bad = max(below)
        if lo_ok - hi_bad <= step:
            break
        share = (tried[hi_bad] - limit) / max(tried[hi_bad] - tried[lo_ok], 1)
        mid = hi_bad + share * (lo_ok - hi_bad)
        candidates = sorted({n for n in (int(mid) - step, int(mid), int(mid) + step) if hi_bad < n < lo_ok})
        if not candidates:
            break
    under = sorted(n for n, z in tried.items() if z < limit)
    if not under:
        raise RuntimeError(f"no drop under the ceiling within {rounds} rounds: {tried}")
    best = under[0]
    return {"drop": best, "zip_bytes_without_windows": tried[best], "step": step, "tried": {str(n): z for n, z in sorted(tried.items())},
            "highest_drop_over": max([n for n, z in tried.items() if z >= limit and n < best], default=None)}


def run_full(out: Path, name: str, candidates: list[int], rounds: int, reserve: int = RESERVE,
             bt: np.ndarray | None = None, drop: int = 0) -> dict[str, Any]:
    """Build candidate thresholds in parallel processes, bisect between over/under, keep the lowest T under. A candidate already
    measured by an earlier run on the same prepared cache is reused. ``bt`` (``second_chunk_tokens``) lets a round without any T
    over the ceiling re-estimate T from the measured slack instead of stepping blindly."""
    tried: dict[int, dict[str, Any]] = {}
    for r in range(rounds):
        todo = []
        for t in candidates:
            if t in tried:
                continue
            old = reusable(out, name, t, drop)
            if old is not None:
                tried[t] = old
                _log(f"t={t} reused zip={old['zip_bytes']} under={old['under_ceiling']}")
            else:
                todo.append(t)
        _log(f"round {r + 1}: thresholds {todo}")
        procs = [(t, subprocess.Popen([sys.executable, "-m", "r2ai.s38.compose", "write", "--name", name, "--out", str(out),
                                       "--threshold", str(t), "--reserve", str(reserve)] + (["--drop", str(drop)] if drop else []),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
                 for t in todo]
        for t, p in procs:
            _, err = p.communicate()
            if p.returncode:
                raise RuntimeError(f"write t={t} failed: {err.decode('utf-8', 'replace')[-2000:]}")
            tried[t] = json.loads((out / f"{label_of(name, drop, t)}.compose.json").read_text(encoding="utf-8"))
            _log(f"t={t} zip={tried[t]['zip_bytes']} under={tried[t]['under_ceiling']}")
        under = sorted(t for t, e in tried.items() if e["under_ceiling"])
        over = sorted(t for t, e in tried.items() if not e["under_ceiling"])
        if not under:
            hi = max(over)
            candidates = [hi + 25 * 4 ** r, hi + 50 * 4 ** r, hi + 100 * 4 ** r]
            continue
        lo_ok = under[0]
        below = [t for t in over if t < lo_ok]
        if not below:
            candidates = [max(FULL_MIN_T, lo_ok - 25 * 4 ** r), max(FULL_MIN_T, lo_ok - 10 * 4 ** r)] if lo_ok > FULL_MIN_T else []
            if bt is not None and lo_ok > FULL_MIN_T:
                e = tried[lo_ok]
                k = int(e["second_chunks"] + max(e["limit_bytes"] - e["zip_bytes"], 0) / RATE_FULL)   # windows the measured slack still buys
                candidates = sorted({t_for_count(bt, int(k * 1.02)), t_for_count(bt, k), t_for_count(bt, int(k * 0.98))})
            candidates = [t for t in candidates if t < lo_ok]
            if not candidates:
                break
            continue
        hi_bad = max(below)
        if lo_ok - hi_bad <= 1:
            break
        step = max(1, (lo_ok - hi_bad) // 3)
        candidates = sorted({hi_bad + step, hi_bad + 2 * step, lo_ok - 1} - {hi_bad, lo_ok})
    under = sorted(t for t, e in tried.items() if e["under_ceiling"])
    over = sorted(t for t, e in tried.items() if not e["under_ceiling"])
    best = under[0] if under else None
    return {"tried": {str(t): {k: e[k] for k in ("zip_bytes", "under_ceiling", "second_chunks", "chunks")} | {
                "validator_ok": e.get("validator", {}).get("ok")} for t, e in sorted(tried.items())},
            "lowest_t_under_ceiling": best, "highest_t_over_ceiling_below_it": max([t for t in over if best and t < best], default=None),
            "exact_to_one_token": bool(best and any(t == best - 1 for t in over))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "write", "run", "estimate", "validate"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=None, help="default RUNS_DIR/s38-chunk-metric-v2/compose")
    parser.add_argument("--base-zip", type=Path)
    parser.add_argument("--docsel", type=Path)
    parser.add_argument("--rule", choices=list(RULES))
    parser.add_argument("--no-base64", action="store_true", help="clean: no base64 cut on baophutho.vn")
    parser.add_argument("--skip-docsel-check", action="store_true")
    parser.add_argument("--carriers", type=Path, default=None, help="parquet (query_id, doc_id): the chunk carriers (default: those of the base bundle)")
    parser.add_argument("--no-thin", action="store_true", help="keep the thin (base64 / placeholder) carriers")
    parser.add_argument("--threshold", type=int, action="append", default=None, help="full+w768: body tokens above which the window is added")
    parser.add_argument("--reserve", type=int, default=RESERVE, help="bytes kept free below the BTC ceiling 104,857,600 (default 2,257,600 -> limit 102,600,000)")
    parser.add_argument("--rounds", type=int, default=4, help="full+w768 run: parallel build rounds")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--drop", type=int, default=0, help="write: drop this many lowest-P4B carriers (their chunks, not the docs)")
    parser.add_argument("--fit-carriers", action="store_true", help="run: drop the fewest lowest-P4B carriers that let the ZIP fit, then search T")
    parser.add_argument("--fit-step", type=int, default=25, help="fit-carriers: resolution in carriers")
    parser.add_argument("--query-set", type=Path, default=None,
                        help="S40: run dir of r2ai.newq (a new query set; recorded in prepare.json, later commands reuse it)")
    parser.add_argument("--no-p1", action="store_true", help="S40: no P1 chunk text (recorded in prepare.json, later commands reuse it)")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or run_root() / "compose")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    global NO_P1
    query_set, NO_P1 = args.query_set, args.no_p1
    if (out / f"{args.name}.prepare.json").exists():
        prep0 = json.loads((out / f"{args.name}.prepare.json").read_text(encoding="utf-8"))
        query_set, NO_P1 = query_set or prep0.get("query_set"), NO_P1 or bool(prep0.get("no_p1"))
    if query_set:
        reg.use_query_set(paths.resolve_path(query_set))
    if args.command in ("prepare", "run") and not (out / f"{args.name}.prepare.json").exists():
        if not (args.rule and (args.base_zip or args.docsel)):
            parser.error("prepare needs --rule and --base-zip (or --docsel alone: the base bundle is built from the docsel)")
        rep = prepare(out, args.name, paths.resolve_path(args.base_zip) if args.base_zip else None,
                      paths.resolve_path(args.docsel) if args.docsel else None, args.rule, not args.no_base64,
                      args.skip_docsel_check, not args.no_thin, paths.resolve_path(args.carriers) if args.carriers else None)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.command == "prepare":
        return 0
    if args.command == "validate":
        res = validate(out, args.name, (args.threshold or [None])[0], args.drop)
        print(json.dumps(res, indent=1))
        return 0 if res.get("ok") else 1
    prep = json.loads((out / f"{args.name}.prepare.json").read_text(encoding="utf-8"))
    if args.command == "estimate" or (args.command == "run" and prep["rule"] == "full+w768" and not args.threshold):
        cache = pd.read_parquet(out / f"{args.name}.cache.parquet")
        base_bytes, drop, fit = prep.get("base_zip_bytes_effective", prep["base_zip_bytes"]), args.drop, None
        if args.command == "run" and args.fit_carriers and not args.drop:
            fit = fit_carriers(out, args.name, base_bytes, args.reserve, args.rounds, int((cache["rank"] == 1).sum()), args.fit_step)
            drop, base_bytes = fit["drop"], fit["zip_bytes_without_windows"]
            _log(f"fit-carriers: drop {drop} lowest-P4B carriers, ZIP without windows {base_bytes}")
        cache = without(cache, lowest_p_carriers(prep, cache, drop))
        est = estimate_thresholds(cache, base_bytes, args.reserve)
        print(json.dumps(est, indent=1))
        if args.command == "estimate":
            return 0
        res = run_full(out, args.name, est["candidates"], args.rounds, args.reserve, second_chunk_tokens(cache), drop)
        res["drop"], res["fit_carriers"] = drop, fit
        res["reserve_bytes"] = args.reserve
        res["estimate"] = est
        _atomic_json(out / f"{label_of(args.name, drop, None)}.threshold.json", res)
        print(json.dumps(res, indent=1))
        return 0
    for t in (args.threshold or [None]):
        print(json.dumps(write(out, args.name, t, not args.no_validate, args.reserve, args.drop), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
