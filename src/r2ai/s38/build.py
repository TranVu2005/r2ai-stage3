"""Build the S38 CH-* bundles: J3-thrC rows with each whole-body chunk replaced by the chosen pieces.

``python -m r2ai.s38.build [--variant CH-t256-1 ...]`` -> ``out/runs/s38-chunk-metric-v2/build/``

* ``relevant_docs`` and the chunk-carrying docs are J3-thrC's, row by row; only ``chunk_text`` changes. A carrier
  with two chosen pieces (CH-t256-2) gets two chunks of its doc_id, in score order, at the place of its J3 chunk.
* writer = the S29 one (one compact JSON row per line, ZIP_LZMA dict 256 MiB, fixed member date).
* ``CH-win384`` / ``CH-win768`` take their texts from ``scores/windows.parquet`` (``r2ai.s38.window``).
* ``CH-full`` is J3-thrC itself (no rebuild); ``--lengths`` adds its chunk lengths to the report.
* per variant: chunks, chunk length (chars, raw BGE-M3 tokens, metric tokens = BGE-M3 ids of the
  metric-normalised text), ZIP bytes / SHA256, JSON SHA256. Run validator S26 on every ZIP afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.compose.base_io import lzma_dictionary
from r2ai.s38.score import VARIANTS, scores_dir
from r2ai.s38.window import SPECS
from r2ai.s38.sources import (J3_JSON_SHA256, J3_ZIP, J3_ZIP_SHA256, _atomic_json, _sha256, describe, read_bundle,
                              run_root)

CEILING = 102_600_000
WINDOW_VARIANTS = tuple(SPECS)
ALL_VARIANTS = (*WINDOW_VARIANTS, *VARIANTS)
MEMBER_DATE = (2026, 10, 23, 0, 0, 0)


def replace_chunks(row: dict[str, Any], chosen: dict[tuple[int, int], list[str]]) -> dict[str, Any]:
    """J3 row with every chunk replaced by the chosen texts of its (query, doc); ``None`` in the list, or a carrier
    without choice (no submittable piece), keeps its J3 chunk, so the chunk-carrying docs stay those of J3-thrC."""
    qid = int(row["id"])
    chunks = [{"doc_id": int(c["doc_id"]), "chunk_text": t}
              for c in row["relevant_chunks"]
              for t in (x if x is not None else c["chunk_text"] for x in chosen.get((qid, int(c["doc_id"])), [None]))]
    return {"id": qid, "relevant_docs": [int(d) for d in row["relevant_docs"]], "relevant_chunks": chunks}


def write_bundle(rows: Sequence[dict[str, Any]], target: Path, member: str) -> dict[str, Any]:
    temp = target.with_name(target.name + ".tmp")
    digest = hashlib.sha256()
    n = 0
    with lzma_dictionary(256), zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_LZMA) as archive:
        info = zipfile.ZipInfo(member, date_time=MEMBER_DATE)
        info.compress_type = zipfile.ZIP_LZMA
        with archive.open(info, "w", force_zip64=True) as stream:
            for k, row in enumerate(rows):
                data = (b"[\n" if k == 0 else b",\n") + json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                stream.write(data)
                digest.update(data)
                n += len(data)
            stream.write(b"\n]\n")
            digest.update(b"\n]\n")
            n += 3
    temp.replace(target)
    return {"json_bytes": n, "json_sha256": digest.hexdigest()}


def length_report(texts: list[str], tok) -> dict[str, Any]:
    from r2ai.eval.scorer import normalize_metric

    raw, met = [], []
    for begin in range(0, len(texts), 20_000):
        part = texts[begin:begin + 20_000]
        raw += [len(x) for x in tok(part, add_special_tokens=False)["input_ids"]]
        norm = [normalize_metric(t) for t in part]
        met += [len(x) for x in tok(norm, add_special_tokens=False)["input_ids"]]
    return {"chars": describe([len(t) for t in texts]), "tokens_raw": describe(raw), "tokens_metric": describe(met)}


def build(out: Path, variants: Sequence[str], lengths_full: bool) -> dict[str, Any]:
    started = time.monotonic()
    zip_path = paths.OUT_DIR / J3_ZIP
    if _sha256(zip_path) != J3_ZIP_SHA256:
        raise ValueError(f"{zip_path}: ZIP SHA256 differs from J3-thrC")
    rows = read_bundle(zip_path, J3_JSON_SHA256)
    sel = pd.read_parquet(scores_dir() / "selection.parquet")
    pieces = pd.read_parquet(run_root() / "sources" / "pieces.parquet", columns=["text"])
    windows = None
    if any(v in WINDOW_VARIANTS for v in variants):
        windows = pd.read_parquet(scores_dir() / "windows.parquet")
        windows = windows[windows.variant.isin(variants)]
        windows["rank"] = windows["rank"].fillna(1) if "rank" in windows else 1
        windows = windows.sort_values(["variant", "query_id", "doc_id", "rank"], kind="stable")
    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    report: dict[str, Any] = {"j3_zip_sha256": J3_ZIP_SHA256, "j3_json_sha256": J3_JSON_SHA256, "variants": {}}
    if lengths_full:
        texts = [c["chunk_text"] for r in rows for c in r["relevant_chunks"]]
        report["variants"]["CH-full"] = {"zip": str(zip_path), "zip_bytes": zip_path.stat().st_size,
                                         "zip_sha256": J3_ZIP_SHA256, "json_sha256": J3_JSON_SHA256,
                                         "docs": sum(len(r["relevant_docs"]) for r in rows), "chunks": len(texts),
                                         **length_report(texts, tok)}
        del texts
    for name in variants:
        chosen: dict[tuple[int, int], list[str]] = {}
        if name in WINDOW_VARIANTS:
            win = windows[windows.variant == name]
            for qid, did, t in zip(win.query_id.astype(int), win.doc_id.astype(int), win.text):
                chosen.setdefault((qid, did), []).append(None if t is None or t != t else t)
        else:
            pick = sel[sel.variant == name].sort_values(["query_id", "doc_id", "rank"])
            text = pieces.text.to_numpy()
            for qid, did, row in zip(pick.query_id.astype(int), pick.doc_id.astype(int), pick.piece_row.astype(int)):
                chosen.setdefault((qid, did), []).append(text[row])
        if not chosen:
            raise ValueError(f"{name}: no chosen chunk (run score select / window first)")
        new_rows = [replace_chunks(r, chosen) for r in rows]
        target = out / f"{name}.zip"
        paths.assert_writable(target)
        paths.assert_writable(target.with_name(target.name + ".tmp"))
        res = write_bundle(new_rows, target, f"{name}.json")
        texts = [c["chunk_text"] for r in new_rows for c in r["relevant_chunks"]]
        carriers = sum(len(r["relevant_chunks"]) for r in rows)
        entry = {"zip": str(target), "zip_bytes": target.stat().st_size, "zip_sha256": _sha256(target), **res,
                 "docs": sum(len(r["relevant_docs"]) for r in new_rows), "chunks": len(texts),
                 "carriers_j3": carriers, "carriers_kept_j3_chunk": carriers - len(chosen),
                 "carriers_changed": sum(any(t is not None for t in v) for v in chosen.values()),
                 "chunks_per_query": describe([len(r["relevant_chunks"]) for r in new_rows]),
                 "docs_equal_j3": all(a["relevant_docs"] == b["relevant_docs"] for a, b in zip(rows, new_rows)),
                 "under_ceiling": target.stat().st_size < CEILING}
        if name in WINDOW_VARIANTS:     # token counts from r2ai.s38.window (raw BGE-M3), no re-tokenisation
            entry.update({"chars": describe([len(t) for t in texts]), "tokens_raw": describe(win.tokens)})
        else:
            entry.update(length_report(texts, tok))
        report["variants"][name] = entry
        print(json.dumps({name: {k: entry[k] for k in ("chunks", "zip_bytes", "zip_sha256")}}), flush=True)
    report["seconds"] = round(time.monotonic() - started, 1)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", choices=list(ALL_VARIANTS), default=None)
    parser.add_argument("--lengths", action="store_true", help="also report the chunk lengths of CH-full (J3-thrC)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or run_root() / "build")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    paths.assert_writable(out / "build.json")
    paths.assert_writable(out / "build.json.tmp")
    out.mkdir(parents=True, exist_ok=True)
    report = build(out, args.variant or list(ALL_VARIANTS), args.lengths)
    previous = json.loads((out / "build.json").read_text(encoding="utf-8")) if (out / "build.json").exists() else {}
    previous.setdefault("variants", {}).update(report.pop("variants"))
    previous.update(report)
    _atomic_json(out / "build.json", previous)
    print(json.dumps({k: {x: v.get(x) for x in ("chunks", "zip_bytes")} for k, v in previous["variants"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
