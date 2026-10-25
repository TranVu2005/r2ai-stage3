"""Compose an S31 bundle from the doc lists of one ZIP and the chunks of another (CPU).

``python -m r2ai.s31.mix --docs-from A.zip --chunks-from B.zip --name N`` writes, per
query, ``relevant_docs`` of A and the ``relevant_chunks`` of B whose doc_id is in A's
docs, in B's order and with B's text unchanged (no new chunk, no cut). Used to measure
the doc side alone:

* K4-thr-C-extD = docs of K4-thr-C-ext, chunks of K4-thr-C (every chunk doc is kept);
* K4-free-D = docs of K4-free, chunks of K4-thr-C on docs K4-free still selects.

Same JSON layout and ZIP settings as ``r2ai.s29.build`` (compact rows, one per line,
LZMA dict 256 MiB). Fails if the ZIP reaches the 102,600,000-byte ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Sequence

from r2ai import paths
from r2ai.compose.base_io import lzma_dictionary
from r2ai.s29.qwen_judge import _atomic_json, _sha256

CEILING = 102_600_000


def read_rows(zip_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected one JSON file")
        return json.loads(archive.read(names[0]))


def mix_rows(docs_rows: list[dict[str, Any]], chunk_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks_by_q = {int(r["id"]): r["relevant_chunks"] for r in chunk_rows}
    if set(chunks_by_q) != {int(r["id"]) for r in docs_rows}:
        raise ValueError("The two bundles cover different queries")
    rows, dropped, changed = [], 0, 0
    for row in docs_rows:
        qid = int(row["id"])
        docs = set(int(d) for d in row["relevant_docs"])
        kept = [c for c in chunks_by_q[qid] if int(c["doc_id"]) in docs]
        dropped += len(chunks_by_q[qid]) - len(kept)
        changed += len(kept) != len(chunks_by_q[qid])
        rows.append({"id": qid, "relevant_docs": [int(d) for d in row["relevant_docs"]], "relevant_chunks": kept})
    stats = {"queries": len(rows), "docs_total": sum(len(r["relevant_docs"]) for r in rows),
             "chunks_total": sum(len(r["relevant_chunks"]) for r in rows),
             "chunks_dropped_doc_not_selected": dropped, "queries_chunks_differ_from_chunk_source": changed}
    return rows, stats


def write_zip(out: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    target, temp = out / f"{name}.zip", out / f"{name}.{os.getpid()}.zip.tmp"
    for path in (target, temp):
        paths.assert_writable(path)
    digest = hashlib.sha256()
    with lzma_dictionary(256), zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_LZMA) as archive:
        info = zipfile.ZipInfo(f"{name}.json", date_time=(2026, 10, 19, 0, 0, 0))
        info.compress_type = zipfile.ZIP_LZMA
        with archive.open(info, "w", force_zip64=True) as stream:
            for index, row in enumerate(rows):
                data = (b"[\n" if index == 0 else b",\n") + json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                stream.write(data)
                digest.update(data)
            stream.write(b"\n]\n")
            digest.update(b"\n]\n")
    size = temp.stat().st_size
    if size >= CEILING:
        raise ValueError(f"{name}: ZIP {size} B reaches the {CEILING} B ceiling; left at {temp}")
    temp.replace(target)
    return {"zip": str(target), "zip_bytes": size, "zip_sha256": _sha256(target), "json_sha256": digest.hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-from", type=Path, required=True)
    parser.add_argument("--chunks-from", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-all-chunks", action="store_true",
                        help="fail unless every chunk of --chunks-from is kept")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or paths.RUNS_DIR / "s31-strong-judge" / "build-mix")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"Output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    docs_zip, chunks_zip = paths.resolve_path(args.docs_from), paths.resolve_path(args.chunks_from)
    rows, stats = mix_rows(read_rows(docs_zip), read_rows(chunks_zip))
    if args.require_all_chunks and stats["chunks_dropped_doc_not_selected"]:
        raise ValueError(f"{stats['chunks_dropped_doc_not_selected']} chunks lost their doc")
    report = {"name": args.name, "status": "built_cpu_not_submitted",
              "docs_from": str(docs_zip), "docs_from_zip_sha256": _sha256(docs_zip),
              "chunks_from": str(chunks_zip), "chunks_from_zip_sha256": _sha256(chunks_zip),
              "rule": "relevant_docs of docs-from; relevant_chunks of chunks-from whose doc is selected, same order and text",
              **stats, **write_zip(out, args.name, rows)}
    _atomic_json(out / f"{args.name}.mix.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
