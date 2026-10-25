"""S25 phase 1 step 2: every text version of every corpus-v2 doc_id, read-only.

    python -m r2ai.canonical.versions --out out/runs/s25-canonical/versions

Sources are all doc-level Parquet files below the inventory roots: segment docs (`doc_text`) and extract output
(`title/question/answer/body`, turned into doc_text exactly like `r2ai.index.chunk.doc_layout`). Chunk files are not
doc versions. A version is (doc_id, sha256(doc_text)); every v2 ID of a row (`doc_ids_v2` > `doc_ids` >
`doc_ids_group` > `doc_id`, intersected with v2) gets one row per source file. Hard-linked files are read once
and listed as aliases. Quality is measured on the body after whitespace collapse: code points and Han count.
"""
from __future__ import annotations

from r2ai.paths import resolve_path

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.canonical.guard import RUN_ROOT, guarded, write_parquet
from r2ai.canonical.inventory import MAIN_REPO, extra_roots, main_roots, walk

ID_PRIORITY = ('doc_ids_v2', 'doc_ids', 'doc_ids_group', 'doc_id')
EXTRACT_COLS = ('title', 'question', 'answer', 'body', 'paragraphs', 'status', 'extractor')
WS = re.compile(r'\s+')
HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]')


def doc_text_of(r: dict) -> tuple[str, str]:
    """-> (doc_text, body) with the doc_text rule of r2ai.index.chunk.doc_layout (no paragraph spans needed)."""
    title = (r.get('title') or '').strip()
    q = (r.get('question') or '').strip()
    body = r.get('body') or ''
    parts = [title] if title else []
    if q and q not in body:
        parts.append(q)
    parts.append(body)
    return '\n\n'.join(parts), body


def body_of_doc_text(doc_text: str, title: str | None) -> str:
    """Segment docs store doc_text only; drop the leading title block when present (question block, if any, stays)."""
    t = (title or '').strip()
    if t and doc_text.startswith(t + '\n\n'):
        return doc_text[len(t) + 2:]
    return doc_text


def quality(body: str) -> tuple[int, int]:
    clean = WS.sub(' ', body).strip()
    return len(clean), len(HAN.findall(clean))


def classify(path: Path):
    """-> (kind, id_col, text_mode) or None. kind: segment-docs | extract | overlay."""
    try:
        names = pq.read_schema(path).names
    except Exception:
        return None
    if 'chunk_id' in names:
        return None
    id_col = next((c for c in ID_PRIORITY if c in names), None)
    if id_col is None:
        return None
    if 'doc_text' in names:
        return 'segment-docs', id_col, 'doc_text'
    if 'body' in names:
        kind = 'overlay' if 'url' not in names else 'extract'
        return kind, id_col, 'fields'
    return None


def scan_file(args):
    path, src_id, id_col, mode, v2 = args
    names = pq.read_schema(path).names
    cols = [c for c in ID_PRIORITY if c in names]
    if mode == 'doc_text':
        cols += ['doc_text'] + (['title'] if 'title' in names else [])
    else:
        cols += [c for c in EXTRACT_COLS if c in names and c != 'paragraphs']
    out = {k: [] for k in ('doc_id', 'phys_id', 'src', 'sha', 'doc_cp', 'body_cp', 'han', 'status', 'extractor')}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        for r in batch.to_pylist():
            ids = next((r[c] for c in ID_PRIORITY if c in r and r[c] is not None), None)
            ids = [ids] if isinstance(ids, int) else list(ids or [])
            if 'doc_id' in r and r['doc_id'] is not None and id_col != 'doc_id':
                phys = int(r['doc_id'])
            else:
                phys = int(ids[0]) if ids else -1
            keep = [int(i) for i in ids if int(i) in v2]
            if not keep:
                continue
            if mode == 'doc_text':
                text = r['doc_text'] or ''
                body = body_of_doc_text(text, r.get('title'))
                status, extractor = None, None
            else:
                text, body = doc_text_of(r)
                status, extractor = r.get('status'), r.get('extractor')
            sha = hashlib.sha256(text.encode('utf-8')).digest()
            cp, han = quality(body)
            for i in dict.fromkeys(keep):
                out['doc_id'].append(i)
                out['phys_id'].append(phys)
                out['src'].append(src_id)
                out['sha'].append(sha)
                out['doc_cp'].append(len(text))
                out['body_cp'].append(cp)
                out['han'].append(han)
                out['status'].append(status)
                out['extractor'].append(extractor)
    return out


def discover(roots):
    files = []
    for owner, root in roots:
        if not root.exists() or root.name == 'hf_model_cache':
            continue
        for d, filenames in walk(root):
            if owner != 'main' and d.relative_to(root).parts[:1] and d.relative_to(root).parts[0] in ('src', 'tests'):
                continue
            files += [(owner, d / f) for f in filenames if f.endswith('.parquet')]
    return files


_V2 = None


def _init(v2_path):
    global _V2
    _V2 = set(pq.read_table(v2_path, columns=['id'])['id'].to_pylist())


def _scan(args):
    path, src_id, id_col, mode = args
    return src_id, scan_file((path, src_id, id_col, mode, _V2))


def build(out: Path, corpus: Path, workers: int, include_extra: bool) -> dict:
    t0 = time.time()
    roots = main_roots() + (extra_roots() if include_extra else [])
    files = discover(roots)
    sources, tasks, seen_inode = [], [], {}
    for owner, p in files:
        st = p.stat()
        c = classify(p)
        if c is None:
            continue
        kind, id_col, mode = c
        key = (st.st_dev, st.st_ino)
        src_id = len(sources)
        alias_of = seen_inode.get(key) if st.st_ino else None
        sources.append(dict(src=src_id, owner=owner, path=str(p), kind=kind, id_col=id_col, bytes=st.st_size,
                            mtime=st.st_mtime, nlink=st.st_nlink, alias_of=-1 if alias_of is None else alias_of))
        if alias_of is None:
            seen_inode[key] = src_id
            tasks.append((str(p), src_id, id_col, mode))
    print(f'[{time.time()-t0:.0f}s] doc-level sources={len(sources)} unique={len(tasks)}', flush=True)
    parts = {k: [] for k in ('doc_id', 'phys_id', 'src', 'sha', 'doc_cp', 'body_cp', 'han', 'status', 'extractor')}
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(corpus),)) as ex:
        for src_id, res in ex.map(_scan, tasks, chunksize=64):
            for k, v in res.items():
                parts[k].extend(v)
            done += 1
            if done % 5000 == 0:
                print(f'[{time.time()-t0:.0f}s] files {done}/{len(tasks)} rows={len(parts["doc_id"])}', flush=True)
    table = pa.table({
        'doc_id': pa.array(parts['doc_id'], pa.int64()), 'phys_id': pa.array(parts['phys_id'], pa.int64()),
        'src': pa.array(parts['src'], pa.int32()), 'sha256': pa.array(parts['sha'], pa.binary(32)),
        'doc_cp': pa.array(parts['doc_cp'], pa.int32()), 'body_cp': pa.array(parts['body_cp'], pa.int32()),
        'han': pa.array(parts['han'], pa.int32()),
        'status': pa.array(parts['status'], pa.string()).dictionary_encode(),
        'extractor': pa.array(parts['extractor'], pa.string()).dictionary_encode(),
    })
    write_parquet(table, out / 'versions.parquet')
    write_parquet(pa.Table.from_pylist(sources), out / 'sources.parquet')
    summary = dict(sources=len(sources), unique_files=len(tasks), rows=table.num_rows,
                   doc_ids=int(len(np.unique(table['doc_id'].to_numpy()))), seconds=round(time.time() - t0, 1))
    guarded(out / 'versions-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RUN_ROOT / 'versions'))
    ap.add_argument('--corpus', default=str(MAIN_REPO / 'data' / 'corpus_v2_ids.parquet'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-extra', action='store_true')
    a = ap.parse_args(argv)
    out = guarded(a.out)
    out.mkdir(parents=True)
    print(json.dumps(build(out, resolve_path(a.corpus), a.workers, not a.no_extra), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
