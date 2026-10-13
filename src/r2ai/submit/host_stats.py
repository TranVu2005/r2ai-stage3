"""Per-host statistics of a finished submission (read-only).

    python -m r2ai.submit.host_stats SUB.(zip|json) --host vinmec.com [--out stats.json]

Reports doc ids, chunks, chunk_text bytes (UTF-8), per-query means and how many queries contain at least one doc /
chunk of the host, next to the submission totals. Host membership comes from corpus v2 (`data/corpus_v2_ids.parquet`).
"""
from __future__ import annotations

from r2ai.paths import assert_writable

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from r2ai import corpus_v2 as C
from r2ai.submit.corpus_filter import _open_src


def _rows(path):
    fin, zin = _open_src(Path(path))
    try:
        for raw in fin:
            line = raw.decode('utf-8')
            if line in ('[\n', ']\n'):
                continue
            yield json.loads(line.rstrip('\n').removesuffix(','))
    finally:
        fin.close()
        if zin is not None:
            zin.close()


def host_stats(src, hosts, ids_path=None) -> dict:
    hosts = list(hosts)
    ids = C.host_ids(hosts, ids_path)
    in_host = lambda arr: np.isin(np.asarray(arr, dtype=np.int64), ids)
    acc = {k: 0 for k in ('docs', 'chunks', 'chunk_text_bytes', 'queries_with_doc', 'queries_with_chunk')}
    tot = {k: 0 for k in ('docs', 'chunks', 'chunk_text_bytes')}
    docs_d: set[int] = set()
    chunk_docs_d: set[int] = set()
    outside = 0
    q = 0
    for row in _rows(src):
        q += 1
        docs = row['relevant_docs']
        chunks = row['relevant_chunks']
        dm = in_host(docs)
        cm = in_host([c['doc_id'] for c in chunks])
        outside += int((~C.member_mask(docs, ids_path)).sum())
        sizes = np.array([len(c['chunk_text'].encode('utf-8')) for c in chunks], dtype=np.int64)
        tot['docs'] += len(docs)
        tot['chunks'] += len(chunks)
        tot['chunk_text_bytes'] += int(sizes.sum())
        acc['docs'] += int(dm.sum())
        acc['chunks'] += int(cm.sum())
        acc['chunk_text_bytes'] += int(sizes[cm].sum())
        acc['queries_with_doc'] += bool(dm.any())
        acc['queries_with_chunk'] += bool(cm.any())
        docs_d.update(d for d, m in zip(docs, dm) if m)
        chunk_docs_d.update(c['doc_id'] for c, m in zip(chunks, cm) if m)
    acc['docs_distinct'] = len(docs_d)
    acc['chunk_docs_distinct'] = len(chunk_docs_d)
    for k in ('docs', 'chunks', 'chunk_text_bytes'):
        acc[f'{k}_per_query'] = acc[k] / q if q else 0.0
        acc[f'share_{k}'] = acc[k] / tot[k] if tot[k] else 0.0
    return {'src': str(src), 'host': hosts, 'queries': q, 'host_ids_in_v2': int(ids.size), 'host_stats': acc,
            'total': tot, 'docs_outside_v2': outside}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src')
    ap.add_argument('--host', action='append', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    st = host_stats(args.src, args.host)
    text = json.dumps(st, ensure_ascii=False, indent=1)
    if args.out:
        out = assert_writable(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
