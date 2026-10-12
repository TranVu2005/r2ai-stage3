"""CC longchau docs -> t256 chunks -> BGE-M3 segment (own manifest, never touches data/index/t256 or the zh segments).

  python -m r2ai.cc_longchau.build chunk     # r2ai.index.chunk (t256, min 50 body tokens), verbatim gate, one segment row
  python -m r2ai.cc_longchau.build embed     # GPU: BGE-M3 fp16 batch 16, 1000-chunk blocks, SHA/norm/NaN/sparse checks on resume
  python -m r2ai.cc_longchau.build verify    # independent integrity report (no GPU)

Reuses the zh-full ingest machinery (manifest, block checkpoints, immutable segment writer) with this shard's roots.
Segment id is fixed to SEGMENT_ID so global chunk ids (2**48 + id * 2**32 + row) never clash with vi or zh segments.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from r2ai.zh_full_ingest import ingest as I
from r2ai.zh_full_ingest.common import MemoryGuard, Roots as ZRoots, global_start

from .common import DOMAIN, SEGMENT_ID, Roots, atomic_json, sha256_file

BATCH = 'cc-longchau'


def zroots(roots: Roots) -> ZRoots:
    return ZRoots(docs=roots.docs, chunks=roots.chunks, index=roots.index, run=roots.run, state=roots.state)


def ensure_segment(zr: ZRoots) -> dict:
    """Segment row with id SEGMENT_ID (allocated once; later calls return the same row)."""
    c = I.connect(zr)
    try:
        row = c.execute("SELECT segment_id, name, status FROM segments WHERE batch=? AND domain=? AND status!='retired'", (BATCH, DOMAIN)).fetchone()
        if row:
            return {'segment_id': row[0], 'name': row[1], 'status': row[2]}
        if c.execute('SELECT 1 FROM segments').fetchone():
            raise ValueError('manifest already holds other segments; refusing to allocate')
        with c:
            c.execute("UPDATE sqlite_sequence SET seq=? WHERE name='segments'", (SEGMENT_ID - 1,))
    finally:
        c.close()
    seg = I.allocate_segment(zr, BATCH, DOMAIN, kind='cc')
    if seg['segment_id'] != SEGMENT_ID:
        raise ValueError(f'segment id {seg["segment_id"]} != {SEGMENT_ID}')
    return {**seg, 'status': 'allocated'}


def chunk(roots: Roots, *, batch_docs=5000) -> dict:
    import shutil
    from r2ai.index.chunk import main as chunk_main
    zr = zroots(roots)
    roots.preflight()
    seg = ensure_segment(zr)
    if seg['status'] != 'allocated':
        return seg
    name = seg['name']
    ver = json.loads((roots.run / 'extract_summary.json').read_text(encoding='utf-8'))
    version = f'{ver["extractor"]}+trafilatura-{ver["trafilatura"]}'
    files = sorted(roots.docs.glob('*.parquet'))
    ids = []
    for f in files:
        for r in pq.read_table(f, columns=['doc_ids', 'status', 'n_tokens_bge_m3', 'body']).to_pylist():
            if r['status'] == 'ok' and (r['n_tokens_bge_m3'] or 0) >= I.MIN_BODY_TOKENS and r['body']:
                ids.append(r['doc_ids'][0])
    if len(ids) != len(set(ids)):
        raise ValueError('primary doc ids are not unique')
    out, staging = zr.guard(roots.chunks / name), zr.guard(roots.chunks / (name + '.partial'))
    if out.exists():
        raise ValueError(f'chunks already exist for allocated segment {name}')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    t0 = time.perf_counter()
    chunk_main(['--docs-dir', str(roots.docs), '--out-dir', str(staging), '--targets', str(I.TARGET),
                '--min-body-tokens', str(I.MIN_BODY_TOKENS), '--batch-docs', str(batch_docs)])
    report = json.loads((staging / 'chunk_report.json').read_text(encoding='utf-8'))
    cfg = report['configs'][str(I.TARGET)]
    if cfg['verbatim_ok'] != f"{cfg['chunks_all']}/{cfg['chunks_all']}":
        raise ValueError(f'verbatim gate failed: {cfg["verbatim_ok"]}')
    os.replace(staging, out)
    cf = out / f'chunks_t{I.TARGET}.parquet'
    chunker = {'target': I.TARGET, 'min_body_tokens': I.MIN_BODY_TOKENS, 'batch_docs': batch_docs, 'module': 'r2ai.index.chunk'}
    stats = {'chunk_seconds': time.perf_counter() - t0, 'docs': report['docs'], 'chunks': cfg['chunks_all'], 'eligible_docs': len(ids)}
    I._update_segment(zr, name, status='chunked', rows=cfg['chunks_all'], docs_count=report['docs'], chunks_sha256=sha256_file(cf),
                      extractor_version=version, chunker=json.dumps(chunker), model=json.dumps(I.MODEL), stats=json.dumps(stats))
    return {**seg, 'status': 'chunked', **stats}


def embed(roots: Roots, *, min_available_gib=3.0, max_pauses=400) -> dict:
    zr = zroots(roots)
    roots.preflight()
    seg = ensure_segment(zr)
    return I.embed_segment(zr, seg['name'], memory=MemoryGuard(zr, minimum_gib=min_available_gib, max_pauses=max_pauses))


def verify(roots: Roots) -> dict:
    """No GPU. Chunk text verbatim vs docs.parquet doc_text; segment shape/ids/norms/NaN/sparse vs chunks; SHA of chunks file."""
    from r2ai.index.segments import load_segment
    zr = zroots(roots)
    seg = ensure_segment(zr)
    c = I.connect(zr)
    try:
        rec = c.execute('SELECT status, rows, docs_count, chunks_sha256, global_start FROM segments WHERE name=?', (seg['name'],)).fetchone()
    finally:
        c.close()
    cdir = roots.chunks / seg['name']
    cf = cdir / f'chunks_t{I.TARGET}.parquet'
    docs = {r['doc_id']: r['doc_text'] for r in pq.read_table(cdir / 'docs.parquet', columns=['doc_id', 'doc_text']).to_pylist()}
    n = bad = 0
    for b in pq.ParquetFile(cf).iter_batches(batch_size=50000, columns=['doc_id', 'char_start', 'char_end', 'text', 'field']):
        for d, s, e, t in zip(*(b.column(k).to_pylist() for k in ('doc_id', 'char_start', 'char_end', 'text'))):
            n += 1
            bad += docs[d][s:e] != t or not t.strip()
    out = {'segment': seg['name'], 'manifest': dict(zip(('status', 'rows', 'docs', 'chunks_sha256', 'global_start'), rec)),
           'chunks': n, 'verbatim_bad': bad, 'chunks_sha256_ok': sha256_file(cf) == rec[3], 'docs_in_chunks': len(docs)}
    if rec[0] == 'active':
        s = load_segment(roots.index / seg['name'])
        dense = np.asarray(s.dense[:], dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1)
        out.update(segment_rows=len(s), global_start_ok=s.global_start == global_start(SEGMENT_ID), dense_nan=int((~np.isfinite(dense)).sum()),
                   norm_min=float(norms.min()), norm_max=float(norms.max()),
                   doc_id_match=bool(np.array_equal(np.asarray(s.doc_id, dtype=np.int64), pq.read_table(cf, columns=['doc_id']).column(0).to_numpy())),
                   sparse_empty_rows=int((np.diff(np.asarray(s.sparse[2])) == 0).sum()), sparse_nnz=int(len(s.sparse[0])))
    atomic_json(roots, roots.run / 'verify.json', out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['chunk', 'embed', 'verify'])
    a = ap.parse_args(argv)
    r = {'chunk': chunk, 'embed': embed, 'verify': verify}[a.step](Roots())
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
