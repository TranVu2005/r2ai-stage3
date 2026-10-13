"""v2-filtered copy of the ACTIVE zh-full segments: same vectors, no re-embedding (S11 step 3).

A chunk is kept when its doc (`doc_id` = primary id) has at least one id of `doc_ids_group` in the v2 zh ids. Dense fp16 rows,
CSR rows and texts of kept chunks are copied byte for byte in the original row order (docs stay contiguous, so doc-aligned
scan blocks and tie order are unchanged); only the local row numbers change. New segment = same name and global_start as its
source, rows renumbered 0..n-1; `row_map.parquet` gives new -> old local row. qihuangzhishu is left out (D5).

    python -m r2ai.zh_v2.index build [--verify]
    python -m r2ai.zh_v2.index status
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, WORK_DATA_DIR, assert_writable, resolve_path
from r2ai.index.segments import ZH_INDEX_ROOT, ZH_V2_INDEX_ROOT, SegmentWriter, load_segment
from r2ai.retrieve.exact import take_texts

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import scope

CHUNKS_SRC = WORK_DATA_DIR / 'chunks_zh_full'
CHUNKS_V2 = WORK_DATA_DIR / 'chunks_zh_v2'
RUN = RUNS_DIR / 'zh-v2'
MANIFEST = ZH_V2_INDEX_ROOT / 'manifest.json'
EXCLUDED_DOMAINS = ('qihuangzhishu.com',)       # D5
BLOCK = 8192


def guard(path) -> Path:
    p = assert_writable(path)
    if not any(p.is_relative_to(resolve_path(r)) for r in (ZH_V2_INDEX_ROOT, CHUNKS_V2, RUN)):
        raise ValueError(f'Not a zh-v2 output: {p}')
    return p


def sha256_file(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def keep_docs(docs_file, v2_ids: np.ndarray) -> np.ndarray:
    """Primary doc ids of docs.parquet whose group holds at least one v2 id."""
    t = pl.read_parquet(docs_file, columns=['doc_id', 'doc_ids_group'])
    g = t.explode('doc_ids_group', empty_as_null=True)
    hit = g.join(pl.DataFrame({'doc_ids_group': v2_ids}), on='doc_ids_group', how='semi').select('doc_id').unique()
    return np.sort(hit['doc_id'].to_numpy())


def ranges(lo: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of arange(lo[i], lo[i] + lens[i])."""
    total = int(lens.sum())
    if not total:
        return np.zeros(0, np.int64)
    offsets = np.cumsum(lens) - lens
    return np.repeat(lo - offsets, lens) + np.arange(total, dtype=np.int64)


def copy_block(seg, rows: np.ndarray, texts_col, vocabulary: int):
    """(dense fp16, csr, doc_id, texts) of the source rows `rows` (ascending)."""
    from scipy.sparse import csr_matrix
    dense = np.asarray(seg.dense[rows])
    data_a, ind_a, ptr_a = seg.sparse
    lo, hi = np.asarray(ptr_a[rows]), np.asarray(ptr_a[rows + 1])
    lens = hi - lo
    idx = ranges(lo, lens)
    if len(idx):
        data, ind = np.asarray(data_a[idx]), np.asarray(ind_a[idx])
    else:
        data, ind = np.zeros(0, np.float32), np.zeros(0, np.int32)
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    csr = csr_matrix((data, ind, indptr), shape=(len(rows), vocabulary))
    return dense, csr, np.asarray(seg.doc_id[rows], dtype=np.int64), take_texts(texts_col, rows)


def build_segment(src: dict, v2_ids: np.ndarray, *, block=BLOCK, out_root=None, chunks_out=None, chunks_src=None) -> dict:
    """Write one filtered segment + its chunks/docs/row_map. Returns the manifest row (rows may be 0 = nothing written)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    out_root, chunks_out, chunks_src = out_root or ZH_V2_INDEX_ROOT, chunks_out or CHUNKS_V2, chunks_src or CHUNKS_SRC
    name = src['name']
    seg = load_segment(ZH_INDEX_ROOT / name)
    meta = json.loads((ZH_INDEX_ROOT / name / 'segment.json').read_text('utf-8'))
    keep = keep_docs(chunks_src / name / 'docs.parquet', v2_ids)
    doc_id = np.asarray(seg.doc_id, dtype=np.int64)
    rows = np.flatnonzero(np.isin(doc_id, keep))
    row = {'name': name, 'segment_id': src['segment_id'], 'batch': src['batch'], 'domain': src['domain'],
           'rows_source': len(seg), 'rows_v2': int(len(rows)), 'docs_v2': int(len(np.unique(doc_id[rows]))),
           'global_start': int(seg.global_start), 'source_chunks_sha256': src.get('chunks_sha256')}
    if not len(rows):
        return row
    final = guard(out_root / name)
    if final.exists():
        raise ValueError(f'v2 segment {name} exists')
    with pa.memory_map(str(seg.text_path), 'r') as source:
        texts_col = ipc.open_file(source).read_all().column('text')
        w = SegmentWriter(out_root, name + '.assembling', rows=len(rows), dimension=meta['dimension'],
                          vocabulary=meta['vocabulary'], global_start=seg.global_start)
        try:
            for lo in range(0, len(rows), block):
                dense, csr, d, texts = copy_block(seg, rows[lo:lo + block], texts_col, meta['vocabulary'])
                w.append(dense, csr, d, texts)
            w.close()
        except BaseException:
            for f in w.files.values():
                f.close()
            raise
        del texts_col
    os.replace(guard(out_root / (name + '.assembling')), final)
    # chunks / docs / row_map
    cdir = guard(chunks_out / name)
    cdir.mkdir(parents=True, exist_ok=True)
    renumber = np.full(len(seg), -1, np.int64)
    renumber[rows] = np.arange(len(rows))
    pf = pq.ParquetFile(chunks_src / name / 'chunks_t256.parquet')
    if pf.metadata.num_rows != len(seg):
        raise ValueError(f'{name}: chunk rows {pf.metadata.num_rows} != segment rows {len(seg)}')
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(guard(cdir / 'chunks_t256.parquet.partial'), schema, compression='zstd')
    at = 0
    try:
        for batch in pf.iter_batches(batch_size=65536):
            n = len(batch)
            sel = np.flatnonzero(renumber[at:at + n] >= 0)
            if len(sel):
                t = pa.Table.from_batches([batch]).take(pa.array(sel))
                t = t.set_column(schema.get_field_index('chunk_id'), 'chunk_id',
                                 pa.array(renumber[at + sel], type=schema.field('chunk_id').type))
                writer.write_table(t)
            at += n
    finally:
        writer.close()
    os.replace(cdir / 'chunks_t256.parquet.partial', cdir / 'chunks_t256.parquet')
    docs = pl.read_parquet(chunks_src / name / 'docs.parquet').join(pl.DataFrame({'doc_id': keep}), on='doc_id', how='semi')
    v2set = set(v2_ids.tolist())
    docs = docs.with_columns(pl.Series('doc_ids_v2', [[i for i in g if i in v2set] for g in docs['doc_ids_group'].to_list()],
                                       dtype=pl.List(pl.Int64)))
    docs.write_parquet(guard(cdir / 'docs.parquet'), compression='zstd')
    pl.DataFrame({'new_local': np.arange(len(rows), dtype=np.int64), 'old_local': rows.astype(np.int64)}) \
        .write_parquet(guard(cdir / 'row_map.parquet'), compression='zstd')
    row['docs_file_docs'] = docs.height
    return row


def verify_segment(name: str, *, block=BLOCK * 4, out_root=None, chunks_out=None) -> dict:
    """Every kept row equals its source row: dense, CSR, doc_id, text; and the chunks parquet agrees (text, doc_id)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    out_root, chunks_out = out_root or ZH_V2_INDEX_ROOT, chunks_out or CHUNKS_V2
    new, old = load_segment(out_root / name), load_segment(ZH_INDEX_ROOT / name)
    old_local = pl.read_parquet(chunks_out / name / 'row_map.parquet').sort('new_local')['old_local'].to_numpy()
    if len(old_local) != len(new) or new.global_start != old.global_start:
        raise ValueError(f'{name}: row_map/global_start mismatch')
    ctext = pq.read_table(chunks_out / name / 'chunks_t256.parquet', columns=['chunk_id', 'doc_id', 'text'])
    if not np.array_equal(ctext['chunk_id'].to_numpy(), np.arange(len(new))):
        raise ValueError(f'{name}: chunks parquet ids are not 0..n-1')
    with pa.memory_map(str(new.text_path), 'r') as a, pa.memory_map(str(old.text_path), 'r') as b:
        tn, to = ipc.open_file(a).read_all().column('text'), ipc.open_file(b).read_all().column('text')
        for lo in range(0, len(new), block):
            hi = min(len(new), lo + block)
            o = old_local[lo:hi]
            if not np.array_equal(np.asarray(new.dense[lo:hi]), np.asarray(old.dense[o])):
                raise ValueError(f'{name}: dense differs in rows {lo}..{hi}')
            if not np.array_equal(np.asarray(new.doc_id[lo:hi]), np.asarray(old.doc_id[o])):
                raise ValueError(f'{name}: doc_id differs in rows {lo}..{hi}')
            got = take_texts(tn, np.arange(lo, hi))
            if got != take_texts(to, o) or got != ctext['text'].slice(lo, hi - lo).to_pylist():
                raise ValueError(f'{name}: text differs in rows {lo}..{hi}')
            ids = np.arange(lo, hi)
            n_d, n_i, n_p = new.sparse
            o_d, o_i, o_p = old.sparse
            a_lo, a_hi = np.asarray(n_p[ids]), np.asarray(n_p[ids + 1])
            b_lo, b_hi = np.asarray(o_p[o]), np.asarray(o_p[o + 1])
            if not np.array_equal(a_hi - a_lo, b_hi - b_lo):
                raise ValueError(f'{name}: sparse row lengths differ in rows {lo}..{hi}')
            ia, ib = ranges(a_lo, a_hi - a_lo), ranges(b_lo, b_hi - b_lo)
            if len(ia) and not (np.array_equal(np.asarray(n_i[ia]), np.asarray(o_i[ib]))
                                and np.array_equal(np.asarray(n_d[ia]), np.asarray(o_d[ib]))):
                raise ValueError(f'{name}: sparse values differ in rows {lo}..{hi}')
    return {'name': name, 'rows': len(new), 'verified': True}


def source_segments(manifest=None) -> list[dict]:
    import sqlite3
    path = manifest or scope.MANIFEST
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        cols = ('segment_id', 'name', 'batch', 'domain', 'rows', 'chunks_sha256', 'global_start')
        rows = [dict(zip(cols, r)) for r in c.execute(
            f"SELECT {','.join(cols)} FROM segments WHERE status='active' ORDER BY segment_id")]
    finally:
        c.close()
    return [r for r in rows if r['domain'] not in EXCLUDED_DOMAINS]


def build(*, verify=False, only=None) -> dict:
    ids = scope.v2_zh()['id'].to_numpy()
    ids = np.sort(ids)
    ZH_V2_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    done = {r['name']: r for r in json.loads(MANIFEST.read_text('utf-8'))['segments']} if MANIFEST.exists() else {}
    rows, t0 = [], time.perf_counter()
    for src in source_segments():
        if only and src['name'] not in only:
            continue
        if src['name'] in done:
            rows.append(done[src['name']])
            continue
        ts = time.perf_counter()
        row = build_segment(src, ids)
        if row['rows_v2'] and verify:
            row['verified'] = verify_segment(src['name'])['verified']
        row['seconds'] = round(time.perf_counter() - ts, 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_manifest(rows, ids)
    write_manifest(rows, ids)
    return {'segments': len(rows), 'rows_v2': sum(r['rows_v2'] for r in rows), 'docs_v2': sum(r['docs_v2'] for r in rows),
            'seconds': round(time.perf_counter() - t0, 1)}


def write_manifest(rows, ids):
    body = {'built_at': time.time(), 'v2_zh_ids': int(len(ids)), 'v2_ids_sha256': sha256_file(scope.IDS_PATH),
            'excluded_domains': list(EXCLUDED_DOMAINS), 'segments': rows}
    tmp = guard(MANIFEST.with_name('manifest.json.tmp'))
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, guard(MANIFEST))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=('build', 'status'))
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args(argv)
    if a.step == 'status':
        m = json.loads(MANIFEST.read_text('utf-8'))
        print(json.dumps({k: v for k, v in m.items() if k != 'segments'}, indent=1))
        print(pl.DataFrame(m['segments']).select('name', 'rows_source', 'rows_v2', 'docs_v2'))
        return 0
    print(json.dumps(build(verify=a.verify, only=a.only)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
