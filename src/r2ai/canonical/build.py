"""S25 phase 2 step 4: write the canonical corpus to a NEW directory (nothing existing is modified).

    python -m r2ai.canonical.build --versions out/runs/s25-canonical/versions --out data/canonical-20261018

Outputs (all below --out, refused if present):
  docs.parquet        one row per v2 doc_id (vi, zh): text, quality, rule, chosen source
  chunks_t256.parquet chunks of the chosen versions only (rule a, or quality good), verbatim vs docs.doc_text,
                      global chunk IDs kept (global_start of the source segment + local chunk_id)
  vectors/            dense.npy fp16 + CSR (data/indices/indptr .bin) + source_row.bin, row-aligned with the
                      chunks_t256.parquet rows that have has_vector=True; a vector is copied only when the source
                      index text of that row equals the chunk text
  reembed.parquet     canonical chunks with no matching vector (listed, never encoded here)
  decisions.parquet   one row per doc_id with more than one distinct text
  provenance.parquet  per doc_id: chosen source file + SHA-256, extractor/status, raw pointers when recorded
  manifest.json       counts, input pins, SHA-256 of every output file
Docs and chunk files are streamed in doc_id order (they are doc_id-sorted), so peak RAM stays bounded.
Chosen versions that exist only as extract output are chunked with r2ai.index.chunk (t256, same rules) into the
private S25 range, segment 25000; those chunks always go to reembed.
"""
from __future__ import annotations

from r2ai.paths import resolve_path

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from r2ai.canonical import select as S
from r2ai.canonical.guard import CANONICAL_ROOT, RUN_ROOT, guarded, write_parquet
from r2ai.canonical.versions import doc_text_of
from r2ai.retrieve.exact import take_texts

S25_SEGMENT = 25000
S25_NAME = f'seg{S25_SEGMENT:05d}-s25-canonical'
CHUNK_SCHEMA = pa.schema([('chunk_id', pa.int64()), ('doc_id', pa.int64()), ('doc_ids', pa.list_(pa.int64())),
                          ('field', pa.string()), ('char_start', pa.int32()), ('char_end', pa.int32()),
                          ('text', pa.string()), ('n_tokens', pa.int32()), ('segment', pa.string()),
                          ('has_vector', pa.bool_())])
DOC_SCHEMA = pa.schema([('doc_id', pa.int64()), ('lang', pa.string()), ('host', pa.string()), ('url', pa.string()),
                        ('phys_id', pa.int64()), ('doc_text', pa.string()), ('body_cp', pa.int32()), ('han', pa.int32()),
                        ('quality', pa.string()), ('rule', pa.string()), ('text_sha256', pa.string()),
                        ('segment', pa.string()), ('source_path', pa.string())])
ID_COLS = ('doc_ids_v2', 'doc_ids', 'doc_ids_group', 'doc_id')


def sha_file(p: Path) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def global_start_of(name: str) -> int:
    """Production convention: global_start = (65536 + segment_id) << 32; base-t256 starts at 0."""
    if name == 'base-t256':
        return 0
    return (65536 + int(name[3:8])) << 32


def segment_of(docs_path: str, live_dirs: dict) -> str:
    p = Path(docs_path)
    if p.parent.name == 'chunks' and p.parent.parent.name.lower() == 'data':
        return 'base-t256'
    if p.parent.name == 'segment':                         # data/s22_reextract/<batch>/segment
        return live_dirs.get(S.norm(p.parent), p.parent.parent.name)
    return p.parent.name


class IndexRef:
    """Row-aligned vectors of one chunk file (base t256 or an ingest segment)."""

    def __init__(self, path: Path, rows: int):
        self.path, self.rows = path, rows
        self.dense = np.load(path / 'dense.npy', mmap_mode='r')
        if (path / 'sparse_mmap').is_dir():
            self.data = np.load(path / 'sparse_mmap/data.npy', mmap_mode='r')
            self.indices = np.load(path / 'sparse_mmap/indices.npy', mmap_mode='r')
            self.indptr = np.load(path / 'sparse_mmap/indptr.npy', mmap_mode='r')
        else:
            nnz = int(json.loads((path / 'segment.json').read_text(encoding='utf-8'))['nnz'])
            mm = lambda n, dt, k: np.memmap(path / n, dtype=dt, mode='r', shape=(k,)) if k else np.zeros(0, dt)
            self.data, self.indices = mm('data.bin', np.float32, nnz), mm('indices.bin', np.int32, nnz)
            self.indptr = mm('indptr.bin', np.int64, rows + 1)
        if len(self.dense) != rows or len(self.indptr) != rows + 1:
            raise ValueError(f'index rows differ from chunk rows: {path}')


def find_index(name: str, chunk_rows: int, index_names: dict) -> Path | None:
    if name == 'base-t256':
        return S.DATA / 'index/t256'
    for cand in index_names.get(name, []):
        meta = json.loads((cand / 'segment.json').read_text(encoding='utf-8'))
        if int(meta['rows']) == chunk_rows and (cand / 'text.arrow').is_file():
            return cand
    return None


def pick_sources(chosen: pl.DataFrame, rows: pl.DataFrame, sources: pl.DataFrame) -> pl.DataFrame:
    """Concrete file for each chosen (doc_id, sha256): live > has chunks > segment docs > under data/ > newest."""
    src = sources.with_columns(pl.col('path').map_elements(lambda p: str(Path(p).parent / 'chunks_t256.parquet'),
                                                           return_dtype=pl.Utf8).alias('chunks_path'))
    src = src.with_columns(pl.col('chunks_path').map_elements(lambda p: Path(p).is_file(), return_dtype=pl.Boolean)
                           .alias('has_chunks'))
    cand = (rows.join(chosen.select('doc_id', 'sha256', 'rule', 'quality'), on=['doc_id', 'sha256'])
            .join(src.select('src', 'path', 'chunks_path', 'has_chunks'), left_on='real', right_on='src', how='left'))
    cand = cand.with_columns((pl.col('kind') == 'segment-docs').alias('is_seg'),
                             pl.col('path').str.to_lowercase().str.contains(r'\\github\\r2ai-stage3\\data\\',
                                                                            literal=True).alias('in_data'))
    best = (cand.sort(['doc_id', 'live', 'has_chunks', 'is_seg', 'in_data', 'mtime'],
                      descending=[False, True, True, True, True, True])
            .group_by('doc_id', maintain_order=True).first())
    return best.select('doc_id', 'sha256', 'rule', 'quality', 'phys_id', 'real', 'path', 'kind', 'has_chunks',
                       'chunks_path', 'lang', 'host', 'body_cp', 'han', 'extractor', 'status', 'live')


def iter_texts(path: str, kind: str, wanted: set[int], batch_size: int = 20000):
    """Yield batches of (sorted v2 ids in `wanted`, physical id, doc_text) in file order."""
    names = pq.read_schema(path).names
    id_cols = [c for c in ID_COLS if c in names]
    cols = id_cols + (['doc_text'] if kind == 'segment-docs'
                      else [c for c in ('title', 'question', 'answer', 'body') if c in names])
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=cols):
        out = []
        for r in b.to_pylist():
            ids = next((r[c] for c in id_cols if r[c] is not None), None)
            ids = [ids] if isinstance(ids, int) else list(ids or [])
            hit = sorted({int(i) for i in ids if int(i) in wanted})
            if not hit:
                continue
            phys = int(r['doc_id']) if r.get('doc_id') is not None else int(ids[0])
            text = r['doc_text'] if kind == 'segment-docs' else doc_text_of(r)[0]
            out.append((hit, phys, text))
        if out:
            yield out


class SegmentChunks:
    """Chunks of one segment file restricted to chosen docs, verbatim-checked; vector rows where index text matches."""

    def __init__(self, chunks_path: str, seg: str, index_names: dict, s25: dict):
        self.path, self.seg, self.s25 = chunks_path, seg, s25
        self.total = pq.ParquetFile(chunks_path).metadata.num_rows
        self.index = find_index(seg, self.total, index_names)
        self.ref = IndexRef(self.index, self.total) if self.index is not None else None
        if self.index is not None and self.index != S.DATA / 'index/t256':
            self.gs = int(json.loads((self.index / 'segment.json').read_text(encoding='utf-8'))['global_start'])
        elif seg == 'base-t256' or (seg.startswith('seg') and seg[3:8].isdigit()):
            self.gs = global_start_of(seg)
        else:
            self.gs = None                                 # unnamed copy without index: new S25 IDs
        self.ds = ds.dataset(chunks_path, format='parquet')
        self.text_col = None
        if self.ref is not None:
            self._mm = pa.memory_map(str(self.index / 'text.arrow'), 'r')
            self.text_col = pa.ipc.open_file(self._mm).read_all().column('text')
            if len(self.text_col) != self.total:
                raise ValueError(f'text.arrow rows differ: {self.index}')

    def emit(self, groups: dict, texts: dict, reembed: list):
        phys = np.fromiter(groups, dtype=np.int64)
        f = (ds.field('doc_id') >= int(phys.min())) & (ds.field('doc_id') <= int(phys.max()))
        t = self.ds.to_table(columns=['chunk_id', 'doc_id', 'field', 'char_start', 'char_end', 'text', 'n_tokens'],
                             filter=f)
        t = t.filter(pa.array(np.isin(t['doc_id'].to_numpy(), phys)))
        local = t['chunk_id'].to_numpy()
        idx_texts = take_texts(self.text_col, local) if self.text_col is not None else None
        rows, vec_rows, miss = [], [], 0
        for i, r in enumerate(t.to_pylist()):
            if texts[r['doc_id']][r['char_start']:r['char_end']] != r['text']:
                raise ValueError(f'chunk not verbatim: {self.path} chunk {r["chunk_id"]}')
            has_vec = idx_texts is not None and idx_texts[i] == r['text']
            miss += int(idx_texts is not None and not has_vec)
            if self.gs is not None:
                gid = self.gs + int(r['chunk_id'])
            else:
                gid = global_start_of(f'seg{S25_SEGMENT:05d}') + self.s25['next']
                self.s25['next'] += 1
            rows.append({'chunk_id': gid, 'doc_id': r['doc_id'], 'doc_ids': sorted(groups[r['doc_id']]),
                         'field': r['field'], 'char_start': r['char_start'], 'char_end': r['char_end'],
                         'text': r['text'], 'n_tokens': r['n_tokens'], 'segment': self.seg, 'has_vector': has_vec})
            if has_vec:
                vec_rows.append(int(r['chunk_id']))
            else:
                reembed.append({'chunk_id': gid, 'doc_id': r['doc_id'], 'segment': self.seg, 'field': r['field'],
                                'reason': 'no index for segment' if self.ref is None else 'index text differs'})
        return rows, vec_rows, miss


def merge_plan(plan):
    merged = []
    for idx, total, rows in plan:
        if merged and merged[-1][0] == idx:
            merged[-1] = (idx, total, np.concatenate([merged[-1][2], rows]))
        else:
            merged.append((idx, total, rows))
    return merged


def build(versions_dir: Path, out: Path, *, vectors: bool = True) -> dict:
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)
    out = guarded(out)
    corpus = pl.read_parquet(S.DATA / 'corpus_v2_ids.parquet')
    versions = pl.read_parquet(versions_dir / 'versions.parquet')
    sources = pl.read_parquet(versions_dir / 'sources.parquet')
    live = S.live_segments()
    live_dirs = {S.norm(s.docs.parent): s.name for s in live}
    chosen, decisions, per_version, rows = S.choose(versions, sources, corpus, live)
    log(f'chosen {chosen.height} doc_ids; decisions {decisions.height}')
    picks = pick_sources(chosen, rows, sources)
    if picks.height != chosen.height or picks['doc_id'].n_unique() != picks.height:
        raise ValueError('source pick is not one row per doc_id')
    out.mkdir(parents=True)
    (out / 'vectors').mkdir()
    write_parquet(decisions.to_arrow(), out / 'decisions.parquet')

    url = dict(zip(corpus['id'].to_list(), corpus['url'].to_list()))
    index_names = S.index_dirs_by_name()
    doc_writer = pq.ParquetWriter(guarded(out / 'docs.parquet'), DOC_SCHEMA, compression='zstd')
    chunk_writer = pq.ParquetWriter(guarded(out / 'chunks_t256.parquet'), CHUNK_SCHEMA, compression='zstd')
    vec_plan, reembed, extract_docs = [], [], []
    n_docs = n_chunks = n_vec = n_text_mismatch = 0
    s25 = {'next': 0}
    files = picks.partition_by('path', as_dict=True)
    for k, ((path,), part) in enumerate(sorted(files.items())):
        kind = part['kind'][0]
        seg = segment_of(path, live_dirs) if kind == 'segment-docs' else ''
        meta = {r['doc_id']: r for r in part.to_dicts()}
        chunked = {d for d, r in meta.items() if r['rule'] == 'a' or r['quality'] == 'good'}
        job = None
        if kind == 'segment-docs' and part['has_chunks'][0]:
            job = SegmentChunks(part['chunks_path'][0], seg, index_names, s25)
        elif chunked:
            extract_docs += [(path, d, meta[d]['sha256']) for d in sorted(chunked)]
        for batch_rows in iter_texts(path, kind, set(meta)):
            batch, groups, texts = [], defaultdict(list), {}
            for ids, phys, text in batch_rows:
                sha = hashlib.sha256(text.encode('utf-8')).digest()
                for d in ids:
                    r = meta[d]
                    if sha != r['sha256']:
                        raise ValueError(f'text changed since catalog: {path} doc {d}')
                    batch.append({'doc_id': d, 'lang': r['lang'], 'host': r['host'], 'url': url[d], 'phys_id': phys,
                                  'doc_text': text, 'body_cp': r['body_cp'], 'han': r['han'],
                                  'quality': r['quality'], 'rule': r['rule'], 'text_sha256': sha.hex(),
                                  'segment': seg, 'source_path': path})
                    if d in chunked:
                        groups[phys].append(d)
                texts[phys] = text
            doc_writer.write_table(pa.Table.from_pylist(batch, schema=DOC_SCHEMA))
            n_docs += len(batch)
            if job is not None and groups:
                crow, vrows, miss = job.emit(groups, texts, reembed)
                if crow:
                    chunk_writer.write_table(pa.Table.from_pylist(crow, schema=CHUNK_SCHEMA))
                n_chunks, n_vec, n_text_mismatch = n_chunks + len(crow), n_vec + len(vrows), n_text_mismatch + miss
                if vrows:
                    vec_plan.append((job.index, job.total, np.asarray(vrows, np.int64)))
        if k % 200 == 0:
            log(f'file {k+1}/{len(files)}: {n_docs} docs, {n_chunks} chunks ({n_vec} with vector)')
    doc_writer.close()
    if n_docs != chosen.height:
        raise ValueError(f'docs written {n_docs} != chosen {chosen.height}')
    log(f'docs {n_docs}; segment chunks {n_chunks}, with vector {n_vec}, index text mismatch {n_text_mismatch}')
    n_new = chunk_extract_docs(extract_docs, chunk_writer, reembed, s25) if extract_docs else 0
    n_chunks += n_new
    log(f'S25-chunked {len(extract_docs)} extract-only docs -> {n_new} chunks (all to reembed)')
    chunk_writer.close()
    write_parquet(pa.Table.from_pylist(reembed) if reembed else pa.table({'chunk_id': pa.array([], pa.int64())}),
                  out / 'reembed.parquet')
    vec_plan = merge_plan(vec_plan)
    if vectors:
        write_vectors(out / 'vectors', vec_plan, n_vec, log)

    file_sha = {p: sha_file(Path(p)) for p in picks['path'].unique().to_list()}
    prov = picks.select('doc_id', 'lang', 'host', 'rule', 'quality', pl.col('path').alias('source_path'), 'kind',
                        'phys_id', 'extractor', 'status', 'live',
                        pl.col('sha256').bin.encode('hex').alias('text_sha256'))
    prov = prov.with_columns(pl.col('source_path').replace_strict(file_sha, return_dtype=pl.Utf8)
                             .alias('source_file_sha256'))
    prov = prov.join(raw_pointers(), on='doc_id', how='left')
    write_parquet(prov.to_arrow(), out / 'provenance.parquet')
    counts = {'docs': n_docs, 'chunks': n_chunks, 'chunks_with_vector': n_vec, 'reembed': len(reembed),
              'index_text_mismatch': n_text_mismatch, 's25_new_chunks': n_new, 'decisions': decisions.height,
              'rules': {r: n for r, n in chosen.group_by('rule').len().iter_rows()},
              'quality': {q: n for q, n in chosen.group_by('quality').len().iter_rows()},
              'by_lang': {l: n for l, n in chosen.group_by('lang').len().iter_rows()}}
    manifest = {
        'schema': 'r2ai.canonical.v1', 'built_utc7': time.strftime('%Y-%m-%d %H:%M:%S'), 'counts': counts,
        'inputs': {str(p): sha_file(p) for p in (S.DATA / 'corpus_v2_ids.parquet', S.VI_MANIFEST, S.ZH_MANIFEST,
                                                 S.P1_FACADE, S.S22_SUPERSEDED, versions_dir / 'versions.parquet',
                                                 versions_dir / 'sources.parquet')},
        'live_segments': [{'name': s.name, 'lang': s.lang, 'docs': str(s.docs), 'superseded': len(s.superseded)}
                          for s in live],
        'vector_segments': [{'index': str(p), 'rows': int(len(r))} for p, _, r in vec_plan],
        'rules': S.__doc__,
    }
    manifest['files'] = {str(p.relative_to(out)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha_file(p)}
                         for p in sorted(out.rglob('*')) if p.is_file()}
    guarded(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    log('done')
    return counts


def chunk_extract_docs(items, writer, reembed, s25) -> int:
    """t256 chunking with the production chunker for chosen extract-only versions (S25 segment 25000)."""
    from transformers import AutoTokenizer
    from r2ai.index.chunk import chunk_paragraphs, doc_layout
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    ntok = lambda x: len(tok(x, add_special_tokens=False)['input_ids'])
    gs = global_start_of(f'seg{S25_SEGMENT:05d}')
    by_file = defaultdict(dict)
    for path, d, sha in items:
        by_file[path][d] = sha
    n = 0
    for path, wanted in sorted(by_file.items()):
        rows = []
        for r in pq.read_table(path).to_pylist():
            ids = sorted({int(i) for i in (r.get('doc_ids') or []) if int(i) in wanted})
            if not ids:
                continue
            r = {**r, 'body': r['body'] or '', 'paragraphs': r['paragraphs'] or []}
            doc_text, spans, paras = doc_layout(r)
            if hashlib.sha256(doc_text.encode('utf-8')).digest() != wanted[ids[0]]:
                raise ValueError(f'extract text drift {path} {ids[0]}')
            ptoks = [ntok(doc_text[s:e]) for _, s, e in paras]
            for f, s, e in list(spans) + chunk_paragraphs(doc_text, paras, ptoks, 256, ntok):
                gid = gs + s25['next']
                s25['next'] += 1
                rows.append({'chunk_id': gid, 'doc_id': ids[0], 'doc_ids': ids, 'field': f, 'char_start': s,
                             'char_end': e, 'text': doc_text[s:e], 'n_tokens': ntok(doc_text[s:e]),
                             'segment': S25_NAME, 'has_vector': False})
                reembed.append({'chunk_id': gid, 'doc_id': ids[0], 'segment': S25_NAME, 'field': f,
                                'reason': 'new S25 chunk'})
                n += 1
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA))
    return n


def write_vectors(vdir: Path, plan, n_rows: int, log) -> None:
    """Concatenate selected rows: dense.npy fp16 [n,1024], CSR data f32 / indices i32 / indptr i64, source rows."""
    dense_f = open(guarded(vdir / 'dense.npy'), 'xb')
    np.lib.format.write_array_header_2_0(dense_f, {'descr': np.dtype(np.float16).str, 'fortran_order': False,
                                                   'shape': (n_rows, 1024)})
    files = {k: open(guarded(vdir / f'{k}.bin'), 'xb') for k in ('data', 'indices', 'indptr', 'source_row')}
    np.array([0], np.int64).tofile(files['indptr'])
    nnz = written = 0
    for idx_path, total, rows in plan:
        ref = IndexRef(idx_path, total)
        for a in range(0, len(rows), 65536):
            r = rows[a:a + 65536]
            np.asarray(ref.dense[r], dtype=np.float16).tofile(dense_f)
            lo, hi = np.asarray(ref.indptr[r], np.int64), np.asarray(ref.indptr[r + 1], np.int64)
            lens = hi - lo
            sel = np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(lo - (np.cumsum(lens) - lens), lens)
            np.asarray(ref.data[sel], np.float32).tofile(files['data'])
            np.asarray(ref.indices[sel], np.int32).tofile(files['indices'])
            (np.cumsum(lens) + nnz).astype(np.int64).tofile(files['indptr'])
            r.astype(np.int64).tofile(files['source_row'])
            nnz += int(lens.sum())
            written += len(r)
        log(f'vectors {written}/{n_rows} ({idx_path.name})')
    for f in (dense_f, *files.values()):
        f.close()
    if written != n_rows:
        raise ValueError('vector row count differs from plan')
    guarded(vdir / 'vectors.json').write_text(json.dumps({
        'format': 'r2ai.canonical-vectors.v1', 'rows': n_rows, 'dimension': 1024, 'vocabulary': 250002, 'nnz': nnz,
        'model': 'BAAI/bge-m3 (dense CLS L2 fp16; sparse relu(sparse_linear) max per token id)',
        'alignment': 'row i = i-th chunks_t256.parquet row with has_vector=True, in file order',
        'segments': [{'index': str(p), 'rows': int(len(r))} for p, _, r in plan]}, indent=2), encoding='utf-8')


def raw_pointers() -> pl.DataFrame:
    """doc_id -> raw provenance (WARC record / raw shard) from every provenance*.parquet that has doc_id."""
    frames = []
    for root in (S.DATA, S.OUT / 'runs'):
        for p in root.rglob('provenance*.parquet'):
            names = pq.read_schema(p).names
            if 'doc_id' not in names:
                continue
            cols = [c for c in ('doc_id', 'warc_filename', 'warc_offset', 'warc_length', 'raw_shard', 'source')
                    if c in names]
            df = pl.read_parquet(p, columns=cols).with_columns(pl.lit(str(p)).alias('raw_provenance_file'))
            df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in ('warc_filename', 'raw_shard', 'source') if c in df.columns]
                                 + [pl.col(c).cast(pl.Int64) for c in ('warc_offset', 'warc_length') if c in df.columns])
            frames.append(df.with_columns(pl.col('doc_id').cast(pl.Int64)))
    if not frames:
        return pl.DataFrame({'doc_id': pl.Series([], dtype=pl.Int64)})
    return (pl.concat(frames, how='diagonal_relaxed').sort('raw_provenance_file', descending=True)
            .unique('doc_id', keep='first'))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--versions', default=str(RUN_ROOT / 'versions'))
    ap.add_argument('--out', default=str(CANONICAL_ROOT))
    ap.add_argument('--no-vectors', action='store_true', help='dry layout run: skip the vector copy')
    a = ap.parse_args(argv)
    print(json.dumps(build(resolve_path(a.versions), resolve_path(a.out), vectors=not a.no_vectors), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
