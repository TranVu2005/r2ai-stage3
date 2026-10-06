"""Immutable CPU dense/CSR/text segments with exact global chunk-ID top-k.

Dense and CSR blocks are read into private buffers, never whole-file mmap views.
Appending a segment leaves existing files unchanged. Virtual segments only hold
read-only ranges of an existing index; cuts follow its doc-aligned scan blocks.
Tie order is (score descending, global chunk ID descending), as retrieve.exact.
This module is deliberately separate from the production retrieval pipeline.
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, WORK_DATA_DIR, assert_writable, resolve_path

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from r2ai.retrieve.exact import QuerySparse, dense_topk, doc_aligned_bounds, sparse_topk

OUTPUT_ROOT = RUNS_DIR / 'zh-full' / 'segments'
ZH_INDEX_ROOT = WORK_DATA_DIR / 'index' / 'zh_full_segments'


def guarded_output(path: str | Path) -> Path:
    """Preflight guard for every runtime destination, including overrides."""
    path = assert_writable(path)
    if not path.is_relative_to(resolve_path(OUTPUT_ROOT)):
        raise ValueError(f'Segment output must stay below {OUTPUT_ROOT}')
    return path


def guarded_segment_output(path: str | Path) -> Path:
    """Immutable ingest artifacts use their private ZH index or runtime root only."""
    path = assert_writable(path)
    if not any(path.is_relative_to(resolve_path(root)) for root in (OUTPUT_ROOT, ZH_INDEX_ROOT)):
        raise ValueError(f'Segment index must stay below {ZH_INDEX_ROOT} or {OUTPUT_ROOT}')
    return path


class ArrayRows:
    """Read-only range; mmap reads use fromfile so scanned pages do not accumulate."""

    def __init__(self, array, start=0, stop=None, subtract=0):
        self.array = array
        self.start = int(start)
        self.stop = len(array) if stop is None else int(stop)
        self.subtract = subtract
        self.shape = (self.stop - self.start,) + array.shape[1:]
        self.dtype = array.dtype
        self.file_offset = None
        if isinstance(array, np.memmap):
            if not array.flags.c_contiguous:
                raise ValueError('disk-backed row ranges must be C contiguous')
            base = array
            while isinstance(base.base, np.memmap):
                base = base.base
            self.file_offset = int(base.offset) + int(array.ctypes.data-base.ctypes.data)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            lo, hi, step = key.indices(len(self))
            if step != 1:
                return self[np.arange(lo, hi, step)]
            lo, hi = self.start + lo, self.start + hi
            a = self.array
            if isinstance(a, np.memmap):
                width = int(np.prod(a.shape[1:], dtype=np.int64))
                with open(a.filename, 'rb') as f:
                    f.seek(self.file_offset + lo * width * a.itemsize)
                    value = np.fromfile(f, dtype=a.dtype, count=(hi - lo) * width)
                value = value.reshape((hi - lo,) + a.shape[1:])
            else:
                value = np.asarray(a[lo:hi])
        else:
            keys = np.asarray(key)
            if np.any(keys < 0) or np.any(keys >= len(self)):
                raise IndexError('row outside segment')
            value = np.asarray(self.array[self.start + keys])
        return value - self.subtract if self.subtract else value


@dataclass
class Segment:
    dense: object
    sparse: tuple
    doc_id: object
    global_start: int
    row_start: int = 0
    path: Path | None = None
    text_path: Path | None = None

    def __len__(self):
        return len(self.dense)


def virtual_segments(dense, sparse, doc_id, *, count=4, block_rows=16384, global_start=0):
    """Partition existing arrays without writing/copying their index files.

    Cuts are a subset of baseline block boundaries, preserving float32 kernel
    shapes and sparse summation order for a fixed query batch and block size.
    """
    if count < 1 or block_rows < 1 or len(dense) != len(doc_id):
        raise ValueError('invalid segment count, block size or row alignment')
    data, indices, indptr = (sparse.data, sparse.indices, sparse.indptr) if hasattr(sparse, 'indptr') else sparse
    if len(indptr) != len(dense) + 1:
        raise ValueError('CSR row count differs from dense')
    bounds = doc_aligned_bounds(doc_id, block_rows)
    if count > len(bounds) - 1:
        raise ValueError('fewer doc-aligned blocks than requested segments')
    cuts = bounds[np.rint(np.linspace(0, len(bounds)-1, count+1)).astype(np.int64)]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        begin, end = int(indptr[lo]), int(indptr[hi])
        out.append(Segment(ArrayRows(dense, lo, hi),
                           (ArrayRows(data, begin, end), ArrayRows(indices, begin, end),
                            ArrayRows(indptr, lo, hi+1, subtract=begin)),
                           np.asarray(doc_id[lo:hi]), global_start+int(lo), row_start=int(lo)))
    return out


def merge_topk(results, k):
    """Merge per-query segment top-k; retain finite scores and global tie order."""
    if k < 1:
        raise ValueError('k must be positive')
    results = list(results)
    if not results:
        return [], []
    nq = len(results[0][0])
    ids, scores = [], []
    for j in range(nq):
        ci = np.concatenate([r[0][j] for r in results]).astype(np.int64, copy=False)
        cv = np.concatenate([r[1][j] for r in results]).astype(np.float32, copy=False)
        finite = np.isfinite(cv) & (ci >= 0)
        ci, cv = ci[finite], cv[finite]
        order = np.lexsort((-ci, -cv))[:k]
        ids.append(ci[order])
        scores.append(cv[order])
    return ids, scores


def _validate_segments(segments):
    if not segments:
        raise ValueError('at least one segment is required')
    end = -1
    dim = segments[0].dense.shape[1]
    for s in sorted(segments, key=lambda s: s.global_start):
        if s.global_start < end or s.global_start < 0 or s.dense.shape[1] != dim:
            raise ValueError('overlapping/negative global IDs or incompatible dense dimensions')
        if len(s.doc_id) != len(s) or len(s.sparse[2]) != len(s) + 1:
            raise ValueError('segment row counts differ')
        end = s.global_start + len(s)


def scan_topk(segments, qd, qsp, k=200, *, block_rows=16384, query_batch=128, progress=None):
    """Exact CPU scan of every segment then global merge, bounded private buffers.

    Every dense block is read once; public exact scanners handle each query batch.
    Returns independent dense/sparse candidate IDs and scores plus measured times.
    """
    segments = list(segments)
    _validate_segments(segments)
    if block_rows < 1 or query_batch < 1 or len(qd) != len(qsp):
        raise ValueError('invalid blocks, query batch or query alignment')
    nq = len(qd)
    empty = lambda: ([np.zeros(0, np.int64) for _ in range(nq)],
                     [np.zeros(0, np.float32) for _ in range(nq)])
    total_d, total_s = empty(), empty()
    dense_seconds = sparse_seconds = 0.0
    started = time.perf_counter()
    vq = QuerySparse(qsp)
    scanned = 0
    for segno, s in enumerate(segments):
        local_d, local_s = empty(), empty()
        bounds = doc_aligned_bounds(s.doc_id, block_rows)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            t0 = time.perf_counter()
            # Convert exactly once before public dense_topk's bounded copies.
            block = np.asarray(s.dense[int(lo):int(hi)], dtype=np.float32)
            block_result = empty()
            for a in range(0, nq, query_batch):
                ids, scores = dense_topk(block, qd[a:a+query_batch], k, block_rows=len(block))
                block_result[0][a:a+query_batch] = [i+s.global_start+int(lo) for i in ids]
                block_result[1][a:a+query_batch] = scores
            del block
            local_d = merge_topk([local_d, block_result], k)
            dense_seconds += time.perf_counter()-t0
            t0 = time.perf_counter()
            data, indices, indptr = s.sparse
            # CSR int64 slices may alias a caller's read-only/input array.
            ip = np.array(indptr[int(lo):int(hi)+1], dtype=np.int64, copy=True)
            begin, end = int(ip[0]), int(ip[-1])
            values = np.asarray(data[begin:end], dtype=np.float32)
            cols = np.asarray(indices[begin:end], dtype=np.int32)
            ip -= begin
            ids, scores = sparse_topk(values, cols, ip, vq, k, block_rows=int(hi-lo))
            local_s = merge_topk([local_s, ([i+s.global_start+int(lo) for i in ids], scores)], k)
            del values, cols, ip
            sparse_seconds += time.perf_counter()-t0
            scanned += int(hi-lo)
            if progress is not None:
                progress({'segment': segno, 'rows_scanned': scanned,
                          'dense_seconds': dense_seconds, 'sparse_seconds': sparse_seconds})
        total_d = merge_topk([total_d, local_d], k)
        total_s = merge_topk([total_s, local_s], k)
    return {'dense': total_d, 'sparse': total_s,
            'stats': {'dense_seconds': dense_seconds, 'sparse_seconds': sparse_seconds,
                      'elapsed_seconds': time.perf_counter()-started, 'rows': scanned,
                      'queries': nq, 'segments': len(segments), 'block_rows': block_rows,
                      'query_batch': query_batch}}


class SegmentWriter:
    """Streaming writer: one immutable ingest batch, dense/CSR/text row aligned.

    Manifest is written last. Incomplete directories are never loadable, and an
    existing directory is refused rather than overwritten or reassembled.
    """
    def __init__(self, root, name, *, rows, dimension, vocabulary, global_start):
        import pyarrow as pa
        if Path(name).name != name or not name or name in ('.', '..'):
            raise ValueError('segment name must be a single path component')
        if rows < 1 or dimension < 1 or vocabulary < 1 or global_start < 0:
            raise ValueError('invalid segment shape/global start')
        self.path = guarded_segment_output(Path(root) / name)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows, self.dimension, self.vocabulary, self.global_start = rows, dimension, vocabulary, global_start
        self.written = self.nnz = 0
        self.files = {}
        for key in ('dense.npy', 'data.bin', 'indices.bin', 'indptr.bin', 'doc_id.bin'):
            self.files[key] = open(self.path/key, 'xb')
        np.lib.format.write_array_header_2_0(self.files['dense.npy'],
            {'descr': np.dtype(np.float16).str, 'fortran_order': False, 'shape': (rows, dimension)})
        np.array([0], np.int64).tofile(self.files['indptr.bin'])
        self.text_file = pa.OSFile(str(self.path/'text.arrow'), 'wb')
        self.text_writer = pa.ipc.new_file(self.text_file, pa.schema([('text', pa.string())]))

    def append(self, dense, sparse, doc_id, texts):
        import pyarrow as pa
        n = len(dense)
        if sparse.shape != (n, self.vocabulary) or np.shape(dense) != (n, self.dimension) or len(doc_id) != n or len(texts) != n:
            raise ValueError('dense/CSR/doc/text row alignment differs')
        if self.written+n > self.rows:
            raise ValueError('too many rows')
        if not sparse.has_sorted_indices or not sparse.has_canonical_format:
            raise ValueError('CSR indices must be sorted and unique')
        np.asarray(dense, dtype=np.float16).tofile(self.files['dense.npy'])
        np.asarray(sparse.data, dtype=np.float32).tofile(self.files['data.bin'])
        np.asarray(sparse.indices, dtype=np.int32).tofile(self.files['indices.bin'])
        (np.asarray(sparse.indptr[1:], dtype=np.int64)+self.nnz).tofile(self.files['indptr.bin'])
        np.asarray(doc_id, dtype=np.int64).tofile(self.files['doc_id.bin'])
        self.text_writer.write_table(pa.table({'text': pa.array(texts, type=pa.string())}))
        self.written += n
        self.nnz += sparse.nnz

    def close(self):
        self.text_writer.close()
        self.text_file.close()
        for f in self.files.values():
            f.close()
        if self.written != self.rows:
            raise ValueError('incomplete segment: manifest not published')
        meta = {'format': 1, 'rows': self.rows, 'dimension': self.dimension,
                'vocabulary': self.vocabulary, 'global_start': self.global_start, 'nnz': int(self.nnz)}
        with open(self.path/'segment.json', 'x', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        return load_segment(self.path)


def write_segment(root, name, dense, sparse, doc_id, texts, *, global_start):
    """Write one in-memory ingest batch. Large ingests use SegmentWriter.append."""
    if len(doc_id) != len(dense) or len(texts) != len(dense) or sparse.shape[0] != len(dense):
        raise ValueError('dense/CSR/doc/text row alignment differs')
    writer = SegmentWriter(root, name, rows=len(dense), dimension=dense.shape[1],
                           vocabulary=sparse.shape[1], global_start=global_start)
    writer.append(dense, sparse, doc_id, texts)
    return writer.close()


def load_segment(path):
    """Open an existing immutable ingest segment read-only."""
    path = resolve_path(path)
    meta = json.loads((path/'segment.json').read_text(encoding='utf-8'))
    n, nnz = meta['rows'], meta['nnz']
    dense = np.load(path/'dense.npy', mmap_mode='r')
    raw = lambda name, dtype, length: (np.memmap(path/name, mode='r', dtype=dtype, shape=(length,))
                                       if length else np.zeros(0, dtype=dtype))
    return Segment(ArrayRows(dense),
        (ArrayRows(raw('data.bin', np.float32, nnz)), ArrayRows(raw('indices.bin', np.int32, nnz)),
         ArrayRows(raw('indptr.bin', np.int64, n+1))),
        raw('doc_id.bin', np.int64, n), meta['global_start'], path=path, text_path=path/'text.arrow')


def take_segment_texts(segments, ids):
    """Retrieve global IDs across text segments, preserving caller order/duplicates."""
    import pyarrow as pa
    from r2ai.retrieve.exact import take_texts
    segments = list(segments)
    _validate_segments(segments)
    ids = np.asarray(ids, dtype=np.int64)
    out = [None]*len(ids)
    for s in segments:
        pos = np.flatnonzero((ids >= s.global_start) & (ids < s.global_start+len(s)))
        if len(pos):
            if s.text_path is None:
                raise ValueError('segment has no text artifact')
            with pa.memory_map(str(s.text_path), 'r') as source:
                col = pa.ipc.open_file(source).read_all().column('text')
                texts = take_texts(col, ids[pos]-s.global_start)
                for p, text in zip(pos, texts):
                    out[p] = text
                del col
    if any(t is None for t in out):
        raise IndexError('global chunk ID not covered by segments')
    return out


def memory_metrics():
    """Native process peaks on Windows; current private bytes reported separately."""
    import psutil
    mi = psutil.Process().memory_info()
    return {'peak_rss_gib': getattr(mi, 'peak_wset', mi.rss)/2**30,
            'rss_gib': mi.rss/2**30,
            'peak_private_gib': getattr(mi, 'peak_pagefile', None)/2**30
                if getattr(mi, 'peak_pagefile', None) is not None else None,
            'private_gib': getattr(mi, 'private', None)/2**30
                if getattr(mi, 'private', None) is not None else None}


def scan_memory_budget(*, dimension, queries, block_rows, query_batch,
                       max_block_nnz=0, query_vocabulary=0, metadata_rows=0,
                       writer_block=0, k=200):
    """Conservative private-buffer estimate, independent of dense disk payload.

    Dense: fp16 read + float32 conversion + public scanner copy (10 bytes/value),
    scores and int64 selection. Sparse: 48 bytes/nnz for restricted CSR copies,
    row/column remaps/masks; both score matrix layouts plus selection scratch.
    Add query matrices, all doc metadata, eight running/result top-k copies,
    and 256 MiB for Python/Arrow/BLAS. Largest doc-aligned block/CSR nnz must be
    supplied for VI; a very large document cannot silently defeat the guard.
    """
    if min(dimension, queries, block_rows, query_batch) < 1 or min(max_block_nnz, query_vocabulary, metadata_rows, writer_block) < 0:
        raise ValueError('invalid memory budget sizes')
    dense_phase = block_rows*dimension*10 + block_rows*min(queries, query_batch)*4 + block_rows*min(queries, query_batch, 256)*9
    sparse_phase = max_block_nnz*48 + block_rows*queries*8 + block_rows*min(queries, 256)*9
    writer_phase = writer_block*dimension*10 + max_block_nnz*48 + writer_block*256
    components = {'runtime_reserve': 256*2**20,
                  'query_matrices': query_vocabulary*queries*4 + queries*dimension*4,
                  'doc_metadata': metadata_rows*8,
                  'topk_results': queries*k*12*8,
                  'dense_phase': dense_phase, 'sparse_phase': sparse_phase,
                  'writer_phase': writer_phase}
    peak = sum(components[x] for x in ('runtime_reserve', 'query_matrices', 'doc_metadata', 'topk_results')) + max(dense_phase, sparse_phase, writer_phase)
    return {'estimated_peak_private_bytes': int(peak), 'components_bytes': components,
            'largest_block_rows': block_rows, 'largest_block_nnz': max_block_nnz}


def require_scan_memory(budget):
    """No CLI bypass: require >=2 GiB free, budget+1 GiB reserve, and <=1.5 GiB private."""
    import psutil
    available = psutil.virtual_memory().available/2**30
    peak = budget['estimated_peak_private_bytes']/2**30
    if peak > 1.5:
        raise RuntimeError(f'Estimated private buffers {peak:.3f} GiB exceed 1.5 GiB; reduce block/query batch sizes')
    minimum = max(2.0, peak+1.0)
    if available < minimum:
        raise RuntimeError(f'CPU scan requires >={minimum:.3f} GiB available memory; measured {available:.3f} GiB; bounded buffers {peak:.3f} GiB')
    return {**budget, 'estimated_peak_private_gib': peak,
            'available_memory_gib': available, 'minimum_available_gib': minimum}


def _save_json(path, value):
    # Parent has already passed guarded_output's recursive preflight.
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def synthetic_benchmark(root, *, rows=8_000_000, dimension=1024, queries=1200,
                        segment_rows=1_000_000, block_rows=16384, writer_block=8192,
                        query_batch=128, seed=42, vocabulary=8192, tokens_per_row=8):
    """Actually write and scan synthetic fp16/CSR/text data (no sparse files/model).

    Generation, writing and scanning are timed; the report contains physical
    file sizes and native Windows process peaks. No automatic deletion.
    """
    import psutil
    from scipy.sparse import csr_matrix
    from threadpoolctl import threadpool_info, threadpool_limits
    if min(rows, dimension, queries, segment_rows, block_rows, writer_block, query_batch) < 1:
        raise ValueError('benchmark sizes must be positive')
    if tokens_per_row < 1 or tokens_per_row > vocabulary:
        raise ValueError('invalid sparse token count')
    largest_block = min(segment_rows, rows, block_rows+3)  # synthetic docs have four chunks
    budget = scan_memory_budget(dimension=dimension, queries=queries, block_rows=largest_block,
        query_batch=query_batch, max_block_nnz=largest_block*tokens_per_row,
        query_vocabulary=min(vocabulary, queries*16), metadata_rows=rows,
        writer_block=min(writer_block, rows))
    preflight = require_scan_memory(budget)
    available = preflight['available_memory_gib']
    root = guarded_output(root)
    root.mkdir(parents=True, exist_ok=False)
    _save_json(root/'SYNTHETIC_MARKER.json', {'owner': 'r2ai.index.segments', 'root': str(root),
        'seed': seed, 'rows': rows, 'dimension': dimension, 'queries': queries})
    rng = np.random.default_rng(seed)
    segments = []
    generation = 0.0
    write_started = time.perf_counter()
    with threadpool_limits(limits=4, user_api='blas'):
        pools = threadpool_info()
        for start in range(0, rows, segment_rows):
            n = min(segment_rows, rows-start)
            w = SegmentWriter(root, f's{len(segments):03d}', rows=n, dimension=dimension,
                              vocabulary=vocabulary, global_start=start)
            for local in range(0, n, writer_block):
                b = min(writer_block, n-local)
                t0 = time.perf_counter()
                d = rng.standard_normal((b, dimension), dtype=np.float32)
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                d = d.astype(np.float16)
                global_ids = np.arange(start+local, start+local+b, dtype=np.int64)
                # Sorted unique CSR tokens; nonzero float32 weights, deterministic.
                columns = np.sort((global_ids[:, None]*tokens_per_row+
                    np.arange(tokens_per_row)) % vocabulary, axis=1).astype(np.int32)
                weights = rng.random(b*tokens_per_row, dtype=np.float32)
                csr = csr_matrix((weights, columns.ravel(),
                                  np.arange(0, (b+1)*tokens_per_row, tokens_per_row, dtype=np.int64)),
                                 shape=(b, vocabulary))
                texts = [f'synthetic-seed{seed}-chunk={i}' for i in global_ids]
                generation += time.perf_counter()-t0
                w.append(d, csr, global_ids//4, texts)
                del d, csr, columns, weights, texts
            segments.append(w.close())
            _save_json(root/'writer-progress.json', {'rows_written': start+n,
                'elapsed_seconds': time.perf_counter()-write_started, 'memory': memory_metrics()})
        writer_seconds = time.perf_counter()-write_started
        writer_memory = memory_metrics()
        qd = rng.standard_normal((queries, dimension), dtype=np.float32)
        qd /= np.linalg.norm(qd, axis=1, keepdims=True)
        qd = qd.astype(np.float16)
        qsp = []
        for _ in range(queries):
            tokens = np.sort(rng.choice(vocabulary, min(16, vocabulary), replace=False)).astype(np.int32)
            qsp.append((tokens, rng.random(len(tokens), dtype=np.float32)))
        last_log = 0.0
        def progress(st):
            nonlocal last_log
            if time.perf_counter()-last_log >= 10 or st['rows_scanned'] == rows:
                _save_json(root/'scan-progress.json', {**st, 'memory': memory_metrics()})
                last_log = time.perf_counter()
        result = scan_topk(segments, qd, qsp, min(200, rows), block_rows=block_rows,
                          query_batch=query_batch, progress=progress)
        sample_ids = [0, rows//2, rows-1, 0]
        texts = take_segment_texts(segments, sample_ids)
        text_verified = texts == [f'synthetic-seed{seed}-chunk={i}' for i in sample_ids]
        dense_bytes = sum((s.path/'dense.npy').stat().st_size for s in segments)
        file_bytes = sum(p.stat().st_size for s in segments for p in s.path.iterdir())
        # Dense format stores exactly a header plus the complete dense payload.
        actual_payload = sum(np.prod(s.dense.shape, dtype=np.int64)*s.dense.dtype.itemsize for s in segments)
        if actual_payload != rows*dimension*2 or result['stats']['rows'] != rows:
            raise AssertionError('synthetic dense size/full scan mismatch')
        np.savez(root/'topk.npz', dense_ids=np.stack(result['dense'][0]),
                 dense_scores=np.stack(result['dense'][1]))
        report = {'seed': seed, 'rows': rows, 'dimension': dimension, 'queries': queries,
                  'segments': len(segments), 'dense_payload_bytes': int(actual_payload),
                  'dense_disk_bytes': dense_bytes, 'total_segment_disk_bytes': file_bytes,
                  'rows_scanned': result['stats']['rows'], 'all_dense_files_read': True,
                  'text_roundtrip_verified': text_verified, 'generation_seconds': generation,
                  'writer_elapsed_seconds': writer_seconds, 'writer_memory': writer_memory,
                  'scan': result['stats'], 'memory': memory_metrics(),
                  'available_memory_gib_before_run': available, 'memory_preflight': preflight,
                  'blas_thread_limit': 4,
                  'threadpools': pools, 'process_id': psutil.Process().pid}
        _save_json(root/'benchmark.json', report)
        return report


def _source_inventory(paths):
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths}


def vi_gate(index_dir, chunks_file, query_cache, root, *, block_rows=16384, query_batch=128, k=200):
    """Full cached-query gate, one vs four virtual segments, index strictly read-only."""
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_info, threadpool_limits
    from r2ai.paths import require_inputs
    from r2ai.retrieve.exact import load_sparse_mmap
    index_dir, chunks_file, query_cache = map(resolve_path, (index_dir, chunks_file, query_cache))
    cache = index_dir/'sparse_mmap'
    source_files = [index_dir/'dense.npy', chunks_file, query_cache, cache/'shape.json'] + [cache/f'{a}.npy' for a in ('data', 'indices', 'indptr')]
    require_inputs(*source_files)
    before = _source_inventory(source_files)
    with open(query_cache, 'rb') as f:
        q = pickle.load(f)
    if len(q['qd']) != 1200 or len(q['qsp']) != 1200:
        raise ValueError('VI gate requires all 1200 cached query vectors')
    dense = np.load(index_dir/'dense.npy', mmap_mode='r')
    data, indices, indptr, shape = load_sparse_mmap(cache)
    docs = pq.read_table(chunks_file, columns=['doc_id'])['doc_id'].to_numpy()
    if len(docs) != len(dense) or shape[0] != len(dense):
        raise ValueError('index/chunks bundle row mismatch')
    bounds = doc_aligned_bounds(docs, block_rows)
    largest_block = int(np.diff(bounds).max())
    largest_nnz = max(int(indptr[int(hi)])-int(indptr[int(lo)]) for lo, hi in zip(bounds[:-1], bounds[1:]))
    query_vocabulary = len(np.unique(np.concatenate([np.asarray(t, dtype=np.int64) for t, _ in q['qsp']])))
    budget = scan_memory_budget(dimension=dense.shape[1], queries=1200, block_rows=largest_block,
        query_batch=query_batch, max_block_nnz=largest_nnz, query_vocabulary=query_vocabulary,
        metadata_rows=len(docs), k=k)
    preflight = require_scan_memory(budget)
    available = preflight['available_memory_gib']
    root = guarded_output(root)
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    with threadpool_limits(limits=4, user_api='blas'):
        pools = threadpool_info()
        for count in (1, 4):
            parts = virtual_segments(dense, (data, indices, indptr), docs, count=count, block_rows=block_rows)
            last_log = 0.0
            def progress(st):
                nonlocal last_log
                if time.perf_counter()-last_log >= 10 or st['rows_scanned'] == len(dense):
                    _save_json(root/f'progress-{count}.json', {**st, 'memory': memory_metrics()})
                    last_log = time.perf_counter()
            result = scan_topk(parts, q['qd'], q['qsp'], k, block_rows=block_rows,
                              query_batch=query_batch, progress=progress)
            result['stats']['memory'] = memory_metrics()
            result['stats']['bounds'] = [[s.global_start, s.global_start+len(s)] for s in parts]
            with open(root/f'topk-{count}.pkl', 'xb') as f:
                pickle.dump(result, f)
            runs.append(result)
        differences = {}
        for key in ('dense', 'sparse'):
            same = 0
            max_abs = 0.0
            differing = []
            for j in range(1200):
                ai, av = runs[0][key][0][j], runs[0][key][1][j]
                bi, bv = runs[1][key][0][j], runs[1][key][1][j]
                if np.array_equal(ai, bi):
                    same += 1
                    if len(av):
                        max_abs = max(max_abs, float(np.max(np.abs(av-bv))))
                else:
                    differing.append(j)
            differences[key] = {'ids_equal_queries': same, 'max_abs_score_delta_same_ids': max_abs,
                                'differing_query_positions': differing}
        union_same = sum(np.array_equal(np.union1d(runs[0]['dense'][0][j], runs[0]['sparse'][0][j]),
                            np.union1d(runs[1]['dense'][0][j], runs[1]['sparse'][0][j])) for j in range(1200))
        after = _source_inventory(source_files)
        report = {'queries': 1200, 'rows': len(dense), 'comparisons': differences,
                  'union_ids_equal_queries': union_same, 'one_segment': runs[0]['stats'],
                  'four_segments': runs[1]['stats'], 'source_inventory_before': before,
                  'source_inventory_after': after, 'source_stat_unchanged': before == after,
                  'available_memory_gib_before_run': available, 'memory_preflight': preflight,
                  'blas_thread_limit': 4, 'threadpools': pools,
                  'passed': union_same == 1200 and all(x['ids_equal_queries'] == 1200 for x in differences.values()) and before == after}
        _save_json(root/'gate.json', report)
        return report


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    bench = sub.add_parser('benchmark')
    bench.add_argument('--out-dir', default=str(OUTPUT_ROOT/'synthetic'))
    bench.add_argument('--rows', type=int, default=8_000_000)
    bench.add_argument('--dimension', type=int, default=1024)
    bench.add_argument('--queries', type=int, default=1200)
    bench.add_argument('--segment-rows', type=int, default=1_000_000)
    bench.add_argument('--writer-block', type=int, default=8192)
    bench.add_argument('--seed', type=int, default=42)
    gate = sub.add_parser('gate')
    gate.add_argument('--index-dir', required=True)
    gate.add_argument('--chunks', required=True)
    gate.add_argument('--query-cache', required=True)
    gate.add_argument('--out-dir', default=str(OUTPUT_ROOT/'vi-gate'))
    for parser in (bench, gate):
        parser.add_argument('--block-rows', type=int, default=16384)
        parser.add_argument('--query-batch', type=int, default=128)
    a = p.parse_args(argv)
    if a.command == 'benchmark':
        report = synthetic_benchmark(a.out_dir, rows=a.rows, dimension=a.dimension, queries=a.queries,
            segment_rows=a.segment_rows, writer_block=a.writer_block, seed=a.seed,
            block_rows=a.block_rows, query_batch=a.query_batch)
    else:
        report = vi_gate(a.index_dir, a.chunks, a.query_cache, a.out_dir,
                        block_rows=a.block_rows, query_batch=a.query_batch)
        if not report['passed']:
            print(json.dumps(report, indent=2))
            return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
