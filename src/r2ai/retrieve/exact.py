"""Exact first-stage candidates by block scan over the memory-mapped index (no FAISS, no full CSC).

For the fixed offline query set (all queries at once) this replaces, in retrieve.run.Index.candidates:
  dense : FAISS top-k over float32(dense.npy)   -> dense_topk: Q @ block.T per row block of dense.npy (mmap)
  sparse: CSC column slice @ query weights       -> sparse_topk: per row block of the CSR (mmap cache), columns
                                                    remapped to the union of query tokens V_q, block @ Q_s
Both keep a running top-k per query, ordered by (score desc, row id desc), so ties are deterministic and the
result does not depend on the block size. Higher ids win ties as in faiss IndexFlatIP (measured on the legacy index:
duplicated chunks at the k-th place were the only differences with the lower-id rule). Sparse keeps only positive scores, as the legacy path does.
The union / dense re-score / sparse score of every candidate are computed exactly like Index.candidates.

Sparse mmap cache: split_sparse_npz writes data.npy, indices.npy, indptr.npy, shape.json next to sparse.npz
(one array in RAM at a time); sparse.npz is left unchanged.
"""
from __future__ import annotations

import json
import mmap
import os
import time
from pathlib import Path

import numpy as np

SPARSE_CACHE_FILES = ('data', 'indices', 'indptr')
SEL_ROWS = 256                                      # queries per selection step (bounds the argpartition temp)


def split_sparse_npz(npz_path: str | Path, cache_dir: str | Path) -> dict:
    """sparse.npz (scipy CSR) -> cache_dir/{data,indices,indptr}.npy + shape.json; reuses a complete cache."""
    npz_path, cache_dir = Path(npz_path), Path(cache_dir)
    meta_f = cache_dir / 'shape.json'
    st = npz_path.stat()
    if meta_f.exists():
        info = json.loads(meta_f.read_text(encoding='utf-8'))
        if info.get('source_size') == st.st_size and info.get('source_mtime') == st.st_mtime and \
                all((cache_dir / f'{k}.npy').exists() for k in SPARSE_CACHE_FILES):
            return {**info, 'reused': True}
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = {'source': str(npz_path), 'source_size': st.st_size, 'source_mtime': st.st_mtime}
    with np.load(npz_path) as z:
        fmt = z['format'].item()
        fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
        assert fmt == 'csr', f'expected a CSR sparse.npz, got {fmt}'
        info['shape'] = [int(x) for x in z['shape']]
        for k in SPARSE_CACHE_FILES:                # one array in memory at a time
            a = z[k]
            info[f'{k}_dtype'] = str(a.dtype)
            if k == 'indptr':
                info['nnz'] = int(a[-1])
            tmp = cache_dir / f'{k}.tmp.npy'
            np.save(tmp, a)
            del a
            os.replace(tmp, cache_dir / f'{k}.npy')
    tmp = cache_dir / 'shape.tmp.json'
    tmp.write_text(json.dumps(info, indent=1), encoding='utf-8')
    os.replace(tmp, meta_f)
    return {**info, 'reused': False}


def load_sparse_mmap(cache_dir: str | Path):
    """-> data, indices, indptr (np.memmap, read-only), shape."""
    cache_dir = Path(cache_dir)
    info = json.loads((cache_dir / 'shape.json').read_text(encoding='utf-8'))
    arrs = [np.load(cache_dir / f'{k}.npy', mmap_mode='r') for k in SPARSE_CACHE_FILES]
    return (*arrs, tuple(info['shape']))


def text_mmap(parquet: str | Path, cache: str | Path):
    """Text column of a chunks parquet as a memory-mapped Arrow IPC file (zero-copy, file-backed pages instead of
    ~7 GiB of Arrow pool memory for 3.9M chunks). Built once by streaming row groups; rebuilt if the parquet changes.
    -> (ChunkedArray, info)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet, cache = Path(parquet), Path(cache)
    st = parquet.stat()
    side = cache.with_suffix('.json')
    src = {'source': str(parquet), 'source_size': st.st_size, 'source_mtime': st.st_mtime}
    reused = False
    if cache.exists() and side.exists():
        info = json.loads(side.read_text(encoding='utf-8'))
        reused = all(info.get(k) == v for k, v in src.items())
    if not reused:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix('.tmp.arrow')
        pf = pq.ParquetFile(parquet)
        with pa.OSFile(str(tmp), 'wb') as f, pa.ipc.new_file(f, pa.schema([('text', pa.string())])) as w:
            for i in range(pf.metadata.num_row_groups):            # one row group in memory at a time
                w.write_table(pf.read_row_group(i, columns=['text']))
        os.replace(tmp, cache)
        tmp = side.with_suffix('.tmp.json')
        tmp.write_text(json.dumps({**src, 'rows': pf.metadata.num_rows}, indent=1), encoding='utf-8')
        os.replace(tmp, side)
        del pf
        pa.default_memory_pool().release_unused()
    col = pa.ipc.open_file(pa.memory_map(str(cache), 'r')).read_all().column('text')
    return col, {**src, 'cache': str(cache), 'reused': reused}


def take_texts(col, ids) -> list[str]:
    """col[ids] as Python strings, taken chunk by chunk (ChunkedArray.take concatenates the chunks first, which
    overflows 32-bit string offsets once the column holds more than 2 GiB of text)."""
    ids = np.asarray(ids, dtype=np.int64)
    out: list = [None] * len(ids)
    if not len(ids):
        return []
    bounds = np.cumsum([0] + [len(c) for c in col.chunks])
    ci = np.searchsorted(bounds, ids, side='right') - 1
    for c in np.unique(ci):
        pos = np.flatnonzero(ci == c)
        for k, t in zip(pos, col.chunk(int(c)).take(ids[pos] - bounds[c]).to_pylist()):
            out[k] = t
    return out


def _read_rows(a, lo: int, hi: int) -> np.ndarray:
    """Rows lo..hi-1 of a. For a whole-file np.memmap (np.load mmap_mode='r') read them with a plain file read:
    the copy is private and freed after the block, so a full scan does not keep the mapped file pages in the
    process working set. Other arrays: plain slice."""
    fn = getattr(a, 'filename', None)
    if isinstance(a, np.memmap) and fn and a.flags.c_contiguous and isinstance(a.base, mmap.mmap):   # not a view
        row = int(np.prod(a.shape[1:], dtype=np.int64))
        with open(fn, 'rb') as f:
            f.seek(int(a.offset) + lo * row * a.itemsize)
            out = np.fromfile(f, dtype=a.dtype, count=(hi - lo) * row)
        return out.reshape((hi - lo,) + a.shape[1:])
    return np.asarray(a[lo:hi])


def doc_aligned_bounds(doc_id: np.ndarray | None, n_or_block: int, block_rows: int | None = None) -> np.ndarray:
    """Row bounds [0, b1, ..., N] of ~block_rows rows, each cut at the first doc start >= the target.

    doc_aligned_bounds(doc_id, B) (doc_id contiguous per doc); doc_aligned_bounds(None, N, B) = plain blocks."""
    if doc_id is None:
        n, b = n_or_block, block_rows
        return np.r_[np.arange(0, n, b), n].astype(np.int64)
    n, b = len(doc_id), n_or_block
    starts = np.flatnonzero(np.r_[True, doc_id[1:] != doc_id[:-1]])
    out = [0]
    while out[-1] < n:
        target = out[-1] + b
        j = np.searchsorted(starts, target, side='left')
        out.append(n if j >= len(starts) else int(starts[j]))
    return np.asarray(out, dtype=np.int64)


def _block_topk(S: np.ndarray, k: int):
    """Top-k per row of S [q, B] by (score desc, column desc) -> local cols [q, k'], scores [q, k']."""
    q, B = S.shape
    if B <= k:
        return np.broadcast_to(np.arange(B), (q, B)).copy(), S.copy()
    P = np.argpartition(S, B - k, axis=1)[:, B - k:]
    V = np.take_along_axis(S, P, 1)
    t = V.min(1)
    n_ge = (S >= t[:, None]).sum(1)
    for r in np.flatnonzero((n_ge > k) & np.isfinite(t)):   # ties straddle the k-th place: keep the highest ids
        gt = np.flatnonzero(S[r] > t[r])
        eq = np.flatnonzero(S[r] == t[r])
        P[r] = np.r_[gt, eq[len(eq) - (k - len(gt)):]]
        V[r] = S[r, P[r]]
    return P, V


class _RunningTopK:
    def __init__(self, nq: int, k: int):
        self.k = k
        self.v = np.full((nq, k), -np.inf, np.float32)
        self.i = np.full((nq, k), -1, np.int64)

    def push(self, S: np.ndarray, offset: int):
        """S [nq, B] scores of rows offset..offset+B-1."""
        for a in range(0, S.shape[0], SEL_ROWS):
            P, V = _block_topk(S[a:a + SEL_ROWS], self.k)
            cv = np.concatenate([self.v[a:a + SEL_ROWS], V], 1)
            ci = np.concatenate([self.i[a:a + SEL_ROWS], P.astype(np.int64) + offset], 1)
            o = np.lexsort((-ci, -cv), axis=1)[:, :self.k]
            self.v[a:a + SEL_ROWS] = np.take_along_axis(cv, o, 1)
            self.i[a:a + SEL_ROWS] = np.take_along_axis(ci, o, 1)

    def result(self):
        """-> per query: row ids, scores (finite entries only, ordered)."""
        keep = np.isfinite(self.v)
        return [self.i[j][keep[j]] for j in range(len(self.v))], [self.v[j][keep[j]] for j in range(len(self.v))]


def dense_topk(dense, qd: np.ndarray, k: int, block_rows: int = 100_000, doc_id=None, device: str = 'cpu'):
    """Exact inner-product top-k of every query over dense [N, d] fp16 (mmap). cpu: float32 math (as IndexFlatIP);
    cuda: fp16 matmul, scores cast to float32 (candidate set only, the re-score is always float32)."""
    n = len(dense)
    bounds = doc_aligned_bounds(doc_id, block_rows) if doc_id is not None else doc_aligned_bounds(None, n, block_rows)
    run = _RunningTopK(len(qd), k)
    if device == 'cpu':
        q = np.ascontiguousarray(qd, dtype=np.float32)
    else:
        import torch
        q = torch.from_numpy(np.ascontiguousarray(qd, dtype=np.float16)).to(device)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        blk = _read_rows(dense, int(lo), int(hi))
        if device == 'cpu':
            S = q @ blk.astype(np.float32).T
        else:
            import torch
            S = (q @ torch.from_numpy(np.ascontiguousarray(blk, dtype=np.float16)).to(device).T).float().cpu().numpy()
        del blk
        run.push(S, int(lo))
        del S
    return run.result()


class QuerySparse:
    """Union vocabulary V_q of the query sparse vectors and Q_s [|V_q|, n_query] float32."""

    def __init__(self, qsp):
        toks = [np.asarray(t, dtype=np.int64) for t, _ in qsp]
        self.vq = np.unique(np.concatenate(toks)) if any(len(t) for t in toks) else np.zeros(0, np.int64)
        self.remap = np.full(int(self.vq.max()) + 1 if len(self.vq) else 1, -1, np.int64)
        self.remap[self.vq] = np.arange(len(self.vq))
        self.Q = np.zeros((len(self.vq), len(qsp)), np.float32)
        self.has = np.array([len(t) > 0 for t in toks])
        for j, (t, w) in enumerate(qsp):
            self.Q[self.remap[np.asarray(t, dtype=np.int64)], j] = np.asarray(w).astype(np.float32)

    def cols(self, idx: np.ndarray) -> np.ndarray:
        """Token ids -> columns in V_q (-1 if not a query token)."""
        idx = np.asarray(idx, dtype=np.int64)
        m = len(self.remap)
        return np.where(idx < m, self.remap[np.minimum(idx, m - 1)], -1)


def _rows_csr(data, indices, indptr, lo: int, hi: int, vq: QuerySparse):
    """CSR [hi-lo, |V_q|] of rows lo..hi-1 restricted to query tokens (column order kept)."""
    from scipy.sparse import csr_matrix
    ip = _read_rows(indptr, lo, hi + 1).astype(np.int64)
    s, e = int(ip[0]), int(ip[-1])
    col = vq.cols(_read_rows(indices, s, e))
    m = col >= 0
    lens = np.diff(ip)
    rows = np.repeat(np.arange(hi - lo), lens)[m]
    ip = np.zeros(hi - lo + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=hi - lo), out=ip[1:])
    return csr_matrix((_read_rows(data, s, e).astype(np.float32, copy=False)[m], col[m].astype(np.int32), ip),
                      shape=(hi - lo, len(vq.vq)))


def sparse_topk(data, indices, indptr, vq: QuerySparse, k: int, block_rows: int = 100_000, doc_id=None):
    """Exact lexical top-k (positive scores only) of every query, scanning the CSR by row blocks."""
    n = len(indptr) - 1
    bounds = doc_aligned_bounds(doc_id, block_rows) if doc_id is not None else doc_aligned_bounds(None, n, block_rows)
    run = _RunningTopK(vq.Q.shape[1], k)
    if len(vq.vq):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            S = np.ascontiguousarray((_rows_csr(data, indices, indptr, int(lo), int(hi), vq) @ vq.Q).T)
            S[S <= 0] = -np.inf
            run.push(S, int(lo))
            del S
    ids, sc = run.result()
    ids = [i if vq.has[j] else i[:0] for j, i in enumerate(ids)]
    sc = [s if vq.has[j] else s[:0] for j, s in enumerate(sc)]
    return ids, sc


def sparse_scores(data, indices, indptr, vq: QuerySparse, j: int, cand: np.ndarray) -> np.ndarray:
    """Lexical score of query j for each candidate row (same float32 summation order as the CSC path)."""
    from scipy.sparse import csr_matrix
    cand = np.asarray(cand, dtype=np.int64)
    if not vq.has[j] or not len(cand):
        return np.zeros(len(cand), np.float32)
    starts = np.asarray(indptr[cand], dtype=np.int64)
    lens = np.asarray(indptr[cand + 1], dtype=np.int64) - starts
    off = np.r_[0, np.cumsum(lens)]
    pos = np.arange(off[-1]) + np.repeat(starts - off[:-1], lens)
    col = vq.cols(np.asarray(indices[pos]))
    m = col >= 0
    rows = np.repeat(np.arange(len(cand)), lens)[m]
    ip = np.zeros(len(cand) + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=len(cand)), out=ip[1:])
    blk = csr_matrix((np.asarray(data[pos], dtype=np.float32)[m], col[m].astype(np.int32), ip), shape=(len(cand), len(vq.vq)))
    return np.asarray(blk @ np.ascontiguousarray(vq.Q[:, j])).ravel().astype(np.float32, copy=False)


def exact_candidates(dense, sparse, doc_id, qd: np.ndarray, qsp, k: int = 200, block_rows: int = 100_000,
                     device: str = 'cpu'):
    """-> [(cand, dense scores, sparse scores)] per query, as Index.candidates (no field filter), and stats."""
    data, indices, indptr = sparse
    t0 = time.perf_counter()
    di, _ = dense_topk(dense, qd, k, block_rows, doc_id, device)
    t_dense = time.perf_counter() - t0
    t0 = time.perf_counter()
    vq = QuerySparse(qsp)
    si, _ = sparse_topk(data, indices, indptr, vq, k, block_rows, doc_id)
    t_sparse = time.perf_counter() - t0
    t0 = time.perf_counter()
    out, n_cand, n_docs = [], [], []
    for j in range(len(qd)):
        cand = np.union1d(di[j], si[j])
        dsc = np.asarray(dense[cand], dtype=np.float32) @ qd[j].astype(np.float32)      # cand is sorted
        ssc = sparse_scores(data, indices, indptr, vq, j, cand)
        out.append((cand, dsc, ssc))
        n_cand.append(len(cand))
        n_docs.append(len(np.unique(np.asarray(doc_id)[cand])))
    stats = {'dense_seconds': t_dense, 'sparse_seconds': t_sparse, 'union_rescore_seconds': time.perf_counter() - t0,
             'n_vq': int(len(vq.vq)), 'n_candidates': np.asarray(n_cand), 'n_docs': np.asarray(n_docs)}
    return out, stats
