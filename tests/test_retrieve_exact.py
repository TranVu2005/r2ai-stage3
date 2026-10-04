"""Exact block-scan candidates (r2ai.retrieve.exact) vs full-matrix numpy and the legacy CSC path."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz

from r2ai.retrieve import exact

N, VOCAB, NQ, K = 5000, 300, 20, 50


@pytest.fixture(scope='module')
def fx():
    rng = np.random.RandomState(42)
    doc_id = np.repeat(np.arange(N), rng.randint(1, 11, N))[:N]          # contiguous, 1..10 chunks per doc
    d = rng.randn(N, 1024).astype(np.float32)
    d[100:110] = d[99]                                                  # duplicated chunks -> exact ties
    dense = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float16)
    rows = []
    for i in range(N):
        t = np.sort(rng.choice(VOCAB, rng.randint(0, 40), replace=False)).astype(np.int32)
        rows.append((t, rng.rand(len(t)).astype(np.float16)))
    for i in range(200, 215):
        rows[i] = rows[199]
    indptr = np.r_[0, np.cumsum([len(t) for t, _ in rows])]
    csr = csr_matrix((np.concatenate([v for _, v in rows]).astype(np.float32),
                      np.concatenate([t for t, _ in rows]), indptr), shape=(N, VOCAB))
    qd = dense[rng.choice(N, NQ)] + rng.randn(NQ, 1024).astype(np.float16) * np.float16(0.05)
    qd = (qd / np.linalg.norm(qd.astype(np.float32), axis=1, keepdims=True)).astype(np.float16)
    qd[0] = dense[99]                                                   # query hitting the tied block
    qsp = []
    for j in range(NQ):
        t = np.sort(rng.choice(VOCAB, rng.randint(1, 12), replace=False)).astype(np.int32)
        qsp.append((t, rng.rand(len(t)).astype(np.float16)))
    qsp[0] = rows[199]
    qsp.append((np.zeros(0, np.int32), np.zeros(0, np.float16)))      # query without sparse tokens
    qd = np.concatenate([qd, qd[:1]])
    return {'doc_id': doc_id, 'dense': dense, 'csr': csr, 'qd': qd, 'qsp': qsp}


def ref_topk(scores: np.ndarray, k: int, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k of one score vector: score desc, then row id desc (IndexFlatIP keeps the higher id)."""
    order = np.lexsort((-np.arange(len(scores)), -scores))[:k]
    if positive:
        order = order[scores[order] > 0]
    return order, scores[order]


def legacy_sparse(csr, toks, w, k):
    """Exactly the scoring of retrieve.run.Index.candidates (CSC column slice @ weights)."""
    csc = csr.tocsc()
    return np.asarray(csc[:, toks.astype(np.int64)] @ w.astype(np.float32)).ravel()


@pytest.mark.parametrize('block', [1, 7, 1000, 10**6])
def test_exact_dense_matches_full_matrix(fx, block):
    ids, sc = exact.dense_topk(fx['dense'], fx['qd'], K, block_rows=block, doc_id=fx['doc_id'])
    full = fx['qd'].astype(np.float32) @ fx['dense'].astype(np.float32).T
    for j in range(len(fx['qd'])):
        ri, rs = ref_topk(full[j], K)
        assert np.array_equal(ids[j], ri), j
        np.testing.assert_allclose(sc[j], rs, atol=1e-5)
    assert set(ids[0][:11]) == set(range(99, 110))                     # all ties kept


def test_ties_at_the_boundary_keep_the_highest_ids():
    dense = np.ones((10, 1024), np.float16) / np.float16(32)
    ids, _ = exact.dense_topk(dense, dense[:1], 3, block_rows=4)
    assert ids[0].tolist() == [9, 8, 7]
    m = csr_matrix(np.ones((10, 3), np.float32))
    ids, _ = exact.sparse_topk(m.data, m.indices, m.indptr, exact.QuerySparse([(np.array([1], np.int32), np.array([1.0], np.float16))]), 3, block_rows=4)
    assert ids[0].tolist() == [9, 8, 7]


@pytest.mark.parametrize('block', [1, 7, 1000, 10**6])
def test_exact_sparse_matches_legacy_csc(fx, block):
    vq = exact.QuerySparse(fx['qsp'])
    ids, sc = exact.sparse_topk(fx['csr'].data, fx['csr'].indices, fx['csr'].indptr, vq, K,
                                block_rows=block, doc_id=fx['doc_id'])
    for j, (t, w) in enumerate(fx['qsp']):
        if not len(t):
            assert len(ids[j]) == 0
            continue
        ri, rs = ref_topk(legacy_sparse(fx['csr'], t, w, K), K, positive=True)
        assert np.array_equal(ids[j], ri), j
        assert np.array_equal(sc[j], rs), j                            # same summation order -> bit-identical
    assert set(ids[0][:16]) == set(range(199, 215))


def test_sparse_scores_of_candidates_bit_identical(fx):
    vq = exact.QuerySparse(fx['qsp'])
    cand = np.array([0, 5, 99, 199, 200, 4999])
    for j, (t, w) in enumerate(fx['qsp']):
        got = exact.sparse_scores(fx['csr'].data, fx['csr'].indices, fx['csr'].indptr, vq, j, cand)
        want = legacy_sparse(fx['csr'], t, w, K)[cand] if len(t) else np.zeros(len(cand), np.float32)
        assert got.dtype == np.float32 and np.array_equal(got, want), j


def test_doc_aligned_bounds_never_split_a_doc(fx):
    for block in (1, 7, 1000, 10**6):
        b = exact.doc_aligned_bounds(fx['doc_id'], block)
        assert b[0] == 0 and b[-1] == N and np.all(np.diff(b) > 0)
        starts = b[1:-1]
        assert np.all(fx['doc_id'][starts] != fx['doc_id'][starts - 1])
        if block >= 10:
            assert np.diff(b).max() <= block + 10                       # a block exceeds B by < one doc


def test_split_sparse_npz_roundtrip_and_source_untouched(fx, tmp_path):
    src = tmp_path / 'sparse.npz'
    save_npz(src, fx['csr'])
    before = src.read_bytes()
    cache = tmp_path / 'sparse_mmap'
    info = exact.split_sparse_npz(src, cache)
    assert src.read_bytes() == before and info['nnz'] == fx['csr'].nnz
    data, indices, indptr, shape = exact.load_sparse_mmap(cache)
    assert isinstance(data, np.memmap) and tuple(shape) == fx['csr'].shape
    assert np.array_equal(data, fx['csr'].data) and np.array_equal(indices, fx['csr'].indices)
    assert np.array_equal(indptr, fx['csr'].indptr)
    assert exact.split_sparse_npz(src, cache)['reused'] is True        # second call keeps the cache


def test_exact_candidates_union_and_scores_match_legacy(fx):
    """Per query: union of dense/sparse top-k, dense rescored from fp16 rows, sparse = legacy sp_all[cand]."""
    csr = fx['csr']
    res, stats = exact.exact_candidates(fx['dense'], (csr.data, csr.indices, csr.indptr), fx['doc_id'],
                                        fx['qd'], fx['qsp'], k=K, block_rows=333)
    full = fx['qd'].astype(np.float32) @ fx['dense'].astype(np.float32).T
    for j, (cand, dsc, ssc) in enumerate(res):
        t, w = fx['qsp'][j]
        di, _ = ref_topk(full[j], K)
        sp_all = legacy_sparse(csr, t, w, K) if len(t) else None
        si = ref_topk(sp_all, K, positive=True)[0] if len(t) else np.zeros(0, np.int64)
        assert np.array_equal(cand, np.union1d(di, si)), j
        assert np.array_equal(dsc, np.asarray(fx['dense'][cand], dtype=np.float32) @ fx['qd'][j].astype(np.float32))
        assert np.array_equal(ssc, sp_all[cand] if sp_all is not None else np.zeros(len(cand), np.float32))
    assert stats['n_candidates'][0] == len(res[0][0])
    assert stats['n_docs'][0] == len(set(fx['doc_id'][res[0][0]]))


def test_read_rows_from_memmap_is_a_private_copy(fx, tmp_path):
    np.save(tmp_path / 'dense.npy', fx['dense'])
    mm = np.load(tmp_path / 'dense.npy', mmap_mode='r')
    got = exact._read_rows(mm, 17, 1234)
    assert type(got) is np.ndarray and np.array_equal(got, fx['dense'][17:1234])
    np.save(tmp_path / 'ip.npy', fx['csr'].indptr)
    ip = np.load(tmp_path / 'ip.npy', mmap_mode='r')
    assert type(exact._read_rows(ip, 3, 9)) is np.ndarray and np.array_equal(exact._read_rows(ip, 3, 9), fx['csr'].indptr[3:9])
    assert np.array_equal(exact._read_rows(fx['dense'], 0, 5), fx['dense'][:5])   # plain arrays: slice
    assert np.array_equal(exact._read_rows(mm[10:], 0, 5), fx['dense'][10:15])     # memmap view (offset not shifted)
    ids_mm, _ = exact.dense_topk(mm, fx['qd'], K, block_rows=777, doc_id=fx['doc_id'])
    ids, _ = exact.dense_topk(fx['dense'], fx['qd'], K, block_rows=777, doc_id=fx['doc_id'])
    assert all(np.array_equal(a, b) for a, b in zip(ids_mm, ids))


def _chunked_texts(tmp_path, n=1000, rg=97):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.RandomState(42)
    texts = [('xin chào ' * rng.randint(0, 5)) + str(i) for i in range(n)]
    src = tmp_path / 'chunks.parquet'
    pq.write_table(pa.table({'chunk_id': np.arange(n), 'text': texts}), src, row_group_size=rg)
    return src, texts


def test_take_texts_across_chunks_keeps_order_and_duplicates(tmp_path):
    import pyarrow.parquet as pq
    src, texts = _chunked_texts(tmp_path)
    col = pq.read_table(src, columns=['text'])['text']
    assert col.num_chunks > 1
    ids = np.array([999, 0, 96, 97, 500, 0, 998, 97])
    assert exact.take_texts(col, ids) == [texts[i] for i in ids]
    assert exact.take_texts(col, np.zeros(0, np.int64)) == []


def test_text_mmap_cache_equals_column_and_is_reused(tmp_path):
    import os
    import pyarrow.parquet as pq
    src, texts = _chunked_texts(tmp_path)
    cache = tmp_path / 'idx' / 'text.arrow'
    col, info = exact.text_mmap(src, cache)
    assert info['reused'] is False and col.to_pylist() == texts
    assert exact.take_texts(col, np.array([5, 900])) == [texts[5], texts[900]]
    del col
    col, info = exact.text_mmap(src, cache)
    assert info['reused'] is True and col.to_pylist() == texts
    del col
    st = src.stat()
    os.utime(src, (st.st_atime, st.st_mtime + 10))                     # source changed -> rebuilt
    col, info = exact.text_mmap(src, cache)
    assert info['reused'] is False and col.num_chunks >= 1
