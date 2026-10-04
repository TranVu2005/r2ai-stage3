"""run_retrieval_k100 --candidates exact gives the same K100 outputs as the legacy FAISS-flat + CSC path (CPU only)."""
import json
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_matrix, save_npz

N_DOCS, VOCAB = 400, 120


def _vec(text: str) -> np.ndarray:
    v = np.random.RandomState(zlib.crc32(text.encode()) % 2**31).randn(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _toks(text: str):
    t = np.unique(np.array([zlib.crc32(w.encode()) % VOCAB for w in text.split()], np.int32))
    w = np.array([(zlib.crc32(f'{x}|{text}'.encode()) % 997 + 1) / 997 for x in t], np.float16)
    return t, w


class FakeEnc:
    def __init__(self, **kw):
        pass

    def encode_batch(self, texts):
        return np.stack([_vec(t) for t in texts]).astype(np.float16), [_toks(t) for t in texts]


class FakeRR:
    def __init__(self, **kw):
        pass

    def score(self, q, passages, batch_size=32):
        return np.array([zlib.crc32(f'{q}|{p}'.encode()) % 10007 / 10007 for p in passages], np.float32)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import faiss
    import torch
    from r2ai.index import bge_m3
    from r2ai.retrieve import run, run_retrieval_k100 as k100
    rng = np.random.RandomState(42)
    words = [f'w{i}' for i in range(300)]
    doc_id, field, text = [], [], []
    for d in range(N_DOCS):
        for c in range(rng.randint(1, 6)):
            doc_id.append(d)
            field.append('title' if c == 0 else 'body')
            text.append(' '.join(rng.choice(words, rng.randint(3, 15))))
    n = len(text)
    cf = tmp_path / 'chunks_t256.parquet'
    pq.write_table(pa.table({'chunk_id': np.arange(n), 'doc_id': np.array(doc_id, np.int64), 'field': field, 'text': text}), cf)
    idx = tmp_path / 'index' / 't256'
    idx.mkdir(parents=True)
    dense = np.stack([_vec(t) for t in text]).astype(np.float16)
    np.save(idx / 'dense.npy', dense)
    sp = [_toks(t) for t in text]
    ip = np.r_[0, np.cumsum([len(t) for t, _ in sp])]
    save_npz(idx / 'sparse.npz', csr_matrix((np.concatenate([w for _, w in sp]).astype(np.float32),
                                             np.concatenate([t for t, _ in sp]), ip), shape=(n, 250002)))
    fi = faiss.IndexFlatIP(1024)
    fi.add(dense.astype(np.float32))
    faiss.write_index(fi, str(idx / 'faiss.index'))
    (idx / 'meta.json').write_text(json.dumps({'n_chunks': n}), encoding='utf-8')
    qf = tmp_path / 'query.parquet'
    pq.write_table(pa.table({'id': np.arange(30), 'query': [' '.join(rng.choice(words, 6)) for _ in range(30)]}), qf)
    for m in (run, k100):
        monkeypatch.setattr(m, 'chunks_file', lambda t: cf)
        monkeypatch.setattr(m, 'index_dir', lambda t: idx)
        monkeypatch.setattr(m, 'assert_writable', lambda p: p)
    monkeypatch.setattr(bge_m3, 'M3Encoder', FakeEnc)
    monkeypatch.setattr(bge_m3, 'Reranker', FakeRR)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    return {'k100': k100, 'q': qf, 'tmp': tmp_path, 'idx': idx}


def test_k100_exact_equals_faiss(env):
    k100 = env['k100']
    base = ['--queries', str(env['q'])]
    assert k100.main(base + ['--out-dir', str(env['tmp'] / 'old')]) == 0
    assert k100.main(base + ['--out-dir', str(env['tmp'] / 'new'), '--candidates', 'exact', '--block-rows', '97']) == 0
    for f in ('vi_k100.parquet', 'vi_k100_chunk_scores.parquet'):
        assert pq.read_table(env['tmp'] / 'old' / f).equals(pq.read_table(env['tmp'] / 'new' / f)), f
    meta = json.loads((env['tmp'] / 'new' / 'vi_k100.meta.json').read_text(encoding='utf-8'))
    assert meta['candidates'] == 'exact' and meta['distinct_docs_before_rerank']['min'] >= 1
    assert meta['peak_rss_gib'] > 0 and 'dense_seconds' in meta['exact']
    cs = pq.read_table(env['tmp'] / 'new' / 'vi_k100.candidates.parquet')
    assert cs.num_rows == 30 and cs.column_names == ['query_id', 'n_candidates', 'n_docs']
    assert (env['idx'] / 'sparse_mmap' / 'shape.json').exists()


def test_k100_exact_does_not_need_faiss(env):
    (env['idx'] / 'faiss.index').unlink()
    assert env['k100'].main(['--queries', str(env['q']), '--out-dir', str(env['tmp'] / 'x'), '--candidates', 'exact']) == 0


def test_exact_gate_stages_agree_on_fixture(env):
    import pickle
    from r2ai.retrieve import exact_gate
    out = env['tmp'] / 'gate'
    out.mkdir()
    qs = pq.read_table(env['q']).to_pylist()
    d, s = FakeEnc().encode_batch([r['query'] for r in qs])
    pickle.dump({'ids': [r['id'] for r in qs], 'qd': d, 'qsp': s}, open(out / 'queries.pkl', 'wb'))
    common = ['--out-dir', str(out), '--chunks', str(env['tmp'] / 'chunks_t256.parquet')]
    assert exact_gate.main(['legacy', '--index-dir', str(env['idx'])] + common) == 0
    assert exact_gate.main(['exact', '--index-dir', str(env['idx']), '--block-rows', '50'] + common) == 0
    assert exact_gate.main(['compare'] + common) == 0
    res = json.loads((out / 'compare.json').read_text(encoding='utf-8'))
    assert res['k100_prerank_same_set'] == 30 and res['candidate_set_identical'] == 30 and res['diffs'] == []
    assert json.loads((out / 'legacy.json').read_text(encoding='utf-8'))['index_files_unchanged'] is True
