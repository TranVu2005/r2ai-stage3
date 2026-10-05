import pytest
import numpy as np
from scipy.sparse import csr_matrix, load_npz, random as sprandom, save_npz, vstack

from r2ai.index.build import assemble_sparse, build_faiss


def test_assemble_sparse_equals_vstack(tmp_path):
    sh = tmp_path / 'shards'
    sh.mkdir()
    mats = [sprandom(r, 50, density=0.2, format='csr', dtype=np.float32, random_state=i) for i, r in enumerate((7, 7, 3))]
    for i, m in enumerate(mats):
        save_npz(sh / f'{i:05d}.sparse.npz', m)
    dest = tmp_path / 'out.npz'
    nnz = assemble_sparse(sh, 3, 17, dest)
    ref = vstack(mats, format='csr')
    got = load_npz(dest)
    assert nnz == ref.nnz and got.shape == ref.shape
    assert (got != ref).nnz == 0


def test_build_faiss_ivf_sq8_finds_self(tmp_path):
    rng = np.random.RandomState(0)
    x = rng.randn(3000, 1024).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    index, kind = build_faiss(x.astype(np.float16), flat_max_gb=0.0, nprobe=32)
    assert 'IVFScalarQuantizer' in kind and index.ntotal == 3000
    _, ids = index.search(x[:50], 1)
    assert (ids[:, 0] == np.arange(50)).mean() > 0.9
    flat, kind = build_faiss(x.astype(np.float16), flat_max_gb=4.0)
    assert kind == 'IndexFlatIP'


def _fake_shards(sh, sizes, vocab=50, seed=0):
    rng = np.random.RandomState(seed)
    dense, mats = [], []
    for i, r in enumerate(sizes):
        d = rng.randn(r, 1024).astype(np.float32)
        d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float16)
        m = sprandom(r, vocab, density=0.2, format='csr', dtype=np.float32, random_state=seed + i)
        np.save(sh / f'{i:05d}.dense.npy', d)
        save_npz(sh / f'{i:05d}.sparse.npz', m)
        dense.append(d)
        mats.append(m)
    return np.concatenate(dense), vstack(mats, format='csr')


def test_assemble_index_no_ann_writes_meta_without_faiss(tmp_path):
    from r2ai.index.build import assemble_index
    sh = tmp_path / 'shards'
    sh.mkdir()
    dense, csr = _fake_shards(sh, (4, 4, 2))
    info = assemble_index(tmp_path, n=10, shard_size=4, flat_max_gb=4.0, ann=False)
    assert not (tmp_path / 'faiss.index').exists()
    assert info['faiss'] is None and info['sparse_nnz'] == csr.nnz and info['faiss_float32_gb'] == 0.0
    assert np.array_equal(np.load(tmp_path / 'dense.npy'), dense)
    assert (load_npz(tmp_path / 'sparse.npz') != csr).nnz == 0


def test_assemble_index_ann_writes_faiss(tmp_path):
    import faiss
    from r2ai.index.build import assemble_index
    sh = tmp_path / 'shards'
    sh.mkdir()
    _fake_shards(sh, (4, 3))
    info = assemble_index(tmp_path, n=7, shard_size=4, flat_max_gb=4.0, ann=True)
    assert info['faiss'] == 'IndexFlatIP'
    assert faiss.read_index(str(tmp_path / 'faiss.index')).ntotal == 7


def test_assemble_cli_writes_meta_from_shards(tmp_path, monkeypatch):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.index import build
    cf = tmp_path / 'chunks_t256.parquet'
    pq.write_table(pa.table({'chunk_id': np.arange(6), 'text': ['x'] * 6}), cf)
    out = tmp_path / 'idx'
    (out / 'shards').mkdir(parents=True)
    _fake_shards(out / 'shards', (4, 2))
    monkeypatch.setattr(build, 'chunks_file', lambda t: cf)
    monkeypatch.setattr(build, 'index_dir', lambda t: out)
    monkeypatch.setattr(build, 'INDEX_DIR', tmp_path)
    monkeypatch.setattr(build, 'assert_writable', lambda p: p)
    assert build.main(['assemble', '--shard-size', '4', '--no-ann']) == 0
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['n_chunks'] == 6 and meta['faiss'] is None and not (out / 'faiss.index').exists()


@pytest.mark.parametrize('frozen_clock', [False, True])
def test_build_end_to_end_with_fake_encoder_no_gpu(tmp_path, monkeypatch, frozen_clock):
    """cmd_build shard loop + assemble on CPU: fake encoder, torch.cuda calls stubbed (no GPU touched).
    frozen_clock: a shard finishing within one timer tick (Windows ~15.6 ms) must not break the speed log line."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from r2ai.index import bge_m3, build

    class FakeEnc:
        max_len, vocab = 512, 40

        def __init__(self, **kw):
            self.model = type('M', (), {'config': type('C', (), {'hidden_size': 1024})()})()

        def tok(self, texts, **kw):
            return {'input_ids': [[0] * (1 + len(t)) for t in texts]}

        def encode_batch(self, texts):
            d = np.stack([np.full(1024, len(t), np.float32) for t in texts])
            d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float16)
            return d, [(np.array([len(t) % 40], np.int32), np.array([1.0], np.float16)) for t in texts]

    for f in ('reset_peak_memory_stats', 'synchronize', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, f, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a, **k: 0)
    monkeypatch.setattr(bge_m3, 'M3Encoder', FakeEnc)
    cf = tmp_path / 'chunks_t256.parquet'
    pq.write_table(pa.table({'chunk_id': np.arange(5), 'text': ['a', 'bb', 'ccc', 'dddd', 'eeeee']}), cf)
    out = tmp_path / 'idx'
    monkeypatch.setattr(build, 'chunks_file', lambda t: cf)
    monkeypatch.setattr(build, 'index_dir', lambda t: out)
    monkeypatch.setattr(build, 'INDEX_DIR', tmp_path)
    monkeypatch.setattr(build, 'assert_writable', lambda p: p)
    if frozen_clock:
        monkeypatch.setattr(build.time, 'time', lambda: 1000.0)
    assert build.main(['build', '--shard-size', '2', '--batch-size', '2', '--no-ann']) == 0
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['n_chunks'] == 5 and meta['faiss'] is None and meta['sparse_nnz'] == 5
    sp = load_npz(out / 'sparse.npz').tocsr()
    assert sp.indices.tolist() == [1, 2, 3, 4, 5] and np.load(out / 'dense.npy').shape == (5, 1024)
