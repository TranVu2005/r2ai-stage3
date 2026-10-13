import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_matrix

import r2ai.index.segments as segments
from r2ai.index.segments import load_segment, write_segment
from r2ai.zh_v2 import index as zi

GS = (1 << 48) + (7 << 32)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    src_idx, v2_idx, chunks_src, chunks_v2 = (tmp_path / n for n in ('src_idx', 'v2_idx', 'chunks_src', 'chunks_v2'))
    for p in (src_idx, v2_idx):
        p.mkdir()
    monkeypatch.setattr(segments, 'ZH_INDEX_ROOT', src_idx)
    monkeypatch.setattr(segments, 'ZH_V2_INDEX_ROOT', v2_idx)
    monkeypatch.setattr(zi, 'ZH_INDEX_ROOT', src_idx)
    monkeypatch.setattr(zi, 'ZH_V2_INDEX_ROOT', v2_idx)
    monkeypatch.setattr(zi, 'CHUNKS_V2', chunks_v2)
    monkeypatch.setattr(zi, 'CHUNKS_SRC', chunks_src)
    monkeypatch.setattr(zi, 'RUN', tmp_path / 'run')
    return src_idx, v2_idx, chunks_src, chunks_v2


def make_source(roots, name='seg00007-b001-x.com'):
    src_idx, _, chunks_src, _ = roots
    rng = np.random.default_rng(1)
    doc = np.array([10, 10, 10, 20, 30, 30], np.int64)          # three docs: 3 + 1 + 2 chunks
    n, vocab = len(doc), 50
    dense = rng.standard_normal((n, 8)).astype(np.float16)
    lens = np.array([3, 1, 2, 4, 2, 5])
    indptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
    cols = np.concatenate([np.sort(rng.choice(vocab, k, replace=False)) for k in lens]).astype(np.int32)
    sp = csr_matrix((rng.random(int(lens.sum())).astype(np.float32), cols, indptr), shape=(n, vocab))
    texts = [f'chunk-{i}' for i in range(n)]
    write_segment(src_idx, name, dense, sp, doc, texts, global_start=GS)
    cdir = chunks_src / name
    cdir.mkdir(parents=True)
    pq.write_table(pa.table({'chunk_id': np.arange(n, dtype=np.int64), 'doc_id': doc,
                             'doc_ids_group': [[d] if d != 20 else [20, 99] for d in doc],
                             'field': ['title'] * n, 'char_start': np.zeros(n, np.int32), 'char_end': np.ones(n, np.int32),
                             'text': texts, 'n_tokens': np.ones(n, np.int32)}), cdir / 'chunks_t256.parquet')
    pq.write_table(pa.table({'doc_id': [10, 20, 30], 'doc_ids_group': [[10], [20, 99], [30]], 'domain': ['x.com'] * 3,
                             'url': ['u10', 'u20', 'u30'], 'title': ['a', 'b', 'c'], 'doc_text': ['', '', '']}),
                   cdir / 'docs.parquet')
    return {'name': name, 'segment_id': 7, 'batch': 'b001', 'domain': 'x.com', 'chunks_sha256': 'h'}


def test_keep_docs_uses_any_group_member(roots):
    src = make_source(roots)
    docs = roots[2] / src['name'] / 'docs.parquet'
    assert zi.keep_docs(docs, np.array([99, 30])).tolist() == [20, 30]       # doc 20 kept by its non-primary id 99
    assert zi.keep_docs(docs, np.array([5])).tolist() == []


def test_build_segment_copies_kept_rows_byte_for_byte_and_verifies(roots):
    src = make_source(roots)
    row = zi.build_segment(src, np.array([30, 99]))
    assert (row['rows_source'], row['rows_v2'], row['docs_v2'], row['global_start']) == (6, 3, 2, GS)
    new, old = load_segment(roots[1] / src['name']), load_segment(roots[0] / src['name'])
    assert new.global_start == old.global_start and new.doc_id.tolist() == [20, 30, 30]
    assert np.array_equal(np.asarray(new.dense), np.asarray(old.dense)[[3, 4, 5]])
    assert zi.verify_segment(src['name']) == {'name': src['name'], 'rows': 3, 'verified': True}
    chunks = pq.read_table(roots[3] / src['name'] / 'chunks_t256.parquet').to_pylist()
    assert [(c['chunk_id'], c['doc_id'], c['text']) for c in chunks] == [(0, 20, 'chunk-3'), (1, 30, 'chunk-4'), (2, 30, 'chunk-5')]
    docs = pq.read_table(roots[3] / src['name'] / 'docs.parquet').to_pylist()
    assert [(d['doc_id'], d['doc_ids_v2']) for d in docs] == [(20, [99]), (30, [30])]
    assert pq.read_table(roots[3] / src['name'] / 'row_map.parquet').to_pydict() == {'new_local': [0, 1, 2], 'old_local': [3, 4, 5]}


def test_verify_detects_a_changed_vector(roots):
    src = make_source(roots)
    zi.build_segment(src, np.array([30]))
    arr = np.load(roots[1] / src['name'] / 'dense.npy', mmap_mode='r+')
    arr[0, 0] = arr[0, 0] + np.float16(1)
    arr.flush()
    del arr
    with pytest.raises(ValueError, match='dense differs'):
        zi.verify_segment(src['name'])


def test_segment_without_v2_docs_writes_nothing_and_existing_output_is_refused(roots):
    src = make_source(roots)
    assert zi.build_segment(src, np.array([1234]))['rows_v2'] == 0
    assert not (roots[1] / src['name']).exists()
    zi.build_segment(src, np.array([10]))
    with pytest.raises(ValueError, match='exists'):
        zi.build_segment(src, np.array([10]))


def test_guard_blocks_everything_outside_the_zh_v2_roots(tmp_path):
    with pytest.raises(ValueError, match='Not a zh-v2 output'):
        zi.guard(tmp_path / 'x')
