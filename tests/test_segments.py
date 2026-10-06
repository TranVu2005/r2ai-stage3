"""CPU immutable segments: global ties, CSR offsets, text and append parity."""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from r2ai.paths import RUNS_DIR, WORK_DATA_DIR, assert_writable
from r2ai.retrieve import exact


def _segments():
    # A missing feature is an assertion failure during the initial RED run.
    assert importlib.util.find_spec('r2ai.index.segments') is not None, 'segments module missing'
    return importlib.import_module('r2ai.index.segments')


@pytest.fixture
def output():
    root = assert_writable(RUNS_DIR / 'zh-full' / 'segments' / 'tests')
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='case-', dir=root))


def test_merge_ties_use_global_id_and_ignore_nonfinite():
    seg = _segments()
    ids, scores = seg.merge_topk([
        ([np.array([3, 1, -1])], [np.array([1., 1., -np.inf], np.float32)]),
        ([np.array([10, 8])], [np.array([1., 1.], np.float32)])], 3)
    assert ids[0].tolist() == [10, 8, 3]
    assert scores[0].tolist() == [1., 1., 1.]


def test_virtual_dense_and_sparse_preserve_doc_blocks_and_ties():
    seg = _segments()
    docs = np.repeat(np.arange(12), 3)
    dense = np.ones((36, 16), np.float16)
    sparse = csr_matrix(np.ones((36, 3), np.float32))
    qd = np.ones((3, 16), np.float16)
    qsp = [(np.array([1]), np.array([1.], np.float32)),
           (np.array([2]), np.array([2.], np.float32)),
           (np.array([], np.int32), np.array([], np.float32))]
    whole = seg.virtual_segments(dense, sparse, docs, count=1, block_rows=4)
    parts = seg.virtual_segments(dense, sparse, docs, count=4, block_rows=4)
    assert len(parts) == 4
    assert all(docs[s.row_start] != docs[s.row_start-1] for s in parts[1:])
    one = seg.scan_topk(whole, qd, qsp, 5, block_rows=4, query_batch=2)
    four = seg.scan_topk(parts, qd, qsp, 5, block_rows=4, query_batch=2)
    for key in ('dense', 'sparse'):
        for a, b in zip(one[key][0], four[key][0]):
            assert np.array_equal(a, b)
        for a, b in zip(one[key][1], four[key][1]):
            assert np.array_equal(a, b)
    assert four['dense'][0][0].tolist() == [35, 34, 33, 32, 31]
    assert four['sparse'][0][2].tolist() == []


def test_ingest_append_keeps_existing_bytes_and_reads_text(output):
    seg = _segments()
    dense = np.array([[1, 0], [0, 1], [1, 1]], np.float16)
    sparse = csr_matrix(np.array([[0, 2], [1, 0], [2, 1]], np.float32))
    docs = np.array([2, 2, 3])
    first = seg.write_segment(output, 'a', dense, sparse, docs, ['a', 'b', 'c'], global_start=0)
    before = {p.name: p.read_bytes() for p in first.path.iterdir()}
    second = seg.write_segment(output, 'b', dense, sparse, docs + 10, ['d', 'e', 'f'], global_start=3)
    assert {p.name: p.read_bytes() for p in first.path.iterdir()} == before
    got = seg.scan_topk([first, second], np.array([[1, 1]], np.float16),
                        [(np.array([0, 1]), np.ones(2, np.float32))], 2, block_rows=1)
    assert got['dense'][0][0].tolist() == [5, 2]
    assert got['sparse'][0][0].tolist() == [5, 2]
    assert seg.take_segment_texts([first, second], [5, 0, 3, 5]) == ['f', 'a', 'd', 'f']
    with pytest.raises(FileExistsError):
        seg.write_segment(output, 'a', dense, sparse, docs, ['a', 'b', 'c'], global_start=0)


def test_invalid_rows_or_overlapping_global_ids_rejected(output):
    seg = _segments()
    dense = np.ones((2, 4), np.float16)
    sparse = csr_matrix(np.ones((2, 2), np.float32))
    with pytest.raises(ValueError):
        seg.write_segment(output, 'invalid', dense, sparse, np.array([0]), ['a', 'b'], global_start=0)
    assert not (output / 'invalid').exists()
    a = seg.write_segment(output, 'a', dense, sparse, np.array([0, 1]), ['a', 'b'], global_start=0)
    with pytest.raises(ValueError):
        seg.scan_topk([a, a], dense[:1], [(np.array([], np.int32), np.array([], np.float32))], 1)


def test_seeded_virtual_partition_matches_public_exact_scores():
    seg = _segments()
    rng = np.random.default_rng(42)
    dense = rng.normal(size=(300, 64)).astype(np.float16)
    docs = np.repeat(np.arange(60), 5)
    dense[45:60] = dense[40]
    sparse = csr_matrix(rng.integers(0, 3, size=(300, 9)).astype(np.float32))
    qd = dense[[40, 123, 299]]
    qsp = [(np.array([1, 3, 8]), np.array([.1, .7, .3], np.float32))] * 3
    got = seg.scan_topk(seg.virtual_segments(dense, sparse, docs, count=4, block_rows=31),
                        qd, qsp, 20, block_rows=31, query_batch=3)
    di, ds = exact.dense_topk(dense, qd, 20, 31, docs)
    si, ss = exact.sparse_topk(sparse.data, sparse.indices, sparse.indptr, exact.QuerySparse(qsp), 20, 31, docs)
    for key, ids, scores in [('dense', di, ds), ('sparse', si, ss)]:
        for j in range(3):
            assert np.array_equal(got[key][0][j], ids[j])
            np.testing.assert_allclose(got[key][1][j], scores[j], atol=1e-5, rtol=0)


def test_streaming_writer_rejects_incomplete_manifest_and_unsafe_names(output):
    seg = _segments()
    writer = seg.SegmentWriter(output, 'partial', rows=4, dimension=2, vocabulary=3, global_start=0)
    writer.append(np.ones((2, 2), np.float16), csr_matrix(np.ones((2, 3), np.float32)),
                  np.array([0, 0]), ['a', 'b'])
    with pytest.raises(ValueError, match='incomplete'):
        writer.close()
    assert not (output/'partial'/'segment.json').exists()
    with pytest.raises(ValueError):
        seg.SegmentWriter(output, '../escape', rows=1, dimension=2, vocabulary=3, global_start=0)


def test_streaming_batches_preserve_csr_offsets_empty_rows_and_text(output):
    seg = _segments()
    writer = seg.SegmentWriter(output, 'stream', rows=5, dimension=2, vocabulary=3, global_start=100)
    writer.append(np.ones((2, 2), np.float16), csr_matrix(np.array([[0, 2, 0], [0, 0, 0]], np.float32)),
                  np.array([0, 0]), ['a', 'b'])
    writer.append(np.ones((3, 2), np.float16), csr_matrix(np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]], np.float32)),
                  np.array([1, 1, 2]), ['c', 'd', 'e'])
    segment = writer.close()
    got = seg.scan_topk([segment], np.ones((1, 2), np.float16),
                        [(np.array([1]), np.array([1.], np.float32))], 5, block_rows=1)
    assert got['sparse'][0][0].tolist() == [103, 100]
    assert got['sparse'][1][0].tolist() == [4., 2.]
    assert seg.take_segment_texts([segment], [104, 101, 102]) == ['e', 'b', 'c']


def test_guard_refuses_output_escape_and_text_missing_id(output):
    seg = _segments()
    with pytest.raises(ValueError):
        seg.guarded_output(RUNS_DIR/'outside-segments')
    d = np.ones((1, 2), np.float16)
    s = seg.write_segment(output, 'one', d, csr_matrix([[1.]]), np.array([0]), ['x'], global_start=7)
    with pytest.raises(IndexError):
        seg.take_segment_texts([s], [6])


def test_real_synthetic_benchmark_writes_and_scans_all_rows(output):
    seg = _segments()
    assert hasattr(seg, 'synthetic_benchmark'), 'synthetic benchmark missing'
    report = seg.synthetic_benchmark(output/'synthetic', rows=48, dimension=16, queries=5,
                                      segment_rows=16, block_rows=8, writer_block=8,
                                      query_batch=3, seed=42, vocabulary=32, tokens_per_row=3)
    assert report['rows_scanned'] == 48 and report['queries'] == 5
    assert report['dense_payload_bytes'] == 48*16*2
    assert report['dense_disk_bytes'] > report['dense_payload_bytes']
    assert report['segments'] == 3 and report['all_dense_files_read']
    assert report['memory']['peak_rss_gib'] > 0
    assert report['text_roundtrip_verified']


def test_read_only_mmap_view_uses_correct_disk_offset(output):
    seg = _segments()
    dense = np.arange(60, dtype=np.float16).reshape(20, 3)
    np.save(output/'view.npy', dense)
    mm = np.load(output/'view.npy', mmap_mode='r')
    view = seg.ArrayRows(mm[5:15], start=2, stop=7)
    got = view[:]
    assert type(got) is np.ndarray
    np.testing.assert_array_equal(got, dense[7:12])


@pytest.mark.parametrize('readonly', [False, True])
def test_csr_int64_indptr_is_unchanged_across_repeated_scans(readonly):
    seg = _segments()
    dense = np.ones((6, 2), np.float16)
    data = np.arange(1, 7, dtype=np.float32)
    indices = np.zeros(6, np.int32)
    indptr = np.arange(7, dtype=np.int64)
    if readonly:
        indptr.flags.writeable = False
    before = indptr.tobytes()
    segments = seg.virtual_segments(dense, (data, indices, indptr),
                                    np.repeat(np.arange(3), 2), count=1, block_rows=2,
                                    global_start=100)
    qsp = [(np.array([0]), np.array([1.], np.float32))]
    for _ in range(2):
        result = seg.scan_topk(segments, np.ones((1, 2), np.float16), qsp, 3,
                               block_rows=2, query_batch=1)
        assert result['dense'][0][0].tolist() == [105, 104, 103]
        assert result['sparse'][0][0].tolist() == [105, 104, 103]
        assert result['sparse'][1][0].tolist() == [6., 5., 4.]
        assert indptr.tobytes() == before


def test_memory_guard_uses_bounded_buffers_instead_of_disk_payload(monkeypatch):
    seg = _segments()
    assert hasattr(seg, 'scan_memory_budget'), 'bounded memory budget missing'
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=int(2.25*2**30)))
    for rows in (16_384, 8_000_000):
        budget = seg.scan_memory_budget(dimension=1024, queries=1200, block_rows=16384,
            query_batch=128, max_block_nnz=16384*8, query_vocabulary=8192,
            metadata_rows=rows, writer_block=8192)
        checked = seg.require_scan_memory(budget)
        assert checked['minimum_available_gib'] == 2.0
        assert checked['estimated_peak_private_gib'] < 0.7
        assert checked['available_memory_gib'] > 2
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=int(1.99*2**30)))
    with pytest.raises(RuntimeError, match='available memory'):
        seg.require_scan_memory(budget)


def test_memory_guard_refuses_unbounded_blocks_even_with_free_ram(monkeypatch):
    seg = _segments()
    assert hasattr(seg, 'scan_memory_budget'), 'bounded memory budget missing'
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=8*2**30))
    budget = seg.scan_memory_budget(dimension=1024, queries=1200, block_rows=1_000_000,
                                    query_batch=128, max_block_nnz=8_000_000,
                                    query_vocabulary=8192, metadata_rows=8_000_000)
    with pytest.raises(RuntimeError, match='1.5 GiB'):
        seg.require_scan_memory(budget)


def test_segment_index_root_is_private_zh_and_never_vi_or_sample():
    seg = _segments()
    assert hasattr(seg, 'guarded_segment_output'), 'private ZH index guard missing'
    own = WORK_DATA_DIR/'index'/'zh_full_segments'/'s000'
    assert seg.guarded_segment_output(own) == own.resolve()
    for forbidden in (WORK_DATA_DIR/'index'/'t256'/'s000',
                      WORK_DATA_DIR/'index'/'zh_sample_t256'/'s000',
                      WORK_DATA_DIR/'index'/'zh_full_segments'/'..'/'t256'):
        with pytest.raises(ValueError):
            seg.guarded_segment_output(forbidden)
    with pytest.raises(ValueError):
        seg.guarded_output(own)  # runtime benchmark/gate cannot redirect into index
