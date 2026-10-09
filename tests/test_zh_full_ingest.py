import json
import os
import sqlite3
import time

import numpy as np
import pytest

ART = '中医认为这种病多由气血不足所致，治疗以补益为主。' * 12


def html_cnkang(text):
    return f'<html><head><title>t</title></head><body><h1>标题</h1><div class="detailc"><p>{text}</p></div></body></html>'


@pytest.fixture(autouse=True)
def segment_root(tmp_path, monkeypatch):
    # segments.SegmentWriter keeps its own guard; point it at this test's private index root.
    import r2ai.index.segments as segments
    monkeypatch.setattr(segments, 'ZH_INDEX_ROOT', tmp_path)


def make_roots(tmp_path):
    from r2ai.zh_full_ingest.common import Roots
    t = tmp_path
    inputs = dict(raw=t / 'raw_full', crawl_db=t / 'state/zh_full/crawl.db', sample_raw=t / 'raw_sample',
                  sample_db=t / 'state/crawl_zh_sample.db', sample_docs=t / 'docs_sample/bundle',
                  sample_chunks=t / 'chunks_sample', sample_index=t / 'index/zh_sample_t256')
    roots = Roots(**inputs, docs=t / 'docs_zh_full', chunks=t / 'chunks_zh_full', index=t / 'index/zh_full_segments',
                  run=t / 'run', state=t / 'state/zh_full_ingest',
                  forbidden=(t / 'raw_full', t / 'state/zh_full', t / 'raw_sample', t / 'docs_sample',
                             t / 'chunks_sample', t / 'index/zh_sample_t256', t / 'vi'))
    return roots


def write_shard(directory, name, records):
    import zstandard
    directory.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode()
    (directory / name).write_bytes(zstandard.ZstdCompressor().compress(payload))
    past = time.time() - 60  # a shard renamed before the snapshot
    os.utime(directory / name, (past, past))
    return directory / name


def rec(u, doc_ids, status='ok', text=ART, domain='cnkang.com', fetched_at=1.0):
    return {'url_norm': u, 'url': 'https://' + u, 'doc_ids': doc_ids, 'domain': domain, 'status': status,
            'html': html_cnkang(text), 'final_url': 'https://' + u, 'fetched_at': fetched_at, 'http_status': 200}


def make_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE urls(url_norm TEXT PRIMARY KEY, url TEXT, doc_ids TEXT, domain TEXT, rank INT, status TEXT, '
              'http_status INT, reason TEXT, attempts INT, fetched_at REAL, next_try_at REAL, shard TEXT, final_url TEXT)')
    c.executemany('INSERT INTO urls(url_norm,url,doc_ids,domain,rank,status,shard,fetched_at) VALUES (?,?,?,?,?,?,?,?)',
                  [(u, 'https://' + u, json.dumps(ids), d, i, s, sh, 1.0) for i, (u, ids, d, s, sh) in enumerate(rows)])
    c.commit()
    c.close()


def fake_tok(text):
    return len(text)


@pytest.fixture
def full_crawl(tmp_path):
    roots = make_roots(tmp_path)
    d = roots.raw / 'cnkang.com'
    write_shard(d, 's1.jsonl.zst', [rec('cnkang.com/a', [10]), rec('cnkang.com/b', [11], status='network_error')])
    write_shard(d, 's2.jsonl.zst', [rec('cnkang.com/b', [11], text=ART + '第二版'), rec('cnkang.com/c', [12], status='thin', text=ART[:120])])
    write_shard(d, 'orphan.jsonl.zst', [rec('cnkang.com/z', [99])])
    write_shard(d, 's3.jsonl.zst.tmp', [rec('cnkang.com/t', [98])])
    make_db(roots.crawl_db, [('cnkang.com/a', [10], 'cnkang.com', 'ok', 's1.jsonl.zst'),
                             ('cnkang.com/b', [11], 'cnkang.com', 'ok', 's2.jsonl.zst'),
                             ('cnkang.com/c', [12], 'cnkang.com', 'thin', 's2.jsonl.zst'),
                             ('cnkang.com/m', [13], 'cnkang.com', 'ok', 'missing.jsonl.zst'),
                             ('cnkang.com/p', [14], 'cnkang.com', 'pending', None),
                             ('cnkang.com/t', [98], 'cnkang.com', 'ok', 's3.jsonl.zst')])
    return roots


def test_global_ids_are_int64_and_disjoint_from_vi():
    from r2ai.zh_full_ingest.common import global_start, split_global_id, GLOBAL_ID_BASE
    assert global_start(1) == 2**48 + 2**32 and global_start(2) - global_start(1) == 2**32
    assert GLOBAL_ID_BASE > 3_909_585 and global_start(32767) + 2**32 < 2**53 < 2**63
    assert split_global_id(global_start(7) + 5) == (7, 5)
    with pytest.raises(ValueError):
        split_global_id(3_909_585)
    with pytest.raises(ValueError):
        global_start(0)


def test_guard_resolves_paths_and_rejects_inputs_case_insensitively(tmp_path):
    roots = make_roots(tmp_path)
    assert roots.guard(roots.docs / 'b001' / 'x.parquet')
    for bad in (roots.raw / 'cnkang.com' / 'x', roots.crawl_db, tmp_path / 'elsewhere',
                roots.docs / '..' / 'raw_full' / 'x', str(roots.raw).upper() + os.sep + 'x'):
        with pytest.raises(ValueError):
            roots.guard(bad)
    link = roots.docs / 'link'
    roots.docs.mkdir(parents=True)
    (tmp_path / 'vi').mkdir()
    try:
        os.symlink(tmp_path / 'vi', link, target_is_directory=True)
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(tmp_path / 'vi'), str(link))
    with pytest.raises(ValueError):
        roots.guard(link / 'x')


def test_cut_takes_only_committed_closed_shards(full_crawl):
    from r2ai.zh_full_ingest.ingest import cut_batch
    roots = full_crawl
    later = time.time() + 3600
    os.utime(roots.raw / 'cnkang.com' / 's3.jsonl.zst.tmp', (later, later))
    out = cut_batch(roots, 'b001', ['cnkang.com'])
    assert out['shards'] == 2 and out['urls'] == 3
    assert out['skipped'] == {'missing_file': 1, 'not_final_name_or_open': 1}
    c = sqlite3.connect(roots.manifest)
    assert sorted(r[0] for r in c.execute('SELECT url_norm FROM batch_urls')) == ['cnkang.com/a', 'cnkang.com/b', 'cnkang.com/c']
    assert c.execute("SELECT count(*) FROM shards WHERE path LIKE '%orphan%'").fetchone()[0] == 0
    sizes = dict(c.execute('SELECT path, size FROM shards'))
    assert all(v > 0 for v in sizes.values())
    with pytest.raises(ValueError, match='exists'):
        cut_batch(roots, 'b001', ['cnkang.com'])


def test_cut_skips_shards_newer_than_the_snapshot(full_crawl, monkeypatch):
    from r2ai.zh_full_ingest import ingest
    roots = full_crawl
    future = time.time() + 3600
    os.utime(roots.raw / 'cnkang.com' / 's2.jsonl.zst', (future, future))
    out = ingest.cut_batch(roots, 'b001', ['cnkang.com'])
    assert out['shards'] == 1 and out['skipped']['newer_than_snapshot'] == 2


def test_next_batch_only_takes_new_urls(full_crawl):
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch
    roots = full_crawl
    cut_batch(roots, 'b001', ['cnkang.com'])
    extract_batch(roots, 'b001', tok=fake_tok)
    write_shard(roots.raw / 'cnkang.com', 's4.jsonl.zst', [rec('cnkang.com/d', [15])])
    c = sqlite3.connect(roots.crawl_db)
    c.execute("INSERT INTO urls(url_norm,url,doc_ids,domain,rank,status,shard) VALUES ('cnkang.com/d','x','[15]','cnkang.com',9,'ok','s4.jsonl.zst')")
    c.commit()
    c.close()
    out = cut_batch(roots, 'b002', ['cnkang.com'])
    assert out['urls'] == 1 and out['shards'] == 1


def docs_of(roots, batch):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((roots.docs / batch).rglob('*.parquet')):
        rows += pq.read_table(f).to_pylist()
    return sorted(rows, key=lambda r: r['url_norm'])


def test_extract_uses_pointer_records_once_and_is_idempotent(full_crawl):
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch
    roots = full_crawl
    cut_batch(roots, 'b001', ['cnkang.com'])
    first = extract_batch(roots, 'b001', tok=fake_tok)
    assert first['shards_done'] == 2 and first['docs'] == 3
    docs = docs_of(roots, 'b001')
    assert [d['url_norm'] for d in docs] == ['cnkang.com/a', 'cnkang.com/b', 'cnkang.com/c']
    assert docs[1]['body'].endswith('第二版') and docs[1]['doc_ids'] == [11]
    assert [d['status'] for d in docs] == ['ok', 'ok', 'thin']
    again = extract_batch(roots, 'b001', tok=fake_tok)
    assert again['shards_done'] == 0 and again['docs'] == 0
    assert docs_of(roots, 'b001') == docs
    c = sqlite3.connect(roots.manifest)
    row = c.execute("SELECT status, n_docs, sha256, extractor_version FROM shards WHERE path LIKE '%s2.jsonl.zst'").fetchone()
    from r2ai.zh_full_ingest.extract import extractor_version
    assert row[0] == 'done' and row[1] == 2 and len(row[2]) == 64 and row[3] == extractor_version()


def test_extract_resumes_after_interrupt_with_identical_output(full_crawl, tmp_path):
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch
    roots = full_crawl
    cut_batch(roots, 'b001', ['cnkang.com'])
    calls = []

    def stop_after_first(where):
        calls.append(where)
        if len(calls) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        extract_batch(roots, 'b001', tok=fake_tok, before_shard=stop_after_first)
    assert not list((roots.docs / 'b001').rglob('*.partial'))
    extract_batch(roots, 'b001', tok=fake_tok)
    assert [d['url_norm'] for d in docs_of(roots, 'b001')] == ['cnkang.com/a', 'cnkang.com/b', 'cnkang.com/c']


def test_extract_refuses_a_shard_changed_after_the_cut(full_crawl):
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch
    roots = full_crawl
    cut_batch(roots, 'b001', ['cnkang.com'])
    write_shard(roots.raw / 'cnkang.com', 's1.jsonl.zst', [rec('cnkang.com/a', [10], text=ART + '改')])
    with pytest.raises(ValueError, match='changed since cut'):
        extract_batch(roots, 'b001', tok=fake_tok)


class FakeEnc:
    max_len, vocab = 512, 40

    def __init__(self, **kw):
        self.model = type('M', (), {'config': type('C', (), {'hidden_size': 1024})()})()
        self.calls = 0

    def tok(self, texts, **kw):
        return {'input_ids': [[0] * (1 + len(t)) for t in texts]}

    def encode_batch(self, texts):
        self.calls += 1
        d = np.stack([np.r_[np.full(1023, 1.0, np.float32), np.float32(len(t))] for t in texts])
        d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float16)
        return d, [(np.array([len(t) % 40], np.int32), np.array([1.0], np.float16)) for t in texts]


@pytest.fixture
def no_cuda(monkeypatch):
    import torch
    for f in ('synchronize', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, f, lambda *a, **k: None)


def build_segment(roots, batch='b001', domain='cnkang.com', block=2, interrupt_after=None):
    from r2ai.zh_full_ingest.ingest import chunk_segment, embed_segment
    seg = chunk_segment(roots, batch, domain, batch_docs=2)
    calls = []

    def before_block(i):
        calls.append(i)
        if interrupt_after is not None and len(calls) > interrupt_after:
            raise KeyboardInterrupt

    try:
        embed_segment(roots, seg['name'], encoder_factory=FakeEnc, block=block, batch_size=2,
                      gpu_check=lambda: None, before_block=before_block)
    except KeyboardInterrupt:
        embed_segment(roots, seg['name'], encoder_factory=FakeEnc, block=block, batch_size=2, gpu_check=lambda: None)
    return seg


def test_chunk_and_embed_make_an_int64_segment_with_verbatim_text(full_crawl, no_cuda):
    import pyarrow.parquet as pq
    from r2ai.index.segments import load_segment, take_segment_texts
    from r2ai.zh_full_ingest.common import global_start
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch, active_segments
    roots = full_crawl
    cut_batch(roots, 'b001', ['cnkang.com'])
    extract_batch(roots, 'b001', tok=fake_tok)
    seg = build_segment(roots)
    assert seg['segment_id'] == 2  # 1 is reserved for the retired old sample
    s = load_segment(roots.index / seg['name'])
    assert s.global_start == global_start(2)
    chunks = pq.read_table(roots.chunks / seg['name'] / 'chunks_t256.parquet')
    assert chunks.schema.field('chunk_id').type == 'int64' and chunks.schema.field('doc_id').type == 'int64'
    assert s.doc_id.dtype == np.int64 and s.sparse[2].dtype == np.int64 and s.dense.dtype == np.float16
    ids = [global_start(2) + i for i in range(len(s))]
    texts = take_segment_texts([s], ids)
    assert texts == chunks['text'].to_pylist()
    doc_text = dict(zip(*pq.read_table(roots.chunks / seg['name'] / 'docs.parquet', columns=['doc_id', 'doc_text']).to_pydict().values()))
    bodies = {d['doc_ids'][0]: d['body'] for d in docs_of(roots, 'b001')}
    for row in chunks.to_pylist():
        assert row['text'] == doc_text[row['doc_id']][row['char_start']:row['char_end']]
        assert row['field'] == 'title' or row['text'] in bodies[row['doc_id']]
    assert {d['doc_id'] for d in chunks.to_pylist()} == {10, 11}  # thin doc 12 not indexed
    assert [x.path.name for x in active_segments(roots)] == [seg['name']]


def test_embed_resume_gives_byte_identical_segment(tmp_path, no_cuda):
    import shutil
    from r2ai.index.segments import load_segment
    from r2ai.zh_full_ingest.ingest import cut_batch, extract_batch
    outs = []
    for k, interrupt in enumerate((None, 1)):
        base = tmp_path / f'r{k}'
        base.mkdir()
        roots = make_roots(base)
        d = roots.raw / 'cnkang.com'
        write_shard(d, 's1.jsonl.zst', [rec(f'cnkang.com/{i}', [20 + i], text=ART * (1 + i % 3)) for i in range(5)])
        make_db(roots.crawl_db, [(f'cnkang.com/{i}', [20 + i], 'cnkang.com', 'ok', 's1.jsonl.zst') for i in range(5)])
        cut_batch(roots, 'b001', ['cnkang.com'])
        extract_batch(roots, 'b001', tok=fake_tok)
        seg = build_segment(roots, block=2, interrupt_after=interrupt)
        p = roots.index / seg['name']
        outs.append({f: (p / f).read_bytes() for f in ('dense.npy', 'data.bin', 'indices.bin', 'indptr.bin', 'doc_id.bin')})
        assert not (roots.index / (seg['name'] + '.assembling')).exists()
    assert outs[0] == outs[1]


def test_parity_one_segment_vs_many(tmp_path):
    from scipy.sparse import random as sprandom
    from r2ai.index.segments import write_segment
    from r2ai.zh_full_ingest.common import global_start
    from r2ai.zh_full_ingest.retrieve import dense_candidates
    rng = np.random.default_rng(42)
    n = 60
    dense = rng.standard_normal((n, 1024)).astype(np.float32)
    dense = (dense / np.linalg.norm(dense, axis=1, keepdims=True)).astype(np.float16)
    sparse = sprandom(n, 30, density=.2, format='csr', dtype=np.float32, random_state=1)
    sparse.sort_indices()
    doc = np.repeat(np.arange(20), 3).astype(np.int64)
    texts = [f't{i}' for i in range(n)]
    roots = make_roots(tmp_path)
    one = [write_segment(roots.index, 'one', dense, sparse, doc, texts, global_start=global_start(5))]
    many = []
    for j, (lo, hi) in enumerate(((0, 21), (21, 39), (39, 60))):
        many.append(write_segment(roots.index, f'm{j}', dense[lo:hi], sparse[lo:hi], doc[lo:hi], texts[lo:hi],
                                  global_start=global_start(10 + j)))
    qd = rng.standard_normal((7, 1024)).astype(np.float32)
    qd = (qd / np.linalg.norm(qd, axis=1, keepdims=True)).astype(np.float16)
    a, b = dense_candidates(one, qd, 15), dense_candidates(many, qd, 15)
    for (ia, sa), (ib, sb) in zip(a, b):
        # Same ids; fp32 dot products may differ by 1 ULP when BLAS blocks differ.
        assert np.max(np.abs(sa - sb)) < 1e-6
        ta = [texts[int(i - global_start(5))] for i in ia]
        offsets = {10: 0, 11: 21, 12: 39}
        tb = [texts[offsets[(int(i) - 2**48) // 2**32] + (int(i) - 2**48) % 2**32] for i in ib]
        assert ta == tb and ib.dtype == np.int64


def fake_old_sample(roots, docs):
    """docs: list of (doc_id, domain, text). Writes an old-format index + chunks like zh_sample."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix, save_npz
    roots.sample_chunks.mkdir(parents=True)
    roots.sample_index.mkdir(parents=True)
    rows = []
    for doc_id, domain, text in docs:
        rows.append({'chunk_id': len(rows), 'doc_id': doc_id, 'doc_ids_group': [doc_id], 'field': 'body',
                     'char_start': 0, 'char_end': len(text), 'text': text, 'n_tokens': len(text)})
    pq.write_table(pa.Table.from_pylist(rows), roots.sample_chunks / 'chunks_t256.parquet')
    pq.write_table(pa.table({'doc_id': [d for d, _, _ in docs], 'domain': [m for _, m, _ in docs]}),
                   roots.sample_chunks / 'docs.parquet')
    n = len(rows)
    dense = np.random.default_rng(3).standard_normal((n, 1024)).astype(np.float16)
    np.save(roots.sample_index / 'dense.npy', dense)
    sp = csr_matrix((np.arange(1, n + 1, dtype=np.float32), np.arange(n) % 7, np.arange(n + 1)), shape=(n, 11))
    save_npz(roots.sample_index / 'sparse.npz', sp)
    return dense, sp


def test_old_sample_becomes_retired_segment_1_byte_identical(tmp_path):
    from r2ai.index.segments import load_segment
    from r2ai.zh_full_ingest.common import global_start
    from r2ai.zh_full_ingest.ingest import convert_old_sample, active_segments
    roots = make_roots(tmp_path)
    dense, sp = fake_old_sample(roots, [(5, 'a.cn', 'x' * 9), (5, 'a.cn', 'y' * 9), (7, 'b.cn', 'z' * 9)])
    out = convert_old_sample(roots, writer_block=2)
    assert out['segment_id'] == 1 and out['status'] == 'retired'
    s = load_segment(roots.index / out['name'])
    assert s.global_start == global_start(1)
    assert np.array_equal(np.asarray(s.dense[0:3]).view(np.uint16), dense.view(np.uint16))
    assert np.array_equal(np.asarray(s.sparse[0][0:3]), sp.data) and np.array_equal(np.asarray(s.sparse[2][0:4]), sp.indptr)
    assert list(np.asarray(s.doc_id)) == [5, 5, 7]
    assert active_segments(roots) == []
    assert [x.path.name for x in active_segments(roots, include_retired=True)] == [out['name']]
    with pytest.raises(ValueError):
        convert_old_sample(roots)


def test_copy_domain_vectors_only_when_chunks_are_identical(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.index.segments import load_segment
    from r2ai.zh_full_ingest.ingest import copy_sample_vectors, allocate_segment
    roots = make_roots(tmp_path)
    dense, sp = fake_old_sample(roots, [(5, 'a.cn', 'x' * 9), (6, 'b.cn', 'y' * 9), (7, 'a.cn', 'z' * 9)])
    old = pq.read_table(roots.sample_chunks / 'chunks_t256.parquet').to_pylist()

    def new_chunks(name, rows):
        d = roots.chunks / name
        d.mkdir(parents=True)
        rows = [{**r, 'chunk_id': i} for i, r in enumerate(rows)]
        pq.write_table(pa.Table.from_pylist(rows), d / 'chunks_t256.parquet')

    seg = allocate_segment(roots, 'sample-v2', 'a.cn', kind='sample')
    new_chunks(seg['name'], [old[0], old[2]])
    out = copy_sample_vectors(roots, seg['name'], 'a.cn', writer_block=1)
    s = load_segment(roots.index / seg['name'])
    assert np.array_equal(np.asarray(s.dense[0:2]).view(np.uint16), dense[[0, 2]].view(np.uint16))
    assert out['vectors'] == 'copied' and out['old_rows'] == [0, 2]
    bad = allocate_segment(roots, 'sample-v2', 'b.cn', kind='sample')
    new_chunks(bad['name'], [{**old[1], 'text': 'changed'}])
    with pytest.raises(ValueError, match='differ'):
        copy_sample_vectors(roots, bad['name'], 'b.cn')


def test_segment_ids_are_never_reused(tmp_path):
    from r2ai.zh_full_ingest.ingest import allocate_segment, retire_segment
    roots = make_roots(tmp_path)
    a = allocate_segment(roots, 'b001', 'x.cn')
    retire_segment(roots, a['name'], 'test')
    b = allocate_segment(roots, 'b001', 'y.cn')
    assert (a['segment_id'], b['segment_id']) == (2, 3)


def test_memory_guard_waits_then_stops_after_too_many_pauses(tmp_path):
    from r2ai.zh_full_ingest.common import MemoryGuard
    values = iter([3.0, 5.0, 3.0, 3.0, 3.0, 3.0])
    slept = []
    clock = iter(range(0, 1000, 10))
    g = MemoryGuard(make_roots(tmp_path), available=lambda: next(values), sleep=slept.append, clock=lambda: next(clock))
    assert g.check('a') == 5.0 and slept == [60]
    with pytest.raises(RuntimeError, match='MEMORY_PAUSES'):
        g.check('b')


def test_cut_sample_keeps_the_latest_copy_like_zh_sample(tmp_path):
    from r2ai.zh_full_ingest.ingest import cut_sample, extract_batch, compare_sample_docs, _write_parquet
    roots = make_roots(tmp_path)
    d = roots.sample_raw / 'cnkang.com'
    write_shard(d, '20261005T01-a.jsonl.zst', [rec('cnkang.com/x', [1], fetched_at=5.0), rec('cnkang.com/y', [2], fetched_at=1.0)])
    write_shard(d, '20261005T02-b.jsonl.zst', [rec('cnkang.com/x', [1], text=ART + '新', fetched_at=6.0),
                                               rec('cnkang.com/y', [2], text=ART + '同时', fetched_at=1.0),
                                               rec('cnkang.com/z', [3], status='network_error')])
    out = cut_sample(roots, 'sample-v2')
    assert out['urls'] == 2 and out['by_domain'] == {'cnkang.com': 2}
    extract_batch(roots, 'sample-v2', tok=fake_tok)
    docs = docs_of(roots, 'sample-v2')
    assert docs[0]['body'].endswith('新') and not docs[1]['body'].endswith('同时')  # tie keeps the earlier shard
    _write_parquet(roots, docs, roots.docs / 'scratch' / 'old.parquet')
    roots.sample_docs.mkdir(parents=True)
    os.replace(roots.docs / 'scratch' / 'old.parquet', roots.sample_docs / 'cnkang.com.parquet')
    assert compare_sample_docs(roots, 'sample-v2', 'cnkang.com')['equal']
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(roots.sample_docs / 'cnkang.com.parquet').to_pylist()
    t[0]['title'] = '别的'
    pq.write_table(pa.Table.from_pylist(t), roots.sample_docs / 'cnkang.com.parquet')
    assert not compare_sample_docs(roots, 'sample-v2', 'cnkang.com')['equal']


def test_monitor_alerts_only_after_two_slow_windows(tmp_path):
    from r2ai.zh_full_ingest.monitor import evaluate
    assert not evaluate(2.0, [1.5])
    assert not evaluate(2.0, [1.5, 1.7])
    assert evaluate(2.0, [1.7, 1.5, 1.5])
    assert not evaluate(0.0, [0.0, 0.0])


def test_cli_stops_before_the_next_unit_on_a_monitor_alert(tmp_path):
    from r2ai.zh_full_ingest.__main__ import _stop_on_alert
    roots = make_roots(tmp_path)
    check = _stop_on_alert(roots)
    check('x')
    roots.run.mkdir(parents=True)
    (roots.run / 'monitor_alert.json').write_text('{}')
    with pytest.raises(SystemExit, match='CRAWL_SLOWDOWN'):
        check('x')
