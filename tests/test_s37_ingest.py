import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from r2ai.s37 import ingest as I

HTML = ('<html><body><h1>头痛三天</h1><div class="b_askcont"><p>我最近头痛已经三天了，伴有恶心，没有发烧，吃了止痛药也不见好转，'
        '请问这是什么原因引起的，需要做什么检查，平时饮食和睡眠应该注意什么，谢谢医生</p></div>'
        '<div class="b_anscont_cont"><p>您好，根据您的描述考虑为紧张性头痛或偏头痛，建议注意休息，避免熬夜，保持情绪稳定，必要时到神经内科就诊做相关检查。</p></div>'
        '</body></html>')
THIN = '<html><body><h1>t</h1><div class="b_askcont"><p>你好</p></div></body></html>'


def record(doc_id, html, url='https://www.120ask.com/question/1.htm'):
    return {'doc_id': str(doc_id), 'url': url, 'final_url': url, 'http': 200, 'fetched_at': 0, 'encoding': 'utf-8', 'html': html}


def make_tar(tmp_path, records):
    blob = gzip.compress(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode('utf-8'))
    name = 'r1-00000.jsonl.gz'
    manifest = {'batch': 1, 'shards': [{'name': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}],
                'urls': len(records), 'ok': len(records), 'suspect': 0, 'failed': 0}
    tar = tmp_path / 'batch-0001.tar'
    with tarfile.open(tar, 'w') as t:
        for arc, data in ((f'shards/{name}', blob), ('MANIFEST.json', json.dumps(manifest).encode()),
                          ('log.jsonl', b'')):
            ti = tarfile.TarInfo(arc)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    (tmp_path / 'batch-0001.tar.sha256').write_text(f'{I.sha256_file(tar)}  batch-0001.tar\n')
    return tar


def test_read_tar_verifies_hashes_and_keeps_last_record(tmp_path):
    tar = make_tar(tmp_path, [record(1, THIN), record(1, HTML), record(2, HTML)])
    _, _, recs = I.read_tar(tar)
    assert {r['doc_id']: r['html'] for r in recs}['1'] == HTML
    (tmp_path / 'batch-0001.tar.sha256').write_text('0' * 64 + '  x\n')
    with pytest.raises(ValueError):
        I.read_tar(tar)


def test_build_docs_han_threshold_shared_url_and_existing_ids():
    recs = [record(10, HTML), record(20, THIN), record(30, HTML)]
    groups = {'10': [10, 11], '20': [20], '30': [30]}
    rows, drop, srcs = I.build_docs(recs, groups, have={30}, segment='seg37001-test')
    assert sorted(r['doc_id'] for r in rows) == [10, 11]                       # THIN dropped, 30 already has text
    assert drop['thin'] == 1 and drop['already_have'] == 1
    assert rows[0]['phys_id'] == 10 and rows[1]['phys_id'] == 10 and rows[0]['text_sha256'] == rows[1]['text_sha256']
    assert rows[0]['han'] >= I.MIN_HAN and rows[0]['quality'] == 'good' and rows[0]['doc_text'].startswith('头痛三天')


def test_pair_rows_entry_rank_and_pool_flags():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ref = np.sort(rng.uniform(0.2, 0.8, size=(3, 6)), axis=1)[:, ::-1].astype(np.float32)   # descending old scores
    pool = {'q_ids': np.array([7, 8, 9]), 'scores': ref, 'size13': np.array([4, 4, 4]), 'thr211': ref[:, 2], 'thr': ref[:, 3]}
    d = np.stack([q[0], -q[1]]).astype(np.float16)            # doc 0 is the query-0 vector (cos 1), doc 1 opposes query 1
    df = I.pair_rows(pool, q, np.array([100, 101]), d, 'segX', 'b0001')
    top = df.filter(df['query_id'] == 7)
    assert top['doc_id'].to_list() == [100] and top['entry_rank'].to_list() == [1] and top['in_top211'].to_list() == [True]
    assert (df['cosine'].to_numpy() > np.array(pool['thr'])[np.searchsorted(pool['q_ids'], df['query_id'].to_numpy())]).all()


def test_top_scores_matches_full_sort():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4, 16)).astype(np.float16)
    d = rng.normal(size=(1000, 16)).astype(np.float16)
    s, idx = I.top_scores(q, d, 25, block=300)
    full = q.astype(np.float32) @ d.astype(np.float32).T
    assert np.allclose(s, -np.sort(-full, axis=1)[:, :25], atol=1e-5)
    assert np.allclose(np.take_along_axis(full, idx, 1), s, atol=1e-5)


def test_chunk_segment_verbatim_one_group_per_text():
    rows, _, srcs = I.build_docs([record(10, HTML)], {'10': [10, 11]}, set(), 'seg37001-test')
    ch = I.chunk_segment(srcs, 'seg37001-test')
    doc_text = rows[0]['doc_text']
    assert ch.height >= 3 and set(ch['doc_id']) == {10} and ch['doc_ids'][0].to_list() == [10, 11]
    assert {'title', 'question', 'answer'} <= set(ch['field']) or 'body' in set(ch['field'])
    assert all(doc_text[s:e] == t for s, e, t in zip(ch['char_start'], ch['char_end'], ch['text']))
    assert not ch['has_vector'].any()


def test_batch_defers_embed_then_embeds_when_gpu_free(tmp_path, monkeypatch):
    monkeypatch.setattr(I, 'STORE', tmp_path / 'store')
    monkeypatch.setattr(I, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(I, 'existing_ids', lambda: set())
    urls = tmp_path / 'urls.tsv'
    urls.write_text('1\thttps://www.120ask.com/question/1.htm\t1\n2\thttps://www.120ask.com/question/2.htm\t2\n', encoding='utf-8')
    tar = make_tar(tmp_path, [record(1, HTML), record(2, THIN)])
    res = I.run_batch(tar, 'defer', urls=urls)
    assert res['status'] == 'extracted' and res['state'] == 'waiting_gpu' and res['docs_kept'] == 1 and res['chunks_t256'] >= 3
    assert I.status()['waiting_gpu'] == [res['segment']]
    assert I.run_batch(tar, 'defer', urls=urls)['status'] == 'already_extracted'
    monkeypatch.setattr(I, 'gpu_free', lambda *a, **k: (False, 'busy'))
    assert I.embed_pending('cuda')['status'] == 'gpu_busy'
    assert I.status()['waiting_gpu'] == [res['segment']]
    q = np.zeros((2, 1024), np.float32)
    q[0, 0] = q[1, 1] = 1
    pool = {'q_ids': np.array([1, 2]), 'scores': np.full((2, 4), 0.5, np.float32), 'size13': np.array([2, 2]),
            'thr211': np.full(2, 0.5, np.float32), 'thr': np.full(2, 0.5, np.float32), 'cut': np.array([2, 2])}
    monkeypatch.setattr(I, 'build_pool', lambda force=False: pool)
    monkeypatch.setattr(I, 'load_queries', lambda: (pool['q_ids'], q))
    monkeypatch.setattr(I, 'parity', lambda p: {})
    vec = np.zeros((1, 1024), np.float16)
    vec[0, 0] = 1
    monkeypatch.setattr(I, 'embed_texts', lambda texts, device: np.repeat(vec, len(texts), axis=0))
    monkeypatch.setattr(I, 'gpu_free', lambda *a, **k: (True, 'idle'))
    out = I.embed_pending('cuda')
    assert out['status'] == 'embedded' and out['cumulative']['pairs'] == 1 and I.status()['waiting_gpu'] == []
    pairs = __import__('polars').read_parquet(tmp_path / 'run' / 'pairs_new.parquet')
    assert pairs['query_id'].to_list() == [1] and pairs['in_top211'].to_list() == [True]
