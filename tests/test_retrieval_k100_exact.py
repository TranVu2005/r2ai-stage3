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
    assert meta['peak_rss_gib'] > 0 and 'dense_seconds' in meta['exact'] and 'peak_vram_allocated_mib' in meta
    cs = pq.read_table(env['tmp'] / 'new' / 'vi_k100.candidates.parquet')
    assert cs.num_rows == 30 and cs.column_names == ['query_id', 'n_candidates', 'n_docs']
    assert (env['idx'] / 'sparse_mmap' / 'shape.json').exists() and (env['idx'] / 'text.arrow').exists()


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


def test_candidates_only_full_ranking_matches_k100(env, monkeypatch):
    from r2ai.index import bge_m3
    k100 = env['k100']
    base = ['--queries', str(env['q']), '--candidates', 'exact']
    assert k100.main(base + ['--out-dir', str(env['tmp'] / 'full')]) == 0

    class NoRR:
        def __init__(self, **kw):
            raise AssertionError('reranker must not be loaded')
    monkeypatch.setattr(bge_m3, 'Reranker', NoRR)
    out = env['tmp'] / 'cand'
    assert k100.main(base + ['--out-dir', str(out), '--candidates-only', '--block-rows', '97']) == 0
    assert not (out / '_partial.pkl').exists() and not (out / 'vi_k100.parquet').exists()
    full = env['tmp'] / 'full'
    assert pq.read_table(out / 'vi_cand.candidates.parquet').equals(pq.read_table(full / 'vi_k100.candidates.parquet'))
    docs = pq.read_table(out / 'vi_cand.docs.parquet').to_pandas()
    chunks = pq.read_table(out / 'vi_cand.chunks.parquet').to_pandas()
    k = pq.read_table(full / 'vi_k100.parquet').to_pandas()
    for q, g in docs.groupby('query_id'):
        assert list(g['rank']) == list(range(1, len(g) + 1)) and g['doc_id'].is_unique
        assert (np.diff(g['score'].to_numpy()) <= 0).all()
        c = chunks[chunks.query_id == q]
        assert set(g['doc_id']) == set(c['doc_id'])
        assert np.allclose(g.set_index('doc_id')['score'], c.groupby('doc_id')['hybrid'].max().loc[g['doc_id']])
        kq = k[k.query_id == q]
        assert set(kq['doc_id']) <= set(g['doc_id'])
        t1 = set(kq[kq.tier == 1]['doc_id'])
        rest = [d for d in g['doc_id'] if d not in t1][:int((kq.tier == 2).sum())]
        assert set(rest) == set(kq[kq.tier == 2]['doc_id'])          # tier 2 = next docs by hybrid order
    with pytest.raises(ValueError, match='overwrite'):
        k100.main(base + ['--out-dir', str(out), '--candidates-only'])
    with pytest.raises(SystemExit):
        k100.main(['--queries', str(env['q']), '--out-dir', str(env['tmp'] / 'y'), '--candidates-only'])


def _run(env, name, *extra):
    out = env['tmp'] / name
    assert env['k100'].main(['--queries', str(env['q']), '--candidates', 'exact', '--out-dir', str(out), *extra]) == 0
    return out


def test_tier_flags_default_values_reproduce_default(env):
    base = _run(env, 'base')
    same = _run(env, 'same', '--tier1-docs', '50', '--tier2-docs', '50', '--chunk-score-docs', '50', '--checkpoint-every', '7')
    for f in ('vi_k100.parquet', 'vi_k100_chunk_scores.parquet', 'vi_k100.candidates.parquet'):
        assert pq.read_table(base / f).equals(pq.read_table(same / f)), f
    assert not (base / 'vi_k100_pairs.parquet').exists()
    assert 'tier1_docs' not in json.loads((base / 'vi_k100.meta.json').read_text(encoding='utf-8'))


def test_deep_tier1_keeps_old_tier1_scores_and_takes_next_docs(env):
    base = _run(env, 'base')
    cand = _run(env, 'cand', '--candidates-only')
    deep = _run(env, 'deep', '--tier1-docs', '180', '--tier2-docs', '20', '--chunk-score-docs', '0', '--pair-scores')
    old = pq.read_table(base / 'vi_k100.parquet').to_pandas()
    new = pq.read_table(deep / 'vi_k100.parquet').to_pandas()
    hyb = pq.read_table(cand / 'vi_cand.docs.parquet').to_pandas()
    chunks = pq.read_table(cand / 'vi_cand.chunks.parquet').to_pandas()
    pairs = pq.read_table(deep / 'vi_k100_pairs.parquet').to_pandas()
    assert pq.read_table(deep / 'vi_k100_chunk_scores.parquet').num_rows == 0
    extended = 0
    for q, g in new.groupby('query_id'):
        h = hyb[hyb.query_id == q].sort_values('rank')['doc_id'].tolist()
        t1 = g[g.tier == 1]
        assert set(t1['doc_id']) == set(h[:180]) and set(g[g.tier == 2]['doc_id']) == set(h[180:200])
        assert list(g['rank']) == list(range(1, len(g) + 1)) and (np.diff(t1['score'].to_numpy()) <= 0).all()
        o = old[(old.query_id == q) & (old.tier == 1)]
        sc = dict(zip(t1['doc_id'], t1['score']))
        assert np.array_equal([sc[d] for d in o['doc_id']], o['score'].to_numpy())          # same pairs, same scores
        assert [d for d in t1['doc_id'] if d in set(o['doc_id'])] == o['doc_id'].tolist()   # same relative order
        c = chunks[chunks.query_id == q].sort_values('hybrid', ascending=False, kind='stable')
        top200 = c.iloc[:200]
        p1 = pairs[(pairs.query_id == q) & (pairs.tier == 1)]
        for d, pg in p1.groupby('doc_id'):
            mine = set(top200[top200.doc_id == d]['chunk_id'])
            if mine:
                assert set(pg['chunk_id']) == mine
            else:
                extended += 1
            assert np.isclose(pg['score'].max(), sc[d])
        assert set(p1['doc_id']) == set(t1['doc_id'])
    assert extended > 0                                                     # the fixture exercises the extension


def test_checkpoint_resume_after_interrupt_and_config_guard(env, monkeypatch):
    from r2ai.index import bge_m3
    full = _run(env, 'full', '--tier1-docs', '60', '--pair-scores')
    calls = {'n': 0}

    class StopRR(FakeRR):
        def score(self, q, passages, batch_size=32):
            calls['n'] += 1
            if calls['n'] == 400:
                raise KeyboardInterrupt
            return super().score(q, passages, batch_size)
    monkeypatch.setattr(bge_m3, 'Reranker', StopRR)
    out = env['tmp'] / 'cut'
    common = ['--queries', str(env['q']), '--candidates', 'exact', '--out-dir', str(out), '--pair-scores']
    args = common + ['--tier1-docs', '60', '--checkpoint-every', '4']
    assert env['k100'].main(args) == 130
    import pickle
    saved = pickle.load(open(out / '_partial.pkl', 'rb'))
    assert 0 < len(saved) < 30 and not (out / 'vi_k100.parquet').exists()
    monkeypatch.setattr(bge_m3, 'Reranker', FakeRR)
    with pytest.raises(ValueError, match='config'):
        env['k100'].main(common + ['--tier1-docs', '70', '--checkpoint-every', '4'])
    assert env['k100'].main(args) == 0
    for f in ('vi_k100.parquet', 'vi_k100_chunk_scores.parquet', 'vi_k100_pairs.parquet', 'vi_k100.candidates.parquet'):
        assert pq.read_table(full / f).equals(pq.read_table(out / f)), f


def test_sample_draws_seed42_queries(env):
    import random
    out = _run(env, 's', '--sample', '7')
    random.seed(42)
    want = sorted(random.sample(range(30), 7))
    got = sorted(set(pq.read_table(out / 'vi_k100_sample.parquet')['query_id'].to_pylist()))
    assert got == want
    meta = json.loads((out / 'vi_k100_sample.meta.json').read_text(encoding='utf-8'))
    assert meta['query_ids'] == want and meta['sample'] == 7
    with pytest.raises(SystemExit):
        env['k100'].main(['--queries', str(env['q']), '--out-dir', str(out), '--sample', '3', '--limit', '3'])


def test_deep_gate_passes_on_fixture_and_flags_changes(env):
    from r2ai.retrieve import deep_gate
    base = _run(env, 'base')
    deep = _run(env, 'deep', '--tier1-docs', '120', '--pair-scores')
    same = _run(env, 'same', '--sample', '9')
    assert deep_gate.main(['deep', '--old', str(base), '--new', str(deep), '--out', str(env['tmp'] / 'g1.json')]) == 0
    g1 = json.loads((env['tmp'] / 'g1.json').read_text(encoding='utf-8'))
    assert g1['pairs_within_tol_share'] == 1.0 and g1['same_order_share'] == 1.0 and g1['chunk_level']['common_rows'] > 0
    assert deep_gate.main(['same', '--old', str(base), '--new', str(same), '--tag', '_sample',
                           '--out', str(env['tmp'] / 'g2.json')]) == 0
    g2 = json.loads((env['tmp'] / 'g2.json').read_text(encoding='utf-8'))
    assert g2['queries'] == 9 and g2['rank_doc_tier_identical'] and g2['chunk_rows_identical']
    t = pq.read_table(deep / 'vi_k100.parquet').to_pandas()
    t.loc[t.index[0], 'score'] += 0.01                                    # one changed old tier-1 score -> gate fails
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), deep / 'vi_k100.parquet')
    assert deep_gate.main(['deep', '--old', str(base), '--new', str(deep), '--out', str(env['tmp'] / 'g3.json'),
                           '--min-score-share', '1.0']) == 1


def test_deep_tier1_keeps_default_scores_with_batch_dependent_reranker(env, monkeypatch):
    """fp16 scores move with the batch composition; the top-50 pairs must still see the default run's batches."""
    from r2ai.index import bge_m3

    class BatchRR(FakeRR):
        def score(self, q, passages, batch_size=32):
            return super().score(q, passages, batch_size) + np.float32(len(passages) * 1e-3)
    monkeypatch.setattr(bge_m3, 'Reranker', BatchRR)
    base = _run(env, 'base')
    deep = _run(env, 'deep', '--tier1-docs', '120', '--chunk-score-docs', '0')
    old = pq.read_table(base / 'vi_k100.parquet').to_pandas()
    new = pq.read_table(deep / 'vi_k100.parquet').to_pandas()
    m = old[old.tier == 1].merge(new, on=['query_id', 'doc_id'], suffixes=('_old', '_new'))
    assert len(m) == (old.tier == 1).sum() and (m['tier_new'] == 1).all()
    assert np.array_equal(m['score_old'].to_numpy(), m['score_new'].to_numpy())


def test_second_run_on_same_out_dir_is_refused_while_first_lives(env):
    import subprocess
    import sys
    out = env['tmp'] / 'locked'
    out.mkdir()
    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', 'run_retrieval_k100'])
    try:
        (out / '_partial.pkl.lock').write_text(str(p.pid), encoding='utf-8')
        with pytest.raises(RuntimeError, match='already using'):
            env['k100'].main(['--queries', str(env['q']), '--candidates', 'exact', '--out-dir', str(out)])
    finally:
        p.kill()
        p.wait()
    assert env['k100'].main(['--queries', str(env['q']), '--candidates', 'exact', '--out-dir', str(out)]) == 0   # stale lock


def _batch_rr(monkeypatch):
    from r2ai.index import bge_m3

    class BatchRR(FakeRR):
        def score(self, q, passages, batch_size=32):
            return super().score(q, passages, batch_size) + np.float32(len(passages) * 1e-3)
    monkeypatch.setattr(bge_m3, 'Reranker', BatchRR)


def test_tier1_base_nests_base_run_scores_with_batch_dependent_reranker(env, monkeypatch):
    """--tier1-base N0: the N0-doc run's tier-1 docs, pairs and scores are kept exactly; plain deeper tier 1 is not."""
    _batch_rr(monkeypatch)
    base = _run(env, 'base', '--tier1-docs', '80', '--tier2-docs', '0', '--chunk-score-docs', '0', '--pair-scores')
    nest = _run(env, 'nest', '--tier1-docs', '160', '--tier1-base', '80', '--tier2-docs', '0', '--chunk-score-docs', '0', '--pair-scores')
    plain = _run(env, 'plain', '--tier1-docs', '160', '--tier2-docs', '0', '--chunk-score-docs', '0', '--pair-scores')
    old = pq.read_table(base / 'vi_k100.parquet').to_pandas()
    new = pq.read_table(nest / 'vi_k100.parquet').to_pandas()
    pl = pq.read_table(plain / 'vi_k100.parquet').to_pandas()
    m = old.merge(new, on=['query_id', 'doc_id'], suffixes=('_old', '_new'))
    assert len(m) == len(old) and (m['tier_new'] == 1).all()
    assert np.array_equal(m['score_old'].to_numpy(), m['score_new'].to_numpy())
    for q, g in new.groupby('query_id'):                                    # same tier-1 doc set as the plain deep run
        assert set(g['doc_id']) == set(pl[pl.query_id == q]['doc_id'])
        assert (np.diff(g['score'].to_numpy()) <= 0).all()
    op = pq.read_table(base / 'vi_k100_pairs.parquet').to_pandas()
    npairs = pq.read_table(nest / 'vi_k100_pairs.parquet').to_pandas()
    mp = op.merge(npairs, on=['query_id', 'doc_id', 'chunk_id'], suffixes=('_old', '_new'))
    assert len(mp) == len(op) and np.array_equal(mp['score_old'].to_numpy(), mp['score_new'].to_numpy())
    kept = set(zip(old['query_id'], old['doc_id']))                         # no extra chunk for a base doc
    extra = npairs[[k in kept for k in zip(npairs['query_id'], npairs['doc_id'])]]
    assert len(extra) == len(op)
    mpl = old.merge(pl, on=['query_id', 'doc_id'], suffixes=('_old', '_new'))
    assert not np.array_equal(mpl['score_old'].to_numpy(), mpl['score_new'].to_numpy())   # plain run drifts
    meta = json.loads((nest / 'vi_k100.meta.json').read_text(encoding='utf-8'))
    assert meta['tier1_base'] == 80 and 'base_doc_chunks_not_reranked' in meta
    cfg = json.loads((plain / '_partial.pkl.config.json').read_text(encoding='utf-8'))
    assert 'tier1_base' not in cfg


def test_tier1_base_levels_nest_a_nested_run(env, monkeypatch):
    """--tier1-base N0,N1: the reranker calls of a --tier1-docs N1 --tier1-base N0 run are kept (50 | 51..N0 | N0+1..N1),
    docs N1+1..N go to a call of their own; a single level N1 merges two of those calls and drifts."""
    _batch_rr(monkeypatch)
    args = ('--tier2-docs', '0', '--chunk-score-docs', '0', '--pair-scores')
    cache = _run(env, 'cache', '--tier1-docs', '120', '--tier1-base', '60', *args)
    multi = _run(env, 'multi', '--tier1-docs', '180', '--tier1-base', '60,120', *args)
    single = _run(env, 'single', '--tier1-docs', '180', '--tier1-base', '120', *args)
    plain = _run(env, 'plain', '--tier1-docs', '180', *args)
    old = pq.read_table(cache / 'vi_k100_pairs.parquet').to_pandas()
    key = ['query_id', 'doc_id', 'chunk_id']
    for d, same in ((multi, True), (single, False)):
        m = old.merge(pq.read_table(d / 'vi_k100_pairs.parquet').to_pandas(), on=key, suffixes=('_old', '_new'))
        assert len(m) == len(old)
        assert np.array_equal(m['score_old'].to_numpy(), m['score_new'].to_numpy()) == same
    od = pq.read_table(cache / 'vi_k100.parquet').to_pandas()
    nd = pq.read_table(multi / 'vi_k100.parquet').to_pandas()
    m = od.merge(nd, on=['query_id', 'doc_id'], suffixes=('_old', '_new'))
    assert len(m) == len(od) and np.array_equal(m['score_old'].to_numpy(), m['score_new'].to_numpy())
    pl = pq.read_table(plain / 'vi_k100.parquet').to_pandas()
    assert (nd.groupby('query_id').size() > 120).any()                     # docs past the second level exist
    for q, g in nd.groupby('query_id'):                                     # same tier-1 doc set as the plain run
        assert set(g['doc_id']) == set(pl[pl.query_id == q]['doc_id'])
    np_ = pq.read_table(multi / 'vi_k100_pairs.parquet').to_pandas()
    kept = set(zip(od['query_id'], od['doc_id']))                           # no extra chunk for a cache doc
    assert sum(k in kept for k in zip(np_['query_id'], np_['doc_id'])) == len(old)
    meta = json.loads((multi / 'vi_k100.meta.json').read_text(encoding='utf-8'))
    assert meta['tier1_base'] == [60, 120]
    assert json.loads((multi / '_partial.pkl.config.json').read_text(encoding='utf-8'))['tier1_base'] == [60, 120]
    assert json.loads((cache / '_partial.pkl.config.json').read_text(encoding='utf-8'))['tier1_base'] == 60


def test_tier1_base_argument_checks(env):
    for bad in (['--tier1-docs', '100', '--tier1-base', '40'], ['--tier1-docs', '100', '--tier1-base', '100'],
                ['--tier1-docs', '180', '--tier1-base', '60,60'], ['--tier1-docs', '180', '--tier1-base', '120,60'],
                ['--tier1-docs', '180', '--tier1-base', '40,120'], ['--tier1-docs', '180', '--tier1-base', '60,180'],
                ['--tier1-docs', '180', '--tier1-base', '60,x']):
        with pytest.raises(SystemExit):
            env['k100'].main(['--queries', str(env['q']), '--candidates', 'exact', '--out-dir', str(env['tmp'] / 'z'), *bad])


def test_deep_gate_pairs_flag(env, monkeypatch):
    from r2ai.retrieve import deep_gate
    _batch_rr(monkeypatch)
    base = _run(env, 'base', '--tier1-docs', '80', '--tier2-docs', '10', '--chunk-score-docs', '0', '--pair-scores')
    nest = _run(env, 'nest', '--tier1-docs', '160', '--tier1-base', '80', '--tier2-docs', '0', '--chunk-score-docs', '0', '--pair-scores')
    g = env['tmp'] / 'g.json'
    assert deep_gate.main(['deep', '--old', str(base), '--new', str(nest), '--out', str(g), '--tol', '0', '--pairs']) == 0
    r = json.loads(g.read_text(encoding='utf-8'))
    assert r['pair_level']['old_tier1_pairs'] > 0 and r['pair_level']['missing'] == 0 and r['pair_level']['rows_over_tol'] == 0
    assert r['doc_score_abs_delta']['max'] == 0.0
    p = pq.read_table(nest / 'vi_k100_pairs.parquet').to_pandas()
    old = pq.read_table(base / 'vi_k100_pairs.parquet').to_pandas()
    i = p.index[(p.query_id == old.query_id[0]) & (p.chunk_id == old.chunk_id[0])][0]
    p.loc[i, 'score'] = np.float32(p.loc[i, 'score'] - 5.0)                # lower one pair: doc max may hold, pair gate must not
    pq.write_table(pa.Table.from_pandas(p, preserve_index=False), nest / 'vi_k100_pairs.parquet')
    assert deep_gate.main(['deep', '--old', str(base), '--new', str(nest), '--out', str(g), '--tol', '0', '--pairs']) == 1
    assert json.loads(g.read_text(encoding='utf-8'))['pair_level']['rows_over_tol'] == 1
    plain = deep_gate.main(['deep', '--old', str(base), '--new', str(nest), '--out', str(g), '--tol', '0'])
    assert 'pair_level' not in json.loads(g.read_text(encoding='utf-8')) and plain in (0, 1)
