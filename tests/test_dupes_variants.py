"""H1 variants on a small fixture: identity == make_submission byte for byte; dedup / expand change only the intended parts."""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.eval.scorer import whitespace_tokenizer
import r2ai.submit.make_submission as ms
from r2ai.dupes import variants as V

IDS = list(range(10, 120))                  # 110 docs; top-100 cache = IDS[:100] (q1) / reversed (q2), pool = the rest
K_TOTAL, K_CHUNK = 104, 3
GROUPS = {d: [d] + ([d + 1000] if d % 3 == 0 else []) for d in IDS}
CLUSTERS = {0: [10, 12], 1: [11, 50], 2: [101, 110], 3: [20, 119], 4: [14, 60]}           # cluster id -> members


def _body(d):
    return ' '.join(f'd{d}w{w}' for w in range(60))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, 'Tokenizer', whitespace_tokenizer)
    for p in ('runs', 'chunks', 'docs_vi', 'H1/work'):
        (tmp_path / p).mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2], 'query': ['a', 'b']}), tmp_path / 'query.parquet')
    pq.write_table(pa.table({'doc_ids': [GROUPS[d] for d in IDS], 'title': ['t'] * len(IDS), 'question': [None] * len(IDS),
                             'description': [None] * len(IDS), 'answer': [None] * len(IDS), 'body': [_body(d) for d in IDS],
                             'status': ['ok'] * len(IDS)}), tmp_path / 'docs_vi/p0.parquet')
    pq.write_table(pa.table({'doc_id': IDS, 'doc_ids_group': [GROUPS[d] for d in IDS]}), tmp_path / 'chunks/docs.parquet')
    order = {1: IDS, 2: IDS[::-1]}
    run = [(q, k, d) for q in (1, 2) for k, d in enumerate(order[q][:100], 1)]
    run += [(q, 100 + k, d) for q in (1, 2) for k, d in enumerate(order[q][100:], 1)]            # tier-2 rows are ignored by the builder
    pq.write_table(pa.table({'query_id': [r[0] for r in run], 'rank': pa.array([r[1] for r in run], pa.int32()),
                             'doc_id': [r[2] for r in run], 'score': pa.array([0.0] * len(run), pa.float32()),
                             'tier': pa.array([1] * len(run), pa.int8())}), tmp_path / 'runs/vi_k100.parquet')
    rk = [(q, k, d) for q in (1, 2) for k, d in enumerate(order[q], 1)]
    pq.write_table(pa.table({'query_id': [r[0] for r in rk], 'rank': [r[1] for r in rk], 'doc_id': [r[2] for r in rk],
                             'score': pa.array([0.0] * len(rk), pa.float32())}), tmp_path / 'cand.parquet')
    rows = [(d, c) for c, m in CLUSTERS.items() for d in m]
    pq.write_table(pa.table({'doc_id': [r[0] for r in rows], 'cluster_id': [r[1] for r in rows], 'boilerplate': [False] * len(rows)}),
                   tmp_path / 'H1/clusters.parquet')
    return tmp_path


def args(env, *extra):
    return ['--out-dir', str(env / 'H1'), '--runs-dir', str(env / 'runs'), '--doc-ranking', str(env / 'cand.parquet'),
            '--queries', str(env / 'query.parquet'), '--chunks-dir', str(env / 'chunks'), '--docs-dir', str(env / 'docs_vi'),
            '--k-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK), *extra]


def build(env, mode, name, *extra):
    out = env / name / 'sub.zip'
    assert V.main(['build', *args(env, '--mode', mode, '--out', str(out), *extra)]) == 0
    return out.with_suffix('.json'), json.loads(out.with_suffix('.stats.json').read_text(encoding='utf-8'))


def load(js):
    return json.loads(js.read_text(encoding='utf-8'))


@pytest.fixture
def d50(env):
    out = env / 'ms' / 'sub.zip'
    assert ms.main(['--k-doc', '100', '--k-doc-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK), '--chunk-mode', 'full', '--dedupe-scope', 'doc',
                    '--runs-dir', str(env / 'runs'), '--doc-ranking', str(env / 'cand.parquet'), '--chunks-dir', str(env / 'chunks'),
                    '--docs-dir', str(env / 'docs_vi'), '--index-dir', str(env / 'index'), '--queries', str(env / 'query.parquet'),
                    '--out', str(out)]) == 0
    return out.with_suffix('.json')


def test_identity_is_byte_identical_to_make_submission(env, d50):
    js, st = build(env, 'identity', 'ident')
    assert js.read_bytes() == d50.read_bytes()
    assert st['json_sha256'] == hashlib.sha256(d50.read_bytes()).hexdigest()


def test_dedup_walks_the_builder_order_and_refills(env, d50):
    js, st = build(env, 'dedup', 'dd')
    base, new = load(d50), load(js)
    prim = lambda r: [d for d in r['relevant_docs'] if d < 1000]
    q1b, q1, q2b, q2 = base[0], new[0], base[1], new[1]
    assert q1['id'] == 1 and prim(q1b) == list(range(10, 114)) and prim(q2b) == list(range(119, 15, -1))     # D50 order
    # q1: 12 (cluster 0), 50 (1), 60 (4: 14 is higher), 110 (2: 101 is higher) are dropped; docs 110..117 refill in ranking order
    assert prim(q1) == [d for d in range(10, 110) if d not in (12, 50, 60)] + list(range(111, 118))
    assert 1012 not in q1['relevant_docs'] and 101 in q1['relevant_docs']
    # q2 ranks descending: 20 (cluster 3: 119 first), 101 (2: 110 first) and 14 (4: 60 first, found in the pool) are dropped
    assert prim(q2) == [d for d in range(119, 19, -1) if d not in (101, 20)] + [19, 18, 17, 16, 15, 13]
    assert [c['doc_id'] for c in q1['relevant_chunks']] == [10, 11, 13] and [c['doc_id'] for c in q1b['relevant_chunks']] == [10, 11, 12]
    assert q2['relevant_chunks'] == q2b['relevant_chunks']                                                # top-3 of q2 had no duplicates
    assert len(prim(q1)) == len(prim(q2)) == K_TOTAL
    assert st['queries_changed'] == 2 and st['docs_dropped_total'] == 7 and st['queries_short_of_k_total'] == 0


def test_expand_appends_cluster_mates_and_leaves_chunks(env, d50):
    js, st = build(env, 'expand', 'ex')
    base, new = load(d50), load(js)
    for b, n in zip(base, new):
        assert n['relevant_chunks'] == b['relevant_chunks'] and n['id'] == b['id']
        assert n['relevant_docs'][:len(b['relevant_docs'])] == b['relevant_docs']
    added = [n['relevant_docs'][len(b['relevant_docs']):] for b, n in zip(base, new)]
    assert added == [[119], [14, 11]]                       # q1: mate of 20; q2: mate of 60 (rank 60) before mate of 50
    assert st['ids_added_total'] == 3 and st['queries_changed'] == 2


def test_diff_flags_nothing_for_expand_and_dedup(env, d50):
    ex, _ = build(env, 'expand', 'ex')
    dd, _ = build(env, 'dedup', 'dd')
    assert V.main(['diff', '--base', str(d50), '--variant', str(ex), '--mode', 'expand']) == 0
    assert V.main(['diff', '--base', str(d50), '--variant', str(dd), '--mode', 'dedup']) == 0
    assert V.main(['diff', '--base', str(d50), '--variant', str(dd), '--mode', 'expand']) == 1               # dedup changes chunks: out of scope for expand


def test_dedup_and_expand_helpers_are_pure():
    of = {1: 0, 2: 0, 4: 1, 6: 1, 7: 2, 8: 2}
    docs, dropped, short = V.dedup_lists({9: [1, 2, 3, 4]}, {9: [5, 6, 7, 8]}, of, k_total=4)
    assert docs == {9: [1, 3, 4, 5]} and dropped == {9: 1} and short == 0
    docs, dropped, short = V.dedup_lists({9: [1, 2, 3, 4]}, {9: [5, 6, 7, 8]}, of, k_total=6)
    assert docs == {9: [1, 3, 4, 5, 7]} and dropped == {9: 3} and short == 1
    ex = V.expand_lists({10: [1, 3]}, {1: 0, 5: 0, 9: 0, 3: 1}, {0: [1, 5, 9], 1: [3]}, {1: [1], 3: [3], 5: [5, 55], 9: [9]})
    assert ex == {10: [5, 55, 9]}
    assert V.expand_lists({10: [1]}, {}, {}, {1: [1]}) == {10: []}


def test_impact_counts(env):
    feats = {'doc_id': IDS, 'ck': [hashlib.sha1(str(10 if d in (10, 12) else d).encode()).digest() for d in IDS], 'n_ids': [len(GROUPS[d]) for d in IDS]}
    pq.write_table(pa.table(feats), env / 'H1/work/features.parquet')
    assert V.main(['impact', *args(env)]) == 0
    res = json.loads((env / 'H1/impact_d50.json').read_text(encoding='utf-8'))['content']
    w = res['a_slots_in_150_taken_by_same_cluster_docs']                 # q1: 12, 50, 60, 110; q2: 20, 101
    assert (w['max'], w['min'], w['mean']) == (4, 2, 3) and res['a_total_wasted_slots'] == 6 and res['a_queries_with_at_least_one_wasted_slot'] == 2
    assert res['b_full_chunks_top50_identical_text']['max'] == 1 and res['b_queries_with_identical_chunk_text'] == 1     # k_chunk 3: docs 10, 12
    assert res['b_full_chunks_top50_same_cluster_as_earlier']['max'] == 1
    m = res['c_cluster_mates_outside_top150_docs']                       # q1: [119]; q2: [14, 11]
    assert (m['max'], m['min'], res['c_total_mates']) == (2, 1, 3)
    assert (env / 'H1/impact_d50_per_query.csv').exists()
