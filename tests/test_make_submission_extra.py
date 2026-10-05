"""--extra-chunk-docs / --extra-zip-budget-bytes of make_submission on a small fixture (whitespace tokenizer)."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.eval.scorer import whitespace_tokenizer
import r2ai.submit.make_submission as ms

N_DOCS = 7


def _body(d: int) -> str:
    return '\n'.join(' '.join(f'd{d}p{p}w{w}' for w in range(30)) + '.' for p in range(3))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, 'Tokenizer', whitespace_tokenizer)
    runs, chunks, docs, index = (tmp_path / x for x in ('runs', 'chunks', 'docs_vi', 'index'))
    for p in (runs, chunks, docs, index):
        p.mkdir()
    qids = [1, 2]
    pq.write_table(pa.table({'id': qids, 'query': ['q one', 'q two']}), tmp_path / 'query.parquet')
    ids = list(range(10, 10 + N_DOCS))
    groups = {d: [d] + ([d + 100] if d % 3 == 0 else []) for d in ids}
    pq.write_table(pa.table({'doc_ids': [groups[d] for d in ids], 'title': ['t'] * N_DOCS, 'question': [None] * N_DOCS,
                             'description': [None] * N_DOCS, 'answer': [None] * N_DOCS, 'body': [_body(d) for d in ids],
                             'status': ['ok'] * N_DOCS}), docs / 'part0.parquet')
    pq.write_table(pa.table({'doc_id': ids, 'doc_ids_group': [groups[d] for d in ids]}), chunks / 'docs.parquet')
    rows = {'chunk_id': [], 'doc_id': [], 'field': [], 'text': []}
    for d in ids:                                          # 3 body chunks per doc = its 3 paragraphs
        for para in _body(d).split('\n'):
            for k, v in zip(rows, (len(rows['chunk_id']), d, 'body', para)):
                rows[k].append(v)
    pq.write_table(pa.table(rows), chunks / 'chunks_t256.parquet')
    order = {1: ids, 2: ids[::-1]}
    run = [(q, k, d) for q in qids for k, d in enumerate(order[q], 1)]
    pq.write_table(pa.table({'query_id': [r[0] for r in run], 'rank': pa.array([r[1] for r in run], pa.int32()),
                             'doc_id': [r[2] for r in run], 'score': pa.array([0.0] * len(run), pa.float32()),
                             'tier': pa.array([1] * len(run), pa.int8())}), runs / 'vi_k100.parquet')
    cs = {'query_id': [], 'doc_id': [], 'chunk_id': [], 'score': []}
    for q in qids:
        for c, d in zip(rows['chunk_id'], rows['doc_id']):
            # best chunk: 2nd paragraph for query 1; query 2 ties the 1st and 3rd (lower chunk_id wins)
            sc = (1.0 if c % 3 == 1 else 0.0) if q == 1 else (1.0 if c % 3 != 1 else 0.0)
            for k, v in zip(cs, (q, d, c, sc)):
                cs[k].append(v)
    pq.write_table(pa.table({'query_id': cs['query_id'], 'doc_id': cs['doc_id'], 'chunk_id': cs['chunk_id'],
                             'score': pa.array(cs['score'], pa.float32())}), runs / 'vi_k100_chunk_scores.parquet')
    common = ['--k-doc', str(N_DOCS), '--k-chunk', '2', '--chunk-mode', 'full', '--runs-dir', str(runs),
              '--chunks-dir', str(chunks), '--docs-dir', str(docs), '--index-dir', str(index),
              '--queries', str(tmp_path / 'query.parquet')]

    def build(name, *extra):
        out = tmp_path / 'out' / name / 'sub.zip'                    # same entry name: zip size comparable
        assert ms.main(common + ['--out', str(out), *extra]) == 0
        return (json.loads(out.with_suffix('.json').read_text(encoding='utf-8')),
                json.loads(out.with_suffix('.stats.json').read_text(encoding='utf-8')), out)
    build.dirs = {'tmp': tmp_path, 'runs': runs, 'chunks': chunks, 'groups': groups}
    return build, order, rows


def test_extra_chunks_appended_verbatim_best_scored(env):
    build, order, rows = env
    base, bst, _ = build('base')
    sub, st, _ = build('x5', '--extra-chunk-docs', '5')
    assert 'extra_chunk_docs' not in bst                              # default output/stats unchanged
    text = dict(zip(rows['chunk_id'], rows['text']))
    for b, s in zip(base, sub):
        assert s['id'] == b['id'] and s['relevant_docs'] == b['relevant_docs']
        assert s['relevant_chunks'][:2] == b['relevant_chunks']
        ext = s['relevant_chunks'][2:]
        assert [c['doc_id'] for c in ext] == order[b['id']][2:5]
        for c in ext:
            first = 3 * (c['doc_id'] - 10)                            # chunk ids of the doc: first, first+1, first+2
            assert c['chunk_text'] == text[first + 1 if b['id'] == 1 else first]
            assert c['chunk_text'] in _body(c['doc_id'])
    assert st['extra_chunk_docs'] == 5 and st['extra_chunks'] == 6 and st['extra_docs_without_score'] == 0


def test_extra_zip_budget_picks_largest_fitting_n(env):
    build, _, _ = env
    _, _, z4 = build('x4', '--extra-chunk-docs', '4')
    budget = z4.stat().st_size
    sub, st, z = build('budget', '--extra-chunk-docs', str(N_DOCS), '--extra-zip-budget-bytes', str(budget))
    assert st['extra_chunk_docs'] == 4 and z.stat().st_size <= budget
    assert all(len(r['relevant_chunks']) == 4 for r in sub)
    assert all(t['zip_bytes'] > budget for t in st['tried'] if t.get('extra_chunk_docs', 0) > 4)


@pytest.mark.parametrize('args', [['--extra-chunk-docs', '51'], ['--extra-chunk-docs', '1'],
                                  ['--extra-zip-budget-bytes', '100'],
                                  ['--extra-chunk-docs', '5', '--max-zip-mib', '45.7']])
def test_extra_flag_errors(env, args):
    build, _, _ = env
    with pytest.raises(SystemExit):
        build('bad', *args)


def test_k_chunk_budget_adds_full_chunks_in_rank_order(env):
    build, order, _ = env
    base, bst, _ = build('base')
    _, _, z4 = build('kc4', '--k-chunk', '4')
    sub, st, z = build('budget', '--k-chunk-zip-budget-bytes', str(z4.stat().st_size))
    assert st['k_chunk'] == 4 and z.stat().st_size <= z4.stat().st_size
    assert all(t['zip_bytes'] > z4.stat().st_size for t in st['tried'] if t['k_chunk'] > 4)
    assert 'k_doc_total' not in bst and 'extra_chunk_docs' not in st
    for b, s in zip(base, sub):
        assert s['relevant_docs'] == b['relevant_docs'] and s['relevant_chunks'][:2] == b['relevant_chunks']
        assert [c['doc_id'] for c in s['relevant_chunks']] == order[b['id']][:4]
        assert all(c['chunk_text'] == _body(c['doc_id']) for c in s['relevant_chunks'])     # whole body, verbatim


def test_doc_ranking_appends_docs_after_k100(env):
    build, _, _ = env
    dd = build.dirs
    runs2 = dd['tmp'] / 'runs2'
    runs2.mkdir()
    groups = dict(dd['groups']) | {d: [d] for d in range(1000, 1092)} | {d: [d] for d in range(2000, 2011)} | {2001: [2001, 2101]}
    pq.write_table(pa.table({'doc_id': list(groups), 'doc_ids_group': list(groups.values())}), dd['chunks'] / 'docs.parquet')
    q1 = list(range(10, 10 + N_DOCS)) + list(range(1000, 1092))                     # 99 cached docs
    q2 = list(range(10, 10 + N_DOCS))[::-1]
    run = [(1, k, d) for k, d in enumerate(q1, 1)] + [(2, k, d) for k, d in enumerate(q2, 1)]
    pq.write_table(pa.table({'query_id': [r[0] for r in run], 'rank': pa.array([r[1] for r in run], pa.int32()),
                             'doc_id': [r[2] for r in run], 'score': pa.array([0.0] * len(run), pa.float32()),
                             'tier': pa.array([1] * len(run), pa.int8())}), runs2 / 'vi_k100.parquet')
    rk = [(1, 1000), (1, 2000), (1, 10), (1, 2001), (1, 2002), (2, 2005), (2, 12), (2, 2006)]
    pq.write_table(pa.table({'query_id': [q for q, _ in rk], 'rank': [1, 2, 3, 4, 5, 1, 2, 3],
                             'doc_id': [d for _, d in rk]}), dd['tmp'] / 'rank.parquet')
    common = ['--k-doc', '100', '--runs-dir', str(runs2)]
    base, _, _ = build('b100', *common)
    sub, st, _ = build('k101', *common, '--doc-ranking', str(dd['tmp'] / 'rank.parquet'), '--k-doc-total', '101')
    extra = {1: [2000, 2001, 2101], 2: [2005, 2006]}
    for b, s in zip(base, sub):
        assert s['relevant_chunks'] == b['relevant_chunks']
        assert s['relevant_docs'] == b['relevant_docs'] + extra[b['id']] and len(set(s['relevant_docs'])) == len(s['relevant_docs'])
    assert st['docs_from_ranking'] == 4 and st['queries_short_of_k_doc_total'] == 1
    for bad in (['--doc-ranking', 'x.parquet'], ['--k-doc-total', '150'], ['--doc-ranking', 'x', '--k-doc-total', '100'],
                ['--k-doc', '50', '--doc-ranking', 'x', '--k-doc-total', '150']):
        with pytest.raises(SystemExit):
            build('bad', '--runs-dir', str(runs2), *(['--k-doc', '100'] if '--k-doc' not in bad else []), *bad)
