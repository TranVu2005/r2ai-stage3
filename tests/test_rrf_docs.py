"""RRF choice of relevant_docs over the tier-1 = 200 rerank pool (RD150 / RD150c): docs = top n by RRF + H1-expand mates,
chunks either kept byte for byte from the base or the full chunks of the top k_chunk RRF docs."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.submit import rrf_docs as D
from tests.test_best_expand import _h1, env  # noqa: F401  (fixture)
from tests.test_best_rrf import _rr


def _run(env, base, out, chunks, n_docs=4, extra=()):
    rc = D.main(['build', '--base', str(base), '--out', str(out), '--chunks', chunks, '--runs-dir', str(env / 'runs'),
                 '--doc-ranking', str(env / 'cand.parquet'), '--rerank-run', str(_rr(env)), '--queries', str(env / 'query.parquet'),
                 '--chunks-dir', str(env / 'chunks'), '--docs-dir', str(env / 'docs_vi'), '--clusters', str(env / 'H1/clusters.parquet'),
                 '--k-cache', '2', '--k-total', '4', '--k-docs', str(n_docs), '--k-chunk', '3', *extra])
    js = out.with_suffix('.json')
    return rc, json.loads(js.read_text(encoding='utf-8')) if rc == 0 else None, \
        json.loads(out.with_suffix('.stats.json').read_text(encoding='utf-8')) if rc == 0 else None


def test_rrf_docs_order_pool_by_fused_rank():
    # q1 of the fixture: reranker 60, 61, 62, 119, 118, ... 10 (rank 110); hybrid 10, 11, ... 119
    order = [60, 61, 62] + [d for d in range(119, 9, -1) if d not in (60, 61, 62)]
    rr = {d: k for k, d in enumerate(order, 1)}
    hy = {d: k for k, d in enumerate(range(10, 120), 1)}
    assert D.rrf_docs(order, rr, hy, 5) == [60, 61, 62, 10, 11]
    with pytest.raises(ValueError, match='without a hybrid rank'):
        D.rrf_docs(order, rr, {d: r for d, r in hy.items() if d != 60}, 5)


def test_rd150_keeps_base_chunks_and_counts_chunks_outside_docs(env):
    _, base = _h1(env, 'expand')                                            # base chunks: q1 10, 11, 12; q2 119, 118, 117
    rc, rows, st = _run(env, base, env / 'rd' / 'sub.zip', 'base')
    assert rc == 0
    b = json.loads(base.read_text(encoding='utf-8'))
    assert [r['id'] for r in rows] == [r['id'] for r in b] and all(list(r) == list(x) for r, x in zip(rows, b))
    assert [r['relevant_chunks'] for r in rows] == [r['relevant_chunks'] for r in b]
    # q1 docs 60, 61, 62, 10 (+ groups: 60 -> 1060) + mates 14 (of 60), 12 / 1012 (of 10); q2 119..116 (+1117) + mate 20
    assert rows[0]['relevant_docs'] == [60, 1060, 61, 62, 10, 14, 12, 1012]
    assert rows[1]['relevant_docs'] == [119, 118, 117, 1117, 116, 20]
    assert st['pool_docs_without_hybrid_rank'] == 0
    assert st['chunk_docs_outside_relevant_docs'] == {'total': 1, 'queries': 1}  # q1 chunk 11; 12 is back via expand
    assert st['docs_new_vs_d50']['total'] == 3 and st['docs_dropped_vs_d50']['total'] == 3   # D50 q1 = 10, 11, 12, 13
    assert st['docs_same_as_r150']['total'] == 4 and st['docs_not_in_r150']['total'] == 4     # R: 60, 61, 62, 119
    assert st['new_docs_reranker_rank']['max'] == 3 and st['new_docs_hybrid_rank']['max'] == 53


def test_rd150c_chunks_are_top_rrf_full_chunks(env):
    _, base = _h1(env, 'expand')
    rc, rows, st = _run(env, base, env / 'rdc' / 'sub.zip', 'rrf')
    assert rc == 0
    rc2, rows_d, _ = _run(env, base, env / 'rd' / 'sub.zip', 'base')
    assert [r['relevant_docs'] for r in rows] == [r['relevant_docs'] for r in rows_d]
    assert [[c['doc_id'] for c in r['relevant_chunks']] for r in rows] == [[60, 61, 62], [119, 118, 117]]
    assert rows[0]['relevant_chunks'][0]['chunk_text'].startswith('d60w0 ')
    assert rows[1]['relevant_chunks'] == json.loads(base.read_text(encoding='utf-8'))[1]['relevant_chunks']
    assert st['chunk_docs_changed_vs_base']['total'] == 3 and st['chunk_docs_outside_relevant_docs']['total'] == 0
    assert st['docs_missing_text'] == 0
