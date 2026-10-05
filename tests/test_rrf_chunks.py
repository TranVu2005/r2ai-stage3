"""RRF choice of the full-chunk docs: relevant_docs of the base kept byte for byte, only relevant_chunks change."""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.submit import rrf_chunks as R
from tests.test_best_expand import IDS, K_CHUNK, K_TOTAL, _h1, env  # noqa: F401  (fixture)


def test_rrf_choice_orders_by_fused_rank_and_breaks_ties():
    docs = [1, 2, 3, 4]
    rr = {1: 1, 2: 4, 3: 2, 4: 3}
    hy = {1: 4, 2: 1, 3: 2, 4: 3}
    assert R.rrf_choice(docs, rr, hy, 4) == [3, 1, 2, 4]           # 3: (2,2) best; 1 (1,4) ties 2 (4,1) -> lower rr; 4 (3,3) last
    assert R.rrf_choice(docs, {}, hy, 2, missing=201) == [2, 3]    # no reranker rank: hybrid decides


def test_rrf_build_keeps_docs_and_changes_chunks(env):
    _, base = _h1(env, 'expand')
    order = [60, 61, 62] + [d for d in IDS[::-1] if d not in (60, 61, 62)]                        # same reranker order for both queries
    rr = [(q, k, d) for q in (1, 2) for k, d in enumerate(order, 1)]
    pq.write_table(pa.table({'query_id': [r[0] for r in rr], 'rank': [r[1] for r in rr], 'doc_id': [r[2] for r in rr]}),
                   env / 'rr.parquet')
    out = env / 'rrf' / 'sub.zip'
    common = ['--runs-dir', str(env / 'runs'), '--doc-ranking', str(env / 'cand.parquet'), '--rerank-run', str(env / 'rr.parquet'),
              '--queries', str(env / 'query.parquet'), '--k-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK)]
    assert R.main(['build', *common, '--base', str(base), '--out', str(out), '--docs-dir', str(env / 'docs_vi')]) == 0
    b, n = json.loads(base.read_text(encoding='utf-8')), json.loads(out.with_suffix('.json').read_text(encoding='utf-8'))
    for x, y in zip(b, n):
        assert (x['id'], x['relevant_docs']) == (y['id'], y['relevant_docs'])
    # q1 (hybrid 10, 11, 12, ...): 60/61/62 (reranker 1-3, hybrid 51-53) beat 10/11/12 (reranker 110/109/108)
    # q2 (hybrid 119, 118, ...): 119/118/117 are reranker 4-6 and hybrid 1-3 -> unchanged
    assert [c['doc_id'] for c in n[0]['relevant_chunks']] == [60, 61, 62]
    assert n[1]['relevant_chunks'] == b[1]['relevant_chunks']
    st = json.loads(out.with_suffix('.stats.json').read_text(encoding='utf-8'))
    assert st['chunk_docs_changed_vs_base']['queries_with_any'] == 1 and st['new_chunk_docs_total'] == 3
    assert st['new_chunk_docs_reranker_rank_gt_50'] == 0 and st['kept_docs_same_chunk_text'] == 3
    assert all(len(r['relevant_chunks']) == K_CHUNK for r in n)
    assert R.main(['build', *common, '--base', str(base), '--out', str(env / 'x/sub.zip'), '--docs-dir', str(env / 'docs_vi'),
                   '--min-mean', '99']) == 2
