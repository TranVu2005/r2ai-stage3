"""CPU report: what happens to the per-query pools if the 239,771 en PubMed docs are ranked together with vi + zh (+ the new 120ask docs).

Reference ranking per query (no en) = the cached S37 reference (S26 canonical doc vectors + Pharmacity, top-400 cosines) merged with the
doc-level vectors of every S37 120ask segment. With en = the same list plus every en doc. Doc-level BGE-M3 cosines only; nothing is scored.
pool13 size of a query = its S29 pool13 size (`size13`, 1.3 x K_q, 110-339); K_q = round(size13 / 1.3). top-300 = the first 300 of the ranking.
Reported: how many en docs fall into pool13 and into top-300 per query (mean, P50, P90, max, share of queries with >= 1), and the cosine of the
best en doc minus the cosine of the rank-K_q doc of the ranking without en.
"""
from __future__ import annotations

import numpy as np

from r2ai.s37.ingest import STORE, build_pool, load_queries, top_scores
from r2ai.s37.en import SEG

KEEP = 400


def q(a: np.ndarray) -> dict:
    return {'mean': round(float(a.mean()), 3), 'p50': float(np.percentile(a, 50)), 'p90': float(np.percentile(a, 90)), 'max': float(a.max())}


def report() -> dict:
    pool = build_pool()
    q_ids, qv = load_queries()
    assert (pool['q_ids'] == q_ids).all()
    en = np.load(SEG / 'vectors.npz')
    es, _ = top_scores(qv, en['vectors'], KEEP)
    segs = [np.load(p / 'vectors.npz')['dense'] for p in sorted(STORE.glob('seg*')) if (p / 'vectors.npz').exists()]
    new = np.concatenate(segs)
    ns, _ = top_scores(qv, new, KEEP)
    base = np.concatenate([pool['scores'], ns], axis=1)
    base = -np.sort(-base, axis=1)[:, :KEEP]                      # vi + zh + 120ask (reference, no en)
    size13 = pool['size13'].astype(int)
    kq = np.rint(size13 / 1.3).astype(int)
    allsc = np.concatenate([base, es], axis=1)
    is_en = np.concatenate([np.zeros_like(base, bool), np.ones_like(es, bool)], axis=1)
    order = np.argsort(-allsc, axis=1, kind='stable')
    en_sorted = np.take_along_axis(is_en, order, 1)
    n13 = np.array([en_sorted[i, :size13[i]].sum() for i in range(len(q_ids))])
    n300 = en_sorted[:, :300].sum(axis=1)
    n211 = en_sorted[:, :211].sum(axis=1)
    ref_kq = base[np.arange(len(q_ids)), kq - 1]
    delta = es[:, 0] - ref_kq
    ref13 = base[np.arange(len(q_ids)), size13 - 1]
    return {
        'queries': len(q_ids), 'en_docs': int(len(en['doc_ids'])), 'reference_docs_new120ask': int(len(new)),
        'pool13_size_p10_p50_p90': [int(np.percentile(size13, 10)), int(np.percentile(size13, 50)), int(np.percentile(size13, 90))],
        'en_in_pool13': {**q(n13), 'share_queries_ge1': round(float((n13 >= 1).mean()), 4), 'total': int(n13.sum())},
        'en_in_top211': {**q(n211), 'share_queries_ge1': round(float((n211 >= 1).mean()), 4)},
        'en_in_top300': {**q(n300), 'share_queries_ge1': round(float((n300 >= 1).mean()), 4), 'total': int(n300.sum())},
        'best_en_cosine': q(es[:, 0]), 'cosine_at_rank_Kq_without_en': q(ref_kq), 'cosine_at_pool13_end_without_en': q(ref13),
        'best_en_minus_cosine_at_Kq': {**q(delta), 'share_best_en_above_Kq_doc': round(float((delta > 0).mean()), 4)},
        'best_en_above_pool13_cut_share': round(float((es[:, 0] > ref13).mean()), 4),
        'definition': 'ranking = reference (vi+zh+Pharmacity top-400 + new 120ask doc vectors) + en; K_q = round(size13/1.3); pool13 = first size13 of the ranking',
    }
