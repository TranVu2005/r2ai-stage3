"""Equivalence gates for run_retrieval_k100 --tier1-docs (deep rerank) against an existing K100 cache.

    python -m r2ai.retrieve.deep_gate deep  --old out/runs/vi-k100 --new <deep run dir> [--tag _sample] --out gate_deep.json
    python -m r2ai.retrieve.deep_gate same  --old out/runs/vi-k100 --new <tier1-docs-50 run dir> --tag _sample --out gate_same.json

deep: for the queries of the new run, the old tier-1 docs (tier 1 of the old cache) must keep their reranker score
      (|delta| <= --tol for >= --min-score-share of the (query, doc) pairs) and their relative order (identical in
      >= --min-order-share of the queries). Chunk-level: old vi_k100_chunk_scores rows of those docs vs the new
      vi_k100_pairs rows (same query, doc, chunk), |delta| reported. Deviating queries/pairs are listed.
      --pairs (off by default): also old vi_k100_pairs rows of tier 1 vs the new vi_k100_pairs rows (same query, doc,
      chunk); the gate then also needs every old row present and |delta| <= --tol on all of them.
same: the new run (default tiers) must give the same (rank, doc_id, tier) rows as the old cache for its queries, and the
      same chunk-score rows; score deltas are reported.
Exit 0 when the gate passes, 1 otherwise.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, resolve_path

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _read(d: Path, name: str):
    return pq.read_table(d / name).to_pandas()


def _dist(x) -> dict | None:
    x = np.abs(np.asarray(x, dtype=float))
    return {'n': int(len(x)), 'max': float(x.max()), 'p99': float(np.percentile(x, 99)), 'mean': float(x.mean())} if len(x) else None


def gate_deep(old_dir: Path, new_dir: Path, tag: str, tol: float, min_score: float, min_order: float,
              pairs: bool = False) -> dict:
    old, new = _read(old_dir, 'vi_k100.parquet'), _read(new_dir, f'vi_k100{tag}.parquet')
    qs = sorted(new['query_id'].unique())
    old = old[old.query_id.isin(qs)]
    deltas, bad_pairs, bad_order, missing = [], [], [], []
    for q in qs:
        o = old[(old.query_id == q) & (old.tier == 1)].sort_values('rank')
        n = new[new.query_id == q].sort_values('rank')
        n1 = n[n.tier == 1]
        sc = dict(zip(n1['doc_id'], n1['score']))
        for d, s in zip(o['doc_id'], o['score']):
            if d not in sc:
                missing.append([int(q), int(d)])
                continue
            dl = float(sc[d]) - float(s)
            deltas.append(dl)
            if abs(dl) > tol:
                bad_pairs.append([int(q), int(d), float(s), float(sc[d])])
        keep = set(o['doc_id'])
        if [d for d in n1['doc_id'] if d in keep] != o['doc_id'].tolist():
            bad_order.append(int(q))
    n_pairs = len(deltas) + len(missing)
    score_share = (len(deltas) - len(bad_pairs)) / n_pairs if n_pairs else 0.0
    order_share = (len(qs) - len(bad_order)) / len(qs) if qs else 0.0
    res = {'queries': len(qs), 'old_tier1_pairs': n_pairs, 'missing_from_new_tier1': missing[:50], 'n_missing': len(missing),
           'tol': tol, 'pairs_within_tol': len(deltas) - len(bad_pairs), 'pairs_within_tol_share': score_share,
           'doc_score_abs_delta': _dist(deltas), 'pairs_over_tol': bad_pairs[:50], 'n_pairs_over_tol': len(bad_pairs),
           'queries_same_relative_order': len(qs) - len(bad_order), 'same_order_share': order_share,
           'queries_order_changed': bad_order}
    pf = new_dir / f'vi_k100_pairs{tag}.parquet'
    if pf.exists():
        oc = _read(old_dir, 'vi_k100_chunk_scores.parquet')
        oc = oc[oc.query_id.isin(qs)].merge(old[old.tier == 1][['query_id', 'doc_id']], on=['query_id', 'doc_id'])
        m = oc.merge(pq.read_table(pf).to_pandas(), on=['query_id', 'doc_id', 'chunk_id'], suffixes=('_old', '_new'))
        dl = (m['score_new'] - m['score_old']).to_numpy()
        res['chunk_level'] = {'common_rows': int(len(m)), 'abs_delta': _dist(dl), 'rows_over_tol': int((np.abs(dl) > tol).sum())}
    res['pass'] = bool(score_share >= min_score and order_share >= min_order)
    if pairs:
        op = _read(old_dir, 'vi_k100_pairs.parquet')
        op = op[op.query_id.isin(qs) & (op.tier == 1)]
        m = op.merge(_read(new_dir, f'vi_k100_pairs{tag}.parquet'), on=['query_id', 'doc_id', 'chunk_id'], how='left',
                     suffixes=('_old', '_new'))
        miss = m['score_new'].isna().to_numpy()
        dl = (m['score_new'] - m['score_old']).to_numpy()[~miss]
        over = int((np.abs(dl) > tol).sum())
        res['pair_level'] = {'old_tier1_pairs': int(len(op)), 'missing': int(miss.sum()), 'abs_delta': _dist(dl),
                             'rows_over_tol': over}
        res['pass'] = bool(res['pass'] and not miss.any() and over == 0)
    return res


def gate_same(old_dir: Path, new_dir: Path, tag: str, tol: float) -> dict:
    new = _read(new_dir, f'vi_k100{tag}.parquet')
    qs = sorted(new['query_id'].unique())
    old = _read(old_dir, 'vi_k100.parquet')
    old = old[old.query_id.isin(qs)]
    key = ['query_id', 'rank', 'doc_id', 'tier']
    a = old.sort_values(['query_id', 'rank'])[key].reset_index(drop=True)
    b = new.sort_values(['query_id', 'rank'])[key].reset_index(drop=True)
    same_rows = a.equals(b)
    m = old.merge(new, on=key, suffixes=('_old', '_new'))
    dl = (m['score_new'] - m['score_old']).to_numpy()
    oc = _read(old_dir, 'vi_k100_chunk_scores.parquet')
    nc = _read(new_dir, f'vi_k100_chunk_scores{tag}.parquet')
    oc = oc[oc.query_id.isin(qs)]
    ck = ['query_id', 'doc_id', 'chunk_id']
    same_chunk_rows = oc[ck].sort_values(ck).reset_index(drop=True).equals(nc[ck].sort_values(ck).reset_index(drop=True))
    mc = oc.merge(nc, on=ck, suffixes=('_old', '_new'))
    dc = (mc['score_new'] - mc['score_old']).to_numpy()
    per_q = [int(q) for q in qs if not a[a.query_id == q].reset_index(drop=True).equals(b[b.query_id == q].reset_index(drop=True))]
    return {'queries': len(qs), 'query_ids': [int(q) for q in qs], 'rows_old': int(len(a)), 'rows_new': int(len(b)),
            'rank_doc_tier_identical': bool(same_rows), 'queries_differing': per_q, 'doc_score_abs_delta': _dist(dl),
            'chunk_rows_identical': bool(same_chunk_rows), 'chunk_score_abs_delta': _dist(dc),
            'chunk_rows_over_tol': int((np.abs(dc) > tol).sum()), 'tol': tol,
            'pass': bool(same_rows and same_chunk_rows and (len(dl) == 0 or np.abs(dl).max() <= tol))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('deep', 'same'))
    ap.add_argument('--old', required=True, help='existing K100 cache dir (vi_k100.parquet, vi_k100_chunk_scores.parquet)')
    ap.add_argument('--new', required=True, help='run dir of run_retrieval_k100')
    ap.add_argument('--tag', default='', help="file suffix of the new run ('', _sample, _debug)")
    ap.add_argument('--out', required=True, help='result json')
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--min-score-share', type=float, default=0.999)
    ap.add_argument('--min-order-share', type=float, default=0.99)
    ap.add_argument('--pairs', action='store_true', help='deep: also gate every old tier-1 pair (vi_k100_pairs) score')
    a = ap.parse_args(argv)
    out = assert_writable(a.out)
    old, new = resolve_path(a.old), resolve_path(a.new)
    res = gate_deep(old, new, a.tag, a.tol, a.min_score_share, a.min_order_share, a.pairs) if a.mode == 'deep' else \
        gate_same(old, new, a.tag, a.tol)
    res = {'mode': a.mode, 'old': str(old), 'new': str(new), 'tag': a.tag} | res
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return 0 if res['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
