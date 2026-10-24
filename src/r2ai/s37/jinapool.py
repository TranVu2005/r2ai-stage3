"""Jina Reranker v3.5 pools for the new 120ask pairs (S37), scored with the S35 code (`r2ai.s35.jina`, b10s0, run from the S35 worktree).

    python -m r2ai.s37.jinapool build      # out/runs/s37/jina/pool.parquet (rescore set) + out/runs/s37/jina-check/pool.parquet (5 control queries)
    python -m r2ai.s37.jinapool compare    # control queries: new scores vs the S35 scores of the same pool (reproducibility gate)

Jina is listwise: a doc's score depends on the 9 docs sharing its block and on the shuffle, so a query that gains new docs is
rescored over its WHOLE pool (S35 pool + new S37 pairs). Queries without new pairs keep their S35 scores. The old S35 outputs are never written.
Row order inside a query decides the seeded block split: control queries keep the S35 order (sorted by query_id, cosine_rank);
rescored queries are sorted by cosine_score descending, doc_id ascending.
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, assert_writable

import argparse
import json

import numpy as np
import pandas as pd

from r2ai.s37.ingest import RUN, atomic_json, sha256_file

S35 = RUNS_DIR / 's35-part-jina' / 'jina'
OUT = RUN / 'jina'
CHECK = RUN / 'jina-check'
CONTROL_QUERIES = 5
POOL_COLUMNS = ['query_id', 'doc_id', 'cosine_rank', 'cosine_score', 'p4b', 'in_pool211', 'in_pool13', 'lang', 'query_text', 'doc_text', 'doc_tokens']


def new_pairs() -> pd.DataFrame:
    """Every S37 pair (pairs_new.parquet) with the 512-token doc text exported for the 4B subsets."""
    pairs = pd.read_parquet(RUN / 'pairs_new.parquet')
    shards = sorted((RUN / 'kaggle-4b').glob('*/shards/pool-93001.parquet'))
    texts = pd.concat([pd.read_parquet(s, columns=['pair_id', 'lang', 'query_text', 'doc_text', 'doc_tokens']) for s in shards], ignore_index=True)
    texts = texts.drop_duplicates('pair_id')
    merged = pairs.drop(columns=['lang']).merge(texts, on='pair_id', how='left', validate='one_to_one')
    if merged.doc_text.isna().any():
        raise SystemExit(f'{int(merged.doc_text.isna().sum())} S37 pairs have no exported doc text (run export kaggle4b for their batch first)')
    return merged


def build() -> dict:
    old = pd.read_parquet(S35 / 'pool.parquet')
    new = new_pairs()
    affected = np.array(sorted(new.query_id.unique()))
    unaffected = np.array(sorted(set(old.query_id.unique()) - set(affected)))
    control = unaffected[np.linspace(0, len(unaffected) - 1, CONTROL_QUERIES).round().astype(int)]
    add = pd.DataFrame({
        'query_id': new.query_id, 'doc_id': new.doc_id, 'cosine_rank': 20_000 + new.cosine_rank, 'cosine_score': new.cosine_score,
        'p4b': np.nan, 'in_pool211': new.in_top211, 'in_pool13': new.in_pool13, 'lang': new.lang, 'query_text': new.query_text,
        'doc_text': new.doc_text, 'doc_tokens': new.doc_tokens})
    base = old[old.query_id.isin(affected)]
    keep = pd.concat([base, add], ignore_index=True).drop_duplicates(['query_id', 'doc_id'], keep='first')
    keep['source'] = np.where(keep.p4b.isna(), 's37', 's35')
    pool = keep.sort_values(['query_id', 'cosine_score', 'doc_id'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
    ctrl = old[old.query_id.isin(control)].reset_index(drop=True)
    ctrl['source'] = 's35'
    assert_writable(OUT)
    assert_writable(CHECK)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT / 'pool.parquet', index=False)
    ctrl.to_parquet(CHECK / 'pool.parquet', index=False)
    per_q = pool.groupby('query_id').size()
    man = {'rescored_queries': int(len(affected)), 'unaffected_queries_keep_s35_scores': int(len(unaffected)), 'control_queries': [int(x) for x in control],
           'pairs': len(pool), 'pairs_s35': int((pool.source == 's35').sum()), 'pairs_s37_new': int((pool.source == 's37').sum()),
           'pairs_per_query': {'mean': float(per_q.mean()), 'min': int(per_q.min()), 'max': int(per_q.max())},
           'doc_tokens_mean': float(pool.doc_tokens.mean()), 'control_pairs': len(ctrl), 'layout': 'b10s0',
           'order_rule': 'rescored: cosine_score desc, doc_id asc; control: S35 order',
           's35_pool_sha256': sha256_file(S35 / 'pool.parquet'), 'pool_sha256': sha256_file(OUT / 'pool.parquet'),
           'check_pool_sha256': sha256_file(CHECK / 'pool.parquet')}
    atomic_json(OUT / 'pool_manifest.json', man)
    atomic_json(CHECK / 'pool_manifest.json', {**man, 'pairs': len(ctrl), 'pairs_per_query': {'mean': float(ctrl.groupby('query_id').size().mean()),
                                                                                              'min': 0, 'max': 0}})
    return man


def compare(tol: float = 2e-3) -> dict:
    """Control queries: new b10s0 scores vs the S35 b10s0 parts for the same (query, doc) rows."""
    new = pd.concat([pd.read_parquet(f) for f in sorted((CHECK / 'parts-b10s0').glob('part-*.parquet'))], ignore_index=True)
    old = pd.concat([pd.read_parquet(f) for f in sorted((S35 / 'parts-b10s0').glob('part-*.parquet'))], ignore_index=True)
    j = new.merge(old, on=['query_id', 'doc_id'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(j) != len(new):
        raise SystemExit(f'{len(new) - len(j)} control rows missing in the S35 parts')
    d = (j.score_new - j.score_old).abs()
    from scipy.stats import spearmanr
    rho = j.groupby('query_id').apply(lambda g: spearmanr(g.score_new, g.score_old).statistic, include_groups=False)
    res = {'rows': len(j), 'queries': int(j.query_id.nunique()), 'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()),
           'spearman_min': float(rho.min()), 'tolerance': tol, 'passed': bool(d.max() <= tol and rho.min() >= 0.9999)}
    atomic_json(CHECK / 'compare.json', res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('build', 'compare'))
    a = ap.parse_args(argv)
    print(json.dumps(build() if a.command == 'build' else compare(), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
