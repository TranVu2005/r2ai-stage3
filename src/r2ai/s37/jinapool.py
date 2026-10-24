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
    cols = ['pair_id', 'lang', 'query_text', 'doc_text', 'doc_tokens']
    shards = sorted((RUN / 'kaggle-4b').glob('*/shards/pool-93001.parquet'))
    frames = [pd.read_parquet(s, columns=cols) for s in shards]
    have = set(pd.concat(frames).pair_id) if frames else set()
    missing = sorted(set(pairs.pair_id) - have)
    if missing:                                    # pairs that never went into a 4B subset (e.g. the last crawl): same text, no 4B export
        frames.append(texts_from_segments(missing)[cols])
    texts = pd.concat(frames, ignore_index=True).drop_duplicates('pair_id')
    merged = pairs.drop(columns=['lang']).merge(texts, on='pair_id', how='left', validate='one_to_one')
    if merged.doc_text.isna().any():
        raise SystemExit(f'{int(merged.doc_text.isna().sum())} S37 pairs have no doc text')
    return merged


def texts_from_segments(pair_ids: list[int]) -> pd.DataFrame:
    """Query text + doc text cut to 512 Qwen3 tokens (the rule of the 4B subsets) straight from the segments; writes nothing."""
    import polars as pl
    from r2ai.s37 import export as E
    pairs = E.load_pairs(None).filter(pl.col('pair_id').is_in(pair_ids))
    queries = dict(pl.read_parquet(E.QUERY).iter_rows())
    docs = E.truncated(E.tokenizer(), E.doc_texts(sorted(pairs['segment'].unique().to_list())))
    return E.base_rows(pairs, queries, docs).to_pandas()


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


FINAL = RUN / 'jina-final'
FINAL2 = RUN / 'jina-final2'
S36_EXTRA_POOL = RUNS_DIR / 's36-fuse4' / 'extra' / 'jina' / 'pool.parquet'
S36_MISSING = RUNS_DIR / 's36-fuse4' / 'cal' / 'jina-missing-pairs.parquet'
S36_MISSING_SHA = '42f7fc9e0025e5b54154419cd4494df4482c30974382b80e6a3832cdf5e67aa6'
S36_MISSING_PAIRS, S36_MISSING_QUERIES = 21311, 174


def previous_scores(run: Path) -> pd.DataFrame:
    """Every score a finished or partial run owns: reused scores carried into it + its own parts."""
    frames = [pd.read_parquet(f) for f in sorted((run / 'parts-b10s0').glob('part-*.parquet'))]
    if (run / 'reused_scores.parquet').exists():
        frames.append(pd.read_parquet(run / 'reused_scores.parquet'))
    return pd.concat(frames, ignore_index=True).drop_duplicates(['query_id', 'doc_id']) if frames else pd.DataFrame(columns=['query_id', 'doc_id', 'score'])


def final_pool(prev_pool: Path | None = None, prev_run: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Final Jina pool: S35 pool + every S37 pair in pairs_new.parquet + the S36 missing pairs (Vu's file, SHA pinned).
    Returns (full pool of every affected query, rows of the queries still to score, earlier scores that stay valid).
    Earlier scores come from `prev_run` (default: the first, partial run in OUT); a query keeps them only when its ordered rows are unchanged."""
    prev_pool = prev_pool or OUT / 'pool.parquet'
    prev_run = prev_run or OUT
    if sha256_file(S36_MISSING) != S36_MISSING_SHA:
        raise SystemExit('jina-missing-pairs.parquet SHA differs from the one Vu gave')
    miss = pd.read_parquet(S36_MISSING)
    if len(miss) != S36_MISSING_PAIRS or miss.query_id.nunique() != S36_MISSING_QUERIES:
        raise SystemExit('jina-missing-pairs.parquet pair/query counts differ from the ones Vu gave')
    s36 = pd.read_parquet(S36_EXTRA_POOL).drop_duplicates(['query_id', 'doc_id'])
    sup = miss[['query_id', 'doc_id']].merge(s36, on=['query_id', 'doc_id'], how='left', validate='one_to_one')
    if sup.doc_text.isna().any():
        raise SystemExit(f'{int(sup.doc_text.isna().sum())} S36 missing pairs have no text in the S36 extra Jina pool')
    old = pd.read_parquet(S35 / 'pool.parquet')
    new = new_pairs()
    add_new = pd.DataFrame({
        'query_id': new.query_id, 'doc_id': new.doc_id, 'cosine_rank': 20_000 + new.cosine_rank, 'cosine_score': new.cosine_score,
        'p4b': np.nan, 'in_pool211': new.in_top211, 'in_pool13': new.in_pool13, 'lang': new.lang, 'query_text': new.query_text,
        'doc_text': new.doc_text, 'doc_tokens': new.doc_tokens, 'source': 's37'})
    add_sup = pd.DataFrame({
        'query_id': sup.query_id, 'doc_id': sup.doc_id, 'cosine_rank': 30_000 + sup.cosine_rank, 'cosine_score': sup.cosine_score,
        'p4b': np.nan, 'in_pool211': False, 'in_pool13': False, 'lang': sup.lang, 'query_text': sup.query_text,
        'doc_text': sup.doc_text, 'doc_tokens': sup.doc_tokens, 'source': 's36'})
    affected = np.array(sorted(set(new.query_id) | set(sup.query_id)))
    base = old[old.query_id.isin(affected)].assign(source='s35')
    keep = pd.concat([base, add_new, add_sup], ignore_index=True).drop_duplicates(['query_id', 'doc_id'], keep='first')
    pool = keep.sort_values(['query_id', 'cosine_score', 'doc_id'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
    # queries already scored in the first (partial) run whose pool did not change keep those scores: Jina seeds the block split by
    # query_id over that query's own rows, so identical ordered rows give identical scores.
    done = previous_scores(prev_run)
    prev = pd.read_parquet(prev_pool)[['query_id', 'doc_id']]
    same = set()
    prev_docs = prev.groupby('query_id').doc_id.apply(tuple)
    new_docs = pool.groupby('query_id').doc_id.apply(tuple)
    done_q = done.groupby('query_id').size()
    for q, docs in new_docs.items():
        if q in prev_docs.index and prev_docs[q] == docs and done_q.get(q, 0) == len(docs):
            same.add(q)
    todo = pool[~pool.query_id.isin(same)].reset_index(drop=True)
    return pool, todo, done[done.query_id.isin(same)].reset_index(drop=True)


def build_final(second: bool = False) -> dict:
    """`final`: the pool of the first complete run (jina-final). `final2`: top-up after the last crawl: only the queries whose pool
    changed since jina-final are scored (jina-final2), every other query keeps its jina-final score. Never overwrites a run that has a pool."""
    out, prev_pool, prev_run = (FINAL2, FINAL / 'pool_full.parquet', FINAL) if second else (FINAL, None, None)
    if (out / 'pool.parquet').exists():
        raise SystemExit(f'{out} already has a pool (a run may be using it): refusing to overwrite')
    pool, todo, reused = final_pool(prev_pool, prev_run)
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / 'pool_full.parquet', index=False)
    todo.to_parquet(out / 'pool.parquet', index=False)
    reused.to_parquet(out / 'reused_scores.parquet', index=False)
    per_q = todo.groupby('query_id').size()
    man = {'queries_in_final_pool': int(pool.query_id.nunique()), 'pairs_final_pool': len(pool),
           'pairs_by_source': {k: int(v) for k, v in pool.source.value_counts().items()},
           'queries_reused_from_previous_run': int(reused.query_id.nunique()), 'pairs_reused': len(reused),
           'queries_to_score': int(todo.query_id.nunique()), 'pairs_to_score': len(todo),
           'pairs_per_query_to_score': {'mean': float(per_q.mean()) if len(per_q) else 0.0, 'min': int(per_q.min()) if len(per_q) else 0,
                                        'max': int(per_q.max()) if len(per_q) else 0},
           'layout': 'b10s0', 's36_missing_sha256': S36_MISSING_SHA,
           'previous_pool_sha256': sha256_file(prev_pool or OUT / 'pool.parquet'),
           'pool_full_sha256': sha256_file(out / 'pool_full.parquet'), 'pool_sha256': sha256_file(out / 'pool.parquet')}
    atomic_json(out / 'pool_manifest.json', man)
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
    ap.add_argument('command', choices=('build', 'compare', 'final', 'final2'))
    a = ap.parse_args(argv)
    res = {'build': build, 'compare': compare, 'final': build_final, 'final2': lambda: build_final(second=True)}[a.command]()
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
