"""Fetch priority of the longchau URLs that have a CC capture (CPU only).

score(url) = best (smallest) rank the URL reaches over the 1,200 queries in the BM25-on-slug ranking of VS-append
(same pin, k1/b and tokenizer as r2ai.submit.slug_append, ranks over the whole slug pool, ties by pool order, score > 0).
Groups: 0 = rank <= 20 for some query, 1 = <= 40, 2 = <= 100, 3 = the rest (also: never in the first 100 or score 0).
Fetch order = group, best rank, url id.

  python -m r2ai.cc_longchau.priority build     # -> state/cc_longchau/priority.parquet + out/runs/cc-longchau/priority_report.json
  python -m r2ai.cc_longchau.priority report    # groups x fetch status
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai.paths import QUERY_FILE, resolve_path
from r2ai.submit.best import load_config
from r2ai.submit.slug_append import bm25_matrix, query_matrix, sha256

from .common import Roots, atomic_json
from .fetch import load_targets

CONFIG = 'configs/submission-best.yaml'
CUTS = (20, 40, 100)
GROUP_NAMES = ('top20', 'top40', 'top100', 'rest')
DEPTH = max(CUTS)
QUERY_BATCH = 100


def priority_path(roots: Roots):
    return roots.state / 'priority.parquet'


def best_ranks(pin, pin_sha256, k1, b, queries, depth=DEPTH) -> dict[int, int]:
    """doc_id -> best rank (1-based, <= depth) over all queries in the slug BM25 ranking of the pool."""
    pin = resolve_path(pin)
    if sha256(pin) != pin_sha256:
        raise ValueError(f'slug pin SHA256 mismatch: {pin}')
    t = pq.read_table(pin).to_pydict()
    urls, in_pool, doc_id = t['url'], np.array(t['in_pool'], bool), t['doc_id']
    pool = np.flatnonzero(in_pool)
    WT, vocab = bm25_matrix(urls, k1, b)
    WTp = WT[:, pool]
    q = pq.read_table(queries).to_pylist()
    Q = query_matrix([r['query'] for r in q], vocab)
    best: dict[int, int] = {}
    for a in range(0, len(q), QUERY_BATCH):
        S = (Q[a:a + QUERY_BATCH] @ WTp).toarray()
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(len(pool)), -S[i]))
            seen = 0
            for j in order:
                if S[i][j] <= 0 or seen >= depth:
                    break
                d = doc_id[pool[j]]
                if d < 0:
                    continue                                    # ranks count real doc ids only, like the picks of slug_append
                seen += 1
                if d not in best or seen < best[d]:
                    best[d] = seen
    return best


def group_of(rank) -> int:
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return len(CUTS)
    return next((g for g, c in enumerate(CUTS) if rank <= c), len(CUTS))


def build(roots: Roots, member_ids=None) -> pd.DataFrame:
    cfg = load_config(str(resolve_path(CONFIG)))
    sla = cfg['postprocess']['slug_append']
    best = best_ranks(sla['pin'], sla['pin_sha256'], float(sla['k1']), float(sla['b']), str(resolve_path(cfg['queries'])))
    t = load_targets(roots, member_ids)[['url_id']]
    df = pd.DataFrame({'url_id': t.url_id.astype('int64'), 'best_rank': t.url_id.map(best).astype('Int64')})
    df['group'] = [group_of(None if pd.isna(r) else int(r)) for r in df.best_rank]
    df['group_name'] = df.group.map(dict(enumerate(GROUP_NAMES)))
    roots.guard(priority_path(roots)).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(priority_path(roots), index=False)
    return df


def load_order(roots: Roots) -> pd.DataFrame:
    return pd.read_parquet(priority_path(roots))


def report(roots: Roots, df: pd.DataFrame) -> dict:
    db = sqlite3.connect(f'file:{roots.fetch_db.as_posix()}?mode=ro', uri=True)
    try:
        st = dict(db.execute('SELECT url_id, status FROM fetch').fetchall())
    finally:
        db.close()
    df = df.assign(status=df.url_id.map(st).fillna('pending'))
    out = {}
    for g, name in enumerate(GROUP_NAMES):
        s = df[df.group == g].status.value_counts().to_dict()
        n = int((df.group == g).sum())
        out[name] = {'urls': n, 'ok': int(s.get('ok', 0)), 'other_final': int(sum(v for k, v in s.items() if k not in ('ok', 'pending', 'unavailable'))),
                     'unavailable': int(s.get('unavailable', 0)), 'pending': int(s.get('pending', 0)) }
        out[name]['to_do'] = n - out[name]['ok'] - out[name]['other_final']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['build', 'report'])
    ap.add_argument('--member-ids')
    a = ap.parse_args(argv)
    roots = Roots()
    roots.preflight()
    if a.action == 'build':
        df = build(roots, a.member_ids)
        rep = {'urls': len(df), 'rule': __doc__.split('\n\n')[0], 'groups': report(roots, df),
               'priority_sha256': sha256(priority_path(roots)), 'cuts': CUTS}
        atomic_json(roots, roots.run / 'priority_report.json', rep)
    else:
        rep = report(roots, load_order(roots))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
