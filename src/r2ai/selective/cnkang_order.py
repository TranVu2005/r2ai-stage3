"""Pending-cnkang crawl order from a path-prefix prior learned on the 2% zh sample. Read-only on crawl state.

prior(prefix) = smoothed mean top-30 reference hits per sample doc; unseen prefix -> global mean.
Output order only; nothing here writes crawl.db.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from r2ai.selective import step2_zh as z
from r2ai.selective.common import OUT, ZH_DB, ro

DOMAIN = 'cnkang.com'
URL_PER_SEC = 0.335          # zh-full REPORT, cnkang after resume (estimate)
DONE = ('ok', 'thin', 'soft404_or_home', 'blocked_4xx', 'bot_challenge', 'dead_origin')
CURVE_N = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)


def fit_prior(prefixes, hits, m=5.0):
    g = float(np.mean(hits))
    df = pd.DataFrame({'k': list(prefixes), 'h': hits}).groupby('k').h.agg(['sum', 'count'])
    return ((df['sum'] + m * g) / (df['count'] + m)).to_dict(), g


def order_pending(pend, prior, mean):
    """pend: url_norm,url,rank. Returns url_norm,prefix,prior_score,rank(0..n-1),db_rank(old rank values, permuted)."""
    out = pend[['url_norm']].copy()
    out['prefix'] = [z.path_feats(u)[0] for u in pend.url]
    out['prior_score'] = [prior.get(p, mean) for p in out.prefix]
    out = out.sort_values(['prior_score', 'url_norm'], ascending=[False, True], kind='mergesort').reset_index(drop=True)
    out['rank'] = np.arange(len(out))
    out['db_rank'] = np.sort(pend['rank'].to_numpy())
    return out


def recall_curve(score, hits, fractions):
    order = np.lexsort((np.arange(len(score)), -score))
    cum = np.cumsum(np.asarray(hits, float)[order]) / max(float(np.sum(hits)), 1.0)
    return {f: float(cum[max(1, int(round(f * len(score)))) - 1]) for f in fractions}


def sample_cnkang():
    docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'domain', 'url'])
    docs = docs[docs.domain == DOMAIN].reset_index(drop=True)
    refs = z.load_refs(pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id']))
    h = Counter(i for l in refs[30].values() for i in l)
    docs['h30'] = docs.doc_id.map(h).fillna(0).astype(int)
    docs['prefix'] = [z.path_feats(u)[0] for u in docs.url]
    docs['idnum'] = np.nan
    return docs


def main():
    docs = sample_cnkang()
    hits = docs.h30.to_numpy(float)
    prior, mean = fit_prior(docs.prefix, hits)
    cv = z.cv_prior(docs, hits, 'prefix')
    curve = recall_curve(cv, hits, CURVE_N)
    c = ro(ZH_DB)
    marks = ','.join('?' * len(DONE))
    pend = pd.read_sql(f"select url_norm,url,rank from urls where domain=? and status not in ({marks})", c,
                       params=(DOMAIN, *DONE))
    out = order_pending(pend, prior, mean)
    unseen = float((~out.prefix.isin(prior)).mean())
    out.to_parquet(OUT / 'cnkang_order.parquet', index=False)
    n = len(out)
    rows = [{'crawl_pct': f, 'urls': int(round(f * n)), 'ref_recall_est': curve[f], 'random_recall': f,
             'hours_est': round(f * n / URL_PER_SEC / 3600, 1)} for f in CURVE_N]
    pd.DataFrame(rows).to_csv(OUT / 'cnkang_curve.csv', index=False)
    json.dump({'pending': n, 'prefixes_in_sample': len(prior), 'pending_prefix_unseen_frac': unseen,
               'sample_docs': len(docs), 'url_per_sec': URL_PER_SEC}, open(OUT / 'cnkang_order.json', 'w'), indent=1)
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print(n, unseen)


if __name__ == '__main__':
    main()
