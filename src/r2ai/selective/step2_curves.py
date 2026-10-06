"""Recall curves at wider N for the domains with a usable signal; pooled 'rest' summary."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from r2ai.selective import step2_zh as z
from r2ai.selective.common import OUT

NS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)


def main():
    docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'domain', 'url', 'title'])
    refs = z.load_refs(docs)
    from collections import Counter
    h30 = Counter(i for l in refs[30].values() for i in l)
    docs['h30'] = docs.doc_id.map(h30).fillna(0).astype(int)
    pf = docs.url.map(z.path_feats)
    docs['prefix'] = [a for a, _ in pf]
    docs['idnum'] = [b if b is not None else np.nan for _, b in pf]
    rng = np.random.default_rng(42)
    rows = []
    rest = []
    for dom, d in docs.groupby('domain'):
        d = d.reset_index(drop=True)
        h = d.h30.to_numpy(float)
        sc = z.cv_prior(d, h, 'prefix')
        for n in NS:
            rows.append({'domain': dom, 'signal': 'prior_prefix', 'N': n, 'recall': z.recall_static(sc, h, n, rng), 'hits30': h.sum()})
        if dom not in ('120ask.com', 'cnkang.com', 'a-hospital.com', 'msdmanuals.cn'):
            for n in NS:
                m = max(1, int(round(n * len(d))))
                order = np.lexsort((rng.random(len(sc)), -sc))
                rest.append((n, h[order[:m]].sum(), h.sum()))
    qd = pickle.load(open(z.ZS / 'retrieval' / 'queries.pkl', 'rb'))[0].astype(np.float32)
    for dom in ('a-hospital.com', 'msdmanuals.cn'):
        d = docs[docs.domain == dom].reset_index(drop=True)
        pos = {i: k for k, i in enumerate(d.doc_id)}
        E = np.load(OUT / f'emb_{dom}_slug.npy')
        sim = qd @ E.T
        rp = [[pos[i] for i in refs[30][qi] if i in pos] for qi in range(1, 1201)]
        for n in NS:
            rows.append({'domain': dom, 'signal': 'dense_slug', 'N': n, 'recall': z.recall_dense(sim, rp, n), 'hits30': d.h30.sum()})
    df = pd.DataFrame(rest, columns=['N', 'got', 'tot']).groupby('N').sum()
    for n, r in df.iterrows():
        rows.append({'domain': 'REST(11 domains, pooled)', 'signal': 'prior_prefix', 'N': n, 'recall': r.got / r.tot, 'hits30': r.tot})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'zh_curves.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.pivot_table(index=['domain', 'signal', 'hits30'], columns='N', values='recall').round(3))


if __name__ == '__main__':
    main()
