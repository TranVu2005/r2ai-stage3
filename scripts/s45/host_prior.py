"""S45 host prior: per-host retention of BTC's zh pool selection, learned from the S26 kept/dropped labels.

Two rates per host of the pre-v2 2% zh sample (29,385 docs, kept = any id of the doc group in corpus v2):
  overall  kept / docs                                       (mixes topical density with the filter)
  top      kept / docs with min-rank <= TOP under dt512 CLS  (docs BTC's retriever had near the top of some query, so the
           rate isolates the second-stage filter; this is the rate the prior uses, see docs/s45.md)
Shrinkage: empirical-Bayes Beta(a, b) over hosts (method of moments on the per-host rates, weighted by n), posterior mean
(k + a) / (n + a + b), 95% equal-tailed Beta posterior interval. Hosts outside the sample get the prior mean.
Also kept rates by id-group size (1 / 2 ids) inside each host.

  host_prior.py <fp_out> <out_json> [TOP=10]
The JSON is the pinned prior table (`--host-prior`): {host: {"n", "k", "rate", "shrunk", "lo", "hi"}, "_default": ...}.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import beta as B

FP, OUTJ = Path(sys.argv[1]), Path(sys.argv[2])
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 10
docs = pd.read_parquet('D:/_r2ai_archive/data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'doc_ids_group', 'domain'])
docs['y'] = np.load(FP / 'y.npy').astype(int)
S = torch.from_numpy(np.load(FP / 'dt512.cls.npy')).cuda().float()
docs['mr'] = (S.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy(); del S
docs['grp'] = docs.doc_ids_group.map(lambda g: len(json.loads(g) if isinstance(g, str) else g)).clip(upper=2)


def eb(n, k):
    """method-of-moments Beta prior over hosts, weighted by n"""
    r = k / n; w = n / n.sum(); m = float((w * r).sum())
    v = float((w * (r - m) ** 2).sum()) - m * (1 - m) * float((w / n).sum())   # between-host variance, binomial noise removed
    v = max(v, 1e-6); s = max(m * (1 - m) / v - 1, 1.0)
    return m * s, (1 - m) * s


tables, out = {}, {}
for name, sub in [('overall', docs), ('top', docs[docs.mr <= TOP])]:
    g = sub.groupby('domain').agg(n=('y', 'size'), k=('y', 'sum'))
    a, b = eb(g.n.values.astype(float), g.k.values.astype(float))
    g['rate'] = g.k / g.n; g['shrunk'] = (g.k + a) / (g.n + a + b)
    g['lo'] = B.ppf(0.025, g.k + a, g.n - g.k + b); g['hi'] = B.ppf(0.975, g.k + a, g.n - g.k + b)
    for gs in (1, 2):
        x = sub[sub.grp == gs].groupby('domain').y.agg(['size', 'mean'])
        g[f'n_{gs}id'] = x['size']; g[f'rate_{gs}id'] = x['mean']
    g = g.sort_values('n', ascending=False)
    tables[name] = (g, a, b)
    print(f'== {name} (TOP={TOP}): prior Beta({a:.2f}, {b:.2f}), mean {a / (a + b):.4f}, pooled rate {sub.y.mean():.4f}')
    print(g.round(4).to_string())

g, a, b = tables['top']
prior = {h: {'n': int(r.n), 'k': int(r.k), 'rate': round(float(r.rate), 6), 'shrunk': round(float(r.shrunk), 6),
             'lo': round(float(r.lo), 6), 'hi': round(float(r.hi), 6)} for h, r in g.iterrows()}
prior['_default'] = {'shrunk': round(a / (a + b), 6), 'note': 'zh host outside the S26 sample: prior mean'}
prior['_meta'] = {'source': f'S26 zh 2pct sample, kept = in corpus v2; rate among docs with dt512 CLS min-rank <= {TOP}',
                  'beta_prior': [round(a, 4), round(b, 4)], 'top': TOP, 'vi': 'no labels: prior 1 (no change)'}
OUTJ.write_text(json.dumps(prior, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote', OUTJ)
