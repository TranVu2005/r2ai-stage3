"""Item 2: doc_id dispersion of J4 top-20 vs random docs (same lang mix; same host mix), in id space and parquet-row space."""
import pandas as pd, numpy as np, zipfile, json
from urllib.parse import urlsplit
S = __import__('sys').argv[1]
c = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet'); c['row'] = np.arange(len(c))
c['host'] = c.url.map(lambda u: urlsplit(u).netloc.lower())
c['lang'] = np.where(c.id < 766442, 'vi', np.where(c.id <= 4420561, 'zh', 'en'))
c.to_parquet(f'{S}/corpus_meta.parquet')
rows = json.loads(zipfile.ZipFile('out/runs/s31-strong-judge/build-topk/J4-K40.zip').read('J4-K40.json'))
top = {int(r['id']): [int(x) for x in r['relevant_docs']] for r in rows}
ci = c.set_index('id'); ci['id'] = ci.index; rng = np.random.default_rng(0)
by_lang = {l: g.index.values for l, g in ci.groupby('lang')}; by_host = {h: g.index.values for h, g in ci.groupby('host')}
W = [1, 5, 20, 100]
def stats(ids, space):
    v = np.sort(ci.loc[ids, space].values); g = np.diff(v)
    out = {'median_gap': float(np.median(g)), 'hosts': ci.loc[ids, 'host'].nunique()}
    for w in W:
        near = np.zeros(len(v), bool); near[1:] |= g <= w; near[:-1] |= g <= w
        out[f'share_nb_le{w}'] = near.mean(); out[f'runs_le{w}'] = int(((g <= w) & np.r_[True, g[:-1] > w]).sum())  # clusters of >=2
    return out
res = []
for N in [20, 5, 10, 40]:
  for q, ids in top.items():
    ids = [i for i in ids if i in ci.index][:N]
    lang = ci.loc[ids, 'lang'].values; host = ci.loc[ids, 'host'].values
    rl = [rng.choice(by_lang[l]) for l in lang]
    rh = []
    for h in host:
        x = rng.choice(by_host[h]); rh.append(x)
    for space in ['id', 'row']:
        for kind, lst in [('J4', ids), ('rand_lang', rl), ('rand_host', rh)]:
            if len(set(lst)) < len(lst) and kind != 'J4': lst = list(dict.fromkeys(lst))
            res.append({'N': N, 'q': q, 'space': space, 'kind': kind, **stats(lst, space)})
r = pd.DataFrame(res); r.to_parquet(f'{S}/idspread.parquet')
pd.set_option('display.width', 250)
print(r.groupby(['N', 'space', 'kind']).agg(['mean']).drop(columns='q').round(4).to_string())
print(r[(r.N == 20) & (r.space == 'row')].groupby('kind').median_gap.median())
