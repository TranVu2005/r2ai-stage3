"""S26 H4: host / path-section distribution of each query's top-50 docs in a submission (default P1-vi+10)."""
import os, sys
from urllib.parse import urlparse
import numpy as np, pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else 'P1-vi+10'
p = pd.read_parquet(f'{S}/{sub}.docs.parquet'); p = p[p['rank'] <= 50]
u = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet').rename(columns={'id': 'doc_id'})
u['host'] = u.url.map(lambda x: urlparse(x).netloc.lower().removeprefix('www.'))
def section(x):
    seg = [s for s in urlparse(x).path.split('/') if s]
    return '/'.join(seg[:2]) if len(seg) > 2 else (seg[0] if len(seg) > 1 else '(root)')
u['section'] = u.host + ':' + u.url.map(section)
m = p.merge(u, on='doc_id', how='left')
def ent(x):
    c = x.value_counts(normalize=True).values; return float(-(c * np.log2(c)).sum())
g = m.groupby('query_id')
f = pd.DataFrame({'top_host': g.host.agg(lambda x: x.value_counts().index[0]),
                  'top_host_share': g.host.agg(lambda x: x.value_counts(normalize=True).iloc[0]),
                  'n_hosts': g.host.nunique(), 'host_entropy': g.host.agg(ent),
                  'top_section_share': g.section.agg(lambda x: x.value_counts(normalize=True).iloc[0])})
print('overall host share in top-50:', m.host.value_counts(normalize=True).head(12).round(4).to_dict())
print(f.describe(percentiles=[.1, .5, .9]).round(3).to_string())
for t in [0.5, 0.7, 0.9]:
    print(f'queries top_host_share>={t}:', int((f.top_host_share >= t).sum()), f[f.top_host_share >= t].top_host.value_counts().head(6).to_dict())
print('dominant host clusters (top_host):', f.top_host.value_counts().head(10).to_dict())
q = pd.read_parquet(f'{S}/query_stats.parquet').set_index('id')
f = f.join(q[['n']]); f['len_bucket'] = pd.cut(f.n, [0, 15, 30, 60, 400])
print(f.groupby('len_bucket', observed=True)[['top_host_share', 'host_entropy']].mean().round(3).to_string())
print(pd.crosstab(f.len_bucket, f.top_host).loc[:, f.top_host.value_counts().head(6).index].to_string())
f.drop(columns='len_bucket').to_parquet(f'{S}/h4_query_host.parquet')
