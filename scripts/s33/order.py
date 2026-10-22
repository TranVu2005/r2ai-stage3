import pyarrow.parquet as pq, pandas as pd, numpy as np
from urllib.parse import urlsplit
v2 = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet'); v1 = pd.read_parquet('../r2ai-stage3-old/data/raw/links_corpus.parquet')
q = pd.read_parquet('data/raw/query.parquet')
print('query', q.dtypes.to_dict(), len(q), 'id range', q.id.min(), q.id.max(), 'sorted', q.id.is_monotonic_increasing, 'unique', q.id.is_unique)
print('query len chars min/med/max', q['query'].str.len().describe()[['min','50%','max']].tolist())
for i in [0,1,2,600,1199]: print(' Q', q.id[i], ascii(q['query'][i][:120]))
for n, d in [('v2', v2), ('v1', v1)]:
    ids = d.id.values; dif = np.diff(ids)
    print(n, 'rows', len(d), 'id min/max', ids.min(), ids.max(), 'monotonic', bool((dif > 0).all()), 'n_decrease', int((dif < 0).sum()),
          'gap==1 share', round(float((dif == 1).mean()), 4), 'gap quantiles', np.percentile(dif[dif>0], [50, 90, 99, 99.9]).tolist())
    rg = pq.ParquetFile('data/raw/links_corpus_vi_en_zh.parquet' if n == 'v2' else '../r2ai-stage3-old/data/raw/links_corpus.parquet')
    print(' row-group id ranges', [(rg.metadata.row_group(i).column(0).statistics.min, rg.metadata.row_group(i).column(0).statistics.max, rg.metadata.row_group(i).num_rows) for i in range(min(5, rg.num_row_groups))])
    host = d.url.map(lambda u: urlsplit(u).netloc.lower()); d['host'] = host
    # are hosts contiguous blocks in id order?
    runs = (host != host.shift()).sum(); print(' host runs in row order', int(runs), 'distinct hosts', host.nunique())
    # url sorted within host?
    g = d.groupby('host', sort=False)
    srt = g.url.apply(lambda s: s.is_monotonic_increasing); print(' hosts whose urls are sorted in id order', int(srt.sum()), '/', len(srt))
    print(' url globally sorted?', d.url.is_monotonic_increasing, ' (host, url) sorted?', d.sort_values(['host','url']).id.is_monotonic_increasing)
    print(' first 12 hosts by first id', g.id.agg(['min','max','count']).sort_values('min').head(12).to_string())
m = v2.merge(v1, on='id', how='left', suffixes=('', '_v1'))
print('v2 ids in v1', int(m.url_v1.notna().sum()), 'same url', int((m.url == m.url_v1).sum()))
