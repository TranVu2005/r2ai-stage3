import pandas as pd, numpy as np
from urllib.parse import urlsplit
v2 = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet'); v1 = pd.read_parquet('../r2ai-stage3-old/data/raw/links_corpus.parquet', columns=['id'])
v2['host'] = v2.url.map(lambda u: urlsplit(u).netloc.lower()); v2['in_v1'] = v2.id.isin(v1.id)
k = int(np.flatnonzero(np.diff(v2.id.values) < 0)[0]); print('decrease at row', k, v2.iloc[k-1:k+3][['id','url','in_v1']].to_string())
print('in_v1 by row block', v2.groupby(np.arange(len(v2)) // 100000).in_v1.mean().round(3).tolist())
new = v2[~v2.in_v1]; print('new ids', len(new), new.id.min(), new.id.max(), 'rows', new.index.min(), new.index.max())
print('new id gap==1 share', float((np.diff(new.id.values) == 1).mean()))
g = v2.groupby('host').agg(n=('id','size'), idmin=('id','min'), idmax=('id','max'), v1share=('in_v1','mean'), firstrow=('id', lambda s: s.index.min()))
g['tld'] = g.index.str.rsplit('.', n=1).str[-1]
print(g.sort_values('firstrow').to_string())
srt = v2.groupby('host').url.apply(lambda s: s.is_monotonic_increasing); print('unsorted host', srt[~srt].index.tolist())
h = srt[~srt].index[0]; s = v2[v2.host == h]; bad = np.flatnonzero(s.url.values[1:] < s.url.values[:-1])
print(len(bad), s.iloc[bad[0]-1:bad[0]+3].to_string())
