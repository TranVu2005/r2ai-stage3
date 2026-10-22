import pandas as pd, numpy as np
v1 = pd.read_parquet('../r2ai-stage3-old/data/raw/links_corpus.parquet'); u = v1.url.values
bad = np.flatnonzero(u[1:] < u[:-1]); print('v1 url-order breaks', len(bad))
for b in bad: print(v1.iloc[b-1:b+2].to_string())
v2 = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet'); pm = v2[v2.url.str.contains('pubmed')]
pmid = pm.url.str.extract(r'/(\d+)/')[0].astype(int); print('pubmed id order by pmid numeric?', pmid.is_monotonic_increasing, 'by string?', pm.url.is_monotonic_increasing)
print(pd.Series(np.diff(pm.id.values)).describe().to_string())
