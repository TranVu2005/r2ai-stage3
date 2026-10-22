import pandas as pd, numpy as np
from urllib.parse import urlsplit
v1 = pd.read_parquet('../r2ai-stage3-old/data/raw/links_corpus.parquet'); v2 = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id'])
ids = v1.id.values; g = np.diff(ids); h = np.flatnonzero(g > 1)
print('v1 rows', len(v1), 'id span', ids.max() - ids.min() + 1, 'missing ids inside', int((g - 1).sum()), 'hole runs', len(h), 'ids 1..582 missing', int(ids.min() - 1))
v1['host'] = v1.url.map(lambda u: urlsplit(u).netloc.lower())
lang = np.where(ids < 766442, 'vi', 'zh')
miss = pd.DataFrame({'after_id': ids[h], 'n_missing': g[h] - 1, 'host_before': v1.host.values[h], 'host_after': v1.host.values[h + 1], 'lang': lang[h],
                     'url_before': v1.url.values[h], 'url_after': v1.url.values[h + 1]})
print('missing by lang', miss.groupby('lang').n_missing.sum().to_dict())
print('run length quantiles', miss.n_missing.quantile([.5, .9, .99, 1]).tolist())
print(miss.groupby('host_before').n_missing.sum().sort_values(ascending=False).head(15).to_dict())
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(miss.sort_values('n_missing', ascending=False).head(8)[['after_id', 'n_missing', 'url_before', 'url_after']].to_string())
print(miss[['after_id', 'n_missing', 'url_before', 'url_after']].to_string())
vi_v1 = (lang == 'vi').sum(); vi_v2 = int((v2.id < 766442).sum()); print('v1 vi rows', vi_v1, 'v2 vi rows', vi_v2, 'v1 zh rows', (lang == 'zh').sum())
miss.to_parquet(f'{__import__("sys").argv[1]}/v1_holes.parquet')
