"""S26 pool test: are zh docs kept in corpus v2 closer to the 1,200 test queries than v1 zh docs that BTC dropped?

Uses the pre-v2 2% zh sample (index segment seg00001-sample-old-zh_sample_t256, crawled from corpus v1 before v2 existed).
Per doc: max over its chunks and over queries of BGE-M3 dense cosine. Compares kept (any id of the doc group in v2) vs dropped.
"""
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1])
seg = Path('data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256')
meta = json.load(open(seg / 'segment.json'))
dense = np.load(seg / 'dense.npy', mmap_mode='r')
doc = np.fromfile(seg / 'doc_id.bin', dtype=np.int64)
assert len(doc) == len(dense) == meta['rows']
q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda()
best = np.empty(len(dense), np.float32); arg = np.empty(len(dense), np.int32)
for s in range(0, len(dense), 50_000):
    sc = (torch.from_numpy(np.ascontiguousarray(dense[s:s + 50_000])).cuda() @ q.T).float()
    m, i = sc.max(1); best[s:s + len(m)] = m.cpu().numpy(); arg[s:s + len(m)] = i.cpu().numpy()
df = pd.DataFrame({'doc_id': doc, 'best': best, 'q': arg})
per = df.sort_values('best', ascending=False).drop_duplicates('doc_id')
docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'doc_ids_group', 'domain'])
v2 = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id)
docs['kept'] = docs.doc_ids_group.map(lambda g: any(int(x) in v2 for x in g))
per = per.merge(docs, on='doc_id', how='left')
per.to_parquet(S / 'pool_zh_sample.parquet', index=False)
print('docs', len(per), 'kept', int(per.kept.sum()), 'unmatched', int(per.kept.isna().sum()))
print(per.groupby('kept').best.describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
for t in [0.5, 0.55, 0.6, 0.65, 0.7]:
    g = per.groupby('kept').best.apply(lambda x: (x >= t).mean())
    print(f'share best>={t}:', g.round(4).to_dict())
dom = per.groupby(['domain', 'kept']).best.median().unstack()
dom['n_kept'] = per[per.kept == True].groupby('domain').size(); dom['n_drop'] = per[per.kept == False].groupby('domain').size()
print(dom.sort_values('n_drop', ascending=False).head(15).round(4).to_string())
# rank-based: per kept/drop, deciles of best → kept share
per['dec'] = pd.qcut(per.best, 10, labels=False)
print('kept share by best-decile:', per.groupby('dec').kept.mean().round(4).tolist())
from sklearn.metrics import roc_auc_score
m = per.kept.notna()
print('AUC(best -> kept)', roc_auc_score(per.kept[m].astype(int), per.best[m]))
for d_ in per.domain.value_counts().head(6).index:
    x = per[(per.domain == d_) & m]
    if x.kept.nunique() == 2:
        print(' AUC', d_, round(roc_auc_score(x.kept.astype(int), x.best), 4), 'n', len(x), 'kept', int(x.kept.sum()))
