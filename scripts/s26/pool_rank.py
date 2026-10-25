"""S26 pool test 2: per doc of the pre-v2 zh 2% sample, its best rank over the 1,200 queries (doc score = max chunk
dense cosine). If BTC kept top-N/query with a retriever like ours, kept docs concentrate at min-rank ~1-3."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
from sklearn.metrics import roc_auc_score

S = Path(sys.argv[1])
seg = Path('data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256')
dense = torch.from_numpy(np.load(seg / 'dense.npy')).cuda()
doc = np.fromfile(seg / 'doc_id.bin', dtype=np.int64)
u, inv = np.unique(doc, return_inverse=True)
q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda()
sc = (q @ dense.T).float()                                     # [Q, chunks]
D = torch.full((len(q), len(u)), -9.0, device='cuda').scatter_reduce(1, torch.from_numpy(inv).cuda().expand(len(q), -1), sc, 'amax')
rank = D.argsort(1, descending=True).argsort(1) + 1            # rank of each doc per query
minrank = rank.min(0).values.cpu().numpy()
per = pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept', 'domain']]
r = pd.DataFrame({'doc_id': u, 'minrank': minrank}).merge(per, on='doc_id')
print('AUC(-minrank -> kept)', round(roc_auc_score(r.kept.astype(int), -r.minrank), 4))
for k in [1, 2, 3, 5, 10, 20, 50]:
    x = r[r.minrank <= k]; print(f'minrank<={k}: docs {len(x)}, kept share {x.kept.mean():.4f}, kept recall {x.kept.sum() / r.kept.sum():.4f}')
r.to_parquet(S / 'pool_zh_minrank.parquet', index=False)
