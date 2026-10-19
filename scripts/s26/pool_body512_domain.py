"""S26 P-A: per-domain AUC and min-rank profile for the best fingerprint so far (BGE-M3 dense, body only, 512 tokens)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S, out = Path(sys.argv[1]), Path(sys.argv[2])
docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'domain']).merge(
    pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']], on='doc_id')
D = np.load(out / 'bgem3_doc_body512.npy').astype(np.float32); Q = np.load(S / 'q_dense.npy').astype(np.float32)
sc = Q @ D.T
docs['minrank'] = (np.argsort(np.argsort(-sc, 1), 1) + 1).min(0); docs['best'] = sc.max(0)
docs['owner'] = sc.argmax(0)
y = docs.kept.astype(int)
print('pooled AUC -minrank', round(roc_auc_score(y, -docs.minrank), 4))
for k in [1, 2, 3, 5, 10, 20]:
    x = docs[docs.minrank <= k]; print(f'minrank<={k}: docs {len(x)}, kept share {x.kept.mean():.3f}, kept recall {x.kept.sum() / y.sum():.3f}')
rows = []
for d, g in docs.groupby('domain'):
    if g.kept.nunique() == 2 and len(g) >= 100:
        rows.append((d, len(g), int(g.kept.sum()), roc_auc_score(g.kept.astype(int), -g.minrank)))
t = pd.DataFrame(rows, columns=['domain', 'n', 'kept', 'auc']).sort_values('n', ascending=False)
print(t.round(4).to_string(index=False)); print('kept-weighted within-domain AUC', round(np.average(t.auc, weights=t.kept), 4))
own = docs[docs.kept].owner.value_counts().reindex(range(len(Q)), fill_value=0)
print('kept docs per owner query (argmax):', own.describe(percentiles=[.1, .5, .9]).round(1).to_dict(), 'queries with 0:', int((own == 0).sum()))
docs.to_parquet(out / 'body512_scored.parquet', index=False)
