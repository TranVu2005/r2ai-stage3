"""S26 P-A: how high could a perfectly matched retriever score on the kept/dropped zh sample?

Random fill (kept regardless of score) puts kept docs evenly in every score bin, so the kept share of the lowest bin
bounds the fill fraction: f <= share_low / share_all. Also reports per-domain AUC of BGE-M3 -min-rank (domain quotas
would cap a pooled AUC even with the right retriever).
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1])
r = pd.read_parquet(S / 'pool_zh_minrank.parquet').merge(pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'best']], on='doc_id')
base = r.kept.mean()
for col, asc in [('best', True), ('minrank', False)]:
    for q in [10, 20, 50]:
        b = pd.qcut(r[col].rank(method='first', ascending=asc), q, labels=False)
        low = r.kept[b == 0].mean()
        print(f'{col:8s} bins={q:3d}: kept share lowest bin {low:.4f} (n={int((b == 0).sum())}, kept {int(r.kept[b == 0].sum())}); base {base:.4f}; random-fill fraction <= {low / base:.4f}')
print('per-domain AUC(-minrank) and kept rate:')
rows = []
for d, g in r.groupby('domain'):
    if g.kept.nunique() == 2 and len(g) >= 100:
        rows.append((d, len(g), int(g.kept.sum()), g.kept.mean(), roc_auc_score(g.kept.astype(int), -g.minrank)))
t = pd.DataFrame(rows, columns=['domain', 'n', 'kept', 'kept_rate', 'auc']).sort_values('n', ascending=False)
print(t.round(4).to_string(index=False))
w = np.average(t.auc, weights=t.kept)
print(f'kept-weighted within-domain AUC {w:.4f}  (pooled {roc_auc_score(r.kept.astype(int), -r.minrank):.4f})')
