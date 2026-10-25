"""S26 P-A: BGE-M3 dense score-normalisation variants on the zh sample (CPU): raw max-chunk, per-query z-score,
per-doc mean of top-2 chunks, doc = chunk count-normalised sum. Kept = in corpus v2."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1])
seg = Path('data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256')
doc = np.fromfile(seg / 'doc_id.bin', dtype=np.int64)
order = np.argsort(doc, kind='stable'); ds = doc[order]
u, start = np.unique(ds, return_index=True)
D = np.load(seg / 'dense.npy')[order].astype(np.float32)
Q = np.load(S / 'q_dense.npy').astype(np.float32)
M = Q @ D.T                                                    # [Q, chunks] sorted by doc
y = pd.read_parquet(S / 'pool_zh_sample.parquet').set_index('doc_id').kept.reindex(u).astype(int).values


def rep(name, X):
    best = X.max(0); mr = (np.argsort(np.argsort(-X, 1), 1) + 1).min(0)
    print(f'{name:34s} AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  kept share@minrank1 {y[mr == 1].mean():.3f}', flush=True)


mx = np.maximum.reduceat(M, start, axis=1)
rep('raw max-chunk', mx)
rep('per-query z of max-chunk', (mx - mx.mean(1, keepdims=True)) / mx.std(1, keepdims=True))
rep('per-doc z over queries (max-chunk)', (mx - mx.mean(0, keepdims=True)) / mx.std(0, keepdims=True))
cnt = np.diff(np.r_[start, len(ds)])
rep('mean-chunk', np.add.reduceat(M, start, axis=1) / cnt)
Ms = M.copy(); top2 = np.empty_like(mx)
for j, (s0, c) in enumerate(zip(start, cnt)):
    blk = M[:, s0:s0 + c]
    top2[:, j] = blk.max(1) if c == 1 else np.sort(blk, 1)[:, -2:].mean(1)
rep('mean of top-2 chunks', top2)
