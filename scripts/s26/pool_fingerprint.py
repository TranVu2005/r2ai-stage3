"""S26 retriever fingerprint: how well does each scoring variant separate kept (corpus v2) vs dropped zh sample docs?

Variants on the pre-v2 zh 2% sample (BGE-M3 index segment): dense max-chunk, sparse max-chunk, hybrid 0.7/0.3,
first-chunk-only (title/lead) dense. Metric: AUC of per-doc best-over-queries score and of -min-rank.
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, scipy.sparse as sp
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from sklearn.metrics import roc_auc_score
from r2ai.index.bge_m3 import M3Encoder

S = Path(sys.argv[1])
seg = Path('data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256')
meta = json.load(open(seg / 'segment.json')); n, V = meta['rows'], meta['vocabulary']
dense = torch.from_numpy(np.load(seg / 'dense.npy')).cuda()
doc = np.fromfile(seg / 'doc_id.bin', dtype=np.int64)
indptr = np.fromfile(seg / 'indptr.bin', dtype=np.int64); nnz = meta['nnz']
idx_dtype = np.int32 if os.path.getsize(seg / 'indices.bin') == 4 * nnz else np.int64
val_dtype = np.float16 if os.path.getsize(seg / 'data.bin') == 2 * nnz else np.float32
X = sp.csr_matrix((np.fromfile(seg / 'data.bin', dtype=val_dtype).astype(np.float32), np.fromfile(seg / 'indices.bin', dtype=idx_dtype), indptr), shape=(n, V))
qtxt = pd.read_parquet('data/raw/query.parquet').sort_values('id')['query'].tolist()
enc = M3Encoder(); qd, qs = [], []
for i in range(0, len(qtxt), 32):
    d_, s_ = enc.encode_batch(qtxt[i:i + 32]); qd.append(d_); qs += s_
qd = torch.from_numpy(np.concatenate(qd)).cuda()
rows = np.repeat(np.arange(len(qs)), [len(t) for t, _ in qs])
Q = sp.csr_matrix((np.concatenate([v for _, v in qs]).astype(np.float32), (rows, np.concatenate([t for t, _ in qs]))), shape=(len(qs), V))
dn = (qd @ dense.T).float().cpu().numpy()                    # [Q, chunks]
spm = (Q @ X.T).toarray().astype(np.float32)                 # [Q, chunks]
u, inv = np.unique(doc, return_inverse=True)
first = np.zeros(n, bool); first[np.unique(inv, return_index=True)[1]] = True
per = pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']].set_index('doc_id').reindex(u)
y = per.kept.astype(int).values


def doc_scores(M, mask=None):
    M = torch.from_numpy(M if mask is None else np.where(mask[None, :], M, -9)).cuda()
    D = torch.full((M.shape[0], len(u)), -9.0, device='cuda').scatter_reduce(1, torch.from_numpy(inv).cuda().expand(M.shape[0], -1), M, 'amax')
    return D


def report(name, D):
    best = D.max(0).values.cpu().numpy(); mr = (D.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    print(f'{name:28s} AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  kept share@minrank1 {y[mr == 1].mean():.3f}', flush=True)


pass  # dense max-chunk: AUC best 0.7959 / -minrank 0.8333 (first run)
pass  # sparse max-chunk: AUC best 0.5587 / -minrank 0.5550 (first run)
z = lambda M: np.nan_to_num((M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6))
report('hybrid z 0.7d+0.3s', doc_scores(0.7 * z(dn) + 0.3 * z(spm)))
report('dense first chunk only', doc_scores(dn, first))
report('sparse first chunk only', doc_scores(spm, first))
