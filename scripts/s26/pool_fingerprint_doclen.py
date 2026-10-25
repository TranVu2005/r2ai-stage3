"""S26 P-A: BGE-M3 dense doc-level variants on the zh sample (kept = in corpus v2), Vietnamese queries.

Variants: title + body truncated at max_len in {128, 256, 512, 1024, 2048}; title only; body only (512). Doc text is the
extracted docs.parquet text (title, doc_text). Embeddings cached in <out>/bgem3_doc_<variant>.npy.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

S, out = Path(sys.argv[1]), Path(sys.argv[2])
variants = sys.argv[3].split(',') if len(sys.argv) > 3 else ['tb128', 'tb256', 'tb1024', 'tb2048', 'title', 'body512']
docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'title', 'doc_text']).merge(
    pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']], on='doc_id')
y = docs.kept.astype(int).values
Q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda()


def text_of(v):
    if v == 'title':
        return [(t or '') for t in docs.title], 128
    if v == 'body512':
        return [(x or '') for x in docs.doc_text], 512
    L = int(v[2:])
    return [(t or '') + '\n' + (x or '')[:L * 8] for t, x in zip(docs.title, docs.doc_text)], L


for v in variants:
    cache = out / f'bgem3_doc_{v}.npy'
    if cache.exists():
        D = np.load(cache)
    else:
        texts, L = text_of(v); enc = M3Encoder(max_len=L)
        order = np.argsort([len(t) for t in texts]); D = np.zeros((len(texts), 1024), np.float16)
        bs = max(2, 16384 // L)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]; D[b] = enc.encode_batch([texts[k] for k in b])[0]
        np.save(cache, D); del enc; torch.cuda.empty_cache()
    sc = (Q @ torch.from_numpy(D).cuda().T).float()
    best = sc.max(0).values.cpu().numpy(); mr = (sc.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    print(f'bge-m3 doc {v:8s} AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  '
          f'kept share@minrank1 {y[mr == 1].mean():.3f}  R-prec {y[np.argsort(mr, kind="stable")[:y.sum()]].mean():.3f}', flush=True)
