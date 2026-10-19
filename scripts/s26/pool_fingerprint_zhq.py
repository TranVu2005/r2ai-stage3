"""S26 P-A: retriever fingerprint with Chinese-translated queries (and BGE-M3 doc-level variants).

On the pre-v2 zh 2% sample (kept = in corpus v2): BGE-M3 dense/sparse max-chunk over the existing index segment with
zh queries; BGE-M3 dense doc-level (title + body, first 512 tokens; cached) with vi and zh queries; BM25 (jieba,
BM25Okapi k1=1.5 b=0.75) over title + body with zh queries. Prints AUC(best), AUC(-minrank), kept share at minrank 1.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

S, out, qzh_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
seg = Path('data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256')
meta = json.load(open(seg / 'segment.json')); n, V, nnz = meta['rows'], meta['vocabulary'], meta['nnz']
doc = np.fromfile(seg / 'doc_id.bin', dtype=np.int64)
u, inv = np.unique(doc, return_inverse=True)
per = pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']].set_index('doc_id').reindex(u)
y = per.kept.astype(int).values
qzh = pd.read_parquet(qzh_path).sort_values('id')['query_zh'].tolist()
qvi = pd.read_parquet('data/raw/query.parquet').sort_values('id')['query'].tolist()
enc = M3Encoder()


def encode(texts, bs=16):
    d, s = [], []
    for i in range(0, len(texts), bs):
        a, b = enc.encode_batch(texts[i:i + bs]); d.append(a); s += b
    return np.concatenate(d), s


def report(name, M):                     # M: [Q, docs] torch on cuda, doc order = u
    best = M.max(0).values.float().cpu().numpy(); mr = (M.float().argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    print(f'{name:40s} AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  kept share@minrank1 {y[mr == 1].mean():.3f}', flush=True)


def to_docs(M):                          # [Q, chunks] -> [Q, docs] max
    M = torch.as_tensor(M, device='cuda', dtype=torch.float32)
    return torch.full((M.shape[0], len(u)), -1e9, device='cuda').scatter_reduce(1, torch.from_numpy(inv).cuda().expand(M.shape[0], -1), M, 'amax')


# 1) BGE-M3 max-chunk with zh queries
qd, qs = encode(qzh)
dense = torch.from_numpy(np.load(seg / 'dense.npy')).cuda()
report('bge-m3 dense max-chunk, zh query', to_docs(torch.from_numpy(qd).cuda() @ dense.T))
del dense; torch.cuda.empty_cache()
idx_dtype = np.int32 if os.path.getsize(seg / 'indices.bin') == 4 * nnz else np.int64
val_dtype = np.float16 if os.path.getsize(seg / 'data.bin') == 2 * nnz else np.float32
X = sp.csr_matrix((np.fromfile(seg / 'data.bin', dtype=val_dtype).astype(np.float32), np.fromfile(seg / 'indices.bin', dtype=idx_dtype),
                   np.fromfile(seg / 'indptr.bin', dtype=np.int64)), shape=(n, V))
rows = np.repeat(np.arange(len(qs)), [len(t) for t, _ in qs])
Q = sp.csr_matrix((np.concatenate([v for _, v in qs]).astype(np.float32), (rows, np.concatenate([t for t, _ in qs]))), shape=(len(qs), V))
report('bge-m3 sparse max-chunk, zh query', to_docs((Q @ X.T).toarray()))

# 2) BGE-M3 doc-level (title + body, 512 tokens)
docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'title', 'doc_text']).set_index('doc_id').reindex(u)
texts = [(t or '') + '\n' + (x or '')[:4000] for t, x in zip(docs.title, docs.doc_text)]
cache = out / 'bgem3_docs512.npy'
if cache.exists():
    Dd = np.load(cache)
else:
    Dd, _ = encode(texts); np.save(cache, Dd)
Dd = torch.from_numpy(Dd).cuda()
qvd, _ = encode(qvi)
report('bge-m3 dense doc-level, vi query', torch.from_numpy(qvd).cuda() @ Dd.T)
report('bge-m3 dense doc-level, zh query', torch.from_numpy(qd).cuda() @ Dd.T)

# 3) BM25 over title + body, zh queries
import jieba
from rank_bm25 import BM25Okapi
jieba.setLogLevel(60)
bm = BM25Okapi([[w for w in jieba.lcut_for_search(t) if w.strip()] for t in texts])
B = np.stack([bm.get_scores([w for w in jieba.lcut_for_search(q) if w.strip()]) for q in qzh]).astype(np.float32)
np.save(out / 'bm25_zhq.npy', B)
report('bm25 jieba title+body, zh query', torch.from_numpy(B).cuda())
