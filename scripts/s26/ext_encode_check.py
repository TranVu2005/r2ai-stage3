"""S26 extractor fingerprint, last checks: encode variants on the 5,000-doc subsample of the zh sample
(all 2,413 kept + 2,587 dropped, numpy RandomState(42) choice, same as pool_fingerprint_qwen3.py --subsample 5000).

  base   cached fp16 doc vectors of the current doc_text (emb_current.npy), fp16 query vectors (q_dense.npy)
  fp32   doc_text and the 1,200 queries re-encoded with M3Encoder(fp16=False)
  nl1    doc_text rebuilt as title + "\n" + body (current layout joins them with a blank line), fp16
AUC -min-rank / AUC best are computed over the subsample only (ranks among its 5,000 docs).
Usage: ext_encode_check.py <scratch_with_q_dense_and_pool> <extractor_out>
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

S, out = Path(sys.argv[1]), Path(sys.argv[2]); L = 512
docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'title', 'doc_text']).merge(
    pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']], on='doc_id')
rng = np.random.RandomState(42); drop = np.flatnonzero(~docs.kept.values)
sel = np.sort(np.r_[np.flatnonzero(docs.kept.values), rng.choice(drop, 5000 - int(docs.kept.sum()), replace=False)])
sub = docs.iloc[sel].reset_index(drop=True); y = sub.kept.astype(int).values
q = pd.read_parquet('data/raw/query.parquet').sort_values('id'); assert (q.id.values == np.load(S / 'q_ids.npy')).all()


def embed(texts, enc, div=1):
    """div > 1 shrinks the batches (fp32 on a 6 GB card spills into shared memory with the fp16 batch sizes)"""
    order = np.argsort([len(t) for t in texts], kind='stable'); D = np.zeros((len(texts), 1024), np.float32); i = 0
    while i < len(order):
        bs = max(1, (64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16) // div)
        b = order[i:i + bs]; D[b] = enc.encode_batch([texts[k] for k in b])[0]; i += bs
    return D


done = set()
if (out / 'encode_check.jsonl').exists():
    done = {json.loads(x)['variant'] for x in open(out / 'encode_check.jsonl', encoding='utf-8')}


def auc(Q, D, name, extra=None):
    sc = torch.from_numpy(Q).float().cuda() @ torch.from_numpy(D).float().cuda().T
    best = sc.max(0).values.cpu().numpy(); mr = (sc.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    r = {'variant': name, 'docs': len(D), 'auc_best': roc_auc_score(y, best), 'auc_minrank': roc_auc_score(y, -mr),
         'kept_share_minrank1': float(y[mr == 1].mean()), **(extra or {})}
    print(json.dumps(r), flush=True)
    with open(out / 'encode_check.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(r) + '\n')


texts = [(x or '')[:L * 8] for x in sub.doc_text]
if 'base_fp16' not in done:
    auc(np.load(S / 'q_dense.npy'), np.load(out / 'emb_current.npy')[sel], 'base_fp16')
if 'fp32' not in done:
    enc = M3Encoder(fp16=False, max_len=L)
    Q32 = np.concatenate([enc.encode_batch(q['query'].tolist()[i:i + 16])[0] for i in range(0, len(q), 16)]).astype(np.float32)
    auc(Q32, embed(texts, enc, div=4), 'fp32')
    del enc; torch.cuda.empty_cache()
nl, n_fallback = [], 0
for t, x in zip(sub.title, sub.doc_text):
    t, x = t or '', x or ''
    if t and x.startswith(t + '\n\n'):
        nl.append(t + '\n' + x[len(t) + 2:])
    else:
        nl.append(x); n_fallback += 1
if 'title_nl1_body' not in done:
    enc = M3Encoder(max_len=L)
    auc(np.load(S / 'q_dense.npy'), embed([x[:L * 8] for x in nl], enc), 'title_nl1_body', {'docs_unchanged_no_title_prefix': n_fallback})
