"""S26 extractor fingerprint, step 2: BGE-M3 dense doc-level (first 4096 chars, max_len 512, same as full_embed.py) of
each extractor variant of ext_texts.parquet, AUC -min-rank of kept (corpus v2) vs dropped over the 1,200 Vietnamese
queries (same measure as pool_fingerprint_doclen.py). A doc whose variant text is missing/empty is embedded as ''.
`<variant>+title` prefixes the extracted text with the sample title and a blank line (doc_text layout).
Embeddings cached in <out>/emb_<variant>.npy; results appended to <out>/auc.jsonl.
Usage: ext_fingerprint.py <scratch_with_q_dense_and_pool> <out> <variant,...>
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
ext = pd.read_parquet(out / 'ext_texts.parquet'); docs = docs.merge(ext, on='doc_id', how='left')
y = docs.kept.astype(int).values
Q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda()
enc = None
for v in sys.argv[3].split(','):
    cache = out / f'emb_{v}.npy'
    if cache.exists():
        D = np.load(cache)
    else:
        base = v.split('+')[0]
        col = docs.doc_text if base == 'current' else docs[base]
        texts = [(x or '') for x in col]
        if v.endswith('+title'):
            texts = [(t or '') + '\n\n' + x for t, x in zip(docs.title, texts)]
        texts = [x[:L * 8] for x in texts]
        enc = enc or M3Encoder(max_len=L)
        order = np.argsort([len(t) for t in texts], kind='stable'); D = np.zeros((len(texts), 1024), np.float16)
        i = 0
        while i < len(order):
            bs = 64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16
            b = order[i:i + bs]; D[b] = enc.encode_batch([texts[k] for k in b])[0]; i += bs
        np.save(cache, D)
    sc = (Q @ torch.from_numpy(D).cuda().T).float()
    best = sc.max(0).values.cpu().numpy(); mr = (sc.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    col = docs.doc_text if v.split('+')[0] == 'current' else docs[v.split('+')[0]]
    r = {'variant': v, 'auc_best': roc_auc_score(y, best), 'auc_minrank': roc_auc_score(y, -mr),
         'kept_share_minrank1': float(y[mr == 1].mean()), 'r_prec': float(y[np.argsort(mr, kind='stable')[:y.sum()]].mean()),
         'empty_docs': int(col.fillna('').str.len().eq(0).sum()), 'median_chars': float(col.fillna('').str.len().median())}
    print(json.dumps(r), flush=True)
    with open(out / 'auc.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(r) + '\n')
