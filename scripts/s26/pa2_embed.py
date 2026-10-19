"""S26 P-A2 step 2: BGE-M3 dense doc-level embedding (canonical doc_text as-is, max_len 512) for the docs of P1-vi+10.

Same encoder/settings as the best fingerprint (pool_fingerprint_doclen.py variant body512 = doc_text, 512 tokens).
Docs are processed shortest-first in blocks of --block; each block is saved to <out>/pa2_emb/<k>.npz so a restart resumes.
Final: <out>/pa2_doc_dense.npz (doc_id int64, dense fp16 [n, 1024]).
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

out = Path(sys.argv[1]); BLOCK = 5000; L = 512
part = out / 'pa2_emb'; part.mkdir(exist_ok=True)
d = pd.read_parquet(out / 'pa2_docs.parquet', columns=['doc_id', 'doc_text'])
d['doc_text'] = d.doc_text.fillna('').str.slice(0, L * 8)
d = d.iloc[np.argsort(d.doc_text.str.len().values, kind='stable')].reset_index(drop=True)
enc = M3Encoder(max_len=L); t0 = time.time()
for k, s in enumerate(range(0, len(d), BLOCK)):
    f = part / f'{k:04d}.npz'
    if f.exists():
        continue
    blk = d.iloc[s:s + BLOCK]; texts = blk.doc_text.tolist(); D = np.zeros((len(blk), 1024), np.float16)
    i = 0
    while i < len(texts):
        bs = 64 if len(texts[i]) < 600 else 32 if len(texts[i]) < 1500 else 16
        D[i:i + bs] = enc.encode_batch(texts[i:i + bs])[0]; i += bs
    np.savez(f, doc_id=blk.doc_id.values.astype(np.int64), dense=D)
    print(f'{s + len(blk)}/{len(d)} {time.time() - t0:.0f}s', flush=True)
parts = [np.load(f) for f in sorted(part.glob('*.npz'))]
ids = np.concatenate([p['doc_id'] for p in parts]); D = np.concatenate([p['dense'] for p in parts])
assert len(ids) == len(d) == len(set(ids))
np.savez(out / 'pa2_doc_dense.npz', doc_id=ids, dense=D)
print('done', len(ids), time.time() - t0)
