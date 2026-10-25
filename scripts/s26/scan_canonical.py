"""S26: one pass over canonical-20261018 dense vectors (BGE-M3) against the 1,200 query vectors.

Outputs (in <scratch>):
  doc_nearest.parquet : per canonical doc, max cosine over its chunks/queries and the argmax query id (+ lang/host/url)
  q_top_chunks.npz    : per query, top-K chunk rows (dense) for later doc aggregation (H4 host/path analysis)
Read-only on data/.
"""
import os
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1]); K = 800
C = Path('data/canonical-20261018')
dense = np.load(C / 'vectors/dense.npy', mmap_mode='r')
chunks = pq.read_table(C / 'chunks_t256.parquet', columns=['doc_id', 'has_vector']).to_pandas()
doc_of = chunks.doc_id.values[chunks.has_vector.values]
assert len(doc_of) == len(dense)
q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda(); qids = np.load(S / 'q_ids.npy')
best = np.empty(len(dense), np.float16); arg = np.empty(len(dense), np.int16)
ts = torch.full((len(q), K), -9.0, device='cuda'); ti = torch.zeros((len(q), K), dtype=torch.long, device='cuda')
t0 = time.time()
B = 200_000
for s in range(0, len(dense), B):
    blk = torch.from_numpy(np.array(dense[s:s + B])).cuda()
    sc = q @ blk.T                                   # [Q, b] fp16
    m, i = sc.max(0); best[s:s + len(m)] = m.cpu().numpy(); arg[s:s + len(m)] = i.cpu().numpy()
    cs = torch.cat([ts, sc.float()], 1); ci = torch.cat([ti, torch.arange(s, s + blk.shape[0], device='cuda').expand(len(q), -1)], 1)
    ts, idx = cs.topk(K, 1); ti = ci.gather(1, idx)
    if s // B % 5 == 0:
        print(f'{s + len(m)}/{len(dense)} {time.time() - t0:.0f}s', flush=True)
np.savez(S / 'q_top_chunks.npz', score=ts.cpu().numpy(), row=ti.cpu().numpy(), qids=qids)
df = pd.DataFrame({'doc_id': doc_of, 'best': best.astype(np.float32), 'qi': arg})
per = df.sort_values('best', ascending=False).drop_duplicates('doc_id')
per['query_id'] = qids[per.qi.values]
docs = pq.read_table(C / 'docs.parquet', columns=['doc_id', 'lang', 'host', 'url', 'quality']).to_pandas()
per = docs.merge(per.drop(columns='qi'), on='doc_id', how='left')
per.to_parquet(S / 'doc_nearest.parquet', index=False)
np.save(S / 'chunk_doc_of.npy', doc_of)
print('done', time.time() - t0, len(per), per.best.isna().sum())
