"""S26 pool test (en): nearest test query per PubMed doc (BGE-M3 dense max chunk) and docs-per-query balance."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1])
q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda(); qids = np.load(S / 'q_ids.npy')
dense = np.load('data/index/en_pubmed/dense.npy', mmap_mode='r')
doc = pd.read_parquet('data/pubmed_en/chunks.parquet', columns=['doc_id']).doc_id.values
best = np.empty(len(dense), np.float32); arg = np.empty(len(dense), np.int32)
for s in range(0, len(dense), 100_000):
    sc = (torch.from_numpy(np.array(dense[s:s + 100_000])).cuda() @ q.T).float()
    m, i = sc.max(1); best[s:s + len(m)] = m.cpu().numpy(); arg[s:s + len(m)] = i.cpu().numpy()
per = pd.DataFrame({'doc_id': doc, 'best': best, 'query_id': qids[arg]}).sort_values('best', ascending=False).drop_duplicates('doc_id')
per.to_parquet(S / 'en_nearest.parquet', index=False)
print('en docs', len(per)); print('best', per.best.describe(percentiles=[.1, .5, .9]).round(4).to_dict())
cnt = per.query_id.value_counts().reindex(qids, fill_value=0)
print('docs per nearest query', cnt.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_dict(), 'zero', int((cnt == 0).sum()))
cand = pd.read_parquet('artifacts/cand_en.parquet'); mine = pd.read_parquet('out/runs/s26-gold-structure/en-only/EN-only.cand.parquet')
ov = [len(set(cand[cand.query_id == i].doc_id) & set(mine[mine.query_id == i].doc_id)) for i in qids]
print('EN-only dense top50 vs cand_en (hybrid+bge-reranker top50) overlap/query', pd.Series(ov).describe().round(2).to_dict())
