"""S26 retriever fingerprint with intfloat/multilingual-e5-small (doc = first 512 tokens of title+body)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

S = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(os.environ['E5_DIR']); m = AutoModel.from_pretrained(os.environ['E5_DIR']).half().cuda().eval()


@torch.inference_mode()
def emb(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        h = m(**e).last_hidden_state; a = e['attention_mask'].unsqueeze(-1)
        out.append(torch.nn.functional.normalize((h * a).sum(1) / a.sum(1), dim=-1).float())
    return torch.cat(out)


docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'title', 'doc_text'])
per = pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']]
docs = docs.merge(per, on='doc_id')
q = pd.read_parquet('data/raw/query.parquet').sort_values('id')['query'].tolist()
Q = emb(['query: ' + x for x in q]); D = emb(['passage: ' + (t or '') + '\n' + (x or '')[:3000] for t, x in zip(docs.title, docs.doc_text)])
sc = Q @ D.T
best = sc.max(0).values.cpu().numpy(); mr = (sc.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
y = docs.kept.astype(int).values
print(f'e5-small doc-level  AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  kept share@minrank1 {y[mr == 1].mean():.3f}  n {len(y)}')
