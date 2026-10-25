"""S26 P-A: retriever fingerprint with Qwen3-Embedding-4B (bitsandbytes 4-bit NF4, last-token pooling, L2).

Docs of the pre-v2 zh 2% sample are embedded once (title + body, first --doc-len tokens) and cached in <out>/;
queries are embedded with the model-card instruction, both the Vietnamese original and (if given) the zh translation.
Prints AUC(best) / AUC(-minrank) / kept share at minrank 1 for kept (corpus v2) vs dropped docs.
Run with the .venv interpreter (bitsandbytes lives there).
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

ap = argparse.ArgumentParser()
ap.add_argument('--scratch', required=True); ap.add_argument('--out', required=True); ap.add_argument('--model', required=True)
ap.add_argument('--doc-len', type=int, default=512); ap.add_argument('--batch-tokens', type=int, default=2048)
ap.add_argument('--subsample', type=int, default=0, help='keep all kept docs + random dropped up to N docs (seed 42)')
ap.add_argument('--query-zh'); ap.add_argument('--tag', default='qwen3emb4b'); ap.add_argument('--body-only', action='store_true')
a = ap.parse_args()
S, out = Path(a.scratch), Path(a.out); out.mkdir(parents=True, exist_ok=True)
TASK = 'Given a web search query, retrieve relevant passages that answer the query'
tok = AutoTokenizer.from_pretrained(a.model, padding_side='left')
model = AutoModel.from_pretrained(a.model, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.float16), dtype=torch.float16, device_map={'': 0}).eval()


@torch.inference_mode()
def embed(texts, max_len):
    ids = tok([t[:max_len * 6] for t in texts], add_special_tokens=True, truncation=True, max_length=max_len)['input_ids']
    order = np.argsort([len(x) for x in ids]); res = np.zeros((len(texts), model.config.hidden_size), np.float16)
    i, t0 = 0, time.time()
    while i < len(order):
        L = len(ids[order[min(len(order) - 1, i)]]); j = i
        while j < len(order) and (j - i + 1) * len(ids[order[j]]) <= a.batch_tokens:
            j += 1
        j = max(j, i + 1); b = order[i:j]
        enc = tok.pad({'input_ids': [ids[k] for k in b]}, padding=True, return_tensors='pt').to('cuda')
        h = model(**enc).last_hidden_state[:, -1]                        # left padding => last token
        res[b] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        i = j
        if (i // 500) != ((i - len(b)) // 500):
            print(f'  {i}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    print(f'embedded {len(texts)} in {time.time() - t0:.0f}s', flush=True)
    return res


docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'title', 'doc_text']).merge(
    pd.read_parquet(S / 'pool_zh_sample.parquet')[['doc_id', 'kept']], on='doc_id')
ref = np.load(out / 'bgem3_doc_body512.npy') if (out / 'bgem3_doc_body512.npy').exists() else None   # same doc order (pool_fingerprint_doclen.py)
if a.subsample:
    rng = np.random.RandomState(42); drop = np.flatnonzero(~docs.kept.values)
    sel = np.sort(np.r_[np.flatnonzero(docs.kept.values), rng.choice(drop, a.subsample - int(docs.kept.sum()), replace=False)])
    docs = docs.iloc[sel].reset_index(drop=True); ref = ref[sel] if ref is not None else None
cache = out / f'{a.tag}_{"body" if a.body_only else "docs"}{a.doc_len}{f"_sub{a.subsample}" if a.subsample else ""}.npy'
if cache.exists():
    D = np.load(cache)
else:
    D = embed([(x or '') if a.body_only else (t or '') + '\n' + (x or '') for t, x in zip(docs.title, docs.doc_text)], a.doc_len); np.save(cache, D)
q = pd.read_parquet('data/raw/query.parquet').sort_values('id')
variants = {'vi': q['query'].tolist()}
if a.query_zh:
    variants['zh'] = pd.read_parquet(a.query_zh).sort_values('id')['query_zh'].tolist()
y = docs.kept.astype(int).values
Dt = torch.from_numpy(D).cuda()
if ref is not None:
    sc = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda() @ torch.from_numpy(ref).cuda().T
    best = sc.max(0).values.float().cpu().numpy(); mr = (sc.float().argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    print(f'reference bge-m3 body512 on same docs: AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  kept share@minrank1 {y[mr == 1].mean():.3f}', flush=True)
for name, texts in variants.items():
    Q = embed([f'Instruct: {TASK}\nQuery:{t}' for t in texts], 512); np.save(out / f'{a.tag}_q_{name}.npy', Q)
    sc = torch.from_numpy(Q).cuda() @ Dt.T
    best = sc.max(0).values.float().cpu().numpy(); mr = (sc.float().argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    print(f'{a.tag} {"body" if a.body_only else "doc"}{a.doc_len} query={name}: AUC best {roc_auc_score(y, best):.4f}  AUC -minrank {roc_auc_score(y, -mr):.4f}  '
          f'kept share@minrank1 {y[mr == 1].mean():.3f}', flush=True)
