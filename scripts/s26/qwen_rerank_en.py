"""S26 H7: score EN-only (query, best chunk) pairs with Qwen3-Reranker-0.6B (model-card yes/no prompt, BF16).

Prompt/score copied from r2ai.reranker_trial_runner (reranker-trial worktree); only the instruction names English.
Checkpoints every 2,000 pairs to <out>/qwen_scores.part.parquet and resumes from it.
"""
import os
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

INSTRUCTION = 'Given a Vietnamese medical question, judge whether the English passage answers it'
SYSTEM_PROMPT = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
PREFIX = '<|im_start|>system\n' + SYSTEM_PROMPT + '<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

ap = argparse.ArgumentParser()
ap.add_argument('--cand', required=True); ap.add_argument('--out', required=True); ap.add_argument('--model', required=True)
ap.add_argument('--batch-tokens', type=int, default=8000)
a = ap.parse_args()
out = Path(a.out); part = out / 'qwen_scores.part.parquet'
cand = pd.read_parquet(a.cand)
q = pd.read_parquet('data/raw/query.parquet').set_index('id')['query']
ch = pd.read_parquet('data/pubmed_en/chunks.parquet', columns=['text'])['text']
done = pd.read_parquet(part) if part.exists() else pd.DataFrame(columns=['query_id', 'chunk_id', 'qwen'])
key = set(zip(done.query_id, done.chunk_id))
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
no_id, yes_id = tok.convert_tokens_to_ids('no'), tok.convert_tokens_to_ids('yes')
todo = [(int(r.query_id), int(r.chunk_id)) for r in cand.itertuples() if (r.query_id, r.chunk_id) not in key]
ids = tok([f'<Instruct>: {INSTRUCTION}\n<Query>: {q[qi]}\n<Document>: {ch.iat[ci]}' for qi, ci in todo],
          add_special_tokens=False, truncation=False)['input_ids']
order = np.argsort([len(x) for x in ids])
rows, t0, n = [], time.time(), 0
i = 0
while i < len(order):
    L = len(ids[order[min(i + 1, len(order) - 1)]])
    j = i
    while j < len(order) and (j - i + 1) * (len(ids[order[j]]) + len(pre) + len(suf)) <= a.batch_tokens:
        j += 1
    j = max(j, i + 1)
    b = order[i:j]
    enc = tok.pad({'input_ids': [pre + ids[k] + suf for k in b]}, padding=True, return_tensors='pt')
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
        s = torch.log_softmax(torch.stack((lg[:, no_id].float(), lg[:, yes_id].float()), 1), 1)[:, 1].exp().tolist()
    rows += [(todo[k][0], todo[k][1], float(v)) for k, v in zip(b, s)]
    n += len(b); i = j
    if len(rows) >= 2000 or i >= len(order):
        done = pd.concat([done, pd.DataFrame(rows, columns=['query_id', 'chunk_id', 'qwen'])], ignore_index=True)
        done.to_parquet(str(part) + '.tmp', index=False); Path(str(part) + '.tmp').replace(part); rows = []
        print(f'{len(done)}/{len(cand)} {time.time() - t0:.0f}s', flush=True)
assert len(done) == len(cand)
done.to_parquet(out / 'qwen_scores.parquet', index=False)
print('done', time.time() - t0)
