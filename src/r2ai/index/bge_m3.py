"""BGE-M3 dense + sparse encoder (BAAI/bge-m3, local HF cache only), and bge-reranker-v2-m3.

Re-implements what FlagEmbedding's BGEM3FlagModel computes, with plain transformers:
  dense  = L2-normalised hidden state of [CLS]
  sparse = relu(sparse_linear(hidden)) per token; per token id the max weight, special tokens (cls/eos/pad/unk) dropped
  lexical score(q, d) = sum over shared token ids of w_q * w_d
"""
from __future__ import annotations

from r2ai.paths import ROOT

import os
from pathlib import Path

import numpy as np
import torch

M3 = 'BAAI/bge-m3'
RERANKER = 'BAAI/bge-reranker-v2-m3'


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, local_files_only=True))


class M3Encoder:
    def __init__(self, device: str = 'cuda', fp16: bool = True, max_len: int = 512):
        from transformers import AutoModel, AutoTokenizer
        path = _snapshot(M3)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16 if fp16 else torch.float32).to(device).eval()
        self.sparse = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.sparse.load_state_dict(torch.load(path / 'sparse_linear.pt', map_location='cpu'))
        self.sparse = self.sparse.to(device, dtype=self.model.dtype).eval()
        self.device, self.max_len = device, max_len
        self.unused = torch.tensor(sorted({self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id,
                                           self.tok.unk_token_id}), device=device)
        self.vocab = len(self.tok)

    @torch.inference_mode()
    def encode_batch(self, texts: list[str]):
        """-> dense float16 [B, 1024] (numpy), list of (token_ids int32, weights float16) per text."""
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to(self.device)
        h = self.model(**enc).last_hidden_state
        dense = torch.nn.functional.normalize(h[:, 0].float(), dim=-1).half().cpu().numpy()
        w = torch.relu(self.sparse(h)).squeeze(-1).float()                    # [B, L]
        ids = enc['input_ids']
        w = w.masked_fill(torch.isin(ids, self.unused) | (enc['attention_mask'] == 0), 0)
        out = []
        ids_c, w_c = ids.cpu().numpy(), w.cpu().numpy()
        for i in range(len(texts)):
            m = w_c[i] > 0
            t, v = ids_c[i][m], w_c[i][m]
            if len(t):
                order = np.lexsort((-v, t))            # per token id keep max weight
                t, v = t[order], v[order]
                first = np.r_[True, t[1:] != t[:-1]]
                t, v = t[first], v[first]
            out.append((t.astype(np.int32), v.astype(np.float16)))
        return dense, out


class Reranker:
    def __init__(self, device: str = 'cuda', fp16: bool = True, max_len: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = _snapshot(RERANKER)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch.float16 if fp16 else torch.float32).to(device).eval()
        self.device, self.max_len = device, max_len

    @torch.inference_mode()
    def score(self, query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        order = np.argsort([-len(p) for p in passages])           # length-sorted batches, less padding
        for b in range(0, len(passages), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok([query] * len(idx), [passages[i] for i in idx], padding=True, truncation='only_second',
                           max_length=self.max_len, return_tensors='pt').to(self.device)
            out.append((idx, self.model(**enc).logits.view(-1).float().cpu().numpy()))
        s = np.empty(len(passages), dtype=np.float32)
        for idx, v in out:
            s[idx] = v
        return s


os.environ.setdefault('HF_HUB_OFFLINE', '1')
