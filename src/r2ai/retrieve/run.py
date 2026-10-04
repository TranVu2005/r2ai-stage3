"""Hybrid retrieval (BGE-M3 dense + sparse) -> doc aggregation -> bge-reranker-v2-m3, evaluated on the pseudo dev set.

    python -m retrieve.run dev  [--target 256] [--dev data/dev/pseudo_vi_v2.parquet]

Per query (text whitespace-normalised, as the doc text):
  1. dense top-200 chunks (FAISS, cosine) and sparse top-200 chunks (lexical weights, exact over all chunks);
     candidates = union, both scores computed for every candidate, min-max normalised per query over the candidates.
  2. chunk score = w*dense + (1-w)*sparse, w in {0.5, 0.7, 1.0}; keep the top-200 chunks.
  3. doc score = max chunk score ("max") or sum of the doc's top-3 chunk scores ("sum3"); keep the top-50 docs.
  4. rerank: bge-reranker-v2-m3 (fp16) on (query, chunk) for the retrieved top-200 chunks that belong to the top-50
     docs; doc score after rerank = max reranker score of its chunks. Reranker scores are cached per
     (query, chunk) and shared across configs, the time per query is measured on the de-duplicated pairs.
--exclude-fields title,question (diagnostic): those chunks are removed from retrieval (FAISS top-600 filtered to 200,
sparse scores zeroed), so the gold doc must be found from its answer/body text; outputs get a _no-title-question suffix.
Evaluation, known-item mode: gold doc = src_doc_ids_group (source page stays in the index).
  * doc Recall@1/5/10/50 before / after rerank, split by type A / B
  * type B chunk F2 (eval.scorer, gold = answer text on src_doc_id): top-2 answer/body chunks of each top-5 doc;
    chunks are ranked by reranker score (after rerank) or hybrid score (before); all answer/body chunks of a top-5
    doc are scored by the reranker for this, not only the retrieved ones.
"""
from __future__ import annotations

from r2ai.paths import LEGACY_DEV_DIR, OUT_DIR, assert_writable, chunks_file, index_dir, require_inputs, resolve_path

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')

WEIGHTS = (0.5, 0.7, 1.0)
AGGS = ('max', 'sum3')
TOP_CHUNKS, TOP_DOCS = 200, 50
SUBMITTABLE = ('answer', 'body')


def ws(s: str) -> str:
    return ' '.join((s or '').split())


class Index:
    def __init__(self, target: int, exclude_fields: tuple[str, ...] = (), candidates: str = 'faiss',
                 sparse_cache: str | Path | None = None):
        """candidates='faiss': FAISS + full CSC (legacy). 'exact': no FAISS/CSC; dense.npy and a CSR mmap cache
        (sparse_cache, default <index>/sparse_mmap) are block-scanned by retrieve.exact.exact_candidates, and the chunk
        text is a memory-mapped Arrow copy (<index>/text.arrow) instead of an in-memory parquet column."""
        d = index_dir(target)
        self.meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        t0 = time.time()
        self.mode = candidates
        if candidates == 'faiss':
            import faiss
            from scipy.sparse import load_npz
            self.faiss = faiss.read_index(str(d / 'faiss.index'))
            self.csc = load_npz(d / 'sparse.npz').tocsc()
        elif candidates == 'exact':
            from r2ai.retrieve.exact import load_sparse_mmap, split_sparse_npz
            assert not exclude_fields, '--exclude-fields is not supported with exact candidates'
            cache = assert_writable(sparse_cache or d / 'sparse_mmap')
            self.sparse_cache_info = split_sparse_npz(d / 'sparse.npz', cache)
            self.sparse = load_sparse_mmap(cache)[:3]
        else:
            raise ValueError(f'unknown candidates mode: {candidates}')
        self.dense = np.load(d / 'dense.npy', mmap_mode='r')
        if candidates == 'exact':
            from r2ai.retrieve.exact import text_mmap
            ch = pq.read_table(chunks_file(target), columns=['doc_id', 'field'])
            self.text, self.text_cache_info = text_mmap(chunks_file(target), assert_writable(d / 'text.arrow'))
        else:
            ch = pq.read_table(chunks_file(target), columns=['doc_id', 'field', 'text'])
            self.text = ch['text']
        self.doc_id = ch['doc_id'].to_numpy()
        self.field = np.array(ch['field'].to_pylist())
        self.submittable = np.isin(self.field, SUBMITTABLE)
        self.allowed = ~np.isin(self.field, list(exclude_fields)) if exclude_fields else None
        order = np.argsort(self.doc_id, kind='stable')               # doc -> chunk ids
        self._doc_sorted, self._chunk_sorted = self.doc_id[order], order
        print(f'index t{target}: {len(self.doc_id)} chunks loaded in {time.time() - t0:.0f}s', flush=True)

    def chunks_of(self, doc: int) -> np.ndarray:
        lo, hi = np.searchsorted(self._doc_sorted, [doc, doc + 1])
        return self._chunk_sorted[lo:hi]

    def texts(self, ids) -> list[str]:
        from r2ai.retrieve.exact import take_texts
        return take_texts(self.text, ids)

    def candidates(self, qd: np.ndarray, qs, k: int = TOP_CHUNKS):
        """-> chunk ids (union of dense/sparse top-k), dense scores, sparse scores."""
        _, di = self.faiss.search(qd[None].astype(np.float32), k if self.allowed is None else 3 * k)
        di = di[0][di[0] >= 0]
        if self.allowed is not None:
            di = di[self.allowed[di]][:k]
        toks, w = qs
        if len(toks):
            sp_all = np.asarray(self.csc[:, toks.astype(np.int64)] @ w.astype(np.float32)).ravel()
            if self.allowed is not None:
                sp_all = np.where(self.allowed, sp_all, 0).astype(np.float32)
            si = np.argpartition(-sp_all, k)[:k] if len(sp_all) > k else np.arange(len(sp_all))
            si = si[sp_all[si] > 0]
        else:
            sp_all, si = None, np.zeros(0, np.int64)
        cand = np.union1d(di, si)
        dsc = np.asarray(self.dense[np.sort(cand)], dtype=np.float32) @ qd.astype(np.float32)   # cand is sorted
        ssc = sp_all[cand] if sp_all is not None else np.zeros(len(cand), np.float32)
        return cand, dsc, ssc


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rank_docs(index: Index, cand, dsc, ssc, w: float, agg: str):
    """-> top chunk ids (<=200, by hybrid), their hybrid scores, top-50 doc ids (ranked), doc scores."""
    h = w * minmax(dsc) + (1 - w) * minmax(ssc)
    top = np.argsort(-h, kind='stable')[:TOP_CHUNKS]
    ch, hs = cand[top], h[top]
    docs: dict[int, list[float]] = {}
    for c, s in zip(ch, hs):
        docs.setdefault(int(index.doc_id[c]), []).append(float(s))   # already in descending order
    score = {d: (v[0] if agg == 'max' else sum(v[:3])) for d, v in docs.items()}
    ranked = sorted(score, key=lambda d: -score[d])[:TOP_DOCS]
    return ch, hs, ranked, score


def rerank_order(index: Index, ch, ranked: list[int], score: dict[int, float]) -> list[int]:
    """Top-50 docs re-sorted by the max reranker score of their retrieved chunks (stable on ties)."""
    top = set(ranked)
    ds: dict[int, float] = {}
    for c in ch:
        d = int(index.doc_id[c])
        if d in top:
            ds[d] = max(ds.get(d, -1e9), score[int(c)])
    return sorted(ranked, key=lambda d: -ds[d])


def recall_at(ranked: list[int], gold: set[int], ks=(1, 5, 10, 50)) -> dict:
    return {f'R@{k}': float(any(d in gold for d in ranked[:k])) for k in ks}


def cmd_dev(a):
    import pyarrow as pa  # noqa: F401
    import torch
    from r2ai.eval.scorer import Tokenizer, chunk_prf, doc_prf
    from r2ai.index.bge_m3 import M3Encoder, Reranker
    out_dir = assert_writable(OUT_DIR / 'retrieval')
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pq.read_table(a.dev).to_pylist()
    excl = tuple(x for x in a.exclude_fields.split(',') if x)
    index = Index(a.target, excl)
    suffix = '_no-' + '-'.join(excl) if excl else ''

    # 1. query encoding
    enc = M3Encoder(max_len=512)
    t0 = time.perf_counter()
    qd, qs = [], []
    for b in range(0, len(dev), 32):
        d, s = enc.encode_batch([ws(r['query']) for r in dev[b:b + 32]])
        qd.append(d)
        qs += s
    qd = np.concatenate(qd)
    t_qenc = (time.perf_counter() - t0) / len(dev)
    del enc
    torch.cuda.empty_cache()

    # 2. first stage, every (w, agg)
    t0 = time.perf_counter()
    cands = [index.candidates(qd[i], qs[i]) for i in range(len(dev))]
    t_cand = (time.perf_counter() - t0) / len(dev)
    first = {}
    for w in WEIGHTS:
        for agg in AGGS:
            first[(w, agg)] = [rank_docs(index, *cands[i], w, agg) for i in range(len(dev))]

    # 3. rerank pairs: retrieved chunks of top-50 docs (any config) + all answer/body chunks of top-5 docs (any config)
    rr = Reranker(max_len=512)
    rr_score: list[dict[int, float]] = []
    t_rr = []
    n_pairs = []
    for i, r in enumerate(dev):
        q = ws(r['query'])
        # pass 1: retrieved chunks of the top-50 docs (union over configs) -> doc order after rerank
        need = set()
        for key in first:
            ch, _, ranked, _ = first[key][i]
            top = set(ranked)
            need.update(int(c) for c in ch if int(index.doc_id[c]) in top)
        ids = sorted(need)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sc = dict(zip(ids, rr.score(q, index.texts(ids), batch_size=a.rerank_batch).tolist()))
        torch.cuda.synchronize()
        t_rr.append(time.perf_counter() - t0)
        n_pairs.append(len(ids))
        # pass 2 (chunk selection only, not timed): every answer/body chunk of the top-5 docs before and after rerank
        need2 = set()
        for key in first:
            ch, _, ranked, _ = first[key][i]
            for order in (ranked, rerank_order(index, ch, ranked, sc)):
                for d in order[:5]:
                    need2.update(int(c) for c in index.chunks_of(d) if index.submittable[c] and int(c) not in sc)
        ids2 = sorted(need2)
        if ids2:
            sc.update(zip(ids2, rr.score(q, index.texts(ids2), batch_size=a.rerank_batch).tolist()))
        rr_score.append(sc)
        if i % 50 == 0:
            print(f'rerank {i}/{len(dev)}: {len(ids)} + {len(ids2)} pairs {t_rr[-1]:.2f}s', flush=True)

    # 4. evaluate
    tok = Tokenizer()
    rows, per_query = [], {}
    for key, res in first.items():
        for reranked in (False, True):
            acc = {'A': [], 'B': []}
            cf2 = []
            for i, r in enumerate(dev):
                ch, hs, ranked, _ = res[i]
                hyb = dict(zip(ch.tolist(), hs.tolist()))
                if reranked:
                    ranked = rerank_order(index, ch, ranked, rr_score[i])
                gold = set(r['src_doc_ids_group'])
                m = recall_at(ranked, gold)
                acc[r['type']].append(m)
                per_query[(key, reranked, r['qid'])] = ranked[:10]
                if r['type'] == 'B':
                    preds = []
                    for d in ranked[:5]:
                        cs = [int(c) for c in index.chunks_of(d) if index.submittable[c]]
                        sc = (lambda c: rr_score[i][c]) if reranked else (lambda c: hyb.get(c, -1e9))
                        cs.sort(key=lambda c: -sc(c))
                        preds += [(d, x) for x in zip(cs[:2], index.texts(cs[:2]))]
                    pc = [(d, tok.encode(t)) for d, (_, t) in preds]
                    gc = [(r['src_doc_id'], tok.encode(r['gold_text']))]
                    cp, cr, cf = chunk_prf(pc, gc)
                    _, _, df = doc_prf(ranked[:5], gold)
                    cf2.append((cp, cr, cf, df))
            row = {'w': key[0], 'agg': key[1], 'rerank': reranked}
            for t in 'AB':
                for k in (1, 5, 10, 50):
                    row[f'{t}_R@{k}'] = round(float(np.mean([m[f'R@{k}'] for m in acc[t]])), 4)
            row['B_chunkP'] = round(float(np.mean([x[0] for x in cf2])), 4)
            row['B_chunkR'] = round(float(np.mean([x[1] for x in cf2])), 4)
            row['B_chunkF2'] = round(float(np.mean([x[2] for x in cf2])), 4)
            row['B_docF2@5'] = round(float(np.mean([x[3] for x in cf2])), 4)
            rows.append(row)
    timing = {'query_encode_s_per_query': round(t_qenc, 4), 'first_stage_s_per_query': round(t_cand, 4),
              'rerank_s_per_query_mean': round(float(np.mean(t_rr)), 3), 'rerank_s_per_query_p95': round(float(np.percentile(t_rr, 95)), 3),
              'rerank_pairs_per_query_mean': round(float(np.mean(n_pairs)), 1),
              'rerank_pairs_per_s': round(float(np.sum(n_pairs) / np.sum(t_rr)), 1),
              'note': 'rerank time = pass 1 only (retrieved chunks of the top-50 docs, union over the 6 first-stage '
                      'configs); measured on the dev queries; chunk-selection pairs (pass 2) are not timed'}
    res = {'target': a.target, 'exclude_fields': list(excl), 'dev': str(a.dev), 'n': {'A': sum(r['type'] == 'A' for r in dev), 'B': sum(r['type'] == 'B' for r in dev)},
           'index_meta': index.meta, 'timing': timing, 'ablation': rows}
    (out_dir / f'ablation_t{a.target}{suffix}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    with open(out_dir / f'dev_cache_t{a.target}{suffix}.pkl', 'wb') as f:
        pickle.dump({'rr_score': rr_score, 'per_query': per_query}, f)
    print(json.dumps(timing, indent=1))
    hdr = ['w', 'agg', 'rerank'] + [f'{t}_R@{k}' for t in 'AB' for k in (1, 5, 10, 50)] + ['B_chunkP', 'B_chunkR', 'B_chunkF2', 'B_docF2@5']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    lines += ['| ' + ' | '.join(str(r[h]) for h in hdr) + ' |' for r in rows]
    (out_dir / f'ablation_t{a.target}{suffix}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('dev')
    p.add_argument('--target', type=int, default=256)
    p.add_argument('--dev', default=str(LEGACY_DEV_DIR / 'pseudo_vi_v2.parquet'))
    p.add_argument('--rerank-batch', type=int, default=32)
    p.add_argument('--exclude-fields', default='', help='diagnostic: drop chunks of these fields from retrieval, e.g. title,question')
    a = ap.parse_args(argv)
    assert_writable(OUT_DIR / 'retrieval')
    a.dev = str(resolve_path(a.dev))
    require_inputs(a.dev, chunks_file(a.target), *(index_dir(a.target) / f for f in ('meta.json','faiss.index','dense.npy','sparse.npz')))
    if pq.ParquetFile(a.dev).metadata.num_rows == 0 or pq.ParquetFile(chunks_file(a.target)).metadata.num_rows == 0:
        raise ValueError('Empty dev/chunk input')
    return cmd_dev(a)


if __name__ == '__main__':
    sys.exit(main())
