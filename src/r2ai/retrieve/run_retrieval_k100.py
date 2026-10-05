"""Run the sub02 retrieval once with K_max=100 for the 1,200 test queries and cache the result.

    python -m scripts.run_retrieval_k100 [--limit N] [--out-dir data/runs]

Same model / index / config as sub02 (t256 index, hybrid w=0.7, agg=max, bge-reranker-v2-m3, seed 42):
  tier 1 = exactly the sub02 procedure: top-200 chunks -> top-50 docs -> rerank on the retrieved chunks -> order by
           max reranker score. Ranks 1..50 (or fewer when the top-200 chunks cover fewer docs) are therefore the docs
           sub02 used.
  tier 2 = extension needed for K > 50 (the sub02 pipeline stops at 50): docs not in tier 1, ranked by max hybrid
           score over *all* first-stage candidates (dense top-200 U sparse top-200), top 50 of them, then ordered by
           the max reranker score of their candidate chunks. Always ranked after tier 1.
Outputs
  vi_k100.parquet               query_id, rank (1-based), doc_id (primary id of the group), score (max reranker score), tier
  vi_k100_chunk_scores.parquet  query_id, doc_id, chunk_id, score: reranker score of every answer/body chunk of the
                                final top-50 docs (used for the c2 chunk mode: top-2 chunks by this score)
  vi_k100.meta.json
  vi_k100.candidates.parquet    query_id, n_candidates (after union), n_docs (distinct docs among the candidates)
--candidates exact: first-stage candidates by exact block scan over dense.npy / the sparse CSR mmap cache
(retrieve.exact) instead of FAISS + full CSC; same k=200 per branch, union, re-score, mix, rerank. Default faiss.
--candidates-only (with --candidates exact): stop after the first stage, no reranker, no _partial.pkl. Writes, without
overwriting existing files, vi_cand.docs.parquet (query_id, rank, doc_id, score: every candidate doc ordered by its max
hybrid chunk score; ties keep the first chunk of the stable argsort, as the tier-2 loop), vi_cand.chunks.parquet
(query_id, chunk_id, doc_id, dense, sparse, hybrid for every candidate chunk, in candidate order = ascending chunk_id),
vi_cand.candidates.parquet (as vi_k100.candidates.parquet) and vi_cand.meta.json.
"""
from __future__ import annotations

from r2ai.paths import RAW_DATA_DIR, RUNS_DIR, assert_writable, chunks_file, index_dir, require_inputs, resolve_path

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')

W, AGG = 0.7, 'max'
K_MAX, TIER2_DOCS, CHUNK_DOCS = 100, 50, 50


def peak_rss_gib() -> float:
    """Peak resident set of this process (psutil peak_wset on Windows, ru_maxrss elsewhere)."""
    import psutil
    m = psutil.Process().memory_info()
    if hasattr(m, 'peak_wset'):
        return round(m.peak_wset / 2**30, 3)
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def peak_private_gib() -> float | None:
    """Peak private (committed, non file-backed) memory; Windows only (psutil peak_pagefile). Peak RSS also counts
    resident pages of the memory-mapped index files, which the OS can drop."""
    import psutil
    m = psutil.Process().memory_info()
    return round(m.peak_pagefile / 2**30, 3) if hasattr(m, 'peak_pagefile') else None


CAND_SUFFIXES = ('.docs.parquet', '.chunks.parquet', '.candidates.parquet', '.meta.json')


def write_candidates(out_dir: Path, qs_rows, res: dict, index, minmax, meta: dict, t_start: float) -> None:
    """Full first-stage ranking per query (see --candidates-only); res: row index -> (cand, dense, sparse)."""
    docs = {k: [] for k in ('query_id', 'rank', 'doc_id', 'score')}
    chunks = {k: [] for k in ('query_id', 'chunk_id', 'doc_id', 'dense', 'sparse', 'hybrid')}
    cnt = {k: [] for k in ('query_id', 'n_candidates', 'n_docs')}
    for i, r in enumerate(qs_rows):
        qid = int(r['id'])
        cand, dsc, ssc = res[i]
        h = W * minmax(dsc) + (1 - W) * minmax(ssc)
        dids = np.asarray(index.doc_id[cand], dtype=np.int64)
        seen: dict[int, float] = {}
        for j in np.argsort(-h, kind='stable'):
            seen.setdefault(int(dids[j]), float(h[j]))
        for k, (d, sc) in enumerate(seen.items(), 1):
            for key, v in zip(docs, (qid, k, d, sc)):
                docs[key].append(v)
        for key, v in zip(chunks, ([qid] * len(cand), cand.tolist(), dids.tolist(), dsc.tolist(), ssc.tolist(), h.tolist())):
            chunks[key] += v
        for key, v in zip(cnt, (qid, len(cand), len(seen))):
            cnt[key].append(v)
    paths = [out_dir / f'vi_cand{x}' for x in CAND_SUFFIXES]
    pq.write_table(pa.table({'query_id': pa.array(docs['query_id'], pa.int64()), 'rank': pa.array(docs['rank'], pa.int32()),
                             'doc_id': pa.array(docs['doc_id'], pa.int64()), 'score': pa.array(docs['score'], pa.float32())}), paths[0])
    pq.write_table(pa.table({'query_id': pa.array(chunks['query_id'], pa.int64()), 'chunk_id': pa.array(chunks['chunk_id'], pa.int64()),
                             'doc_id': pa.array(chunks['doc_id'], pa.int64()), 'dense': pa.array(chunks['dense'], pa.float32()),
                             'sparse': pa.array(chunks['sparse'], pa.float32()), 'hybrid': pa.array(chunks['hybrid'], pa.float32())}), paths[1])
    pq.write_table(pa.table({'query_id': pa.array(cnt['query_id'], pa.int64()), 'n_candidates': pa.array(cnt['n_candidates'], pa.int32()),
                             'n_docs': pa.array(cnt['n_docs'], pa.int32())}), paths[2])
    nd = np.asarray(cnt['n_docs'])
    meta |= {'seconds_this_run': round(time.time() - t_start), 'peak_rss_gib': peak_rss_gib(), 'peak_private_gib': peak_private_gib(),
             'docs_per_query': {'min': int(nd.min()), 'p5': float(np.percentile(nd, 5)), 'p50': float(np.percentile(nd, 50)),
                                'max': int(nd.max())}, 'rows_docs': len(docs['doc_id']), 'rows_chunks': len(chunks['chunk_id'])}
    paths[3].write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', type=int, default=256)
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--out-dir', default=str(RUNS_DIR / 'vi-k100'))
    ap.add_argument('--limit', type=int, default=0, help='debug: first N queries only')
    ap.add_argument('--candidates', choices=('faiss', 'exact'), default='faiss')
    ap.add_argument('--block-rows', type=int, default=100_000, help='exact: rows per scanned block')
    ap.add_argument('--device', default='cpu', help='exact: dense block matmul on cpu|cuda')
    ap.add_argument('--sparse-cache', default=None, help='exact: CSR mmap cache dir (default <index>/sparse_mmap)')
    ap.add_argument('--candidates-only', action='store_true',
                    help='exact only: write the full hybrid candidate ranking (vi_cand.*), no rerank')
    a = ap.parse_args(argv)
    if a.candidates_only and a.candidates != 'exact':
        ap.error('--candidates-only needs --candidates exact')
    a.out_dir = str(assert_writable(a.out_dir))
    a.queries = str(resolve_path(a.queries))
    need = ('meta.json', 'dense.npy', 'sparse.npz') + (('faiss.index',) if a.candidates == 'faiss' else ())
    require_inputs(a.queries, chunks_file(a.target), *(index_dir(a.target) / f for f in need))
    if pq.ParquetFile(a.queries).metadata.num_rows == 0 or pq.ParquetFile(chunks_file(a.target)).metadata.num_rows == 0:
        raise ValueError('Empty query/chunk input')
    import torch
    from r2ai.index.bge_m3 import M3Encoder, Reranker
    from r2ai.retrieve.run import Index, minmax, rank_docs, rerank_order, ws

    random.seed(42), np.random.seed(42), torch.manual_seed(42)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / ('_partial_debug.pkl' if a.limit else '_partial.pkl')
    qs_rows = pq.read_table(a.queries).to_pylist()
    if a.limit:
        qs_rows = qs_rows[:a.limit]
    if a.candidates_only:
        cand_files = [assert_writable(out_dir / f'vi_cand{x}') for x in CAND_SUFFIXES]
        if any(f.exists() for f in cand_files):
            raise ValueError(f'refusing to overwrite existing candidate outputs in {out_dir}')
    done: dict[int, dict] = {} if a.candidates_only else (pickle.load(open(partial, 'rb')) if partial.exists() else {})
    print(f'{len(done)} queries restored from {partial.name}', flush=True)

    t_start = time.time()
    index = Index(a.target, candidates=a.candidates, sparse_cache=a.sparse_cache)
    enc = M3Encoder(max_len=512)
    qd, qsp = [], []
    for b in range(0, len(qs_rows), 32):
        d, s = enc.encode_batch([ws(r['query']) for r in qs_rows[b:b + 32]])
        qd.append(d)
        qsp += s
    qd = np.concatenate(qd)
    del enc
    torch.cuda.empty_cache()
    exact_res, exact_stats = None, {}
    if a.candidates == 'exact':
        from r2ai.retrieve.exact import exact_candidates
        todo = [i for i, r in enumerate(qs_rows) if int(r['id']) not in done]
        res, st = exact_candidates(index.dense, index.sparse, index.doc_id, qd[todo], [qsp[i] for i in todo],
                                   k=200, block_rows=a.block_rows, device=a.device)
        exact_res = dict(zip(todo, res))
        exact_stats = {'dense_seconds': round(st['dense_seconds'], 1), 'sparse_seconds': round(st['sparse_seconds'], 1),
                       'union_rescore_seconds': round(st['union_rescore_seconds'], 1), 'n_vq': st['n_vq'],
                       'block_rows': a.block_rows, 'device': a.device, 'peak_rss_gib_after_candidates': peak_rss_gib(),
                       'peak_private_gib_after_candidates': peak_private_gib(),
                       'sparse_cache': {k: index.sparse_cache_info.get(k) for k in ('nnz', 'shape', 'reused')}}
        print(f'exact candidates: {json.dumps(exact_stats)}', flush=True)
    if a.candidates_only:
        write_candidates(out_dir, qs_rows, exact_res, index, minmax, {
            'queries': len(qs_rows), 'w': W, 'agg': AGG, 'target': a.target, 'seed': 42, 'k_per_branch': 200,
            'candidates': 'exact', 'exact': exact_stats, 'encoder_device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_initialized() else None,
            'index_meta': index.meta}, t_start)
        return 0
    rr = Reranker(max_len=512)
    cstat: dict[int, tuple[int, int]] = {}

    t_q = []
    for i, r in enumerate(qs_rows):
        qid = int(r['id'])
        if qid in done:
            continue
        t0 = time.perf_counter()
        q = ws(r['query'])
        cand, dsc, ssc = exact_res.pop(i) if exact_res is not None else index.candidates(qd[i], qsp[i])
        nd = len(np.unique(index.doc_id[cand]))
        cstat[qid] = (len(cand), nd)
        if nd < K_MAX:
            print(f'WARNING query {qid}: {nd} distinct docs among {len(cand)} candidates (< {K_MAX})', flush=True)
        # ---- tier 1: identical to submission/build.py with --rerank
        ch, hs, ranked, _ = rank_docs(index, cand, dsc, ssc, W, AGG)
        top = set(ranked)
        pairs = [int(c) for c in ch if int(index.doc_id[c]) in top]
        s = dict(zip(pairs, rr.score(q, index.texts(pairs)).tolist()))
        tier1 = rerank_order(index, ch, ranked, s)
        dscore = {}
        for c in pairs:
            d = int(index.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s[c])
        # ---- tier 2: next docs by max hybrid score over all candidates, reranked among themselves
        h = W * minmax(dsc) + (1 - W) * minmax(ssc)
        seen, tier2 = set(tier1), []
        for j in np.argsort(-h, kind='stable'):
            d = int(index.doc_id[cand[j]])
            if d not in seen:
                seen.add(d)
                tier2.append(d)
                if len(tier2) == TIER2_DOCS:
                    break
        t2set = set(tier2)
        p2 = [int(c) for c in cand if int(index.doc_id[c]) in t2set]
        s2 = dict(zip(p2, rr.score(q, index.texts(p2)).tolist())) if p2 else {}
        for c in p2:
            d = int(index.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s2[c])
        s.update(s2)
        tier2.sort(key=lambda d: -dscore[d])
        order = [(d, 1) for d in tier1] + [(d, 2) for d in tier2]
        order = order[:K_MAX]
        # ---- reranker scores of every answer/body chunk of the final top-50 docs (c2 mode)
        cscore = {}
        for d, _ in order[:CHUNK_DOCS]:
            cs = [int(c) for c in index.chunks_of(d) if index.submittable[c]]
            new = [c for c in cs if c not in s]
            if new:
                s.update(zip(new, rr.score(q, index.texts(new)).tolist()))
            cscore[d] = [(c, float(s[c])) for c in cs]
        done[qid] = {'docs': [(d, float(dscore[d]), t) for d, t in order], 'chunks': cscore}
        t_q.append(time.perf_counter() - t0)
        if len(done) % 25 == 0:
            print(f'{len(done)}/{len(qs_rows)} {t_q[-1]:.2f}s (mean {np.mean(t_q):.2f}s)', flush=True)
        if len(done) % 100 == 0:
            pickle.dump(done, open(partial, 'wb'))

    pickle.dump(done, open(partial, 'wb'))
    rows = {'query_id': [], 'rank': [], 'doc_id': [], 'score': [], 'tier': []}
    crow = {'query_id': [], 'doc_id': [], 'chunk_id': [], 'score': []}
    for r in qs_rows:
        qid = int(r['id'])
        for k, (d, sc, t) in enumerate(done[qid]['docs'], 1):
            for key, v in zip(rows, (qid, k, d, sc, t)):
                rows[key].append(v)
        for d, lst in done[qid]['chunks'].items():
            for c, sc in lst:
                for key, v in zip(crow, (qid, d, c, sc)):
                    crow[key].append(v)
    tag = '_debug' if a.limit else ''
    pq.write_table(pa.table({'query_id': pa.array(rows['query_id'], pa.int64()), 'rank': pa.array(rows['rank'], pa.int32()),
                             'doc_id': pa.array(rows['doc_id'], pa.int64()), 'score': pa.array(rows['score'], pa.float32()),
                             'tier': pa.array(rows['tier'], pa.int8())}), out_dir / f'vi_k100{tag}.parquet')
    pq.write_table(pa.table({'query_id': pa.array(crow['query_id'], pa.int64()), 'doc_id': pa.array(crow['doc_id'], pa.int64()),
                             'chunk_id': pa.array(crow['chunk_id'], pa.int64()), 'score': pa.array(crow['score'], pa.float32())}),
                   out_dir / f'vi_k100_chunk_scores{tag}.parquet')
    cq = [int(r['id']) for r in qs_rows if int(r['id']) in cstat]
    pq.write_table(pa.table({'query_id': pa.array(cq, pa.int64()), 'n_candidates': pa.array([cstat[q][0] for q in cq], pa.int32()),
                             'n_docs': pa.array([cstat[q][1] for q in cq], pa.int32())}), out_dir / f'vi_k100.candidates{tag}.parquet')
    nd_all = np.array([cstat[q][1] for q in cq]) if cq else np.zeros(1)
    n_docs = [len(done[int(r['id'])]['docs']) for r in qs_rows]
    meta = {'queries': len(qs_rows), 'w': W, 'agg': AGG, 'target': a.target, 'seed': 42, 'k_max': K_MAX,
            'docs_per_query': {'min': int(min(n_docs)), 'p50': float(np.percentile(n_docs, 50)), 'max': int(max(n_docs)),
                               'n_lt_100': int(sum(n < K_MAX for n in n_docs)), 'n_lt_50': int(sum(n < 50 for n in n_docs))},
            'seconds_this_run': round(time.time() - t_start), 'seconds_per_query_mean_this_run': round(float(np.mean(t_q)), 3) if t_q else None,
            'candidates': a.candidates, 'exact': exact_stats,
            'distinct_docs_before_rerank': {'queries_this_run': len(cq), 'min': int(nd_all.min()), 'p5': float(np.percentile(nd_all, 5)),
                                            'p50': float(np.percentile(nd_all, 50)), 'n_lt_100': int((nd_all < K_MAX).sum())},
            'peak_rss_gib': peak_rss_gib(), 'peak_private_gib': peak_private_gib(),
            'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_initialized() else None,
            'index_meta': index.meta}
    (out_dir / f'vi_k100{tag}.meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
