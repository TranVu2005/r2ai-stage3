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
Deep rerank (off by default; the defaults reproduce the outputs above):
  --tier1-docs N (default 50) tier 1 = the top-N docs by max hybrid over the top-200 chunks, reranked on their retrieved
                 chunks as above. For N > 50 only: when the top-200 chunks cover fewer than N docs, the hybrid chunk order
                 is followed past rank 200 until N docs (or the candidates run out); a doc first met there is scored on
                 its candidate chunks up to that point, a doc already in the top 200 keeps exactly its top-200 chunks.
                 The pairs of the top-50 docs are scored in a reranker call of their own and the other tier-1 pairs
                 in a second call: fp16 scores depend on the batch (padding) by a few ULP, and this keeps the
                 top-50 docs' scores identical to a default run. All N docs are then ordered together.
  --tier2-docs M (default 50) tier 2 = the next M docs by max hybrid over all candidates, reranked among themselves.
                 The cache keeps N + M docs per query (default 100).
  --chunk-score-docs C (default 50) every answer/body chunk of the final top-C docs is scored (c2/window/extra modes).
  --pair-scores  also write vi_k100_pairs.parquet: query_id, doc_id, chunk_id, score, tier for every reranked pair.
  --sample S     debug: S queries drawn with random.seed(42) (as measure_candidates); outputs get a _sample suffix.
Resume: finished queries are checkpointed atomically (temp file + os.replace) every --checkpoint-every queries and on
Ctrl+C into <out-dir>/_partial[_debug|_sample].pkl; a rerun with the same arguments skips them. The run config is kept
next to it (.config.json); a rerun with a different config is refused.
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
TIER1_DOCS, TOP_CHUNKS = 50, 200


def tier1_pairs(index, cand, dsc, ssc, n_docs: int = TIER1_DOCS):
    """Tier-1 docs (hybrid order) and the chunk ids to rerank for them (see --tier1-docs)."""
    from r2ai.retrieve.run import minmax, rank_docs
    ch, _, ranked, _ = rank_docs(index, cand, dsc, ssc, W, AGG, top_docs=n_docs)
    top = set(ranked)
    pairs = [int(c) for c in ch if int(index.doc_id[c]) in top]
    if n_docs > TIER1_DOCS and len(ranked) < n_docs:
        ranked, in200 = list(ranked), {int(index.doc_id[c]) for c in ch}
        h = W * minmax(dsc) + (1 - W) * minmax(ssc)
        for j in np.argsort(-h, kind='stable')[TOP_CHUNKS:]:
            d = int(index.doc_id[cand[j]])
            if d in in200:
                continue
            if d not in top:
                ranked.append(d)
                top.add(d)
            pairs.append(int(cand[j]))
            if len(ranked) == n_docs:
                break
    return ranked, pairs


def take_lock(lock: Path) -> None:
    """One run per checkpoint: refuse while the pid in `lock` is a live run_retrieval_k100 process; released at exit."""
    import atexit
    import psutil
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding='utf-8').strip())
            alive = psutil.pid_exists(pid) and 'run_retrieval_k100' in ' '.join(psutil.Process(pid).cmdline())
        except (ValueError, psutil.Error):
            alive = False
        if alive and pid != os.getpid():
            raise RuntimeError(f'{lock}: run_retrieval_k100 pid {pid} is already using this --out-dir')
    lock.write_text(str(os.getpid()), encoding='utf-8')

    def release():
        try:
            if lock.read_text(encoding='utf-8').strip() == str(os.getpid()):
                lock.unlink()
        except OSError:
            pass
    atexit.register(release)


def save_atomic(obj, path: Path) -> None:
    tmp = assert_writable(path.with_name(path.name + '.tmp'))
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    ap.add_argument('--tier1-docs', type=int, default=TIER1_DOCS, help='docs reranked together in tier 1 (default 50)')
    ap.add_argument('--tier2-docs', type=int, default=TIER2_DOCS, help='next docs by hybrid, reranked among themselves')
    ap.add_argument('--chunk-score-docs', type=int, default=CHUNK_DOCS, help='score every answer/body chunk of the top-C docs')
    ap.add_argument('--pair-scores', action='store_true', help='also write vi_k100_pairs.parquet (every reranked pair)')
    ap.add_argument('--sample', type=int, default=0, help='debug: S queries drawn with random.seed(42)')
    ap.add_argument('--checkpoint-every', type=int, default=25, help='queries between atomic checkpoints')
    a = ap.parse_args(argv)
    if a.tier1_docs < 1 or a.tier2_docs < 0 or a.chunk_score_docs < 0 or a.checkpoint_every < 1:
        ap.error('need --tier1-docs >= 1, --tier2-docs >= 0, --chunk-score-docs >= 0, --checkpoint-every >= 1')
    if a.limit and a.sample:
        ap.error('--limit and --sample exclude each other')
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
    tag = '_debug' if a.limit else '_sample' if a.sample else ''
    partial = assert_writable(out_dir / f'_partial{tag}.pkl')
    k_max = a.tier1_docs + a.tier2_docs
    qs_rows = pq.read_table(a.queries).to_pylist()
    if a.limit:
        qs_rows = qs_rows[:a.limit]
    if a.sample:
        qs_rows = [qs_rows[i] for i in sorted(random.sample(range(len(qs_rows)), a.sample))]
    run_cfg = {'tier1_docs': a.tier1_docs, 'tier2_docs': a.tier2_docs, 'chunk_score_docs': a.chunk_score_docs,
               'pair_scores': a.pair_scores, 'candidates': a.candidates, 'target': a.target, 'queries': a.queries,
               'query_ids': [int(r['id']) for r in qs_rows]}
    cfg_path = assert_writable(partial.with_name(partial.name + '.config.json'))
    if a.candidates_only:
        cand_files = [assert_writable(out_dir / f'vi_cand{x}') for x in CAND_SUFFIXES]
        if any(f.exists() for f in cand_files):
            raise ValueError(f'refusing to overwrite existing candidate outputs in {out_dir}')
    if not a.candidates_only:
        take_lock(assert_writable(partial.with_name(partial.name + '.lock')))
    done: dict[int, dict] = {} if a.candidates_only else (pickle.load(open(partial, 'rb')) if partial.exists() else {})
    if not a.candidates_only:
        if cfg_path.exists() and json.loads(cfg_path.read_text(encoding='utf-8')) != run_cfg:
            raise ValueError(f'{cfg_path} holds a different run config; use another --out-dir')
        if done and not cfg_path.exists() and k_max != K_MAX:
            raise ValueError(f'{partial} has no run config (default tiers); use another --out-dir')
        cfg_path.write_text(json.dumps(run_cfg), encoding='utf-8')
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

    t_q, n_pairs = [], []

    def one_query(i: int, r: dict) -> None:
        qid = int(r['id'])
        t0 = time.perf_counter()
        q = ws(r['query'])
        cand, dsc, ssc = exact_res.pop(i) if exact_res is not None else index.candidates(qd[i], qsp[i])
        nd = len(np.unique(index.doc_id[cand]))
        cstat[qid] = (len(cand), nd)
        if nd < k_max:
            print(f'WARNING query {qid}: {nd} distinct docs among {len(cand)} candidates (< {k_max})', flush=True)
        # ---- tier 1: identical to submission/build.py with --rerank (default --tier1-docs 50)
        ranked, pairs = tier1_pairs(index, cand, dsc, ssc, a.tier1_docs)
        first = set(ranked[:TIER1_DOCS])
        p_a = [c for c in pairs if int(index.doc_id[c]) in first]          # == pairs when tier1_docs <= 50
        p_b = [c for c in pairs if int(index.doc_id[c]) not in first]
        # the default top-50 pair list is scored in a call of its own: same fp16 batches, so the same scores as a default run
        s = dict(zip(p_a, rr.score(q, index.texts(p_a)).tolist()))
        if p_b:
            s.update(zip(p_b, rr.score(q, index.texts(p_b)).tolist()))
        tier1 = rerank_order(index, pairs, ranked, s)
        dscore = {}
        for c in pairs:
            d = int(index.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s[c])
        # ---- tier 2: next docs by max hybrid score over all candidates, reranked among themselves
        h = W * minmax(dsc) + (1 - W) * minmax(ssc)
        seen, tier2 = set(tier1), []
        for j in np.argsort(-h, kind='stable'):
            if len(tier2) >= a.tier2_docs:
                break
            d = int(index.doc_id[cand[j]])
            if d not in seen:
                seen.add(d)
                tier2.append(d)
        t2set = set(tier2)
        p2 = [int(c) for c in cand if int(index.doc_id[c]) in t2set]
        s2 = dict(zip(p2, rr.score(q, index.texts(p2)).tolist())) if p2 else {}
        for c in p2:
            d = int(index.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s2[c])
        s.update(s2)
        tier2.sort(key=lambda d: -dscore[d])
        order = [(d, 1) for d in tier1] + [(d, 2) for d in tier2]
        order = order[:k_max]
        # ---- reranker scores of every answer/body chunk of the final top-C docs (c2 mode)
        cscore, n_new = {}, 0
        for d, _ in order[:a.chunk_score_docs]:
            cs = [int(c) for c in index.chunks_of(d) if index.submittable[c]]
            new = [c for c in cs if c not in s]
            if new:
                s.update(zip(new, rr.score(q, index.texts(new)).tolist()))
                n_new += len(new)
            cscore[d] = [(c, float(s[c])) for c in cs]
        done[qid] = {'docs': [(d, float(dscore[d]), t) for d, t in order], 'chunks': cscore, 'n_cand': (len(cand), nd)}
        if a.pair_scores:
            done[qid]['pairs'] = [(int(index.doc_id[c]), c, float(s[c]), 1) for c in pairs] +                 [(int(index.doc_id[c]), c, float(s[c]), 2) for c in p2]
        t_q.append(time.perf_counter() - t0)
        n_pairs.append((len(pairs), len(p2), n_new))

    n_todo = sum(int(r['id']) not in done for r in qs_rows)
    try:
        for i, r in enumerate(qs_rows):
            if int(r['id']) in done:
                continue
            one_query(i, r)
            if len(t_q) % a.checkpoint_every == 0:
                save_atomic(done, partial)
                eta = float(np.mean(t_q)) * (n_todo - len(t_q))
                print(f'{len(done)}/{len(qs_rows)} {t_q[-1]:.2f}s (mean {np.mean(t_q):.2f}s, ETA {eta / 3600:.2f} h, '
                      f'peak VRAM {torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_initialized() else 0:.0f} MiB)', flush=True)
    except KeyboardInterrupt:
        save_atomic(done, partial)
        print(f'interrupted: {len(done)}/{len(qs_rows)} queries saved to {partial}; rerun the same command to resume', flush=True)
        return 130

    save_atomic(done, partial)
    for qid, v in done.items():
        if 'n_cand' in v:
            cstat.setdefault(qid, v['n_cand'])
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
    pq.write_table(pa.table({'query_id': pa.array(rows['query_id'], pa.int64()), 'rank': pa.array(rows['rank'], pa.int32()),
                             'doc_id': pa.array(rows['doc_id'], pa.int64()), 'score': pa.array(rows['score'], pa.float32()),
                             'tier': pa.array(rows['tier'], pa.int8())}), out_dir / f'vi_k100{tag}.parquet')
    pq.write_table(pa.table({'query_id': pa.array(crow['query_id'], pa.int64()), 'doc_id': pa.array(crow['doc_id'], pa.int64()),
                             'chunk_id': pa.array(crow['chunk_id'], pa.int64()), 'score': pa.array(crow['score'], pa.float32())}),
                   out_dir / f'vi_k100_chunk_scores{tag}.parquet')
    if a.pair_scores:
        prow = {'query_id': [], 'doc_id': [], 'chunk_id': [], 'score': [], 'tier': []}
        for r in qs_rows:
            qid = int(r['id'])
            for x in done[qid]['pairs']:
                for key, v in zip(prow, (qid, *x)):
                    prow[key].append(v)
        pq.write_table(pa.table({'query_id': pa.array(prow['query_id'], pa.int64()), 'doc_id': pa.array(prow['doc_id'], pa.int64()),
                                 'chunk_id': pa.array(prow['chunk_id'], pa.int64()), 'score': pa.array(prow['score'], pa.float32()),
                                 'tier': pa.array(prow['tier'], pa.int8())}), out_dir / f'vi_k100_pairs{tag}.parquet')
    cq = [int(r['id']) for r in qs_rows if int(r['id']) in cstat]
    pq.write_table(pa.table({'query_id': pa.array(cq, pa.int64()), 'n_candidates': pa.array([cstat[q][0] for q in cq], pa.int32()),
                             'n_docs': pa.array([cstat[q][1] for q in cq], pa.int32())}), out_dir / f'vi_k100.candidates{tag}.parquet')
    nd_all = np.array([cstat[q][1] for q in cq]) if cq else np.zeros(1)
    n_docs = [len(done[int(r['id'])]['docs']) for r in qs_rows]
    meta = {'queries': len(qs_rows), 'w': W, 'agg': AGG, 'target': a.target, 'seed': 42, 'k_max': k_max,
            'docs_per_query': {'min': int(min(n_docs)), 'p50': float(np.percentile(n_docs, 50)), 'max': int(max(n_docs)),
                               'n_lt_100': int(sum(n < K_MAX for n in n_docs)), 'n_lt_50': int(sum(n < 50 for n in n_docs))},
            'seconds_this_run': round(time.time() - t_start), 'seconds_per_query_mean_this_run': round(float(np.mean(t_q)), 3) if t_q else None,
            'candidates': a.candidates, 'exact': exact_stats,
            'distinct_docs_before_rerank': {'queries_this_run': len(cq), 'min': int(nd_all.min()), 'p5': float(np.percentile(nd_all, 5)),
                                            'p50': float(np.percentile(nd_all, 50)), 'n_lt_100': int((nd_all < K_MAX).sum())},
            'peak_rss_gib': peak_rss_gib(), 'peak_private_gib': peak_private_gib(),
            'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_initialized() else None,
            'index_meta': index.meta}
    if (a.tier1_docs, a.tier2_docs, a.chunk_score_docs) != (TIER1_DOCS, TIER2_DOCS, CHUNK_DOCS) or a.pair_scores or a.sample:
        npairs = np.array(n_pairs) if n_pairs else np.zeros((1, 3))
        meta |= {'tier1_docs': a.tier1_docs, 'tier2_docs': a.tier2_docs, 'chunk_score_docs': a.chunk_score_docs,
                 'sample': a.sample, 'query_ids': run_cfg['query_ids'] if a.sample else None,
                 'n_lt_k_max': int(sum(n < k_max for n in n_docs)),
                 'rerank_pairs_per_query_mean_this_run': {'tier1': round(float(npairs[:, 0].mean()), 2),
                                                          'tier2': round(float(npairs[:, 1].mean()), 2),
                                                          'chunk_scores': round(float(npairs[:, 2].mean()), 2)},
                 'seconds_per_query_p95_this_run': round(float(np.percentile(t_q, 95)), 3) if t_q else None}
    (out_dir / f'vi_k100{tag}.meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
