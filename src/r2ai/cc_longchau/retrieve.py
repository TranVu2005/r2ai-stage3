"""1,200 queries over the CC longchau segment: first stage (CPU, exact) then rerank (GPU), one file per query (resumable).

  python -m r2ai.cc_longchau.retrieve first-stage
  python -m r2ai.cc_longchau.retrieve rerank

Same rule as the zh-full run (out/runs/zh-full-ingest/retrieval-b001): dense top-200 U sparse top-200 per query, the union
re-scored on both branches (hybrid 0.7/0.3 min-max), bge-reranker-v2-m3 (fp16, max_len 512, batch 32, seed 42) scores every
candidate chunk, a branch keeps its first-stage top-150 docs scored by max chunk logit. Query vectors are the cached bge-m3
vectors (sha256-checked against query.parquet). Output: <run>/retrieval/{candidates.pkl,rerank/q<id>.json,summary}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from r2ai.index.segments import (load_segment, require_scan_memory, scan_memory_budget, scan_topk, take_segment_texts,
                                 _validate_segments)
from r2ai.paths import QUERY_FILE, RUNS_DIR
from r2ai.retrieve.exact import QuerySparse, doc_aligned_bounds, sparse_scores
from r2ai.retrieve.run import minmax, ws
from r2ai.zh_full_ingest.common import require_gpu_idle

from .build import ensure_segment, zroots
from .common import Roots, atomic_json, log_event, sha256_file

K, BLOCK_ROWS, QUERY_BATCH, TOP_DOCS = 200, 16384, 64, 150
QUERY_CACHE = RUNS_DIR / 'zh-sample' / 'retrieval' / 'queries.pkl'
SAMPLE_CFG = RUNS_DIR / 'zh-sample' / 'retrieval' / 'config.json'


def query_vectors():
    cfg = json.loads(SAMPLE_CFG.read_text('utf-8'))
    if cfg['query_sha256'] != sha256_file(QUERY_FILE):
        raise ValueError('query.parquet differs from the one behind the cached query vectors')
    qd, qsp = pickle.loads(QUERY_CACHE.read_bytes())
    qd = np.asarray(qd)
    if len(qd) != 1200 or len(qsp) != 1200:
        raise ValueError('need 1200 cached query vectors')
    return qd, qsp, cfg['query_sha256']


def open_segment(roots: Roots):
    seg = ensure_segment(zroots(roots))
    from r2ai.zh_full_ingest import ingest as I
    c = I.connect(zroots(roots))
    try:
        status, rows = c.execute('SELECT status, rows FROM segments WHERE name=?', (seg['name'],)).fetchone()
    finally:
        c.close()
    if status != 'active':
        raise ValueError(f'segment {seg["name"]} is {status}, not active')
    s = load_segment(roots.index / seg['name'])
    if len(s) != rows:
        raise ValueError(f'segment rows {len(s)} != manifest {rows}')
    return seg['name'], s


def first_stage(roots: Roots) -> dict:
    t_all = time.perf_counter()
    roots.preflight()
    qd, qsp, qsha = query_vectors()
    name, seg = open_segment(roots)
    _validate_segments([seg])
    b = doc_aligned_bounds(np.asarray(seg.doc_id), BLOCK_ROWS)
    ip = seg.sparse[2]
    vq = QuerySparse(qsp)
    budget = scan_memory_budget(dimension=qd.shape[1], queries=1200, block_rows=int(np.diff(b).max()), query_batch=QUERY_BATCH,
                                max_block_nnz=max(int(ip[int(hi)]) - int(ip[int(lo)]) for lo, hi in zip(b[:-1], b[1:])),
                                query_vocabulary=len(vq.vq), metadata_rows=len(seg), k=K)
    require_scan_memory(budget)
    with threadpool_limits(limits=4, user_api='blas'):
        res = scan_topk([seg], qd, qsp, K, block_rows=BLOCK_ROWS, query_batch=QUERY_BATCH)
    start = seg.global_start
    out = []
    for j in range(1200):
        cand = np.union1d(res['dense'][0][j], res['sparse'][0][j]).astype(np.int64)
        local = cand - start
        q32 = qd[j].astype(np.float32)
        dsc = np.asarray(seg.dense[local], dtype=np.float32) @ q32
        data, indices, indptr = seg.sparse
        ssc = sparse_scores(data, indices, indptr, vq, j, local).astype(np.float32)
        out.append((cand, dsc, ssc))
    path = roots.guard(roots.run / 'retrieval' / 'candidates.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('candidates.tmp.pkl')
    with open(tmp, 'wb') as f:
        pickle.dump({'segment': name, 'global_start': start, 'hybrid': out, 'dense_top': res['dense'], 'sparse_top': res['sparse']}, f, protocol=4)
    os.replace(tmp, path)
    n = np.array([len(o[0]) for o in out])
    summary = {'queries': 1200, 'segment': name, 'rows': len(seg), 'k_per_branch': K, 'query_sha256': qsha,
               'candidates_per_query': {'min': int(n.min()), 'median': float(np.median(n)), 'max': int(n.max())},
               'scan_stats': res['stats'], 'seconds': round(time.perf_counter() - t_all, 1), 'candidates_pkl_sha256': sha256_file(path)}
    atomic_json(roots, roots.run / 'retrieval' / 'first_stage_summary.json', summary)
    return summary


def rerank(roots: Roots) -> dict:
    import torch
    from r2ai.index.bge_m3 import Reranker
    roots.preflight()
    rr_dir = roots.guard(roots.run / 'retrieval' / 'rerank')
    rr_dir.mkdir(parents=True, exist_ok=True)
    first = json.loads((roots.run / 'retrieval' / 'first_stage_summary.json').read_text('utf-8'))
    cand_file = roots.run / 'retrieval' / 'candidates.pkl'
    if sha256_file(cand_file) != first['candidates_pkl_sha256']:
        raise ValueError('candidates.pkl changed since the first stage')
    C = pickle.loads(cand_file.read_bytes())
    name, seg = open_segment(roots)
    if name != C['segment']:
        raise ValueError('segment differs from the first stage')
    qs = pq.read_table(QUERY_FILE).to_pylist()
    cfg = {'first_stage': first, 'top_docs_per_branch': TOP_DOCS, 'hybrid_weight': .7, 'max_len': 512, 'reranker_batch': 32, 'seed': 42,
           'reranker': 'BAAI/bge-reranker-v2-m3'}
    cfg_path = rr_dir / 'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text('utf-8')) != json.loads(json.dumps(cfg)):
        raise ValueError('rerank resume config differs; refusing to mix runs')
    atomic_json(roots, cfg_path, cfg)
    todo = [i for i, q in enumerate(qs) if not (rr_dir / f'q{int(q["id"])}.json').exists()]
    zr = zroots(roots)
    if todo:
        require_gpu_idle(zr)
        np.random.seed(42)
        torch.manual_seed(42)
        rr = Reranker(max_len=512)
        t_run = time.perf_counter()
        doc_ids = np.asarray(seg.doc_id)
        for n_done, i in enumerate(todo, 1):
            if n_done % 25 == 1:
                require_gpu_idle(zr)
            query, qid = qs[i], int(qs[i]['id'])
            t0 = time.perf_counter()
            cand, dsc, ssc = C['hybrid'][i]
            scores = rr.score(ws(query['query']), take_segment_texts([seg], cand))
            pair = {int(c): float(s) for c, s in zip(cand, scores)}
            doc_of = {int(c): int(doc_ids[int(c) - C['global_start']]) for c in cand}
            d_ids = np.asarray(C['dense_top'][0][i], dtype=np.int64)
            d_sc = np.asarray(C['dense_top'][1][i], dtype=np.float32)
            branches = {}
            for branch, (ids, first_scores) in {'hybrid': (cand, .7 * minmax(dsc) + .3 * minmax(ssc)), 'dense': (d_ids, d_sc)}.items():
                first_docs = {}
                for c in np.asarray(ids)[np.argsort(-first_scores, kind='stable')]:
                    first_docs.setdefault(doc_of[int(c)], None)
                selected = set(list(first_docs)[:TOP_DOCS])
                by_doc = {}
                for c in ids:
                    d = doc_of[int(c)]
                    if d in selected:
                        by_doc[d] = max(by_doc.get(d, -float('inf')), pair[int(c)])
                branches[branch] = sorted(by_doc.items(), key=lambda row: -row[1])
            atomic_json(roots, rr_dir / f'q{qid}.json', {'query_id': qid, **branches, 'rr_pairs': len(cand),
                                                         'pair_logits': {str(c): s for c, s in pair.items()},
                                                         'seconds': time.perf_counter() - t0})
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.perf_counter() - t_run
                log_event(roots, 'rerank', {'done': n_done, 'of': len(todo), 'per_query_s': round(el / n_done, 2),
                                            'eta_min': round(el / n_done * (len(todo) - n_done) / 60, 1)})
                print(f'{n_done}/{len(todo)} {el / n_done:.2f}s/q', flush=True)
        del rr
        torch.cuda.empty_cache()
    rows = [json.loads((rr_dir / f'q{int(q["id"])}.json').read_text('utf-8')) for q in qs]
    summary = {'queries': 1200, 'rr_pairs': sum(r['rr_pairs'] for r in rows), 'rr_query_seconds_sum': sum(r['seconds'] for r in rows)}
    atomic_json(roots, rr_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['first-stage', 'rerank'])
    a = ap.parse_args(argv)
    r = (first_stage if a.step == 'first-stage' else rerank)(Roots())
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
