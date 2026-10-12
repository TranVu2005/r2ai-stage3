"""1,200 queries over the ACTIVE CC longchau segments: first stage (CPU, exact) then rerank (GPU), one file per query (resumable).

  python -m r2ai.cc_longchau.retrieve first-stage [--run NAME]
  python -m r2ai.cc_longchau.retrieve rerank [--run NAME]

Same rule as the zh-full run (out/runs/zh-full-ingest/retrieval-b001): dense top-200 U sparse top-200 per query, the union
re-scored on both branches (hybrid 0.7/0.3 min-max), bge-reranker-v2-m3 (fp16, max_len 512, batch 32, seed 42) scores every
candidate chunk, a branch keeps its first-stage top-150 docs scored by max chunk logit. Query vectors are the cached bge-m3
vectors (sha256-checked against query.parquet).

A run covers every ACTIVE segment (batch). Run `retrieval` = the first segment alone (the cc-A run); with more segments the
default is `retrieval_<n>seg`. A later run never overwrites an earlier one and reuses its reranker logits: a (query, global
chunk id) pair that an earlier run already scored is not scored again (chunk ids are global and immutable), so only the pairs
that are new in the larger candidate set reach the GPU. Output: <run dir>/retrieval*/{candidates.pkl,rerank/q<id>.json}.
"""
from __future__ import annotations

import argparse
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

from .build import active_segments, zroots
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


def open_segments(roots: Roots):
    """[(name, Segment)] of every active segment, in global chunk id order; row counts checked against the manifest."""
    out = []
    for s in active_segments(zroots(roots)):
        seg = load_segment(roots.index / s['name'])
        if len(seg) != s['rows']:
            raise ValueError(f'segment {s["name"]} rows {len(seg)} != manifest {s["rows"]}')
        out.append((s['name'], seg))
    if not out:
        raise ValueError('no active segment')
    return out


def default_run(roots: Roots) -> str:
    n = len(active_segments(zroots(roots)))
    return 'retrieval' if n == 1 else f'retrieval_{n}seg'


def run_dir(roots: Roots, run: str):
    return roots.guard(roots.run / run)


def first_stage(roots: Roots, run: str | None = None) -> dict:
    t_all = time.perf_counter()
    roots.preflight()
    run = run or default_run(roots)
    qd, qsp, qsha = query_vectors()
    named = open_segments(roots)
    names, segments = [n for n, _ in named], [s for _, s in named]
    _validate_segments(segments)
    max_rows = max_nnz = 0
    for seg in segments:
        b = doc_aligned_bounds(np.asarray(seg.doc_id), BLOCK_ROWS)
        ip = seg.sparse[2]
        max_rows = max(max_rows, int(np.diff(b).max()))
        max_nnz = max(max_nnz, max(int(ip[int(hi)]) - int(ip[int(lo)]) for lo, hi in zip(b[:-1], b[1:])))
    vq = QuerySparse(qsp)
    budget = scan_memory_budget(dimension=qd.shape[1], queries=1200, block_rows=max_rows, query_batch=QUERY_BATCH, max_block_nnz=max_nnz,
                                query_vocabulary=len(vq.vq), metadata_rows=sum(len(s) for s in segments), k=K)
    require_scan_memory(budget)
    with threadpool_limits(limits=4, user_api='blas'):
        res = scan_topk(segments, qd, qsp, K, block_rows=BLOCK_ROWS, query_batch=QUERY_BATCH)
    starts = np.array([s.global_start for s in segments], dtype=np.int64)
    out = []
    for j in range(1200):
        cand = np.union1d(res['dense'][0][j], res['sparse'][0][j]).astype(np.int64)
        dsc = np.zeros(len(cand), np.float32)
        ssc = np.zeros(len(cand), np.float32)
        owner = np.searchsorted(starts, cand, side='right') - 1
        q32 = qd[j].astype(np.float32)
        for si in np.unique(owner):
            m = owner == si
            local = cand[m] - starts[si]
            seg = segments[si]
            dsc[m] = np.asarray(seg.dense[local], dtype=np.float32) @ q32
            data, indices, indptr = seg.sparse
            ssc[m] = sparse_scores(data, indices, indptr, vq, j, local)
        out.append((cand, dsc, ssc))
    path = roots.guard(roots.run / run / 'candidates.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('candidates.tmp.pkl')
    with open(tmp, 'wb') as f:
        pickle.dump({'segments': names, 'global_starts': starts, 'hybrid': out, 'dense_top': res['dense'], 'sparse_top': res['sparse']}, f,
                    protocol=4)
    os.replace(tmp, path)
    n = np.array([len(o[0]) for o in out])
    summary = {'queries': 1200, 'run': run, 'segments': names, 'rows': sum(len(s) for s in segments), 'k_per_branch': K, 'query_sha256': qsha,
               'candidates_per_query': {'min': int(n.min()), 'median': float(np.median(n)), 'max': int(n.max())},
               'scan_stats': res['stats'], 'seconds': round(time.perf_counter() - t_all, 1), 'candidates_pkl_sha256': sha256_file(path)}
    atomic_json(roots, roots.run / run / 'first_stage_summary.json', summary)
    return summary


def earlier_pair_logits(roots: Roots, run: str, qid: int) -> dict:
    """Logits {global chunk id: logit} that other retrieval runs of this shard already computed for this query."""
    out = {}
    for d in sorted(roots.run.glob('retrieval*')):
        f = d / 'rerank' / f'q{qid}.json'
        if d.name != run and f.exists():
            out.update({int(k): v for k, v in json.loads(f.read_text('utf-8')).get('pair_logits', {}).items()})
    return out


def rerank(roots: Roots, run: str | None = None) -> dict:
    import torch
    from r2ai.index.bge_m3 import Reranker
    roots.preflight()
    run = run or default_run(roots)
    rr_dir = roots.guard(roots.run / run / 'rerank')
    rr_dir.mkdir(parents=True, exist_ok=True)
    first = json.loads((roots.run / run / 'first_stage_summary.json').read_text('utf-8'))
    cand_file = roots.run / run / 'candidates.pkl'
    if sha256_file(cand_file) != first['candidates_pkl_sha256']:
        raise ValueError('candidates.pkl changed since the first stage')
    C = pickle.loads(cand_file.read_bytes())
    named = open_segments(roots)
    if [n for n, _ in named] != C['segments']:
        raise ValueError('active segments differ from the first stage')
    segments = [s for _, s in named]
    starts = np.asarray(C['global_starts'], dtype=np.int64)
    qs = pq.read_table(QUERY_FILE).to_pylist()
    cfg = {'first_stage': first, 'top_docs_per_branch': TOP_DOCS, 'hybrid_weight': .7, 'max_len': 512, 'reranker_batch': 32, 'seed': 42,
           'reranker': 'BAAI/bge-reranker-v2-m3'}
    cfg_path = rr_dir / 'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text('utf-8')) != json.loads(json.dumps(cfg)):
        raise ValueError('rerank resume config differs; refusing to mix runs')
    atomic_json(roots, cfg_path, cfg)
    todo = [i for i, q in enumerate(qs) if not (rr_dir / f'q{int(q["id"])}.json').exists()]
    zr = zroots(roots)
    new_pairs = reused_pairs = 0
    if todo:
        require_gpu_idle(zr)
        np.random.seed(42)
        torch.manual_seed(42)
        rr = Reranker(max_len=512)
        t_run = time.perf_counter()
        doc_arrays = [np.asarray(s.doc_id) for s in segments]
        for n_done, i in enumerate(todo, 1):
            if n_done % 25 == 1:
                require_gpu_idle(zr)
            query, qid = qs[i], int(qs[i]['id'])
            t0 = time.perf_counter()
            cand, dsc, ssc = C['hybrid'][i]
            cache = earlier_pair_logits(roots, run, qid)
            missing = np.array([c for c in cand if int(c) not in cache], dtype=np.int64)
            pair = {int(c): cache[int(c)] for c in cand if int(c) in cache}
            if len(missing):
                scores = rr.score(ws(query['query']), take_segment_texts(segments, missing))
                pair.update({int(c): float(sc) for c, sc in zip(missing, scores)})
            new_pairs += len(missing)
            reused_pairs += len(cand) - len(missing)
            owner = np.searchsorted(starts, cand, side='right') - 1
            doc_of = {}
            for c, o in zip(cand, owner):
                doc_of[int(c)] = int(doc_arrays[o][int(c) - starts[o]])
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
            atomic_json(roots, rr_dir / f'q{qid}.json', {'query_id': qid, **branches, 'rr_pairs': len(cand), 'pairs_new': int(len(missing)),
                                                         'pair_logits': {str(c): s for c, s in pair.items()},
                                                         'seconds': time.perf_counter() - t0})
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.perf_counter() - t_run
                log_event(roots, 'rerank', {'run': run, 'done': n_done, 'of': len(todo), 'per_query_s': round(el / n_done, 2),
                                            'eta_min': round(el / n_done * (len(todo) - n_done) / 60, 1)})
                print(f'{n_done}/{len(todo)} {el / n_done:.2f}s/q new_pairs={new_pairs} reused={reused_pairs}', flush=True)
        del rr
        torch.cuda.empty_cache()
    rows = [json.loads((rr_dir / f'q{int(q["id"])}.json').read_text('utf-8')) for q in qs]
    summary = {'run': run, 'queries': 1200, 'rr_pairs': sum(r['rr_pairs'] for r in rows), 'pairs_new': sum(r.get('pairs_new', r['rr_pairs']) for r in rows),
               'rr_query_seconds_sum': sum(r['seconds'] for r in rows)}
    atomic_json(roots, rr_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['first-stage', 'rerank'])
    ap.add_argument('--run', default=None, help='run name under out/runs/cc-longchau (default: retrieval / retrieval_<n>seg)')
    a = ap.parse_args(argv)
    r = (first_stage if a.step == 'first-stage' else rerank)(Roots(), a.run)
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
