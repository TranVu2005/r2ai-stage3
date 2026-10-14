"""Retrieval over the v2-only zh index (S11 step 4): exact CPU first stage, reranker, candidate export, ZG selection.

Same rule as out/runs/zh-full-ingest/retrieval-b001 (zh_sample.evaluate.retrieve): dense top-200 U sparse top-200 chunks per
query from the cached bge-m3 query vectors, union re-scored on both branches, bge-reranker-v2-m3 (fp16, max_len 512, batch 32)
on every union chunk, per branch the first-stage top-150 docs scored by max chunk logit. Reranker logits are cached per
(query, SOURCE global chunk id) in the per-query files, so a later run over more segments only scores the new pairs.
Logits can differ by fp16 padding ULPs between runs (batches are length-sorted per candidate set).

    python -m r2ai.zh_v2.retrieve first-stage
    python -m r2ai.zh_v2.retrieve rerank [--reuse DIR]      # GPU, resumable, one file per query
    python -m r2ai.zh_v2.retrieve export                    # artifacts/cand_zh_v2.parquet (dense branch, top-50 docs)
    python -m r2ai.zh_v2.retrieve zg                        # ZG on v2: tau 0.96, whole doc_ids_group, cap 10
"""
from __future__ import annotations

from r2ai.paths import QUERY_FILE, ROOT, RUNS_DIR, WORK_DATA_DIR, assert_writable, resolve_path
from r2ai.index.segments import (ZH_V2_INDEX_ROOT, load_segment, memory_metrics, require_scan_memory, scan_memory_budget,
                                 scan_topk, take_segment_texts, _validate_segments)
from r2ai.retrieve.exact import QuerySparse, doc_aligned_bounds, sparse_scores

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import index as zi

OUT = RUNS_DIR / 'zh-v2' / 'retrieval'
ARTIFACTS = ROOT / 'artifacts'
SAMPLE_CFG = RUNS_DIR / 'zh-sample' / 'retrieval' / 'config.json'
QUERY_CACHE = RUNS_DIR / 'zh-sample' / 'retrieval' / 'queries.pkl'
K, BLOCK_ROWS, QUERY_BATCH, TOP_DOCS, HYBRID_W = 200, 16384, 64, 150, 0.7
TAU, CAP = 0.96, 10


def guard(path) -> Path:
    p = assert_writable(path)
    if not p.is_relative_to(resolve_path(RUNS_DIR / 'zh-v2')) and not p.is_relative_to(resolve_path(ARTIFACTS)):
        raise ValueError(f'Not a zh-v2 retrieval output: {p}')
    return p


def atomic_json(path, value):
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.tmp'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log(**kw):
    line = json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False, default=str)
    print(line, flush=True)
    path = guard(OUT / 'retrieval.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def minmax(x):
    return (x - x.min()) / (x.max() - x.min()) if len(x) and x.max() > x.min() else np.zeros(len(x), np.float32)


def load_v2_segments():
    """(names, segments sorted by global_start, source_gid(new_gid) mapping parts)."""
    manifest = json.loads(zi.MANIFEST.read_text('utf-8'))
    rows = [r for r in manifest['segments'] if r['rows_v2']]
    segs = sorted(((r['name'], load_segment(ZH_V2_INDEX_ROOT / r['name']), r) for r in rows), key=lambda t: t[1].global_start)
    for name, s, r in segs:
        if len(s) != r['rows_v2']:
            raise ValueError(f'{name}: rows {len(s)} != manifest {r["rows_v2"]}')
    return manifest, [t[0] for t in segs], [t[1] for t in segs]


def source_gids(names, segments, gids: np.ndarray) -> np.ndarray:
    """v2 global chunk ids -> the ids the same chunks have in the source (zh_full_segments) index."""
    out = np.empty(len(gids), np.int64)
    starts = np.array([s.global_start for s in segments], np.int64)
    owner = np.searchsorted(starts, gids, side='right') - 1
    for si in np.unique(owner):
        m = owner == si
        rm_file = zi.CHUNKS_V2 / names[si] / 'row_map.parquet'
        if rm_file.exists():
            rm = pl.read_parquet(rm_file).sort('new_local')['old_local'].to_numpy()
            out[m] = starts[si] + rm[gids[m] - starts[si]]
        else:                                           # segment ingested straight into the v2 index: ids are already source ids
            out[m] = gids[m]
    return out


def doc_groups(names) -> dict:
    """primary doc id -> {'group': [...], 'v2': [...], 'domain', 'segment'} for every doc in the v2 index."""
    out = {}
    for n in names:
        t = pl.read_parquet(zi.CHUNKS_V2 / n / 'docs.parquet')
        v2col = t['doc_ids_v2'].to_list() if 'doc_ids_v2' in t.columns else t['doc_ids_group'].to_list()
        for d, g, v, dom in zip(t['doc_id'].to_list(), t['doc_ids_group'].to_list(), v2col, t['domain'].to_list()):
            if d in out or g[0] != d:
                raise ValueError(f'Duplicate doc_id or group without primary first: {d}')
            out[d] = {'group': g, 'v2': v, 'domain': dom, 'segment': n}
    return out


def first_stage(limit=None):
    t_all = time.perf_counter()
    cfg = json.loads(SAMPLE_CFG.read_text('utf-8'))
    qsha = hashlib.file_digest(open(QUERY_FILE, 'rb'), 'sha256').hexdigest()
    if cfg['query_sha256'] != qsha:
        raise SystemExit(f'query.parquet sha256 differs from the one behind the cached vectors: {qsha}')
    qd, qsp = pickle.loads(QUERY_CACHE.read_bytes())
    qd = np.asarray(qd)
    if len(qd) != 1200 or len(qsp) != 1200:
        raise SystemExit(f'need 1200 cached query vectors, got {len(qd)}/{len(qsp)}')
    manifest, names, segments = load_v2_segments()
    _validate_segments(segments)
    total_rows = sum(len(s) for s in segments)
    max_rows = max_nnz = 0
    for s in segments:
        b = doc_aligned_bounds(np.asarray(s.doc_id), BLOCK_ROWS)
        ip = s.sparse[2]
        max_rows = max(max_rows, int(np.diff(b).max()))
        max_nnz = max(max_nnz, max(int(ip[int(hi)]) - int(ip[int(lo)]) for lo, hi in zip(b[:-1], b[1:])))
    vq = QuerySparse(qsp)
    budget = scan_memory_budget(dimension=qd.shape[1], queries=1200, block_rows=max_rows, query_batch=QUERY_BATCH,
                                max_block_nnz=max_nnz, query_vocabulary=len(vq.vq), metadata_rows=total_rows, k=K)
    pre = require_scan_memory(budget)
    log(step='segments', n=len(segments), rows=total_rows, estimated_peak_gib=round(pre['estimated_peak_private_gib'], 3))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4, user_api='blas'):
        res = scan_topk(segments, qd, qsp, K, block_rows=BLOCK_ROWS, query_batch=QUERY_BATCH)
    log(step='scan_done', **res['stats'], memory=memory_metrics())
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
            s = segments[si]
            dsc[m] = np.asarray(s.dense[local], dtype=np.float32) @ q32
            data, indices, indptr = s.sparse
            ssc[m] = sparse_scores(data, indices, indptr, vq, j, local)
        out.append((cand, dsc, ssc))
    tmp = guard(OUT / 'candidates.tmp.pkl')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as f:
        pickle.dump({'segments': names, 'global_starts': starts, 'hybrid': out, 'dense_top': res['dense'], 'sparse_top': res['sparse']},
                    f, protocol=4)
    os.replace(tmp, guard(OUT / 'candidates.pkl'))
    n_cand = np.array([len(o[0]) for o in out])
    summary = {'queries': 1200, 'segments': len(segments), 'rows': total_rows, 'k_per_branch': K, 'query_sha256': qsha,
               'v2_ids_sha256': manifest['v2_ids_sha256'],
               'candidates_per_query': {'min': int(n_cand.min()), 'median': float(np.median(n_cand)), 'max': int(n_cand.max())},
               'scan_stats': res['stats'], 'total_seconds': round(time.perf_counter() - t_all, 1),
               'note': 'first stage only: no reranker, no relevance labels, no LB'}
    atomic_json(OUT / 'first_stage_summary.json', summary)
    log(step='first_stage_done', total_seconds=summary['total_seconds'])
    return summary


def gpu_idle():
    import subprocess
    from r2ai.zh_sample.index import compute_processes
    text = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
    if 'Processes:' not in text:
        raise RuntimeError('Cannot parse nvidia-smi process table; refusing GPU')
    active = compute_processes(text, os.getpid())
    if active:
        raise RuntimeError('GPU_BUSY: ' + json.dumps(active))


def rerank(reuse: Path | None = None, limit=None):
    import torch
    import pyarrow.parquet as pq
    from r2ai.index.bge_m3 import Reranker
    from r2ai.retrieve.run import ws
    C = pickle.loads((OUT / 'candidates.pkl').read_bytes())
    names, starts = C['segments'], np.asarray(C['global_starts'], dtype=np.int64)
    manifest, mnames, segments = load_v2_segments()
    if mnames != names:
        raise SystemExit('v2 index changed since the first stage; rerun first-stage')
    qs = pq.read_table(QUERY_FILE).to_pylist()
    rr_dir = guard(OUT / 'rerank')
    rr_dir.mkdir(parents=True, exist_ok=True)
    cfg = {'segments': names, 'v2_ids_sha256': manifest['v2_ids_sha256'], 'top_docs_per_branch': TOP_DOCS, 'hybrid_weight': HYBRID_W,
           'max_len': 512, 'reranker_batch': 32, 'seed': 42, 'model': 'BAAI/bge-reranker-v2-m3',
           'candidates_pkl_sha256': hashlib.file_digest(open(OUT / 'candidates.pkl', 'rb'), 'sha256').hexdigest()}
    cfg_path = rr_dir / 'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text('utf-8')) != json.loads(json.dumps(cfg)):
        raise SystemExit('rerank resume config differs; refusing to mix runs')
    atomic_json(cfg_path, cfg)
    reuse_pairs = {}
    if reuse is not None:
        for f in Path(reuse).glob('q*.json'):
            row = json.loads(f.read_text('utf-8'))
            reuse_pairs[row['query_id']] = {int(k): v for k, v in row['pairs'].items()}
    todo = [i for i, q in enumerate(qs) if not (rr_dir / f'q{int(q["id"])}.json').exists()]
    if limit:
        todo = todo[:limit]
    log(step='rerank_start', todo=len(todo), done=1200 - len(todo), reused_queries=len(reuse_pairs))
    if todo:
        gpu_idle()
        np.random.seed(42)
        torch.manual_seed(42)
        rr = Reranker(max_len=512)
        t_run = time.perf_counter()
        rowmaps = {}
        for n_done, i in enumerate(todo, 1):
            if n_done % 25 == 1:
                gpu_idle()
            query = qs[i]
            qid = int(query['id'])
            t0 = time.perf_counter()
            cand, dsc, ssc = C['hybrid'][i]
            src = source_gids(names, segments, cand)
            have = reuse_pairs.get(qid, {})
            need = np.array([j for j, g in enumerate(src) if int(g) not in have], dtype=np.int64)
            pair_src = {int(src[j]): have[int(src[j])] for j in range(len(cand)) if int(src[j]) in have}
            if len(need):
                texts = take_segment_texts(segments, cand[need])
                for j, sc in zip(need, rr.score(ws(query['query']), texts)):
                    pair_src[int(src[j])] = float(sc)
            pair = {int(c): pair_src[int(g)] for c, g in zip(cand, src)}
            owner = np.searchsorted(starts, cand, side='right') - 1
            doc_of, seg_of = {}, {}
            for si in np.unique(owner):
                m = owner == si
                local = cand[m] - starts[si]
                for c, d in zip(cand[m], np.asarray(segments[si].doc_id[local])):
                    doc_of[int(c)] = int(d)
                    seg_of[int(d)] = names[si]
            d_ids = np.asarray(C['dense_top'][0][i], dtype=np.int64)
            d_sc = np.asarray(C['dense_top'][1][i], dtype=np.float32)
            branches, best = {}, {}
            for branch, (ids, first_scores) in {'hybrid': (cand, HYBRID_W * minmax(dsc) + (1 - HYBRID_W) * minmax(ssc)),
                                                'dense': (d_ids, d_sc)}.items():
                first_docs = {}
                for c in np.asarray(ids)[np.argsort(-first_scores, kind='stable')]:
                    first_docs.setdefault(doc_of[int(c)], None)
                selected = set(list(first_docs)[:TOP_DOCS])
                by_doc, arg = {}, {}
                for c in ids:
                    d = doc_of[int(c)]
                    if d in selected and pair[int(c)] > by_doc.get(d, -float('inf')):
                        by_doc[d], arg[d] = pair[int(c)], int(c)
                branches[branch] = sorted(by_doc.items(), key=lambda row: -row[1])
                best[branch] = {str(d): arg[d] for d in by_doc}
            atomic_json(rr_dir / f'q{qid}.json', {'query_id': qid, **branches, 'best_chunk': best, 'rr_pairs': len(cand),
                        'rr_pairs_scored': int(len(need)), 'pairs': {str(g): s for g, s in pair_src.items()},
                        'doc_segment': {str(d): s for d, s in seg_of.items()}, 'seconds': time.perf_counter() - t0})
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.perf_counter() - t_run
                log(step='rerank', done=n_done, of=len(todo), seconds_this_run=round(el), per_query_s=round(el / n_done, 2),
                    eta_min=round(el / n_done * (len(todo) - n_done) / 60, 1))
        del rr
        torch.cuda.empty_cache()
    log(step='rerank_done', files=len(list(rr_dir.glob('q*.json'))))


def load_cache(qids=None):
    rr_dir = OUT / 'rerank'
    qids = qids if qids is not None else sorted(int(f.stem[1:]) for f in rr_dir.glob('q*.json'))
    return {q: json.loads((rr_dir / f'q{q}.json').read_text('utf-8')) for q in qids}


def export(top=50, out=None):
    """artifacts/cand_zh_v2.parquet: per query the top-`top` docs of the dense branch by max chunk logit."""
    import pyarrow.parquet as pq
    qids = pq.read_table(QUERY_FILE, columns=['id']).column('id').to_pylist()
    cache = load_cache(qids)
    if len(cache) != 1200:
        raise SystemExit(f'rerank incomplete: {len(cache)}/1200 queries')
    _, names, segments = load_v2_segments()
    groups = doc_groups(names)
    rows = []
    for q in qids:
        for rank, (d, logit) in enumerate(cache[q]['dense'][:top], 1):
            g = groups[int(d)]
            rows.append({'query_id': q, 'rank': rank, 'doc_id': int(d), 'logit': float(logit), 'domain': g['domain'],
                         'segment': g['segment'], 'chunk_id_v2': int(cache[q]['best_chunk']['dense'][str(d)]),
                         'doc_ids_group': g['group'], 'doc_ids_v2': g['v2']})
    df = pl.DataFrame(rows, schema={'query_id': pl.Int64, 'rank': pl.Int32, 'doc_id': pl.Int64, 'logit': pl.Float64,
                                    'domain': pl.String, 'segment': pl.String, 'chunk_id_v2': pl.Int64,
                                    'doc_ids_group': pl.List(pl.Int64), 'doc_ids_v2': pl.List(pl.Int64)})
    path = guard(out or ARTIFACTS / 'cand_zh_v2.parquet')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = guard(path.with_name(path.name + '.tmp'))
    df.write_parquet(tmp, compression='zstd')
    os.replace(tmp, path)
    return {'path': str(path), 'rows': df.height, 'queries': df['query_id'].n_unique(),
            'docs_per_query_min': int(df.group_by('query_id').len()['len'].min()),
            'sha256': hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()}


def zg(base_json=None, tau=TAU, cap=CAP):
    """ZG on v2: docs with max logit >= tau (dense branch), at most `cap` per query, not already in the VS-append base,
    whole doc_ids_group appended (primary first) then restricted to v2 ids. Reports docs/ids per query; builds no zip."""
    import statistics
    from r2ai.zh_full_ingest.append_zf import BASE, VS_APPEND_SHA256, select_zf
    from r2ai.zh_sample.append import _rows
    raw = Path(base_json or BASE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != VS_APPEND_SHA256:
        raise ValueError('VS-append SHA256 mismatch; refusing mutable baseline')
    base = raw.decode('utf-8')
    _, names, _ = load_v2_segments()
    groups = doc_groups(names)
    cache = load_cache()
    pairs = {q: [(int(d), s) for d, s in v['dense']] for q, v in cache.items()}
    if len(pairs) != 1200:
        raise SystemExit(f'rerank incomplete: {len(pairs)}/1200 queries')
    from r2ai.corpus_v2 import member_mask
    per_q, docs_n, ids_n, raw_ids_n, above, sel = {}, [], [], [], [], {}
    full_groups = {d: g['group'] for d, g in groups.items()}
    for row, _ in _rows(base):
        qid = row['id']
        present = set(row['relevant_docs'])
        ids, docs, available = select_zf(pairs[qid], full_groups, present, cap, tau=tau)
        v2_ids = [i for i, ok in zip(ids, member_mask(ids)) if ok] if ids else []
        sel[qid] = {'docs': docs, 'ids': v2_ids, 'ids_with_nonmembers': ids}
        docs_n.append(len(docs))
        ids_n.append(len(v2_ids))
        raw_ids_n.append(len(ids))
        above.append(available)
    n = len(docs_n)
    hist = {}
    for x in docs_n:
        hist[x] = hist.get(x, 0) + 1
    summary = {'tau': tau, 'cap': cap, 'variant': 'ZG (whole group, v2 ids only)', 'queries': n, 'base_sha256': VS_APPEND_SHA256,
               'docs_per_query': {'mean': statistics.fmean(docs_n), 'min': min(docs_n), 'max': max(docs_n),
                                  'median': statistics.median(docs_n)},
               'ids_per_query_v2': {'mean': statistics.fmean(ids_n), 'min': min(ids_n), 'max': max(ids_n)},
               'ids_per_query_before_v2_filter': statistics.fmean(raw_ids_n),
               'docs_total': sum(docs_n), 'ids_total_v2': sum(ids_n), 'queries_with_zero_docs': sum(x == 0 for x in docs_n),
               'docs_at_or_above_tau_not_in_base_per_query_mean': statistics.fmean(above),
               'docs_per_query_histogram': {str(k): hist[k] for k in sorted(hist)}}
    atomic_json(OUT / 'zg_v2.json', {'summary': summary, 'selection': {str(q): v for q, v in sel.items()}})
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=('first-stage', 'rerank', 'export', 'zg'))
    ap.add_argument('--reuse', default=None, help='earlier rerank dir whose per-pair logits are reused')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tau', type=float, default=TAU)
    ap.add_argument('--tag', default='retrieval', help='run directory name below out/runs/zh-v2 (one per ingest batch)')
    a = ap.parse_args(argv)
    global OUT
    OUT = RUNS_DIR / 'zh-v2' / a.tag
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    if a.step == 'first-stage':
        print(json.dumps(first_stage(), default=str))
    elif a.step == 'rerank':
        rerank(Path(a.reuse) if a.reuse else None, a.limit)
    elif a.step == 'export':
        print(json.dumps(export()))
    else:
        print(json.dumps(zg(tau=a.tau), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
