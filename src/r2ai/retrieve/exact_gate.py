"""Equivalence gate: legacy first stage (FAISS IndexFlatIP + full CSC) vs exact block scan, no rerank.

    python -m r2ai.retrieve.exact_gate queries --queries Q --out-dir OUT          # BGE-M3 on CPU (fp32), cached
    python -m r2ai.retrieve.exact_gate legacy  --index-dir IDX --chunks C --out-dir OUT
    python -m r2ai.retrieve.exact_gate exact   --index-dir IDX --chunks C --out-dir OUT [--block-rows 100000]
    python -m r2ai.retrieve.exact_gate compare --chunks C --out-dir OUT

Each stage is its own process so elapsed and peak RSS are per method. Both methods get the same query vectors.
legacy: retrieve.run.Index.candidates itself (FAISS + CSC). The FAISS flat index is opened with IO_FLAG_MMAP_IFC and
searched in groups of 16 queries (< faiss distance_compute_blas_threshold 20, i.e. the same per-query
exhaustive_inner_product_seq arithmetic as one query at a time) so its 3.35 GB is not copied into RAM.
compare: K100 doc set before rerank = tier 1 (rank_docs top-50 docs) + tier 2 (next 50 docs by max hybrid over all
candidates), exactly as run_retrieval_k100 picks them before reranking.
The index directory is only read (np.load mmap 'r', faiss read-only); the sparse mmap cache goes to OUT.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs, resolve_path

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')
K = 200
FAISS_GROUP = 16


def _peak_rss_gib() -> float:
    from r2ai.retrieve.run_retrieval_k100 import peak_rss_gib
    return peak_rss_gib()


def _peak_private_gib():
    from r2ai.retrieve.run_retrieval_k100 import peak_private_gib
    return peak_private_gib()


def _fingerprint(paths) -> dict:
    out = {}
    for p in paths:
        st = os.stat(p)
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                h.update(b)
        out[str(p)] = {'size': st.st_size, 'mtime': st.st_mtime, 'sha256': h.hexdigest()}
    return out


def _doc_id(chunks: Path) -> np.ndarray:
    return pq.read_table(chunks, columns=['doc_id'])['doc_id'].to_numpy()


def stage_queries(a, out: Path):
    from r2ai.index.bge_m3 import M3Encoder
    from r2ai.retrieve.run import ws
    rows = pq.read_table(a.queries).to_pylist()
    t0 = time.time()
    enc = M3Encoder(device='cpu', fp16=False, max_len=512)
    qd, qsp = [], []
    for b in range(0, len(rows), 32):
        d, s = enc.encode_batch([ws(r['query']) for r in rows[b:b + 32]])
        qd.append(d)
        qsp += s
    res = {'ids': [int(r['id']) for r in rows], 'qd': np.concatenate(qd), 'qsp': qsp,
           'encoder': 'BGE-M3 cpu fp32 (dense cast to fp16 as encode_batch)', 'seconds': round(time.time() - t0, 1)}
    with open(out / 'queries.pkl', 'wb') as f:
        pickle.dump(res, f)
    return {'n': len(rows), 'seconds': res['seconds'], 'peak_rss_gib': _peak_rss_gib()}


def stage_legacy(a, out: Path):
    import faiss
    from scipy.sparse import load_npz
    from r2ai.retrieve.run import Index
    q = pickle.load(open(out / 'queries.pkl', 'rb'))
    idx = Path(a.index_dir)
    t0 = time.time()
    ix = object.__new__(Index)                                      # the legacy candidates() on the legacy data
    ix.faiss = faiss.read_index(str(idx / 'faiss.index'), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    ix.dense = np.load(idx / 'dense.npy', mmap_mode='r')
    ix.csc = load_npz(idx / 'sparse.npz').tocsc()
    ix.allowed = None
    t_load = time.time() - t0
    rss_load = _peak_rss_gib()
    # dense: FAISS search per group of 16 (same arithmetic as single-query search), then fed to candidates()
    t0 = time.time()
    di_all = []
    for g in range(0, len(q['qd']), FAISS_GROUP):
        _, di = ix.faiss.search(q['qd'][g:g + FAISS_GROUP].astype(np.float32), K)
        di_all += list(di)
    t_dense = time.time() - t0
    from types import SimpleNamespace
    t0 = time.time()
    res = []
    for j in range(len(q['qd'])):
        ix.faiss = SimpleNamespace(search=lambda x, k, _d=di_all[j]: (None, _d[None]))   # this query's group result
        res.append(ix.candidates(q['qd'][j], q['qsp'][j]))
    t_rest = time.time() - t0
    with open(out / 'legacy.pkl', 'wb') as f:
        pickle.dump({'res': res, 'di': di_all}, f)
    return {'load_seconds': round(t_load, 1), 'faiss_dense_seconds': round(t_dense, 1),
            'csc_sparse_union_rescore_seconds': round(t_rest, 1), 'peak_rss_gib_after_load': rss_load,
            'peak_rss_gib': _peak_rss_gib(), 'faiss': 'IndexFlatIP mmap (IO_FLAG_MMAP_IFC), groups of 16'}


def stage_exact(a, out: Path):
    from r2ai.retrieve.exact import exact_candidates, load_sparse_mmap, split_sparse_npz
    q = pickle.load(open(out / 'queries.pkl', 'rb'))
    idx = Path(a.index_dir)
    t0 = time.time()
    cache = assert_writable(out / 'sparse_mmap')
    info = split_sparse_npz(idx / 'sparse.npz', cache)
    t_split = time.time() - t0
    rss_split = _peak_rss_gib()
    data, indices, indptr, shape = load_sparse_mmap(cache)
    dense = np.load(idx / 'dense.npy', mmap_mode='r')
    doc_id = _doc_id(Path(a.chunks))
    assert len(doc_id) == len(dense) == shape[0] == len(indptr) - 1
    res, st = exact_candidates(dense, (data, indices, indptr), doc_id, q['qd'], q['qsp'], k=K,
                               block_rows=a.block_rows, device='cpu')
    with open(out / 'exact.pkl', 'wb') as f:
        pickle.dump({'res': res}, f)
    nd = st['n_docs']
    return {'split_seconds': round(t_split, 1), 'split_reused': info['reused'], 'peak_rss_gib_after_split': rss_split,
            'nnz': info['nnz'], 'shape': info['shape'], 'n_vq': st['n_vq'], 'block_rows': a.block_rows,
            'dense_seconds': round(st['dense_seconds'], 1), 'sparse_seconds': round(st['sparse_seconds'], 1),
            'union_rescore_seconds': round(st['union_rescore_seconds'], 1), 'peak_rss_gib': _peak_rss_gib(),
            'distinct_docs': {'min': int(nd.min()), 'p5': float(np.percentile(nd, 5)), 'p50': float(np.percentile(nd, 50)),
                              'n_lt_100': int((nd < 100).sum())}}


def k100_prerank(doc_id: np.ndarray, cand, dsc, ssc) -> list[int]:
    """Docs of run_retrieval_k100 before rerank: tier 1 (rank_docs) then tier 2 (next 50 by max hybrid)."""
    from types import SimpleNamespace

    from r2ai.retrieve.run import minmax, rank_docs
    from r2ai.retrieve.run_retrieval_k100 import AGG, TIER2_DOCS, W
    _, _, ranked, _ = rank_docs(SimpleNamespace(doc_id=doc_id), cand, dsc, ssc, W, AGG)
    h = W * minmax(dsc) + (1 - W) * minmax(ssc)
    seen, tier2 = set(ranked), []
    for j in np.argsort(-h, kind='stable'):
        d = int(doc_id[cand[j]])
        if d not in seen:
            seen.add(d)
            tier2.append(d)
            if len(tier2) == TIER2_DOCS:
                break
    return list(ranked) + tier2


def stage_compare(a, out: Path):
    q = pickle.load(open(out / 'queries.pkl', 'rb'))
    old = pickle.load(open(out / 'legacy.pkl', 'rb'))['res']
    new = pickle.load(open(out / 'exact.pkl', 'rb'))['res']
    doc_id = _doc_id(Path(a.chunks))
    n = len(old)
    same_set = same_order = same_cand = 0
    max_dd = max_ds = 0.0
    diffs = []
    for j in range(n):
        (c0, d0, s0), (c1, d1, s1) = old[j], new[j]
        if np.array_equal(c0, c1):
            same_cand += 1
            max_dd = max(max_dd, float(np.abs(d0 - d1).max()))
            max_ds = max(max_ds, float(np.abs(s0 - s1).max()))
        k0, k1 = k100_prerank(doc_id, c0, d0, s0), k100_prerank(doc_id, c1, d1, s1)
        same_order += k0 == k1
        if set(k0) == set(k1):
            same_set += 1
            continue
        # cause: chunks in only one candidate set and how far their scores are from the k-th score of the branch
        only_old, only_new = np.setdiff1d(c0, c1), np.setdiff1d(c1, c0)
        qd = q['qd'][j].astype(np.float32)
        full_d0 = dict(zip(c0.tolist(), d0.tolist()))
        full_d1 = dict(zip(c1.tolist(), d1.tolist()))
        kth_dense = float(np.sort(d1)[::-1][min(K, len(d1)) - 1]) if len(d1) else None
        diffs.append({'qid': q['ids'][j], 'only_legacy_chunks': only_old.tolist()[:10], 'only_exact_chunks': only_new.tolist()[:10],
                      'dense_score_only_legacy': [full_d0[c] for c in only_old.tolist()[:10]],
                      'dense_score_only_exact': [full_d1[c] for c in only_new.tolist()[:10]],
                      'docs_only_legacy': sorted(set(k0) - set(k1)), 'docs_only_exact': sorted(set(k1) - set(k0))})
    return {'queries': n, 'k100_prerank_same_set': same_set, 'k100_prerank_same_set_pct': round(100 * same_set / n, 3),
            'k100_prerank_same_order': same_order, 'candidate_set_identical': same_cand,
            'max_abs_dense_score_diff_identical_sets': max_dd, 'max_abs_sparse_score_diff_identical_sets': max_ds,
            'diffs': diffs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('queries', 'legacy', 'exact', 'compare'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--queries')
    ap.add_argument('--index-dir')
    ap.add_argument('--chunks')
    ap.add_argument('--block-rows', type=int, default=100_000)
    a = ap.parse_args(argv)
    out = assert_writable(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'queries':
        require_inputs(a.queries)
    if a.stage in ('legacy', 'exact'):
        a.index_dir = str(resolve_path(a.index_dir))
        require_inputs(*(Path(a.index_dir) / f for f in ('dense.npy', 'sparse.npz') + (('faiss.index',) if a.stage == 'legacy' else ())))
    if a.stage != 'queries':
        require_inputs(a.chunks)
    watched = [Path(a.index_dir) / f for f in ('faiss.index', 'dense.npy', 'sparse.npz')] if a.stage in ('legacy', 'exact') else []
    before = _fingerprint(watched) if watched else {}
    t0 = time.time()
    res = {'stage': a.stage, **globals()[f'stage_{a.stage}'](a, out), 'elapsed_seconds': round(time.time() - t0, 1),
           'peak_private_gib': _peak_private_gib()}
    if watched:
        res['index_files_unchanged'] = _fingerprint(watched) == before
        assert res['index_files_unchanged'], 'index files changed during the gate'
    (out / f'{a.stage}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'diffs'}, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
