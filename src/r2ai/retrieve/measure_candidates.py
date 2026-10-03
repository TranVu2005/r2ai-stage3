"""Size of the first-stage candidate set (no rerank, retrieval config unchanged: t256, hybrid w=0.7, max).

    python -m scripts.measure_candidates [--sample 100] [--seed 42]

  * all 1,200 queries, from data/runs/vi_k100.parquet: docs in tier 1 / tier 2 / total, queries with < 100 docs.
  * random sample of --sample queries (seed 42), first stage only: candidate chunks (dense top-200 U sparse top-200),
    distinct docs among those chunks before rerank, tier-1 docs (rank_docs, <= 50) and the tier-2 pool
    (distinct candidate docs not in tier 1, before the 50-doc cap of run_retrieval_k100).
Writes data/runs/candidate_stats.json.
"""
from __future__ import annotations

from r2ai.paths import RAW_DATA_DIR, RUNS_DIR, auxiliary_disabled

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def d3(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {'min': int(x.min()), 'p50': float(np.percentile(x, 50)), 'max': int(x.max()), 'mean': round(float(x.mean()), 1)}


def main(argv=None):
    auxiliary_disabled()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)
    import torch
    from r2ai.index.bge_m3 import M3Encoder
    from r2ai.retrieve.run import Index, rank_docs, ws

    run = pq.read_table(RUNS_DIR / 'vi-k100/vi_k100.parquet').to_pylist()
    per: dict[int, list[int]] = {}
    for r in run:
        t = per.setdefault(r['query_id'], [0, 0])
        t[r['tier'] - 1] += 1
    t1, t2 = [v[0] for v in per.values()], [v[1] for v in per.values()]
    tot = [x + y for x, y in zip(t1, t2)]
    res = {'all_queries': {'n': len(per), 'tier1_docs': d3(t1), 'tier2_docs_cached(<=50)': d3(t2), 'docs_total_cached(<=100)': d3(tot),
                           'queries_lt_100_docs': int(sum(x < 100 for x in tot)), 'queries_lt_50_tier1': int(sum(x < 50 for x in t1))}}

    rows = pq.read_table(RAW_DATA_DIR / 'query.parquet').to_pylist()
    random.seed(a.seed)
    sample = sorted(random.sample(range(len(rows)), a.sample))
    index = Index(256)
    enc = M3Encoder(max_len=512)
    qd, qs = [], []
    for b in range(0, len(sample), 32):
        d, s = enc.encode_batch([ws(rows[i]['query']) for i in sample[b:b + 32]])
        qd.append(d)
        qs += s
    qd = np.concatenate(qd)
    del enc
    torch.cuda.empty_cache()
    n_cand, n_docs, n_t1, n_pool = [], [], [], []
    for j in range(len(sample)):
        cand, dsc, ssc = index.candidates(qd[j], qs[j])
        _, _, ranked, _ = rank_docs(index, cand, dsc, ssc, 0.7, 'max')
        docs = set(int(index.doc_id[c]) for c in cand)
        n_cand.append(len(cand))
        n_docs.append(len(docs))
        n_t1.append(len(ranked))
        n_pool.append(len(docs - set(ranked)))
    res['sample'] = {'n': len(sample), 'seed': a.seed, 'query_ids': [int(rows[i]['id']) for i in sample[:5]] + ['...'],
                     'candidate_chunks': d3(n_cand), 'distinct_docs_before_rerank': d3(n_docs), 'tier1_docs': d3(n_t1),
                     'tier2_pool_docs(uncapped)': d3(n_pool), 'queries_lt_100_distinct_docs': int(sum(x < 100 for x in n_docs))}
    (RUNS_DIR / 'measure-candidates/candidate_stats.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
