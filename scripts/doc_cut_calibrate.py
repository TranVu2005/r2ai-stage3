"""Calibrate tau of doc_cut A1 on the RRF primary docs (mean primary docs/query after the chunk-doc protection; no pseudo dev):
PYTHONPATH=src python scripts/doc_cut_calibrate.py"""
import json
import numpy as np
import pyarrow.parquet as pq
from r2ai.submit import rrf_chunks as R, rrf_docs as RD, doc_cut as C
from r2ai.dupes import variants as V

M = 'D:/GitHub/r2ai-stage3/'
rerank, hybrid = M + 'out/runs/rerank200/full/vi_k100.parquet', M + 'out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet'
BASE = M + 'out/runs/best-vsappend/gate-best/sub_rrf.json'
qids = sorted(int(x) for x in pq.read_table('D:/GitHub/r2ai-stage3-old/data/raw/query.parquet', columns=['id'])['id'].to_pylist())
rr, hy, sc = R.ranks(rerank), R.ranks(hybrid), C.pool_scores(rerank)
pool = {q: sorted(rr[q], key=rr[q].get) for q in qids}
docs = {q: RD.rrf_docs(pool[q], rr[q], hy[q], 150) for q in qids}
groups = V.load_groups(M + 'data/chunks')
of, members = V.load_clusters(M + 'out/runs/H1/clusters.parquet', 'content')
cids = {}
for line in open(BASE, encoding='utf-8'):
    line = line.strip()
    if line.startswith('{'):
        r = json.loads(line.rstrip(','))
        cids[r['id']] = {int(c['doc_id']) for c in r['relevant_chunks']}
def covers(d):
    ids = {int(x) for x in groups[d]}
    c = of.get(d)
    if c is not None:
        ids |= {int(x) for m in members[c] for x in groups.get(m, [m])}
    return ids
prot = {q: {d for d in docs[q] if cids[q] & covers(d)} for q in qids}
print('protected docs/query mean', round(np.mean([len(prot[q]) for q in qids]), 2), 'p50', np.median([len(prot[q]) for q in qids]))

def stats(fn):
    n, fl, ce = [], 0, 0
    for q in qids:
        k, f, c = fn(docs[q], sc[q], prot[q]); n.append(len(k)); fl += f; ce += c
    n = np.array(n)
    return dict(mean=round(float(n.mean()), 2), p10=float(np.percentile(n, 10)), p50=float(np.percentile(n, 50)), p90=float(np.percentile(n, 90)), floor=fl, ceil=ce)

res = {}
for target in (60, 100, 130):
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        m = np.mean([len(C.cut_a1(docs[q], sc[q], mid, protect=prot[q])[0]) for q in qids])
        lo, hi = (mid, hi) if m > target else (lo, mid)
    t = round((lo + hi) / 2, 2)
    res[target] = t
    print(target, 'tau', t, stats(C.make_cut('a1', tau=t)))
print('a2', stats(C.make_cut('a2')))
print(json.dumps(res))
