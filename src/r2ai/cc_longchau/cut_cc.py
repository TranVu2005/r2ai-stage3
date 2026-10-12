"""cc-A-cut115 = cc-A with the S5 adaptive doc cut (opt-in; nothing in the best config changes).

Order: cut the RRF primary docs (A1, logit tau calibrated to a target mean of kept primary docs/query, chunk docs protected)
-> cluster expansion -> slug ids (BM25 pool of URLs without text, then only those WITHOUT a CC capture kept) -> the CC picks
(same reranker-ranked picks as cc-A, count = slug ids replaced). relevant_chunks stay the RRF-chunk base's bytes.

  python -m r2ai.cc_longchau.cut_cc --target-mean 115
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.dupes import variants as V
from r2ai.paths import QUERY_FILE, resolve_path
from r2ai.submit import doc_cut as C
from r2ai.submit import rrf_chunks as R
from r2ai.submit import rrf_docs as RD
from r2ai.submit.best import load_config, sha256
from r2ai.submit.rrf_docs import rrf_doc_submission
from r2ai.submit.slug_append import slug_append_submission
from r2ai.zh_sample.append import _rows

from .append_cc import load_inputs, run_variant
from .common import Roots, atomic_json

BASE = 'out/runs/best-vsappend/gate-best/sub_rrf.json'      # RRF-chunk JSON of the best replay (config rrf_json_sha256)
CONFIG = 'configs/submission-best.yaml'


def _p(x):
    return str(resolve_path(x))


def calibrate(cfg, target_mean, floor=20, ceil=150):
    """tau (2 decimals) whose mean kept primary docs/query, after protecting chunk docs, hits `target_mean` (same bisect as S5)."""
    s, rds, exp = cfg['submission'], cfg['postprocess']['rrf_doc_set'], cfg['postprocess']['expand_clusters']
    qids = sorted(int(x) for x in pq.read_table(QUERY_FILE, columns=['id'])['id'].to_pylist())
    rr, hy, sc = R.ranks(_p(rds['reranker_cache'])), R.ranks(_p(rds['hybrid_ranks'])), C.pool_scores(_p(rds['reranker_cache']))
    pool = {q: sorted(rr[q], key=rr[q].get) for q in qids}
    docs = {q: RD.rrf_docs(pool[q], rr[q], hy[q], int(rds['k_docs']), int(rds['k'])) for q in qids}
    groups = V.load_groups(_p(s['chunks_dir']))
    of, members = V.load_clusters(_p(exp['clusters']), exp.get('scope', 'content'))
    cids = {}
    with open(_p(BASE), encoding='utf-8') as f:
        for line in f:
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
    mean_at = lambda tau: float(np.mean([len(C.cut_a1(docs[q], sc[q], tau, floor, ceil, prot[q])[0]) for q in qids]))
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_at(mid) > target_mean else (lo, mid)
    tau = round((lo + hi) / 2, 2)
    return tau, mean_at(tau), {'candidates_mean': float(np.mean([len(docs[q]) for q in qids])),
                               'protected_mean': float(np.mean([len(prot[q]) for q in qids]))}


def build(target_mean, name):
    roots = Roots()
    roots.preflight()
    cfg = load_config(_p(CONFIG))
    s, rds, sla, exp = cfg['submission'], cfg['postprocess']['rrf_doc_set'], cfg['postprocess']['slug_append'], cfg['postprocess']['expand_clusters']
    if sha256(Path(_p(BASE))) != s['rrf_json_sha256']:
        raise ValueError('base RRF json differs from the config hash')
    args = s['args']
    arg = lambda k: args[args.index(k) + 1]
    tau, mean, info = calibrate(cfg, target_mean)
    out = roots.guard(roots.run / name)
    out.mkdir(parents=True, exist_ok=True)
    stage = roots.guard(out / '_stage')
    stage.mkdir(exist_ok=True)
    st1 = rrf_doc_submission(_p(BASE), stage / 'mid.zip', chunks='base', runs_dir=_p(s['runs_dir']), doc_ranking=_p(s['doc_ranking']),
                             hybrid_ranks=_p(rds['hybrid_ranks']), rerank_run=_p(rds['reranker_cache']), queries=_p(cfg['queries']),
                             chunks_dir=_p(s['chunks_dir']), docs_dir=_p(s['docs_dir']), clusters=_p(exp['clusters']),
                             scope=exp.get('scope', 'content'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')),
                             k_docs=int(rds['k_docs']), k_chunk=int(arg('--k-chunk')), rrf_k=int(rds['k']),
                             cut=C.make_cut('a1', tau=tau, floor=20, ceil=150, protect_chunks=True))
    mid_json = stage / 'mid.json'
    slug_append_submission(mid_json, stage / 'slug.zip', pin=_p(sla['pin']), pin_sha256=sla['pin_sha256'], queries=_p(cfg['queries']),
                           k=int(sla.get('k', 20)), k1=float(sla['k1']), b=float(sla['b']), scan_depth=int(sla['scan_depth']))
    mid = mid_json.read_text(encoding='utf-8')
    with open(stage / 'slug.json', encoding='utf-8', newline='') as f:
        slugged = {r['id']: r['relevant_docs'] for r, _ in _rows(f.read())}
    slug = {}
    for row, _ in _rows(mid):
        have = row['relevant_docs']
        full = slugged[row['id']]
        if full[:len(have)] != have:
            raise ValueError('slug_append changed the cut relevant_docs prefix')
        slug[row['id']] = full[len(have):]
    qids, _, groups, ranking = load_inputs(roots)
    stats = run_variant(roots, mid, name, qids, slug, groups, ranking, plan='VS-cc-A', base_label=f'{BASE} + doc cut A1 tau={tau}')
    stats |= {'doc_cut': st1.get('doc_cut'), 'tau': tau, 'target_mean': target_mean, 'mean_primary_after_cut': mean, 'calibration': info,
              'slug_per_query_mean_before_cc_filter': float(np.mean([len(v) for v in slug.values()]))}
    atomic_json(roots, out / 'stats.json', stats)
    for f in stage.iterdir():
        f.unlink()
    stage.rmdir()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-mean', type=float, default=115)
    ap.add_argument('--name', default=None)
    a = ap.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    name = a.name or f'VS-cc-A-cut{int(a.target_mean)}'
    s = build(a.target_mean, name)
    print(json.dumps({k: s[k] for k in ('variant', 'tau', 'mean_primary_after_cut', 'zip_bytes', 'zip_mib', 'zip_sha256', 'json_sha256',
                                        'validator', 'gate', 'relevant_docs_per_query', 'ids_appended_per_query', 'doc_cut')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
