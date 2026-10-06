"""RRF choice of relevant_docs over the tier-1 = 200 rerank pool (CPU only): variants RD150 / RD150c of the best submission.

    python -m r2ai.submit.rrf_docs build --base <best.json> --out <dir>/sub.zip --chunks base|rrf

Pool per query = every doc of --rerank-run (vi_k100.parquet of the tier-1 = 200 rerank, ranks 1..200). Each is scored
    rrf = 1 / (k + rank_reranker) + 1 / (k + rank_hybrid),   k = 60
with rank_hybrid = rank in --doc-ranking (the hybrid order rrf_chunks uses); ties as r2ai.submit.rrf_chunks.rrf_choice (lower
rank_reranker, then lower rank_hybrid). A pool doc without a hybrid rank is an error (ValueError), not a fallback rank.
relevant_docs = doc_ids_group of the --k-docs best docs in RRF order, then the H1-expand cluster mates of those docs
(r2ai.dupes.variants.expand_lists with --clusters / --scope, as configs/submission-best.yaml).
--chunks base: relevant_chunks copied from --base (RD150); --chunks rrf: the full chunk (make_submission.load_full_texts) of the
--k-chunk best RRF docs of the pool, in that order (RD150c; a doc without text gets none). id and key order stay as in --base.
Stats (<out>.stats.json): docs new / dropped vs the D50 150 docs (--runs-dir / --doc-ranking / --k-cache / --k-total), their
reranker and hybrid ranks, overlap with R150 (first --k-docs docs of the pool in reranker order), chunks whose doc_id is not
in relevant_docs, chunk docs changed vs --base.
rrf_doc_submission is the same build as a library call, used by r2ai.submit.best when the config enables
postprocess.rrf_doc_set; hybrid_ranks may then differ from doc_ranking (the CLI uses --doc-ranking for both).
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs, resolve_path

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.dupes import variants as V
from r2ai.submit import rrf_chunks as R

DEF = R.DEF | {'chunks_dir': 'data/chunks', 'clusters': 'out/runs/H1/clusters.parquet'}


def rrf_docs(pool: list[int], rr: dict[int, int], hy: dict[int, int], n: int, k: int = 60) -> list[int]:
    """The n docs of pool with the highest RRF (rrf_chunks.rrf_choice); every pool doc must have a hybrid rank."""
    miss = [d for d in pool if d not in hy]
    if miss:
        raise ValueError(f'{len(miss)} pool docs without a hybrid rank, e.g. {miss[:3]}')
    return R.rrf_choice(pool, rr, hy, n, k)


def dist(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    if not len(x):
        return {'total': 0}
    return {'total': int(x.sum()), 'mean': round(float(x.mean()), 3), 'p50': float(np.percentile(x, 50)),
            'p95': float(np.percentile(x, 95)), 'max': int(x.max()), 'queries_with_any': int((x > 0).sum())}


def rank_dist(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    if not len(x):
        return {'n': 0}
    edges = (50, 100, 150, 200)
    hist, lo = {}, 0
    for e in edges:
        hist[f'{lo + 1}-{e}'] = int(((x > lo) & (x <= e)).sum())
        lo = e
    hist[f'>{lo}'] = int((x > lo).sum())
    return {'n': len(x), 'mean': round(float(x.mean()), 2), 'p50': float(np.percentile(x, 50)),
            'p95': float(np.percentile(x, 95)), 'max': int(x.max()), 'hist': hist}


def rrf_doc_submission(base_json, out_zip, *, chunks, runs_dir, doc_ranking, hybrid_ranks, rerank_run, queries, chunks_dir,
                       docs_dir, clusters, scope='content', k_cache=100, k_total=150, k_docs=150, k_chunk=50, rrf_k=60) -> dict:
    """base JSON -> out_zip + its .json (see the module doc); returns the stats."""
    from types import SimpleNamespace
    a = SimpleNamespace(base=base_json, out=out_zip, chunks=chunks, runs_dir=runs_dir, doc_ranking=doc_ranking,
                        hybrid_ranks=hybrid_ranks, rerank_run=rerank_run, queries=queries, chunks_dir=chunks_dir, docs_dir=docs_dir,
                        clusters=clusters, scope=scope, k_cache=k_cache, k_total=k_total, k_docs=k_docs, k_chunk=k_chunk, rrf_k=rrf_k)
    return _build(a)


def _build(a) -> dict:
    from r2ai.submit.make_submission import load_full_texts
    out_zip = assert_writable(a.out)
    out_json = assert_writable(out_zip.with_suffix('.json'))
    require_inputs(a.base, a.rerank_run, a.doc_ranking, a.hybrid_ranks, a.clusters, resolve_path(a.chunks_dir) / 'docs.parquet',
                   a.docs_dir)
    t0 = time.time()
    qids = sorted(int(x) for x in pq.read_table(resolve_path(a.queries), columns=['id'])['id'].to_pylist())
    rr, hy = R.ranks(a.rerank_run), R.ranks(a.hybrid_ranks)
    pool = {q: sorted(rr.get(q, {}), key=rr.get(q, {}).get) for q in qids}
    pool_missing_hy = sum(d not in hy.get(q, {}) for q in qids for d in pool[q])
    docs = {q: rrf_docs(pool[q], rr.get(q, {}), hy.get(q, {}), a.k_docs, a.rrf_k) for q in qids}
    top, rest = V.load_lists(a.runs_dir, a.doc_ranking, qids, a.k_cache)
    d50 = V.d50_docs(top, rest, a.k_total)
    groups = V.load_groups(a.chunks_dir)
    of, members = V.load_clusters(a.clusters, a.scope)
    extra = V.expand_lists(docs, of, members, groups)
    chunk_docs = {q: rrf_docs(pool[q], rr.get(q, {}), hy.get(q, {}), a.k_chunk, a.rrf_k) for q in qids} if a.chunks == 'rrf' else {}
    full, src = ({}, {})
    if a.chunks == 'rrf':
        full, src, _ = load_full_texts({d for q in qids for d in chunk_docs[q]}, resolve_path(a.docs_dir))
    new_in, dropped, same_r, not_r, out_docs, changed, missing = [], [], [], [], [], [], 0
    new_rr, new_hy, out_q, n = [], [], 0, 0
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(a.base, 'r', encoding='utf-8', newline='') as fi, open(out_json, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write('[\n')
        for line in fi:
            line = line.rstrip('\n')
            if line in ('[', ']'):
                continue
            row = json.loads(line.rstrip(','))
            q = row['id']
            row['relevant_docs'] = list(dict.fromkeys(int(x) for d in docs[q] for x in groups[d])) + extra[q]
            if a.chunks == 'rrf':
                old = {c['doc_id'] for c in row['relevant_chunks']}
                chunks = []
                for d in chunk_docs[q]:
                    t = full.get(d, '')
                    if not t.strip():
                        missing += 1
                        continue
                    chunks.append({'doc_id': int(d), 'chunk_text': t})
                changed.append(sum(c['doc_id'] not in old for c in chunks))
                row['relevant_chunks'] = chunks
            have = set(row['relevant_docs'])
            k_out = sum(c['doc_id'] not in have for c in row['relevant_chunks'])
            out_docs.append(k_out)
            out_q += k_out > 0
            s, s50, r150 = set(docs[q]), set(d50[q]), set(pool[q][:a.k_docs])
            add = [d for d in docs[q] if d not in s50]
            new_in.append(len(add))
            dropped.append(len(s50 - s))
            same_r.append(len(s & r150))
            not_r.append(len(s - r150))
            new_rr += [rr[q][d] for d in add]
            new_hy += [hy[q][d] for d in add]
            fo.write(('' if n == 0 else ',\n') + json.dumps(row, ensure_ascii=False, separators=(', ', ': ')))
            n += 1
        fo.write('\n]\n')
    if n != len(qids):
        raise ValueError(f'{a.base}: {n} queries, expected {len(qids)}')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out_json, out_json.name)
    m = {'base': str(a.base), 'chunks': a.chunks, 'queries': n, 'k_docs': a.k_docs, 'k_chunk': a.k_chunk, 'rrf_k': a.rrf_k,
         'scope': a.scope, 'pool_docs_per_query': dist(len(pool[q]) for q in qids), 'pool_docs_without_hybrid_rank': pool_missing_hy,
         'primary_docs_per_query': dist(len(docs[q]) for q in qids), 'd50_docs_per_query': dist(len(d50[q]) for q in qids),
         'docs_new_vs_d50': dist(new_in), 'docs_dropped_vs_d50': dist(dropped),
         'new_docs_reranker_rank': rank_dist(new_rr), 'new_docs_hybrid_rank': rank_dist(new_hy),
         'docs_same_as_r150': dist(same_r), 'docs_not_in_r150': dist(not_r),
         'expand_ids_added': dist(len(extra[q]) for q in qids),
         'chunk_docs_outside_relevant_docs': {'total': int(sum(out_docs)), 'queries': out_q}}
    if a.chunks == 'rrf':
        m |= {'chunk_docs_changed_vs_base': dist(changed), 'docs_missing_text': missing, 'full_text_sources': src}
    m |= {'json_bytes': out_json.stat().st_size, 'json_sha256': R._sha256(out_json), 'zip_bytes': out_zip.stat().st_size,
          'zip_sha256': R._sha256(out_zip), 'max_zip_bytes': 104857600, 'zip_within_ceiling': out_zip.stat().st_size <= 104857600,
          'elapsed_s': round(time.time() - t0, 1)}
    return m


def cmd_build(a) -> int:
    a.hybrid_ranks = a.doc_ranking
    m = _build(a)
    Path(a.out).with_suffix('.stats.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print(json.dumps(m, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--base', required=True, help='base submission JSON (id order; relevant_chunks kept with --chunks base)')
    b.add_argument('--out', required=True)
    b.add_argument('--chunks', choices=('base', 'rrf'), required=True)
    b.add_argument('--runs-dir', default=DEF['runs_dir'], help='D50 deep-rerank cache (for the D50 comparison only)')
    b.add_argument('--doc-ranking', default=DEF['doc_ranking'], help='hybrid ranks (RRF) and D50 order after --k-cache')
    b.add_argument('--rerank-run', default=DEF['rerank_run'], help='tier-1 = 200 rerank cache: the pool and its reranker ranks')
    b.add_argument('--queries', default=DEF['queries'])
    b.add_argument('--chunks-dir', default=DEF['chunks_dir'])
    b.add_argument('--docs-dir', default=DEF['docs_dir'])
    b.add_argument('--clusters', default=DEF['clusters'])
    b.add_argument('--scope', choices=('content', 'all'), default='content')
    b.add_argument('--k-cache', type=int, default=100)
    b.add_argument('--k-total', type=int, default=150)
    b.add_argument('--k-docs', type=int, default=150)
    b.add_argument('--k-chunk', type=int, default=50)
    b.add_argument('--rrf-k', type=int, default=60)
    a = ap.parse_args(argv)
    return cmd_build(a)


if __name__ == '__main__':
    sys.exit(main())
