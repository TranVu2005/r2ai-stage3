"""RRF variant of the full-chunk doc choice (CPU only): keep relevant_docs of a base submission, change only which docs get
the full chunk.

    python -m r2ai.submit.rrf_chunks measure --base <best.json> --out-dir <dir>
    python -m r2ai.submit.rrf_chunks build   --base <best.json> --out <dir>/sub.zip

Primary docs per query = the D50 list (r2ai.dupes.variants.load_lists / d50_docs on --runs-dir / --doc-ranking: deep-rerank
ranks 1..100, then hybrid order up to --k-total). Each is scored
    rrf = 1 / (k + rank_reranker) + 1 / (k + rank_hybrid),   k = 60
rank_reranker = rank in --rerank-run (vi_k100.parquet of the tier-1 = 200 rerank; a doc missing there gets rank --missing-rank),
rank_hybrid = rank in --doc-ranking (the candidate order the D50 builder used). The --k-chunk docs with the highest rrf
(ties: lower rank_reranker, then lower rank_hybrid) get a chunk, in that order; the chunk text is the builder's full-chunk
rule (make_submission.load_full_texts; a doc without text gets none, as in the builder). relevant_docs and id stay as in
--base, byte for byte; the JSON is written in the writer.format_submission layout and zipped like make_submission.
measure: changed chunk docs vs the base (the first --k-chunk primary docs) per query, and among the newly chosen docs how
many have rank_reranker > 50 / rank_hybrid > 100; writes rrf_measure.json. build refuses when the mean change is < --min-mean
(default 1.0 doc/query) unless --force.
rrf_chunk_submission is the same build as a library call (no --min-mean), used by r2ai.submit.best when the config enables
postprocess.rrf_chunk_docs; hybrid_ranks may then differ from doc_ranking (the CLI uses --doc-ranking for both).
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs, resolve_path

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.dupes import variants as V

DEF = {'runs_dir': 'out/runs/deep-rerank-2026-10-05/full', 'doc_ranking': 'out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet',
       'rerank_run': 'out/runs/rerank200/full/vi_k100.parquet', 'queries': 'D:/GitHub/r2ai-stage3-old/data/raw/query.parquet',
       'docs_dir': 'data/docs_vi'}


def ranks(path) -> dict[int, dict[int, int]]:
    t = pq.read_table(resolve_path(path), columns=['query_id', 'rank', 'doc_id']).to_pydict()
    out: dict[int, dict[int, int]] = {}
    for q, r, d in zip(t['query_id'], t['rank'], t['doc_id']):
        out.setdefault(int(q), {}).setdefault(int(d), int(r))
    return out


def rrf_choice(docs: list[int], rr: dict[int, int], hy: dict[int, int], n: int, k: int = 60, missing: int = 201) -> list[int]:
    """The n docs of `docs` with the highest 1/(k+rank_rr) + 1/(k+rank_hy); ties: lower rank_rr, then lower rank_hy."""
    def key(d):
        a, b = rr.get(d, missing), hy[d]
        return (-(1.0 / (k + a) + 1.0 / (k + b)), a, b)
    return sorted(docs, key=key)[:n]


def dist(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    return {'mean': round(float(x.mean()), 3), 'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)),
            'max': int(x.max()), 'queries_with_any': int((x > 0).sum())}


def choose(a) -> tuple[list[int], dict, dict, dict]:
    return choose_docs(runs_dir=a.runs_dir, doc_ranking=a.doc_ranking, hybrid_ranks=a.doc_ranking, rerank_run=a.rerank_run,
                       queries=a.queries, k_cache=a.k_cache, k_total=a.k_total, k_chunk=a.k_chunk, rrf_k=a.rrf_k,
                       missing_rank=a.missing_rank)


def choose_docs(*, runs_dir, doc_ranking, hybrid_ranks, rerank_run, queries, k_cache=100, k_total=150, k_chunk=50, rrf_k=60,
                missing_rank=201) -> tuple[list[int], dict, dict, dict]:
    """(qids, D50 primary docs, RRF-chosen chunk docs, measure) per query."""
    qids = sorted(int(x) for x in pq.read_table(resolve_path(queries), columns=['id'])['id'].to_pylist())
    top, pool = V.load_lists(runs_dir, doc_ranking, qids, k_cache)
    docs = V.d50_docs(top, pool, k_total)
    rr, hy = ranks(rerank_run), ranks(hybrid_ranks)
    new = {q: rrf_choice(docs[q], rr.get(q, {}), hy[q], k_chunk, rrf_k, missing_rank) for q in qids}
    changed, n_rr50, n_hy100, n_both, missing = [], 0, 0, 0, 0
    for q in qids:
        old = set(docs[q][:k_chunk])
        add = [d for d in new[q] if d not in old]
        changed.append(len(add))
        for d in add:
            r, h = rr.get(q, {}).get(d, missing_rank), hy[q][d]
            n_rr50 += r > 50
            n_hy100 += h > 100
            n_both += r > 50 and h > 100
        missing += sum(d not in rr.get(q, {}) for d in docs[q])
    m = {'queries': len(qids), 'k_chunk': k_chunk, 'rrf_k': rrf_k,
         'primary_docs_per_query': dist(len(docs[q]) for q in qids),
         'primary_docs_without_reranker_rank': missing,
         'chunk_docs_changed_vs_base': dist(changed), 'new_chunk_docs_total': int(sum(changed)),
         'new_chunk_docs_reranker_rank_gt_50': n_rr50, 'new_chunk_docs_hybrid_rank_gt_100': n_hy100,
         'new_chunk_docs_both': n_both}
    return qids, docs, new, m


def cmd_measure(a) -> int:
    out = assert_writable(a.out_dir)
    _, _, _, m = choose(a)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'rrf_measure.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print(json.dumps(m, indent=1))
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_chunks(base, out_zip, qids: list[int], new: dict, docs_dir) -> dict:
    """base JSON -> out_zip + its .json with relevant_chunks = full chunks of new[q] (in that order); the rest unchanged."""
    from r2ai.submit.make_submission import load_full_texts
    out_zip = assert_writable(out_zip)
    out_json = assert_writable(out_zip.with_suffix('.json'))
    require_inputs(base, docs_dir)
    full, src, _ = load_full_texts({d for q in qids for d in new[q]}, resolve_path(docs_dir))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    n, missing, same_text, diff_text = 0, 0, 0, 0
    with open(base, 'r', encoding='utf-8', newline='') as fi, open(out_json, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write('[\n')
        for line in fi:
            line = line.rstrip('\n')
            if line in ('[', ']'):
                continue
            row = json.loads(line.rstrip(','))
            q = row['id']
            base_text = {c['doc_id']: c['chunk_text'] for c in row['relevant_chunks']}
            chunks = []
            for d in new[q]:
                t = full.get(d, '')
                if not t.strip():
                    missing += 1
                    continue
                if d in base_text:                                     # same doc -> must be the same full chunk
                    same_text += base_text[d] == t
                    diff_text += base_text[d] != t
                chunks.append({'doc_id': int(d), 'chunk_text': t})
            row['relevant_chunks'] = chunks
            fo.write(('' if n == 0 else ',\n') + json.dumps(row, ensure_ascii=False, separators=(', ', ': ')))
            n += 1
        fo.write('\n]\n')
    if n != len(qids) or diff_text:
        raise ValueError(f'{n} queries (expected {len(qids)}), {diff_text} kept docs with a different chunk text')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out_json, out_json.name)
    return {'base': str(base), 'docs_missing_text': missing, 'kept_docs_same_chunk_text': same_text, 'full_text_sources': src,
            'json': str(out_json), 'json_bytes': out_json.stat().st_size, 'json_sha256': _sha256(out_json),
            'zip_bytes': out_zip.stat().st_size, 'zip_sha256': _sha256(out_zip), 'max_zip_bytes': 104857600,
            'zip_within_ceiling': out_zip.stat().st_size <= 104857600}


def rrf_chunk_submission(base_json, out_zip, *, runs_dir, doc_ranking, hybrid_ranks, rerank_run, queries, docs_dir, k_cache=100,
                         k_total=150, k_chunk=50, rrf_k=60, missing_rank=201) -> dict:
    """Library form of `build` (same bytes, no --min-mean gate): returns the measure + write stats."""
    qids, _, new, m = choose_docs(runs_dir=runs_dir, doc_ranking=doc_ranking, hybrid_ranks=hybrid_ranks, rerank_run=rerank_run,
                                  queries=queries, k_cache=k_cache, k_total=k_total, k_chunk=k_chunk, rrf_k=rrf_k,
                                  missing_rank=missing_rank)
    return m | write_chunks(base_json, out_zip, qids, new, docs_dir)


def cmd_build(a) -> int:
    out_zip = assert_writable(a.out)
    assert_writable(out_zip.with_suffix('.json'))
    require_inputs(a.base, a.docs_dir)
    t0 = time.time()
    qids, docs, new, m = choose(a)
    if m['chunk_docs_changed_vs_base']['mean'] < a.min_mean and not a.force:
        print(json.dumps(m, indent=1))
        print(f'mean change {m["chunk_docs_changed_vs_base"]["mean"]} < {a.min_mean} doc/query: not built', flush=True)
        return 2
    w = write_chunks(a.base, out_zip, qids, new, a.docs_dir)
    w.pop('json')
    m |= w | {'elapsed_s': round(time.time() - t0, 1)}
    out_zip.with_suffix('.stats.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print(json.dumps(m, indent=1))
    return 0 if m['zip_within_ceiling'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('measure', 'build'):
        p = sub.add_parser(name)
        p.add_argument('--runs-dir', default=DEF['runs_dir'])
        p.add_argument('--doc-ranking', default=DEF['doc_ranking'])
        p.add_argument('--rerank-run', default=DEF['rerank_run'])
        p.add_argument('--queries', default=DEF['queries'])
        p.add_argument('--k-cache', type=int, default=100)
        p.add_argument('--k-total', type=int, default=150)
        p.add_argument('--k-chunk', type=int, default=50)
        p.add_argument('--rrf-k', type=int, default=60)
        p.add_argument('--missing-rank', type=int, default=201, help='rank_reranker of a doc absent from --rerank-run')
    sub.choices['measure'].add_argument('--out-dir', required=True)
    b = sub.choices['build']
    b.add_argument('--base', required=True, help='base submission JSON (relevant_docs kept)')
    b.add_argument('--out', required=True)
    b.add_argument('--docs-dir', default=DEF['docs_dir'])
    b.add_argument('--min-mean', type=float, default=1.0)
    b.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    return cmd_measure(a) if a.cmd == 'measure' else cmd_build(a)


if __name__ == '__main__':
    sys.exit(main())
