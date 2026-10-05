"""H1 on D50: duplicate-cluster impact per query, and the two variants H1-dedup / H1-expand (CPU only, no model).

    python -m r2ai.dupes.variants impact --out-dir out/runs/H1 [--scope content|all]
    python -m r2ai.dupes.variants build  --mode identity|dedup|expand --out out/runs/H1/<name>/sub.zip [--scope content|all]
    python -m r2ai.dupes.variants diff   --base D50.json --variant V.json --mode dedup|expand

The ranked doc lists are rebuilt exactly as r2ai.submit.make_submission does for D50 (configs/submission-best.yaml):
  top[q]   rank <= 100 rows of <runs_dir>/vi_k100.parquet in rank order (deep-rerank order),
  pool[q]  rows of <doc_ranking> (hybrid order) that are not in top[q], in rank order,
  D50 docs = top[q] + pool[q][:150 - len(top[q])]; relevant_docs = their doc_ids_group, unique, in that order;
  relevant_chunks = whole-doc chunk (make_submission.load_full_texts) of the first 50 docs, rank order.
identity  that list unchanged: must reproduce the D50 JSON byte for byte (SHA256 gate).
dedup     cluster-aware list: walk top[q] + pool[q] (the order above), drop a doc whose cluster already has a kept doc, stop at
          150 kept docs; chunks of the first 50 kept docs. relevant_docs/relevant_chunks change only through this list.
expand    D50 unchanged, then the doc_ids (whole doc_ids_group) of every doc in a cluster with a top-150 doc are appended to
          relevant_docs (anchor rank order, then doc_id); relevant_chunks untouched.
scope     content: clusters not flagged boilerplate (short / same-template); all: every cluster of size >= 2.
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

from r2ai.dupes.scan import rss_gib

BEST = {'runs_dir': 'out/runs/deep-rerank-2026-10-05/full', 'doc_ranking': 'out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet',
        'queries': 'D:/GitHub/r2ai-stage3-old/data/raw/query.parquet', 'chunks_dir': 'data/chunks', 'docs_dir': 'data/docs_vi',
        'k_cache': 100, 'k_total': 150, 'k_chunk': 50}


def load_lists(runs_dir, doc_ranking, qids, k_cache=100):
    """(top, pool) per query, as make_submission builds them (rows with rank <= k_cache; ranking docs not in the cache)."""
    run = pq.read_table(resolve_path(runs_dir) / 'vi_k100.parquet').to_pylist()
    top = {q: [] for q in qids}
    for r in sorted(run, key=lambda r: (r['query_id'], r['rank'])):
        if r['rank'] <= k_cache:
            top[r['query_id']].append(r['doc_id'])
    pool = {q: [] for q in qids}
    rk = pq.read_table(resolve_path(doc_ranking), columns=['query_id', 'rank', 'doc_id']).to_pandas().sort_values(['query_id', 'rank'], kind='stable')
    seen = {q: set(v) for q, v in top.items()}
    for q, d in zip(rk['query_id'].tolist(), rk['doc_id'].tolist()):
        if q in pool and d not in seen[q]:
            pool[q].append(d)
            seen[q].add(d)
    return top, pool


def d50_docs(top, pool, k_total=150):
    return {q: top[q] + pool[q][:max(0, k_total - len(top[q]))] for q in top}


def load_clusters(path, scope='content'):
    """doc_id -> cluster id (clusters of size >= 2 in scope), cluster id -> member doc_ids sorted."""
    t = pq.read_table(resolve_path(path), columns=['doc_id', 'cluster_id', 'boilerplate']).to_pydict()
    of, members = {}, {}
    for d, c, b in zip(t['doc_id'], t['cluster_id'], t['boilerplate']):
        if scope == 'all' or not b:
            of[d] = c
            members.setdefault(c, []).append(d)
    for m in members.values():
        m.sort()
    return of, members


def load_groups(chunks_dir):
    return {r['doc_id']: r['doc_ids_group'] for r in pq.read_table(resolve_path(chunks_dir) / 'docs.parquet', columns=['doc_id', 'doc_ids_group']).to_pylist()}


def dedup_lists(top, pool, of, k_total=150):
    """Cluster-aware doc list per query: first doc of each cluster in (top + pool) order, up to k_total docs."""
    out, dropped, short = {}, {}, 0
    for q in top:
        seen, keep, drop = set(), [], 0
        for d in top[q] + pool[q]:
            c = of.get(d)
            if c is not None:
                if c in seen:
                    drop += 1
                    continue
                seen.add(c)
            keep.append(d)
            if len(keep) == k_total:
                break
        out[q], dropped[q] = keep, drop
        short += len(keep) < k_total
    return out, dropped, short


def expand_lists(docs, of, members, groups):
    """Per query: ids (whole groups) of cluster mates of the docs that are not already in relevant_docs, anchor order."""
    out = {}
    for q, lst in docs.items():
        have = {int(x) for d in lst for x in groups[d]}
        extra, seen = [], set()
        for d in lst:
            c = of.get(d)
            if c is None or c in seen:
                continue
            seen.add(c)
            for m in members[c]:
                for x in groups.get(m, [m]):
                    if int(x) not in have:
                        have.add(int(x))
                        extra.append(int(x))
        out[q] = extra
    return out


def dist(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    return {'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)), 'mean': round(float(x.mean()), 3),
            'max': float(x.max()), 'min': float(x.min())}


def cmd_impact(a) -> int:
    out = assert_writable(a.out_dir)
    qids = sorted(int(x) for x in pq.read_table(resolve_path(a.queries), columns=['id'])['id'].to_pylist())
    top, pool = load_lists(a.runs_dir, a.doc_ranking, qids, a.k_cache)
    docs = d50_docs(top, pool, a.k_total)
    feats = pq.read_table(out / 'work' / 'features.parquet', columns=['doc_id', 'ck', 'n_ids']).to_pydict()
    ck = dict(zip(feats['doc_id'], feats['ck']))
    nid = dict(zip(feats['doc_id'], feats['n_ids']))
    groups = load_groups(a.chunks_dir)
    res = {}
    per = []
    for scope in ('content', 'all'):
        of, members = load_clusters(out / 'clusters.parquet', scope)
        wasted, wasted50, wasted100, chunk_same_cluster, chunk_same_text, extra_docs, extra_ids, extra_docs50 = ([] for _ in range(8))
        in_cluster, queries_hit = [], 0
        for q in qids:
            lst = docs[q]
            seen, w = set(), 0
            w_by_rank = []
            for d in lst:
                c = of.get(d)
                if c is not None and c in seen:
                    w += 1
                elif c is not None:
                    seen.add(c)
                w_by_rank.append(w)
            wasted.append(w)
            wasted100.append(w_by_rank[min(99, len(lst) - 1)])
            wasted50.append(w_by_rank[min(49, len(lst) - 1)])
            in_cluster.append(sum(d in of for d in lst))
            queries_hit += w > 0
            ch = lst[:a.k_chunk]
            seen_c, n_same_cluster = set(), 0
            for d in ch:
                c = of.get(d)
                if c is not None and c in seen_c:
                    n_same_cluster += 1
                elif c is not None:
                    seen_c.add(c)
            chunk_same_cluster.append(n_same_cluster)
            chunk_same_text.append(len(ch) - len({ck[d] for d in ch}))
            inl = set(lst)
            mates, ids, mates50 = set(), 0, set()
            for rank, d in enumerate(lst):
                c = of.get(d)
                if c is None:
                    continue
                for m in members[c]:
                    if m not in inl and m not in mates:
                        mates.add(m)
                        ids += len(groups.get(m, [m]))
                        if rank < a.k_chunk:
                            mates50.add(m)
            extra_docs.append(len(mates))
            extra_ids.append(ids)
            extra_docs50.append(len(mates50))
        res[scope] = {
            'a_slots_in_150_taken_by_same_cluster_docs': dist(wasted), 'a_within_top100': dist(wasted100), 'a_within_top50': dist(wasted50),
            'a_docs_of_top150_that_are_in_a_cluster': dist(in_cluster),
            'a_queries_with_at_least_one_wasted_slot': queries_hit, 'a_total_wasted_slots': int(sum(wasted)),
            'b_full_chunks_top50_same_cluster_as_earlier': dist(chunk_same_cluster), 'b_full_chunks_top50_identical_text': dist(chunk_same_text),
            'b_queries_with_identical_chunk_text': int(sum(x > 0 for x in chunk_same_text)),
            'c_cluster_mates_outside_top150_docs': dist(extra_docs), 'c_cluster_mates_outside_top150_ids_with_groups': dist(extra_ids),
            'c_mates_of_top50_docs_outside_top150': dist(extra_docs50), 'c_total_mates': int(sum(extra_docs)),
        }
        per.append((scope, wasted, chunk_same_cluster, chunk_same_text, extra_docs, extra_ids))
    res['meta'] = {'queries': len(qids), 'k_total': a.k_total, 'k_chunk': a.k_chunk, 'docs_per_query': dist(len(v) for v in docs.values()),
                   'source': {'runs_dir': a.runs_dir, 'doc_ranking': a.doc_ranking}, 'peak_rss_gib': round(rss_gib(), 3)}
    (out / 'impact_d50.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    with open(out / 'impact_d50_per_query.csv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('query_id,' + ','.join(f'{s}_{k}' for s, *_ in per for k in ('wasted', 'chunk_same_cluster', 'chunk_same_text', 'mates_docs', 'mates_ids')) + '\n')
        for i, q in enumerate(qids):
            f.write(f'{q},' + ','.join(str(v[i]) for _, *cols in per for v in cols) + '\n')
    print(json.dumps(res, indent=1))
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cmd_build(a) -> int:
    from r2ai.submit.make_submission import load_full_texts
    out_zip = assert_writable(a.out)
    out_json = assert_writable(out_zip.with_suffix('.json'))
    require_inputs(a.queries, resolve_path(a.runs_dir) / 'vi_k100.parquet', a.doc_ranking, resolve_path(a.chunks_dir) / 'docs.parquet', a.docs_dir)
    t0 = time.time()
    qids = sorted(int(x) for x in pq.read_table(resolve_path(a.queries), columns=['id'])['id'].to_pylist())
    top, pool = load_lists(a.runs_dir, a.doc_ranking, qids, a.k_cache)
    groups = load_groups(a.chunks_dir)
    base = d50_docs(top, pool, a.k_total)
    stats = {'mode': a.mode, 'scope': a.scope}
    extra = {q: [] for q in qids}
    docs = base
    if a.mode in ('dedup', 'expand'):
        out_dir = assert_writable(a.out_dir)
        of, members = load_clusters(out_dir / 'clusters.parquet', a.scope)
    if a.mode == 'dedup':
        docs, dropped, short = dedup_lists(top, pool, of, a.k_total)
        changed = [q for q in qids if docs[q] != base[q]]
        stats |= {'queries_changed': len(changed), 'docs_dropped_total': int(sum(dropped.values())), 'queries_short_of_k_total': short,
                  'dropped_per_query': dist(dropped.values()),
                  'docs_new_vs_d50_per_query': dist(len(set(docs[q]) - set(base[q])) for q in qids)}
    elif a.mode == 'expand':
        extra = expand_lists(base, of, members, groups)
        stats |= {'queries_changed': sum(bool(extra[q]) for q in qids), 'ids_added_total': int(sum(map(len, extra.values()))),
                  'ids_added_per_query': dist(map(len, extra.values()))}
    need = {d for q in qids for d in docs[q][:a.k_chunk]}
    full, src, _ = load_full_texts(need, resolve_path(a.docs_dir))
    stats['peak_rss_gib_after_texts'] = round(rss_gib(), 3)
    n_docs, n_chunks, missing = [], [], 0
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8', newline='\n') as f:        # same bytes as writer.format_submission, one query at a time
        f.write('[\n')
        for i, q in enumerate(qids):
            ids = list(dict.fromkeys(int(x) for d in docs[q] for x in groups[d])) + extra[q]
            chunks = []
            for d in docs[q][:a.k_chunk]:
                t = full.get(d, '')
                if t.strip():
                    chunks.append({'doc_id': int(d), 'chunk_text': t})
                else:
                    missing += 1
            f.write(('' if i == 0 else ',\n') + json.dumps({'id': q, 'relevant_docs': ids, 'relevant_chunks': chunks}, ensure_ascii=False, separators=(', ', ': ')))
            n_docs.append(len(ids))
            n_chunks.append(len(chunks))
        f.write('\n]\n')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out_json, out_json.name)
    stats |= {'queries': len(qids), 'docs_per_query': dist(n_docs), 'chunks_per_query': dist(n_chunks), 'docs_missing_text': missing,
              'json_bytes': out_json.stat().st_size, 'json_sha256': sha256(out_json), 'zip_bytes': out_zip.stat().st_size,
              'zip_sha256': sha256(out_zip), 'max_zip_bytes': 104857600, 'zip_within_ceiling': out_zip.stat().st_size <= 104857600,
              'elapsed_s': round(time.time() - t0, 1), 'peak_rss_gib': round(rss_gib(), 3), 'full_text_source_counts': src}
    out_zip.with_suffix('.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))
    return 0


def _lines(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('[') and len(line) <= 2 or line.rstrip('\n') == ']':
                continue
            yield line.rstrip('\n').rstrip(',')


def cmd_diff(a) -> int:
    """Per-query comparison of two submission JSONs (one query per line). Exit 1 when anything outside the intended scope
    differs: expand may only append ids to relevant_docs; dedup may only change relevant_docs / relevant_chunks, and every
    variant chunk must belong to a doc of its relevant_docs."""
    res = {'queries': 0, 'identical_lines': 0, 'changed_queries': 0, 'same_keys_and_ids': True, 'docs_changed': 0, 'chunks_changed': 0,
           'docs_prefix_kept_and_ids_appended': 0, 'docs_count_base': 0, 'docs_count_variant': 0}
    bad = []
    for lb, lv in zip(_lines(a.base), _lines(a.variant)):
        res['queries'] += 1
        if lb == lv:
            res['identical_lines'] += 1
            continue
        b, v = json.loads(lb), json.loads(lv)
        res['changed_queries'] += 1
        if b['id'] != v['id'] or set(b) != set(v):
            res['same_keys_and_ids'] = False
            bad.append(b['id'])
        res['docs_changed'] += b['relevant_docs'] != v['relevant_docs']
        res['chunks_changed'] += b['relevant_chunks'] != v['relevant_chunks']
        res['docs_count_base'] += len(b['relevant_docs'])
        res['docs_count_variant'] += len(v['relevant_docs'])
        appended = v['relevant_docs'][:len(b['relevant_docs'])] == b['relevant_docs'] and len(v['relevant_docs']) > len(b['relevant_docs'])
        res['docs_prefix_kept_and_ids_appended'] += appended
        if a.mode == 'expand' and (b['relevant_chunks'] != v['relevant_chunks'] or not appended):
            bad.append(b['id'])
        if a.mode == 'dedup' and not {c['doc_id'] for c in v['relevant_chunks']} <= set(v['relevant_docs']):
            bad.append(b['id'])
    res['violations_outside_scope'] = sorted(set(bad))
    print(json.dumps(res, indent=1))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('impact', 'build'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', default='out/runs/H1', help='H1 run dir with clusters.parquet and work/')
        p.add_argument('--scope', choices=('content', 'all'), default='content')
        p.add_argument('--runs-dir', default=BEST['runs_dir'])
        p.add_argument('--doc-ranking', default=BEST['doc_ranking'])
        p.add_argument('--queries', default=BEST['queries'])
        p.add_argument('--chunks-dir', default=BEST['chunks_dir'])
        p.add_argument('--docs-dir', default=BEST['docs_dir'])
        p.add_argument('--k-cache', type=int, default=BEST['k_cache'])
        p.add_argument('--k-total', type=int, default=BEST['k_total'])
        p.add_argument('--k-chunk', type=int, default=BEST['k_chunk'])
    sub.choices['build'].add_argument('--mode', choices=('identity', 'dedup', 'expand'), required=True)
    sub.choices['build'].add_argument('--out', required=True)
    d = sub.add_parser('diff')
    d.add_argument('--base', required=True)
    d.add_argument('--variant', required=True)
    d.add_argument('--mode', choices=('dedup', 'expand'), required=True)
    a = ap.parse_args(argv)
    return {'impact': cmd_impact, 'build': cmd_build, 'diff': cmd_diff}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
