"""VS-cc-A / VS-cc-A-union: CC longchau docs (reranker order) appended to the byte-pinned RD150 in place of / next to the slug ids.

  VS-cc-A        RD150 + [VS-append slug ids of every query that are NOT longchau docs with CC text] + the top-n CC docs by
                 max reranker logit (dense branch, not already present), n = number of slug ids removed (count preserved).
  VS-cc-A-union  RD150 + all 20 slug ids of VS-append (unchanged) + the same CC picks (dedup by doc id).
relevant_chunks stay RD150's byte for byte (gate); no GPU, no model call here (the rerank cache comes from retrieve.py).

  python -m r2ai.cc_longchau.append_cc [--variants VS-cc-A VS-cc-A-union]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from r2ai.paths import CORPUS_FILE, QUERY_FILE, ROOT, RUNS_DIR, WORK_DATA_DIR
from r2ai.zh_sample.append import _addition, _rows, pinned_base, verify_gate

from .common import Roots, atomic_json, sha256_file

RD150 = ROOT / 'out/runs/rrf-docs-2026-10-06/RD150/sub_RD150.json'
SLUG = RUNS_DIR / 'selective-check/VS-append/query_append.json'
VS_APPEND_JSON_SHA256 = '9c91c952863c4e33b992053d665d547ba1f483cd0be487047386ba09af43e2e0'
ZIP_CEILING = 104_857_600
KNOWN_ACCEPTED_ZIP = 101_870_962         # VS40, largest ZIP the leaderboard has taken (97.15 MiB)
VARIANTS = ('VS-cc-A', 'VS-cc-A-union')


def cc_ranking(rows):
    """[(doc, logit)] -> docs by max logit descending (stable)."""
    best = {}
    for doc, score in rows:
        if type(doc) is not int or not math.isfinite(score):
            raise ValueError(f'invalid cached pair {doc!r} {score!r}')
        best[doc] = max(best.get(doc, -math.inf), score)
    return sorted(best, key=lambda d: -best[d]), best


def plan_query(variant, rd150_ids, slug, cc_ids, ranked):
    """-> (tail ids appended after RD150, cc picks, removed slug ids)."""
    have = set(rd150_ids)
    replaced = [d for d in slug if d in cc_ids]
    kept = [d for d in slug if d not in cc_ids]
    n = len(replaced)
    block = set(have) | (set(kept) if variant == 'VS-cc-A' else set(slug))
    picks = []
    for d in ranked:
        if len(picks) >= n:
            break
        if d not in block:
            picks.append(d)
    if variant == 'VS-cc-A':
        return kept + picks, picks, replaced
    return list(slug) + picks, picks, []


def render(base, variant, slug_by_q, cc_ids, ranking, groups, write):
    cursor, sel = 0, {}
    for row, spans in _rows(base):
        qid = row['id']
        ranked, _ = ranking[qid]
        tail, picks, removed = plan_query(variant, row['relevant_docs'], slug_by_q[qid], cc_ids, ranked)
        ids, seen = [], set(row['relevant_docs'])
        for d in tail:
            for g in (groups[d] if d in groups else [d]):
                if g not in seen:
                    seen.add(g)
                    ids.append(g)
        end = spans['relevant_docs'][1]
        write(base[cursor:end - 1])
        write(_addition(row['relevant_docs'], ids))
        cursor = end - 1
        sel[qid] = {'appended': ids, 'cc_picks': picks, 'slug_removed': removed, 'slug_total': len(slug_by_q[qid])}
    write(base[cursor:])
    return sel


def load_inputs(roots: Roots):
    qids = pq.read_table(QUERY_FILE, columns=['id']).column('id').to_pylist()
    if len(qids) != 1200:
        raise ValueError('need 1200 queries')
    slug = {int(q): v['appended'] for q, v in json.loads(SLUG.read_text('utf-8')).items()}
    seg_dirs = sorted(p for p in roots.chunks.iterdir() if p.is_dir() and (p / 'docs.parquet').exists())
    if len(seg_dirs) != 1:
        raise ValueError(f'expected one chunked segment, found {len(seg_dirs)}')
    t = pq.read_table(seg_dirs[0] / 'docs.parquet', columns=['doc_id', 'doc_ids_group'])
    groups = dict(zip(t.column('doc_id').to_pylist(), t.column('doc_ids_group').to_pylist()))
    rr = roots.run / 'retrieval' / 'rerank'
    ranking = {}
    for qid in qids:
        r = json.loads((rr / f'q{qid}.json').read_text('utf-8'))
        if r['query_id'] != qid:
            raise ValueError(f'cache mismatch q{qid}')
        ranking[qid] = cc_ranking([(int(d), s) for d, s in r['dense']])
    return qids, slug, groups, ranking


def run_variant(roots: Roots, base, variant, qids, slug, groups, ranking):
    t0 = time.perf_counter()
    cc_ids = set(groups)
    out = roots.guard(roots.run / variant)
    out.mkdir(parents=True, exist_ok=True)
    stem = variant.replace('-', '_')
    tmp_json, tmp_zip = out / f'{stem}.json.tmp', out / f'{stem}.pending.zip'
    with open(roots.guard(tmp_json), 'w', encoding='utf-8', newline='') as f:
        sel = render(base, variant, slug, cc_ids, ranking, groups, f.write)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_json, encoding='utf-8', newline='') as f:
        gate = verify_gate(base, f.read(), sel)
    if gate['queries'] != 1200 or set(sel) != set(qids):
        raise ValueError('need the exact 1200-query byte gate')
    with zipfile.ZipFile(roots.guard(tmp_zip), 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(tmp_json, arcname=f'{stem}.json')
    from r2ai.zh_full_ingest.append_zf import validate
    validator = validate(tmp_zip, out / 'validator.log')
    final_json, final_zip = roots.guard(out / f'{stem}.json'), roots.guard(out / f'{stem}.zip')
    os.replace(tmp_json, final_json)
    os.replace(tmp_zip, final_zip)
    n_ids = [len(s['appended']) for s in sel.values()]
    slug_total = sum(s['slug_total'] for s in sel.values())
    lc_with_cc = sum(len([d for d in slug[q] if d in cc_ids]) for q in sel)
    picks = [d for s in sel.values() for d in s['cc_picks']]
    overlap = sum(d in set(slug[q]) for q, s in sel.items() for d in s['cc_picks'])
    stats = {'variant': variant, 'diagnostic_only': True, 'LB': 'chua do', 'base': str(RD150), 'gate': gate, 'validator': validator,
             'json_bytes': final_json.stat().st_size, 'json_sha256': sha256_file(final_json), 'zip_bytes': final_zip.stat().st_size,
             'zip_mib': round(final_zip.stat().st_size / 2 ** 20, 3), 'zip_sha256': sha256_file(final_zip),
             'zip_vs_largest_accepted_VS40_bytes': final_zip.stat().st_size - KNOWN_ACCEPTED_ZIP,
             'ids_appended_per_query': {'mean': statistics.fmean(n_ids), 'min': min(n_ids), 'max': max(n_ids)},
             'slug_ids_total_VS_append': slug_total, 'slug_ids_longchau_with_cc_text': lc_with_cc,
             'slug_ids_replaced': sum(len(s['slug_removed']) for s in sel.values()),
             'slug_ids_replaced_pct_of_slug': round(100 * sum(len(s['slug_removed']) for s in sel.values()) / slug_total, 2),
             'cc_picks_total': len(picks), 'cc_picks_per_query_mean': len(picks) / 1200,
             'cc_picks_also_in_slug_pct': round(100 * overlap / max(1, len(picks)), 2), 'unique_cc_docs_picked': len(set(picks)),
             'queries_with_cc_pick': sum(bool(s['cc_picks']) for s in sel.values()), 'seconds': time.perf_counter() - t0}
    atomic_json(roots, out / 'stats.json', stats)
    atomic_json(roots, out / 'query_append.json', {str(q): s for q, s in sel.items()})
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    a = ap.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    roots = Roots()
    roots.preflight()
    if sha256_file(RUNS_DIR / 'selective-check/VS-append/VS_append.json') != VS_APPEND_JSON_SHA256:
        raise ValueError('VS-append JSON SHA256 mismatch')
    base = pinned_base(RD150)
    qids, slug, groups, ranking = load_inputs(roots)
    for v in a.variants:
        s = run_variant(roots, base, v, qids, slug, groups, ranking)
        print(json.dumps({k: s[k] for k in ('variant', 'zip_mib', 'zip_sha256', 'json_sha256', 'validator', 'gate', 'ids_appended_per_query',
                                            'slug_ids_replaced_pct_of_slug', 'cc_picks_also_in_slug_pct')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
