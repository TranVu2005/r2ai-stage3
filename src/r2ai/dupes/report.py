"""H1 corpus-level duplicate statistics from the scan outputs (no corpus re-read).

    python -m r2ai.dupes.report --out-dir out/runs/H1

Reads <out>/work/features.parquet, <out>/clusters.parquet, <out>/cluster_table.parquet; writes <out>/cluster_report.json.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, resolve_path

import argparse
import json
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pyarrow.parquet as pq

BINS = [(2, 2), (3, 3), (4, 4), (5, 9), (10, 49), (50, 99), (100, 10**9)]


def size_hist(sizes) -> dict:
    sizes = list(sizes)
    out = {f'{lo}' if lo == hi else (f'{lo}-{hi}' if hi < 10**9 else f'>={lo}'): sum(lo <= s <= hi for s in sizes) for lo, hi in BINS}
    return out | {'max': max(sizes) if sizes else 0, 'mean': round(float(np.mean(sizes)), 3) if sizes else 0,
                  'p50': float(np.percentile(sizes, 50)) if sizes else 0, 'p95': float(np.percentile(sizes, 95)) if sizes else 0}


def exact_report(f) -> dict:
    n = len(f['doc_id'])
    ek, ts = f['ek'], f['tsha']
    g_norm, g_sha = Counter(ek), Counter(ts)
    first = {}
    for k, s in zip(ek, ts):
        first.setdefault(s, set()).add(k)
    split_sha = sum(len(v) > 1 for v in first.values())           # a text_sha1 group spread over several normalised groups (must be 0)
    norm_to_sha: dict = {}
    for k, s in zip(ek, ts):
        norm_to_sha.setdefault(k, set()).add(s)
    ge2 = lambda c: [v for v in c.values() if v >= 2]
    return {'docs_ok_with_body': n,
            'normalised_groups_ge2': len(ge2(g_norm)), 'normalised_docs_in_groups_ge2': int(sum(ge2(g_norm))),
            'normalised_redundant_docs': int(sum(v - 1 for v in ge2(g_norm))),
            'text_sha1_groups_ge2': len(ge2(g_sha)), 'text_sha1_docs_in_groups_ge2': int(sum(ge2(g_sha))),
            'text_sha1_redundant_docs': int(sum(v - 1 for v in ge2(g_sha))),
            'text_sha1_groups_split_by_normalisation': split_sha,
            'normalised_groups_merging_several_text_sha1': sum(len(v) > 1 for v in norm_to_sha.values()),
            'docs_gained_by_normalisation': int(sum(v - 1 for v in ge2(g_norm)) - sum(v - 1 for v in ge2(g_sha))),
            'normalised_group_size': size_hist(ge2(g_norm))}


def scope_report(c: dict, ct: dict, mask) -> dict:
    ids = [i for i, m in enumerate(ct['cluster_id']) if mask(i)]
    sizes = [ct['size'][i] for i in ids]
    sel = set(ct['cluster_id'][i] for i in ids)
    docs = [i for i, cid in enumerate(c['cluster_id']) if cid in sel]
    kinds = Counter(ct['kind'][i] for i in ids)
    doms = Counter()
    same_dom = multi_dom = 0
    pairs = Counter()
    members: dict[int, set] = {}
    for i in docs:
        members.setdefault(c['cluster_id'][i], Counter())[c['domain'][i]] += 1
    for cid, dc in members.items():
        if len(dc) == 1:
            same_dom += 1
            doms[next(iter(dc))] += sum(dc.values())
        else:
            multi_dom += 1
            for a, b in combinations(sorted(dc), 2):
                pairs[(a, b)] += 1
    return {'clusters': len(ids), 'docs': len(docs), 'redundant_docs': len(docs) - len(ids), 'size': size_hist(sizes),
            'kind': dict(kinds), 'exact_only_cluster_docs': sum(ct['size'][i] for i in ids if ct['kind'][i] == 'exact'),
            'clusters_single_domain': same_dom, 'clusters_multi_domain': multi_dom,
            'top_domains_single_domain_clusters_by_docs': doms.most_common(15),
            'top_domain_pairs_by_clusters': [[f'{a} | {b}', n] for (a, b), n in pairs.most_common(15)],
            'clusters_with_min_jaccard_lt_0_9': sum(1 for i in ids if ct['min_jaccard'][i] is not None and ct['min_jaccard'][i] < 0.9)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default='out/runs/H1')
    a = ap.parse_args(argv)
    out = assert_writable(a.out_dir)
    f = pq.read_table(out / 'work' / 'features.parquet', columns=['doc_id', 'ek', 'tsha', 'ntok', 'n_ids']).to_pydict()
    c = pq.read_table(out / 'clusters.parquet').to_pydict()
    ct = pq.read_table(out / 'cluster_table.parquet').to_pydict()
    n = len(f['doc_id'])
    res = {'exact': exact_report(f)}
    res['scopes'] = {
        'all': scope_report(c, ct, lambda i: True),
        'content': scope_report(c, ct, lambda i: not (ct['short'][i] or ct['template'][i])),
        'boilerplate': scope_report(c, ct, lambda i: ct['short'][i] or ct['template'][i]),
        'boilerplate_short': scope_report(c, ct, lambda i: ct['short'][i]),
        'boilerplate_template_not_short': scope_report(c, ct, lambda i: ct['template'][i] and not ct['short'][i]),
    }
    for k, v in res['scopes'].items():
        v['pct_of_ok_docs_in_clusters'] = round(100 * v['docs'] / n, 3)
        v['pct_of_ok_docs_redundant'] = round(100 * v['redundant_docs'] / n, 3)
    ntok = np.asarray(f['ntok'])
    res['ok_docs'] = n
    res['ok_docs_lt_50_tokens'] = int((ntok < 50).sum())
    res['ok_docs_lt_100_tokens'] = int((ntok < 100).sum())
    res['http_https_www_pairs_note'] = 'rows of docs_vi are one per url_norm (doc_ids = whole group); not counted as duplicates'
    res['docs_with_group_of_2_ids'] = int(sum(x > 1 for x in f['n_ids']))
    (out / 'cluster_report.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
