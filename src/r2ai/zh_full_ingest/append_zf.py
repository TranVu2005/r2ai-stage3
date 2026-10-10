"""CPU-only: append <= k zh-full docs/query (reranker order) to the byte-pinned VS-append.

Cache = rerank of the 1,200 queries over every ACTIVE zh-full segment (sample-v2 + b001), branch `dense`
(same branch as Z-append). Doc = max cached chunk logit; the doc's whole doc_ids_group is appended, primary first.
Nothing before the appended ids changes (byte gate); no GPU, no model call, no LB.
Run from D:\\GitHub\\r2ai-stage3:
  .venv\\Scripts\\python.exe -B -m r2ai.zh_full_ingest.append_zf --k 100 10   (PYTHONPATH=<worktree>/src)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sqlite3
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from r2ai.paths import QUERY_FILE, CORPUS_FILE, RUNS_DIR, WORK_DATA_DIR

from .common import Roots, atomic_json, exclusive, sha256_file

VS_APPEND_SHA256 = '9c91c952863c4e33b992053d665d547ba1f483cd0be487047386ba09af43e2e0'
BASE = RUNS_DIR / 'selective-check/VS-append/VS_append.json'
ZIP_CEILING = 104_857_600
BANDS = ((1, 10), (11, 50), (51, 100))


def select_zf(pairs, groups, present, k):
    """Top-k docs by max logit that are not in `present`; returns (ids, docs, n_available)."""
    best = {}
    for doc, score in pairs:
        if type(doc) is not int or doc not in groups or type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f'Unknown zh primary id or invalid score: {doc!r}')
        best[doc] = max(best.get(doc, -math.inf), score)
    have, ids, docs, available = set(present), [], [], 0
    for doc in sorted(best, key=lambda d: -best[d]):          # stable: cache order breaks ties
        if doc in have:
            continue
        available += 1
        if len(docs) < k:
            new = [i for i in groups[doc] if i not in have]
            ids.extend(new)
            have.update(new)
            docs.append(doc)
    return ids, docs, available


def render(base, cache, groups, k, write):
    from r2ai.zh_sample.append import _rows, _addition
    cursor, selections = 0, {}
    for row, spans in _rows(base):
        qid = row['id']
        ids, docs, available = select_zf(cache[qid], groups, row['relevant_docs'], k)
        end = spans['relevant_docs'][1]
        write(base[cursor:end - 1])
        write(_addition(row['relevant_docs'], ids))
        cursor = end - 1
        selections[qid] = {'appended': ids, 'docs': docs, 'available': available}
    if set(selections) != set(cache):
        raise ValueError('Cached query ids differ from the base query ids')
    write(base[cursor:])
    return selections


def load_inputs(roots: Roots):
    rr = roots.run / 'retrieval-b001' / 'rerank'
    cfg = json.loads((rr / 'config.json').read_text('utf-8'))
    import pyarrow.parquet as pq
    qids = pq.read_table(QUERY_FILE, columns=['id']).column('id').to_pylist()
    if len(qids) != 1200 or len(set(qids)) != 1200:
        raise ValueError('Need 1200 unique query ids')
    c = sqlite3.connect(f'file:{roots.manifest.as_posix()}?mode=ro', uri=True)
    try:
        active = {r[0]: (r[1], r[2]) for r in c.execute("SELECT name, batch, domain FROM segments WHERE status='active'")}
    finally:
        c.close()
    if set(cfg['segments']) != set(active):
        raise ValueError('Rerank cache segments differ from the ACTIVE segments')
    groups, owner = {}, {}
    for name in cfg['segments']:
        t = pq.read_table(roots.chunks / name / 'docs.parquet', columns=['doc_id', 'doc_ids_group'])
        for d, g in zip(t.column('doc_id').to_pylist(), t.column('doc_ids_group').to_pylist()):
            if d in groups or g[0] != d:
                raise ValueError(f'Duplicate doc_id or group without primary first: {d}')
            groups[d], owner[d] = g, active[name]
    cache = {}
    for qid in qids:
        row = json.loads((rr / f'q{qid}.json').read_text('utf-8'))
        if row['query_id'] != qid:
            raise ValueError(f'Cache query mismatch q{qid}')
        cache[qid] = [(int(d), s) for d, s in row['dense']]
    return cfg, groups, owner, cache, qids


def summarise(selections, owner, merged_top, k):
    n_docs = [len(s['docs']) for s in selections.values()]
    pct = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    by_domain, by_batch, bands = Counter(), Counter(), {str(b): Counter() for b in BANDS}
    in_merged = total = 0
    for qid, s in selections.items():
        for rank, d in enumerate(s['docs'], 1):
            batch, domain = owner[d]
            by_domain[domain] += 1
            by_batch[batch] += 1
            total += 1
            in_merged += d in merged_top.get(qid, ())
            for b in BANDS:
                if b[0] <= rank <= b[1]:
                    bands[str(b)][domain] += 1
    share = lambda c: {k: {'docs': v, 'share': round(v / total, 4)} for k, v in c.most_common()}
    return {'docs_per_query': {'mean': statistics.fmean(n_docs), 'min': min(n_docs), 'p5': pct(n_docs, .05),
                               'p50': pct(n_docs, .5), 'max': max(n_docs), 'queries_below_k': sum(n < k for n in n_docs)},
            'ids_appended_total': sum(len(s['appended']) for s in selections.values()), 'docs_appended_total': total,
            'by_domain': share(by_domain), 'by_batch': share(by_batch),
            'rank_band_domain_share': {b: {k: round(v / sum(c.values()), 4) for k, v in c.most_common()} for b, c in bands.items() if c},
            'appended_docs_inside_merged_top150_by_logit': {'docs': in_merged, 'share': round(in_merged / total, 4)}}


def rank_logits(selections, cache):
    out = {}
    for rank in (1, 2, 5, 10, 25, 50, 100):
        v = []
        for qid, s in selections.items():
            if len(s['docs']) >= rank:
                d = s['docs'][rank - 1]
                v.append(max(sc for dd, sc in cache[qid] if dd == d))
        if v:
            v.sort()
            out[str(rank)] = {'queries': len(v), 'p5': v[int(.05 * len(v))], 'p50': v[len(v) // 2], 'p95': v[min(len(v) - 1, int(.95 * len(v)))]}
    return out


def validate(zip_path, log_path):
    from r2ai.submit.validate_submission import main as run
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        code = run([str(zip_path), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                    '--docs-dir', str(WORK_DATA_DIR / 'docs_vi'), '--max-zip-mib', '100'])
    text = capture.getvalue()
    Path(log_path).write_text(text, encoding='utf-8')
    result = json.loads(text.strip())
    if code != 0 or result.get('errors') != 0 or result.get('ok') is not True or result.get('rows') != 1200:
        raise ValueError(f'Validator failed ({code}): {text[:300]}')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, nargs='+', default=[100, 10])
    a = ap.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    roots = Roots()
    roots.preflight()
    raw = BASE.read_bytes()
    if __import__('hashlib').sha256(raw).hexdigest() != VS_APPEND_SHA256:
        raise ValueError('VS-append SHA256 mismatch; refusing mutable baseline')
    base = raw.decode('utf-8')
    del raw
    cfg, groups, owner, cache, qids = load_inputs(roots)
    merged = json.loads((roots.run / 'retrieval-b001/rerank/dense_merged.json').read_text('utf-8'))
    merged_top = {int(q): {d for d, _ in rows if d in groups} for q, rows in merged.items()}
    from r2ai.zh_sample.append import verify_gate
    with exclusive(roots, 'append'):
        for k in a.k:
            t0 = time.perf_counter()
            out = roots.guard(roots.run / 'ZF' / f'ZF-{k}')
            out.mkdir(parents=True, exist_ok=True)
            name = f'ZF_{k}'
            tmp_json, tmp_zip = out / f'{name}.json.tmp', out / f'{name}.pending.zip'
            with open(roots.guard(tmp_json), 'w', encoding='utf-8', newline='') as f:
                selections = render(base, cache, groups, k, f.write)
                f.flush()
                os.fsync(f.fileno())
            with open(tmp_json, encoding='utf-8', newline='') as f:
                gate = verify_gate(base, f.read(), selections)
            if gate['queries'] != 1200 or set(selections) != set(qids):
                raise ValueError('Need the exact 1200-query byte gate')
            with zipfile.ZipFile(roots.guard(tmp_zip), 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.write(tmp_json, arcname=f'{name}.json')
            if tmp_zip.stat().st_size > ZIP_CEILING:
                raise ValueError(f'ZIP exceeds {ZIP_CEILING} bytes: {tmp_zip.stat().st_size}')
            validator = validate(tmp_zip, out / 'validator.log')
            final_json, final_zip = roots.guard(out / f'{name}.json'), roots.guard(out / f'{name}.zip')
            atomic_json(roots, out / 'stats.json', {'complete': False, 'status': 'publishing validated staging artifacts'})
            os.replace(tmp_json, final_json)
            os.replace(tmp_zip, final_zip)
            stats = {'diagnostic_only': True, 'LB': 'chua do', 'k': k, 'base': str(BASE), 'base_sha256': VS_APPEND_SHA256,
                     'branch': 'dense', 'aggregation': 'max cached reranker chunk logit per primary doc; whole doc_ids_group appended',
                     'rerank_config_segments': len(cfg['segments']), 'rerank_candidates_pkl_sha256': cfg['candidates_pkl_sha256'],
                     'gate': gate, 'validator': validator, 'json_bytes': final_json.stat().st_size, 'json_sha256': sha256_file(final_json),
                     'zip_bytes': final_zip.stat().st_size, 'zip_sha256': sha256_file(final_zip), 'zip_ceiling_bytes': ZIP_CEILING,
                     **summarise(selections, owner, merged_top, k), 'logit_at_zh_rank': rank_logits(selections, cache),
                     'available_zh_not_in_base': {'mean': statistics.fmean(s['available'] for s in selections.values()),
                                                  'min': min(s['available'] for s in selections.values())},
                     'new_model_calls': 0, 'new_gpu_seconds': 0, 'seconds': time.perf_counter() - t0, 'complete': True}
            atomic_json(roots, out / 'stats.json', stats)
            atomic_json(roots, out / 'query_append.json', {str(q): s for q, s in selections.items()})
            print(json.dumps({k_: stats[k_] for k_ in ('k', 'zip_bytes', 'zip_sha256', 'json_sha256', 'validator', 'gate', 'docs_per_query')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
