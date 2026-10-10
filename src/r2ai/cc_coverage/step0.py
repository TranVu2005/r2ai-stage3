"""Step 0 of the Common Crawl coverage measurement for nhathuoclongchau.com.vn (index metadata only; no WARC, no site request).

    python -m r2ai.cc_coverage.step0 --ua "<agent string>" [--only CC-MAIN-2026-39]

Measured layout (2026-10-11, CC-MAIN-2026-39): 300 parquet files, each a disjoint key range, but file order is NOT key order, so a
binary search over footers does not work. Per crawl: read the footer of every file (one 64 KiB tail request each) to find the row
group(s) whose url_surtkey statistics overlap the domain prefix, then read only those row groups and only the needed columns.
The footer map of a crawl is cached (footer_map_<crawl>.json) and verified against the footer of the file read.
Stops at the first anomaly (403/429, a cap, an oversized row group, missing columns) and says why.
Writes only below out/runs/vi-longchau/cc-coverage; ledger.json is cumulative across runs (caps 2 GB / 5,000 requests overall).
"""
from __future__ import annotations

from r2ai.paths import CORPUS_FILE, LEGACY_OUT_DIR, RUNS_DIR, assert_writable

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import choose as CH
from . import crawls as CR
from . import index_scan as IX
from .net import Blocked, BudgetExceeded, Client, InvalidResponse, Ledger, Unavailable
from .surt import surt_key

OUT = RUNS_DIR / 'vi-longchau' / 'cc-coverage'
HOST_PREFIX = 'vn,com,nhathuoclongchau)/'
COLLINFO = 'https://index.commoncrawl.org/collinfo.json'
STEP_MAX_BYTES, STEP_MAX_REQUESTS, RG_CAP = 600_000_000, 800, 250_000_000


def corpus_urls():
    t = pq.read_table(CORPUS_FILE, columns=['id', 'url']).to_pylist()
    return [(r['id'], r['url']) for r in t if 'nhathuoclongchau.com.vn' in r['url']]


def old_key(url: str) -> str:
    s = urlsplit(url)
    return (s.hostname or '').removeprefix('www.') + s.path


def old_sample():
    """(id, url, old cc_status, crawl of the old CC record or None) for the 50 sample URLs of the 2026-10-02 run (read only)."""
    with open(LEGACY_OUT_DIR / 'archive_coverage.csv', encoding='utf-8') as f:
        rows = [(int(r['id']), r['url'], r['cc_status']) for r in csv.DictReader(f) if 'nhathuoclongchau' in r['domain']]
    lookup = json.loads((LEGACY_OUT_DIR / 'archive_domain_indexes' / 'nhathuoclongchau.com.vn.json').read_text(encoding='utf-8'))['cc_lookup']
    return [(i, u, st, (lookup.get(old_key(u)) or {}).get('crawl')) for i, u, st in rows]


def records_of(table: pa.Table, crawl: str):
    cols = table.to_pydict()
    return [{'crawl': crawl, **{c: cols[c][i] for c in cols}} for i in range(table.num_rows)]


def scan_crawl(client: Client, crawl: str, out: Path) -> dict:
    t0, b0, r0 = time.time(), client.ledger.bytes, client.ledger.requests
    paths = IX.list_parquet_paths(client, crawl)
    info = {'crawl': crawl, 'parquet_files': len(paths)}
    if not paths:
        return info | {'status': 'no_files'}
    cache = out / f'footer_map_{crawl}.json'
    if cache.exists():
        fm = json.loads(cache.read_text(encoding='utf-8'))
        info['footer_map_from_cache'] = True
    else:
        fm = IX.footer_map(client, paths, HOST_PREFIX)
        cache.write_text(json.dumps(fm, ensure_ascii=False), encoding='utf-8')
        info['footer_map_from_cache'] = False
    ranged = sorted((f for f in fm['file_list'] if f['range']), key=lambda f: f['range'][0])
    info['footer_map'] = {k: v for k, v in fm.items() if k not in ('file_list', 'row_group_hits')}
    info['file_index_in_key_order'] = [f['i'] for f in ranged] == sorted(f['i'] for f in ranged)
    info['row_group_hits'] = fm['row_group_hits']
    if len(paths) != fm['files']:
        return info | {'status': 'footer_map_mismatch'}
    cost = sum(h['cost_columns_bytes'] for h in fm['row_group_hits'])
    if cost > RG_CAP:
        return info | {'status': 'row_groups_too_big', 'cost_columns_bytes': cost}
    rows_file = out / f'rows_{crawl}.parquet'
    if rows_file.exists():                                    # rows of the domain already read in an earlier run
        table = pq.read_table(rows_file)
        tables = None
    else:
        table = None
        tables = []
    for fi in sorted({h['file'] for h in fm['row_group_hits']}) if tables is not None else []:
        pf, rf = IX.open_parquet(client, paths[fi])
        want = next(f for f in fm['file_list'] if f['i'] == fi)
        got = IX.file_range(IX.key_ranges(pf))
        if rf.size != want['file_bytes'] or list(got) != list(want['range']):
            return info | {'status': 'footer_changed_since_map', 'file': fi}
        for h in (h for h in fm['row_group_hits'] if h['file'] == fi):
            t = pf.read_row_group(h['row_group'], columns=list(IX.COLUMNS), use_threads=False)
            tables.append(t.filter(pc.starts_with(t['url_surtkey'], HOST_PREFIX)))
    if table is None:
        table = pa.concat_tables(tables) if tables else pa.table({c: [] for c in IX.COLUMNS})
        pq.write_table(table, assert_writable(rows_file))

    recs = records_of(table, crawl)
    return info | {'status': 'ok', 'rows_from_cache': tables is None, 'domain_rows': table.num_rows, 'usable_rows_200_html': sum(1 for r in recs if CH.usable(r)),
                   'revisit_rows': sum(1 for r in recs if r['content_mime_type'] == 'warc/revisit'),
                   'non_200_rows': sum(1 for r in recs if str(r['fetch_status']) != '200'),
                   'distinct_urls_200_html': len({r['url_surtkey'] for r in recs if CH.usable(r)}),
                   'bytes': client.ledger.bytes - b0, 'requests': client.ledger.requests - r0, 'seconds': round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ua', required=True, help='User-Agent string (descriptive; no contact unless the user supplies one)')
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--only', default='', help='comma list of crawl ids (default: the six crawls of the old sample, newest first)')
    a = ap.parse_args(argv)
    out = assert_writable(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / 'ledger.json')
    client = Client(ua=a.ua, ledger=ledger, log_path=out / 'requests.jsonl', rps=1.0)
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'ua': a.ua, 'crawls': {}}
    start_b, start_r, t_all = ledger.bytes, ledger.requests, time.time()

    def finish(status, **kw):
        report.update({'status': status, **kw, 'ledger': {'bytes': ledger.bytes, 'requests': ledger.requests},
                       'step_bytes': ledger.bytes - start_b, 'step_requests': ledger.requests - start_r,
                       'seconds': round(time.time() - t_all, 1)})
        (out / 'step0_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ('crawls', 'selected_crawls')}, ensure_ascii=False, default=str)[:1500])
        return 0 if status == 'ok' else 2

    try:
        info = json.loads(client.get(COLLINFO).content)
        sel = CR.select_crawls(info)
        report.update({'collinfo_total': len(info), 'selected_crawls': sel})
        ids = {c['id'] for c in info}
        want = [c for c in a.only.split(',') if c] or list(reversed(CR.OLD_SIX))
        todo = [c for c in want if c in ids]
        report['requested_crawls_missing_in_collinfo'] = [c for c in want if c not in ids]
        for crawl in todo:
            res = scan_crawl(client, crawl, out)
            report['crawls'][crawl] = res
            print(crawl, res['status'], res.get('domain_rows'), res.get('bytes'), res.get('requests'), flush=True)
            if res['status'] != 'ok':
                return finish('stopped_layout', reason=f"{crawl}: {res['status']}")
            if ledger.bytes - start_b > STEP_MAX_BYTES or ledger.requests - start_r > STEP_MAX_REQUESTS:
                return finish('stopped_step_cap', reason='step cap reached')
    except Blocked as e:
        return finish('stopped_blocked', reason=str(e))
    except BudgetExceeded as e:
        return finish('stopped_budget', reason=str(e))
    except (Unavailable, InvalidResponse, IX.TooBig, FileNotFoundError) as e:
        return finish('stopped_error', reason=f'{type(e).__name__}: {e}')
    finally:
        client.close()

    # sample comparison and by-product coverage of the processed crawls over all corpus URLs (no extra request)
    done = list(report['crawls'])
    by_key = {}
    for c in done:
        for r in records_of(pq.read_table(out / f'rows_{c}.parquet'), c):
            by_key.setdefault(r['url_surtkey'], []).append(r)
    cmp = []
    for i, url, old, old_crawl in old_sample():
        recs = [r for r in by_key.get(surt_key(url), []) if CH.usable(r)]
        cmp.append({'id': i, 'url': url, 'old_cc_status': old, 'old_crawl': old_crawl, 'new_found_in_processed': bool(recs),
                    'new_crawls': sorted({r['crawl'] for r in recs})})
    in_scope = [x for x in cmp if x['old_crawl'] in done]
    report['sample_vs_old'] = {
        'processed_crawls': done, 'n': len(cmp), 'old_found_any_crawl': sum(x['old_cc_status'] == 'found' for x in cmp),
        'old_records_in_processed_crawls': len(in_scope), 'of_those_found_again': sum(x['new_found_in_processed'] for x in in_scope),
        'new_found_in_processed': sum(x['new_found_in_processed'] for x in cmp),
        'new_found_but_old_not_found': [x['url'] for x in cmp if x['old_cc_status'] != 'found' and x['new_found_in_processed']],
        'old_in_scope_not_found_again': [x['url'] for x in in_scope if not x['new_found_in_processed']]}
    corpus = corpus_urls()
    prefix_of = {i: ('/benh' if '/benh/' in u else '/bai-viet') for i, u in corpus}
    chosen = {i: CH.choose(by_key.get(surt_key(u), []), CR.CORPUS_DATE) for i, u in corpus}
    report['by_product_processed_crawls'] = CH.summarize(chosen, prefix_of)
    return finish('ok')


if __name__ == '__main__':
    sys.exit(main())
