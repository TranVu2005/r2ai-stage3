"""Common Crawl INDEX scan (metadata only) for v2 zh URLs the crawler may not fetch (S11 D13 item 3c / 4).

Reads the columnar index (parquet, `subset=warc`) of the 13 crawls of the corpus window (r2ai.cc_coverage.crawls): per crawl one
footer per file (cached once in footers_<crawl>.json, so a later host needs no footer request), then only the row groups whose
key statistics overlap a wanted SURT prefix, 7 columns. Never requests a WARC and never contacts the target site: S11 does NOT
download WARC; `cc_queue.parquet` (URL, crawl, WARC file, offset, length) is handed to the S3 fetcher.
Rate <= 2 req/s through the metered cc_coverage Client (ledger caps 5,000 requests / 2 GB per ledger file, backoff, stops on 403/429).

    python -m r2ai.zh_v2.cc_queue scan --hosts zysjonline.com baidu.com [--rps 2]
    python -m r2ai.zh_v2.cc_queue queue --hosts zysjonline.com baidu.com        # build cc_queue.parquet from cached rows
    python -m r2ai.zh_v2.cc_queue scan --hosts 120ask.com --tag 120ask --only-missing   # own ledger/budget, cc_queue_120ask.parquet
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, assert_writable
from r2ai.cc_coverage import choose as CH
from r2ai.cc_coverage import crawls as CR
from r2ai.cc_coverage import index_scan as IX
from r2ai.cc_coverage.net import Blocked, BudgetExceeded, Client, InvalidResponse, Ledger, Unavailable
from r2ai.cc_coverage.surt import surt_key

import argparse
import json
import os
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = RUNS_DIR / 'zh-v2' / 'cc'
COLLINFO = 'https://index.commoncrawl.org/collinfo.json'
UA = 'R2AI-Stage3-CCIndexCoverage/1.0 (research; index metadata only; <=1 req/s)'    # the descriptive UA used for the longchau scan
MAX_RG_BYTES = 250_000_000


def guard(path) -> Path:
    p = assert_writable(path)
    if not p.is_relative_to(assert_writable(OUT)):
        raise ValueError(f'Not a zh-v2 cc output: {p}')
    return p


def common_prefix(keys: list[str]) -> str:
    return os.path.commonprefix(keys)


def host_urls(hosts, ids_path=None) -> pl.DataFrame:
    from r2ai.paths import WORK_DATA_DIR
    df = pl.read_parquet(ids_path or WORK_DATA_DIR / 'corpus_v2_ids.parquet', columns=['id', 'url', 'lang', 'host'])
    df = df.filter((pl.col('lang') == 'zh') & pl.col('host').is_in(list(hosts)))
    return df.with_columns(pl.Series('surt', [surt_key(u) for u in df['url']]))


def prefixes_of(df: pl.DataFrame) -> dict:
    """host -> longest common SURT prefix of its v2 URLs (at least 'tld,domain)/')."""
    out = {}
    for host, sub in df.group_by('host'):
        host = host[0] if isinstance(host, tuple) else host
        out[host] = common_prefix(sub['surt'].to_list())
    return out


def footer_summary(client: Client, crawl: str, paths: list[str], cache: Path, workers: int = 1) -> list[dict]:
    """Per file: size and per row group (min, max, rows, cost of the 7 columns). One 64 KiB request per file, cached."""
    if cache.exists():
        return json.loads(cache.read_text('utf-8'))
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        pf, rf = IX.open_parquet(client, paths[i])
        ranges = IX.key_ranges(pf)
        return {'i': i, 'file_bytes': rf.size, 'rows': pf.metadata.num_rows,
                'groups': [{'g': g, 'range': r, 'rows': pf.metadata.row_group(g).num_rows, 'cost': IX.row_group_cost(pf, g)}
                           for g, r in enumerate(ranges)]}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        files = list(ex.map(one, range(len(paths))))
    tmp = guard(cache.with_name(cache.name + '.tmp'))
    tmp.write_text(json.dumps(files), encoding='utf-8')
    os.replace(tmp, guard(cache))
    return files


def hits(files: list[dict], prefixes: list[str]) -> list[dict]:
    out = []
    for f in files:
        for g in f['groups']:
            r = g['range']
            if r is None or any(r[0] < IX.upper_bound(p) and r[1] >= p for p in prefixes):
                out.append({'file': f['i'], **g})
    return out


def scan_crawl(client, crawl, prefixes: dict, out: Path, workers=1) -> dict:
    t0, b0, r0 = time.time(), client.ledger.bytes, client.ledger.requests
    paths = IX.list_parquet_paths(client, crawl)
    files = footer_summary(client, crawl, paths, out / f'footers_{crawl}.json', workers)
    if len(files) != len(paths):
        return {'crawl': crawl, 'status': 'footer_cache_mismatch'}
    res = {'crawl': crawl, 'files': len(paths), 'hosts': {}}
    pending = {h: p for h, p in prefixes.items() if not (out / f'rows_{crawl}_{h}.parquet').exists()}
    if pending:
        todo = hits(files, list(pending.values()))
        cost = sum(g['cost'] for g in todo)
        if cost > MAX_RG_BYTES:
            return {'crawl': crawl, 'status': 'row_groups_too_big', 'cost_bytes': cost}
        tables = {h: [] for h in pending}
        for fi in sorted({g['file'] for g in todo}):
            pf, _ = IX.open_parquet(client, paths[fi])
            for g in (g for g in todo if g['file'] == fi):
                t = pf.read_row_group(g['g'], columns=list(IX.COLUMNS), use_threads=False)
                for h, p in pending.items():
                    tables[h].append(t.filter(pc.starts_with(t['url_surtkey'], p)))
        for h in pending:
            table = pa.concat_tables(tables[h]) if tables[h] else pa.table({c: [] for c in IX.COLUMNS})
            pq.write_table(table, guard(out / f'rows_{crawl}_{h}.parquet'))
    for h in prefixes:
        t = pq.read_table(out / f'rows_{crawl}_{h}.parquet')
        res['hosts'][h] = t.num_rows
    return {**res, 'status': 'ok', 'bytes': client.ledger.bytes - b0, 'requests': client.ledger.requests - r0,
            'seconds': round(time.time() - t0, 1)}


def indexed_v2_ids() -> set:
    """v2 zh ids already covered by a doc in the zh v2 index (their pages were crawled, no CC fetch needed)."""
    import json
    from . import index as zindex
    manifest = json.loads(zindex.MANIFEST.read_text('utf-8'))
    out = set()
    for r in manifest['segments']:
        t = zindex.segment_docs(manifest, r['name'])
        out.update(i for g in t['doc_ids_v2' if 'doc_ids_v2' in t.columns else 'doc_ids_group'].to_list() for i in g)
    return out


def build_queue(hosts, out: Path = OUT, ids_path=None, tag: str = '', only_missing: bool = False) -> dict:
    """cc_queue.parquet: for every v2 URL of `hosts` the chosen snapshot (newest usable <= corpus date, else oldest later)."""
    df = host_urls(hosts, ids_path)
    if only_missing:
        have = indexed_v2_ids()
        df = df.filter(~pl.col('id').is_in(sorted(have)))
    rows = []
    by_key = {}
    crawls = sorted((f.name[5:].rsplit('_', 1)[0] for f in out.glob('rows_*_*.parquet')), key=lambda c: CR.crawl_key(c) or (0, 0))
    for crawl in dict.fromkeys(crawls):
        for h in hosts:
            f = out / f'rows_{crawl}_{h}.parquet'
            if not f.exists():
                continue
            cols = pq.read_table(f).to_pydict()
            for i in range(len(cols['url_surtkey'])):
                by_key.setdefault(cols['url_surtkey'][i], []).append({'crawl': crawl, **{c: cols[c][i] for c in cols}})
    chosen = {}
    for id_, url, host, key in zip(df['id'], df['url'], df['host'], df['surt']):
        recs = by_key.get(key, [])
        pick = CH.choose(recs, CR.CORPUS_DATE)
        chosen[id_] = pick
        rows.append({'id': id_, 'url': url, 'host': host, 'surt': key, 'n_snapshots': sum(1 for r in recs if CH.usable(r)),
                     **{c: ((str(pick[c]) if c == 'fetch_time' and pick[c] is not None else pick[c]) if pick else None)
                        for c in ('crawl', 'fetch_time', 'warc_filename', 'warc_record_offset',
                                                                 'warc_record_length', 'after_corpus')}})
    q = pl.DataFrame(rows, schema={'id': pl.Int64, 'url': pl.String, 'host': pl.String, 'surt': pl.String, 'n_snapshots': pl.Int64,
                                   'crawl': pl.String, 'fetch_time': pl.String, 'warc_filename': pl.String,
                                   'warc_record_offset': pl.Int64, 'warc_record_length': pl.Int64, 'after_corpus': pl.Boolean},
                     strict=False) if rows else pl.DataFrame()
    q = q.with_columns(pl.col('fetch_time').cast(pl.String))
    found = q.filter(pl.col('crawl').is_not_null())
    found.write_parquet(guard(out / f'cc_queue{tag}.parquet'), compression='zstd')
    report = {'hosts': {h: {'v2_urls': int((q['host'] == h).sum()), 'with_200_html': int(((q['host'] == h) & q['crawl'].is_not_null()).sum()),
                            'warc_bytes': int(found.filter(pl.col('host') == h)['warc_record_length'].sum() or 0)} for h in hosts},
              'queue_rows': found.height, 'by_crawl': dict(found.group_by('crawl').len().sort('crawl').iter_rows()),
              'after_corpus': int(found['after_corpus'].sum() or 0), 'note': 'metadata only; no WARC downloaded by S11'}
    report['coverage'] = {h: (v['with_200_html'] / v['v2_urls'] if v['v2_urls'] else None) for h, v in report['hosts'].items()}
    report['only_missing'] = only_missing
    atomic_report(out / f'cc_queue_report{tag}.json', report)
    return report


def atomic_report(path, value):
    p = guard(path)
    tmp = guard(p.with_name(p.name + '.tmp'))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['scan', 'queue'])
    ap.add_argument('--hosts', nargs='+', required=True)
    ap.add_argument('--rps', type=float, default=2.0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--tag', default='', help='suffix for ledger/log/report/queue files (own budget per host group)')
    ap.add_argument('--only-missing', action='store_true', help='queue/coverage only over v2 ids not yet in the zh v2 index')
    a = ap.parse_args(argv)
    if a.rps > 2.0:
        raise SystemExit('rps above 2 is not allowed for this scan')
    out = guard(OUT)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f'_{a.tag}' if a.tag else ''
    if a.step == 'queue':
        print(json.dumps(build_queue(a.hosts, out, tag=suffix, only_missing=a.only_missing), ensure_ascii=False, indent=1))
        return 0
    prefixes = prefixes_of(host_urls(a.hosts))
    ledger = Ledger(out / f'ledger{suffix}.json')
    client = Client(ua=UA, ledger=ledger, log_path=out / f'requests{suffix}.jsonl', rps=a.rps)
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'ua': UA, 'rps': a.rps, 'prefixes': prefixes, 'crawls': {}}
    try:
        try:
            info = json.loads(client.get(COLLINFO).content)
            crawls = [c['id'] for c in sorted(CR.select_crawls(info), key=lambda c: CR.crawl_key(c['id']), reverse=True)]
        except Unavailable:    # index.commoncrawl.org down: reuse the window already scanned (footers cache names it)
            crawls = sorted((f.name[8:-5] for f in out.glob('footers_CC-MAIN-*.json')), key=lambda c: CR.crawl_key(c), reverse=True)
            if not crawls:
                raise
            report['collinfo'] = 'unavailable; crawl window taken from cached footers'
        for crawl in crawls:
            res = scan_crawl(client, crawl, prefixes, out, a.workers)
            report['crawls'][crawl] = res
            print(json.dumps(res, ensure_ascii=False), flush=True)
            if res['status'] != 'ok':
                report['status'] = 'stopped_anomaly'
                break
        else:
            report['status'] = 'ok'
        report['ledger'] = {'bytes': ledger.bytes, 'requests': ledger.requests}
    except (Blocked, BudgetExceeded, Unavailable, InvalidResponse, IX.TooBig, FileNotFoundError) as e:
        report['status'] = f'stopped: {type(e).__name__}: {e}'
    finally:
        client.close()
        atomic_report(out / f'scan_report{suffix}.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'crawls'}, ensure_ascii=False, default=str))
    if report.get('status') == 'ok':
        print(json.dumps(build_queue(a.hosts, out, tag=suffix, only_missing=a.only_missing), ensure_ascii=False))
    return 0 if report.get('status') == 'ok' else 2


if __name__ == '__main__':
    sys.exit(main())
