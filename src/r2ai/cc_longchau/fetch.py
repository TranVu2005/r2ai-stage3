"""Fetch the newest HTTP-200 Common Crawl capture of every corpus longchau URL by HTTP Range from data.commoncrawl.org.

  python -m r2ai.cc_longchau.fetch [--limit N] [--workers 4] [--rps 4] [--shard-size 500] [--dry-run]

Targets = cc-coverage/coverage.parquet (newest 200 HTML capture per corpus URL: crawl, WARC file, offset, length, time).
Only corpus ids are ever requested; each record is checked (WARC response, target URI == corpus URL after url_norm,
HTTP 200) before it enters a raw shard. Raw shards use the vicrawl layout (`<domain>/<name>.jsonl.zst`, one closed zstd
frame) plus a `cc` provenance dict per record. State: state/cc_longchau/fetch.db. Resumable: a rerun only asks for
URLs without a final row. 503/429/5xx/timeouts back off exponentially and slow every worker; 403 stops the run.
Budget guard: <= MAX_BYTES read and <= MAX_REQUESTS requests (ledger persisted across runs).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from r2ai.cc_coverage.net import Ledger, BudgetExceeded, Blocked
from vicrawl.fetch import decode_html
from vicrawl.shards import write_shard
from vicrawl.urlnorm import domain_of, normalize_url

from .common import CC_BASE, DOMAIN, UA, Roots, atomic_json, log_event, sha256_bytes
from .warc import WarcError, gunzip_member, parse_response

MAX_BYTES, MAX_REQUESTS = 6_000_000_000, 90_000
BACKOFF_S = (5, 15, 45, 120, 300)
MAX_GAP_S = 1.0
TIMEOUT = httpx.Timeout(connect=20.0, read=90.0, write=20.0, pool=60.0)
FINAL = ('ok', 'bad_http_status', 'url_mismatch', 'parse_error', 'range_mismatch', 'not_html')
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fetch(
  url_id INTEGER PRIMARY KEY, url TEXT, crawl TEXT, warc_filename TEXT, warc_offset INTEGER, warc_length INTEGER, fetch_time REAL,
  status TEXT, reason TEXT, attempts INTEGER DEFAULT 0, shard TEXT, http_status INTEGER, body_bytes INTEGER, record_sha256 TEXT,
  html_sha256 TEXT, warc_record_id TEXT, truncated TEXT, updated_at REAL);
CREATE INDEX IF NOT EXISTS fetch_status ON fetch(status);
"""


class Unavailable(Exception):
    pass


def load_targets(roots: Roots) -> pd.DataFrame:
    """One row per corpus longchau URL that has a selected capture: url_id, url, crawl, warc_filename, offset, length, time."""
    corpus = pd.read_parquet(roots.corpus)
    corpus = corpus[corpus.url.map(domain_of) == DOMAIN][['id', 'url']].rename(columns={'id': 'url_id'})
    cov = pd.read_parquet(roots.coverage)
    cov = cov[cov.crawl.notna()]
    t = cov.merge(corpus, on='url_id', how='inner', validate='one_to_one')
    if len(t) != len(cov):
        raise ValueError(f'{len(cov) - len(t)} coverage ids are not corpus {DOMAIN} URLs')
    t['warc_record_offset'] = t.warc_record_offset.astype('int64')
    t['warc_record_length'] = t.warc_record_length.astype('int64')
    t['fetch_epoch'] = t.fetch_time.map(lambda x: x.timestamp())
    return t.sort_values('url_id').reset_index(drop=True)


def connect(roots: Roots) -> sqlite3.Connection:
    roots.guard(roots.fetch_db).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(roots.fetch_db), timeout=60, isolation_level=None, check_same_thread=False)
    c.execute('PRAGMA journal_mode=WAL')
    c.executescript(SCHEMA_SQL)
    return c


class RangeClient:
    """Metered, paced Range GET. 403 -> Blocked (stop); 429/5xx/timeouts -> backoff, and every worker waits."""

    def __init__(self, *, ledger: Ledger, log_path, rps=4.0, transport=None, sleep=time.sleep, clock=time.monotonic, ua=UA):
        self.ledger, self.log_path, self.sleep, self.clock = ledger, Path(log_path), sleep, clock
        self.base_gap = self.gap = 1.0 / rps
        self.ok_streak = 0
        self.lock = threading.Lock()
        self.next_at = 0.0
        self.pause_until = 0.0
        self.http = httpx.Client(headers={'User-Agent': ua, 'Accept-Encoding': 'identity'}, timeout=TIMEOUT, transport=transport,
                                 follow_redirects=True)

    def _pace(self):
        with self.lock:
            now = self.clock()
            start = max(now, self.next_at, self.pause_until)
            self.next_at = start + self.gap
        if start > now:
            self.sleep(start - now)

    def _pause(self, seconds):
        with self.lock:
            self.pause_until = max(self.pause_until, self.clock() + seconds)
            self.gap = min(self.gap * 1.1, MAX_GAP_S)                 # throttled: ask for less, recover after a clean streak
            self.ok_streak = 0

    def _ok(self):
        with self.lock:
            self.ok_streak += 1
            if self.ok_streak >= 30 and self.gap > self.base_gap:
                self.gap, self.ok_streak = max(self.base_gap, self.gap * 0.92), 0

    def _log(self, **kw):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False) + '\n')

    def get_range(self, url: str, offset: int, length: int) -> bytes:
        rng = f'bytes={offset}-{offset + length - 1}'
        for attempt in range(len(BACKOFF_S) + 1):
            self.ledger.check(length)
            self._pace()
            t0 = self.clock()
            status, body, retry_after, err = None, b'', None, ''
            try:
                r = self.http.get(url, headers={'Range': rng})
                status, body, retry_after = r.status_code, r.content, r.headers.get('retry-after')
            except httpx.HTTPError as e:
                err = repr(e)[:160]
            self.ledger.add(len(body))
            self._log(url=url, range=rng, status=status, bytes=len(body), seconds=round(self.clock() - t0, 3), attempt=attempt, error=err)
            if status in (200, 206):
                self._ok()
                return body
            if status == 403:
                raise Blocked(f'HTTP 403 from {url}')
            if status is not None and status < 500 and status != 429:
                raise Unavailable(f'HTTP {status} {url}')         # 404/416 etc.: not retryable, recorded as a failure
            if attempt == len(BACKOFF_S):
                raise Unavailable(f'{url}: {status or err} after {attempt + 1} attempts')
            delay = BACKOFF_S[attempt]
            if retry_after and retry_after.strip().isdigit():
                delay = max(delay, int(retry_after))
            self._log(event='backoff', status=status, delay_s=delay)
            self._pause(delay)

    def close(self):
        self.http.close()


def fetch_one(client: RangeClient, row) -> dict:
    """-> {'row': db fields, 'record': raw shard record or None}."""
    base = {'url_id': int(row.url_id), 'url': row.url, 'crawl': row.crawl, 'warc_filename': row.warc_filename,
            'warc_offset': int(row.warc_record_offset), 'warc_length': int(row.warc_record_length), 'fetch_time': float(row.fetch_epoch)}
    try:
        data = client.get_range(CC_BASE + row.warc_filename, base['warc_offset'], base['warc_length'])
    except Unavailable as e:
        return {'row': {**base, 'status': 'unavailable', 'reason': str(e)[:200]}, 'record': None}
    if len(data) != base['warc_length']:
        return {'row': {**base, 'status': 'range_mismatch', 'reason': f'{len(data)} bytes != {base["warc_length"]}'}, 'record': None}
    base['record_sha256'] = sha256_bytes(data)
    try:
        resp = parse_response(gunzip_member(data))
    except WarcError as e:
        return {'row': {**base, 'status': 'parse_error', 'reason': str(e)[:200]}, 'record': None}
    base.update(http_status=resp.status, warc_record_id=resp.warc.get('warc-record-id', ''), truncated=resp.truncated)
    if normalize_url(resp.target_uri) != normalize_url(row.url):
        return {'row': {**base, 'status': 'url_mismatch', 'reason': f'WARC target {resp.target_uri[:150]}'}, 'record': None}
    if resp.status != 200:
        return {'row': {**base, 'status': 'bad_http_status', 'reason': f'HTTP {resp.status}'}, 'record': None}
    ctype = resp.http.get('content-type', '')
    if ctype and 'html' not in ctype.lower():
        return {'row': {**base, 'status': 'not_html', 'reason': ctype[:80]}, 'record': None}
    html, enc = decode_html(resp.body, ctype)
    base.update(body_bytes=len(resp.body), html_sha256=sha256_bytes(resp.body))
    record = {'url_norm': normalize_url(row.url), 'url': row.url, 'doc_ids': [int(row.url_id)], 'domain': DOMAIN, 'attempts': 1,
              'final_url': row.url, 'status': 'ok', 'reason': '', 'http_status': 200, 'fetched_at': base['fetch_time'], 'html': html,
              'source': 'common_crawl',
              'cc': {'crawl': row.crawl, 'warc_filename': row.warc_filename, 'warc_offset': base['warc_offset'],
                     'warc_length': base['warc_length'], 'warc_date': resp.warc.get('warc-date', ''),
                     'warc_record_id': base['warc_record_id'], 'warc_payload_digest': resp.warc.get('warc-payload-digest', ''),
                     'record_sha256': base['record_sha256'], 'body_sha256': base['html_sha256'], 'content_type': ctype,
                     'encoding': enc, 'truncated': resp.truncated}}
    return {'row': {**base, 'status': 'ok', 'reason': ''}, 'record': record}


def pending_targets(roots: Roots, db, targets: pd.DataFrame) -> pd.DataFrame:
    done = {r[0] for r in db.execute(f"SELECT url_id FROM fetch WHERE status IN ({','.join('?' * len(FINAL))})", FINAL)}
    return targets[~targets.url_id.isin(done)]


def _commit(roots: Roots, db, results: list[dict], records: list[dict]):
    shard = ''
    if records:
        shard = write_shard(roots.guard(roots.raw), DOMAIN, records).name
    now = time.time()
    db.execute('BEGIN IMMEDIATE')
    try:
        for r in results:
            d = r['row']
            old = db.execute('SELECT attempts FROM fetch WHERE url_id=?', (d['url_id'],)).fetchone()
            db.execute('INSERT OR REPLACE INTO fetch(url_id,url,crawl,warc_filename,warc_offset,warc_length,fetch_time,status,reason,attempts,'
                       'shard,http_status,body_bytes,record_sha256,html_sha256,warc_record_id,truncated,updated_at) '
                       'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (d['url_id'], d['url'], d['crawl'], d['warc_filename'], d['warc_offset'], d['warc_length'], d['fetch_time'],
                        d['status'], d.get('reason', ''), (old[0] if old else 0) + 1, shard if d['status'] == 'ok' else None,
                        d.get('http_status'), d.get('body_bytes'), d.get('record_sha256'), d.get('html_sha256'),
                        d.get('warc_record_id'), d.get('truncated'), now))
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise


def slug_priority_ids(path) -> set:
    """Doc ids that VS-append appended through slug_append (union over the 1,200 queries)."""
    return {int(d) for v in json.loads(Path(path).read_text('utf-8')).values() for d in v['appended']}


def run(roots: Roots, *, limit=None, workers=4, rps=4.0, shard_size=500, client=None, targets=None, priority_ids=None) -> dict:
    roots.preflight()
    targets = load_targets(roots) if targets is None else targets
    db = connect(roots)
    todo = pending_targets(roots, db, targets)
    if priority_ids:                                         # stable: priority URLs first, each group in url_id order
        todo = todo.assign(_p=~todo.url_id.isin(priority_ids)).sort_values(['_p', 'url_id'], kind='stable').drop(columns='_p')
    if limit is not None:
        todo = todo.head(limit)
    ledger = Ledger(roots.guard(roots.run / 'fetch_ledger.json'), max_bytes=MAX_BYTES, max_requests=MAX_REQUESTS)
    client = client or RangeClient(ledger=ledger, log_path=roots.guard(roots.run / 'fetch_requests.jsonl'), rps=rps)
    t0, done, stop = time.time(), 0, None
    rows = list(todo.itertuples(index=False))
    try:
        with ThreadPoolExecutor(workers) as ex:
            for a in range(0, len(rows), shard_size):
                results = list(ex.map(lambda r: fetch_one(client, r), rows[a:a + shard_size]))
                records = [r['record'] for r in results if r['record'] is not None]
                _commit(roots, db, results, records)
                done += len(results)
                log_event(roots, 'fetch', {'done': done, 'of': len(rows), 'ok': len(records), 'seconds': round(time.time() - t0, 1),
                                           'ledger_bytes': ledger.bytes, 'ledger_requests': ledger.requests})
                print(f'{done}/{len(rows)} ok+{len(records)} {time.time() - t0:.0f}s {ledger.bytes / 1e6:.0f} MB', flush=True)
    except (Blocked, BudgetExceeded) as e:
        stop = f'{type(e).__name__}: {e}'
    finally:
        client.close()
    summary = status_summary(roots, db, targets)
    summary.update(run_seconds=round(time.time() - t0, 1), run_urls=done, stopped=stop, ledger={'bytes': ledger.bytes, 'requests': ledger.requests})
    atomic_json(roots, roots.run / 'fetch_summary.json', summary)
    return summary


def status_summary(roots: Roots, db, targets: pd.DataFrame) -> dict:
    counts = dict(db.execute('SELECT status, COUNT(*) FROM fetch GROUP BY status').fetchall())
    return {'targets': int(len(targets)), 'by_status': counts, 'not_attempted': int(len(targets) - sum(counts.values())),
            'ok_bytes': int(db.execute("SELECT COALESCE(SUM(body_bytes),0) FROM fetch WHERE status='ok'").fetchone()[0]),
            'truncated_records': int(db.execute("SELECT COUNT(*) FROM fetch WHERE status='ok' AND truncated<>''").fetchone()[0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rps', type=float, default=4.0)
    ap.add_argument('--shard-size', type=int, default=500)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--priority-slug', help='query_append.json of VS-append: fetch its ids first')
    a = ap.parse_args(argv)
    roots = Roots()
    if a.dry_run:
        t = load_targets(roots)
        print(json.dumps({'targets': len(t), 'warc_bytes': int(t.warc_record_length.sum()), 'by_crawl': t.crawl.value_counts().to_dict()}, indent=1))
        return 0
    print(json.dumps(run(roots, limit=a.limit, workers=a.workers, rps=a.rps, shard_size=a.shard_size,
                         priority_ids=slug_priority_ids(a.priority_slug) if a.priority_slug else None), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
