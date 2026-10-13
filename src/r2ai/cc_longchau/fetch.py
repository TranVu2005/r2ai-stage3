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
import email.utils
import json
import random
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

MAX_BYTES, MAX_REQUESTS = 6_000_000_000, 400_000      # budget ceilings; the request cap counts retried 503s too
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


class Stopped(Exception):
    """Stop requested (deadline, STOP file, signal): the URL in flight stays pending."""


def retry_after_seconds(value, now=None):
    """Retry-After header -> seconds (delta-seconds or HTTP-date); None when absent or unparsable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - (now if now is not None else time.time()))
    except (TypeError, ValueError):
        return None


def load_targets(roots: Roots, member_ids=None) -> pd.DataFrame:
    """One row per corpus longchau URL that has a selected capture: url_id, url, crawl, warc_filename, offset, length, time."""
    corpus = pd.read_parquet(roots.corpus)
    corpus = corpus[corpus.url.map(domain_of) == DOMAIN][['id', 'url']].rename(columns={'id': 'url_id'})
    cov = pd.read_parquet(roots.coverage)
    cov = cov[cov.crawl.notna()]
    t = cov.merge(corpus, on='url_id', how='inner', validate='one_to_one')
    if len(t) != len(cov):
        raise ValueError(f'{len(cov) - len(t)} coverage ids are not corpus {DOMAIN} URLs')
    if member_ids is not None:                              # corpus v2 membership: a URL outside v2 can never be a positive
        keep = set(pd.read_parquet(member_ids, columns=['id']).id)
        t = t[t.url_id.isin(keep)]
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

    def __init__(self, *, ledger: Ledger, log_path, rps=4.0, transport=None, sleep=time.sleep, clock=time.monotonic, ua=UA,
                 rate=None, jitter=False, rng=None, stop: threading.Event | None = None):
        """rate: r2ai.cc_longchau.throttle.AutoRate (None = the older gap adaptation); jitter: backoff x U(0.8, 1.2)."""
        self.ledger, self.log_path, self.sleep, self.clock = ledger, Path(log_path), sleep, clock
        self.rate, self.jitter, self.rng, self.stop = rate, jitter, rng or random.Random(), stop
        self.requests = self.throttled = 0
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
            gap = 1.0 / self.rate.rps if self.rate else self.gap
            start = max(now, self.next_at, self.pause_until)
            self.next_at = start + gap
        if start > now:
            self.sleep(start - now)

    def _pause(self, seconds):
        with self.lock:
            self.pause_until = max(self.pause_until, self.clock() + seconds)
            if not self.rate:
                self.gap = min(self.gap * 1.1, MAX_GAP_S)             # throttled: ask for less, recover after a clean streak
            self.ok_streak = 0

    def _ok(self):
        with self.lock:
            self.ok_streak += 1
            if not self.rate and self.ok_streak >= 30 and self.gap > self.base_gap:
                self.gap, self.ok_streak = max(self.base_gap, self.gap * 0.92), 0

    def _log(self, **kw):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False) + '\n')

    def get_range(self, url: str, offset: int, length: int) -> bytes:
        rng = f'bytes={offset}-{offset + length - 1}'
        for attempt in range(len(BACKOFF_S) + 1):
            if self.stop is not None and self.stop.is_set():
                raise Stopped()
            self.ledger.check(length)
            self._pace()
            if self.stop is not None and self.stop.is_set():
                raise Stopped()
            t0 = self.clock()
            status, body, retry_after, err = None, b'', None, ''
            try:
                r = self.http.get(url, headers={'Range': rng})
                status, body, retry_after = r.status_code, r.content, r.headers.get('retry-after')
            except httpx.HTTPError as e:
                err = repr(e)[:160]
            self.ledger.add(len(body))
            self.requests += 1
            self.throttled += status in (429, 503)
            if self.rate:
                self.rate.record(status)
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
            delay = BACKOFF_S[attempt] * (self.rng.uniform(0.8, 1.2) if self.jitter else 1.0)
            ra = retry_after_seconds(retry_after)
            if ra is not None:
                delay = max(delay, ra)                                 # Retry-After is a floor
            delay = round(delay, 2)
            self._log(event='backoff', status=status, delay_s=delay, retry_after=retry_after)
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


def run(roots: Roots, *, limit=None, workers=4, rps=4.0, shard_size=500, client=None, targets=None, priority_ids=None, member_ids=None,
        order=None, stop: threading.Event | None = None, progress: dict | None = None, flush_s=300.0, deadline=None) -> dict:
    """Fetch the pending targets. order = priority.load_order() frame (url_id, group, best_rank): fetch group by group.

    Commits a raw shard every `shard_size` results or `flush_s` seconds. `stop` (Event) ends the run after the request in
    flight; the unfinished URL stays pending. `progress` is a dict the heartbeat reads."""
    roots.preflight()
    targets = load_targets(roots, member_ids) if targets is None else targets
    db = connect(roots)
    todo = pending_targets(roots, db, targets)
    group = {}
    if order is not None:                                    # group, then best slug rank, then url_id
        o = order.set_index('url_id')
        todo = todo.assign(_g=todo.url_id.map(o.group).fillna(len(set(o.group))).astype(int),
                           _r=todo.url_id.map(o.best_rank).astype('float64').fillna(1e9))
        todo = todo.sort_values(['_g', '_r', 'url_id'], kind='stable')
        group = dict(zip(todo.url_id, todo._g))
        todo = todo.drop(columns=['_g', '_r'])
    elif priority_ids:                                       # stable: priority URLs first, each group in url_id order
        todo = todo.assign(_p=~todo.url_id.isin(priority_ids)).sort_values(['_p', 'url_id'], kind='stable').drop(columns='_p')
    names = ('top20', 'top40', 'top100', 'rest')
    pending_by_group = pd.Series(list(group.get(u, 0) for u in todo.url_id), dtype=int).value_counts().to_dict() if group else {}
    if limit is not None:
        todo = todo.head(limit)
    stop = stop or threading.Event()
    progress = progress if progress is not None else {}
    if client is None:
        ledger = Ledger(roots.guard(roots.run / 'fetch_ledger.json'), max_bytes=MAX_BYTES, max_requests=MAX_REQUESTS)
        client = RangeClient(ledger=ledger, log_path=roots.guard(roots.run / 'fetch_requests.jsonl'), rps=rps, stop=stop)
    ledger = client.ledger
    base_ok = db.execute("SELECT COUNT(*) FROM fetch WHERE status='ok'").fetchone()[0]
    rows = list(todo.itertuples(index=False))
    left: dict[int, int] = {}
    for r in rows:
        left[group.get(r.url_id, 0)] = left.get(group.get(r.url_id, 0), 0) + 1
    names = ('top20', 'top40', 'top100', 'rest')
    finished_groups: dict = {}
    for g in range(len(names)):
        if order is not None and pending_by_group.get(g, 0) == 0:
            finished_groups[names[g]] = 'complete before ' + time.strftime('%Y-%m-%d %H:%M:%S')
    progress.update(ok_base=base_ok, ok_run=0, urls_run=0, todo_left=len(rows), current_group=None, last_ok_at=None, queue=len(rows))
    t0, done, stopped = time.time(), 0, None
    buf_res: list = []
    buf_rec: list = []
    last_flush = time.time()

    def flush():
        nonlocal buf_res, buf_rec, last_flush
        if buf_res:
            _commit(roots, db, buf_res, buf_rec)
            for r in buf_res:
                g = group.get(r['row']['url_id'], 0)
                if r['row']['status'] in FINAL:
                    left[g] -= 1
                    if left[g] == 0 and order is not None:
                        finished_groups[names[g]] = time.strftime('%Y-%m-%d %H:%M:%S')
            progress['ok_run'] += len(buf_rec)
            progress['todo_left'] = sum(left.values())
            if buf_rec:
                progress['last_ok_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            log_event(roots, 'fetch', {'done': done, 'of': len(rows), 'ok': len(buf_rec), 'seconds': round(time.time() - t0, 1),
                                       'ledger_bytes': ledger.bytes, 'ledger_requests': ledger.requests})
            print(f'{done}/{len(rows)} ok+{len(buf_rec)} {time.time() - t0:.0f}s {ledger.bytes / 1e6:.0f} MB', flush=True)
            buf_res, buf_rec = [], []
        last_flush = time.time()
        mark_groups()

    def mark_groups():
        f = roots.state / 'groups_done.json'
        prev = json.loads(f.read_text('utf-8')) if f.exists() else {}
        if any(k not in prev for k in finished_groups):
            atomic_json(roots, f, {**finished_groups, **prev})

    mark_groups()
    ex = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        i = 0
        while i < len(rows):
            if deadline is not None and time.time() >= deadline:
                stop.set()
                stopped = stopped or 'deadline'
            if stop.is_set():
                break
            n = 1 if ex is None else max(1, shard_size - len(buf_res))
            batch = rows[i:i + n]
            progress['current_group'] = names[group.get(batch[0].url_id, 0)] if order is not None else None
            try:
                results = [fetch_one(client, batch[0])] if ex is None else list(ex.map(lambda r: fetch_one(client, r), batch))
            except Stopped:
                break
            i += len(batch)
            done += len(batch)
            progress['urls_run'] = done
            buf_res += results
            buf_rec += [r['record'] for r in results if r['record'] is not None]
            if len(buf_res) >= shard_size or time.time() - last_flush >= flush_s:
                flush()
        flush()
    except (Blocked, BudgetExceeded) as e:
        stopped = f'{type(e).__name__}: {e}'
        flush()
    finally:
        if ex is not None:
            ex.shutdown()
        client.close()
    summary = status_summary(roots, db, targets)
    summary.update(run_seconds=round(time.time() - t0, 1), run_urls=done, stopped=stopped or ('stop requested' if stop.is_set() else None),
                   ledger={'bytes': ledger.bytes, 'requests': ledger.requests})
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
    ap.add_argument('--rps', type=float, default=4.0, help='start (and ceiling) of the request rate; with --priority-order it is auto-adjusted')
    ap.add_argument('--shard-size', type=int, default=500)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--member-ids', help='corpus_v2_ids.parquet: only URLs whose id is a member are requested')
    ap.add_argument('--priority-slug', help='query_append.json of VS-append: fetch its ids first')
    ap.add_argument('--priority-order', action='store_true',
                    help='production mode: priority groups of state/cc_longchau/priority.parquet, auto rate, heartbeat, exit record, STOP file')
    ap.add_argument('--deadline', help='local time YYYY-MM-DDTHH:MM: stop fetching then (production mode)')
    a = ap.parse_args(argv)
    roots = Roots()
    if a.dry_run:
        t = load_targets(roots, a.member_ids)
        print(json.dumps({'targets': len(t), 'warc_bytes': int(t.warc_record_length.sum()), 'by_crawl': t.crawl.value_counts().to_dict()}, indent=1))
        return 0
    if not a.priority_order:
        print(json.dumps(run(roots, limit=a.limit, workers=a.workers, rps=a.rps, shard_size=a.shard_size,
                             priority_ids=slug_priority_ids(a.priority_slug) if a.priority_slug else None, member_ids=a.member_ids), indent=1))
        return 0
    from . import fetchctl, priority
    from .throttle import AutoRate
    if a.workers != 1:
        raise SystemExit('--priority-order runs one worker')
    roots.preflight()
    p = fetchctl.paths(roots)
    stop, progress = threading.Event(), {}
    rate = AutoRate(a.rps, path=roots.guard(p['throttle']))
    ledger = Ledger(roots.guard(roots.run / 'fetch_ledger.json'), max_bytes=MAX_BYTES, max_requests=MAX_REQUESTS)
    client = RangeClient(ledger=ledger, log_path=roots.guard(roots.run / 'fetch_requests.jsonl'), rps=a.rps, rate=rate, jitter=True,
                         stop=stop, sleep=lambda s: stop.wait(s))
    deadline = fetchctl.parse_deadline(a.deadline) if a.deadline else None

    beat = fetchctl.Beat(roots, lambda: snapshot(), stop, deadline=deadline)

    def snapshot():
        n10, r10 = rate.recent(600)
        return {'ok_run': progress.get('ok_run', 0), 'ok_total': progress.get('ok_base', 0) + progress.get('ok_run', 0),
                'urls_run': progress.get('urls_run', 0), 'todo_left': progress.get('todo_left'), 'current_group': progress.get('current_group'),
                'last_ok_at': progress.get('last_ok_at'), 'requests_run': client.requests, 'throttled_run': client.throttled,
                'rate503_run': round(client.throttled / client.requests, 3) if client.requests else None,
                'rate503_10m': {'attempts': n10, 'share': round(r10, 3)}, 'rps': rate.rps,
                'rps_changes': [(time.strftime('%H:%M:%S', time.localtime(t)), o, n, w) for t, o, n, w in rate.changes[-5:]],
                'stop_reason': beat.reason}

    with beat:
        beat.install_signals()
        s = run(roots, limit=a.limit, workers=1, shard_size=a.shard_size, client=client, member_ids=a.member_ids, stop=stop, progress=progress,
                order=priority.load_order(roots), deadline=deadline)
        if beat.reason is None and s.get('stopped'):
            beat.reason = s['stopped']
    print(json.dumps(s, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
