"""Polite, resumable direct fetch for the v2 120ask/iiyi allowlist only.

No robots fetch, proxy, redirect following, browser emulation, or challenge solving is used.
The single request loop is intentionally global (at most 1 request/s total), which is
stricter than the approved per-host ceiling and keeps one stable egress path.
"""
from __future__ import annotations

# Load configured roots before any later HF/ML imports.
from r2ai.paths import ROOT, RUNS_DIR, STATE_DIR, WORK_DATA_DIR, OLD_ROOT, assert_writable, resolve_path

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

from vicrawl.urlnorm import group_urls, normalize_url


HOSTS = ('120ask.com', 'article.iiyi.com', 'bingli.iiyi.com')
USER_AGENT = 'r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)'
REQUEST_GAP_S = 1.0
JITTER_S = 0.1
BACKOFF_CAP_S = 300.0
ERROR_WINDOW = 500
ERROR_LIMIT = 0.20
RAW_BATCH_SIZE = 128
INGEST_BATCH_DOCS = 10_000
RUN = RUNS_DIR / 'zh-v2-direct'
RAW = WORK_DATA_DIR / 'raw_zh_v2'
STATE = STATE_DIR / 'zh_v2_direct'
DB = STATE / 'crawl.db'
IDS_PATH = WORK_DATA_DIR / 'corpus_v2_ids.parquet'
INDEX_MANIFEST = WORK_DATA_DIR / 'index' / 'zh_v2_segments' / 'manifest.json'
CHUNKS = WORK_DATA_DIR / 'chunks_zh_v2'
STOP_FILE = RUN / 'stop_reason.json'
TARGETS_FILE = RUN / 'targets.jsonl'
INVENTORY_FILE = RUN / 'inventory.json'
EVENTS_FILE = RUN / 'events.jsonl'
HEARTBEAT_FILE = RUN / 'heartbeat.json'
LOCK_FILE = RUN / 'direct.lock'

_BOT_PATTERNS = re.compile(
    r'just a moment|attention required|access denied|verify you are human|are you a robot|'
    r'captcha|验证码|滑块验证|安全验证|人机验证|请完成验证|访问频繁|访问过于频繁|'
    r'challenge-platform|cf-chl|cf_chl|geetest', re.I)


def guard(path) -> Path:
    p = resolve_path(path)
    if OLD_ROOT is not None and p.is_relative_to(resolve_path(OLD_ROOT)):
        raise ValueError(f'Write to OLD is forbidden: {p}')
    roots = (RUN, RAW, STATE)
    if not any(p == resolve_path(r) or p.is_relative_to(resolve_path(r)) for r in roots):
        raise ValueError(f'Not a zh-v2 direct output: {p}')
    # Existing directory trees are scanned only by preflight, not per record/shard.
    return p if p.is_dir() else assert_writable(p)


def preflight() -> None:
    for path in (RUN, RAW, STATE):
        p = resolve_path(path)
        if OLD_ROOT is not None and p.is_relative_to(resolve_path(OLD_ROOT)):
            raise ValueError(f'Output root resolves into OLD: {p}')
        if p.exists():
            for child in p.rglob('*'):
                q = resolve_path(child)
                if OLD_ROOT is not None and q.is_relative_to(resolve_path(OLD_ROOT)):
                    raise ValueError(f'Output child resolves into OLD: {q}')
                if not any(q == resolve_path(r) or q.is_relative_to(resolve_path(r)) for r in (RUN, RAW, STATE)):
                    raise ValueError(f'Output child escapes zh-v2 direct roots: {q}')
        else:
            assert_writable(p)
    for path in (DB, DB.with_name(DB.name + '-wal'), DB.with_name(DB.name + '-shm'), STOP_FILE):
        guard(path)


def _manifest_live_ids(manifest_path: Path, chunks_root: Path) -> set[int]:
    manifest = json.loads(Path(manifest_path).read_text('utf-8'))
    live: set[int] = set()
    for segment in manifest.get('segments', []):
        if segment.get('status') in ('retired', 'inactive'):
            continue
        if int(segment.get('docs_v2', 1)) == 0 or int(segment.get('rows_v2', 1)) == 0:
            continue
        docs_path = Path(chunks_root) / segment['name'] / 'docs.parquet'
        if not docs_path.is_file():
            raise ValueError(f"active v2 segment has no docs parquet: {docs_path}")
        names = pl.read_parquet_schema(docs_path).names()
        column = 'doc_ids_v2' if 'doc_ids_v2' in names else 'doc_ids_group'
        if column not in names:
            raise ValueError(f"active v2 segment docs missing doc_ids_v2/doc_ids_group: {docs_path}")
        for group in pl.read_parquet(docs_path, columns=[column])[column].to_list():
            live.update(int(value) for value in (group or []))
    return live


def build_targets(corpus_path=IDS_PATH, manifest_path=INDEX_MANIFEST, chunks_root=CHUNKS) -> list[dict]:
    """Return one representative v2 URL per normalized missing-live URL group."""
    corpus = pl.read_parquet(corpus_path, columns=['id', 'url', 'lang', 'host'])
    scoped = corpus.filter((pl.col('lang') == 'zh') & pl.col('host').is_in(HOSTS))
    if scoped.height == 0:
        raise ValueError('v2 corpus contains no 120ask/iiyi targets')
    live = _manifest_live_ids(Path(manifest_path), Path(chunks_root))
    missing = scoped.filter(~pl.col('id').is_in(live))
    rows = []
    for row in missing.select('id', 'url', 'host').to_dicts():
        url = str(row['url'])
        parsed = urlsplit(url)
        domain = str(row['host'])
        actual = parsed.netloc.rsplit('@', 1)[-1].lower()
        actual = actual[4:] if actual.startswith('www.') else actual
        if parsed.scheme not in ('http', 'https') or actual not in HOSTS or domain != actual:
            raise ValueError(f'v2 target URL host/scheme mismatch: id={row["id"]} url={url!r} host={domain!r}')
        rows.append((int(row['id']), url))
    groups = group_urls(rows)
    by_domain: dict[str, list[dict]] = {d: [] for d in HOSTS}
    missing_ids = {doc_id for doc_id, _ in rows}
    for group in groups:
        domain = group['domain']
        if domain not in HOSTS or not set(group['doc_ids']) <= missing_ids:
            raise ValueError(f'group escaped the missing v2 allowlist: {group}')
        if normalize_url(group['url']) != group['url_norm']:
            raise ValueError(f'URL normalization mismatch: {group}')
        by_domain[domain].append(group)
    out = []
    for domain in HOSTS:
        for rank, group in enumerate(sorted(by_domain[domain], key=lambda x: x['url_norm'])):
            out.append({**group, 'rank': rank})
    if {int(i) for target in out for i in target['doc_ids']} != missing_ids:
        raise ValueError('target URL groups do not cover every missing v2 id exactly')
    if len({i for target in out for i in target['doc_ids']}) != len(missing_ids):
        raise ValueError('a missing v2 id appears in more than one target group')
    return out


def client_options() -> dict:
    import httpx
    return {
        'headers': {'User-Agent': USER_AGENT, 'Accept': 'text/html,application/xhtml+xml'},
        'trust_env': False,
        'follow_redirects': False,
        'timeout': httpx.Timeout(30.0, connect=15.0),
        'limits': httpx.Limits(max_connections=1, max_keepalive_connections=1),
    }


def interval_seconds(sample: float, *, jitter_s: float = JITTER_S) -> float:
    if not 0 <= sample <= 1 or jitter_s < 0:
        raise ValueError('sample must be in [0,1] and jitter_s nonnegative')
    return REQUEST_GAP_S + sample * jitter_s


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (ValueError, AttributeError):
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, dt.timestamp() - (time.time() if now is None else now))
    except (TypeError, ValueError, OverflowError):
        return None


def backoff_seconds(streak: int, *, retry_after_s: float | None = None, cap_s: float = BACKOFF_CAP_S) -> float:
    if streak < 1 or cap_s < 0:
        raise ValueError('streak must be >=1 and cap_s nonnegative')
    exponential = min(2.0 ** (streak - 1), cap_s)
    return max(exponential, max(0.0, retry_after_s or 0.0))


class RollingErrors:
    def __init__(self, *, window: int = ERROR_WINDOW, threshold: float = ERROR_LIMIT, recent=()):
        self.window, self.threshold = window, threshold
        self.recent = deque((bool(x) for x in recent), maxlen=window)
        self.total = len(self.recent)

    @property
    def errors(self) -> int:
        return sum(self.recent)

    @property
    def rate(self) -> float | None:
        return self.errors / len(self.recent) if self.recent else None

    def add(self, is_error: bool) -> bool:
        self.recent.append(bool(is_error))
        self.total += 1
        return len(self.recent) == self.window and self.errors / self.window > self.threshold


def is_bot_challenge(html: str) -> bool:
    if not html:
        return False
    from vicrawl.classify import analyze_html
    stats = analyze_html(html)
    short = stats.text_len < 2000 and stats.visible_len < 4000
    title = stats.title or ''
    if re.search(r'^(?:just a moment|attention required|access denied|安全验证|人机验证|滑动验证|请完成验证)$', title, re.I):
        return True
    return short and bool(_BOT_PATTERNS.search(title + '\n' + stats.visible_head))


def classify_response(http_status: int, headers: dict, html: str, *, url: str = 'https://v2-allowlist.invalid/page') -> tuple[str, str]:
    code = int(http_status)
    content_type = str(headers.get('content-type', '')).lower()
    if 300 <= code < 400:
        return 'redirect_disallowed', f'http_{code}_not_followed'
    if code == 429 or code == 403:
        return 'blocked_4xx', f'http_{code}'
    if code >= 500:
        return 'dead_origin', f'http_{code}'
    if 400 <= code < 500 and code not in (404, 410):
        return 'blocked_4xx', f'http_{code}'
    is_html = any(x in content_type for x in ('text/html', 'application/xhtml+xml', 'application/xml'))
    if not content_type and html.lstrip().lower().startswith(('<!doctype html', '<html')):
        is_html = True
    if is_html and is_bot_challenge(html):
        return 'bot_challenge', 'challenge_signature'
    if not is_html:
        return 'non_html', f'content_type={content_type[:100]}'
    from vicrawl.classify import analyze_html, classify_response as classify
    state, reason = classify(stats=analyze_html(html), raw=html, url=url,
                             final_url=url, http_status=code)
    return state, reason


def fetch_once(client, target: dict, *, clock=time.time) -> dict:
    """Fetch exactly one allowlisted URL, never follow redirects, and return a raw-record candidate."""
    url, domain = str(target['url']), str(target['domain'])
    parsed = urlsplit(url)
    actual = parsed.netloc.rsplit('@', 1)[-1].lower()
    actual = actual[4:] if actual.startswith('www.') else actual
    if domain not in HOSTS or actual != domain or normalize_url(url) != target['url_norm']:
        raise ValueError(f'not an exact v2 direct target: {domain} {url}')
    at = clock()
    latency = None
    http_status = None
    headers = {}
    html = None
    body_bytes = 0
    try:
        started = time.perf_counter()
        response = client.get(url, follow_redirects=False)
        latency = round(time.perf_counter() - started, 3)
        http_status = response.status_code
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        body_bytes = len(response.content or b'')
        content_type = headers.get('content-type', '')
        raw, encoding = '', ''
        if response.content and any(x in content_type.lower() for x in ('text/html', 'application/xhtml+xml', 'application/xml')):
            from vicrawl.fetch import decode_html
            raw, encoding = decode_html(response.content, content_type)
        elif response.content and not content_type:
            candidate, encoding = response.content.decode('utf-8', errors='replace'), 'utf-8'
            if candidate.lstrip().lower().startswith(('<!doctype html', '<html')):
                raw = candidate
        state, reason = classify_response(http_status, headers, raw, url=url)
        if state in ('ok', 'thin'):
            html = raw
        retry_after = parse_retry_after(headers.get('retry-after'), now=at)
        err = None
    except Exception as exc:  # network/TLS/protocol error; never switch client or proxy
        state, reason, retry_after = 'network_error', f'{type(exc).__name__}: {str(exc)[:250]}', None
        encoding, content_type = '', ''
        err = reason
    record = {
        'url_norm': target['url_norm'], 'url': url, 'final_url': url, 'domain': domain,
        'doc_ids': [int(x) for x in target['doc_ids']], 'status': state, 'reason': reason,
        'http_status': http_status, 'fetched_at': at, 'content_type': headers.get('content-type', ''),
        'encoding': encoding, 'latency': latency, 'html': html,
        'source': 'direct', 'provenance_timestamp': at,
    }
    return {'record': record, 'is_error': state not in ('ok', 'thin') or not html,
            'retry_after_s': retry_after, 'error': err, 'body_bytes': body_bytes}


def _target_digest(targets: list[dict]) -> str:
    payload = [{k: t[k] for k in ('url_norm', 'url', 'domain', 'doc_ids')} for t in targets]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls(
  url_norm TEXT PRIMARY KEY, url TEXT NOT NULL, doc_ids TEXT NOT NULL, domain TEXT NOT NULL,
  rank INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', http_status INTEGER,
  reason TEXT, attempts INTEGER NOT NULL DEFAULT 0, fetched_at REAL, next_try_at REAL,
  shard TEXT, final_url TEXT, content_type TEXT, source TEXT, provenance_timestamp REAL, staged_record BLOB
);
CREATE INDEX IF NOT EXISTS urls_due ON urls(status, next_try_at, domain, rank);
CREATE TABLE IF NOT EXISTS hosts(
  domain TEXT PRIMARY KEY, block_streak INTEGER NOT NULL DEFAULT 0, next_request_at REAL,
  last_block_at REAL, last_status TEXT, updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests(
  request_id INTEGER PRIMARY KEY AUTOINCREMENT, url_norm TEXT NOT NULL, domain TEXT NOT NULL,
  requested_at REAL NOT NULL, status TEXT NOT NULL, http_status INTEGER, is_error INTEGER NOT NULL,
  latency REAL, body_bytes INTEGER NOT NULL DEFAULT 0, retry_after_s REAL, retry_delay_s REAL, reason TEXT
);
CREATE INDEX IF NOT EXISTS requests_recent ON requests(request_id DESC);
CREATE TABLE IF NOT EXISTS ingest_batches(
  batch TEXT PRIMARY KEY, n_urls INTEGER NOT NULL, n_v2_ids INTEGER NOT NULL DEFAULT 0,
  crawl_db TEXT NOT NULL, state TEXT NOT NULL,
  created_at REAL NOT NULL, launched_at REAL, pid INTEGER, finished_at REAL, exit_code INTEGER, reason TEXT,
  launch_attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ingest_urls(
  url_norm TEXT PRIMARY KEY, batch TEXT NOT NULL REFERENCES ingest_batches(batch)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _connect(db_path=DB) -> sqlite3.Connection:
    path = guard(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=60)
    c.execute('PRAGMA busy_timeout=60000')
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.executescript(DB_SCHEMA)
    columns = {r[1] for r in c.execute('PRAGMA table_info(urls)')}
    if 'staged_record' not in columns:
        c.execute('ALTER TABLE urls ADD COLUMN staged_record BLOB')
    batch_columns = {r[1] for r in c.execute('PRAGMA table_info(ingest_batches)')}
    if 'n_v2_ids' not in batch_columns:
        c.execute('ALTER TABLE ingest_batches ADD COLUMN n_v2_ids INTEGER NOT NULL DEFAULT 0')
    if 'launch_attempts' not in batch_columns:
        c.execute('ALTER TABLE ingest_batches ADD COLUMN launch_attempts INTEGER NOT NULL DEFAULT 0')
    c.commit()
    return c


def _atomic_json(path: Path, value) -> None:
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _log(event: str, **fields) -> None:
    path = guard(EVENTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'at': time.time(), 'event': event, **fields}, ensure_ascii=False, default=str) + '\n')


def prepare(*, corpus_path=IDS_PATH, manifest_path=INDEX_MANIFEST, chunks_root=CHUNKS,
            db_path=DB, run_root=RUN, scan_outputs=True) -> dict:
    """Create or verify the durable direct queue from exactly the missing-live v2 IDs."""
    if scan_outputs and Path(run_root) == RUN and Path(db_path) == DB:
        preflight()
    target_file = Path(run_root) / 'targets.jsonl'
    inventory_file = Path(run_root) / 'inventory.json'
    for p in (target_file, inventory_file, db_path):
        guard(p)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    c = _connect(db_path)
    try:
        old = c.execute("SELECT value FROM meta WHERE key='target_digest'").fetchone()
        if old:
            rows = c.execute('SELECT url_norm,url,doc_ids,domain,rank FROM urls ORDER BY domain,rank').fetchall()
            targets = [{'url_norm': r[0], 'url': r[1], 'doc_ids': json.loads(r[2]), 'domain': r[3], 'rank': r[4]} for r in rows]
            digest = _target_digest(targets)
            if old[0] != digest:
                raise ValueError('durable queue contents differ from its frozen target digest')
            _validate_frozen_targets(targets, corpus_path)
        else:
            targets = build_targets(corpus_path, manifest_path, chunks_root)
            digest = _target_digest(targets)
            with c:
                c.executemany('INSERT INTO urls(url_norm,url,doc_ids,domain,rank) VALUES (?,?,?,?,?)',
                              [(t['url_norm'], t['url'], json.dumps(t['doc_ids']), t['domain'], int(t['rank'])) for t in targets])
                c.executemany('INSERT INTO hosts(domain) VALUES (?)', [(h,) for h in HOSTS])
                c.executemany('INSERT INTO meta(key,value) VALUES (?,?)', [
                    ('target_digest', digest), ('target_count', str(len(targets))), ('prepared_at', str(time.time())),
                    ('corpus_sha256', _sha256(Path(corpus_path))),
                ])
        count = c.execute('SELECT count(*) FROM urls').fetchone()[0]
        if count != len(targets):
            raise ValueError(f'durable queue count {count} != v2 target count {len(targets)}')
    finally:
        c.close()
    counts = {h: sum(1 for t in targets if t['domain'] == h) for h in HOSTS}
    summary = {'target_count': len(targets), 'v2_doc_ids': len({i for t in targets for i in t['doc_ids']}),
               'url_groups': len(targets), 'domains': counts, 'target_sha256': digest,
               'corpus_sha256': _sha256(Path(corpus_path)), 'prepared_at': time.time()}
    with open(target_file.with_name(target_file.name + '.partial'), 'w', encoding='utf-8') as f:
        for t in targets:
            f.write(json.dumps(t, ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(target_file.with_name(target_file.name + '.partial'), target_file)
    _atomic_json(inventory_file, summary)
    return summary


def _validate_frozen_targets(targets: list[dict], corpus_path: Path) -> None:
    corpus = pl.read_parquet(corpus_path, columns=['id', 'url', 'lang', 'host'])
    rows = {int(r['id']): r for r in corpus.filter((pl.col('lang') == 'zh') & pl.col('host').is_in(HOSTS)).to_dicts()}
    seen = set()
    for target in targets:
        if target['domain'] not in HOSTS or target['url_norm'] != normalize_url(target['url']):
            raise ValueError(f'frozen queue has invalid URL/domain: {target}')
        group_ids = {int(i) for i in target['doc_ids']}
        if not group_ids or seen.intersection(group_ids):
            raise ValueError(f'frozen queue duplicates or empties ID group: {target["url_norm"]}')
        if not group_ids <= rows.keys():
            raise ValueError(f'frozen queue contains ID outside zh v2 host allowlist: {sorted(group_ids - rows.keys())[:5]}')
        if not any(rows[i]['url'] == target['url'] and rows[i]['host'] == target['domain'] for i in group_ids):
            raise ValueError(f'frozen representative URL is not attached to its v2 ID: {target["url"]}')
        seen.update(group_ids)


def _sha256(path: Path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _ensure_prepared(db_path=DB) -> tuple[sqlite3.Connection, str]:
    c = _connect(db_path)
    row = c.execute("SELECT value FROM meta WHERE key='target_digest'").fetchone()
    if not row:
        c.close()
        raise RuntimeError('direct queue is not prepared; run prepare first')
    if c.execute('SELECT count(*) FROM urls').fetchone()[0] != int(c.execute(
            "SELECT value FROM meta WHERE key='target_count'").fetchone()[0]):
        c.close()
        raise RuntimeError('direct queue count does not match prepared inventory')
    return c, row[0]


def _recent_errors(c: sqlite3.Connection) -> tuple[int, list[bool]]:
    total = c.execute('SELECT count(*) FROM requests').fetchone()[0]
    recent = [bool(r[0]) for r in c.execute('SELECT is_error FROM requests ORDER BY request_id DESC LIMIT ?', (ERROR_WINDOW,))]
    recent.reverse()
    return total, recent


def _write_raw_records(domain: str, records: list[dict]) -> Path:
    from vicrawl.shards import write_shard
    raw_dir = guard(RAW / domain)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = write_shard(RAW, domain, records)
    guarded = guard(path)
    if not guarded.is_file() or not guarded.name.endswith('.jsonl.zst'):
        raise ValueError(f'raw shard was not closed: {guarded}')
    return guarded


def _flush_buffers(c: sqlite3.Connection, buffers: dict[str, list[dict]]) -> int:
    flushed = 0
    for domain, records in list(buffers.items()):
        if not records:
            continue
        shard = _write_raw_records(domain, records)
        with c:
            for record in records:
                changed = c.execute("UPDATE urls SET status=?, shard=?, next_try_at=NULL, reason=?, http_status=?, fetched_at=?, "
                                    "final_url=?, content_type=?, source='direct', provenance_timestamp=?, staged_record=NULL "
                                    "WHERE url_norm=? AND status='staged'",
                                    (record['status'], str(shard), record['reason'], record['http_status'], record['fetched_at'],
                                     record['final_url'], record['content_type'], record['provenance_timestamp'], record['url_norm'])).rowcount
                if changed != 1:
                    raise RuntimeError(f'raw shard closed but URL checkpoint was not staged exactly once: {record["url_norm"]}')
        flushed += len(records)
        _log('raw_shard_closed', domain=domain, path=str(shard), records=len(records), sha256=_sha256(shard))
        records.clear()
    return flushed


def _stop_snapshot(c: sqlite3.Connection, *, reason: str) -> dict:
    total, recent = _recent_errors(c)
    errors = sum(recent)
    return {'reason': reason, 'at': time.time(), 'requests_total': total, 'window_size': len(recent),
            'window_errors': errors, 'window_error_rate': errors / len(recent) if recent else None,
            'threshold': ERROR_LIMIT, 'window': ERROR_WINDOW}


def _next_target(c: sqlite3.Connection, now: float, first_domain: str | None = None,
                 domains: tuple[str, ...] = HOSTS):
    marks = ','.join('?' for _ in domains)
    domain_order = 'CASE u.domain ' + ' '.join(f'WHEN ? THEN {i}' for i in range(len(domains))) + ' ELSE 99 END'
    return c.execute(
        "SELECT u.url_norm,u.url,u.doc_ids,u.domain,u.rank,u.attempts,h.block_streak,h.next_request_at,u.next_try_at "
        f"FROM urls u JOIN hosts h USING(domain) WHERE u.status='pending' AND u.domain IN ({marks}) "
        "AND (u.next_try_at IS NULL OR u.next_try_at<=?) AND (h.next_request_at IS NULL OR h.next_request_at<=?) "
        f"ORDER BY CASE WHEN u.domain=? THEN 0 ELSE 1 END,{domain_order},CASE WHEN u.attempts=0 THEN 0 ELSE 1 END,u.rank LIMIT 1",
        (*domains, now, now, first_domain or '', *domains)).fetchone()


def _next_due(c: sqlite3.Connection, now: float, domains: tuple[str, ...] = HOSTS) -> float | None:
    marks = ','.join('?' for _ in domains)
    rows = c.execute(f"SELECT u.next_try_at,h.next_request_at FROM urls u JOIN hosts h USING(domain) "
                     f"WHERE u.status='pending' AND u.domain IN ({marks})", domains).fetchall()
    if not rows:
        return None
    return min(max(now, r[0] or now, r[1] or now) for r in rows)


def _retryable_block(result: dict) -> bool:
    r = result['record']
    return (r['status'] in ('bot_challenge', 'cookie_challenge', 'network_error', 'dead_origin') or
            (r['status'] == 'blocked_4xx' and r['http_status'] in (403, 429)))


def _write_ingest_source(path: Path, rows: list[tuple]) -> None:
    """Atomically create the minimal read-only crawl DB S15 cut_batch consumes."""
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        guard(tmp).unlink()
    out = sqlite3.connect(tmp)
    try:
        out.execute('CREATE TABLE urls(url_norm TEXT PRIMARY KEY, domain TEXT NOT NULL, status TEXT NOT NULL, shard TEXT NOT NULL)')
        out.executemany('INSERT INTO urls VALUES (?,?,?,?)', rows)
        out.commit()
        if out.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError(f'ingest source DB failed integrity check: {tmp}')
    finally:
        out.close()
    if path.exists():
        guard(path).unlink()
    os.replace(tmp, path)


def _assign_ingest_batches(c: sqlite3.Connection, domains: tuple[str, ...], *, finalize: bool = False) -> list[dict]:
    """Freeze successful, unassigned URL groups into source DBs of ~10k v2 IDs."""
    if not domains:
        return []
    marks = ','.join('?' for _ in domains)
    rows = c.execute(
        "SELECT u.url_norm,u.doc_ids,u.domain,u.status,u.shard,u.rank FROM urls u "
        "LEFT JOIN ingest_urls i USING(url_norm) WHERE i.url_norm IS NULL "
        f"AND u.domain IN ({marks}) AND u.status IN ('ok','thin') AND u.shard IS NOT NULL",
        domains).fetchall()
    order = {domain: i for i, domain in enumerate(domains)}
    rows.sort(key=lambda r: (order[r[2]], r[5], r[0]))
    pending = rows
    assigned = []
    while pending:
        selected, v2_ids = [], 0
        for row in pending:
            selected.append(row)
            v2_ids += len(json.loads(row[1]))
            if v2_ids >= INGEST_BATCH_DOCS:
                break
        if v2_ids < INGEST_BATCH_DOCS and not finalize:
            break
        meta = c.execute("SELECT value FROM meta WHERE key='next_ingest_batch'").fetchone()
        number = int(meta[0]) if meta else 1
        batch = f'zhv2d-{number:05d}'
        source = guard(STATE / f'ingest-source-{batch}.db')
        source_rows = [(row[0], row[2], row[3], row[4]) for row in selected]
        if c.execute('SELECT 1 FROM ingest_batches WHERE batch=?', (batch,)).fetchone():
            raise RuntimeError(f'ingest batch sequence already exists: {batch}')
        _write_ingest_source(source, source_rows)
        with c:
            c.execute('INSERT INTO ingest_batches(batch,n_urls,n_v2_ids,crawl_db,state,created_at) VALUES (?,?,?,?,?,?)',
                      (batch, len(selected), v2_ids, str(source), 'queued', time.time()))
            c.executemany('INSERT INTO ingest_urls(url_norm,batch) VALUES (?,?)', [(row[0], batch) for row in selected])
            c.execute("INSERT INTO meta(key,value) VALUES ('next_ingest_batch',?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(number + 1),))
        assigned.append({'batch': batch, 'n_urls': len(selected), 'n_v2_ids': v2_ids,
                         'crawl_db': str(source), 'state': 'queued'})
        chosen = {row[0] for row in selected}
        pending = [row for row in pending if row[0] not in chosen]
        if not finalize and sum(len(json.loads(row[1])) for row in pending) < INGEST_BATCH_DOCS:
            break
    return assigned


def _advance_ingest(c: sqlite3.Connection) -> dict:
    """Poll S15's single ingest driver and start the next durable batch if available."""
    active = c.execute("SELECT batch,crawl_db,launch_attempts,launched_at FROM ingest_batches "
                        "WHERE state='running' ORDER BY created_at LIMIT 1").fetchone()
    queued = c.execute("SELECT batch,crawl_db,launch_attempts FROM ingest_batches WHERE state='queued' "
                       "ORDER BY created_at LIMIT 1").fetchone()
    if not active and not queued:
        return {'state': 'idle'}
    from r2ai.zh_v2 import ingest as zingest
    roots = zingest.V2Roots()
    if active:
        batch, source, attempts, launched_at = active
        driver = zingest.drv.driver_status(roots, batch)
        if driver['state'] == 'alive' or (driver['state'] == 'not_started' and time.time() - (launched_at or 0) < 120):
            return {'state': 'running', 'batch': batch, 'driver': driver}
        if driver['state'] == 'exited' and driver['code'] == 0:
            with c:
                c.execute("UPDATE ingest_batches SET state='done',finished_at=?,exit_code=0,reason=? WHERE batch=?",
                          (time.time(), driver['reason'], batch))
            _log('ingest_batch_done', batch=batch, driver=driver)
        else:
            reason = driver.get('reason', driver.get('state', 'unknown'))
            if driver.get('state') == 'exited':
                reason = f'{reason} (exit={driver.get("code")})'
            if int(attempts) >= 3:
                with c:
                    c.execute("UPDATE ingest_batches SET state='failed',finished_at=?,exit_code=?,reason=? WHERE batch=?",
                              (time.time(), driver.get('code'), reason, batch))
                _log('ingest_batch_failed', batch=batch, attempts=attempts, reason=reason)
                return {'state': 'failed', 'batch': batch, 'reason': reason, 'driver': driver}
            with c:
                c.execute("UPDATE ingest_batches SET state='queued',reason=?,pid=NULL WHERE batch=?", (reason, batch))
            _log('ingest_batch_retry', batch=batch, attempt=attempts + 1, reason=reason)

    queued = c.execute("SELECT batch,crawl_db,launch_attempts FROM ingest_batches WHERE state='queued' "
                       "ORDER BY created_at LIMIT 1").fetchone()
    if not queued:
        return {'state': 'idle'}
    batch, source, attempts = queued
    if int(attempts) >= 3:
        with c:
            c.execute("UPDATE ingest_batches SET state='failed',finished_at=?,reason=COALESCE(reason,'launch retries exhausted') WHERE batch=?",
                      (time.time(), batch))
        return {'state': 'failed', 'batch': batch, 'reason': 'launch retries exhausted'}
    try:
        result = zingest.launch(batch, crawl_db=source)
    except Exception as exc:
        reason = f'launch error: {type(exc).__name__}: {exc}'
        next_attempts = int(attempts) + 1
        state = 'failed' if next_attempts >= 3 else 'queued'
        with c:
            c.execute('UPDATE ingest_batches SET state=?,launch_attempts=?,finished_at=?,reason=? WHERE batch=?',
                      (state, next_attempts, time.time() if state == 'failed' else None, reason, batch))
        _log('ingest_batch_launch_error', batch=batch, attempt=next_attempts, state=state, reason=reason)
        return {'state': state, 'batch': batch, 'reason': reason}
    with c:
        c.execute("UPDATE ingest_batches SET state='running',launched_at=?,pid=?,launch_attempts=launch_attempts+1,reason=NULL WHERE batch=?",
                  (time.time(), result['pid'], batch))
    _log('ingest_batch_launch', batch=batch, pid=result['pid'], crawl_db=source, launch_attempt=attempts + 1)
    return {'state': 'running', 'batch': batch, 'pid': result['pid'], 'crawl_db': source}


def _manage_ingest(c: sqlite3.Connection, domains: tuple[str, ...], *, finalize: bool = False) -> dict:
    assigned = _assign_ingest_batches(c, domains, finalize=finalize)
    for batch in assigned:
        _log('ingest_batch_queued', **batch)
    advanced = _advance_ingest(c)
    return {'assigned': assigned, 'driver': advanced}


def _commit_outcome(c: sqlite3.Connection, target: dict, result: dict, *, when: float) -> float | None:
    record = result['record']
    domain, status = target['domain'], record['status']
    retry_delay = None
    with c:
        host = c.execute('SELECT block_streak FROM hosts WHERE domain=?', (domain,)).fetchone()
        if _retryable_block(result):
            streak = int(host[0]) + 1
            retry_delay = backoff_seconds(streak, retry_after_s=result['retry_after_s'])
            next_at = when + retry_delay
            c.execute('UPDATE hosts SET block_streak=?, next_request_at=?, last_block_at=?, last_status=?, updated_at=? WHERE domain=?',
                      (streak, next_at, status, when, when, domain))
            c.execute("UPDATE urls SET status='pending', http_status=?, reason=?, attempts=attempts+1, fetched_at=?, next_try_at=?, "
                      "final_url=?, content_type=?, source='direct', provenance_timestamp=?, shard=NULL WHERE url_norm=?",
                      (record['http_status'], record['reason'], when, next_at, record['final_url'], record['content_type'], when, target['url_norm']))
        elif status in ('ok', 'thin') and record.get('html'):
            c.execute("UPDATE hosts SET block_streak=0, next_request_at=NULL, last_status=?, updated_at=? WHERE domain=?",
                      (status, when, domain))
            c.execute("UPDATE urls SET status='staged', http_status=?, reason=?, attempts=attempts+1, fetched_at=?, next_try_at=NULL, "
                      "final_url=?, content_type=?, source='direct', provenance_timestamp=?, shard=NULL, staged_record=? WHERE url_norm=?",
                      (record['http_status'], record['reason'], when, record['final_url'], record['content_type'], when,
                       zlib.compress(json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), level=3),
                       target['url_norm']))
        else:
            if status in ('ok', 'thin'):
                status = 'empty_response'
                record['status'] = status
                record['reason'] = 'empty_response_body'
                result['is_error'] = True
            c.execute('UPDATE hosts SET block_streak=0, next_request_at=NULL, last_status=?, updated_at=? WHERE domain=?',
                      (status, when, domain))
            c.execute('UPDATE urls SET status=?, http_status=?, reason=?, attempts=attempts+1, fetched_at=?, next_try_at=NULL, '
                      "final_url=?, content_type=?, source='direct', provenance_timestamp=?, shard=NULL WHERE url_norm=?",
                      (status, record['http_status'], record['reason'], when, record['final_url'], record['content_type'], when, target['url_norm']))
        c.execute('INSERT INTO requests(url_norm,domain,requested_at,status,http_status,is_error,latency,body_bytes,retry_after_s,retry_delay_s,reason) '
                  'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                  (target['url_norm'], domain, when, record['status'], record['http_status'], int(result['is_error'] or status == 'empty_response'),
                   record['latency'], result.get('body_bytes', 0), result['retry_after_s'], retry_delay, record['reason']))
    return retry_delay


def _exclusive(path: Path):
    import msvcrt
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+b')
    f.seek(0, os.SEEK_END)
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as e:
        f.close()
        raise RuntimeError('direct crawler already running') from e
    return f


def _release_lock(f):
    import msvcrt
    try:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        f.close()


def run(*, db_path=DB, max_requests: int | None = None, first_domain: str | None = None,
        domains: tuple[str, ...] = HOSTS,
        sleep=time.sleep, clock=time.time, random_sample=random.random, client=None,
        heartbeat_every=30.0) -> dict:
    """Sequential one-IP fetcher; each closed shard is checkpointed before queue pointers commit."""
    if max_requests is not None and max_requests < 1:
        raise ValueError('max_requests must be >= 1')
    if first_domain is not None and first_domain not in HOSTS:
        raise ValueError(f'first_domain must be one of {HOSTS}')
    domains = tuple(dict.fromkeys(domains))
    if not domains or set(domains) - set(HOSTS):
        raise ValueError(f'domains must be a nonempty subset of {HOSTS}')
    if first_domain is not None and first_domain not in domains:
        raise ValueError('first_domain must be included in domains')
    if not Path(db_path).exists():
        raise RuntimeError('direct queue is not prepared; run prepare first')
    if STOP_FILE.exists():
        raise RuntimeError(f'direct crawl is stopped: {STOP_FILE.read_text("utf-8")}')
    lock = _exclusive(LOCK_FILE)
    own_client = client is None
    c = None
    try:
        if own_client:
            import httpx
            client = httpx.Client(**client_options())
        c, digest = _ensure_prepared(db_path)
    except BaseException:
        if own_client and client is not None:
            client.close()
        _release_lock(lock)
        raise
    buffers: dict[str, list[dict]] = {h: [] for h in HOSTS}
    started, request_start, last_heartbeat = clock(), None, 0.0
    request_count = 0
    pipeline_failure = None
    try:
        # Success HTML is compressed into SQLite per URL until its closed raw shard is committed.
        staged = c.execute("SELECT url_norm,domain,staged_record FROM urls WHERE status='staged' ORDER BY rank").fetchall()
        legacy_staged = []
        for url_norm, domain, blob in staged:
            if not blob:
                legacy_staged.append((url_norm,))
                continue
            buffers[domain].append(json.loads(zlib.decompress(blob).decode('utf-8')))
        if legacy_staged:
            with c:
                c.executemany("UPDATE urls SET status='pending',next_try_at=NULL WHERE url_norm=?", legacy_staged)
        with c:
            requeued = c.execute("UPDATE urls SET status='pending' WHERE status IN "
                                 "('network_error','dead_origin','bot_challenge','cookie_challenge') OR "
                                 "(status='blocked_4xx' AND http_status IN (403,429))").rowcount
        if requeued:
            _log('legacy_retryable_requeued', count=requeued)
        for domain in HOSTS:
            if len(buffers[domain]) >= RAW_BATCH_SIZE:
                _flush_buffers(c, {domain: buffers[domain]})
        pipeline = _manage_ingest(c, domains)
        if pipeline['driver']['state'] == 'failed':
            pipeline_failure = pipeline['driver']
        total, recent = _recent_errors(c)
        rolling = RollingErrors(recent=recent)
        if len(recent) == ERROR_WINDOW and rolling.rate is not None and rolling.rate > ERROR_LIMIT:
            snapshot = _stop_snapshot(c, reason='rolling error rate already above threshold at resume')
            _atomic_json(STOP_FILE, snapshot)
            raise RuntimeError('rolling error threshold exceeded before resume')
        while not pipeline_failure:
            now = clock()
            selected = _next_target(c, now, first_domain, domains)
            if selected is None:
                due = _next_due(c, now, domains)
                if due is None:
                    break
                wait_s = max(0.05, min(10.0, due - now))
                sleep(wait_s)
                _atomic_json(HEARTBEAT_FILE, {'pid': os.getpid(), 'at': clock(), 'requests_this_run': request_count,
                                              'requests_total': total + request_count, 'sleeping_until': due,
                                              'domains': domains, 'target_sha256': digest})
                continue
            target = {'url_norm': selected[0], 'url': selected[1], 'doc_ids': json.loads(selected[2]),
                      'domain': selected[3], 'rank': selected[4], 'attempts': selected[5]}
            if request_start is not None:
                wait_s = interval_seconds(random_sample()) - (now - request_start)
                if wait_s > 0:
                    sleep(wait_s)
            request_start = clock()
            result = fetch_once(client, target, clock=clock)
            when = result['record']['fetched_at']
            _commit_outcome(c, target, result, when=when)
            request_count += 1
            first_domain = None
            if result['record']['status'] in ('ok', 'thin') and result['record'].get('html'):
                buffers[target['domain']].append(result['record'])
                if len(buffers[target['domain']]) >= RAW_BATCH_SIZE:
                    _flush_buffers(c, {target['domain']: buffers[target['domain']]})
                    pipeline = _manage_ingest(c, domains)
                    if pipeline['driver']['state'] == 'failed':
                        pipeline_failure = pipeline['driver']
                        break
            if request_count % 30 == 0 or request_count == 1:
                _log('request_progress', requests_this_run=request_count, url_norm=target['url_norm'],
                     domain=target['domain'], status=result['record']['status'], error_rate_500=rolling.rate)
            # Add once after the durable request row has been committed.
            stopped = rolling.add(bool(result['is_error']))
            if stopped:
                _flush_buffers(c, buffers)
                pipeline = _manage_ingest(c, domains)
                if pipeline['driver']['state'] == 'failed':
                    pipeline_failure = pipeline['driver']
                snapshot = _stop_snapshot(c, reason='rolling error rate exceeded 20% after 500 requests')
                _atomic_json(STOP_FILE, snapshot)
                _log('stopped_error_rate', **snapshot)
                break
            if max_requests is not None and request_count >= max_requests:
                _flush_buffers(c, buffers)
                pipeline = _manage_ingest(c, domains)
                if pipeline['driver']['state'] == 'failed':
                    pipeline_failure = pipeline['driver']
                break
            if clock() - last_heartbeat >= heartbeat_every:
                counts = dict(c.execute('SELECT status,count(*) FROM urls GROUP BY status'))
                _atomic_json(HEARTBEAT_FILE, {'pid': os.getpid(), 'at': clock(), 'requests_this_run': request_count,
                                              'requests_total': total + request_count, 'statuses': counts,
                                              'error_rate_500': rolling.rate, 'target_sha256': digest})
                last_heartbeat = clock()
        _flush_buffers(c, buffers)
        counts = dict(c.execute('SELECT status,count(*) FROM urls GROUP BY status'))
        total_requests, recent_now = _recent_errors(c)
        pending_scope = c.execute("SELECT count(*) FROM urls WHERE domain IN (" + ','.join('?' for _ in domains) + ") "
                                  "AND status IN ('pending','staged')", domains).fetchone()[0]
        pipeline = _manage_ingest(c, domains, finalize=(pending_scope == 0 or STOP_FILE.exists()))
        if pipeline['driver']['state'] == 'failed':
            pipeline_failure = pipeline['driver']
        batch_states = dict(c.execute('SELECT state,count(*) FROM ingest_batches GROUP BY state'))
        summary = {'state': 'stopped_error_rate' if STOP_FILE.exists() else
                   ('ingest_failed' if pipeline_failure else
                    ('paused' if pending_scope and max_requests else ('running' if pending_scope else 'scope_complete'))),
                   'requests_this_run': request_count, 'requests_total': total_requests, 'statuses': counts,
                   'error_rate_500': sum(recent_now) / len(recent_now) if recent_now else None,
                   'target_sha256': digest, 'started_at': started, 'finished_at': clock(),
                   'elapsed_s': round(clock() - started, 1), 'domains': domains, 'pending_in_scope': pending_scope,
                   'ingest_batches': batch_states, 'ingest_driver': pipeline['driver'],
                   'ingest_failure': pipeline_failure}
        _atomic_json(HEARTBEAT_FILE, {'pid': os.getpid(), 'at': clock(), **summary})
        _log('run_exit', **summary)
        return summary
    finally:
        if own_client:
            client.close()
        if c is not None:
            c.close()
        _release_lock(lock)


def status(*, db_path=DB) -> dict:
    if not Path(db_path).exists():
        return {'state': 'not_prepared', 'database': str(db_path)}
    c = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        counts = {s: n for s, n in c.execute('SELECT status,count(*) FROM urls GROUP BY status')}
        domains = {d: {s: n for s, n in c.execute('SELECT status,count(*) FROM urls WHERE domain=? GROUP BY status', (d,))}
                   for d in HOSTS}
        total, recent = _recent_errors(c)
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'ingest_batches' in tables:
            columns = {r[1] for r in c.execute('PRAGMA table_info(ingest_batches)')}
            n_ids = 'n_v2_ids' if 'n_v2_ids' in columns else '0'
            attempts = 'launch_attempts' if 'launch_attempts' in columns else '0'
            ingest_batches = [dict(zip(('batch', 'n_urls', 'n_v2_ids', 'state', 'launch_attempts', 'pid', 'reason'), row))
                              for row in c.execute(f'SELECT batch,n_urls,{n_ids},state,{attempts},pid,reason '
                                                   'FROM ingest_batches ORDER BY created_at')]
        else:
            ingest_batches = []
        request_statuses = {s: n for s, n in c.execute('SELECT status,count(*) FROM requests GROUP BY status')}
        return {'state': 'stopped_error_rate' if STOP_FILE.exists() else
                (json.loads(HEARTBEAT_FILE.read_text('utf-8')).get('state', 'running_or_paused')
                 if HEARTBEAT_FILE.exists() else 'not_started'),
                'requests_total': total, 'error_rate_500': sum(recent) / len(recent) if recent else None,
                'window_n': len(recent), 'statuses': counts, 'domains': domains,
                'request_statuses': request_statuses, 'ingest_batches': ingest_batches,
                'hosts': [dict(zip(('domain','block_streak','next_request_at','last_block_at','last_status'), row))
                          for row in c.execute('SELECT domain,block_streak,next_request_at,last_block_at,last_status FROM hosts')],
                'heartbeat': json.loads(HEARTBEAT_FILE.read_text('utf-8')) if HEARTBEAT_FILE.exists() else None,
                'stop': json.loads(STOP_FILE.read_text('utf-8')) if STOP_FILE.exists() else None}
    finally:
        c.close()


def launch(*, hosts: tuple[str, ...] = ('120ask.com',), first_domain: str | None = None) -> dict:
    """Start the crawler as a hidden, detached local process with durable stdout/stderr logs."""
    hosts = tuple(dict.fromkeys(hosts))
    if not hosts or set(hosts) - set(HOSTS):
        raise ValueError(f'hosts must be a nonempty subset of {HOSTS}')
    if first_domain is not None and first_domain not in hosts:
        raise ValueError('first_domain must be included in hosts')
    if not DB.is_file():
        raise RuntimeError('direct queue is not prepared; run prepare first')
    if STOP_FILE.exists():
        raise RuntimeError(f'direct crawl is stopped: {STOP_FILE.read_text("utf-8")}')
    preflight()
    RUN.mkdir(parents=True, exist_ok=True)
    stdout, stderr = guard(RUN / 'crawler.stdout.log'), guard(RUN / 'crawler.stderr.log')
    args = ['-B', '-m', 'r2ai.zh_v2.direct', 'run', '--hosts', ','.join(hosts)]
    if first_domain:
        args.extend(['--first-domain', first_domain])
    ps_quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    ps_args = ','.join(ps_quote(x) for x in args)
    ps = (f"$env:PYTHONPATH={ps_quote(str(ROOT / 'src'))}; "
          f"$p = Start-Process -FilePath {ps_quote(sys.executable)} -ArgumentList @({ps_args}) "
          f"-WorkingDirectory {ps_quote(ROOT)} -RedirectStandardOutput {ps_quote(stdout)} "
          f"-RedirectStandardError {ps_quote(stderr)} -WindowStyle Hidden -PassThru; "
          "Write-Output $p.Id")
    result = subprocess.run(['powershell', '-NoProfile', '-Command', ps], cwd=ROOT, capture_output=True, text=True, check=True)
    pid = int(result.stdout.strip().splitlines()[-1])
    launch_info = {'pid': pid, 'at': time.time(), 'python': sys.executable, 'hosts': hosts, 'args': args,
                   'stdout': str(stdout), 'stderr': str(stderr), 'target_sha256':
                   json.loads(INVENTORY_FILE.read_text('utf-8'))['target_sha256']}
    _atomic_json(guard(RUN / 'launch.json'), launch_info)
    return launch_info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=('prepare', 'run', 'launch', 'status'))
    ap.add_argument('--max-requests', type=int)
    ap.add_argument('--first-domain', choices=HOSTS)
    ap.add_argument('--hosts', default=','.join(HOSTS))
    a = ap.parse_args(argv)
    if a.action == 'prepare':
        print(json.dumps(prepare(), ensure_ascii=False, indent=1))
    elif a.action == 'run':
        preflight()
        hosts = tuple(x for x in a.hosts.split(',') if x)
        print(json.dumps(run(max_requests=a.max_requests, first_domain=a.first_domain, domains=hosts), ensure_ascii=False, indent=1))
    elif a.action == 'launch':
        hosts = tuple(x for x in a.hosts.split(',') if x)
        print(json.dumps(launch(hosts=hosts, first_domain=a.first_domain), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
