"""Resumable Common Crawl recovery for the mapped S27 Pharmacity URLs."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from r2ai.paths import RUNS_DIR
from r2ai.cc_coverage import index_scan as IX
from r2ai.cc_coverage.net import Blocked, Client, Ledger
from r2ai.cc_coverage.surt import surt_key
from r2ai.cc_longchau.common import CC_BASE, UA as CC_UA
from r2ai.cc_longchau.fetch import RangeClient, Unavailable
from r2ai.cc_longchau.warc import WarcError, gunzip_member, parse_response
from vicrawl.fetch import decode_html
from vicrawl.urlnorm import normalize_url

from . import pipeline as P


CC_CRAWLS_NEWEST_FIRST = (
    'CC-MAIN-2026-39', 'CC-MAIN-2026-34', 'CC-MAIN-2026-30', 'CC-MAIN-2026-25', 'CC-MAIN-2026-21',
    'CC-MAIN-2026-17', 'CC-MAIN-2026-12', 'CC-MAIN-2026-08', 'CC-MAIN-2026-04', 'CC-MAIN-2025-51',
    'CC-MAIN-2025-47', 'CC-MAIN-2025-43', 'CC-MAIN-2025-38',
)
CC_HOST_PREFIX = 'vn,pharmacity'
INDEX_SCAN_VERSION = 3
INDEX_MAX_ROW_BYTES = 250_000_000
WARC_MAX_BYTES = 6_000_000_000
WARC_MAX_REQUESTS = 400_000
WARC_WORKERS = 5

INDEX_SCHEMA = pa.schema([
    ('doc_id', pa.int64()), ('crawl', pa.string()), ('url_surtkey', pa.string()), ('fetch_time', pa.string()),
    ('fetch_status', pa.int32()), ('content_mime_type', pa.string()), ('warc_filename', pa.string()),
    ('warc_record_offset', pa.int64()), ('warc_record_length', pa.int64()),
])

CC_DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS targets(
  doc_id INTEGER PRIMARY KEY, outcome TEXT NOT NULL, captures INTEGER NOT NULL,
  best_title TEXT, best_ratio REAL, best_crawl TEXT, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS capture_results(
  doc_id INTEGER NOT NULL, crawl TEXT NOT NULL, warc_offset INTEGER NOT NULL, warc_length INTEGER NOT NULL,
  index_status INTEGER, index_mime TEXT, status TEXT NOT NULL, reason TEXT NOT NULL,
  target_uri TEXT, match_type TEXT, http_status INTEGER, title TEXT, body_chars INTEGER,
  title_ratio REAL, title_match_count INTEGER, title_token_count INTEGER, fetched_at TEXT,
  warc_date TEXT, warc_record_id TEXT, warc_sha256 TEXT, raw_path TEXT, raw_sha256 TEXT,
  PRIMARY KEY(doc_id,crawl,warc_offset,warc_length));
'''


def pending_cc_ids(mapping_ids, page_ids) -> list[int]:
    """The CC scope is exactly mapped IDs absent from the live/checkpoint state."""
    return sorted({int(x) for x in mapping_ids} - {int(x) for x in page_ids})


def index_surt_keys(mapped_url: str) -> set[str]:
    """The CC index may retain or canonicalize the www label; look up both SURT forms."""
    base = surt_key(mapped_url)
    host, suffix = base.split(')', 1)
    return {base, host + ',www)' + suffix}


def index_lookup_key(url_surtkey: str) -> str:
    """Return the host/path SURT without a query so query variants can be classified as captures, not misses."""
    return url_surtkey.split('?', 1)[0]


def match_warc_target(mapped_url: str, target_uri: str) -> str:
    if target_uri == mapped_url:
        return 'exact'
    if normalize_url(target_uri) == normalize_url(mapped_url):
        return 'normalized'
    return 'mismatch'


def classify_capture_group(captures: list[dict], results: list[dict]) -> str:
    if not captures:
        return 'no_capture'
    if all(int(row.get('fetch_status') or 0) in (404, 410) for row in captures):
        return 'dead'
    if any(row.get('status') == 'ok' for row in results):
        return 'usable'
    return 'unusable'


def _stamp(value) -> str:
    if value is None:
        return ''
    if hasattr(value, 'isoformat'):
        out = value.isoformat()
    else:
        out = str(value)
    if out and not (out.endswith('Z') or '+' in out[-6:]):
        out += 'Z'
    return out


def _epoch(value) -> float:
    if value is None:
        return 0.0
    if hasattr(value, 'timestamp'):
        try:
            return float(value.replace(tzinfo=timezone.utc).timestamp() if value.tzinfo is None else value.timestamp())
        except (OverflowError, OSError, ValueError):
            return 0.0
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (OverflowError, OSError, TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def _connect_cc(roots: P.Roots) -> sqlite3.Connection:
    path = roots.guard(roots.root / 'cc-backfill' / 'state.sqlite')
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=60, isolation_level=None, check_same_thread=False)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.executescript(CC_DB_SCHEMA)
    return c


def _scope(roots: P.Roots) -> list[dict]:
    path = roots.guard(roots.root / 'cc-backfill' / 'scope.parquet')
    mapping = pq.read_table(roots.mapping).to_pylist()
    by_id = {int(row['doc_id']): row for row in mapping}
    if path.exists():
        rows = pq.read_table(path).to_pylist()
        if any(int(row['doc_id']) not in by_id or by_id[int(row['doc_id'])]['mapped_url'] != row['mapped_url'] for row in rows):
            raise ValueError('saved CC scope no longer matches the S27 mapping')
        return rows
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        page_ids = {int(row[0]) for row in c.execute('SELECT doc_id FROM pages')}
    finally:
        c.close()
    ids = [doc_id for doc_id in pending_cc_ids(by_id, page_ids) if doc_id != 317850]
    if len(ids) != 5182:
        raise ValueError(f'expected 5,182 unmapped-to-state CC targets, got {len(ids)}')
    rows = [by_id[doc_id] for doc_id in ids]
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=P.MAPPING_SCHEMA), tmp, compression='zstd')
    os.replace(tmp, path)
    return rows


def _mark_317850_unknown(roots: P.Roots) -> bool:
    c = sqlite3.connect(roots.state, timeout=60)
    try:
        cur = c.execute("UPDATE pages SET status='unknown',reason='unknown_404_challenge_never_retry',updated_at=? "
                        "WHERE doc_id=317850 AND status='halted_captcha'", (time.time(),))
        c.commit()
        if cur.rowcount:
            return True
        row = c.execute('SELECT status,reason FROM pages WHERE doc_id=317850').fetchone()
        return bool(row and row[0] == 'unknown' and row[1] == 'unknown_404_challenge_never_retry')
    finally:
        c.close()


def _write_json(roots: P.Roots, path: Path, value: dict) -> None:
    P.atomic_json(roots, path, value)


def _write_index_rows(roots: P.Roots, path: Path, rows: list[dict]) -> None:
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA), tmp, compression='zstd')
    os.replace(tmp, path)


def _scan_crawl(client: Client, roots: P.Roots, crawl: str, scope: list[dict], target_by_surt: dict[str, list[int]],
                footer_dir: Path) -> dict:
    out = roots.guard(roots.root / 'cc-backfill' / 'index' / f'rows_{crawl}.parquet')
    marker = roots.guard(roots.root / 'cc-backfill' / 'index' / f'meta_{crawl}.json')
    if out.is_file() and marker.is_file() and json.loads(marker.read_text(encoding='utf-8')).get('scan_version') == INDEX_SCAN_VERSION:
        rows = pq.read_table(out).to_pylist()
        scope_ids = {doc_id for ids in target_by_surt.values() for doc_id in ids}
        if any(row['crawl'] != crawl or int(row['doc_id']) not in scope_ids for row in rows):
            raise ValueError(f'cached CC index rows outside the target scope: {crawl}')
        return {'crawl': crawl, 'status': 'cached', 'matched_rows': len(rows),
                'matched_doc_ids': len({int(row['doc_id']) for row in rows})}

    paths = IX.list_parquet_paths(client, crawl)
    footer_path = roots.guard(roots.root / 'cc-backfill' / 'footers' / f'footer_map_{crawl}.json')
    if footer_path.is_file():
        fm = json.loads(footer_path.read_text(encoding='utf-8'))
        source = 's27-cache'
    else:
        cached = footer_dir / f'footer_map_{crawl}.json'
        if cached.is_file():
            fm = json.loads(cached.read_text(encoding='utf-8'))
            source = 'read-only-vi-longchau-cache'
        else:
            fm = IX.footer_map(client, paths, CC_HOST_PREFIX, workers=1)
            source = 'network'
        footer_path.parent.mkdir(parents=True, exist_ok=True)
        footer_path.write_text(json.dumps(fm, ensure_ascii=False), encoding='utf-8')
    if len(paths) != int(fm.get('files', -1)) or len(fm.get('file_list', [])) != len(paths):
        raise ValueError(f'{crawl}: Common Crawl file list differs from cached footer map')
    missing_stats = int(fm.get('file_ranges_missing_stats', 0))
    if missing_stats:
        raise ValueError(f'{crawl}: {missing_stats} files have no key range; refusing to skip possible target captures')

    upper = IX.upper_bound(CC_HOST_PREFIX)
    file_rows = sorted(fm['file_list'], key=lambda f: int(f['i']))
    candidates = [f for f in file_rows if f.get('range') and f['range'][0] < upper and f['range'][1] >= CC_HOST_PREFIX]
    wanted = set(target_by_surt)
    matched = []
    read_groups = 0
    for file_row in candidates:
        fi = int(file_row['i'])
        pf, ranged = IX.open_parquet(client, paths[fi])
        got_range = IX.file_range(IX.key_ranges(pf))
        if ranged.size != int(file_row['file_bytes']) or list(got_range or ()) != list(file_row['range']):
            raise ValueError(f'{crawl}: footer cache mismatch for parquet file {fi}')
        table, details = IX.read_prefix(pf, CC_HOST_PREFIX, columns=IX.COLUMNS, max_bytes=INDEX_MAX_ROW_BYTES)
        read_groups += len(details['row_groups_read'])
        path_keys = pc.list_element(pc.split_pattern(table['url_surtkey'], '?'), 0)
        table = table.filter(pc.is_in(path_keys, value_set=pa.array(sorted(wanted), type=pa.string())))
        for row in table.to_pylist():
            for doc_id in target_by_surt.get(row['url_surtkey'], []):
                matched.append({
                    'doc_id': doc_id, 'crawl': crawl, 'url_surtkey': row['url_surtkey'],
                    'fetch_time': _stamp(row.get('fetch_time')), 'fetch_status': int(row.get('fetch_status') or 0),
                    'content_mime_type': row.get('content_mime_type') or '', 'warc_filename': row['warc_filename'],
                    'warc_record_offset': int(row['warc_record_offset']), 'warc_record_length': int(row['warc_record_length']),
                })
    matched.sort(key=lambda r: (r['doc_id'], r['warc_record_offset'], r['warc_record_length']))
    _write_index_rows(roots, out, matched)
    _write_json(roots, marker, {'scan_version': INDEX_SCAN_VERSION, 'crawl': crawl,
                                'index_prefix': CC_HOST_PREFIX, 'scope_urls': len(scope), 'rows': len(matched)})
    return {'crawl': crawl, 'status': 'scanned', 'parquet_files': len(paths), 'candidate_files': len(candidates),
            'row_groups_read': read_groups, 'matched_rows': len(matched),
            'matched_doc_ids': len({r['doc_id'] for r in matched}), 'footer_map_source': source}


def _persist_capture(c: sqlite3.Connection, lock: threading.Lock, result: dict) -> None:
    columns = ('doc_id','crawl','warc_offset','warc_length','index_status','index_mime','status','reason','target_uri',
               'match_type','http_status','title','body_chars','title_ratio','title_match_count','title_token_count',
               'fetched_at','warc_date','warc_record_id','warc_sha256','raw_path','raw_sha256')
    with lock:
        c.execute('INSERT OR REPLACE INTO capture_results(' + ','.join(columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')',
                  tuple(result.get(k) for k in columns))


def _fetch_candidate(range_client: RangeClient, roots: P.Roots, target: dict, capture: dict) -> dict:
    base = {
        'doc_id': int(target['doc_id']), 'crawl': capture['crawl'],
        'warc_offset': int(capture['warc_record_offset']), 'warc_length': int(capture['warc_record_length']),
        'index_status': int(capture['fetch_status']), 'index_mime': capture['content_mime_type'],
        'fetched_at': capture['fetch_time'], 'status': '', 'reason': '',
    }
    if base['index_status'] != 200 or 'html' not in (base['index_mime'] or '').lower() or 'revisit' in (base['index_mime'] or '').lower():
        return base | {'status': 'skipped_index_capture', 'reason': 'not_http_200_html'}
    url = CC_BASE + capture['warc_filename']
    try:
        data = range_client.get_range(url, base['warc_offset'], base['warc_length'])
    except (Blocked, Unavailable):
        raise
    if len(data) != base['warc_length']:
        return base | {'status': 'range_mismatch', 'reason': f'{len(data)} bytes != {base["warc_length"]}'}
    base['warc_sha256'] = hashlib.sha256(data).hexdigest()
    try:
        response = parse_response(gunzip_member(data))
    except WarcError as exc:
        return base | {'status': 'parse_error', 'reason': str(exc)[:200]}
    target_uri = response.target_uri
    match_type = match_warc_target(target['mapped_url'], target_uri)
    base.update(target_uri=target_uri, match_type=match_type, http_status=response.status,
                warc_date=response.warc.get('warc-date', ''), warc_record_id=response.warc.get('warc-record-id', ''))
    body_bytes = response.body
    raw_dir = roots.guard(roots.root / 'cc-backfill' / 'raw' / str(target['doc_id']))
    raw_path = roots.guard(raw_dir / f'{capture["crawl"]}-{base["warc_offset"]}.html.gz')
    raw_dir.mkdir(parents=True, exist_ok=True)
    P.atomic_bytes(roots, raw_path, gzip.compress(body_bytes, compresslevel=6, mtime=0))
    base.update(raw_path=str(raw_path), raw_sha256=hashlib.sha256(body_bytes).hexdigest())
    if match_type == 'mismatch':
        return base | {'status': 'url_mismatch', 'reason': 'warc_target_does_not_match_mapped_url'}
    if response.status != 200:
        return base | {'status': 'bad_http_status', 'reason': f'HTTP {response.status}'}
    content_type = response.http.get('content-type', '')
    if content_type and 'html' not in content_type.lower():
        return base | {'status': 'not_html', 'reason': content_type[:120]}
    html, _encoding = decode_html(body_bytes, content_type)
    page = P.selector_document(html)
    title_score = P.title_match(target['slug'] or '', page['title'])
    base.update(title=page['title'], body_chars=len(page['body']), title_ratio=title_score['ratio'],
                title_match_count=title_score['matched'], title_token_count=title_score['terms'])
    if not page['selector_found']:
        return base | {'status': 'selector_missing', 'reason': 'pmc_content_blog_detail_missing'}
    if len(page['body']) < P.MIN_BODY_CP:
        return base | {'status': 'body_too_short', 'reason': 'body_below_200_codepoints'}
    from vicrawl.classify import analyze_html, classify_response
    kind, reason = classify_response(stats=analyze_html(html), raw=html, url=target['mapped_url'],
                                     final_url=target_uri, http_status=response.status)
    if kind in ('soft404_or_home', 'bot_challenge', 'cookie_challenge'):
        return base | {'status': 'nonarticle', 'reason': reason or kind}
    if not title_score['accepted']:
        return base | {'status': 'low_title_match', 'reason': 'title_match_below_50_percent'}
    return base | {'status': 'ok', 'reason': ''}


def _record_target_outcome(c: sqlite3.Connection, pages_db: sqlite3.Connection, lock: threading.Lock, target: dict,
                           captures: list[dict], results: list[dict], outcome: str) -> None:
    low = [r for r in results if r.get('title_ratio') is not None]
    best = max(low, key=lambda r: (float(r['title_ratio']), int(r.get('body_chars') or 0)), default={})
    with lock:
        c.execute('INSERT OR REPLACE INTO targets(doc_id,outcome,captures,best_title,best_ratio,best_crawl,updated_at) VALUES(?,?,?,?,?,?,?)',
                  (target['doc_id'], outcome, len(captures), best.get('title'), best.get('title_ratio'), best.get('crawl'), time.time()))
        if outcome == 'usable':
            ok = next(r for r in results if r.get('status') == 'ok')
            raw_path = Path(ok['raw_path'])
            fetched_at = ok.get('warc_date') or ok.get('fetched_at') or datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
            pages_db.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,http_status,final_url,title,title_match_ratio,
                         title_match_count,title_token_count,body_chars,fetched_at,raw_path,raw_sha256,attempts,reason,updated_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO NOTHING''',
                      (target['doc_id'], target['original_url'], target['mapped_url'], 'ok', 200, ok.get('target_uri'),
                       ok.get('title'), ok.get('title_ratio'), ok.get('title_match_count'), ok.get('title_token_count'),
                       ok.get('body_chars'), fetched_at, str(raw_path), ok.get('raw_sha256'), 0, 'common_crawl', time.time()))
        elif outcome == 'dead':
            latest = max(captures, key=lambda r: (_epoch(r.get('fetch_time')), r['crawl']))
            pages_db.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,http_status,final_url,fetched_at,attempts,reason,updated_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO NOTHING''',
                      (target['doc_id'], target['original_url'], target['mapped_url'], 'cc_dead', latest['fetch_status'],
                       target['mapped_url'], latest.get('fetch_time'), 0, 'all_cc_captures_404_or_410', time.time()))
        c.commit()
        pages_db.commit()


def _process_target(range_client: RangeClient, roots: P.Roots, c: sqlite3.Connection, pages_db: sqlite3.Connection, lock: threading.Lock,
                    target: dict, captures: list[dict]) -> dict:
    if not captures:
        _record_target_outcome(c, pages_db, lock, target, captures, [], 'no_capture')
        return {'doc_id': target['doc_id'], 'outcome': 'no_capture'}
    if all(int(row.get('fetch_status') or 0) in (404, 410) for row in captures):
        _record_target_outcome(c, pages_db, lock, target, captures, [], 'dead')
        return {'doc_id': target['doc_id'], 'outcome': 'dead'}
    with lock:
        done = {(r[0], int(r[1])) for r in c.execute(
            'SELECT crawl,warc_offset FROM capture_results WHERE doc_id=? AND status NOT IN (\'fetch_error\')',
            (target['doc_id'],)).fetchall()}
        prev = c.execute('SELECT outcome FROM targets WHERE doc_id=?', (target['doc_id'],)).fetchone()
        if prev and prev[0] == 'usable':
            return {'doc_id': target['doc_id'], 'outcome': 'usable', 'cached': True}
    results = []
    usable = False
    ordered = sorted(captures, key=lambda r: (_epoch(r.get('fetch_time')), r['crawl']), reverse=True)
    for capture in ordered:
        key = (capture['crawl'], int(capture['warc_record_offset']))
        if key in done:
            with lock:
                row = c.execute('''SELECT status,reason,target_uri,match_type,http_status,title,body_chars,title_ratio,
                                          title_match_count,title_token_count,fetched_at,raw_path,raw_sha256,warc_sha256,warc_date,
                                          warc_record_id,crawl,warc_offset,warc_length,index_status,index_mime
                                   FROM capture_results WHERE doc_id=? AND crawl=? AND warc_offset=?''',
                                (target['doc_id'], capture['crawl'], int(capture['warc_record_offset']))).fetchone()
            if row:
                cols = ('status','reason','target_uri','match_type','http_status','title','body_chars','title_ratio',
                        'title_match_count','title_token_count','fetched_at','raw_path','raw_sha256','warc_sha256','warc_date',
                        'warc_record_id','crawl','warc_offset','warc_length','index_status','index_mime')
                previous_result = dict(zip(cols, row))
                results.append(previous_result)
                if previous_result['status'] == 'ok':
                    usable = True
                    break
            continue
        if index_lookup_key(capture['url_surtkey']) != capture['url_surtkey']:
            mismatch = {'doc_id': target['doc_id'], 'crawl': capture['crawl'],
                        'warc_offset': int(capture['warc_record_offset']), 'warc_length': int(capture['warc_record_length']),
                        'index_status': int(capture.get('fetch_status') or 0), 'index_mime': capture.get('content_mime_type') or '',
                        'status': 'url_mismatch_index', 'reason': 'index_query_variant_does_not_normalize_to_mapped_url'}
            _persist_capture(c, lock, mismatch)
            results.append(mismatch)
            continue
        if int(capture.get('fetch_status') or 0) != 200 or 'html' not in (capture.get('content_mime_type') or '').lower():
            skipped = {'doc_id': target['doc_id'], 'crawl': capture['crawl'],
                       'warc_offset': int(capture['warc_record_offset']), 'warc_length': int(capture['warc_record_length']),
                       'index_status': int(capture.get('fetch_status') or 0), 'index_mime': capture.get('content_mime_type') or '',
                       'status': 'skipped_index_capture', 'reason': 'not_http_200_html'}
            _persist_capture(c, lock, skipped)
            results.append(skipped)
            continue
        try:
            result = _fetch_candidate(range_client, roots, target, capture)
        except Unavailable as exc:
            result = {'doc_id': target['doc_id'], 'crawl': capture['crawl'],
                      'warc_offset': int(capture['warc_record_offset']), 'warc_length': int(capture['warc_record_length']),
                      'index_status': int(capture.get('fetch_status') or 0), 'index_mime': capture.get('content_mime_type') or '',
                      'status': 'fetch_error', 'reason': str(exc)[:200]}
        _persist_capture(c, lock, result)
        results.append(result)
        if result['status'] == 'ok':
            usable = True
            break
    errored = any(r.get('status') == 'fetch_error' for r in results)
    if usable:
        outcome = 'usable'
    elif errored:
        outcome = 'pending'
    else:
        outcome = 'unusable'
    _record_target_outcome(c, pages_db, lock, target, captures, results, outcome)
    return {'doc_id': target['doc_id'], 'outcome': outcome, 'tried_captures': len(results)}


def _summary(roots: P.Roots, scope: list[dict], crawls: list[dict], c: sqlite3.Connection,
             started: str, unknown_marked: bool) -> dict:
    outcomes = dict(c.execute('SELECT outcome,COUNT(*) FROM targets GROUP BY outcome').fetchall())
    exact = c.execute("SELECT COUNT(*) FROM capture_results WHERE match_type='exact'").fetchone()[0]
    normalized = c.execute("SELECT COUNT(*) FROM capture_results WHERE match_type='normalized'").fetchone()[0]
    mismatch = c.execute("SELECT COUNT(*) FROM capture_results WHERE match_type='mismatch'").fetchone()[0]
    low = c.execute("SELECT COUNT(DISTINCT doc_id) FROM capture_results WHERE status='low_title_match'").fetchone()[0]
    low_examples = []
    # Read title examples from the scope map and result rows, keeping only the 20 lowest ratios.
    scope_by_id = {int(row['doc_id']): row for row in scope}
    for row in c.execute("SELECT doc_id,crawl,title,title_ratio,body_chars,target_uri,reason FROM capture_results WHERE status='low_title_match' ORDER BY title_ratio ASC,doc_id ASC LIMIT 20"):
        doc_id, crawl, title, ratio, chars, target_uri, reason = row
        low_examples.append({'doc_id': int(doc_id), 'mapped_url': scope_by_id[int(doc_id)]['mapped_url'], 'crawl': crawl,
                             'title': title, 'title_match_ratio': ratio, 'body_chars': chars,
                             'warc_target_uri': target_uri, 'reason': reason})
    total = len(scope)
    report = {
        'batch': P.BATCH, 'source': 'common_crawl', 'started_at_utc': started,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
        'scope_urls': total, 'crawl_ids_newest_first': list(CC_CRAWLS_NEWEST_FIRST), 'index_surt_prefix': CC_HOST_PREFIX,
        'crawls': crawls,
        'outcomes': {'usable_text': int(outcomes.get('usable',0)), 'a_no_capture': int(outcomes.get('no_capture',0)),
                     'b_captured_but_no_usable_text': int(outcomes.get('unusable',0)),
                     'dead_all_captures_404_or_410': int(outcomes.get('dead',0)),
                     'pending_fetch_error': int(outcomes.get('pending',0))},
        'warc_target_url_matches': {'exact': int(exact), 'normalized': int(normalized), 'mismatch': int(mismatch)},
        'low_title_match_unique_docs': int(low), 'low_title_match_pct_of_scope': round(100 * low / max(total,1), 4),
        'low_title_match_review_threshold_pct': 5.0, 'low_title_match_examples_lowest_20': low_examples,
        'doc_id_317850_marked_unknown_without_retry': unknown_marked,
        'shared_index_written': False, 'vectors_written': False,
    }
    out = roots.guard(roots.root / 'reports' / 'cc_backfill.json')
    _write_json(roots, out, report)
    return report


def run_cc_backfill(roots: P.Roots, *, workers: int = WARC_WORKERS) -> dict:
    """Scan all 13 CC indexes and resume WARC Range recovery; no Pharmacity requests are made."""
    if not 1 <= int(workers) <= WARC_WORKERS:
        raise ValueError('Common Crawl WARC workers must be in 1..5')
    roots.preflight()
    if not roots.mapping.is_file() or not roots.state.is_file():
        raise FileNotFoundError('S27 mapping and crawl state are required before CC recovery')
    started = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
    unknown_marked = _mark_317850_unknown(roots)
    scope = _scope(roots)
    target_by_surt: dict[str, list[int]] = {}
    by_id = {}
    for row in scope:
        by_id[int(row['doc_id'])] = row
        for key in index_surt_keys(row['mapped_url']):
            target_by_surt.setdefault(key, []).append(int(row['doc_id']))

    cc_root = roots.guard(roots.root / 'cc-backfill')
    cc_root.mkdir(parents=True, exist_ok=True)
    index_ledger = Ledger(roots.guard(cc_root / 'index_ledger.json'))
    index_client = Client(ua=CC_UA, ledger=index_ledger, log_path=roots.guard(cc_root / 'index_requests.jsonl'), rps=1.0)
    crawls = []
    footer_dir = RUNS_DIR / 'vi-longchau' / 'cc-coverage'
    try:
        for crawl in CC_CRAWLS_NEWEST_FIRST:
            crawls.append(_scan_crawl(index_client, roots, crawl, scope, target_by_surt, footer_dir))
    finally:
        index_client.close()

    captures_by_id: dict[int, list[dict]] = {doc_id: [] for doc_id in by_id}
    for crawl in CC_CRAWLS_NEWEST_FIRST:
        path = roots.guard(cc_root / 'index' / f'rows_{crawl}.parquet')
        if not path.is_file():
            raise FileNotFoundError(f'missing complete CC URL index results: {crawl}')
        for row in pq.read_table(path).to_pylist():
            captures_by_id[int(row['doc_id'])].append(row)

    db = _connect_cc(roots)
    for doc_id, captures in captures_by_id.items():
        old = db.execute('SELECT outcome,captures FROM targets WHERE doc_id=?', (doc_id,)).fetchone()
        if old and old[0] != 'usable' and int(old[1]) != len(captures):
            db.execute('UPDATE targets SET outcome=?,captures=?,updated_at=? WHERE doc_id=?',
                       ('pending', len(captures), time.time(), doc_id))
    db.commit()
    pages_db = sqlite3.connect(roots.state, timeout=60, isolation_level=None, check_same_thread=False)
    pages_db.execute('PRAGMA synchronous=FULL')
    lock = threading.Lock()
    range_ledger = Ledger(roots.guard(cc_root / 'warc_ledger.json'), max_bytes=WARC_MAX_BYTES, max_requests=WARC_MAX_REQUESTS)
    range_client = RangeClient(ledger=range_ledger, log_path=roots.guard(cc_root / 'warc_requests.jsonl'),
                               rps=4.0, ua=CC_UA)
    try:
        work = []
        for doc_id, target in by_id.items():
            done = db.execute('SELECT outcome FROM targets WHERE doc_id=?', (doc_id,)).fetchone()
            if done and done[0] in ('usable','unusable','no_capture','dead'):
                continue
            work.append((target, captures_by_id[doc_id]))
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_process_target, range_client, roots, db, pages_db, lock, target, captures)
                       for target, captures in work]
            for future in as_completed(futures):
                result = future.result()
                if result.get('outcome') == 'pending':
                    continue
        range_client.close()
        report = _summary(roots, scope, crawls, db, started, unknown_marked)
        report['index_ledger'] = {'bytes': index_ledger.bytes, 'requests': index_ledger.requests}
        report['warc_ledger'] = {'bytes': range_ledger.bytes, 'requests': range_ledger.requests, 'workers': int(workers)}
        _write_json(roots, roots.guard(roots.root / 'reports' / 'cc_backfill.json'), report)
        return report
    finally:
        range_client.close()
        db.close()
        pages_db.close()


def cc_backfill_status(roots: P.Roots) -> dict:
    path = roots.guard(roots.root / 'reports' / 'cc_backfill.json')
    if not path.is_file():
        return {'batch': P.BATCH, 'status': 'not_started'}
    return json.loads(path.read_text(encoding='utf-8'))
