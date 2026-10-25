"""Pharmacity URL remap, respectful fetch, S15 extraction/chunk handoff, and gated S26 embeddings."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.paths import DATA_DIR, OLD_ROOT, assert_writable, resolve_path, run_dir

SOURCE_HOST = 'pmc-ecm-healthblog.beta.pharmacity.io'
TARGET_HOST = 'www.pharmacity.vn'
DOMAIN = 'pharmacity.vn'
BODY_SELECTOR = '#pmc-content-blog-detail'
BATCH = 'cc-s27-pharmacity'
USER_AGENT = 'r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)'
MIN_REQUEST_INTERVAL = 1.05
MIN_BODY_CP = 200
MIN_BODY_TOKENS = 50
TARGET_CHUNK_TOKENS = 256
S26_DOC_COUNT = 887_151
EXPECTED_BETA_URLS = 5_333
EXPECTED_MAPPING_ROWS = 5_337
CRAWL_TERMINAL = frozenset(('ok', 'low_title_match', 'soft404_or_nonarticle', 'redirect_slug_mismatch', 'unparseable',
                            'unknown', 'cc_dead'))
S26_FULL_VECTOR_CANDIDATES = (
    run_dir('s26-gold-structure') / 'full' / 'full_doc_dense.npz',
    run_dir('s26-gold-structure') / 'full_doc_dense.npz',
)

# User-approved fixed Vietnamese stopword set, accent-stripped.
STOPWORDS = frozenset('''
va la cua cho co khong the nao gi nhung cac mot voi khi de bi duoc nen tai sao lam nhu trong ve tu den
hay hoac can biet cach o se da dang rat nay do ai bao nhieu
'''.split())

MISSING_PHARMACITY = {
    720437: 'https://www.pharmacity.vn/dau-hieu-thieu-canxi.htm',
    720467: 'https://www.pharmacity.vn/thieu-hut-collagen.htm',
    720468: 'https://www.pharmacity.vn/thieu-hut-omega-3.htm',
    720435: 'https://www.pharmacity.vn/collagen-la-gi.htm',
}
SAMPLE_IDS = (317905, 318613, 319707, 319954, 322939)

MAPPING_SCHEMA = pa.schema([
    ('doc_id', pa.int64()), ('original_url', pa.string()), ('mapped_url', pa.string()),
    ('slug', pa.string()), ('source_type', pa.string()), ('slug_parseable', pa.bool_()),
])
PROVENANCE_SCHEMA = pa.schema([
    ('doc_id', pa.int64()), ('original_url', pa.string()), ('mapped_url', pa.string()),
    ('final_url', pa.string()), ('fetched_at', pa.string()), ('http_status', pa.int32()),
    ('body_selector', pa.string()), ('body_chars', pa.int32()), ('raw_sha256', pa.string()),
    ('title_match_ratio', pa.float32()), ('title_match_count', pa.int32()), ('title_token_count', pa.int32()),
    ('fetch_status', pa.string()), ('reason', pa.string()),
])


def _under(path: Path, root: Path) -> bool:
    p = os.path.normcase(str(resolve_path(path)))
    r = os.path.normcase(str(resolve_path(root)))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


@dataclass(frozen=True)
class Roots:
    """Every S27 artifact lives in its isolated run directory, never the shared index."""
    root: Path = run_dir('s27-pharmacity')

    @property
    def mapping(self) -> Path: return self.root / 'mapping.parquet'
    @property
    def state(self) -> Path: return self.root / 'state' / 'crawl.sqlite'
    @property
    def raw(self) -> Path: return self.root / 'raw'
    @property
    def docs(self) -> Path: return self.root / 'docs'
    @property
    def chunks(self) -> Path: return self.root / 'chunks'
    @property
    def reports(self) -> Path: return self.root / 'reports'
    @property
    def vectors(self) -> Path: return self.root / 'vectors'

    def guard(self, path: str | Path) -> Path:
        p = resolve_path(path)
        if not _under(p, self.root):
            raise ValueError(f'Not an S27 batch output: {p}')
        if OLD_ROOT is not None and _under(p, OLD_ROOT):
            raise ValueError(f'Write to OLD is forbidden: {p}')
        return p

    def preflight(self) -> None:
        # Full recursive write-target scan is intentionally CLI-only, never called per URL/shard.
        assert_writable(self.guard(self.root))


def atomic_bytes(roots: Roots, path: str | Path, data: bytes) -> None:
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(roots: Roots, path: str | Path, value) -> None:
    atomic_bytes(roots, path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8'))


def _accentless(value: str) -> str:
    value = value.lower().replace('đ', 'd')
    return ''.join(c for c in unicodedata.normalize('NFD', value) if unicodedata.category(c) != 'Mn')


def _tokens(value: str) -> list[str]:
    return [x for x in re.findall(r'[a-z]+', _accentless(value)) if len(x) > 1 and x not in STOPWORDS]


def parse_beta_slug(url: str) -> str | None:
    try:
        p = urlsplit(url)
    except ValueError:
        return None
    segments = [x for x in p.path.split('/') if x]
    if (p.scheme != 'https' or (p.hostname or '').lower() != SOURCE_HOST or p.query or p.fragment
            or len(segments) != 1 or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', segments[0])):
        return None
    return segments[0]


def mapped_url(url: str) -> str | None:
    slug = parse_beta_slug(url)
    return f'https://{TARGET_HOST}/{slug}.htm' if slug is not None else None


def title_match(slug: str, title: str) -> dict:
    terms = _tokens(slug.replace('-', ' '))
    title_words = set(_tokens(title))
    matched = sum(word in title_words for word in terms)
    ratio = matched / len(terms) if terms else 0.0
    return {'matched': matched, 'terms': len(terms), 'ratio': ratio, 'accepted': bool(terms) and ratio >= 0.5}


def _meta(tree, keys: tuple[str, ...]) -> str:
    for key in keys:
        vals = tree.xpath(f'//meta[@property="{key}" or @name="{key}"]/@content')
        for value in vals:
            if value.strip():
                return value.strip()
    return ''


def selector_document(raw_html: str) -> dict:
    """Extract title and article blocks from the observed Pharmacity selector only."""
    from lxml import html as lhtml
    from vicrawl.extractors.base import GENERIC_JUNK, norm, strip_site_junk

    try:
        tree = lhtml.fromstring(raw_html)
    except (TypeError, ValueError, lhtml.ParserError):
        return {'title': '', 'description': '', 'body': '', 'paragraphs': [], 'headings': [], 'selector_found': False}
    title_nodes = tree.xpath('//h1[normalize-space()]')
    title = norm(title_nodes[0].text_content()) if title_nodes else norm(_meta(tree, ('og:title', 'twitter:title'))) or norm(' '.join(tree.xpath('//title/text()')))
    description = _meta(tree, ('og:description', 'description'))
    nodes = tree.cssselect(BODY_SELECTOR)
    if not nodes:
        return {'title': title, 'description': description, 'body': '', 'paragraphs': [], 'headings': [], 'selector_found': False}
    article = nodes[0]
    paragraphs, headings = [], []
    for node in article.xpath('.//p|.//h2|.//h3|.//h4|.//li|.//blockquote|.//td|.//th'):
        if node.tag == 'li' and node.xpath('.//p|.//li'):
            continue
        value = norm(node.text_content())
        if not value:
            continue
        if GENERIC_JUNK.match(value):
            continue
        if node.tag in ('h2', 'h3', 'h4'):
            headings.append(value)
        if not paragraphs or paragraphs[-1] != value:
            paragraphs.append(value)
    if not paragraphs:
        value = norm(article.text_content())
        if value:
            paragraphs = [value]
    paragraphs = strip_site_junk(paragraphs, DOMAIN)
    body = '\n\n'.join(paragraphs)
    return {'title': title, 'description': description, 'body': body, 'paragraphs': paragraphs,
            'headings': [h for h in headings if h in set(paragraphs)], 'selector_found': True}


def _slug_for_url(url: str) -> str | None:
    p = urlsplit(url)
    parts = [x for x in p.path.split('/') if x]
    if len(parts) != 1:
        return None
    return re.sub(r'\.htm$', '', parts[0], flags=re.I).lower()


def redirect_changes_slug(mapped: str, final: str) -> bool:
    return _slug_for_url(mapped) != _slug_for_url(final)


def allowed_redirect_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    return (parts.scheme == 'https' and (parts.hostname or '').lower() in (TARGET_HOST, DOMAIN)
            and port in (None, 443) and parts.username is None and parts.password is None)


def build_mapping(corpus_path: Path | None = None) -> tuple[list[dict], dict]:
    """Map every beta doc_id and the four Pharmacity source URLs missing from S26."""
    corpus_path = resolve_path(corpus_path or DATA_DIR / 'corpus_v2_ids.parquet')
    table = pq.read_table(corpus_path, columns=['id', 'url'])
    by_id = {int(i): u for i, u in zip(table['id'].to_pylist(), table['url'].to_pylist())}
    rows, parseable, unparseable = [], 0, []
    for doc_id, url in by_id.items():
        host = (urlsplit(url).hostname or '').lower()
        if host == SOURCE_HOST:
            slug = parse_beta_slug(url)
            if slug is None:
                unparseable.append({'doc_id': doc_id, 'original_url': url})
                rows.append({'doc_id': doc_id, 'original_url': url, 'mapped_url': None, 'slug': None,
                             'source_type': 'beta', 'slug_parseable': False})
            else:
                parseable += 1
                rows.append({'doc_id': doc_id, 'original_url': url, 'mapped_url': mapped_url(url), 'slug': slug,
                             'source_type': 'beta', 'slug_parseable': True})
    for doc_id, target in MISSING_PHARMACITY.items():
        if by_id.get(doc_id) != target:
            raise ValueError(f'Corpus mapping changed for missing Pharmacity doc_id={doc_id}')
        slug = _slug_for_url(target)
        rows.append({'doc_id': doc_id, 'original_url': target, 'mapped_url': target, 'slug': slug,
                     'source_type': 'pharmacity_missing', 'slug_parseable': slug is not None})
    rows.sort(key=lambda r: r['doc_id'])
    if len({r['doc_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate doc_id in S27 mapping')
    if len(rows) != EXPECTED_MAPPING_ROWS or parseable != EXPECTED_BETA_URLS or unparseable:
        raise ValueError(f'corpus scope changed: expected {EXPECTED_BETA_URLS} parseable beta URLs + '
                         f'{len(MISSING_PHARMACITY)} direct missing URLs, got rows={len(rows)}, '
                         f'parseable_beta={parseable}, unparseable_beta={len(unparseable)}')
    report = {'batch': BATCH, 'rows': len(rows), 'beta_rows': parseable + len(unparseable),
              'beta_slug_parseable': parseable, 'beta_slug_unparseable': len(unparseable),
              'direct_missing_rows': len(MISSING_PHARMACITY), 'unparseable': unparseable,
              'source_corpus': str(corpus_path), 'source_corpus_sha256': _sha256_file(corpus_path)}
    return rows, report


def _connect(roots: Roots) -> sqlite3.Connection:
    path = roots.guard(roots.state)
    roots.guard(path.with_name(path.name + '-wal'))
    roots.guard(path.with_name(path.name + '-shm'))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=60)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.executescript('''
        CREATE TABLE IF NOT EXISTS pages(
          doc_id INTEGER PRIMARY KEY, original_url TEXT NOT NULL, mapped_url TEXT NOT NULL,
          status TEXT NOT NULL, http_status INTEGER, final_url TEXT, title TEXT, title_match_ratio REAL,
          title_match_count INTEGER, title_token_count INTEGER, body_chars INTEGER, fetched_at TEXT,
          raw_path TEXT, raw_sha256 TEXT, attempts INTEGER NOT NULL DEFAULT 0, reason TEXT, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    return c


def _write_raw(roots: Roots, doc_id: int, content: bytes) -> tuple[str, str]:
    path = roots.guard(roots.raw / f'{doc_id}.html.gz')
    payload = gzip.compress(content, compresslevel=6, mtime=0)
    atomic_bytes(roots, path, payload)
    return str(path), hashlib.sha256(content).hexdigest()


def _request_delay(*, elapsed: float, minimum: float, jitter_s: float, jitter_value: float) -> float:
    delay = max(0.0, minimum - elapsed)
    if delay > 0:
        delay += max(0.0, jitter_s) * min(1.0, max(0.0, jitter_value))
    return delay


class RequestObserverError(RuntimeError):
    """Stop safely when the durable per-request audit marker cannot be written."""


def _request_once(session, c, url: str, timeout: float = 40, *, min_interval: float = MIN_REQUEST_INTERVAL,
                  jitter_s: float = 0.0, rng_value: float = 0.0, request_observer=None):
    last = c.execute("SELECT value FROM meta WHERE key='last_request_wall'").fetchone()
    previous_started = float(last[0]) if last else None
    if last:
        wait = _request_delay(elapsed=time.time() - previous_started, minimum=min_interval,
                              jitter_s=jitter_s, jitter_value=rng_value)
        if wait > 0:
            time.sleep(wait)
    started = time.time()
    c.execute("INSERT INTO meta(key,value) VALUES('last_request_wall',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
              (str(started),))
    c.commit()
    gap = started - previous_started if previous_started is not None else None
    if request_observer:
        try:
            request_observer({'phase': 'start', 'url': url,
                              'requested_at_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(
                                  timespec='milliseconds').replace('+00:00', 'Z'), 'gap_from_previous_start_s': gap})
        except Exception as exc:
            raise RequestObserverError(f'cannot persist request start marker: {exc}') from exc
    try:
        response = session.get(url, allow_redirects=False, timeout=timeout)
    except Exception as exc:
        if request_observer:
            try:
                request_observer({'phase': 'finish', 'url': url, 'http_status': None,
                                  'error': f'{type(exc).__name__}: {exc}'})
            except Exception as observer_exc:
                raise RequestObserverError(f'cannot persist request result marker: {observer_exc}') from observer_exc
        raise
    if request_observer:
        try:
            request_observer({'phase': 'finish', 'url': url, 'http_status': int(response.status_code)})
        except Exception as exc:
            raise RequestObserverError(f'cannot persist request result marker: {exc}') from exc
    return response, started


def _challenge(raw: bytes, status: int) -> tuple[bool, str]:
    from vicrawl.classify import analyze_html, classify_response

    decoded = raw.decode('utf-8', errors='replace')
    stats = analyze_html(decoded)
    kind, reason = classify_response(stats=stats, raw=decoded, url='https://www.pharmacity.vn/',
                                     final_url='https://www.pharmacity.vn/', http_status=status)
    return kind in ('bot_challenge', 'cookie_challenge'), reason


def _fetch_target(roots: Roots, c: sqlite3.Connection, row: dict, session, *,
                  min_request_interval: float = MIN_REQUEST_INTERVAL, jitter_s: float = 0.0,
                  rng=None, max_hops: int = 6, request_observer=None) -> dict:
    if not 1 <= int(max_hops) <= 6:
        raise ValueError('max_hops must be in 1..6')
    doc_id, original, target, expected_slug = row['doc_id'], row['original_url'], row['mapped_url'], row['slug']
    current, final_url, status, raw, started = target, target, None, b'', None
    redirect_mismatch = False
    redirect_reason = 'redirect_target_has_different_slug'
    for _hop in range(int(max_hops)):
        try:
            response, started = _request_once(
                session, c, current, min_interval=min_request_interval, jitter_s=jitter_s,
                rng_value=(rng.random() if rng is not None else 0.0), request_observer=request_observer)
        except RequestObserverError:
            raise
        except Exception as exc:  # noqa: BLE001 - saved as resumable network failure
            c.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,final_url,attempts,reason,updated_at)
                         VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET status=excluded.status,
                         attempts=pages.attempts+1,reason=excluded.reason,updated_at=excluded.updated_at''',
                      (doc_id, original, target, 'retryable_error', current, 1, f'{type(exc).__name__}: {exc}', time.time()))
            c.commit()
            return {'doc_id': doc_id, 'status': 'retryable_error', 'reason': str(exc)}
        status, raw = response.status_code, response.content
        final_url = current
        if status in (403, 429):
            raw_path, raw_sha = _write_raw(roots, doc_id, raw)
            fetched_at = datetime.fromtimestamp(started, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            c.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,http_status,final_url,fetched_at,raw_path,raw_sha256,
                         attempts,reason,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET
                         status=excluded.status,http_status=excluded.http_status,final_url=excluded.final_url,
                         fetched_at=excluded.fetched_at,
                         raw_path=excluded.raw_path,raw_sha256=excluded.raw_sha256,attempts=pages.attempts+1,
                         reason=excluded.reason,updated_at=excluded.updated_at''',
                      (doc_id, original, target, 'halted_block', status, current, fetched_at, raw_path, raw_sha, 1,
                       f'http_{status}', time.time()))
            c.commit()
            raise RuntimeError(f'crawl halted on HTTP {status}, doc_id={doc_id}, URL={current}')
        is_challenge, challenge_reason = _challenge(raw, status)
        if is_challenge:
            raw_path, raw_sha = _write_raw(roots, doc_id, raw)
            fetched_at = datetime.fromtimestamp(started, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            c.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,http_status,final_url,fetched_at,raw_path,raw_sha256,
                         attempts,reason,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET
                         status=excluded.status,http_status=excluded.http_status,final_url=excluded.final_url,
                         fetched_at=excluded.fetched_at,
                         raw_path=excluded.raw_path,raw_sha256=excluded.raw_sha256,attempts=pages.attempts+1,
                         reason=excluded.reason,updated_at=excluded.updated_at''',
                      (doc_id, original, target, 'halted_captcha', status, current, fetched_at, raw_path, raw_sha, 1,
                       challenge_reason, time.time()))
            c.commit()
            raise RuntimeError(f'crawl halted on CAPTCHA/challenge, doc_id={doc_id}, HTTP {status}, URL={current}')
        if 300 <= status < 400:
            location = response.headers.get('Location')
            if not location:
                break
            next_url = urljoin(current, location)
            if not allowed_redirect_url(next_url):
                final_url = next_url
                redirect_mismatch = True
                redirect_reason = 'redirect_outside_pharmacity_domain'
                break
            if redirect_changes_slug(target, next_url):
                final_url = next_url
                redirect_mismatch = True
                break
            current = next_url
            final_url = current
            continue
        break
    fetched_at = datetime.fromtimestamp(started or time.time(), timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    raw_path, raw_sha = _write_raw(roots, doc_id, raw)
    decoded = raw.decode('utf-8', errors='replace')
    page = selector_document(decoded) if status is not None and 200 <= status < 300 else {'title': '', 'body': '', 'selector_found': False}
    score = title_match(expected_slug or '', page['title'])
    from vicrawl.classify import analyze_html, classify_response
    page_kind, page_reason = classify_response(stats=analyze_html(decoded), raw=decoded, url=target,
                                                final_url=final_url, http_status=status)
    if redirect_mismatch:
        state, reason = 'redirect_slug_mismatch', redirect_reason
    elif page_kind == 'soft404_or_home' or status in (404, 410) or not page.get('selector_found'):
        state, reason = 'soft404_or_nonarticle', page_reason or ('http_not_found' if status in (404, 410) else 'article_selector_missing')
    elif status is None or status >= 500 or status < 200 or status >= 400:
        state, reason = 'retryable_error', f'http_{status}'
    elif not score['accepted']:
        state, reason = 'low_title_match', 'title_match_below_50_percent'
    else:
        state, reason = 'ok', ''
    c.execute('''INSERT INTO pages(doc_id,original_url,mapped_url,status,http_status,final_url,title,title_match_ratio,
                 title_match_count,title_token_count,body_chars,fetched_at,raw_path,raw_sha256,attempts,reason,updated_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET
                 status=excluded.status,http_status=excluded.http_status,final_url=excluded.final_url,title=excluded.title,
                 title_match_ratio=excluded.title_match_ratio,title_match_count=excluded.title_match_count,
                 title_token_count=excluded.title_token_count,body_chars=excluded.body_chars,fetched_at=excluded.fetched_at,
                 raw_path=excluded.raw_path,raw_sha256=excluded.raw_sha256,attempts=pages.attempts+1,
                 reason=excluded.reason,updated_at=excluded.updated_at''',
              (doc_id, original, target, state, status, final_url, page.get('title', ''), score['ratio'], score['matched'],
               score['terms'], len(page.get('body', '')), fetched_at, raw_path, raw_sha, 1, reason, time.time()))
    c.commit()
    return {'doc_id': doc_id, 'status': state, 'http_status': status, 'title_match_ratio': score['ratio'],
            'body_chars': len(page.get('body', '')), 'selector_found': bool(page.get('selector_found')),
            'title': page.get('title', ''), 'final_url': final_url, 'reason': reason}


def _load_mapping(roots: Roots) -> list[dict]:
    rows = pq.read_table(roots.mapping).to_pylist()
    report_path = roots.guard(roots.reports / 'mapping.json')
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    source_corpus = Path(report['source_corpus']) if report.get('source_corpus') else DATA_DIR / 'corpus_v2_ids.parquet'
    expected, source_report = build_mapping(source_corpus)
    if report.get('source_corpus_sha256') and report['source_corpus_sha256'] != source_report['source_corpus_sha256']:
        raise ValueError('source corpus changed since S27 mapping was recorded')
    if rows != expected:
        raise ValueError('stored S27 mapping differs from source corpus mapping')
    return rows


def crawl_coverage(mapping: list[dict], statuses: dict[int, str]) -> dict:
    """Summarize whether every mapped URL has a terminal result."""
    crawlable = [row for row in mapping if row['mapped_url']]
    incomplete = [{'doc_id': row['doc_id'], 'status': statuses.get(row['doc_id'], 'missing')}
                  for row in crawlable if statuses.get(row['doc_id']) not in CRAWL_TERMINAL]
    return {'complete': not incomplete, 'crawlable_urls': len(crawlable),
            'terminal_urls': len(crawlable) - len(incomplete), 'incomplete_urls': len(incomplete),
            'incomplete_examples': incomplete[:20]}


def _crawl_coverage(roots: Roots) -> dict:
    mapping = _load_mapping(roots)
    if not roots.state.exists():
        return crawl_coverage(mapping, {})
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        statuses = dict(c.execute('SELECT doc_id,status FROM pages'))
    finally:
        c.close()
    return crawl_coverage(mapping, statuses)


def current_crawl_status(roots: Roots) -> tuple[str, dict | None]:
    if not roots.mapping.exists():
        return 'not_started', None
    coverage = _crawl_coverage(roots)
    if roots.state.exists():
        c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            statuses = [row[0] for row in c.execute('SELECT status FROM pages')]
        finally:
            c.close()
    else:
        statuses = []
    if any(status in ('halted_block', 'halted_captcha') for status in statuses):
        return 'halted', coverage
    if coverage['complete']:
        return 'complete', coverage
    return ('incomplete' if statuses else 'not_started'), coverage


def require_crawl_complete(roots: Roots) -> dict:
    coverage = _crawl_coverage(roots)
    if not coverage['complete']:
        raise RuntimeError(f"S27 crawl is incomplete; downstream step blocked: {coverage}")
    return coverage


def _next_handoff_ids(eligible_ids, previous_ids, *, limit: int) -> list[int]:
    """Choose a stable, duplicate-free incremental handoff lot (later lots are capped at 1,000 docs)."""
    if not 1 <= int(limit) <= 1000:
        raise ValueError('handoff lot limit must be in 1..1000')
    previous = {int(x) for x in previous_ids}
    return sorted({int(x) for x in eligible_ids} - previous)[:int(limit)]


def _refresh_title_scores(c: sqlite3.Connection, mapping: list[dict]) -> int:
    """Recompute cached article scores using the current fixed title normalization."""
    slugs = {row['doc_id']: row['slug'] or '' for row in mapping}
    rows = c.execute("SELECT doc_id,title FROM pages WHERE http_status=200 AND status IN ('ok','low_title_match')").fetchall()
    for doc_id, title in rows:
        score = title_match(slugs.get(doc_id, ''), title or '')
        status = 'ok' if score['accepted'] else 'low_title_match'
        reason = '' if score['accepted'] else 'title_match_below_50_percent'
        c.execute('UPDATE pages SET status=?,title_match_ratio=?,title_match_count=?,title_token_count=?,reason=? WHERE doc_id=?',
                  (status, score['ratio'], score['matched'], score['terms'], reason, doc_id))
    if rows:
        c.commit()
    return len(rows)


def _sample_report(roots: Roots, c: sqlite3.Connection, mapping: list[dict]) -> dict:
    by_id = {r['doc_id']: r for r in mapping}
    results = []
    for doc_id in SAMPLE_IDS:
        row = by_id[doc_id]
        got = c.execute('SELECT status,http_status,final_url,title,title_match_ratio,body_chars,reason FROM pages WHERE doc_id=?',
                         (doc_id,)).fetchone()
        if not got:
            raise ValueError(f'missing sample fetch record {doc_id}')
        state, code, final_url, title, ratio, body_chars, reason = got
        if state.startswith('halted_'):
            raise RuntimeError(f'sample halted: doc_id={doc_id} state={state} reason={reason}')
        results.append({'doc_id': doc_id, 'url': row['mapped_url'], 'http_status': code, 'final_url': final_url,
                        'title': title, 'title_match_ratio': ratio, 'body_chars': body_chars, 'status': state,
                        'body_selector': BODY_SELECTOR, 'reason': reason})
    if any(r['http_status'] != 200 or r['status'] != 'ok' for r in results):
        raise ValueError('sample validation failed: each of five must be HTTP 200 and title-match accepted')
    summary = {'samples': results, 'all_http_200': True, 'all_title_match_accepted': True,
               'inferred_body_selector': BODY_SELECTOR}
    atomic_json(roots, roots.reports / 'sample.json', summary)
    return summary


def map_batch(roots: Roots, corpus_path: Path | None = None) -> dict:
    roots.preflight()
    rows, report = build_mapping(corpus_path)
    path = roots.guard(roots.mapping)
    if path.exists():
        if pq.read_table(path).to_pylist() != rows:
            raise ValueError('existing S27 mapping differs; refusing to replace a crawl snapshot')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = roots.guard(path.with_name(path.name + '.partial'))
        pq.write_table(pa.Table.from_pylist(rows, schema=MAPPING_SCHEMA), tmp, compression='zstd')
        os.replace(tmp, path)
    atomic_json(roots, roots.reports / 'mapping.json', report)
    return report


@contextmanager
def _exclusive_crawl(roots: Roots):
    """Serialize all crawl processes so separate invocations cannot violate the global request interval."""
    import msvcrt

    path = roots.guard(roots.root / 'crawl.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('another S27 crawl process is active') from exc
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def sample_and_crawl(roots: Roots, *, sample_only: bool = False, corpus_path: Path | None = None,
                     progress_every: int = 100, resume_after_block: bool = False) -> dict:
    with _exclusive_crawl(roots):
        return _sample_and_crawl_locked(roots, sample_only=sample_only, corpus_path=corpus_path,
                                        progress_every=progress_every, resume_after_block=resume_after_block)


def _sample_and_crawl_locked(roots: Roots, *, sample_only: bool = False, corpus_path: Path | None = None,
                             progress_every: int = 100, resume_after_block: bool = False) -> dict:
    roots.preflight()
    if not roots.mapping.exists():
        map_batch(roots, corpus_path)
    mapping = _load_mapping(roots)
    c = _connect(roots)
    if progress_every < 1:
        c.close()
        raise ValueError('progress_every must be >= 1')
    import requests
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'User-Agent': USER_AGENT, 'Accept': 'text/html,application/xhtml+xml'})
    try:
        blocked = c.execute("SELECT doc_id,status,http_status FROM pages WHERE status IN ('halted_block','halted_captcha') LIMIT 1").fetchone()
        if blocked:
            crawl_halt_report(roots)
            if not resume_after_block:
                raise RuntimeError(f'crawl remains halted at doc_id={blocked[0]} status={blocked[1]} HTTP={blocked[2]}')
            c.execute("UPDATE pages SET status='retryable_error',reason='explicit_resume_after_block',updated_at=? WHERE status IN ('halted_block','halted_captcha')",
                      (time.time(),))
            c.commit()
        by_id = {r['doc_id']: r for r in mapping}
        _refresh_title_scores(c, mapping)
        # Sample is a gate and its fetched raw pages are reused by the full crawl.
        try:
            for doc_id in SAMPLE_IDS:
                got = c.execute('SELECT status FROM pages WHERE doc_id=?', (doc_id,)).fetchone()
                if not got or got[0] in ('retryable_error', 'pending'):
                    _fetch_target(roots, c, by_id[doc_id], session)
        except RuntimeError:
            crawl_halt_report(roots)
            raise
        sample = _sample_report(roots, c, mapping)
        if sample_only:
            return {'sample': sample, 'crawl': 'not_started'}
        terminal = CRAWL_TERMINAL
        statuses = {r[0]: r[1] for r in c.execute('SELECT doc_id,status FROM pages')}
        pending = [r for r in mapping if r['mapped_url'] and statuses.get(r['doc_id']) not in terminal]
        counts = {}
        for i, row in enumerate(pending, 1):
            try:
                result = _fetch_target(roots, c, row, session)
            except RuntimeError:
                crawl_halt_report(roots)
                raise
            counts[result['status']] = counts.get(result['status'], 0) + 1
            if i % progress_every == 0 or i == len(pending):
                print(json.dumps({'attempted_this_run': i, 'pending_at_start': len(pending), 'counts': counts,
                                  'last_doc_id': row['doc_id'], 'last_status': result['status']}, ensure_ascii=False), flush=True)
        all_counts = {s: n for s, n in c.execute('SELECT status,count(*) FROM pages GROUP BY status')}
        coverage = crawl_coverage(mapping, dict(c.execute('SELECT doc_id,status FROM pages')))
        report = {'batch': BATCH, 'mapped_docs': len(mapping), 'attempted_this_run': len(pending),
                  'status_counts': all_counts, 'crawl_coverage': coverage, 'complete': coverage['complete'],
                  'sample': {'count': len(sample['samples']), 'selector': BODY_SELECTOR}}
        atomic_json(roots, roots.reports / 'crawl.json', report)
        return report
    finally:
        session.close()
        c.close()


def crawl_halt_report(roots: Roots) -> dict:
    """Read a stopped checkpoint, classify its saved challenge body, and write an offline report."""
    roots.preflight()
    from vicrawl.classify import analyze_html, classify_response

    if not roots.state.exists():
        return {'batch': BATCH, 'status': 'not_started'}
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        counts = dict(c.execute('SELECT status,count(*) FROM pages GROUP BY status'))
        n = c.execute('SELECT count(*) FROM pages').fetchone()[0]
        last = c.execute('''SELECT doc_id,original_url,mapped_url,http_status,final_url,status,reason,fetched_at,
                            raw_path,raw_sha256,updated_at FROM pages WHERE status IN ('halted_block','halted_captcha')
                            ORDER BY updated_at DESC LIMIT 1''').fetchone()
        under50 = c.execute('SELECT count(*) FROM pages WHERE http_status=200 AND coalesce(title_match_ratio,0)<0.5').fetchone()[0]
        n200 = c.execute('SELECT count(*) FROM pages WHERE http_status=200').fetchone()[0]
    finally:
        c.close()
    mapping = _load_mapping(roots)
    if not last:
        report = {'batch': BATCH, 'status': 'no_active_halt', 'records': n, 'status_counts': counts}
    else:
        doc_id, original, mapped, code, final, state, reason, fetched_at, raw_path, raw_sha, updated = last
        raw = gzip.open(roots.guard(raw_path), 'rb').read() if raw_path and Path(raw_path).exists() else b''
        decoded = raw.decode('utf-8', errors='replace')
        stats = analyze_html(decoded)
        kind, classification_reason = classify_response(stats=stats, raw=decoded, url=mapped, final_url=final or mapped,
                                                        http_status=code)
        markers = [m for m in ('/cdn-cgi/challenge-platform', 'cf-chl', 'cf_chl', 'g-recaptcha', 'hcaptcha')
                   if m.lower() in decoded.lower()]
        fetched_time = fetched_at or datetime.fromtimestamp(updated, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
        report = {'batch': BATCH, 'status': 'halted', 'crawl_complete': False, 'halt': {
                      'doc_id': doc_id, 'original_url': original, 'mapped_url': mapped, 'final_url': final,
                      'http_status': code, 'state': state, 'stored_reason': reason,
                      'classification': kind, 'classification_reason': classification_reason,
                      'challenge_markers': markers, 'fetched_at_utc': fetched_time,
                      'response_html_bytes': len(raw), 'visible_codepoints': stats.visible_len,
                      'main_text_codepoints': stats.text_len, 'page_title': stats.title, 'raw_sha256': raw_sha},
                  'processed_urls': n, 'mapped_urls': len(mapping), 'processed_fraction': n / max(1, len(mapping)),
                  'remaining_urls': max(0, len(mapping) - n), 'status_counts': counts,
                  'partial_title_match': {'http_200': n200, 'below_50_percent': under50,
                                          'below_50_percent_of_total_mapping': under50 / max(1, len(mapping)),
                                          'final_distribution_pending': True},
                  'extract_chunk_embed': 'not_started', 'shared_index_written': False}
    if last:
        atomic_json(roots, roots.reports / 'crawl_halted.json', report)
    return report


def title_match_report(roots: Roots) -> dict:
    roots.preflight()
    c = _connect(roots)
    try:
        _refresh_title_scores(c, _load_mapping(roots))
        rows = c.execute('SELECT doc_id,original_url,mapped_url,final_url,title,title_match_ratio,status,reason '
                         'FROM pages WHERE http_status=200 ORDER BY title_match_ratio,doc_id').fetchall()
    finally:
        c.close()
    total_mapping = len(_load_mapping(roots))
    matched = [r for r in rows if r[6] == 'ok']
    under = [r for r in rows if (r[5] or 0) < 0.5]
    buckets = {'0%': 0, '(0, 50%)': 0, '50%': 0, '(50%, 75%)': 0, '[75%, 100%)': 0, '100%': 0}
    for r in rows:
        x = r[5] or 0.0
        if x == 0: buckets['0%'] += 1
        elif x < 0.5: buckets['(0, 50%)'] += 1
        elif x == 0.5: buckets['50%'] += 1
        elif x < 0.75: buckets['(50%, 75%)'] += 1
        elif x < 1: buckets['[75%, 100%)'] += 1
        else: buckets['100%'] += 1
    limit = 20 if len(under) / max(1, total_mapping) > 0.05 else 0
    examples = [{'doc_id': r[0], 'original_url': r[1], 'mapped_url': r[2], 'final_url': r[3], 'title': r[4],
                 'title_match_ratio': r[5], 'status': r[6], 'reason': r[7]} for r in under[:limit]]
    report = {'batch': BATCH, 'eligible_pages_http_200': len(rows), 'accepted_title_match': len(matched),
              'below_50_percent': len(under), 'below_50_percent_of_total_mapping': len(under) / max(1, total_mapping),
              'below_50_threshold_for_review': limit > 0, 'distribution': buckets,
              'below_50_examples': examples,
              'stopwords': sorted(STOPWORDS), 'token_policy': 'accentless lowercase; alpha tokens; drop numeric/1-char/stopwords; count slug token occurrences found in normalized title'}
    atomic_json(roots, roots.reports / 'title_match.json', report)
    return report


def extract_batch(roots: Roots) -> dict:
    roots.preflight()
    crawl_gate = require_crawl_complete(roots)
    from vicrawl.extract_pipeline import SCHEMA, default_tokenizer, detect_lang
    from vicrawl.urlnorm import normalize_url
    from vicrawl.extractors.base import norm
    c = _connect(roots)
    try:
        _refresh_title_scores(c, _load_mapping(roots))
        fetched = c.execute('''SELECT doc_id,original_url,mapped_url,final_url,status,http_status,title,title_match_ratio,
                              title_match_count,title_token_count,body_chars,fetched_at,raw_path,raw_sha256
                              FROM pages ORDER BY doc_id''').fetchall()
    finally:
        c.close()
    tok = default_tokenizer()
    rows, prov, dropped = [], [], {'low_title_match': 0, 'nonarticle_or_error': 0}
    for (doc_id, original, mapped, final, status, code, title, ratio, match_n, term_n, _, fetched_at,
         raw_path, raw_sha) in fetched:
        if status == 'low_title_match':
            dropped['low_title_match'] += 1
            continue
        if status != 'ok' or not raw_path or not fetched_at:
            dropped['nonarticle_or_error'] += 1
            continue
        with gzip.open(roots.guard(raw_path), 'rb') as f:
            raw = f.read().decode('utf-8', errors='replace')
        doc = selector_document(raw)
        body = doc['body']
        state = 'ok' if len(body) >= MIN_BODY_CP else 'thin'
        tokens = tok(body) if body else 0
        rows.append({'doc_ids': [doc_id], 'url': mapped, 'final_url': final, 'domain': DOMAIN,
                     'title': doc['title'] or title or '', 'description': doc['description'] or '', 'question': '', 'answer': '',
                     'body': body, 'paragraphs': doc['paragraphs'], 'lang': detect_lang(body), 'n_tokens_bge_m3': tokens,
                     'text_sha1': hashlib.sha1(body.encode('utf-8')).hexdigest(),
                     'fetched_at': datetime.fromisoformat(fetched_at.replace('Z', '+00:00')).timestamp(),
                     'url_norm': normalize_url(mapped), 'status': state, 'extractor': BODY_SELECTOR, 'headings': doc['headings']})
        prov.append({'doc_id': doc_id, 'original_url': original, 'mapped_url': mapped, 'final_url': final,
                     'fetched_at': fetched_at, 'http_status': code, 'body_selector': BODY_SELECTOR,
                     'body_chars': len(body), 'raw_sha256': raw_sha, 'title_match_ratio': ratio or 0.0,
                     'title_match_count': match_n or 0, 'title_token_count': term_n or 0,
                     'fetch_status': status, 'reason': ''})
    # Preserve per-URL provenance for flagged raw pages as well as the accepted handoff docs.
    all_prov = [{'doc_id': r[0], 'original_url': r[1], 'mapped_url': r[2], 'final_url': r[3],
                 'fetched_at': r[11], 'http_status': r[5], 'body_selector': BODY_SELECTOR,
                 'body_chars': r[10] or 0, 'raw_sha256': r[13], 'title_match_ratio': r[7] or 0.0,
                 'title_match_count': r[8] or 0, 'title_token_count': r[9] or 0,
                 'fetch_status': r[4], 'reason': ''} for r in fetched]
    docs_path = roots.guard(roots.docs / 'docs.parquet')
    prov_path = roots.guard(roots.docs / 'provenance.parquet')
    all_prov_path = roots.guard(roots.docs / 'provenance_all.parquet')
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_doc, tmp_prov = roots.guard(docs_path.with_suffix('.partial')), roots.guard(prov_path.with_suffix('.partial'))
    tmp_all_prov = roots.guard(all_prov_path.with_suffix('.partial'))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp_doc, compression='zstd')
    pq.write_table(pa.Table.from_pylist(prov, schema=PROVENANCE_SCHEMA), tmp_prov, compression='zstd')
    pq.write_table(pa.Table.from_pylist(all_prov, schema=PROVENANCE_SCHEMA), tmp_all_prov, compression='zstd')
    os.replace(tmp_doc, docs_path)
    os.replace(tmp_prov, prov_path)
    os.replace(tmp_all_prov, all_prov_path)
    summary = {'batch': BATCH, 'crawl_coverage': crawl_gate, 'docs': len(rows), 'provenance': len(prov), 'provenance_all_fetched': len(all_prov),
               'body_ge_200_codepoints': sum(r['status'] == 'ok' for r in rows),
               'body_lt_200_codepoints': sum(r['status'] == 'thin' for r in rows),
               'eligible_for_t256': sum(r['status'] == 'ok' and r['n_tokens_bge_m3'] >= MIN_BODY_TOKENS for r in rows),
               'dropped': dropped, 'doc_schema': str(SCHEMA), 'provenance_schema': str(PROVENANCE_SCHEMA)}
    atomic_json(roots, roots.reports / 'extract.json', summary)
    return summary


def chunk_batch(roots: Roots) -> dict:
    roots.preflight()
    crawl_gate = require_crawl_complete(roots)
    if not roots.guard(roots.reports / 'extract.json').exists():
        raise FileNotFoundError('S27 extraction report is required before chunking')
    if not roots.guard(roots.docs / 'docs.parquet').exists():
        raise FileNotFoundError('S27 extraction output is required before chunking')
    from r2ai.index.chunk import main as chunk_main
    out = roots.guard(roots.chunks)
    out.mkdir(parents=True, exist_ok=True)
    chunk_main(['--docs-dir', str(roots.guard(roots.docs)), '--out-dir', str(out), '--targets', str(TARGET_CHUNK_TOKENS),
                '--min-body-tokens', str(MIN_BODY_TOKENS), '--batch-docs', '5000'])
    report_path = roots.guard(out / 'chunk_report.json')
    chunk_report = json.loads(report_path.read_text(encoding='utf-8'))
    cfg = chunk_report['configs'][str(TARGET_CHUNK_TOKENS)]
    if cfg['verbatim_ok'] != f"{cfg['chunks_all']}/{cfg['chunks_all']}":
        raise ValueError(f'chunk verbatim gate failed: {cfg["verbatim_ok"]}')
    summary = {'batch': BATCH, 'crawl_coverage': crawl_gate, 'target_tokens': TARGET_CHUNK_TOKENS, 'min_body_tokens': MIN_BODY_TOKENS,
               'docs': chunk_report['docs'], 'chunks': cfg['chunks_all'], 'verbatim_ok': cfg['verbatim_ok'],
               'schema_docs': str(pq.read_schema(out / 'docs.parquet')),
               'schema_chunks': str(pq.read_schema(out / 'chunks_t256.parquet'))}
    atomic_json(roots, roots.reports / 'chunk.json', summary)
    return summary


def verify_batch(roots: Roots) -> dict:
    roots.preflight()
    crawl_gate = require_crawl_complete(roots)
    extracted_file = roots.guard(roots.docs / 'docs.parquet')
    doc_file = roots.guard(roots.chunks / 'docs.parquet')
    chunk_file = roots.guard(roots.chunks / 'chunks_t256.parquet')
    prov_file = roots.guard(roots.docs / 'provenance.parquet')
    extracted = pq.read_table(extracted_file, columns=['doc_ids', 'status', 'n_tokens_bge_m3', 'body',
                                                       'title', 'question', 'answer', 'paragraphs']).to_pylist()
    docs = pq.read_table(doc_file, columns=['doc_id', 'doc_text']).to_pylist()
    chunks = pq.read_table(chunk_file, columns=['doc_id', 'char_start', 'char_end', 'text']).to_pylist()
    prov = pq.read_table(prov_file, columns=['doc_id']).column(0).to_pylist()
    by_id = {r['doc_id']: r['doc_text'] for r in docs}
    bad = sum(by_id.get(r['doc_id'], '')[r['char_start']:r['char_end']] != r['text'] for r in chunks)
    id_checks = verify_document_ids(extracted, list(by_id), prov, [r['doc_id'] for r in chunks])
    expected_texts = {r['doc_id']: r['doc_text'] for r in _document_level_rows(extracted)
                      if r['status'] == 'ok' and (r['n_tokens_bge_m3'] or 0) >= MIN_BODY_TOKENS and r['body']}
    handoff_texts_match = all(by_id.get(doc_id) == text for doc_id, text in expected_texts.items())
    hashes = _verification_hashes(roots)
    result = {'batch': BATCH, 'crawl_coverage': crawl_gate, 'docs': len(docs), 'chunks': len(chunks), 'provenance': len(prov),
              **id_checks, 'handoff_texts_match_s15_doc_layout': handoff_texts_match,
              'verbatim_bad': bad, 'artifacts_sha256': hashes, 'complete': True, 'shared_index_written': False}
    if (bad or not result['handoff_doc_ids_unique'] or not result['extracted_doc_ids_unique']
            or not result['provenance_ids_equal'] or not result['handoff_doc_ids_match_eligible']
            or not result['chunk_doc_ids_present'] or not handoff_texts_match):
        raise ValueError(f'S27 S15 verification failed: {result}')
    atomic_json(roots, roots.reports / 'verify.json', result)
    return result


def verify_document_ids(extracted_rows: list[dict], handoff_doc_ids: list[int], provenance_ids: list[int],
                        chunk_doc_ids: list[int]) -> dict:
    extracted_ids = [int(doc_id) for row in extracted_rows for doc_id in (row.get('doc_ids') or [])]
    eligible_ids = sorted(int(doc_id) for row in extracted_rows
                          if row.get('status') == 'ok' and (row.get('n_tokens_bge_m3') or 0) >= MIN_BODY_TOKENS
                          and row.get('body')
                          for doc_id in (row.get('doc_ids') or []))
    handoff_set = set(handoff_doc_ids)
    return {'extracted_doc_ids_unique': len(extracted_ids) == len(set(extracted_ids)),
            'handoff_doc_ids_unique': len(handoff_doc_ids) == len(handoff_set),
            'provenance_ids_equal': sorted(extracted_ids) == sorted(int(x) for x in provenance_ids),
            'handoff_doc_ids_match_eligible': sorted(handoff_doc_ids) == eligible_ids,
            'chunk_doc_ids_present': all(doc_id in handoff_set for doc_id in chunk_doc_ids)}


def _document_level_rows(extracted_rows: list[dict]) -> list[dict]:
    from r2ai.index.chunk import doc_layout

    rows = []
    for row in extracted_rows:
        if row.get('status') != 'ok' or not row.get('body'):
            continue
        doc_text = doc_layout(row)[0]
        for doc_id in (row.get('doc_ids') or []):
            rows.append({'doc_id': int(doc_id), 'doc_text': doc_text, 'status': row['status'],
                         'n_tokens_bge_m3': row.get('n_tokens_bge_m3') or 0, 'body': row['body']})
    rows.sort(key=lambda row: row['doc_id'])
    if len({row['doc_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate S27 document id in doc-level embedding inputs')
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _embedding_cache_hashes(doc_ids: np.ndarray, texts: list[str]) -> tuple[str, str]:
    if len(doc_ids) != len(texts):
        raise ValueError('embedding cache input IDs/texts length mismatch')
    input_hash = hashlib.sha256()
    for doc_id, text in zip(doc_ids, texts):
        encoded = text.encode('utf-8')
        input_hash.update(int(doc_id).to_bytes(8, 'big', signed=True))
        input_hash.update(len(encoded).to_bytes(8, 'big'))
        input_hash.update(encoded)
    config = {'encoder': 'r2ai.index.bge_m3.M3Encoder', 'model': 'BAAI/bge-m3', 'max_len': 512,
              'input_prefix_chars': 512 * 8, 'dense': 'L2-normalized-CLS', 'dimension': 1024,
              'dtype': 'float16'}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return input_hash.hexdigest(), config_hash


def _embedding_cache_matches(cache, doc_ids: np.ndarray, input_hash: str, config_hash: str) -> bool:
    try:
        dense = cache['dense']
        return (np.array_equal(cache['doc_id'], doc_ids)
                and dense.shape == (len(doc_ids), 1024)
                and dense.dtype == np.dtype(np.float16)
                and np.isfinite(dense).all()
                and str(cache['input_sha256'].item()) == input_hash
                and str(cache['config_sha256'].item()) == config_hash)
    except (KeyError, ValueError, AttributeError):
        return False


def _verification_hashes(roots: Roots) -> dict:
    paths = {'extracted_docs': roots.docs / 'docs.parquet', 'provenance': roots.docs / 'provenance.parquet',
             'chunk_docs': roots.chunks / 'docs.parquet', 'chunks_t256': roots.chunks / 'chunks_t256.parquet'}
    return {name: _sha256_file(roots.guard(path)) for name, path in paths.items()}


def _s26_completion_marker_ok(log_text: str) -> bool:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == f'done {S26_DOC_COUNT}'


def verification_is_fresh(report: dict, current_hashes: dict) -> bool:
    return report.get('complete') is True and report.get('artifacts_sha256') == current_hashes


def s26_embedding_gate() -> dict:
    import zipfile
    from numpy.lib import format as npformat

    evidence = []
    for path in S26_FULL_VECTOR_CANDIDATES:
        if not path.exists():
            evidence.append({'path': str(path), 'complete': False, 'reason': 'full_vector_missing'})
            continue
        log_path = path.parent / 'embed.log'
        try:
            log_text = log_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            log_text = ''
        if not _s26_completion_marker_ok(log_text) or not log_path.exists() or path.stat().st_mtime > log_path.stat().st_mtime:
            evidence.append({'path': str(path), 'complete': False, 'reason': 's26_done_marker_missing_or_stale',
                             'completion_log': str(log_path)})
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                if not {'doc_id.npy', 'dense.npy'} <= names:
                    evidence.append({'path': str(path), 'complete': False, 'reason': 'npz_members_missing'})
                    continue
                with zf.open('dense.npy') as dense_file:
                    version = npformat.read_magic(dense_file)
                    if version == (1, 0):
                        shape, _, dtype = npformat.read_array_header_1_0(dense_file)
                    elif version == (2, 0):
                        shape, _, dtype = npformat.read_array_header_2_0(dense_file)
                    else:
                        raise ValueError(f'unsupported dense.npy header version: {version}')
                with zf.open('doc_id.npy') as id_file:
                    version = npformat.read_magic(id_file)
                    if version == (1, 0):
                        id_shape, _, id_dtype = npformat.read_array_header_1_0(id_file)
                    elif version == (2, 0):
                        id_shape, _, id_dtype = npformat.read_array_header_2_0(id_file)
                    else:
                        raise ValueError(f'unsupported doc_id.npy header version: {version}')
                    ids = np.frombuffer(id_file.read(), dtype=id_dtype)
            if (shape == (S26_DOC_COUNT, 1024) and np.dtype(dtype) == np.dtype(np.float16)
                    and id_shape == (S26_DOC_COUNT,) and np.dtype(id_dtype) == np.dtype(np.int64)
                    and len(np.unique(ids)) == S26_DOC_COUNT):
                return {'complete': True, 'path': str(path), 'docs': len(ids), 'dense_shape': list(shape),
                        'completion_log': str(log_path), 'completion_marker': f'done {S26_DOC_COUNT}'}
            evidence.append({'path': str(path), 'complete': False, 'reason': 'array_shape_dtype_or_doc_id_mismatch'})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            evidence.append({'path': str(path), 'complete': False, 'reason': 'unreadable_or_incomplete_npz'})
            continue
    return {'complete': False, 'required_docs': S26_DOC_COUNT, 'checked': evidence}


def embed_batch(roots: Roots) -> dict:
    roots.preflight()
    crawl_gate = require_crawl_complete(roots)
    verify_path = roots.guard(roots.reports / 'verify.json')
    if not verify_path.exists():
        raise FileNotFoundError('S27 verification output is required before embedding')
    verification = json.loads(verify_path.read_text(encoding='utf-8'))
    if not verification_is_fresh(verification, _verification_hashes(roots)):
        raise RuntimeError('S27 verification is absent, incomplete, or stale for current batch artifacts')
    gate = s26_embedding_gate()
    if not gate['complete']:
        return {'batch': BATCH, 'status': 'waiting_for_s26', 's26_gate': gate}
    from r2ai.index.bge_m3 import M3Encoder
    extracted_file = roots.guard(roots.docs / 'docs.parquet')
    extracted = pq.read_table(extracted_file, columns=['doc_ids', 'status', 'n_tokens_bge_m3', 'body',
                                                       'title', 'question', 'answer', 'paragraphs']).to_pylist()
    doc_rows = _document_level_rows(extracted)
    ids = np.asarray([row['doc_id'] for row in doc_rows], dtype=np.int64)
    texts = [row['doc_text'][:512 * 8] for row in doc_rows]
    parts_dir = roots.guard(roots.vectors / 'blocks')
    parts_dir.mkdir(parents=True, exist_ok=True)
    encoder = M3Encoder(max_len=512)
    t0 = time.time()
    for block_start in range(0, len(ids), 5000):
        block_ids = ids[block_start:block_start + 5000]
        block_texts = texts[block_start:block_start + len(block_ids)]
        input_hash, config_hash = _embedding_cache_hashes(block_ids, block_texts)
        part = roots.guard(parts_dir / f'{block_start // 5000:04d}.npz')
        if part.exists():
            try:
                with np.load(part, allow_pickle=False) as prev:
                    if _embedding_cache_matches(prev, block_ids, input_hash, config_hash):
                        continue
            except (OSError, ValueError, EOFError):
                pass
        order = np.argsort([len(x) for x in block_texts], kind='stable')
        dense = np.zeros((len(block_ids), 1024), dtype=np.float16)
        i = 0
        while i < len(order):
            bs = 64 if len(block_texts[order[i]]) < 600 else 32 if len(block_texts[order[i]]) < 1500 else 16
            sel = order[i:i + bs]
            dense[sel] = encoder.encode_batch([block_texts[j] for j in sel])[0]
            i += bs
        tmp = roots.guard(part.with_name(part.stem + '.partial.npz'))
        np.savez(tmp, doc_id=block_ids, dense=dense, input_sha256=np.asarray(input_hash),
                 config_sha256=np.asarray(config_hash))
        os.replace(tmp, part)
        print(json.dumps({'embedded': min(block_start + len(block_ids), len(ids)), 'total': len(ids),
                          'seconds': round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
    expected_part_count = (len(ids) + 4999) // 5000
    expected_parts = [roots.guard(parts_dir / f'{index:04d}.npz') for index in range(expected_part_count)]
    if any(not path.is_file() for path in expected_parts):
        raise ValueError('S27 embedding is missing one or more expected blocks')
    parts = [np.load(p, allow_pickle=False) for p in expected_parts]
    out_ids = np.concatenate([p['doc_id'] for p in parts]) if parts else np.array([], dtype=np.int64)
    out_dense = np.concatenate([p['dense'] for p in parts]) if parts else np.zeros((0, 1024), np.float16)
    if not np.array_equal(out_ids, ids) or out_dense.shape != (len(ids), 1024) or not np.isfinite(out_dense).all():
        raise ValueError('S27 embedding block integrity check failed')
    out = roots.guard(roots.vectors / 'dense_doc_level.npz')
    tmp = roots.guard(roots.vectors / 'dense_doc_level.partial.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, doc_id=out_ids, dense=out_dense)
    os.replace(tmp, out)
    summary = {'batch': BATCH, 'status': 'complete', 'crawl_coverage': crawl_gate, 'docs': len(out_ids), 'dense_shape': list(out_dense.shape),
               'dtype': str(out_dense.dtype), 'encoder': 'r2ai.index.bge_m3.M3Encoder', 'max_len': 512,
               'doc_text_prefix_chars': 4096, 's26_gate': gate, 'vector_file': str(out)}
    atomic_json(roots, roots.reports / 'embed.json', summary)
    return summary


def embed_handoff_lot(roots: Roots, lot_id: str) -> dict:
    """Create an isolated BGE-M3 doc-level vector handoff for one verified S15 lot."""
    roots.preflight()
    gate = s26_embedding_gate()
    if not gate.get('complete'):
        return {'batch': BATCH, 'lot_id': lot_id, 'status': 'waiting_for_s26', 's26_gate': gate,
                'shared_index_written': False}
    if not re.fullmatch(r'lot-[0-9]{4,}', str(lot_id)):
        raise ValueError(f'invalid S27 handoff lot ID: {lot_id!r}')
    lot_root = roots.guard(roots.root / 'handoff' / BATCH / 'lots' / lot_id)
    lot_manifest_path = roots.guard(lot_root / 'lot_manifest.json')
    docs_path = roots.guard(lot_root / 'docs' / 'docs.parquet')
    if not lot_manifest_path.is_file() or not docs_path.is_file():
        raise FileNotFoundError(f'S15 lot documents and manifest are required: {lot_id}')
    lot_manifest = json.loads(lot_manifest_path.read_text(encoding='utf-8'))
    if lot_manifest.get('batch') != BATCH or lot_manifest.get('contract') != 'S15' or lot_manifest.get('lot_id') != lot_id:
        raise ValueError(f'invalid S15 lot manifest: {lot_manifest_path}')
    rows = pq.read_table(docs_path).to_pylist()
    doc_rows = _document_level_rows(rows)
    ids = np.asarray([row['doc_id'] for row in doc_rows], dtype=np.int64)
    expected_ids = sorted(int(doc_id) for doc_id in lot_manifest.get('doc_ids', []))
    if not len(ids) or ids.tolist() != expected_ids or len(ids) != len(set(ids.tolist())):
        raise ValueError(f'S15 lot doc_text IDs differ from its manifest: {lot_id}')
    texts = [row['doc_text'][:512 * 8] for row in doc_rows]
    lot_sha = _sha256_file(lot_manifest_path)
    docs_sha = _sha256_file(docs_path)
    full_input_sha, config_sha = _embedding_cache_hashes(ids, texts)
    out_dir = roots.guard(roots.vectors / 'handoff' / lot_id)
    blocks_dir = roots.guard(out_dir / 'blocks')
    blocks_dir.mkdir(parents=True, exist_ok=True)
    dense_parts = []
    encoder = None
    started = time.time()
    for block_start in range(0, len(ids), 5000):
        block_ids = ids[block_start:block_start + 5000]
        block_texts = texts[block_start:block_start + len(block_ids)]
        input_sha, block_config_sha = _embedding_cache_hashes(block_ids, block_texts)
        part = roots.guard(blocks_dir / f'{block_start // 5000:04d}.npz')
        dense = None
        if part.is_file():
            try:
                with np.load(part, allow_pickle=False) as cached:
                    if _embedding_cache_matches(cached, block_ids, input_sha, block_config_sha):
                        dense = cached['dense'].copy()
            except (OSError, ValueError, EOFError):
                dense = None
        if dense is None:
            if encoder is None:
                from r2ai.index.bge_m3 import M3Encoder
                encoder = M3Encoder(max_len=512)
            order = np.argsort([len(text) for text in block_texts], kind='stable')
            dense = np.zeros((len(block_ids), 1024), dtype=np.float16)
            i = 0
            while i < len(order):
                size = 64 if len(block_texts[order[i]]) < 600 else 32 if len(block_texts[order[i]]) < 1500 else 16
                selected = order[i:i + size]
                dense[selected] = encoder.encode_batch([block_texts[j] for j in selected])[0]
                i += size
            tmp_part = roots.guard(part.with_name(part.stem + '.partial.npz'))
            np.savez(tmp_part, doc_id=block_ids, dense=dense, input_sha256=np.asarray(input_sha),
                     config_sha256=np.asarray(block_config_sha))
            os.replace(tmp_part, part)
        if dense.shape != (len(block_ids), 1024) or dense.dtype != np.dtype(np.float16) or not np.isfinite(dense).all():
            raise ValueError(f'S27 vector block failed shape/dtype/finiteness checks: {part}')
        dense_parts.append(dense)

    all_dense = np.concatenate(dense_parts, axis=0)
    if all_dense.shape != (len(ids), 1024) or not np.isfinite(all_dense).all():
        raise ValueError(f'S27 lot dense vector output is invalid: {lot_id}')
    vector_path = roots.guard(out_dir / 'dense_doc_level.npz')
    tmp_vector = roots.guard(out_dir / 'dense_doc_level.partial.npz')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(tmp_vector, doc_id=ids, dense=all_dense)
    os.replace(tmp_vector, vector_path)
    vector_manifest = {
        'batch': BATCH, 'contract': 'S26-P-A4-vector-handoff', 'lot_id': lot_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'docs': len(ids), 'doc_ids': ids.tolist(), 'input': 'doc_text (S15 doc_layout)', 'input_prefix_chars': 4096,
        'encoder': 'r2ai.index.bge_m3.M3Encoder', 'model': 'BAAI/bge-m3', 'max_len': 512,
        'dense': 'L2-normalized-CLS', 'dense_shape': list(all_dense.shape), 'dtype': str(all_dense.dtype),
        'input_sha256': full_input_sha, 'config_sha256': config_sha,
        's15_lot_manifest_sha256': lot_sha, 's15_docs_sha256': docs_sha,
        's26_gate': gate, 'vector_file': str(vector_path), 'vector_sha256': _sha256_file(vector_path),
        'elapsed_seconds': round(time.time() - started, 2), 'shared_index_written': False,
    }
    vector_manifest_path = roots.guard(out_dir / 'vector_manifest.json')
    atomic_json(roots, vector_manifest_path, vector_manifest)

    handoff_root = roots.guard(roots.vectors / 'handoff')
    lot_manifests = []
    seen_ids = set()
    for path in sorted(handoff_root.glob('lot-*/vector_manifest.json')):
        saved = json.loads(roots.guard(path).read_text(encoding='utf-8'))
        saved_ids = [int(doc_id) for doc_id in saved.get('doc_ids', [])]
        if seen_ids.intersection(saved_ids):
            raise ValueError('duplicate doc_id across S27 vector handoff lots')
        seen_ids.update(saved_ids)
        lot_manifests.append({'lot_id': saved['lot_id'], 'docs': len(saved_ids),
                              'vector_file': saved['vector_file'], 'vector_sha256': saved['vector_sha256'],
                              'vector_manifest': str(path)})
    handoff_manifest = {'batch': BATCH, 'contract': 'S26-P-A4-vector-handoff',
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                        'lots': lot_manifests, 'docs': len(seen_ids), 's26_gate': gate,
                        'shared_index_written': False}
    atomic_json(roots, roots.guard(handoff_root / 'manifest.json'), handoff_manifest)
    return {'batch': BATCH, 'lot_id': lot_id, 'status': 'complete', 'docs': len(ids),
            'dense_shape': list(all_dense.shape), 'dtype': str(all_dense.dtype),
            'vector_file': str(vector_path), 'vector_sha256': vector_manifest['vector_sha256'],
            'vector_manifest': str(vector_manifest_path), 'elapsed_seconds': vector_manifest['elapsed_seconds'],
            'shared_index_written': False}
