"""Shared storage and verification; no search-engine requests."""
from __future__ import annotations

from r2ai.paths import DATA_DIR, OUT_DIR, RAW_DATA_DIR, ROOT

import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

OUT = OUT_DIR
STRATA = ('short', 'medium', 'long')
FOUND = ('verbatim', 'partial', 'paraphrase', 'none')
PAGE_TYPES = ('hỏi đáp', 'bài viết', 'khác')
MODEL = 'BAAI/bge-m3'


def data_path(name):
    for p in (DATA_DIR / name, RAW_DATA_DIR / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def normalize_url(url, loose=False):
    u = urlsplit(url if '://' in url else 'https://' + url)
    host = (u.hostname or '').lower().removeprefix('www.')
    port = ':' + str(u.port) if u.port is not None else ''
    return host + port + u.path.rstrip('/') + (('?' + u.query) if u.query and not loose else '')


def domain(url):
    return (urlsplit(url if '://' in url else 'https://' + url).hostname or '').lower().removeprefix('www.')


def stratum(n):
    return 'short' if n < 15 else 'medium' if n <= 40 else 'long'


def normalized_with_offsets(text):
    """NFC/lowercase, punctuation -> spaces; map every char to original span."""
    chars, starts, ends = [], [], []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and unicodedata.combining(text[j]):
            j += 1
        chunk = unicodedata.normalize('NFC', text[i:j]).lower()
        for c in chunk:
            c = ' ' if c.isspace() or unicodedata.category(c).startswith('P') else c
            if c == ' ' and (not chars or chars[-1] == ' '):
                continue
            chars.append(c)
            starts.append(i)
            ends.append(j)
        i = j
    if chars and chars[-1] == ' ':
        chars.pop(); starts.pop(); ends.pop()
    return ''.join(chars), starts, ends


def normalize_text(text):
    return normalized_with_offsets(text)[0]


def match_excerpt(query, text):
    q = normalize_text(query)
    normalized, starts, ends = normalized_with_offsets(text)
    pos = normalized.find(q) if q else -1
    if pos < 0:
        return dict(verbatim_hit=False, match_start=None, match_end=None, after_match='')
    start, end = starts[pos], ends[pos + len(q) - 1]
    return dict(verbatim_hit=True, match_start=start, match_end=end, after_match=text[end:end + 300])


def extracted_page_text(body):
    """Preserve omitted title and explicit FAQ question beside trafilatura answer."""
    from r2ai.probe.fetcher import extract
    main, title, _ = extract(body)
    if main:
        from lxml import html
        from trafilatura.utils import decode_file
        try:
            tree = html.fromstring(decode_file(body))
            questions = tree.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' faq-item ')]/*[contains(concat(' ', normalize-space(@class), ' '), ' faq-item-desc ')]")
            if questions:
                question = ' '.join(questions[0].text_content().split())
                if question and normalize_text(question) not in normalize_text(main):
                    main = question + '\n' + main
        except (ValueError, TypeError):
            pass
    if main and title and normalize_text(title) not in normalize_text(main):
        return title + '\n' + main, len(title) + 1
    return main, 0


def page_match(query, text, body_start):
    body = match_excerpt(query, text[body_start:])
    if body['verbatim_hit']:
        body['match_start'] += body_start
        body['match_end'] += body_start
        body.update(match_source='body', body_verbatim_hit=True)
        return body
    result = match_excerpt(query, text)
    result.update(match_source='title' if result['verbatim_hit'] else '', body_verbatim_hit=False)
    return result


def lcs_length(query_tokens, text_tokens):
    """Exact bit-parallel LCS, O(len(text)*ceil(len(query)/word_size))."""
    masks = {}
    for i, token in enumerate(query_tokens):
        masks[token] = masks.get(token, 0) | (1 << i)
    state = 0
    for token in text_tokens:
        x = state | masks.get(token, 0)
        state = x & ~(x - ((state << 1) | 1))
    return state.bit_count()


def load_tokenizer(path=None, download=False):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path or MODEL, local_files_only=not download, use_fast=True)
    except Exception as e:
        raise RuntimeError('Chưa có tokenizer BAAI/bge-m3. Chạy app với --download-tokenizer hoặc --tokenizer PATH. Không dùng tokenizer thay thế.') from e


class CorpusIndex:
    """Disk-backed exact/loose index, streamed build and atomic cache refresh."""
    VERSION = 1

    def __init__(self, source, cache):
        self.source, self.cache = Path(source).resolve(), Path(cache)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        stat = self.source.stat()
        signature = json.dumps([str(self.source), stat.st_size, stat.st_mtime_ns, self.VERSION])
        valid = False
        if self.cache.exists():
            with closing(sqlite3.connect(self.cache)) as db:
                try:
                    valid = db.execute('SELECT value FROM meta WHERE key=?', ('signature',)).fetchone() == (signature,)
                except sqlite3.DatabaseError:
                    pass
        if not valid:
            self._build(signature)

    def _build(self, signature):
        import pyarrow.parquet as pq
        temp = self.cache.with_suffix('.building.sqlite')
        if temp.exists():
            temp.unlink()
        counts = Counter()
        try:
            with closing(sqlite3.connect(temp)) as db, db:
                db.execute('PRAGMA journal_mode=OFF')
                db.execute('PRAGMA cache_size=-32768')
                db.execute('PRAGMA temp_store=FILE')
                db.execute('CREATE TABLE urls(norm TEXT, loose TEXT, doc_id TEXT)')
                db.execute('CREATE TABLE domains(domain TEXT PRIMARY KEY, n INTEGER)')
                db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)')
                total = 0
                for batch in pq.ParquetFile(self.source).iter_batches(batch_size=50000, columns=['id', 'url']):
                    records = []
                    for r in batch.to_pylist():
                        if not r['url']:
                            continue
                        records.append((normalize_url(r['url']), normalize_url(r['url'], True), str(r['id'])))
                        counts[domain(r['url'])] += 1
                    db.executemany('INSERT INTO urls VALUES(?,?,?)', records)
                    db.commit()
                    total += len(records)
                    if total % 500000 < 50000:
                        print(f'Corpus index: {total:,} rows', flush=True)
                db.execute('CREATE INDEX exact_idx ON urls(norm)')
                db.execute('CREATE INDEX loose_idx ON urls(loose)')
                db.executemany('INSERT INTO domains VALUES(?,?)', counts.items())
                db.execute('INSERT INTO meta VALUES(?,?)', ('signature', signature))
            os.replace(temp, self.cache)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def lookup(self, url):
        with closing(sqlite3.connect(self.cache)) as db:
            ids = sorted({r[0] for r in db.execute('SELECT doc_id FROM urls WHERE norm=?', (normalize_url(url),))})
            suggestions = [] if ids else sorted({r[0] for r in db.execute('SELECT doc_id FROM urls WHERE loose=?', (normalize_url(url, True),))})
        return dict(in_corpus=bool(ids), doc_ids=ids, suggested_doc_ids=suggestions)

    def domain_counts(self):
        with closing(sqlite3.connect(self.cache)) as db:
            return dict(db.execute('SELECT domain,n FROM domains'))


FIELDS = ['query_id', 'query', 'word_count', 'stratum', 'found', 'page_type', 'url_found', 'url_page_type', 'notes', 'url', 'url_norm', 'domain', 'in_corpus', 'doc_ids', 'suggested_doc_ids', 'final_url', 'final_in_corpus', 'final_doc_ids', 'verification_status', 'fetch_state', 'http_status', 'error', 'text_chars', 'verbatim_hit', 'lcs_ratio', 'lcs_tokens', 'query_tokens', 'tokenizer', 'match_source', 'body_verbatim_hit', 'match_kind', 'match_start', 'match_end', 'after_match', 'html_path', 'text_path', 'saved_at']


class ResultsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.rows = []
        if self.path.exists():
            with self.path.open(encoding='utf-8-sig', newline='') as f:
                self.rows = list(csv.DictReader(f))
            for row in self.rows:
                for key in ('doc_ids', 'suggested_doc_ids', 'final_doc_ids'):
                    if row.get(key):
                        row[key] = json.loads(row[key])

    def for_query(self, query_id):
        with self.lock:
            return [dict(r) for r in self.rows if str(r['query_id']) == str(query_id)]

    def replace_query(self, query_id, rows):
        with self.lock:
            updated = [r for r in self.rows if str(r['query_id']) != str(query_id)] + rows
            self._write(updated)
            self.rows = updated

    def _write(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.csv.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)


SEARCH_HOSTS = ('google.com', 'google.com.vn', 'bing.com', 'coccoc.com', 'search.coccoc.com', 'googleusercontent.com')


def validate_url(url):
    u = urlsplit(url)
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password:
        raise ValueError('URL phải là http(s), có hostname, không có thông tin đăng nhập.')
    if '%' in u.hostname:
        raise ValueError('Hostname không được chứa percent-encoding.')
    host = u.hostname.encode('idna').decode('ascii').lower().rstrip('.')
    if any(host == s or host.endswith('.' + s) for s in SEARCH_HOSTS) or re.search(r'(^|\.)google\.', host):
        raise ValueError('Không fetch Google/Bing/Cốc Cốc, kể cả redirect.')
    _ = u.port
    return url


def make_fetcher():
    from r2ai.probe.fetcher import Fetcher

    class PastedURLFetcher(Fetcher):
        def request(self, url, **kwargs):
            validate_url(url)  # also protects robots and every redirect
            return super().request(url, **kwargs)

    return PastedURLFetcher(timeout=15)


def verify_url(query, url, index, fetcher, tokenizer, pages):
    from r2ai.probe.fetcher import extract
    result = dict(verification_status='error', tokenizer=MODEL, error='', verbatim_hit=None,
                  lcs_ratio=None, lcs_tokens=None, query_tokens=None, match_kind='none',
                  match_start=None, match_end=None, after_match='', match_source='', body_verbatim_hit=None)
    pages = Path(pages)
    pages.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((str(query['id']) + '\0' + url).encode()).hexdigest()[:24]
    html_path, text_path = pages / (key + '.html'), pages / (key + '.txt')
    try:
        validate_url(url)
        meta, body = fetcher.fetch(url)
        html_path.write_bytes(body)
        text, body_start = extracted_page_text(body)  # trafilatura body + omitted title
        text_path.write_text(text, encoding='utf-8')
        result.update(html_path=str(html_path.resolve()), text_path=str(text_path.resolve()), final_url=meta.get('final_url', url), fetch_state=meta.get('state'), http_status=meta.get('http_status'), text_chars=len(text), error=meta.get('error', ''))
        final_lookup = index.lookup(result['final_url'])
        result.update(final_in_corpus=final_lookup['in_corpus'], final_doc_ids=final_lookup['doc_ids'])
        if meta.get('error') or meta.get('http_status') != 200 or meta.get('state') not in ('ok', 'thin') or not text:
            result['error'] = result['error'] or 'HTTP/extraction non-success; metrics unavailable'
            return result
        result.update(page_match(query['query'], text, body_start))
        q = tokenizer(normalize_text(query['query']), add_special_tokens=False, truncation=False)['input_ids']
        encoded = tokenizer(normalize_text(text), add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        t = encoded['input_ids']
        length = lcs_length(q, t)
        result.update(lcs_ratio=length / len(q) if q else 0.0, lcs_tokens=length, query_tokens=len(q), verification_status='ok', match_kind='verbatim' if result['verbatim_hit'] else 'none')
        # If no full match: report longest contiguous token match, explicitly partial.
        if not result['verbatim_hit']:
            previous, best, best_end = {}, 0, 0
            positions = {}
            for i, token in enumerate(q):
                positions.setdefault(token, []).append(i)
            for j, token in enumerate(t):
                current = {i: previous.get(i - 1, 0) + 1 for i in positions.get(token, [])}
                for size in current.values():
                    if size > best:
                        best, best_end = size, j + 1
                previous = current
            if best:
                normalized, starts, ends = normalized_with_offsets(text)
                offsets = encoded['offset_mapping']
                a, b = offsets[best_end - best][0], offsets[best_end - 1][1]
                if b > a and b <= len(ends):
                    start, end = starts[a], ends[b - 1]
                    result.update(match_kind='partial_tokens', match_start=start, match_end=end, after_match=text[end:end + 300])
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return result


def timestamp():
    return datetime.now(timezone.utc).isoformat()
