"""Pure S20 recovery helpers and the S15 CC provenance contract."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import pyarrow as pa


PROV_SCHEMA = pa.schema([
    ('doc_id', pa.int64()), ('url', pa.string()), ('source', pa.string()), ('crawl', pa.string()),
    ('warc_filename', pa.string()), ('warc_offset', pa.int64()), ('warc_length', pa.int64()),
    ('warc_date', pa.string()), ('capture_time_utc', pa.string()), ('warc_record_id', pa.string()),
    ('warc_payload_digest', pa.string()), ('record_sha256', pa.string()), ('body_sha256', pa.string()),
    ('content_type', pa.string()), ('truncated', pa.string()), ('raw_shard', pa.string()),
    ('extractor', pa.string()), ('extractor_name', pa.string()), ('trafilatura', pa.string()),
    ('status', pa.string()), ('n_chars', pa.int32()), ('n_tokens_bge_m3', pa.int32()), ('text_sha256', pa.string()),
])
CC_MANIFEST_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches(batch TEXT PRIMARY KEY, kind TEXT NOT NULL, snapshot_at REAL NOT NULL,
    created_at REAL NOT NULL, domains TEXT NOT NULL, source_counts TEXT NOT NULL, skipped TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, n_urls INTEGER NOT NULL,
    status TEXT NOT NULL, n_records INTEGER, n_docs INTEGER, docs_file TEXT, extractor_version TEXT, done_at REAL);
CREATE TABLE IF NOT EXISTS batch_urls(url_norm TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    shard TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs(url_norm TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    doc_id INTEGER NOT NULL, status TEXT NOT NULL, docs_file TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS segments(segment_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
    batch TEXT NOT NULL, domain TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, rows INTEGER,
    docs_count INTEGER, global_start INTEGER, chunks_sha256 TEXT, extractor_version TEXT, chunker TEXT,
    model TEXT, vectors TEXT, note TEXT, stats TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS embed_blocks(segment TEXT NOT NULL, block INTEGER NOT NULL, rows INTEGER NOT NULL,
    dense_sha256 TEXT NOT NULL, sparse_sha256 TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(segment, block));
"""
CC_TARGET = 256
CC_MIN_BODY_TOKENS = 50
CC_MODEL = {
    'model': 'BAAI/bge-m3', 'max_len': 512, 'fp16': True, 'batch_size': 16, 'block': 1000,
    'dense': 'CLS, L2-normalised, fp16', 'sparse': 'relu(sparse_linear) max per token id',
}


def cleaned_length(text: str | None) -> int:
    """S20 threshold: collapse all Unicode whitespace, then count code points."""
    return len(' '.join((text or '').split()))


def good_text(text: str | None) -> bool:
    return cleaned_length(text) >= 200


def primary_doc_id(row: dict) -> int | None:
    ids = row.get('doc_ids') or []
    return int(ids[0]) if ids else None


def is_s20_failure(row: dict, host: str, v2_ids: set[int]) -> bool:
    """Select only the primary v2 ID; 120ask uses the S20 extract_ok proxy."""
    doc_id = primary_doc_id(row)
    if doc_id is None or doc_id not in v2_ids:
        return False
    if host == '120ask.com':
        return row.get('status') != 'ok'
    return not good_text(row.get('body'))


def select_sample(host: str, rows: list[dict], limit: int = 20) -> list[dict]:
    """Order-independent deterministic sample, deduplicated on primary doc ID."""
    unique = {}
    for row in rows:
        doc_id = row.get('doc_id', primary_doc_id(row))
        if doc_id is None:
            continue
        unique.setdefault(int(doc_id), row)
    ordered = sorted(unique.values(), key=lambda row: (
        hashlib.sha256(f'{host}\0{int(row.get("doc_id", primary_doc_id(row)))}'.encode()).digest(),
        int(row.get('doc_id', primary_doc_id(row))),
    ))
    return ordered[:limit]


def batch_label(host: str) -> str:
    return f'cc-s22-{host}'


def provenance_row(*, doc_id: int, url: str, raw_shard: str, status: str, body: str,
                    extractor: str, extractor_name: str, fetched_at: float,
                    n_tokens: int, trafilatura: str = '') -> dict:
    return {
        'doc_id': int(doc_id), 'url': url, 'source': 'reextract-raw', 'crawl': None,
        'warc_filename': None, 'warc_offset': None, 'warc_length': None, 'warc_date': None,
        'capture_time_utc': None, 'warc_record_id': None, 'warc_payload_digest': None,
        'record_sha256': None, 'body_sha256': None, 'content_type': None, 'truncated': None,
        'raw_shard': raw_shard, 'extractor': extractor, 'extractor_name': extractor_name,
        'trafilatura': trafilatura, 'status': status, 'n_chars': len(body),
        'n_tokens_bge_m3': int(n_tokens), 'text_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
    }


def encoding_suspect(text: str) -> bool:
    """Conservative damage flags; valid language text is not classified by charset label alone."""
    if '\ufffd' in text:
        return True
    if any(marker in text for marker in ('锟斤拷', 'Ã¤', 'Ã©', 'â€™', 'ï¿½')):
        return True
    bad = sum(unicodedata.category(ch) == 'Co' for ch in text)
    return bool(text) and bad / len(text) > 0.005


def strip_null_bytes(text: str) -> str:
    """Remove NUL characters that some stored Youlai HTML contains inside visible Chinese text."""
    return text.replace('\x00', '')


def looks_blocked(text: str, status: int | None = None) -> bool:
    if status is not None and not 200 <= int(status) < 300:
        return True
    lower = text[:200_000].lower()
    return any(token in lower for token in (
        'cf-chl-', '/cdn-cgi/challenge-platform', 'just a moment...', 'attention required! | cloudflare',
        'captcha', '<title>access denied', '<title>403 forbidden', '<h1>403 forbidden',
        'request blocked', 'verify you are human', 'too many requests', '<title>service unavailable',
    ))
