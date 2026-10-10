"""Snapshot choice and coverage summary (pure functions)."""
from __future__ import annotations

from collections import Counter

HTML = 'text/html'


def ts(x) -> str:
    """fetch_time as 'YYYY-MM-DDTHH:MM:SS' whether it is a datetime (parquet timestamp) or an ISO string."""
    return (x.isoformat() if hasattr(x, 'isoformat') else str(x))[:19]


def usable(rec: dict) -> bool:
    """HTTP 200, HTML, not a revisit record (revisits have mime warc/revisit)."""
    return str(rec['fetch_status']) == '200' and rec['content_mime_type'] == HTML


def choose(records: list[dict], corpus_date: str) -> dict | None:
    """records: dicts with crawl, fetch_time (ISO 'YYYY-MM-DDTHH:MM:SS...'), warc_record_length, fetch_status, content_mime_type.
    Newest usable record not after corpus_date; ties -> larger warc_record_length. Nothing usable -> None.
    If every usable record is after the corpus date: the oldest of them, flagged after_corpus."""
    cut = corpus_date + 'T23:59:59'
    ok = [r for r in records if usable(r)]
    if not ok:
        return None
    before = [r for r in ok if ts(r['fetch_time']) <= cut]
    if before:
        return max(before, key=lambda r: (ts(r['fetch_time']), r['warc_record_length'])) | {'after_corpus': False}
    return min(ok, key=lambda r: (ts(r['fetch_time']), -r['warc_record_length'])) | {'after_corpus': True}


def summarize(chosen: dict, prefix_of: dict) -> dict:
    """chosen: url_id -> chosen record or None. Counts overall / per prefix / per crawl-id and the exact WARC byte sum."""
    hit = {i: r for i, r in chosen.items() if r}
    return {'urls': len(chosen), 'with_200_html': len(hit),
            'by_prefix': {p: {'urls': sum(1 for i in chosen if prefix_of[i] == p), 'with_200_html': sum(1 for i in hit if prefix_of[i] == p)}
                          for p in sorted(set(prefix_of.values()))},
            'chosen_by_crawl': dict(sorted(Counter(r['crawl'] for r in hit.values()).items())),
            'after_corpus_chosen': sum(1 for r in hit.values() if r['after_corpus']),
            'warc_bytes_chosen': int(sum(r['warc_record_length'] for r in hit.values()))}
