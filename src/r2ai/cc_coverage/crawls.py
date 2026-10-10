"""Crawl list from collinfo.json: the 12 months up to the corpus crawl, plus every later crawl (flagged after_corpus)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

ANCHOR = 'CC-MAIN-2026-39'
CORPUS_DATE = '2026-09-29'
OLD_SIX = ('CC-MAIN-2026-17', 'CC-MAIN-2026-21', 'CC-MAIN-2026-25', 'CC-MAIN-2026-30', 'CC-MAIN-2026-34', 'CC-MAIN-2026-39')


def crawl_key(crawl_id: str):
    m = re.fullmatch(r'CC-MAIN-(\d{4})-(\d{2})', crawl_id)
    return (int(m[1]), int(m[2])) if m else None


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def select_crawls(collinfo: list[dict], *, anchor: str = ANCHOR, window_days: int = 365) -> list[dict]:
    """collinfo entries: id, from, to. Window = crawls whose end date is within window_days before the anchor's end date,
    plus every crawl after the anchor (after_corpus). Oldest first."""
    by_id = {c['id']: c for c in collinfo if crawl_key(c['id'])}
    if anchor not in by_id:
        raise ValueError(f'anchor crawl {anchor} not in collinfo')
    start = _dt(by_id[anchor]['to']) - timedelta(days=window_days)
    out = []
    for c in sorted(by_id.values(), key=lambda c: crawl_key(c['id'])):
        after = crawl_key(c['id']) > crawl_key(anchor)
        if after or _dt(c['to']) >= start:
            out.append({'id': c['id'], 'from': c['from'], 'to': c['to'], 'after_corpus': after})
    return out
