"""Read-only helpers for the selective-crawl signal check (no network, no crawl state writes)."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from urllib.parse import unquote, urlsplit

from r2ai.paths import CORPUS_FILE, OUT_DIR, RUNS_DIR, STATE_DIR

OUT = RUNS_DIR / 'selective-check'
VI_DB = STATE_DIR / 'crawl.db'
ZH_DB = STATE_DIR / 'zh_full' / 'crawl.db'
CJK = re.compile(r'[㐀-鿿]')
_D = str.maketrans({'đ': 'd', 'Đ': 'D'})


def ro(path):
    return sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)


def host_of(url: str) -> str:
    h = (urlsplit(url).hostname or '').lower()
    return h[4:] if h.startswith('www.') else h


def strip_marks(s: str) -> str:
    s = unicodedata.normalize('NFD', s.translate(_D))
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')


def slug_text(url: str) -> str:
    """Path + query values, percent-decoded, no scheme/host."""
    p = urlsplit(url)
    return unquote(p.path + (' ' + p.query if p.query else ''))


def slug_tokens(url: str) -> list[str]:
    """Latin alnum tokens of the slug, diacritics stripped, lowercased."""
    return re.findall(r'[a-z0-9]+', strip_marks(slug_text(url)).lower())


def alpha_tokens(url: str) -> list[str]:
    return [t for t in slug_tokens(url) if t.isalpha() and len(t) >= 2]
