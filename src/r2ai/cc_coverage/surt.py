"""SURT key of a corpus URL as in the CC index column url_surtkey (plain hosts, no userinfo / port handling needed here)."""
from __future__ import annotations

from urllib.parse import urlsplit


def surt_key(url: str) -> str:
    s = urlsplit(url.strip())
    host = (s.hostname or '').lower()
    if host.startswith('www.'):
        host = host[4:]
    path = s.path or '/'
    q = ('?' + '&'.join(sorted(s.query.split('&')))) if s.query else ''
    return ','.join(reversed(host.split('.'))) + ')' + path + q
