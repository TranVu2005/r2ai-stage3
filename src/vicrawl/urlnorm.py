"""URL normalisation and duplicate grouping (http/https/www/fragment/trailing-slash variants)."""
from __future__ import annotations

from r2ai.paths import ROOT

from urllib.parse import urlsplit


def _host(netloc: str, scheme: str) -> str:
    host = netloc.rsplit('@', 1)[-1].lower()
    if host.startswith('www.'):
        host = host[4:]
    for default in (':80', ':443'):
        if host.endswith(default):
            host = host[: -len(default)]
            break
    return host


def normalize_url(url: str) -> str:
    """Drop scheme, `www.`, fragment, trailing `/`; lowercase host; KEEP path case and query."""
    u = urlsplit(url.strip())
    path = u.path.rstrip('/')
    return _host(u.netloc, u.scheme) + path + (('?' + u.query) if u.query else '')


def domain_of(url: str) -> str:
    u = urlsplit(url.strip())
    host = u.netloc.rsplit('@', 1)[-1].lower()
    return host[4:] if host.startswith('www.') else host


def _rank(url: str, doc_id: int):
    """Smaller is better: https first, then shortest url, then lowest id."""
    return (0 if url.lower().startswith('https://') else 1, len(url), doc_id)


def group_urls(rows) -> list[dict]:
    """rows: iterable of (doc_id, url). One group per url_norm with the preferred representative url."""
    groups: dict[str, dict] = {}
    for doc_id, url in rows:
        key = normalize_url(url)
        g = groups.get(key)
        if g is None:
            groups[key] = {'url_norm': key, 'url': url, 'domain': domain_of(url), 'doc_ids': [doc_id], '_rank': _rank(url, doc_id)}
            continue
        g['doc_ids'].append(doc_id)
        r = _rank(url, doc_id)
        if r < g['_rank']:
            g['url'], g['_rank'] = url, r
    out = []
    for g in groups.values():
        g['doc_ids'].sort()
        g.pop('_rank')
        out.append(g)
    return out
