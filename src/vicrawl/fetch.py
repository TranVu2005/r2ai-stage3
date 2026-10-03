"""HTTP layer: per-domain httpx client (HTTP/2, keep-alive), pacing limiter, redirect + cookie-challenge handling."""
from __future__ import annotations

from r2ai.paths import ROOT

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx

from .classify import find_cookie_challenge

log = logging.getLogger('vicrawl.fetch')

CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
MAX_BODY = 8_000_000
TOTAL_TIMEOUT = 60.0
MAX_REDIRECTS = 10
REDIRECTS = (301, 302, 303, 307, 308)

try:
    import brotli  # noqa: F401
    ACCEPT_ENCODING = 'gzip, deflate, br'
except ImportError:  # pragma: no cover
    ACCEPT_ENCODING = 'gzip, deflate'

_META_CHARSET = re.compile(rb'<meta[^>]+charset\s*=\s*["\']?\s*([\w-]+)', re.I)


def decode_html(body: bytes, content_type: str = '') -> tuple[str, str]:
    """Return (text, encoding). Header charset > strict utf-8 > <meta charset> > utf-8 with replacement."""
    m = re.search(r'charset\s*=\s*["\']?([\w-]+)', content_type or '', re.I)
    for enc in ([m[1]] if m else []) + ['utf-8']:
        try:
            return body.decode(enc), enc.lower()
        except (LookupError, UnicodeDecodeError):
            continue
    m2 = _META_CHARSET.search(body[:4096])
    if m2:
        enc = m2[1].decode('ascii', 'ignore')
        try:
            return body.decode(enc), enc.lower()
        except (LookupError, UnicodeDecodeError):
            pass
    return body.decode('utf-8', errors='replace'), 'utf-8'


@dataclass
class Fetched:
    url: str
    final_url: str = ''
    status_code: int | None = None
    headers: dict = field(default_factory=dict)
    body: bytes = b''
    error: str = ''
    latency: float = 0.0       # seconds, last hop
    t0: float = 0.0            # wall clock of first request
    t1: float = 0.0
    discounted: bool = False   # failure/latency caused by our own connectivity (sleep, wifi): ignore
    cookie_solved: bool = False
    redirects: int = 0


class Limiter:
    """Pacing (min interval between request starts) + adaptive concurrency for one domain."""

    def __init__(self, tuner, wait_running, clock=time.monotonic, sleep=asyncio.sleep, jitter=0.1):
        self.tuner, self.wait_running, self.clock, self.sleep, self.jitter = tuner, wait_running, clock, sleep, jitter
        self.active = 0
        self.next_at = 0.0

    def reset(self):
        self.next_at = 0.0

    async def acquire(self):
        while self.active >= self.tuner.conns:
            await self.sleep(0.02)
        self.active += 1
        try:
            while True:
                await self.wait_running()
                now = self.clock()
                until = max(self.next_at, self.tuner.paused_until)
                if now >= until:
                    self.next_at = now + self.tuner.interval() * (1 + random.uniform(0, self.jitter))
                    return
                await self.sleep(min(until - now, 1.0))
        except BaseException:
            self.active -= 1
            raise

    def release(self):
        self.active -= 1


class DomainFetcher:
    def __init__(self, domain: str, *, tuner, limiter: Limiter, netwatch, global_sem: asyncio.Semaphore, max_conns: int, http2: bool = True, ua: str = CHROME_UA):
        self.domain, self.tuner, self.limiter, self.netwatch, self.global_sem = domain, tuner, limiter, netwatch, global_sem
        self.max_conns, self.http2, self.ua = max_conns, http2, ua
        self.client = self._new_client()

    def _new_client(self, cookies=None) -> httpx.AsyncClient:
        try:
            import h2  # noqa: F401
            http2 = self.http2
        except ImportError:  # pragma: no cover
            http2 = False
        return httpx.AsyncClient(
            http2=http2, follow_redirects=False, cookies=cookies,
            limits=httpx.Limits(max_connections=self.max_conns + 2, max_keepalive_connections=self.max_conns + 2, keepalive_expiry=30),
            timeout=httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=30.0),
            headers={'User-Agent': self.ua, 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5',
                     'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.5', 'Accept-Encoding': ACCEPT_ENCODING})

    def reset_client(self):
        """After wake / network change the pooled sockets are dead: start a fresh pool, keep cookies."""
        old, self.client = self.client, self._new_client(cookies=self.client.cookies)
        asyncio.ensure_future(old.aclose())

    async def aclose(self):
        await self.client.aclose()

    async def _hop(self, url: str):
        """One paced request. Returns (response_info or None, error, latency, t0, t1, discounted)."""
        await self.limiter.acquire()
        t0 = time.time()
        m0 = time.monotonic()
        try:
            async with self.global_sem:
                async with asyncio.timeout(TOTAL_TIMEOUT):
                    async with self.client.stream('GET', url) as r:
                        chunks, size = [], 0
                        async for chunk in r.aiter_bytes():
                            chunks.append(chunk)
                            size += len(chunk)
                            if size >= MAX_BODY:
                                break
                        info = (r.status_code, {k.lower(): v for k, v in r.headers.items()}, b''.join(chunks)[:MAX_BODY], str(r.url))
            t1 = time.time()
            return info, '', time.monotonic() - m0, t0, t1, self.netwatch.tainted(t0, t1)
        except (httpx.HTTPError, OSError, asyncio.TimeoutError, ValueError, RuntimeError) as e:
            t1 = time.time()
            err = f'{type(e).__name__}: {str(e)[:200]}'
            discounted = await self.netwatch.report_network_error(t0) or self.netwatch.tainted(t0, t1)
            return None, err, time.monotonic() - m0, t0, t1, discounted
        finally:
            self.limiter.release()

    async def fetch(self, url: str, *, solve_cookie: bool = True) -> Fetched:
        out = Fetched(url=url, final_url=url)
        current, cookie_tries = url, 0
        for _ in range(MAX_REDIRECTS + 2):
            info, err, latency, t0, t1, discounted = await self._hop(current)
            out.t0 = out.t0 or t0
            out.t1, out.latency = t1, latency
            out.discounted = out.discounted or discounted
            if info is None:
                out.error, out.final_url, out.status_code, out.body = err, current, None, b''
                return out
            code, headers, body, final = info
            out.status_code, out.headers, out.body, out.final_url = code, headers, body, current
            if code in REDIRECTS and headers.get('location'):
                target = urljoin(current, headers['location'])
                if urlsplit(target).scheme not in ('http', 'https') or out.redirects >= MAX_REDIRECTS:
                    out.error = 'redirect_limit_or_invalid_scheme'
                    return out
                out.redirects += 1
                current = target
                out.final_url = current
                continue
            if solve_cookie and cookie_tries < 1 and 200 <= code < 400 and len(body) < 8000:
                ch = find_cookie_challenge(body.decode('utf-8', errors='replace'))
                if ch:
                    self.client.cookies.set(ch[0], ch[1], domain=urlsplit(current).hostname or self.domain, path='/')
                    cookie_tries += 1
                    out.cookie_solved = True
                    continue
            return out
        out.error = 'redirect_limit_or_invalid_scheme'
        return out
