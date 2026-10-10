"""HTTP client for the Common Crawl index measurement: required UA, pacing, byte/request ledger with hard caps, backoff.

Policy (user, 2026-10-11): 503 / "Slow Down" / 5xx / timeouts = unavailable -> exponential backoff and a breaker cool-down, never
"blocked"; only 403 / 429 are blocked and stop everything. The ledger is cumulative across runs (caps for the whole measurement:
<= 2 GB read and <= 5,000 requests); hitting a cap raises BudgetExceeded and the caller must stop and ask.
No default User-Agent: the caller must pass one (a contact string must come from the user).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import httpx

BACKOFF_S = (5, 15, 45, 120, 600)
BREAKER_FAILS = 3
BREAKER_COOLDOWN_S = 600
MAX_BYTES, MAX_REQUESTS = 2_000_000_000, 5_000
TIMEOUT = httpx.Timeout(connect=15.0, read=90.0, write=15.0, pool=30.0)


class Blocked(Exception):
    """HTTP 403 / 429: stop the whole measurement."""


class Unavailable(Exception):
    """5xx / 503 SlowDown / timeouts persisted after every backoff step."""


class InvalidResponse(Exception):
    """Unexpected 4xx (not 403/429/404) or malformed answer."""


class BudgetExceeded(Exception):
    """The byte or request cap of the whole measurement was reached."""


class Ledger:
    """Cumulative bytes / requests, persisted atomically; check() before every request, add() after."""

    def __init__(self, path, *, max_bytes=MAX_BYTES, max_requests=MAX_REQUESTS):
        self.path, self.max_bytes, self.max_requests = Path(path), max_bytes, max_requests
        self.lock = threading.Lock()
        self.bytes = self.requests = 0
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding='utf-8'))
            self.bytes, self.requests = int(d['bytes']), int(d['requests'])

    def check(self, planned_bytes: int = 0):
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f'request cap {self.max_requests} reached ({self.requests})')
        if self.bytes + planned_bytes > self.max_bytes:
            raise BudgetExceeded(f'byte cap {self.max_bytes} would be exceeded ({self.bytes} + {planned_bytes})')

    def add(self, nbytes: int):
        with self.lock:
            self.bytes += int(nbytes)
            self.requests += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + '.tmp')
            tmp.write_text(json.dumps({'bytes': self.bytes, 'requests': self.requests, 'max_bytes': self.max_bytes,
                                       'max_requests': self.max_requests}), encoding='utf-8')
            os.replace(tmp, self.path)


def classify(status: int | None) -> str:
    if status is None or status >= 500:
        return 'unavailable'
    if status in (403, 429):
        return 'blocked'
    if status in (200, 206):
        return 'ok'
    if status == 404:
        return 'missing'
    return 'invalid'


class Client:
    def __init__(self, *, ua: str, ledger: Ledger, log_path, rps: float = 1.0, transport=None, sleep=time.sleep,
                 clock=time.monotonic):
        if not ua or not ua.strip():
            raise ValueError('User-Agent is required (no default: the contact string must come from the user)')
        self.ua, self.ledger, self.log_path = ua, ledger, Path(log_path)
        self.gap, self.sleep, self.clock = 1.0 / rps, sleep, clock
        self.lock = threading.Lock()
        self.next_at = 0.0
        self.http = httpx.Client(headers={'User-Agent': ua, 'Accept-Encoding': 'identity'}, timeout=TIMEOUT, transport=transport,
                                 follow_redirects=True)

    def _pace(self):
        with self.lock:
            now = self.clock()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.gap
        if wait:
            self.sleep(wait)

    def _log(self, **kw):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False) + '\n')

    def get(self, url: str, *, byte_range: tuple | None = None, planned_bytes: int = 0) -> httpx.Response:
        """GET, optionally a Range: (start, end_inclusive), (start, None) = open end, (-n, None) = the last n bytes.
        Returns ok (200/206) and 404 responses; raises Blocked / InvalidResponse / Unavailable / BudgetExceeded."""
        headers = {}
        if byte_range is not None:
            a, b = byte_range
            headers['Range'] = f'bytes={a}' if (b is None and a < 0) else (f'bytes={a}-' if b is None else f'bytes={a}-{b}')
        fails = 0
        for attempt in range(len(BACKOFF_S) + 1):
            self.ledger.check(planned_bytes)
            self._pace()
            t0 = self.clock()
            status, body, retry_after, err, r = None, b'', None, '', None
            try:
                r = self.http.get(url, headers=headers)
                status, body = r.status_code, r.content
                retry_after = r.headers.get('retry-after')
            except httpx.HTTPError as e:
                err = repr(e)[:160]
            cls = classify(status)
            self.ledger.add(len(body))
            self._log(url=url, range=headers.get('Range'), status=status, cls=cls, bytes=len(body), seconds=round(self.clock() - t0, 3),
                      attempt=attempt, error=err, ledger_bytes=self.ledger.bytes, ledger_requests=self.ledger.requests)
            if cls in ('ok', 'missing'):
                return r
            if cls == 'blocked':
                raise Blocked(f'HTTP {status} from {url}')
            if cls == 'invalid':
                raise InvalidResponse(f'HTTP {status} from {url}')
            fails += 1
            if attempt == len(BACKOFF_S):
                raise Unavailable(f'{url}: unavailable after {attempt + 1} attempts (last {status or err})')
            delay = BACKOFF_S[attempt]
            if fails % BREAKER_FAILS == 0:
                delay = max(delay, BREAKER_COOLDOWN_S)                       # breaker open: cool down, then try again
            if retry_after and re.fullmatch(r'\d+', retry_after.strip()):
                delay = max(delay, int(retry_after))
            self._log(event='backoff', url=url, delay_s=delay, fails=fails)
            self.sleep(delay)
        raise Unavailable(url)

    def close(self):
        self.http.close()


def content_range_total(r: httpx.Response) -> int | None:
    m = re.search(r'/(\d+)\s*$', r.headers.get('content-range', ''))
    return int(m.group(1)) if m else None
