"""VN laptop only, direct HTTP, robots at every origin/hop; resumable private DB."""
from __future__ import annotations

from r2ai.paths import CONFIG_DIR, require_inputs
from .common import RUN, RAW, DB, preflight, atomic_json, exclusive
from .sample import EXCLUDED
import argparse
import asyncio
import json
import re
import signal
import time
from urllib.parse import urlsplit, urljoin
import httpx
from vicrawl.fetch import CHROME_UA, MAX_BODY, decode_html
from vicrawl.robots import interpret_robots
from vicrawl.classify import analyze_html, classify_response
from vicrawl.shards import write_shard, cleanup_tmp
from vicrawl.state import StateDB
from vicrawl.settings import Config
from vicrawl.tuner import Tuner, start_rate


def decode_zh(body: bytes, content_type: str):
    text, enc = decode_html(body, content_type)
    if '\ufffd' not in text:
        return text, enc
    try:
        return body.decode('gb18030'), 'gb18030'
    except UnicodeDecodeError:
        return text, enc


def classify_zh(raw, url, final_url, code, error=''):
    stats = analyze_html(raw)
    if re.search(r'页面.{0,6}(不存在|未找到|找不到|删除)|找不到.{0,5}页面|错误.{0,3}404', stats.title):
        return 'soft404_or_home', 'zh_soft404_title'
    return classify_response(stats=stats, raw=raw, url=url, final_url=final_url, http_status=code, error=error)


def defer_robots_row(db, row, retry_at):
    db.conn.execute("UPDATE urls SET status='pending',next_try_at=?,reason='robots_unavailable' WHERE url_norm=?",
                    (retry_at, row['url_norm']))


def next_retry_at(db, domain, limit):
    scope = '' if limit is None else ' AND rank < ?'
    args = (domain,) if limit is None else (domain, limit)
    return db.conn.execute("SELECT MIN(COALESCE(next_try_at,0)) FROM urls WHERE domain=?" + scope +
        " AND (status='pending' OR next_try_at IS NOT NULL)", args).fetchone()[0]


class ZhTuner(Tuner):
    """Opt-in zh slow-server cap, without changing the default vi tuner."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('clock', time.perf_counter)
        super().__init__(*args, **kwargs)

    def record(self, latency, http_status, error):
        events = super().record(latency, http_status, error)
        if self.p50 > 3 and self.cap > 1:
            self.cap, self.rate = 1, min(self.rate, 1)
            events.append({'event': 'zh_slow_cap', 'rate': self.rate, 'p50': self.p50})
        return events


class DirectNetwork:
    def __init__(self):
        self.country, self.checked_at = None, 0.0
        self.lock = asyncio.Lock()

    async def check(self):
        async with self.lock:
            if time.time() - self.checked_at < 60 and self.country == 'VN':
                return
            self.country = None
            async with httpx.AsyncClient(trust_env=False, timeout=15) as c:
                for url in ('https://ipinfo.io/json', 'https://api.country.is/'):
                    try:
                        r = await c.get(url)
                        if r.status_code == 200:
                            self.country = r.json().get('country')
                            if self.country:
                                break
                    except (httpx.HTTPError, ValueError):
                        pass
            self.checked_at = time.time()
            atomic_json(RUN / 'egress.json', {'country': self.country, 'checked_at': self.checked_at,
                                            'direct': True, 'proxy': False})
            if self.country is None:
                raise RuntimeError('EGRESS_UNAVAILABLE: country unknown; no page request allowed')
            if self.country != 'VN':
                raise RuntimeError(f'EGRESS country={self.country}; need VN. Crawl stopped.')


class PoliteClient:
    def __init__(self, domain, cfg, net, db, stop, audit):
        self.domain, self.net, self.stop, self.audit = domain, net, stop, audit
        saved = db.get_domain(domain) or {}
        self.tuner = ZhTuner(cfg.rate_cap, cfg.max_conns, rate=start_rate(saved.get('rate'), cfg.rate_cap),
                           baseline_p95=saved.get('baseline_p95'), crawl_delay=saved.get('crawl_delay'))
        self.next_at = 0.0
        self.pace_lock = asyncio.Lock()
        self.robots_lock = asyncio.Lock()
        self.robots = {}
        self.consecutive_challenges = 0
        if saved.get('state') == 'halted':
            self.tuner.halted, self.tuner.halt_reason = True, saved.get('halt_reason') or 'persisted halt'
        self.client = httpx.AsyncClient(trust_env=False, http2=True, follow_redirects=False,
                            timeout=httpx.Timeout(25, connect=10),
                            limits=httpx.Limits(max_connections=cfg.max_conns, max_keepalive_connections=cfg.max_conns),
                            headers={'User-Agent': CHROME_UA, 'Accept-Language': 'zh-CN,zh;q=0.9',
                                     'Accept': 'text/html,application/xhtml+xml,*/*;q=0.5'})

    async def hop(self, url, kind):
        async with self.pace_lock:
            clock = self.tuner.clock
            while True:
                while clock() < max(self.next_at, self.tuner.paused_until):
                    if self.stop.is_set():
                        raise asyncio.CancelledError
                    await asyncio.sleep(min(1, max(self.next_at, self.tuner.paused_until) - clock()))
                if self.stop.is_set():
                    raise asyncio.CancelledError
                # Geo refresh can await while another response sets a pause or Ctrl+C arrives.
                await self.net.check()
                if self.stop.is_set():
                    raise asyncio.CancelledError
                if clock() >= max(self.next_at, self.tuner.paused_until):
                    break
            self.next_at = clock() + self.tuner.interval()
        t0 = time.time()
        code, headers, body, error = None, {}, b'', ''
        try:
            async with self.client.stream('GET', url) as r:
                code, headers = r.status_code, dict(r.headers)
                parts, size = [], 0
                async for chunk in r.aiter_bytes():
                    parts.append(chunk)
                    size += len(chunk)
                    if size >= MAX_BODY:
                        break
                body = b''.join(parts)[:MAX_BODY]
        except httpx.HTTPError as e:
            error = f'{type(e).__name__}: {str(e)[:200]}'
        dt = time.time() - t0
        self.audit.write(json.dumps({'domain': self.domain, 'url': url, 'kind': kind, 't0': t0,
                                    't1': time.time(), 'latency': dt, 'http_status': code, 'error': error,
                                    'bytes': len(body)}, ensure_ascii=False) + '\n')
        self.audit.flush()
        for event in self.tuner.record(dt, code, bool(error)):
            print(self.domain, event, flush=True)
        return code, headers, body, error, dt

    async def allowed(self, url):
        u = urlsplit(url)
        if u.scheme not in ('http', 'https') or u.username or u.password or u.hostname is None:
            return False, 'invalid_url'
        host = u.hostname.lower().removeprefix('www.')
        if host in EXCLUDED:
            return False, 'excluded_domain'
        origin = f'{u.scheme}://{u.netloc}'
        async with self.robots_lock:
            if origin not in self.robots:
                robot_url = origin + '/robots.txt'
                code, body, error = None, b'', ''
                for _ in range(6):
                    code, headers, body, error, _ = await self.hop(robot_url, 'robots')
                    if code in (301, 302, 303, 307, 308) and headers.get('location'):
                        target = urljoin(robot_url, headers['location'])
                        v = urlsplit(target)
                        # A robots redirect may move scheme/www only; fail closed across other hosts.
                        if v.scheme not in ('http', 'https') or v.username or v.password or (v.hostname or '').removeprefix('www.') != host:
                            error = 'robots_cross_host_redirect'
                            break
                        robot_url = target
                    else:
                        break
                else:
                    error = 'robots_redirect_limit'
                text, _ = decode_zh(body, '')
                robots = interpret_robots(None if error else code, text, CHROME_UA, error=error)
                self.robots[origin] = robots
                if robots.crawl_delay:
                    self.tuner.crawl_delay = max(self.tuner.crawl_delay or 0, robots.crawl_delay)
                path = RUN / 'robots' / (self.domain + '_' + u.scheme + '_' + (u.hostname or '') + '.json')
                atomic_json(path, {'origin': origin, 'url': robot_url, 'http_status': code,
                                  'state': robots.state, 'crawl_delay': robots.crawl_delay, 'text': text,
                                  'checked_at': time.time(), 'error': error})
            robots = self.robots[origin]
        return robots.can_fetch(url), 'robots_' + robots.state.lower()

    async def fetch(self, row):
        url, current = row['url'], row['url']
        code, headers, body, error, latency = None, {}, b'', '', 0.0
        for _ in range(11):
            allowed, reason = await self.allowed(current)
            if not allowed:
                return {**row, 'domain': self.domain, 'html': None, 'encoding': '',
                        'status': 'network_error' if reason == 'robots_unavailable' else 'blocked_4xx',
                        'reason': reason, 'http_status': None, 'final_url': current, 'fetched_at': time.time(),
                        'attempts': row['attempts'] + 1}
            code, headers, body, error, latency = await self.hop(current, 'page')
            if code in (301, 302, 303, 307, 308) and headers.get('location'):
                current = urljoin(current, headers['location'])
            else:
                break
        else:
            error = 'redirect_limit'
        ctype = headers.get('content-type', '')
        text, enc = decode_zh(body, ctype)
        status, reason = classify_zh(text, url, current, code, error)
        if status in ('bot_challenge', 'cookie_challenge'):
            self.consecutive_challenges += 1
            self.tuner.rate = max(.25, self.tuner.rate / 2)
            self.tuner.paused_until = self.tuner.clock() + 60
            if self.consecutive_challenges >= 12:
                self.tuner.halted, self.tuner.halt_reason = True, '12 consecutive challenges; no bypass attempted'
        else:
            self.consecutive_challenges = 0
        if ctype and not any(t in ctype.lower() for t in ('html', 'xml', 'text/plain')):
            status, reason, text = 'thin', 'non_html', ''
        return {**row, 'domain': self.domain, 'html': text, 'encoding': enc, 'status': status, 'reason': reason,
                'content_type': ctype, 'latency': latency, 'http_status': code, 'final_url': current,
                'fetched_at': time.time(), 'attempts': row['attempts'] + 1}


async def crawl(limit=None, domains=None):
    preflight()
    require_inputs(DB, RUN / 'sample.json', CONFIG_DIR / 'domains-zh-sample.yaml')
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    def interrupt(signum, frame):
        loop.call_soon_threadsafe(stop.set)
    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, interrupt)
    net = DirectNetwork()
    await net.check()
    cleanup_tmp(RAW)
    db = StateDB(DB)
    db.recover()
    # Repair only this experiment's early smoke records: unavailable robots are retryable, never permanent.
    db.conn.execute("UPDATE urls SET status='pending', attempts=MAX(0,attempts-1), next_try_at=NULL "
                    "WHERE reason='robots_unavailable' AND status IN ('network_error','blocked_4xx')")
    cfg = Config.load(CONFIG_DIR / 'domains-zh-sample.yaml')
    selected = sorted({g['domain'] for g in json.loads((RUN / 'sample.json').read_text('utf-8'))})
    if domains:
        selected = [d for d in selected if d in domains]
    with (RUN / 'requests.jsonl').open('a', encoding='utf-8') as audit:
        async def domain_run(domain):
            c = PoliteClient(domain, cfg.for_domain(domain), net, db, stop, audit)
            db.upsert_domain(domain, state='active', updated_at=time.time())
            t0, n = time.time(), 0
            buffer = []
            robot_pause_until = 0.0
            def flush():
                if not buffer:
                    return
                shard = write_shard(RAW, domain, buffer)
                db.commit_results([{**r, 'shard': shard.name} for r in buffer])
                buffer.clear()
            async def worker():
                nonlocal n, robot_pause_until
                try:
                    while not stop.is_set() and not c.tuner.halted and not robot_pause_until:
                        rows = db.next_batch(domain, 1, time.time(), limit)
                        if not rows:
                            break
                        rec = await c.fetch(rows[0])
                        if rec['reason'] == 'robots_unavailable':
                            robot_pause_until = max((r.retry_at or 0 for r in c.robots.values() if r.state == 'UNAVAILABLE'), default=0)
                            robot_pause_until = max(robot_pause_until, time.time() + 1800)
                            defer_robots_row(db, rows[0], robot_pause_until)
                            return
                        buffer.append(rec)
                        n += 1
                        if len(buffer) >= 10:
                            flush()
                except Exception:
                    stop.set()
                    raise
            try:
                while not stop.is_set():
                    results = await asyncio.gather(*(worker() for _ in range(cfg.for_domain(domain).max_conns)), return_exceptions=True)
                    failure = next((r for r in results if isinstance(r, Exception)), None)
                    if failure:
                        raise failure
                    flush()
                    if c.tuner.halted:
                        break
                    retry_at = robot_pause_until or next_retry_at(db, domain, limit)
                    if retry_at is None:
                        break
                    waiting = 'waiting_robots' if robot_pause_until else 'waiting_retry'
                    db.upsert_domain(domain, state=waiting, updated_at=time.time())
                    print(f'{domain}: {waiting}, retry in {max(0,retry_at-time.time()):.1f}s', flush=True)
                    while time.time() < retry_at and not stop.is_set():
                        await asyncio.sleep(1)
                    c.robots = {o:r for o,r in c.robots.items() if r.state != 'UNAVAILABLE'}
                    robot_pause_until = 0.0
                    db.upsert_domain(domain, state='active', updated_at=time.time())
            except Exception:
                stop.set()
                raise
            finally:
                flush()
                db.upsert_domain(domain, rate=c.tuner.rate, conns=c.tuner.conns,
                    baseline_p95=c.tuner.baseline_p95, crawl_delay=c.tuner.crawl_delay,
                    state='halted' if c.tuner.halted else ('stopped' if stop.is_set() else 'sample_finished'),
                    halt_reason=c.tuner.halt_reason, updated_at=time.time())
                await c.client.aclose()
                with (RUN / 'crawl_sessions.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'domain': domain, 't0': t0, 't1': time.time(), 'url_attempts': n,
                                       'limit': limit, 'halted': c.tuner.halted}) + '\n')
                print(f'{domain}: {n} URL attempts in {time.time()-t0:.1f}s', flush=True)
        tasks = [asyncio.create_task(domain_run(d)) for d in selected]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [str(r) for r in results if isinstance(r, Exception)]
            if failures:
                raise RuntimeError('; '.join(failures))
            # A safety halt is an explicit censored cutoff, not a completed sample.
            atomic_json(RUN / ('smoke_completion.json' if limit else 'crawl_completion.json'), {
                'limit': limit, 'complete_sample': not stop.is_set() and not any((db.get_domain(d) or {}).get('state') == 'halted' for d in selected),
                'domains': {d: {'state': (db.get_domain(d) or {}).get('state'),
                               'halt_reason': (db.get_domain(d) or {}).get('halt_reason'),
                               'remaining_retry_at': next_retry_at(db,d,limit)} for d in selected},
                'note': 'halted domains are censored by safety policy; no automatic bypass/restart'})
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            db.recover()
            db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            db.conn.close()
    return 130 if stop.is_set() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, help='rank cutoff per domain; smoke = 50')
    ap.add_argument('--domains', help='comma-separated sampled domains')
    a = ap.parse_args()
    with exclusive('crawl'):
        return run_crawl(a.limit, set(a.domains.split(',')) if a.domains else None)


def run_crawl(limit=None, domains=None):
    previous = {getattr(signal, name): signal.getsignal(getattr(signal, name))
                for name in ('SIGINT', 'SIGTERM', 'SIGBREAK') if hasattr(signal, name)}
    try:
        return asyncio.run(crawl(limit, domains))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
