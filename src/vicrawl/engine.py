"""Crawl orchestrator: per-domain runners, shard flushing, progress, graceful stop, sleep/offline handling."""
from __future__ import annotations

from r2ai.paths import ROOT, cli_script

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import dns_cache
from .classify import PageStats, analyze_html, classify_response
from .fetch import DomainFetcher, Limiter, decode_html
from .netwatch import NetWatch
from .robots import interpret_robots
from .settings import QA_GATE_DOCS, QA_GATE_MIN_URLS, ROOT, Config, Paths, safe_name
from .shards import cleanup_tmp, write_shard
from .state import DBWriter
from .tuner import Tuner, start_rate

log = logging.getLogger('vicrawl')

EGRESS_URL = 'https://ipinfo.io/json'
EGRESS_FALLBACKS = ('https://api.country.is/',)
TCP_PROBES = (('1.1.1.1', 443), ('8.8.8.8', 53), ('9.9.9.9', 443))
FLUSH_DOCS, FLUSH_SECS, FLUSH_BYTES = 500, 30.0, 8_000_000
MILESTONE_OK = 100_000


@dataclass
class RunOptions:
    domains: set | None = None
    group: str | None = None
    limit: int | None = None
    max_conns: int = 64
    hours: float | None = None
    until: str | None = None
    egress_url: str = EGRESS_URL
    probe_url: str = ''
    progress_every: float = 60.0
    report_every: float = 3600.0
    stop_grace: float = 30.0
    flush_docs: int = FLUSH_DOCS
    flush_secs: float = FLUSH_SECS
    tick: float = 5.0
    skip_report: bool = False
    skip_qa_subprocess: bool = False
    milestone_ok: int = MILESTONE_OK
    qa_docs: int = QA_GATE_DOCS
    pause_s: float = 60.0


def compute_deadline(hours: float | None, until: str | None, now: float) -> float | None:
    """--hours H => now + H*3600.  --until HH:MM => next local occurrence of that wall-clock time."""
    if hours:
        return now + hours * 3600
    if until:
        hh, mm = (int(x) for x in until.split(':'))
        t = time.localtime(now)
        target = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0, -1))
        return target if target > now else target + 86400
    return None


# --------------------------------------------------------------------------------- network probes
async def egress_country(url: str = EGRESS_URL, timeout: float = 10.0) -> str | None:
    urls = [url] + ([] if url != EGRESS_URL else list(EGRESS_FALLBACKS))
    async with httpx.AsyncClient(timeout=timeout, headers={'User-Agent': 'curl/8.0'}) as c:
        for u in urls:
            try:
                r = await c.get(u)
                if r.status_code == 200:
                    j = r.json()
                    cc = j.get('country') or j.get('country_code')
                    if cc:
                        return str(cc).upper()
            except (httpx.HTTPError, ValueError, OSError):
                continue
    return None


async def check_online(probe_url: str = '') -> bool:
    if probe_url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (await c.get(probe_url)).status_code == 200
        except (httpx.HTTPError, OSError):
            return False
    async def tcp(host, port):
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(host, port), 3.0)
            w.close()
            return True
        except (OSError, asyncio.TimeoutError):
            return False
    results = await asyncio.gather(*(tcp(h, p) for h, p in TCP_PROBES))
    return any(results)


# --------------------------------------------------------------------------------- domain runner
class DomainRunner:
    def __init__(self, eng: 'Engine', domain: str, counts: dict, dom_row: dict | None):
        self.eng, self.domain = eng, domain
        self.cfg = eng.cfg.for_domain(domain)
        self.counts = counts
        self.total = sum(counts.values())
        if eng.opts.limit:
            self.total = min(self.total, eng.opts.limit)
        self.done = sum(n for s, n in counts.items() if s not in ('pending', 'in_progress'))
        self.ok_total = counts.get('ok', 0)
        self.approved = bool((dom_row or {}).get('approved'))
        self.db_state = (dom_row or {}).get('state')
        self.db_halt_reason = (dom_row or {}).get('halt_reason')
        learned = (dom_row or {}).get('rate')
        rate = start_rate(learned, self.cfg.rate_cap) if learned else min(self.cfg.rate_cap, self.cfg.start_rate)
        self.tuner = Tuner(self.cfg.rate_cap, self.cfg.max_conns, rate=rate, baseline_p95=(dom_row or {}).get('baseline_p95'),
                           crawl_delay=(dom_row or {}).get('crawl_delay'), pause_s=eng.opts.pause_s)
        self.tuner.conns = min(self.cfg.max_conns, max(1, int((dom_row or {}).get('conns') or 1)))
        self.limiter = Limiter(self.tuner, eng.netwatch.wait_running)
        self.fetcher = DomainFetcher(domain, tuner=self.tuner, limiter=self.limiter, netwatch=eng.netwatch, global_sem=eng.global_sem,
                                     max_conns=self.cfg.max_conns, http2=self.cfg.http2)
        self.queue: deque = deque()
        self.lock = asyncio.Lock()
        self.origin = None
        self.flush_lock = asyncio.Lock()
        self.buffer: list[dict] = []
        self.buf_bytes = 0
        self.buf_first = 0.0
        self.robots = None
        self.exhausted = False
        self.stop_flag = False
        self.state = 'pending'
        self.req_times: deque = deque()
        self.n_requests = 0
        self.task: asyncio.Task | None = None
        self.ok_buffered = 0

    # -- lifecycle -------------------------------------------------------------------------
    @property
    def qa_gate(self) -> bool:
        if self.cfg.qa_gate is not None:
            return self.cfg.qa_gate
        return sum(self.counts.values()) >= QA_GATE_MIN_URLS

    @property
    def finished(self) -> bool:
        return self.task is not None and self.task.done()

    async def run(self):
        self.state = 'running'
        if self.db_state == 'halted':
            self.state = 'halted'
            log.warning('[%s] skipped: halted by an earlier run (%s). `python crawl.py reset-errors %s` to retry.', self.domain, self.db_halt_reason, self.domain)
            return
        if self.qa_gate and not self.approved and self.ok_total >= self.eng.opts.qa_docs:
            self.stop_flag, self.state = True, 'awaiting_qa'          # gate was already reached in an earlier run
            self.eng.persist_domain(self, state='awaiting_qa')
            if not (self.eng.paths.qa_dir / f'{safe_name(self.domain)}.md').exists():
                self.eng.spawn_qa(self)
            log.warning('[%s] waiting for approval (python crawl.py approve %s)', self.domain, self.domain)
            return
        self.eng.persist_domain(self, state='active')
        workers = [asyncio.create_task(self._worker(i)) for i in range(self.cfg.max_conns)]
        try:
            await asyncio.gather(*workers)
        finally:
            for w in workers:
                w.cancel()
        if self.state == 'running':
            self.state = ('limit_reached' if self.eng.opts.limit else 'done') if self.exhausted else 'stopped'
        self.eng.persist_domain(self, state=self.state if self.state in ('done', 'halted', 'awaiting_qa', 'limit_reached') else 'active')
        log.info('[%s] runner finished: %s (requests this run=%d)', self.domain, self.state, self.n_requests)

    async def _worker(self, idx: int):
        while True:
            row = await self.pop()
            if row is None:
                return
            try:
                await self.process(row)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  one bad page must not kill the domain
                log.exception('[%s] process failed for %s', self.domain, row.get('url'))

    # -- queue -----------------------------------------------------------------------------
    async def pop(self):
        while True:
            if self.stop_flag or self.eng.stopping.is_set():
                return None
            if self.robots is None or self.robots.state == 'UNAVAILABLE':
                async with self.lock:
                    if not await self.ensure_robots():
                        return None
                continue
            if self.queue:
                return self.queue.popleft()
            if self.exhausted:
                return None
            async with self.lock:
                if self.queue:
                    continue
                if not await self.refill():
                    return None

    async def refill(self) -> bool:
        limit = self.eng.opts.limit
        rows = await self.eng.writer.call(lambda db: db.next_batch(self.domain, 100, time.time(), limit))
        if not rows:
            self.exhausted = True
            return False
        self.queue.extend(rows)
        return True

    async def ensure_robots(self) -> bool:
        """False => nothing (more) to do for this domain right now."""
        while True:
            if self.robots is not None and self.robots.state != 'UNAVAILABLE':
                return True
            if self.robots is not None:
                self.state = 'waiting_robots'
                while time.time() < (self.robots.retry_at or 0):
                    if self.stop_flag or self.eng.stopping.is_set():
                        return False
                    await asyncio.sleep(2)
                self.state = 'running'
            if self.origin is None:
                if not self.queue and not await self.refill():
                    return False
                from urllib.parse import urlsplit
                u = urlsplit(self.queue[0]['url'])
                self.origin = f'{u.scheme}://{u.netloc}'
            f = await self.fetcher.fetch(self.origin + '/robots.txt')
            if f.discounted and f.error:
                await self.eng.netwatch.wait_running()
                continue
            text = decode_html(f.body, f.headers.get('content-type', ''))[0] if f.body else ''
            self.robots = interpret_robots(f.status_code, text, self.fetcher.ua, error=f.error)
            if self.robots.crawl_delay:
                self.tuner.crawl_delay = self.robots.crawl_delay
            log.info('[%s] robots=%s crawl_delay=%s', self.domain, self.robots.state, self.robots.crawl_delay)
            self.eng.persist_domain(self, crawl_delay=self.robots.crawl_delay)
            if self.robots.state == 'UNAVAILABLE':
                log.warning('[%s] robots.txt unavailable (%s %s); retry in 30 min, not treated as allowed', self.domain, f.status_code, f.error)

    # -- one URL ---------------------------------------------------------------------------
    async def process(self, row: dict):
        url = row['url']
        base = {'url_norm': row['url_norm'], 'url': url, 'doc_ids': row['doc_ids'], 'domain': self.domain, 'attempts': row['attempts'] + 1}
        if not self.robots.can_fetch(url):
            self._buffer({**base, 'final_url': url, 'status': 'blocked_4xx', 'reason': 'robots_disallowed', 'http_status': None,
                          'fetched_at': time.time(), 'html': None})
            return
        while True:
            f = await self.fetcher.fetch(url)
            if f.discounted and f.error:
                await self.eng.netwatch.wait_running()     # our connectivity, not the site: try the same URL again
                continue
            break
        ctype = f.headers.get('content-type', '')
        html, enc, stats, raw = None, '', PageStats(), ''
        is_html = (not ctype) or any(t in ctype.lower() for t in ('html', 'xml', 'text/plain'))
        if f.body and is_html:
            raw, enc = decode_html(f.body, ctype)
            stats = await asyncio.get_running_loop().run_in_executor(self.eng.pool, analyze_html, raw)
            html = raw
        state, reason = classify_response(stats=stats, raw=raw, url=url, final_url=f.final_url, http_status=f.status_code, error=f.error)
        if f.body and not is_html:
            reason = f'non_html:{ctype[:60]}'
        rec = {**base, 'final_url': f.final_url, 'status': state, 'reason': reason, 'http_status': f.status_code, 'fetched_at': time.time(),
               'content_type': ctype, 'encoding': enc, 'server': f.headers.get('server', ''), 'cookie_solved': f.cookie_solved,
               'latency': round(f.latency, 3), 'html': html}
        self._buffer(rec)
        if state == 'ok':
            self.ok_buffered += 1
            self.check_qa_gate()
        self.n_requests += 1
        self.req_times.append(time.monotonic())
        if not f.discounted:
            soft_block = state in ('bot_challenge', 'cookie_challenge')
            for ev in self.tuner.record(f.latency, f.status_code, bool(f.error) or soft_block):
                self.on_tuner_event(ev)

    def on_tuner_event(self, ev: dict):
        log.info('[%s] %s', self.domain, json.dumps(ev))
        if ev['event'] == 'halt':
            self.stop_flag = True
            self.state = 'halted'
            self.eng.persist_domain(self, state='halted', halt_reason=ev['reason'])
            log.error('[%s] HALTED: %s', self.domain, ev['reason'])
        self.eng.persist_domain(self)

    # -- buffer / shards ---------------------------------------------------------------------
    def _buffer(self, rec: dict):
        if not self.buffer:
            self.buf_first = time.monotonic()
        self.buffer.append(rec)
        self.buf_bytes += len(rec.get('html') or '') + 400

    def flush_due(self, now: float, force: bool = False) -> bool:
        if not self.buffer:
            return False
        return force or len(self.buffer) >= self.eng.opts.flush_docs or self.buf_bytes >= FLUSH_BYTES or now - self.buf_first >= self.eng.opts.flush_secs

    async def flush(self):
        async with self.flush_lock:
            if not self.buffer:
                return
            recs, self.buffer, self.buf_bytes = self.buffer, [], 0
            try:
                path = await asyncio.to_thread(write_shard, self.eng.paths.raw_dir, self.domain, recs)
            except OSError:
                log.exception('[%s] shard write failed; keeping %d records in memory', self.domain, len(recs))
                self.buffer = recs + self.buffer
                self.buf_bytes = sum(len(r.get('html') or '') + 400 for r in self.buffer)
                await asyncio.sleep(5)
                return
            # shard is renamed: only now may the rows be marked done
            rows = [{'url_norm': r['url_norm'], 'status': r['status'], 'http_status': r['http_status'], 'reason': r['reason'],
                     'fetched_at': r['fetched_at'], 'shard': path.name, 'final_url': r['final_url']} for r in recs]
            policy = self.cfg.retry_policy()
            await self.eng.writer.call(lambda db: db.commit_results(rows, policy))
            n_ok = sum(r['status'] == 'ok' for r in recs)
            self.done += len(recs)
            self.ok_total += n_ok
            self.ok_buffered -= n_ok

    def check_qa_gate(self):
        """Counted when a doc is buffered, not when its shard is flushed: at 8 req/s a 30s flush window would overshoot by hundreds."""
        if self.qa_gate and not self.approved and self.ok_total + self.ok_buffered >= self.eng.opts.qa_docs and not self.stop_flag:
            self.stop_flag = True
            self.state = 'awaiting_qa'
            self.eng.persist_domain(self, state='awaiting_qa')
            log.warning('[%s] QA GATE: %d ok docs. Stopping domain; review out/qa_extract/%s.md then: python crawl.py approve %s',
                        self.domain, self.ok_total, safe_name(self.domain), self.domain)
            self.eng.spawn_qa(self)


# --------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, paths: Paths, cfg: Config, opts: RunOptions):
        self.paths, self.cfg, self.opts = paths, cfg, opts
        self.writer = DBWriter(paths.db)
        self.runners: dict[str, DomainRunner] = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='analyze')
        self.stopping = asyncio.Event()
        self.stop_reason = ''
        self.global_sem = asyncio.Semaphore(opts.max_conns)
        self.netwatch: NetWatch | None = None
        self.started = time.time()
        self.deadline: float | None = None
        self.qa_tasks: list[asyncio.Task] = []
        self.explicit = bool(opts.domains)

    # -- control ---------------------------------------------------------------------------
    def request_stop(self, reason: str):
        if not self.stopping.is_set():
            self.stop_reason = reason
            log.warning('STOP requested (%s): finishing in-flight requests, flushing, then exit', reason)
            self.stopping.set()

    def install_signals(self, loop):
        presses = {'n': 0}

        def handler(signum, frame):
            presses['n'] += 1
            if presses['n'] == 1:
                loop.call_soon_threadsafe(self.request_stop, f'signal {signum}')
            else:
                os.write(2, b'second interrupt: exiting immediately (state is crash-safe)\n')
                os._exit(130)
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def persist_domain(self, r: DomainRunner, **extra):
        fields = {'rate': r.tuner.rate, 'conns': r.tuner.conns, 'baseline_p95': r.tuner.baseline_p95, 'updated_at': time.time(), **extra}
        self.writer.post(lambda db: db.upsert_domain(r.domain, **fields))

    # -- QA gate ---------------------------------------------------------------------------
    def spawn_qa(self, r: DomainRunner):
        self.qa_tasks.append(asyncio.create_task(self.make_qa(r)))

    async def make_qa(self, r: DomainRunner):
        await r.flush()
        if self.opts.skip_qa_subprocess:
            return
        cmd = [sys.executable, str(cli_script('extract.py')), 'qa', r.domain, '--raw-dir', str(self.paths.raw_dir), '--out-dir', str(self.paths.out_dir), '--config', str(self.paths.config)]
        try:
            p = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await p.communicate()
            log.info('[%s] QA report: %s', r.domain, (out or b'').decode('utf-8', 'replace').strip()[-300:])
        except OSError:
            log.exception('QA subprocess failed for %s', r.domain)

    # -- main ------------------------------------------------------------------------------
    def _select(self, counts: dict) -> list[str]:
        doms = set(counts)
        if self.opts.domains:
            doms &= self.opts.domains
        if self.opts.group:
            doms = {d for d in doms if self.cfg.for_domain(d).lang == self.opts.group or self.cfg.for_domain(d).group == self.opts.group}
        return sorted(doms)

    def _deadline(self) -> float | None:
        return compute_deadline(self.opts.hours, self.opts.until, time.time())

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        self.install_signals(loop)
        removed = cleanup_tmp(self.paths.raw_dir)
        if removed:
            log.info('removed %d partial shard(s)', removed)
        await self.writer.start()
        recovered = await self.writer.call(lambda db: db.recover())
        if recovered:
            log.info('recovered %d in_progress URL(s) back into the queue', recovered)
        country = await egress_country(self.opts.egress_url)
        if country != 'VN':
            log.error('EGRESS CHECK FAILED: country=%s (need VN). Probe/benchmark results are only valid from a Vietnamese IP. Not crawling.', country)
            await self.writer.stop()
            return 3
        log.info('egress ok (country=VN)')
        dns_cache.install()

        async def _online():
            return await check_online(self.opts.probe_url)

        async def _egress():
            return await egress_country(self.opts.egress_url)

        self.netwatch = NetWatch(check_online=_online, check_egress=_egress, tick=self.opts.tick, on_resume=self._on_resume)
        counts = await self.writer.call(lambda db: db.counts(self.opts.limit))
        doms_db = await self.writer.call(lambda db: db.all_domains())
        selected = self._select(counts)
        if not selected:
            log.error('no domains to crawl (selection empty). Run `python crawl.py init` first.')
            await self.writer.stop()
            return 2
        self.deadline = self._deadline()
        for d in selected:
            self.runners[d] = DomainRunner(self, d, counts[d], doms_db.get(d))
        log.info('selected %d domain(s)%s', len(selected), f', limit={self.opts.limit}/domain' if self.opts.limit else '')
        net_task = asyncio.create_task(self.netwatch.run())
        try:
            await self._supervise()
        finally:
            net_task.cancel()
            await self._shutdown()
        return 0

    def _on_resume(self):
        dns_cache.clear()
        for r in self.runners.values():
            r.limiter.reset()
            r.fetcher.reset_client()

    def _wave_ok(self, r: DomainRunner, active_waves: int) -> bool:
        return self.explicit or r.cfg.background or r.cfg.priority <= active_waves

    async def _supervise(self):
        last_progress = last_report = last_poll = time.monotonic()
        started: set[str] = set()
        while True:
            now = time.monotonic()
            if self.deadline and time.time() >= self.deadline:
                self.request_stop('deadline reached')
            if self.stopping.is_set():
                return
            # waves: lowest priority number not yet finished is active (background domains never block)
            blocking = [r for r in self.runners.values() if not r.cfg.background and not (r.finished)]
            active_wave = min((r.cfg.priority for r in blocking), default=10 ** 6)
            for d, r in self.runners.items():
                if d not in started and self._wave_ok(r, active_wave):
                    started.add(d)
                    r.task = asyncio.create_task(r.run(), name=d)
            for r in self.runners.values():
                if r.flush_due(now) and not r.flush_lock.locked():
                    asyncio.create_task(r.flush())
            if now - last_poll >= 10:
                last_poll = now
                await self._poll_approvals(started)
            if now - last_progress >= self.opts.progress_every:
                last_progress = now
                await self._progress()
            if not self.opts.skip_report and now - last_report >= self.opts.report_every:
                last_report = now
                self._spawn_report()
            if all(r.finished for r in self.runners.values()) and len(started) == len(self.runners):
                return
            await asyncio.sleep(0.5)

    async def _poll_approvals(self, started: set):
        rows = await self.writer.call(lambda db: db.all_domains())
        for d, r in self.runners.items():
            row = rows.get(d) or {}
            if r.state == 'awaiting_qa' and row.get('approved') and r.finished:
                log.info('[%s] approved: resuming', d)
                r.approved, r.stop_flag, r.exhausted, r.state = True, False, False, 'pending'
                r.task = asyncio.create_task(r.run(), name=d)

    async def _progress(self):
        now = time.monotonic()
        total_done = total_all = 0
        lines = []
        for d, r in sorted(self.runners.items()):
            while r.req_times and now - r.req_times[0] > 60:
                r.req_times.popleft()
            cur = len(r.req_times) / 60.0
            remaining = max(0, r.total - r.done)
            eta = f'{remaining / cur / 3600:.1f}h' if cur > 0.01 and remaining else ('-' if not remaining else '?')
            ok_pct = 100.0 * r.ok_total / max(1, r.done)
            total_done += r.done
            total_all += r.total
            if r.state == 'pending' or (r.finished and r.state == 'done' and not r.n_requests):
                continue
            lines.append(f'  {d:<34} {r.done:>7}/{r.total:<7} ok={ok_pct:5.1f}% cur={cur:4.2f}/s target={r.tuner.rate:.2f} conns={r.tuner.conns} {r.state:<13} ETA={eta}')
        elapsed = max(1.0, time.time() - self.started)
        n_req = sum(r.n_requests for r in self.runners.values())
        log.info('PROGRESS total %d/%d  req=%d  avg=%.2f req/s  paused=%s\n%s', total_done, total_all, n_req, n_req / elapsed, self.netwatch.paused, '\n'.join(lines))
        ok = sum(r.ok_total for r in self.runners.values())
        ms = self.paths.out_dir / 'MILESTONE_1.md'
        if ok >= self.opts.milestone_ok and not ms.exists():
            self.paths.out_dir.mkdir(parents=True, exist_ok=True)
            ms.write_text(f'# MILESTONE 1\n\nTổng số doc `ok` (trạng thái lúc crawl): **{ok}** (>= {self.opts.milestone_ok}) tại {time.strftime("%Y-%m-%d %H:%M:%S")}.\n'
                          'Chạy `python extract.py` để trích xuất; baseline có thể bắt đầu từ `data/docs_vi/*.parquet`.\n', encoding='utf-8')
            log.warning('MILESTONE 1 reached: %d ok docs -> %s', ok, ms)

    def _spawn_report(self):
        try:
            subprocess.Popen([sys.executable, str(cli_script('crawl.py')), 'report', '--state-dir', str(self.paths.state_dir), '--out-dir', str(self.paths.out_dir),
                              '--raw-dir', str(self.paths.raw_dir), '--docs-dir', str(self.paths.docs_dir)], cwd=str(ROOT),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            log.exception('report spawn failed')

    async def _shutdown(self):
        self.stopping.set()
        for r in self.runners.values():
            r.stop_flag = True
        tasks = [r.task for r in self.runners.values() if r.task and not r.task.done()]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.opts.stop_grace)
            for t in pending:
                t.cancel()
            if pending:
                log.warning('%d runner(s) still busy after %ss: cancelled (their URLs return to the queue)', len(pending), self.opts.stop_grace)
                await asyncio.gather(*pending, return_exceptions=True)
        for r in self.runners.values():
            try:
                await r.flush()
            except Exception:  # noqa: BLE001  a failed commit leaves rows in_progress => re-fetched next run
                log.exception('[%s] final flush failed', r.domain)
            self.persist_domain(r)
        await self.writer.call(lambda db: db.recover())     # anything still in_progress goes back to pending
        counts = await self.writer.call(lambda db: db.counts())
        if self.qa_tasks:
            await asyncio.wait(self.qa_tasks, timeout=120)       # let the QA reports finish: they are why the domains stopped
        for r in self.runners.values():
            await r.fetcher.aclose()
        self._summary(counts)
        await self.writer.stop()
        try:
            from .report import build_report
            await asyncio.to_thread(build_report, self.paths.db, self.paths.docs_dir, self.paths.out_dir / 'crawl_vi_report.md')
        except Exception:  # noqa: BLE001
            log.exception('final report failed')
        dns_cache.uninstall()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _summary(self, counts: dict):
        lines = [f'--- summary ({self.stop_reason or "finished"}) ---']
        pend_after = 0
        for d in sorted(self.runners):
            c = counts.get(d, {})
            pend = c.get('pending', 0)
            pend_after += pend
            lines.append(f'{d}: ' + ', '.join(f'{k}={v}' for k, v in sorted(c.items())) + f'  [{self.runners[d].state}]')
        lines.append(f'pending total: {pend_after}')
        print('\n'.join(lines), flush=True)
