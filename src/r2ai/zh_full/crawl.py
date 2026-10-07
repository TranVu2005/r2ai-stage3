"""Authorized full-zh crawl: bounded ingestion, four lanes, immutable sample reuse."""
from __future__ import annotations

from r2ai.paths import CORPUS_FILE, CONFIG_DIR, RUNS_DIR, require_inputs
from .common import RUN, RAW, DB, STATE, guard, preflight, atomic_json, exclusive, keep_awake

import argparse
import asyncio
from collections import Counter, defaultdict, deque
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
import signal
import time
import uuid

from vicrawl.state import StateDB
from vicrawl.settings import Config
from vicrawl.urlnorm import group_urls, normalize_url, domain_of
from r2ai.zh_sample.crawl import ZhTuner
from vicrawl.tuner import MIN_RATE, start_rate
from urllib.parse import urlsplit

SAMPLE = RUNS_DIR / 'zh-sample'
LANES = (
    ('120ask.com',),
    ('cnkang.com',),
    ('msdmanuals.cn', 'test.pmphai.com', 'youlai.cn', 'care.39.net', 'zhongyibaodian.net', 'qihuangzhishu.com'),
    ('jb39.com', 'jbk.39.net', 'a-hospital.com', 'wujue.com', 'zydcd.com', 'health.people.com.cn', 'familydoctor.cn'),
)
DOMAINS = frozenset(d for lane in LANES for d in lane)
RATE_CUTS = RUN / 'rate_cuts.json'


class FullTuner(ZhTuner):
    """Transient outages wait at a reduced rate; access/challenge halts stay final."""
    RATE_CUTS = ('rate_down', 'zh_slow_cap', 'full_zh_outage_wait')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transient_errors = deque(maxlen=self._errs.maxlen)
        self.on_rate_cut = None

    def record(self, latency, http_status, error):
        transient = bool(error or http_status == 429 or (http_status is not None and http_status >= 500))
        self.transient_errors.append(transient)
        events = super().record(latency, http_status, error)
        if (self.halted and (self.halt_reason or '').startswith('error_rate')
                and sum(self.transient_errors) == sum(self._errs)):
            self.halted, self.halt_reason = False, None
            events = [e for e in events if e.get('event') != 'halt']
            if transient:
                self.rate = max(.25, self.rate / 2)
                self.paused_until = max(self.paused_until, self.clock() + 60)
                events.append({'event': 'full_zh_outage_wait', 'rate': self.rate, 'wait_seconds': 60})
        if self.on_rate_cut and any(e.get('event') in self.RATE_CUTS for e in events):
            self.on_rate_cut()
        return events


def session_start_rate(saved_rate, cap, had_rate_cut):
    """Resume at the learned rate; only a session that had to slow down restarts 25% lower."""
    if had_rate_cut or not saved_rate:
        return start_rate(saved_rate, cap)
    return max(MIN_RATE, min(cap, saved_rate))


def begin_session(tuner, saved_rate, had_rate_cut):
    tuner.rate = session_start_rate(saved_rate, tuner.cap, had_rate_cut)
    # The sample-run baseline goes stale; rate_up compares against latency measured this session.
    tuner.baseline_p95 = None


def mark_rate_cut(path, domain):
    # Written when the cut happens, so a crashed session still counts as having slowed down.
    cuts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not cuts.get(domain):
        atomic_json(path, {**cuts, domain: True})


def take_rate_cut(path, domain):
    cuts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if domain not in cuts:
        return False
    atomic_json(path, {d: v for d, v in cuts.items() if d != domain})
    return bool(cuts[domain])


def stop_requested(request, pid):
    """A persisted control request only applies to its original process."""
    return (isinstance(request, dict) and type(request.get('pid')) is int
            and request['pid'] > 0 and request['pid'] == pid)


def request_stop():
    runtime_path = RUN / 'runtime_state.json'
    require_inputs(runtime_path)
    state = json.loads(runtime_path.read_text('utf-8'))
    pid = state.get('pid')
    if not stop_requested(state, pid):
        raise ValueError('Runtime state lacks a valid crawler PID')
    value = {'pid': pid, 'requested_at': time.time()}
    atomic_json(RUN / 'stop_request.json', value)
    print(json.dumps(value), flush=True)


async def fetch_record(client, db, row, stop):
    """Requeue interrupted work; an explicit stop ends the worker normally."""
    try:
        return await client.fetch(row)
    except asyncio.CancelledError:
        db.requeue([row['url_norm']])
        if stop.is_set():
            return None
        raise


def direct_limits(geo=False):
    import httpx
    return httpx.Limits(max_connections=1 if geo else 2, max_keepalive_connections=0)


def same_domain_url(url, domain):
    parsed = urlsplit(url)
    return (parsed.scheme in ('http', 'https') and not parsed.username and not parsed.password
            and (parsed.hostname or '').lower().removeprefix('www.') == domain)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def new_groups(rows, domain, reused):
    groups = sorted((g for g in group_urls(rows) if g['domain'] == domain and g['url_norm'] not in reused),
                    key=lambda g: g['url_norm'])
    return [{**g, 'rank': i} for i, g in enumerate(groups)]


def domain_config(cfg, domain):
    original = cfg.for_domain(domain)
    return replace(original, max_conns=min(2, original.max_conns))


def network_runtime():
    # Separate module namespace: default sample runtime is never mutated.
    from r2ai.zh_sample import crawl as original
    spec = importlib.util.spec_from_file_location('r2ai.zh_sample._full_network', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN, module.RAW, module.DB = RUN, RAW, DB
    module.atomic_json, module.preflight = atomic_json, preflight
    module.ZhTuner = FullTuner
    return module


def snapshot_rates(events, domains, now):
    counts = defaultdict(Counter)
    for r in events:
        if r['domain'] not in domains or r['t0'] < domains[r['domain']]['start'] or r['t1'] > now:
            continue
        if r['kind'] == 'page':
            counts[r['domain']]['pages'] += 1
            counts[r['domain']]['errors'] += bool(r.get('error') or (r.get('http_status') or 0) >= 400)
    result = {}
    for d, state in domains.items():
        elapsed = max(0, now - state['start'])
        c = counts[d]
        rate = state['completed'] / elapsed if elapsed else 0
        result[d] = {**state, 'elapsed_seconds': elapsed,
                     'page_req_per_second': c['pages'] / elapsed if elapsed else 0,
                     'page_requests': c['pages'], 'http_errors': c['errors'],
                     'http_error_pct': 100 * c['errors'] / c['pages'] if c['pages'] else None,
                     'url_per_second': rate,
                     'eta_hours_estimated': state['remaining'] / rate / 3600 if rate else None}
    return result


async def supervise_workers(workers, stop):
    async def guarded_worker(work):
        try:
            return await work
        except BaseException:
            stop.set()
            raise
    results = await asyncio.gather(*(guarded_worker(work) for work in workers), return_exceptions=True)
    failure = next((r for r in results if isinstance(r, BaseException)), None)
    if failure:
        raise failure


async def supervise_lanes(lanes, heartbeat, stop):
    lane_job = asyncio.create_task(supervise_workers(lanes, stop))
    tick = asyncio.create_task(heartbeat)
    try:
        done, _ = await asyncio.wait((lane_job, tick), return_when=asyncio.FIRST_COMPLETED)
        if tick in done and not lane_job.done():
            error = tick.exception()
            stop.set()
            await lane_job
            if error:
                raise error
        else:
            await lane_job
            if tick.done() and not tick.cancelled() and tick.exception():
                raise tick.exception()
    except BaseException:
        stop.set()
        raise
    finally:
        for task in (lane_job, tick):
            if not task.done():
                task.cancel()
        await asyncio.gather(lane_job, tick, return_exceptions=True)


_INGEST = """INSERT INTO incoming(url_norm,url,doc_ids,domain,https_order,url_len,representative_id)
 VALUES(?,?,?,?,?,?,?) ON CONFLICT(url_norm) DO UPDATE SET
 doc_ids=CASE WHEN EXISTS(SELECT 1 FROM json_each(incoming.doc_ids) WHERE value=excluded.representative_id)
 THEN incoming.doc_ids ELSE json_insert(incoming.doc_ids,'$[#]',excluded.representative_id) END,
 url=CASE WHEN (excluded.https_order,excluded.url_len,excluded.representative_id)<
 (incoming.https_order,incoming.url_len,incoming.representative_id) THEN excluded.url ELSE incoming.url END,
 https_order=MIN(incoming.https_order,excluded.https_order),
 url_len=CASE WHEN (excluded.https_order,excluded.url_len,excluded.representative_id)<
 (incoming.https_order,incoming.url_len,incoming.representative_id) THEN excluded.url_len ELSE incoming.url_len END,
 representative_id=CASE WHEN (excluded.https_order,excluded.url_len,excluded.representative_id)<
 (incoming.https_order,incoming.url_len,incoming.representative_id) THEN excluded.representative_id ELSE incoming.representative_id END"""


def prepare():
    import pyarrow.parquet as pq
    preflight()
    require_inputs(CORPUS_FILE, SAMPLE / 'sample.json', SAMPLE / 'sample_manifest.json', SAMPLE / 'domain_metrics.json')
    with exclusive('prepare'):
        sample_manifest = json.loads((SAMPLE / 'sample_manifest.json').read_text('utf-8'))
        sample_sha = digest(SAMPLE / 'sample.json')
        if sample_sha != sample_manifest['sample_sha256']:
            raise ValueError('Sample manifest hash mismatch')
        metrics = json.loads((SAMPLE / 'domain_metrics.json').read_text('utf-8'))
        allowed = {r['domain'] for r in metrics if r.get('chunks', 0) > 0 and not r['excluded']}
        if allowed != DOMAINS:
            raise ValueError('Full crawl whitelist must equal the 15 sampled accessible domains')
        marker = RUN / 'prepare_complete.json'
        if marker.exists():
            old = json.loads(marker.read_text('utf-8'))
            if old['sample_sha256'] != sample_sha or old['corpus_sha256'] != digest(CORPUS_FILE):
                raise ValueError('Full crawl inputs changed; refusing resume')
            require_inputs(DB)
            print('Prepared DB resume: immutable inputs verified', flush=True)
            return old
        reused = {g['url_norm'] for g in json.loads((SAMPLE / 'sample.json').read_text('utf-8')) if g['domain'] in DOMAINS}
        corpus_sha = digest(CORPUS_FILE)
        tmp_dir = guard(STATE / 'tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        db = StateDB(guard(DB))
        try:
            db.conn.execute('PRAGMA cache_size=-16384')
            db.conn.execute('PRAGMA mmap_size=0')
            db.conn.execute('PRAGMA temp_store=FILE')
            db.conn.execute("PRAGMA temp_store_directory='" + str(tmp_dir).replace("'", "''") + "'")
            db.conn.execute('''CREATE TABLE IF NOT EXISTS incoming(url_norm TEXT PRIMARY KEY,url TEXT,
                doc_ids TEXT,domain TEXT,https_order INTEGER,url_len INTEGER,representative_id INTEGER)''')
            batches = seen = 0
            for batch in pq.ParquetFile(CORPUS_FILE).iter_batches(batch_size=10_000, columns=['id', 'url'], use_threads=False):
                payload = []
                cols = batch.to_pydict()
                for doc, url in zip(cols['id'], cols['url']):
                    domain = domain_of(url)
                    if domain not in DOMAINS:
                        continue
                    norm = normalize_url(url)
                    if norm in reused:
                        continue
                    payload.append((norm, url, json.dumps([doc]), domain, 0 if url.lower().startswith('https://') else 1, len(url), doc))
                with db._tx():
                    db.conn.executemany(_INGEST, payload)
                seen += len(payload)
                batches += 1
                if batches % 25 == 0:
                    print(f'prepare batch={batches}, eligible corpus rows={seen}', flush=True)
            expected = {r['domain']: r['n_unique'] - r['n_sample'] for r in metrics if r['domain'] in DOMAINS}
            actual = dict(db.conn.execute('SELECT domain,count(*) FROM incoming GROUP BY domain'))
            if any(actual.get(d, 0) != n for d, n in expected.items()):
                raise ValueError(f'Full URL normalization inventory differs: expected={expected}, actual={actual}')
            # External ordering/alias aggregation remains bounded by SQLite's file-backed temp store.
            db.conn.execute('CREATE INDEX IF NOT EXISTS incoming_domain_url ON incoming(domain,url_norm)')
            with db._tx():
                db.conn.execute('''INSERT OR IGNORE INTO urls(url_norm,url,doc_ids,domain,rank)
                    SELECT url_norm,url,(SELECT json_group_array(value) FROM
                      (SELECT value FROM json_each(incoming.doc_ids) ORDER BY value)),domain,
                      ROW_NUMBER() OVER(PARTITION BY domain ORDER BY url_norm)-1 FROM incoming''')
            # Carry learned pacing without importing sample URL states or modifying sample DB.
            import sqlite3
            from r2ai.paths import STATE_DIR
            sample_db = STATE_DIR / 'crawl_zh_sample.db'
            conn = sqlite3.connect(sample_db.as_uri() + '?mode=ro', uri=True)
            try:
                conn.row_factory = sqlite3.Row
                for row in conn.execute('SELECT * FROM domains'):
                    if row['domain'] in DOMAINS:
                        db.upsert_domain(row['domain'], rate=row['rate'], conns=min(2, row['conns'] or 2),
                                         baseline_p95=row['baseline_p95'], crawl_delay=row['crawl_delay'],
                                         approved=1, state='ready', updated_at=time.time())
            finally:
                conn.close()
            value = {'corpus_sha256': corpus_sha, 'sample_sha256': sample_sha, 'sample_reused_url_groups': len(reused),
                     'new_url_groups': sum(expected.values()), 'domains': expected, 'seed': 42,
                     'raw_sample_read_only': str(SAMPLE), 'new_raw_root': str(RAW), 'new_db': str(DB),
                     'prepared_at': time.time(), 'lanes': LANES, 'config': str(CONFIG_DIR / 'domains-zh-sample.yaml')}
            atomic_json(marker, value)
            db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            print(json.dumps(value, ensure_ascii=True), flush=True)
            return value
        finally:
            db.conn.close()


def write_raw(domain, records):
    import zstandard
    directory = guard(RAW / domain)
    directory.mkdir(parents=True, exist_ok=True)
    name = f'{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex}.jsonl.zst'
    final, tmp = guard(directory / name), guard(directory / (name + '.tmp'))
    payload = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in records).encode('utf-8')
    with tmp.open('wb') as f:
        f.write(zstandard.ZstdCompressor(level=6).compress(payload))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return final


async def run():
    import httpx
    import psutil
    preflight()
    require_inputs(DB, RUN / 'prepare_complete.json', CONFIG_DIR / 'domains-zh-sample.yaml')
    runtime = network_runtime()
    stop, global_slots = asyncio.Event(), asyncio.Semaphore(8)
    loop = asyncio.get_running_loop()
    previous = {}
    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    cfg = Config.load(CONFIG_DIR / 'domains-zh-sample.yaml')
    db = StateDB(guard(DB))
    db.recover()
    starts, completed = {}, {}
    counts = defaultdict(Counter)
    audits = guard(RUN / 'requests.jsonl').open('a', encoding='utf-8')

    class Audit:
        def write(self, line):
            audits.write(line)
            row = json.loads(line)
            counts[row['domain']]['pages'] += row['kind'] == 'page'
            counts[row['domain']]['errors'] += row['kind'] == 'page' and bool(row['error'] or (row['http_status'] or 0) >= 400)
        def flush(self):
            audits.flush()

    class Network:
        def __init__(self):
            self.lock, self.country, self.checked_at = asyncio.Lock(), None, 0
        async def check(self):
            async with self.lock:
                if time.time() - self.checked_at >= 60:
                    self.country = None
                    async with httpx.AsyncClient(trust_env=False, timeout=15, limits=direct_limits(True)) as client:
                        for url in ('https://ipinfo.io/json', 'https://api.country.is/'):
                            try:
                                response = await client.get(url)
                                if response.status_code == 200:
                                    self.country = response.json().get('country')
                                    if self.country:
                                        break
                            except (httpx.HTTPError, ValueError):
                                pass
                    self.checked_at = time.time()
                    atomic_json(RUN / 'egress.json', {'country': self.country, 'checked_at': self.checked_at,
                                                    'direct': True, 'proxy': False})
                if self.country is None:
                    raise RuntimeError('EGRESS_UNAVAILABLE: wait without page requests')
                if self.country != 'VN':
                    raise RuntimeError(f'EGRESS country={self.country}; need VN')

    net = Network()

    class Client(runtime.PoliteClient):
        async def allowed(self, url):
            if not same_domain_url(url, self.domain):
                return False, 'cross_domain_redirect_not_allowed'
            return await super().allowed(url)

        async def hop(self, url, kind):
            async with global_slots:
                return await super().hop(url, kind)

    def progress():
        now = time.time()
        result = {}
        for domain, t0 in starts.items():
            remaining = db.conn.execute("SELECT count(*) FROM urls WHERE domain=? AND (status IN ('pending','in_progress') OR next_try_at IS NOT NULL)", (domain,)).fetchone()[0]
            total = db.conn.execute('SELECT count(*) FROM urls WHERE domain=?', (domain,)).fetchone()[0]
            done = total - remaining - completed[domain]
            elapsed = max(now - t0, 1e-9)
            rate = max(done, 0) / elapsed
            c = counts[domain]
            result[domain] = {'start': t0, 'elapsed_seconds': elapsed, 'total_new_urls': total,
                              'completed_this_session': done, 'remaining': remaining,
                              'page_requests': c['pages'], 'http_errors': c['errors'],
                              'page_req_per_second': c['pages'] / elapsed,
                              'url_per_second': rate, 'http_error_pct': 100*c['errors']/c['pages'] if c['pages'] else None,
                              'eta_hours_estimated': remaining / rate / 3600 if rate else None,
                              'state': (db.get_domain(domain) or {}).get('state')}
        proc = psutil.Process()
        mem = proc.memory_info()
        value = {'pid': os.getpid(), 'updated_at': now, 'country': net.country, 'domains': result,
                 'rss_bytes': mem.rss, 'private_bytes': getattr(mem, 'private', None),
                 'available_memory_bytes': psutil.virtual_memory().available,
                 'no_extract': True, 'max_active_domains': 4, 'max_connections_per_domain': 2,
                 'max_connections_total': 8}
        atomic_json(RUN / 'runtime_state.json', value)
        return value

    async def wait(seconds):
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.01, seconds))
        except asyncio.TimeoutError:
            pass

    async def domain_run(domain):
        dc = domain_config(cfg, domain)
        client = Client(domain, dc, net, db, stop, Audit())
        begin_session(client.tuner, (db.get_domain(domain) or {}).get('rate'), take_rate_cut(RATE_CUTS, domain))
        client.tuner.on_rate_cut = lambda: mark_rate_cut(RATE_CUTS, domain)
        # No idle keep-alive sockets: active request slots bound total sockets,
        # including the geo check performed before a page within the same slot.
        await client.client.aclose()
        client.client = httpx.AsyncClient(trust_env=False, http2=True, follow_redirects=False,
            timeout=httpx.Timeout(25, connect=10), limits=direct_limits(False),
            headers={'User-Agent': runtime.CHROME_UA, 'Accept-Language': 'zh-CN,zh;q=0.9',
                     'Accept': 'text/html,application/xhtml+xml,*/*;q=0.5'})
        starts[domain] = time.time()
        remaining_start = db.conn.execute("SELECT count(*) FROM urls WHERE domain=? AND (status IN ('pending','in_progress') OR next_try_at IS NOT NULL)", (domain,)).fetchone()[0]
        total = db.conn.execute('SELECT count(*) FROM urls WHERE domain=?', (domain,)).fetchone()[0]
        completed[domain] = total - remaining_start
        buffer, robots_retry = [], 0
        db.upsert_domain(domain, state='active', updated_at=time.time())
        print(f'lane start {domain}: {remaining_start} pending, max_conns=2', flush=True)

        def flush():
            if buffer:
                shard = write_raw(domain, buffer)
                db.commit_results([{**r, 'shard': shard.name} for r in buffer])
                buffer.clear()

        async def worker():
            nonlocal robots_retry
            while not stop.is_set() and not client.tuner.halted and not robots_retry:
                rows = db.next_batch(domain, 1, time.time())
                if not rows:
                    return
                row = rows[0]
                try:
                    rec = await fetch_record(client, db, row, stop)
                except RuntimeError as e:
                    db.requeue([row['url_norm']])
                    if str(e).startswith('EGRESS_UNAVAILABLE'):
                        await wait(60)
                        continue
                    stop.set()
                    raise
                if rec is None:
                    return
                if rec['reason'] == 'robots_unavailable':
                    robots_retry = max(time.time()+1800, max((r.retry_at or 0 for r in client.robots.values() if r.state=='UNAVAILABLE'), default=0))
                    runtime.defer_robots_row(db, row, robots_retry)
                    return
                buffer.append(rec)
                if len(buffer) >= 10:
                    flush()

        try:
            while not stop.is_set():
                await supervise_workers([worker() for _ in range(dc.max_conns)], stop)
                flush()
                if client.tuner.halted:
                    break
                retry_at = robots_retry or runtime.next_retry_at(db, domain, None)
                if retry_at is None:
                    break
                db.upsert_domain(domain, state='waiting_robots' if robots_retry else 'waiting_retry', updated_at=time.time())
                await wait(retry_at - time.time())
                client.robots = {o:r for o,r in client.robots.items() if r.state!='UNAVAILABLE'}
                robots_retry = 0
                db.upsert_domain(domain, state='active', updated_at=time.time())
        finally:
            flush()
            db.upsert_domain(domain, rate=client.tuner.rate, conns=dc.max_conns,
                baseline_p95=client.tuner.baseline_p95, crawl_delay=client.tuner.crawl_delay,
                state='halted' if client.tuner.halted else ('stopped' if stop.is_set() else 'finished'),
                halt_reason=client.tuner.halt_reason, updated_at=time.time())
            await client.client.aclose()
            print(f'lane finish {domain}, state={(db.get_domain(domain) or {}).get("state")}', flush=True)

    async def lane(domains):
        for domain in domains:
            if stop.is_set():
                return
            await domain_run(domain)

    async def monitor():
        while not stop.is_set():
            request_path = RUN / 'stop_request.json'
            if request_path.exists() and stop_requested(json.loads(request_path.read_text('utf-8')), os.getpid()):
                stop.set()
                return
            value = progress()
            print(f'heartbeat pid={os.getpid()}, rss={value["rss_bytes"]}, country={net.country}', flush=True)
            await wait(60)

    try:
        while not stop.is_set():
            try:
                await net.check()
                break
            except RuntimeError as e:
                if not str(e).startswith('EGRESS_UNAVAILABLE'):
                    raise
                print(str(e), flush=True)
                await wait(60)
        atomic_json(RUN / 'process.json', {'pid': os.getpid(), 'started_at': time.time(), 'lanes': LANES,
                                        'raw': str(RAW), 'db': str(DB), 'config_sha256': digest(CONFIG_DIR / 'domains-zh-sample.yaml')})
        await supervise_lanes([lane(domains) for domains in LANES], monitor(), stop)
        return 130 if stop.is_set() else 0
    finally:
        stop.set()
        try:
            db.recover()
            progress()
            db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        finally:
            db.conn.close()
            audits.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare', 'run', 'stop'])
    args = parser.parse_args(argv)
    if args.step == 'stop':
        request_stop()
        return 0
    if args.step == 'prepare':
        prepare()
        return 0
    with exclusive('crawl'), keep_awake():
        code = None
        try:
            code = asyncio.run(run())
            return code
        finally:
            # The watchdog reads this to tell "all lanes finished" (0) from stop/crash; a hard kill writes nothing.
            atomic_json(RUN / 'last_exit.json', {'pid': os.getpid(), 'code': code, 'at': time.time()})


if __name__ == '__main__':
    raise SystemExit(main())
