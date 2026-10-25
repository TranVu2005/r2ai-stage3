#!/usr/bin/env python3
"""S37 standalone 120ask crawler (Python 3 + requests only; no dependency on the r2ai repo).

    crawler.py run    --urls urls.tsv --out data [--round 1] [--sample 50] [--limit N]
    crawler.py status --out data
    crawler.py pack   --out data [--batch-shards 5] [--final] [--partial]

Politeness: one IP, one request at a time, start-to-start gap 1.0 s + uniform(0, 0.3) s, descriptive UA, no proxy
(trust_env off), no IP rotation, no browser emulation. Redirects are followed by hand (max 3 hops) and only inside the
allowed host suffix. robots.txt is not read (D21).

Output under --out:
  shards/r<round>-<idx:05d>.jsonl.gz  one gzip member per record (appended, so a crash never rewrites old data);
                                      shard idx covers queue lines idx*2000 .. idx*2000+1999
  log.jsonl                           one line per finished URL: doc_id, url, status, http, bytes, sec, attempts, suspect,
                                      round, shard, off (shard size after the record), ts
  STOP.json                           written when a block/challenge is seen; the run refuses to start while it exists
  queue-r<round>.tsv                  round >= 2 only: the URLs whose latest log status is `failed`, frozen at round start

status values: ok | gone (404/410) | failed (retries exhausted, retry in a later round) | offsite (redirect leaves the
allowed host) | http_error (other 4xx/5xx without retry).  403, 429, 401, 451 and challenge pages stop the whole run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import signal
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

UA = 'r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)'
SHARD_URLS = 2000
MIN_GAP, JITTER = 1.0, 0.3
BACKOFFS = (30, 120, 600)
RETRY_HTTP = {500, 502, 503, 504, 520, 521, 522, 523, 524}
STOP_HTTP = {401, 403, 429, 451}
ALLOWED_SUFFIX = '120ask.com'
CONNECT_TIMEOUT, READ_TIMEOUT = 15, 45
MAX_HOPS = 3
COOLDOWN_AFTER, COOLDOWN_S, MAX_COOLDOWNS = 10, 1800, 3
CHALLENGE = re.compile(r'cf-chl|challenge-platform|Just a moment|cf-browser-verification|Attention Required|captcha|'
                       r'verify you are human|验证码|安全验证|人机验证|访问验证|滑动验证', re.I)
ASK, ANS, H1 = re.compile(r'b_askcont'), re.compile(r'b_ans'), re.compile(r'<h1[\s>]', re.I)


class Stop(Exception):
    pass


def classify_page(html: str) -> tuple[bool, bool]:
    """-> (challenge, suspect). A challenge needs a marker and no question block; suspect = not a Q&A page."""
    has_qa = bool(ASK.search(html)) and bool(ANS.search(html)) and bool(H1.search(html))
    challenge = (not ASK.search(html)) and bool(CHALLENGE.search(html))
    return challenge, not has_qa


def decode(content: bytes) -> tuple[str, str]:
    for enc in ('utf-8', 'gb18030'):
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return content.decode('utf-8', 'replace'), 'utf-8-replace'


def host_ok(url: str, suffix: str) -> bool:
    host = (urlsplit(url).hostname or '').lower()
    return host == suffix or host.endswith('.' + suffix)


def read_queue(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text('utf-8').splitlines():
        if line.strip():
            parts = line.split('\t')
            rows.append((parts[0], parts[1]))
    return rows


def read_log(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text('utf-8').splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                break           # torn last line after a crash: ignore it, the URL is fetched again
    return out


def shard_path(out: Path, rnd: int, idx: int) -> Path:
    return out / 'shards' / f'r{rnd}-{idx:05d}.jsonl.gz'


def read_shard(path: Path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def repair_shards(out: Path, log: list[dict]):
    """Cut every shard back to the last size confirmed by the log (drops a torn or unlogged trailing member)."""
    last: dict[str, int] = {}
    for r in log:
        if r.get('shard'):
            last[r['shard']] = r['off']
    for p in (out / 'shards').glob('*.jsonl.gz'):
        keep = last.get(p.name, 0)
        if p.stat().st_size != keep:
            with open(p, 'r+b') as f:
                f.truncate(keep)


class Crawler:
    def __init__(self, out: Path, rnd: int = 1, session=None, suffix: str = ALLOWED_SUFFIX, min_gap: float = MIN_GAP,
                 jitter: float = JITTER, backoffs=BACKOFFS, upgrade_https: bool = True, sleep=time.sleep, clock=time.monotonic,
                 cooldown_s: int = COOLDOWN_S):
        import requests
        self.out, self.rnd, self.suffix = Path(out), rnd, suffix
        self.min_gap, self.jitter, self.backoffs, self.upgrade = min_gap, jitter, tuple(backoffs), upgrade_https
        self.sleep, self.clock, self.cooldown_s = sleep, clock, cooldown_s
        if session is None:
            session = requests.Session()
            session.trust_env = False          # no proxy from the environment
            session.headers.update({'User-Agent': UA})
        self.s = session
        self.last_start = None
        self.halt = False
        self.consec_failed = 0
        self.cooldowns = 0
        (self.out / 'shards').mkdir(parents=True, exist_ok=True)

    def _pace(self):
        if self.last_start is not None:
            wait = self.min_gap + random.uniform(0, self.jitter) - (self.clock() - self.last_start)
            if wait > 0:
                self.sleep(wait)
        self.last_start = self.clock()

    def _get(self, url: str):
        """One logical GET with manual same-host redirects -> (response|None, error|None, offsite_location|None)."""
        import requests
        cur = url
        for _ in range(MAX_HOPS + 1):
            self._pace()
            try:
                r = self.s.get(cur, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=False)
            except requests.RequestException as e:
                return None, type(e).__name__, None
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get('Location'):
                nxt = urljoin(cur, r.headers['Location'])
                if not host_ok(nxt, self.suffix):
                    return r, None, nxt
                cur = nxt
                continue
            return r, None, None
        return None, 'TooManyRedirects', None

    def fetch(self, doc_id: str, url: str) -> tuple[dict, dict | None]:
        """-> (log entry without shard info, shard record or None). Raises Stop on a block or challenge."""
        if self.upgrade and url.startswith('http://'):
            url = 'https://' + url[len('http://'):]
        t0 = time.time()
        attempts, last_err, http = 0, None, None
        for attempt in range(len(self.backoffs) + 1):
            attempts += 1
            r, err, offsite = self._get(url)
            if offsite:
                return self._entry(doc_id, url, 'offsite', r.status_code, 0, t0, attempts, False, err=offsite), None
            if r is None:
                last_err, http = err, None
            else:
                http = r.status_code
                if http in STOP_HTTP:
                    raise Stop({'reason': f'http_{http}', 'url': url, 'doc_id': doc_id})
                if http == 200:
                    html, enc = decode(r.content)
                    challenge, suspect = classify_page(html)
                    if challenge:
                        raise Stop({'reason': 'challenge_page', 'url': url, 'doc_id': doc_id})
                    rec = {'doc_id': doc_id, 'url': url, 'final_url': r.url, 'http': 200, 'fetched_at': time.time(),
                           'encoding': enc, 'html': html}
                    return self._entry(doc_id, url, 'ok', 200, len(r.content), t0, attempts, suspect), rec
                if http in (404, 410):
                    return self._entry(doc_id, url, 'gone', http, len(r.content), t0, attempts, False), None
                if http not in RETRY_HTTP:
                    return self._entry(doc_id, url, 'http_error', http, len(r.content), t0, attempts, False), None
                last_err = f'http_{http}'
            if attempt < len(self.backoffs):
                self.sleep(self.backoffs[attempt])
        return self._entry(doc_id, url, 'failed', http, 0, t0, attempts, False, err=last_err), None

    def _entry(self, doc_id, url, status, http, nbytes, t0, attempts, suspect, err=None) -> dict:
        e = {'doc_id': doc_id, 'url': url, 'status': status, 'http': http, 'bytes': nbytes, 'sec': round(time.time() - t0, 2),
             'attempts': attempts, 'suspect': suspect, 'round': self.rnd}
        if err:
            e['err'] = err
        return e

    def run(self, queue: list[tuple[str, str]], limit: int | None = None) -> dict:
        if (self.out / 'STOP.json').exists():
            raise SystemExit(f'{self.out}/STOP.json exists: read it, decide, delete it to continue')
        log_path = self.out / 'log.jsonl'
        log = read_log(log_path)
        repair_shards(self.out, log)
        done = {r['doc_id'] for r in log if r.get('round') == self.rnd}
        todo = [(i, d, u) for i, (d, u) in enumerate(queue) if d not in done]
        if limit is not None:
            todo = todo[:limit]
        lf = open(log_path, 'a', encoding='utf-8')
        handles: dict[int, object] = {}
        n = 0
        try:
            for i, doc_id, url in todo:
                if self.halt:
                    break
                try:
                    entry, rec = self.fetch(doc_id, url)
                except Stop as s:
                    info = {**s.args[0], 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'done_before_stop': len(done) + n}
                    (self.out / 'STOP.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
                    print('STOP', info, file=sys.stderr)
                    return {'stopped': info, 'fetched': n}
                idx = i // SHARD_URLS
                p = shard_path(self.out, self.rnd, idx)
                entry['shard'], entry['ts'] = p.name, int(time.time())
                if rec is not None:
                    member = gzip.compress((json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'), 6)
                    with open(p, 'ab') as f:
                        f.write(member)
                        f.flush()
                        os.fsync(f.fileno())
                else:
                    p.touch()
                entry['off'] = p.stat().st_size
                lf.write(json.dumps(entry, ensure_ascii=False) + '\n')
                lf.flush()
                n += 1
                self._breaker(entry['status'] == 'failed')
                if n % 50 == 0:
                    print(f'{n}/{len(todo)} last={entry["status"]} http={entry["http"]}', flush=True)
        finally:
            lf.close()
        return {'fetched': n, 'remaining': len(todo) - n, 'halted': self.halt}

    def _breaker(self, failed: bool):
        """Origin down (many failures in a row): pause 30 min, give up after 3 pauses in a row (resume later by hand)."""
        self.consec_failed = self.consec_failed + 1 if failed else 0
        if self.consec_failed >= COOLDOWN_AFTER:
            self.cooldowns += 1
            self.consec_failed = 0
            if self.cooldowns > MAX_COOLDOWNS:
                self.halt = True
                return
            self.sleep(self.cooldown_s)
        elif not failed:
            self.cooldowns = 0


def sample_queue(queue, n):
    step = max(1, len(queue) // n)
    return queue[::step][:n]


def retry_queue(out: Path, rnd: int) -> list[tuple[str, str]]:
    path = out / f'queue-r{rnd}.tsv'
    if path.exists():
        return read_queue(path)
    latest: dict[str, dict] = {}
    for r in read_log(out / 'log.jsonl'):
        latest[r['doc_id']] = r
    rows = [(d, r['url']) for d, r in latest.items() if r['status'] == 'failed']
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(f'{d}\t{u}\n' for d, u in rows), encoding='utf-8')
    os.replace(tmp, path)
    return rows


def cmd_run(a) -> int:
    out = Path(a.out)
    if a.sample:
        queue = sample_queue(read_queue(Path(a.urls)), a.sample)
    elif a.round > 1:
        out.mkdir(parents=True, exist_ok=True)
        queue = retry_queue(out, a.round)
    else:
        queue = read_queue(Path(a.urls))
    c = Crawler(out, a.round, suffix=a.suffix, upgrade_https=not a.no_https_upgrade, min_gap=a.min_gap)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(c, 'halt', True))
    res = c.run(queue, a.limit)
    print(json.dumps(res, ensure_ascii=False))
    if res.get('stopped'):
        return 3            # block / challenge: run.sh must not restart
    return 4 if c.cooldowns > MAX_COOLDOWNS else 0     # 4 = origin kept failing, give up (rerun later)


def summarize(out: Path) -> dict:
    log = read_log(out / 'log.jsonl')
    latest: dict[str, dict] = {}
    for r in log:
        latest[r['doc_id']] = r
    by = {}
    for r in latest.values():
        by[r['status']] = by.get(r['status'], 0) + 1
    ok = [r for r in latest.values() if r['status'] == 'ok']
    suspect = sum(1 for r in ok if r['suspect'])
    recent = log[-300:]
    rate = None
    if len(recent) >= 2 and recent[-1]['ts'] > recent[0]['ts']:
        rate = (len(recent) - 1) / (recent[-1]['ts'] - recent[0]['ts'])
    return {'urls_logged': len(latest), 'by_status': by, 'ok': len(ok), 'suspect': suspect,
            'ok_rate': round(len(ok) / len(latest), 4) if latest else None,
            'sec_per_url_mean': round(sum(r['sec'] for r in log) / len(log), 2) if log else None,
            'rate_per_s_recent': round(rate, 3) if rate else None, 'bytes_ok': sum(r['bytes'] for r in ok),
            'stop': json.loads((out / 'STOP.json').read_text('utf-8')) if (out / 'STOP.json').exists() else None}


def cmd_status(a) -> int:
    out = Path(a.out)
    s = summarize(out)
    total = None
    if a.urls and Path(a.urls).exists():
        total = len(read_queue(Path(a.urls)))
        s['total_urls'] = total
        left = total - s['urls_logged']
        s['remaining'] = left
        if s['rate_per_s_recent']:
            s['eta_hours'] = round(left / s['rate_per_s_recent'] / 3600, 1)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    return 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def complete_shards(out: Path, urls: Path) -> list[Path]:
    """Shards whose every queue line has a log line in that round (a shard is closed once its last URL is logged)."""
    per: dict[str, int] = {}
    for r in read_log(out / 'log.jsonl'):
        per[r['shard']] = per.get(r['shard'], 0) + 1
    totals: dict[int, int] = {}
    res = []
    for name, n in sorted(per.items()):
        rnd, idx = (int(x) for x in re.match(r'r(\d+)-(\d+)\.jsonl\.gz$', name).groups())
        if rnd not in totals:
            totals[rnd] = len(read_queue(urls if rnd == 1 else out / f'queue-r{rnd}.tsv'))
        if n >= min(SHARD_URLS, totals[rnd] - idx * SHARD_URLS):
            res.append(out / 'shards' / name)
    return res


def partial_shards(out: Path, urls: Path, packed: dict) -> list[Path]:
    """Shards with logged records that are not complete (crawl stopped inside them) and not packed with the same row count yet."""
    complete = {p.name for p in complete_shards(out, urls)}
    rows: dict[str, int] = {}
    for r in read_log(out / 'log.jsonl'):
        rows[r['shard']] = rows.get(r['shard'], 0) + 1
    return [out / 'shards' / n for n, k in sorted(rows.items())
            if n not in complete and n not in packed['shards'] and packed.get('partial', {}).get(n) != k]


def cmd_pack(a) -> int:
    out = Path(a.out)
    packed_path = out / 'packed.json'
    packed = json.loads(packed_path.read_text('utf-8')) if packed_path.exists() else {'shards': [], 'batches': []}
    ready = [p for p in complete_shards(out, Path(a.urls)) if p.name not in packed['shards']]
    groups = [ready[i:i + a.batch_shards] for i in range(0, len(ready), a.batch_shards)]
    if groups and len(groups[-1]) < a.batch_shards and not a.final:
        groups.pop()
    if getattr(a, 'partial', False):          # crawl stopped for good: also ship the records of an unfinished shard (own batch)
        repair_shards(out, read_log(out / 'log.jsonl'))      # cut a torn trailing member first
        extra = partial_shards(out, Path(a.urls), packed)
        if extra:
            groups.append(extra)
    (out / 'batches').mkdir(exist_ok=True)
    log = read_log(out / 'log.jsonl')
    extra_set = set(extra) if getattr(a, 'partial', False) else set()
    for g in groups:
        names = [p.name for p in g]
        n = len(packed['batches']) + 1
        tar_path = out / 'batches' / f'batch-{n:04d}.tar'
        rows = [r for r in log if r['shard'] in names]
        manifest = {'batch': n, 'shards': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in g],
                    'urls': len(rows), 'ok': sum(1 for r in rows if r['status'] == 'ok'),
                    'suspect': sum(1 for r in rows if r['status'] == 'ok' and r['suspect']),
                    'failed': sum(1 for r in rows if r['status'] == 'failed')}
        tmp = tar_path.with_name(tar_path.name + '.tmp')
        with tarfile.open(tmp, 'w') as t:
            for p in g:
                t.add(p, arcname=f'shards/{p.name}')
            blob = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
            ti = tarfile.TarInfo('log.jsonl')
            ti.size = len(blob)
            t.addfile(ti, io.BytesIO(blob))
            mb = json.dumps(manifest, indent=1).encode('utf-8')
            ti = tarfile.TarInfo('MANIFEST.json')
            ti.size = len(mb)
            t.addfile(ti, io.BytesIO(mb))
        os.replace(tmp, tar_path)
        (out / 'batches' / f'batch-{n:04d}.tar.sha256').write_text(f'{sha256_file(tar_path)}  {tar_path.name}\n')
        for p in g:
            if p in extra_set:
                packed.setdefault('partial', {})[p.name] = sum(1 for r in log if r['shard'] == p.name)
            else:
                packed['shards'].append(p.name)
        packed['batches'].append({'tar': tar_path.name, **{k: manifest[k] for k in ('urls', 'ok', 'suspect', 'failed')}})
        packed_path.write_text(json.dumps(packed, indent=1), encoding='utf-8')
        print(json.dumps({'packed': tar_path.name, 'shards': len(g), 'urls': manifest['urls'], 'ok': manifest['ok'],
                          'suspect': manifest['suspect'], 'failed': manifest['failed']}))
    if not groups:
        print('no complete batch to pack (use --final for a partial last batch)')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--urls', default='urls.tsv')
    r.add_argument('--out', default='data')
    r.add_argument('--round', type=int, default=1)
    r.add_argument('--sample', type=int, default=0, help='pilot: N URLs spread evenly over the queue')
    r.add_argument('--limit', type=int)
    r.add_argument('--suffix', default=ALLOWED_SUFFIX)
    r.add_argument('--min-gap', type=float, default=MIN_GAP)
    r.add_argument('--no-https-upgrade', action='store_true')
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser('status')
    s.add_argument('--out', default='data')
    s.add_argument('--urls', default='urls.tsv')
    s.set_defaults(fn=cmd_status)
    p = sub.add_parser('pack')
    p.add_argument('--out', default='data')
    p.add_argument('--urls', default='urls.tsv')
    p.add_argument('--batch-shards', type=int, default=5)
    p.add_argument('--final', action='store_true')
    p.add_argument('--partial', action='store_true',
                   help='crawl stopped for good: also pack the logged records of an unfinished shard as its own batch')
    p.set_defaults(fn=cmd_pack)
    a = ap.parse_args(argv)
    if getattr(a, 'min_gap', 1.0) < 1.0:
        raise SystemExit('min-gap below 1.0 s is not allowed')
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
