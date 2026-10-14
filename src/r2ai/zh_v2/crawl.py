"""Crawl of the v2 zh hosts that were never in the zh_full plan (S11, D13): same machinery, separate state.

The engine is r2ai.zh_full.crawl loaded into its own module namespace (the way zh_full loads the zh_sample runtime) with
RUN/RAW/DB/STATE/LANES/DOMAINS replaced, so the running zh-full crawler, its watchdog, its DB and its raw root are untouched.
Politeness is the tuner default group (rate_cap 4 req/s, 2 connections, start 1 req/s, adaptive down on 429/5xx), robots.txt
interpreted by vicrawl.robots exactly as for the other zh hosts, VN egress required, no redirect to another host. Hosts whose
robots.txt disallows (zysjonline.com, baidu.com) are not loaded: they go to the Common Crawl queue.
Same-family hosts share one lane (39.net, familydoctor) so a family is paced as one site.

    python -m r2ai.zh_v2.crawl prepare
    python -m r2ai.zh_v2.crawl run [--deadline ISO8601+TZ]
    python -m r2ai.zh_v2.crawl stop
    python -m r2ai.zh_v2.crawl status
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, STATE_DIR, WORK_DATA_DIR, OLD_ROOT, assert_writable, resolve_path
from vicrawl.state import StateDB
from vicrawl.urlnorm import group_urls

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

RUN = RUNS_DIR / 'zh-v2-crawl'
RAW = WORK_DATA_DIR / 'raw_zh_v2'
STATE = STATE_DIR / 'zh_v2'
DB = STATE / 'crawl.db'
IDS = WORK_DATA_DIR / 'corpus_v2_ids.parquet'
DEFAULT_DEADLINE = '2026-10-20T23:59:00+07:00'

SUBS_39 = ('heart.39.net', 'fk.39.net', 'nk.39.net', 'cancer.39.net', 'tnb.39.net', 'woman.39.net', 'food.39.net', 'ek.39.net',
           'man.39.net', 'shen.39.net', 'gan.39.net', 'wei.39.net', 'sj.39.net', 'fitness.39.net', 'gk.39.net', 'pf.39.net',
           'gc.39.net', 'byby.39.net')
LANES = (
    ('ask.39.net',),
    ('familydoctor.com.cn', 'baidianfeng.familydoctor.com.cn', 'qy.familydoctor.com.cn', 'familydoctor.cn'),
    ('baby.39.net', *SUBS_39),
    # robots.txt answered 5xx at probe time (site down): a lane of their own so they never hold the others up
    ('article.iiyi.com', 'bingli.iiyi.com', 'ask.familydoctor.com.cn'),
)
DOMAINS = frozenset(d for lane in LANES for d in lane)
# D13: the 39.net subdomains answer /robots.txt with a redirect to the 404 page of www.39.net. Under RFC 9309 a 4xx robots.txt
# (also reached through a redirect) means no restrictions, so the redirect inside the 39.net family is followed instead of failing
# closed. Every other cross-host robots redirect still fails closed. Pacing stays the tuner default group.
ROBOTS_REDIRECT_SUFFIX = ('.39.net',)


def guard(path) -> Path:
    p = resolve_path(path)
    if not any(p.is_relative_to(resolve_path(r)) for r in (RUN, RAW, STATE)):
        raise ValueError(f'Not a zh-v2 crawl output: {p}')
    if OLD_ROOT is not None and p.is_relative_to(OLD_ROOT):
        raise ValueError(f'Write to OLD is forbidden: {p}')
    return p if p.is_dir() else assert_writable(p)


def atomic_json(path, value):
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.tmp'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def preflight():
    for root in (RUN, RAW, STATE):
        assert_writable(guard(root))
        if root.exists():
            for child in root.rglob('*'):
                guard(child)
    for suffix in ('', '-wal', '-shm', '-journal'):
        guard(DB.with_name(DB.name + suffix))


@contextmanager
def exclusive(name):
    import msvcrt
    path = guard(RUN / f'{name}.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(f'zh-v2 crawl {name} already running') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def sha256_file(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def engine():
    """r2ai.zh_full.crawl re-executed in its own namespace with this module's roots and lanes."""
    from r2ai.zh_full import crawl as base
    spec = importlib.util.spec_from_file_location('r2ai.zh_full._zh_v2_crawl', base.__file__)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.RUN, m.RAW, m.DB, m.STATE = RUN, RAW, DB, STATE
    m.RATE_CUTS = RUN / 'rate_cuts.json'
    m.LANES, m.DOMAINS = LANES, DOMAINS
    m.MAX_CONNS_OVERRIDE = {}
    m.guard, m.preflight, m.atomic_json, m.exclusive = guard, preflight, atomic_json, exclusive
    load_runtime = m.network_runtime

    def network_runtime():
        runtime = load_runtime()
        runtime.PoliteClient.ROBOTS_REDIRECT_SUFFIX = ROBOTS_REDIRECT_SUFFIX
        return runtime
    m.network_runtime = network_runtime
    return m


def v2_rows(ids_path=IDS) -> pl.DataFrame:
    df = pl.read_parquet(ids_path, columns=['id', 'url', 'lang', 'host'])
    return df.filter((pl.col('lang') == 'zh') & pl.col('host').is_in(sorted(DOMAINS)))


def prepare(ids_path=IDS, db_path=None) -> dict:
    """Load every v2 URL of the lane hosts into the zh-v2 crawl.db (idempotent: INSERT OR IGNORE, counts must match)."""
    db_path = DB if db_path is None else db_path
    preflight()
    rows = v2_rows(ids_path)
    groups = group_urls(zip(rows['id'].to_list(), rows['url'].to_list()))
    by_domain = {}
    for g in groups:
        by_domain.setdefault(g['domain'], []).append(g)
    for d, gs in by_domain.items():
        gs.sort(key=lambda g: g['url_norm'])
        for i, g in enumerate(gs):
            g['rank'] = i
    unknown = set(by_domain) - DOMAINS
    if unknown:
        raise ValueError(f'URLs outside the lane hosts: {sorted(unknown)}')
    guard(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = StateDB(guard(db_path))
    try:
        added = db.add_urls([g for gs in by_domain.values() for g in gs])
        for d in by_domain:
            db.upsert_domain(d, conns=2, approved=1, state='ready', updated_at=time.time())
        counts = dict(db.conn.execute('SELECT domain,count(*) FROM urls GROUP BY domain'))
        want = {d: len(gs) for d, gs in by_domain.items()}
        if counts != want:
            raise ValueError(f'crawl.db inventory differs from the v2 list: {counts} vs {want}')
        value = {'ids_sha256': sha256_file(ids_path), 'v2_ids': rows.height, 'url_groups': sum(want.values()),
                 'rows_added_now': added, 'domains': want, 'lanes': LANES, 'prepared_at': time.time()}
        atomic_json(RUN / 'prepare_complete.json', value)
        db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return value
    finally:
        db.conn.close()


def status() -> dict:
    c = sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)
    try:
        out = {}
        for d, s, n in c.execute('SELECT domain,status,count(*) FROM urls GROUP BY 1,2'):
            out.setdefault(d, {})[s] = n
        states = dict(c.execute('SELECT domain,state FROM domains'))
    finally:
        c.close()
    return {d: {**v, 'state': states.get(d)} for d, v in sorted(out.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=['prepare', 'run', 'stop', 'status'])
    ap.add_argument('--deadline', default=DEFAULT_DEADLINE)
    a = ap.parse_args(argv)
    if a.step == 'status':
        print(json.dumps(status(), ensure_ascii=False, indent=1))
        return 0
    if a.step == 'prepare':
        print(json.dumps(prepare(), ensure_ascii=False))
        return 0
    m = engine()
    if a.step == 'stop':
        m.request_stop()
        return 0
    from r2ai.zh_full.common import keep_awake
    from r2ai.zh_full.selection_queue import deadline_timestamp
    deadline = deadline_timestamp(a.deadline)
    with exclusive('crawl'), keep_awake():
        code = None
        try:
            code = asyncio.run(m.run(deadline))
            return code
        finally:
            atomic_json(RUN / 'last_exit.json', {'pid': os.getpid(), 'code': code, 'at': time.time()})


if __name__ == '__main__':
    raise SystemExit(main())
