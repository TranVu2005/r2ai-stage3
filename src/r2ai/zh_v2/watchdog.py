"""Watchdog of the zh_v2 crawler (new v2 hosts), same idea as r2ai.zh_full.watchdog: a Windows task runs it at startup and every
5 minutes; it reads the crawler heartbeat (runtime_state.json, written every 60 s) and restarts a dead or hung crawler.

Each check appends one line to out/runs/zh-v2-crawl/watchdog.jsonl and rewrites status.json:
  state ALIVE (heartbeat fresh) | DEAD (no process, or a hung one that was killed; restart attempted) | DONE (last exit 0 and
  nothing left to fetch) | PAUSED (watchdog.pause present) | HOLD (restart limit / network down).
`python -m r2ai.zh_v2.watchdog --status` prints status.json and exits 1 when the state is DEAD.
`--dry-run` decides but neither kills nor spawns. Restart limit: 6 per hour. No restart while the network is unreachable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import crawl as zc

STALE_S = 600
MAX_RESTARTS_PER_HOUR = 6
PAUSE = zc.RUN / 'watchdog.pause'
STATUS = zc.RUN / 'status.json'
LOG = zc.RUN / 'watchdog.jsonl'
CRAWL_ARGS = ('-B', '-u', '-m', 'r2ai.zh_v2.crawl', 'run')


def is_crawler_command(cmd) -> bool:
    c = [str(x) for x in (cmd or [])]
    return 'r2ai.zh_v2.crawl' in c and c[-1] == 'run'


def decide(now, procs, heartbeat_at, last_exit, remaining, paused, restarts_last_hour, online) -> dict:
    """Pure decision. procs = live crawler PIDs; heartbeat_at = runtime_state updated_at or None; last_exit = record or None."""
    if paused:
        return {'state': 'PAUSED', 'action': 'none', 'reason': 'watchdog.pause present'}
    age = None if heartbeat_at is None else now - heartbeat_at
    if procs:
        if age is not None and age <= STALE_S:
            return {'state': 'ALIVE', 'action': 'none', 'reason': f'heartbeat {age:.0f}s old'}
        return {'state': 'DEAD', 'action': 'kill_restart', 'reason': f'hung: heartbeat age {age}'}
    if remaining == 0 and last_exit and last_exit.get('code') == 0:
        return {'state': 'DONE', 'action': 'none', 'reason': 'nothing left to fetch'}
    if restarts_last_hour >= MAX_RESTARTS_PER_HOUR:
        return {'state': 'HOLD', 'action': 'none', 'reason': f'{restarts_last_hour} restarts in the last hour'}
    if not online:
        return {'state': 'HOLD', 'action': 'none', 'reason': 'network unreachable'}
    return {'state': 'DEAD', 'action': 'restart', 'reason': 'no crawler process'}


def live_procs() -> list[int]:
    import psutil
    out = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if is_crawler_command(p.info['cmdline']):
                out.append(p.info['pid'])
        except (psutil.Error, TypeError):
            continue
    return out


def kill_tree(pids):
    import psutil
    victims = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            victims += [proc, *proc.children(recursive=True)]
        except psutil.Error:
            pass
    for v in victims:
        try:
            v.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=30)


def spawn() -> int:
    repo = Path(zc.__file__).resolve().parents[3]
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out, err = zc.guard(zc.RUN / f'crawl-wd-{stamp}.stdout.log'), zc.guard(zc.RUN / f'crawl-wd-{stamp}.stderr.log')
    root = Path(zc.RUNS_DIR).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src'), 'PYTHONIOENCODING': 'utf-8'}
    exe = root / '.venv' / 'Scripts' / 'python.exe'
    flags = (subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP | 0x01000000) if os.name == 'nt' else 0
    with out.open('ab') as o, err.open('ab') as e:
        try:
            p = subprocess.Popen([str(exe), *CRAWL_ARGS], cwd=root, env=env, stdout=o, stderr=e, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True)
        except OSError:
            p = subprocess.Popen([str(exe), *CRAWL_ARGS], cwd=root, env=env, stdout=o, stderr=e, stdin=subprocess.DEVNULL,
                                 creationflags=flags & ~0x01000000, close_fds=True)
    return p.pid


def read_json(path):
    try:
        return json.loads(Path(path).read_text('utf-8'))
    except (OSError, ValueError):
        return None


def remaining_rows() -> int | None:
    import sqlite3
    try:
        c = sqlite3.connect(zc.DB.as_uri() + '?mode=ro', uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        return c.execute("SELECT count(*) FROM urls WHERE status IN ('pending','in_progress') OR "
                         "(status NOT IN ('ok','thin','deferred_select') AND next_try_at IS NOT NULL)").fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        c.close()


def restarts_last_hour(now) -> int:
    n = 0
    try:
        for line in LOG.read_text('utf-8').splitlines()[-200:]:
            r = json.loads(line)
            if r.get('action') in ('restart', 'kill_restart') and not r.get('dry_run') and now - r['ts'] < 3600:
                n += 1
    except (OSError, ValueError):
        pass
    return n


def check(dry_run=False, now=None) -> dict:
    from r2ai.zh_full.watchdog import _network_available
    now = time.time() if now is None else now
    zc.RUN.mkdir(parents=True, exist_ok=True)
    runtime = read_json(zc.RUN / 'runtime_state.json') or {}
    decision = decide(now, live_procs(), runtime.get('updated_at'), read_json(zc.RUN / 'last_exit.json'), remaining_rows(),
                      PAUSE.exists(), restarts_last_hour(now), _network_available())
    spawned = None
    if not dry_run and decision['action'] in ('kill_restart', 'restart'):
        if decision['action'] == 'kill_restart':
            kill_tree(live_procs())
        spawned = spawn()
    record = {'ts': now, 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(now)), **decision, 'spawned_pid': spawned,
              'dry_run': dry_run}
    zc.atomic_json(STATUS, record)
    with zc.guard(LOG).open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args(argv)
    if a.status:
        s = read_json(STATUS) or {'state': 'DEAD', 'reason': 'no status.json'}
        print(json.dumps(s, ensure_ascii=False))
        return 1 if s.get('state') == 'DEAD' else 0
    print(json.dumps(check(a.dry_run), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
