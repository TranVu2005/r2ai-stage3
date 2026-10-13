"""Liveness, control and reporting of the CC fetch.

  python -m r2ai.cc_longchau.fetchctl launch [--deadline "2026-10-28 23:59"] [--rps 0.6]   # detached (Win32_Process.Create)
  python -m r2ai.cc_longchau.fetchctl status      # reads the heartbeat; DEAD when older than 5 min, EXITED when an exit record follows it
  python -m r2ai.cc_longchau.fetchctl stop        # graceful: writes the STOP file, the fetch finishes its shard and writes the exit record
  python -m r2ai.cc_longchau.fetchctl report      # daily report: ok per priority group, 24 h speed, 503 rate, ETA per group

Files (state/cc_longchau): heartbeat.json (every 60 s), exit.json (last exit) + exits.jsonl (all), STOP (stop request),
throttle.json (current rps), groups_done.json (time each priority group finished).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
import traceback
from pathlib import Path

from .common import Roots, atomic_json

STALE_S = 300
BEAT_S = 60
DEFAULT_DEADLINE = '2026-10-28 23:59'


def paths(roots: Roots) -> dict:
    s = roots.state
    return {k: s / f for k, f in (('heartbeat', 'heartbeat.json'), ('exit', 'exit.json'), ('exits', 'exits.jsonl'), ('stop', 'STOP'),
                                  ('throttle', 'throttle.json'), ('groups_done', 'groups_done.json'))}


def parse_deadline(text) -> float:
    """Local wall-clock 'YYYY-MM-DD HH:MM' (or with a 'T') -> epoch."""
    return time.mktime(time.strptime(text.replace('T', ' '), '%Y-%m-%d %H:%M'))


class Beat:
    """Daemon thread: heartbeat every `interval` s; also raises `stop` on the deadline or when the STOP file appears."""

    def __init__(self, roots: Roots, snapshot, stop: threading.Event, *, deadline=None, interval=BEAT_S):
        self.roots, self.snapshot, self.stop, self.deadline, self.interval = roots, snapshot, stop, deadline, interval
        self.p = paths(roots)
        self.reason = None
        self.started = time.time()
        self._end = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def write(self, state='running'):
        atomic_json(self.roots, self.p['heartbeat'], {**self.snapshot(), 'at': time.time(), 'at_text': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                      'pid': os.getpid(), 'started_at': self.started, 'state': state,
                                                      'deadline': self.deadline})

    def _check_stop(self):
        if self.reason is None:
            if self.deadline is not None and time.time() >= self.deadline:
                self.reason = 'deadline'
            elif self.p['stop'].exists():
                self.reason = 'STOP file'
            if self.reason:
                self.stop.set()

    def _run(self):
        while not self._end.wait(self.interval):
            try:
                self._check_stop()
                self.write()
            except OSError:
                pass                                                    # a locked file must never kill the fetch

    def __enter__(self):
        for f in (self.p['stop'],):
            if f.exists():
                f.unlink()                                              # a stale STOP request must not stop the new run
        self._check_stop()
        self.write()
        self._thread.start()
        return self

    def __exit__(self, kind, exc, tb):
        self._end.set()
        self._thread.join(timeout=5)
        if exc is None:
            code, reason = 0, self.reason or 'finished'
        elif isinstance(exc, KeyboardInterrupt):
            code, reason = 130, 'interrupt'
        else:
            code, reason = 1, repr(exc)
        rec = {'at': time.time(), 'at_text': time.strftime('%Y-%m-%d %H:%M:%S'), 'pid': os.getpid(), 'code': code, 'reason': reason,
               'last': self.snapshot(), 'error': ''.join(traceback.format_exception(kind, exc, tb))[-3000:] if exc else None}
        try:
            self.write('exited')
            atomic_json(self.roots, self.p['exit'], rec)
            with open(self.roots.guard(self.p['exits']), 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except OSError:
            pass
        return False

    def install_signals(self):
        """SIGINT/SIGTERM/SIGBREAK: first one = graceful stop (shard committed, exit record written), second one = abort."""
        def handler(signum, frame):
            if self.stop.is_set():
                raise KeyboardInterrupt()
            self.reason = f'signal {signum}'
            self.stop.set()
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), handler)


def read_json(path):
    try:
        return json.loads(Path(path).read_text('utf-8'))
    except (OSError, ValueError):
        return None


def status(roots: Roots, now=None) -> dict:
    """RUNNING | DEAD (heartbeat older than STALE_S, no exit record after it) | EXITED | NOT_STARTED."""
    p, now = paths(roots), now or time.time()
    hb, ex = read_json(p['heartbeat']), read_json(p['exit'])
    if hb is None:
        return {'state': 'NOT_STARTED'}
    age = now - hb['at']
    out = {'heartbeat_age_s': round(age), 'heartbeat_at': hb.get('at_text'), 'pid': hb.get('pid')}
    if ex is not None and ex['at'] >= hb.get('started_at', 0):
        out |= {'state': 'EXITED', 'code': ex['code'], 'reason': ex['reason'], 'exit_at': ex['at_text']}
    elif age > STALE_S:
        out |= {'state': 'DEAD'}
    else:
        out |= {'state': 'RUNNING'}
    out |= {k: hb.get(k) for k in ('ok_total', 'ok_run', 'requests_run', 'rps', 'rate503_10m', 'rate503_run', 'current_group', 'todo_left',
                                   'last_ok_at', 'rps_changes')}
    out['deadline'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(hb['deadline'])) if hb.get('deadline') else None
    return out


def launch_command(repo: str, worktree: str, extra: str) -> str:
    return (f'cmd.exe /c "cd /d {repo} && set PYTHONPATH={worktree}\\src&& set PYTHONIOENCODING=utf-8&& '
            rf'.venv\Scripts\python.exe -B -u -m r2ai.cc_longchau.fetch {extra} '
            r'>> out\runs\cc-longchau\fetch_stdout.log 2>> out\runs\cc-longchau\fetch_stderr.log"')


def launch(roots: Roots, extra: str) -> dict:
    repo = str(Path(roots.run).resolve().parents[2])
    worktree = str(Path(__file__).resolve().parents[3])
    cmd = launch_command(repo, worktree, extra)
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          "-Arguments @{CommandLine=$env:DRIVER_CMD; CurrentDirectory=$env:DRIVER_CWD}; "
          "Write-Output ($r.ReturnValue.ToString() + ' ' + $r.ProcessId.ToString())")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, check=True,
                         env={**os.environ, 'DRIVER_CMD': cmd, 'DRIVER_CWD': repo}).stdout.split()
    if out[0] != '0':
        raise RuntimeError(f'Win32_Process.Create failed: {out}')
    return {'pid': int(out[1]), 'command': cmd}


def request_stats(log_path, since_epoch) -> dict:
    """Attempts / 503s / 206s in fetch_requests.jsonl since `since_epoch` (log timestamps are local time)."""
    n = c503 = c429 = ok = 0
    t0 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_epoch))
    if not Path(log_path).exists():
        return {'attempts': 0, 'r503': 0, 'r429': 0, 'ok': 0, 'rate503': None}
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            if '"status"' not in line or '"event"' in line or line[7:26] < t0:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            n += 1
            c503 += r['status'] == 503
            c429 += r['status'] == 429
            ok += r['status'] in (200, 206)
    return {'attempts': n, 'r503': c503, 'r429': c429, 'ok': ok, 'rate503': round(c503 / n, 3) if n else None}


def report(roots: Roots, now=None, write=True) -> dict:
    from .priority import GROUP_NAMES, load_order
    now = now or time.time()
    df = load_order(roots)
    db = sqlite3.connect(f'file:{roots.fetch_db.as_posix()}?mode=ro', uri=True)
    try:
        rows = db.execute('SELECT url_id, status, updated_at FROM fetch').fetchall()
    finally:
        db.close()
    st = {u: s for u, s, _ in rows}
    ok24 = {u for u, s, t in rows if s == 'ok' and t >= now - 86400}
    df = df.assign(status=df.url_id.map(st).fillna('pending'), ok24=df.url_id.isin(ok24))
    total24 = int(df.ok24.sum())
    speed = total24 / 86400
    groups = {}
    for g, name in enumerate(GROUP_NAMES):
        d = df[df.group == g]
        ok = int((d.status == 'ok').sum())
        other = int((~d.status.isin(['ok', 'pending', 'unavailable'])).sum())
        todo = len(d) - ok - other
        groups[name] = {'urls': len(d), 'ok': ok, 'ok_24h': int(d.ok24.sum()), 'to_do': todo,
                        'eta_hours_at_24h_speed': round(todo / speed / 3600, 1) if speed > 0 and todo else (0 if not todo else None)}
    todo_all = sum(v['to_do'] for v in groups.values())
    hb, dl = read_json(paths(roots)['heartbeat']), None
    if hb and hb.get('deadline'):
        dl = hb['deadline']
    left_h = (dl - now) / 3600 if dl else None
    rep = {'at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)), 'fetch': status(roots, now), 'groups': groups,
           'ok_total': int((df.status == 'ok').sum()), 'to_do_total': todo_all, 'ok_24h': total24,
           'speed_24h_url_per_s': round(speed, 4), 'requests_24h': request_stats(roots.run / 'fetch_requests.jsonl', now - 86400),
           'eta_hours_total_at_24h_speed': round(todo_all / speed / 3600, 1) if speed > 0 else None,
           'hours_to_deadline': round(left_h, 1) if left_h is not None else None,
           'expected_ok_by_deadline_at_24h_speed': int(speed * left_h * 3600) if left_h is not None else None}
    if write:
        d = roots.guard(roots.run / 'daily')
        d.mkdir(parents=True, exist_ok=True)
        atomic_json(roots, d / (time.strftime('%Y-%m-%d', time.localtime(now)) + '.json'), rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['launch', 'status', 'stop', 'report'])
    ap.add_argument('--deadline', default=DEFAULT_DEADLINE, help='local time, YYYY-MM-DD HH:MM (default %(default)s)')
    ap.add_argument('--rps', default='0.6')
    ap.add_argument('--shard-size', default='50')
    a = ap.parse_args(argv)
    roots = Roots()
    if a.action == 'launch':
        s = status(roots)
        if s['state'] == 'RUNNING':
            print(json.dumps({'refused': 'fetch already RUNNING', **s}, indent=1))
            return 1
        parse_deadline(a.deadline)                                      # fail early on a bad date
        member = str(Path(roots.run).resolve().parents[2] / 'data' / 'corpus_v2_ids.parquet')
        extra = (f'--workers 1 --rps {a.rps} --shard-size {a.shard_size} --member-ids {member} '
                 f'--priority-order --deadline {a.deadline.replace(" ", "T")}')
        print(json.dumps(launch(roots, extra), indent=1))
        return 0
    if a.action == 'stop':
        roots.guard(paths(roots)['stop']).write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf-8')
        print('STOP file written; the fetch exits after the current request')
        return 0
    out = status(roots) if a.action == 'status' else report(roots)
    print(json.dumps(out, indent=1, default=str))
    return 2 if out.get('state') == 'DEAD' else 0


if __name__ == '__main__':
    raise SystemExit(main())
