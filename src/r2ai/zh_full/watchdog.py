"""One-shot watchdog for the full-zh crawler (run every 5 minutes by Task Scheduler).

    python -m r2ai.zh_full.watchdog            # check once, restart if dead/hung
    python -m r2ai.zh_full.watchdog --dry-run  # decide and log only

Alive = a process running exactly `python -B -u -m r2ai.zh_full.crawl run` whose own
heartbeat (runtime_state.json written by that PID after it started, or its stdout log)
is newer than 10 minutes. A stale heartbeat must be seen on two checks >= 4 minutes
apart before the process tree is killed, so a wake from sleep is not mistaken for a hang.
Restart uses the same command as launcher*.json, in a new hidden console/process group
outside the Task Scheduler job. Exit code 0 of the last run (all lanes finished) and
out/runs/zh-full/watchdog.pause stop restarts; more than 6 restarts per hour gives up.
Every decision is appended to out/runs/zh-full/watchdog.jsonl.
"""
from __future__ import annotations

from r2ai.paths import ROOT
from .common import RUN, guard, atomic_json, exclusive

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

CRAWL_ARGS = ('-B', '-u', '-m', 'r2ai.zh_full.crawl', 'run')
COMMAND_TEXT = 'python -B -u -m r2ai.zh_full.crawl run'      # as recorded in launcher*.json
STALE_SECONDS = 600
SUSPECT_CONFIRM_SECONDS = 240
MAX_RESTARTS_PER_HOUR = 6


@dataclass
class Proc:
    pid: int
    create_time: float
    cmdline: list


def crawler_procs(procs):
    return [p for p in procs if p.cmdline[-3:] == ['-m', 'r2ai.zh_full.crawl', 'run']]


def restarts_last_hour(history, now):
    return sum(now - t < 3600 for t in history)


def decide(now, procs, runtime, log_mtime, last_exit, history, suspect, paused):
    """Pure decision: returns {'action', 'reason', 'pid', 'exit_code', 'suspect', 'heartbeat_age'}."""
    out = {'pid': None, 'exit_code': None, 'suspect': None, 'heartbeat_age': None, 'reason': ''}
    procs = crawler_procs(procs)
    runtime = runtime or {}
    budget_left = restarts_last_hour(history, now) < MAX_RESTARTS_PER_HOUR
    if paused:
        return {**out, 'action': 'paused', 'reason': 'watchdog.pause present', 'suspect': suspect}
    if not procs:
        last_exit = last_exit or {}
        code = last_exit.get('code') if last_exit.get('pid') == runtime.get('pid') else None
        out.update(pid=runtime.get('pid'), exit_code=code)
        if code == 0:
            return {**out, 'action': 'finished', 'reason': 'last run exited 0'}
        if not budget_left:
            return {**out, 'action': 'gave_up', 'reason': f'>={MAX_RESTARTS_PER_HOUR} restarts in 1h'}
        return {**out, 'action': 'restart', 'reason': 'dead'}
    by_pid = {p.pid: p for p in procs}
    main = by_pid.get(runtime.get('pid')) or max(procs, key=lambda p: p.create_time)
    beats = [main.create_time]                                   # startup grace
    if runtime.get('pid') == main.pid and (runtime.get('updated_at') or 0) >= main.create_time:
        beats.append(runtime['updated_at'])
    if log_mtime is not None and log_mtime >= main.create_time:
        beats.append(log_mtime)
    age = now - max(beats)
    out.update(pid=main.pid, heartbeat_age=age)
    if age <= STALE_SECONDS:
        return {**out, 'action': 'ok', 'reason': 'heartbeat fresh'}
    if not suspect or suspect.get('pid') != main.pid:
        return {**out, 'action': 'suspect', 'reason': 'stale heartbeat, first sighting',
                'suspect': {'pid': main.pid, 'since': now}}
    if now - suspect['since'] < SUSPECT_CONFIRM_SECONDS:
        return {**out, 'action': 'suspect', 'reason': 'stale heartbeat, waiting for confirmation', 'suspect': suspect}
    if not budget_left:
        return {**out, 'action': 'gave_up', 'reason': f'>={MAX_RESTARTS_PER_HOUR} restarts in 1h', 'suspect': suspect}
    return {**out, 'action': 'kill_restart', 'reason': 'stale_heartbeat'}


# ---------------------------------------------------------------------------- I/O layer
def _read_json(path):
    try:
        return json.loads(path.read_text('utf-8'))
    except (OSError, ValueError):
        return None


def _history():
    path = RUN / 'watchdog.jsonl'
    out = []
    if path.exists():
        for line in path.read_text('utf-8').splitlines()[-500:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('action') in ('restart', 'kill_restart') and row.get('spawned_pid'):
                out.append(row['ts'])
    return out


def _latest_stdout_mtime():
    logs = list(RUN.glob('crawl*.stdout.log'))
    return max((p.stat().st_mtime for p in logs), default=None)


def _live_procs():
    import psutil
    out = []
    for p in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        try:
            cmd = p.info['cmdline'] or []
            if cmd[-3:] == ['-m', 'r2ai.zh_full.crawl', 'run']:
                out.append(Proc(p.info['pid'], p.info['create_time'], list(cmd)))
        except (psutil.Error, TypeError):
            continue
    return out


def _kill_tree(procs):
    import psutil
    victims = []
    for p in crawler_procs(procs):
        try:
            proc = psutil.Process(p.pid)
            if proc.create_time() != p.create_time:
                continue                                         # PID reused meanwhile
            victims += [proc, *proc.children(recursive=True)]
        except psutil.Error:
            continue
    for v in victims:
        try:
            v.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=30)


def python_executable():
    venv = ROOT / '.venv' / 'Scripts' / 'python.exe'
    return str(venv) if venv.exists() else sys.executable


def spawn():
    stamp = time.strftime('%Y%m%dT%H%M%S')
    stdout, stderr = guard(RUN / f'crawl-wd-{stamp}.stdout.log'), guard(RUN / f'crawl-wd-{stamp}.stderr.log')
    command = [python_executable(), *CRAWL_ARGS]
    flags = 0
    startupinfo = None
    if os.name == 'nt':
        flags = (subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
                 | 0x01000000)                                   # CREATE_BREAKAWAY_FROM_JOB
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0                              # SW_HIDE
    with stdout.open('ab') as out, stderr.open('ab') as err:
        try:
            p = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 creationflags=flags, startupinfo=startupinfo, close_fds=True)
        except OSError:
            if not flags:
                raise
            # The job may forbid breakaway; a new console still survives the task's own exit.
            p = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 creationflags=flags & ~0x01000000, startupinfo=startupinfo, close_fds=True)
    atomic_json(RUN / 'launcher-watchdog.json', {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'launcher_pid': p.pid, 'stdout': str(stdout), 'stderr': str(stderr), 'command': COMMAND_TEXT})
    return p.pid


def check(dry_run=False):
    now = time.time()
    state = _read_json(RUN / 'watchdog_state.json') or {}
    procs = _live_procs()
    decision = decide(now, procs, _read_json(RUN / 'runtime_state.json'), _latest_stdout_mtime(),
                      _read_json(RUN / 'last_exit.json'), _history(), state.get('suspect'),
                      (RUN / 'watchdog.pause').exists())
    row = {'ts': now, 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **decision,
           'restarts_last_hour': restarts_last_hour(_history(), now), 'dry_run': dry_run}
    if not dry_run and decision['action'] in ('restart', 'kill_restart'):
        if decision['action'] == 'kill_restart':
            _kill_tree(procs)
        row['spawned_pid'] = spawn()
    with guard(RUN / 'watchdog.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if not dry_run:
        atomic_json(RUN / 'watchdog_state.json', {'suspect': decision.get('suspect'), 'last_check': now,
                                                  'last_action': decision['action']})
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        with exclusive('watchdog'):
            check(args.dry_run)
    except RuntimeError as e:                                    # another check is running
        print(str(e), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
