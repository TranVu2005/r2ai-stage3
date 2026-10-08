"""One-shot watchdog for the full-zh crawler (run every 5 minutes by Task Scheduler).

    python -m r2ai.zh_full.watchdog            # check once, restart if dead/hung
    python -m r2ai.zh_full.watchdog --dry-run  # decide and log only

Alive = a process running `python -B -u -m r2ai.zh_full.crawl run` (optional --deadline) whose own
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
from .common import RUN, DB, guard, atomic_json, exclusive
from .selection_queue import DEFAULT_DEADLINE, deadline_timestamp, remaining_count

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from vicrawl.urlnorm import normalize_url

CRAWL_ARGS = ('-B', '-u', '-m', 'r2ai.zh_full.crawl', 'run')
COMMAND_TEXT = 'python -B -u -m r2ai.zh_full.crawl run --deadline {deadline}'
STALE_SECONDS = 600
SUSPECT_CONFIRM_SECONDS = 240
MAX_RESTARTS_PER_HOUR = 6
NETWORK_BACKOFF_MAX_SECONDS = 600
KEEPALIVE = RUN / 'keepalive'
STATUS_LOG = KEEPALIVE / 'status.jsonl'
AUDIT_PATH = RUN / 'requests.jsonl'


@dataclass
class Proc:
    pid: int
    create_time: float
    cmdline: list


def is_crawler_command(command):
    try:
        start = command.index('-m')
    except ValueError:
        return False
    if command[start:start+3] != ['-m','r2ai.zh_full.crawl','run']:
        return False
    extra = command[start+3:]
    return (not extra or (len(extra)==2 and extra[0]=='--deadline')
            or (len(extra)==1 and extra[0].startswith('--deadline=')))


def crawler_procs(procs):
    return [p for p in procs if is_crawler_command(p.cmdline)]


def restarts_last_hour(history, now):
    return sum(now - t < 3600 for t in history)


def decide(now, procs, runtime, log_mtime, last_exit, history, suspect, paused,
           deadline=None, remaining=None):
    """Pure decision: returns {'action', 'reason', 'pid', 'exit_code', 'suspect', 'heartbeat_age'}."""
    out = {'pid': None, 'exit_code': None, 'suspect': None, 'heartbeat_age': None, 'reason': ''}
    procs = crawler_procs(procs)
    runtime = runtime or {}
    budget_left = restarts_last_hour(history, now) < MAX_RESTARTS_PER_HOUR
    if deadline is not None and now >= deadline:
        return {**out, 'action': 'completed', 'reason': 'deadline reached', 'suspect': suspect}
    if paused:
        return {**out, 'action': 'paused', 'reason': 'watchdog.pause present', 'suspect': suspect}
    if not procs and remaining == 0:
        return {**out, 'action': 'completed', 'reason': 'no selected URLs remain', 'suspect': None}
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
            if is_crawler_command(cmd):
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


def build_crawl_command(python, deadline):
    return [python, *CRAWL_ARGS, '--deadline', deadline]


def network_retry_delay(failures):
    return min(NETWORK_BACKOFF_MAX_SECONDS, 60 * (2 ** max(0, failures - 1)))


def network_probe_allowed(now, state):
    return now >= state.get('next_network_probe_at', 0)


def gate_network_restart(decision, now, state, online):
    if decision['action'] not in ('restart', 'kill_restart'):
        return online
    if not network_probe_allowed(now, state):
        decision.update(action='waiting_network', reason='network retry backoff')
        return False
    if not online:
        failures = int(state.get('network_failures', 0)) + 1
        delay = network_retry_delay(failures)
        state['network_failures'] = failures
        state['next_network_probe_at'] = now + delay
        decision.update(action='waiting_network', reason=f'network unavailable; retry in {delay}s')
        return False
    state['network_failures'] = 0
    state['next_network_probe_at'] = 0
    return True


def _network_available(timeout=3):
    """Check route connectivity only; do not request any crawl target."""
    try:
        with socket.create_connection(('1.1.1.1', 443), timeout=timeout):
            return True
    except OSError:
        return False


def status_record(now, decision, runtime, domains, robots, deferred_fetched, free_disk_bytes,
                  network_online=None, alerts=None):
    action = decision['action']
    if action in ('restart', 'kill_restart'):
        action = 'started' if decision.get('spawned_pid') else 'waiting_network'
    elif action in ('ok', 'suspect'):
        action = 'alive'
    elif action in ('finished', 'gave_up'):
        action = 'completed' if decision.get('reason') in ('deadline reached', 'last run exited 0',
                                                           'no selected URLs remain') else 'waiting_network'
    return {'at': datetime.fromtimestamp(now, timezone(timedelta(hours=7))).isoformat(),
            'pid': decision.get('spawned_pid') or decision.get('pid') or (runtime or {}).get('pid'),
            'action': action, 'reason': decision.get('reason'), 'network_online': network_online,
            'domains': domains, 'robots': robots, 'deferred_fetched': deferred_fetched,
            'remaining_total': sum(d['remaining'] for d in domains.values()),
            'free_disk_bytes': free_disk_bytes, 'alerts': alerts or []}


def spawn(deadline=DEFAULT_DEADLINE):
    stamp = time.strftime('%Y%m%dT%H%M%S')
    stdout, stderr = guard(RUN / f'crawl-wd-{stamp}.stdout.log'), guard(RUN / f'crawl-wd-{stamp}.stderr.log')
    command = build_crawl_command(python_executable(), deadline)
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
                'launcher_pid': p.pid, 'stdout': str(stdout), 'stderr': str(stderr),
                'command': COMMAND_TEXT.format(deadline=deadline), 'argv': command,
                'cwd': str(ROOT)})
    return p.pid


def _read_new_audit(offset):
    if not AUDIT_PATH.exists():
        return 0, []
    with AUDIT_PATH.open('rb') as f:
        size = f.seek(0, os.SEEK_END)
        offset = max(0, min(size, int(offset)))
        f.seek(offset)
        data = f.read()
    last = data.rfind(b'\n')
    if last < 0:
        return offset, []
    complete = data[:last + 1]
    records = []
    for line in complete.splitlines():
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            continue
    return offset + len(complete), records


def _recent_audit(cutoff):
    """Read a bounded audit tail, enough to cover the recent 15-minute window."""
    stats, robots = {}, {}
    if not AUDIT_PATH.exists():
        return stats, robots
    with AUDIT_PATH.open('rb') as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - 16 * 1024 * 1024)
        f.seek(start)
        data = f.read()
    if start:
        first = data.find(b'\n')
        data = data[first + 1:] if first >= 0 else b''
    for line in data.splitlines():
        try:
            row = json.loads(line)
            if row.get('t0', 0) < cutoff:
                continue
        except (ValueError, UnicodeDecodeError):
            continue
        domain = row.get('domain')
        if row.get('kind') == 'robots':
            robots[domain] = row.get('http_status') if not row.get('error') else row['error'].split(':')[0]
            continue
        s = stats.setdefault(domain, {'requests': 0, 'errors': 0})
        s['requests'] += 1
        s['errors'] += bool(row.get('error') or (row.get('http_status') or 0) >= 400)
    return stats, robots


def bootstrap_monitor_config(now, runtime):
    process = _read_json(RUN / 'process.json') or {}
    starts = [row.get('start') for row in (runtime or {}).get('domains', {}).values()
              if isinstance(row, dict) and isinstance(row.get('start'), (int, float))]
    since = ((runtime or {}).get('started_at') or process.get('started_at')
             or (min(starts) if starts else now))
    robots, offset = {}, 0
    if AUDIT_PATH.exists():
        with AUDIT_PATH.open('rb') as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(0)
            while f.tell() < size:
                line_start = f.tell()
                line = f.readline()
                if not line or not line.endswith(b'\n') or f.tell() > size:
                    offset = line_start
                    break
                offset = f.tell()
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row.get('kind') == 'robots':
                    robots[row.get('domain')] = (row.get('http_status') if not row.get('error')
                                                 else row['error'].split(':')[0])
            else:
                offset = size
    return {'monitor_since': since, 'created_at': now, 'audit_offset': offset,
            'robots_last': robots, 'deadline': DEFAULT_DEADLINE}


def _deferred_in_audit(conn, records):
    urls = sorted({normalize_url(r.get('url', '')) for r in records
                   if r.get('kind') == 'page' and r.get('url')})
    deferred = []
    for start in range(0, len(urls), 800):
        batch = urls[start:start + 800]
        marks = ','.join('?' for _ in batch)
        deferred.extend(row[0] for row in conn.execute(
            f"SELECT url_norm FROM select_meta WHERE new_status='deferred_select' AND url_norm IN ({marks})",
            batch))
    return deferred


def _database_status(now, since):
    db = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.row_factory = sqlite3.Row
        runtime_rows = db.execute('''SELECT domain,
            SUM(CASE WHEN status='ok' AND fetched_at>=? THEN 1 ELSE 0 END) ok_total,
            SUM(CASE WHEN status='ok' AND fetched_at>=? THEN 1 ELSE 0 END) ok_15m
            FROM urls GROUP BY domain''', (since, now - 900)).fetchall()
        totals = {r['domain']: (r['ok_total'] or 0, r['ok_15m'] or 0) for r in runtime_rows}
        domains = {}
        for r in db.execute('SELECT domain,state,updated_at FROM domains ORDER BY domain'):
            rem = remaining_count(db, r['domain'])
            total, last15 = totals.get(r['domain'], (0, 0))
            domains[r['domain']] = {'ok_cumulative': int(total), 'ok_per_second_15m': last15 / 900,
                                    'error_pct_15m': None, 'requests_15m': 0,
                                    'remaining': rem, 'state': r['state'], 'robots': None,
                                    'state_updated_at': r['updated_at'],
                                    'completed_at': r['updated_at'] if r['state'] == 'finished' else None}
        return domains
    finally:
        db.close()


def _status_alerts(now, domains, network_online, state):
    starts = state.setdefault('threshold_since', {})
    alerts = []
    for domain, row in domains.items():
        if row['state'] != 'active':
            starts.pop(domain + ':errors', None)
            starts.pop(domain + ':zero_ok', None)
            continue
        errors = row.get('error_pct_15m')
        err_key, zero_key = domain + ':errors', domain + ':zero_ok'
        if errors is not None and errors > 5:
            starts.setdefault(err_key, now)
            if now - starts[err_key] >= 1800:
                alerts.append({'domain': domain, 'type': 'errors_over_5pct_30m', 'since': starts[err_key]})
        else:
            starts.pop(err_key, None)
        if network_online and row['ok_per_second_15m'] == 0:
            starts.setdefault(zero_key, now)
            if now - starts[zero_key] >= 1800:
                alerts.append({'domain': domain, 'type': 'zero_ok_30m_network_online', 'since': starts[zero_key]})
        else:
            starts.pop(zero_key, None)
    if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
        alerts.append({'type': 'free_disk_below_20GB'})
    return alerts


def check(dry_run=False, deadline=DEFAULT_DEADLINE):
    now = time.time()
    deadline_ts = deadline_timestamp(deadline)
    state = _read_json(RUN / 'watchdog_state.json') or {}
    procs = _live_procs()
    runtime = _read_json(RUN / 'runtime_state.json') or {}
    monitor = _read_json(KEEPALIVE / 'config.json') or {}
    if not monitor:
        monitor = bootstrap_monitor_config(now, runtime)
        if not dry_run:
            atomic_json(KEEPALIVE / 'config.json', monitor)
    since = monitor.get('monitor_since') or runtime.get('started_at') or now
    if 'audit_offset' not in state:
        state['audit_offset'] = monitor.get('audit_offset', AUDIT_PATH.stat().st_size if AUDIT_PATH.exists() else 0)
    state.setdefault('robots_last', monitor.get('robots_last', {}))
    offset, new_records = _read_new_audit(state['audit_offset'])
    state['audit_offset'] = offset
    window, current_robots = _recent_audit(now - 900)
    robots = {**state.get('robots_last', {}), **current_robots}
    domains = _database_status(now, since)
    for domain, row in domains.items():
        recent = window.get(domain, {'requests': 0, 'errors': 0})
        row['requests_15m'] = recent['requests']
        row['error_pct_15m'] = (100 * recent['errors'] / recent['requests']) if recent['requests'] else None
        row['robots'] = robots.get(domain)
    deferred = []
    db = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        deferred = _deferred_in_audit(db, new_records)
    finally:
        db.close()
    deferred_total = int(state.get('deferred_fetched', 0)) + len(deferred)
    runtime_decision = runtime
    history = _history()
    decision = decide(now, procs, _read_json(RUN / 'runtime_state.json'), _latest_stdout_mtime(),
                      _read_json(RUN / 'last_exit.json'), history, state.get('suspect'),
                      (RUN / 'watchdog.pause').exists(), deadline=deadline_ts,
                      remaining=sum(r['remaining'] for r in domains.values()))
    network_online = (_network_available() if network_probe_allowed(now, state)
                      else state.get('last_network_online', False))
    if deferred:
        p = guard(RUN / 'watchdog.pause')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        if procs:
            try:
                from .crawl import request_stop
                request_stop()
            except Exception as exc:
                decision['stop_error'] = str(exc)
        decision.update(action='paused', reason='deferred page fetch; pause created and graceful stop requested')
    elif decision['action'] in ('restart', 'kill_restart'):
        network_online = gate_network_restart(decision, now, state, network_online)
        if decision['action'] in ('restart', 'kill_restart') and network_online and not dry_run:
            if decision['action'] == 'kill_restart':
                _kill_tree(procs)
            decision['spawned_pid'] = spawn(deadline)
    for d, row in domains.items():
        if row['state'] == 'finished' and row['remaining'] == 0 and d == 'a-hospital.com':
            state.setdefault('a_hospital_completed_at', row.get('completed_at') or now)
        if d == '120ask.com' and row.get('robots') == 200:
            state.setdefault('robots_120ask_200_seen_at', now)
    alerts = _status_alerts(now, domains, network_online, state)
    if deferred:
        alerts.append({'type': 'deferred_page_fetched', 'urls': deferred})
    pids = {p.pid for p in crawler_procs(procs)}
    row = {'ts': now, 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **decision,
           'restarts_last_hour': restarts_last_hour(history, now), 'dry_run': dry_run,
           'network_online': network_online, 'remaining': sum(r['remaining'] for r in domains.values())}
    with guard(RUN / 'watchdog.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if not dry_run:
        state.update(suspect=decision.get('suspect'), last_check=now, last_action=decision['action'],
                     deferred_fetched=deferred_total, last_network_online=network_online,
                     robots_last=robots)
        atomic_json(RUN / 'watchdog_state.json', state)
    public = status_record(now, decision, runtime_decision, domains, robots, deferred_total,
                           shutil.disk_usage(ROOT).free, network_online, alerts)
    public['active_pids'] = sorted(pids)
    row.update({'keepalive_action': public['action'], 'deferred_fetched': deferred_total,
                'alerts': alerts})
    KEEPALIVE.mkdir(parents=True, exist_ok=True)
    with guard(STATUS_LOG).open('a', encoding='utf-8') as f:
        f.write(json.dumps(public, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--deadline', default=DEFAULT_DEADLINE)
    args = parser.parse_args(argv)
    try:
        with exclusive('watchdog'):
            check(args.dry_run, deadline=args.deadline)
    except RuntimeError as e:                                    # another check is running
        print(str(e), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
