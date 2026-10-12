"""Detached, resumable driver for one zh-full batch: extract -> chunk per domain -> embed per segment.

Why a driver module instead of a scratch script: the b001 driver died silently twice (10/10 05:14 and 08:31) with empty
stdout/stderr because it was a child of the Claude session. This one is launched through Win32_Process.Create (no
parent), writes a heartbeat file every `interval` seconds and an exit record (code, reason, last step) on every exit it
can observe; a missing exit record with a stale heartbeat therefore means "killed from outside".

  launch (detached):  python -m r2ai.zh_full_ingest.driver launch --batch b002 [--first 120ask.com,cnkang.com]
  status:             python -m r2ai.zh_full_ingest.driver status --batch b002
  run in foreground:  python -m r2ai.zh_full_ingest.driver run --batch b002

Every ingest step is idempotent, so launching again after a reboot continues. Resume verifies each finished embedding
block (rows, shape, norm, NaN, sparse non-empty, SHA256 in the manifest) before skipping it.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .common import MemoryGuard, Roots, atomic_json, exclusive

WAIT_S, MAX_RETRIES = 600, 12
RETRYABLE = ('MEMORY_PAUSES', 'GPU_BUSY', 'already running')
STALE_S = 300


def paths(roots: Roots, batch: str) -> dict:
    return {'heartbeat': roots.run / f'driver-{batch}.heartbeat.json', 'exit': roots.run / f'driver-{batch}.exit.json',
            'log': roots.run / f'driver-{batch}.jsonl', 'stdout': roots.run / f'driver-{batch}.stdout.log',
            'stderr': roots.run / f'driver-{batch}.stderr.log'}


class Heartbeat:
    """Writes {pid, at, step, detail} to `path` every `interval` s from a daemon thread, and on every beat()."""

    def __init__(self, roots: Roots, path: Path, interval: float = 30.0):
        self.roots, self.path, self.interval = roots, path, interval
        self.state = {'pid': os.getpid(), 'started_at': time.time(), 'step': 'starting', 'detail': {}}
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _write(self):
        with self._lock:
            atomic_json(self.roots, self.path, {**self.state, 'at': time.time()})

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._write()
            except OSError:
                pass                                       # a locked file must never kill the driver

    def beat(self, step: str, **detail):
        self.state['step'], self.state['detail'] = step, detail
        self._write()

    def __enter__(self):
        self._write()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


class ExitRecord:
    """Writes exit.json (code, reason, last step, traceback) on normal exit, exception, SIGINT/SIGTERM/SIGBREAK."""

    def __init__(self, roots: Roots, path: Path, heartbeat: Heartbeat):
        self.roots, self.path, self.heartbeat, self.written = roots, path, heartbeat, False

    def write(self, code: int, reason: str, error: str | None = None):
        if self.written:
            return
        self.written = True
        atomic_json(self.roots, self.path, {'pid': os.getpid(), 'at': time.time(), 'code': code, 'reason': reason,
                                            'last_step': self.heartbeat.state['step'], 'detail': self.heartbeat.state['detail'],
                                            'error': error})

    def __enter__(self):
        self.path.unlink(missing_ok=True)                 # a stale record from the previous run must not look current
        atexit.register(lambda: self.write(1, 'process exited without a recorded result'))
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._on_signal)
        return self

    def _on_signal(self, signum, frame):
        self.write(128 + signum, f'signal {signum}')
        raise SystemExit(128 + signum)

    def __exit__(self, kind, exc, tb):
        if exc is None:
            self.write(0, 'ALL_DONE')
        elif isinstance(exc, SystemExit):
            self.write(exc.code if isinstance(exc.code, int) else 1, f'SystemExit({exc.code!r})')
        else:
            self.write(1, repr(exc), ''.join(traceback.format_exception(kind, exc, tb))[-4000:])
        return False


def driver_status(roots: Roots, batch: str, now: float | None = None, stale_s: float = STALE_S) -> dict:
    """alive | exited(code) | died_silently(stale heartbeat, no exit record) | not_started."""
    p, now = paths(roots, batch), now or time.time()
    hb = json.loads(p['heartbeat'].read_text('utf-8')) if p['heartbeat'].exists() else None
    ex = json.loads(p['exit'].read_text('utf-8')) if p['exit'].exists() else None
    if ex is not None:
        return {'state': 'exited', 'code': ex['code'], 'reason': ex['reason'], 'last_step': ex['last_step']}
    if hb is None:
        return {'state': 'not_started'}
    age = now - hb['at']
    state = 'alive' if age <= stale_s else 'died_silently'
    return {'state': state, 'heartbeat_age_s': round(age), 'pid': hb['pid'], 'last_step': hb['step'], 'detail': hb['detail']}


def order_domains(sizes: dict, first: list[str]) -> list[str]:
    head = [d for d in first if d in sizes]
    return head + sorted((d for d in sizes if d not in head), key=lambda d: (sizes[d], d))


class ManifestView:
    def __init__(self, roots: Roots, batch: str):
        self.roots, self.batch = roots, batch

    def _ro(self, sql, *args):
        c = sqlite3.connect(f'file:{self.roots.manifest.as_posix()}?mode=ro', uri=True, timeout=30)
        try:
            return c.execute(sql, args).fetchall()
        finally:
            c.close()

    def pending_shards(self) -> int:
        return self._ro("SELECT count(*) FROM shards WHERE batch=? AND status!='done'", self.batch)[0][0]

    def domain_sizes(self) -> dict:
        return dict(self._ro('SELECT domain, sum(n_urls) FROM shards WHERE batch=? GROUP BY 1', self.batch))

    def chunked_segments(self) -> dict:
        return {d: (name, rows) for name, d, rows in
                self._ro("SELECT name, domain, rows FROM segments WHERE batch=? AND status='chunked'", self.batch)}


def retrying(label, fn, *, log, beat, stop_file: Path, sleep=time.sleep, wait_s=WAIT_S, max_retries=MAX_RETRIES):
    """Run fn; wait and retry on RAM/GPU/lock stops; any other error propagates (exit record keeps the traceback)."""
    for attempt in range(1, max_retries + 1):
        if stop_file.exists():
            raise SystemExit(f'{stop_file.name} present (crawl slowdown); stopping before {label}')
        try:
            t = time.time()
            out = fn()
            return out, round(time.time() - t, 1), attempt
        except RuntimeError as e:
            if not any(k in str(e) for k in RETRYABLE):
                raise
            log(step=label, retry=attempt, wait_s=wait_s, reason=str(e)[:300])
            beat(f'waiting: {label}', retry=attempt, reason=str(e)[:200])
            sleep(wait_s)
    raise RuntimeError(f'too many retries: {label}')


def run_batch(roots: Roots, batch: str, *, view, ingest, first=(), log, beat, stop, extract_guard, chunk_guard, embed_guard,
              sleep=time.sleep, wait_s=WAIT_S) -> None:
    stop_file = roots.run / 'monitor_alert.json'
    kw = dict(log=log, beat=beat, stop_file=stop_file, sleep=sleep, wait_s=wait_s)
    if view.pending_shards():
        def do_extract():
            with exclusive(roots, 'extract'):
                return ingest.extract_batch(roots, batch, memory=extract_guard, before_shard=stop)
        beat('extract')
        out, secs, _ = retrying('extract', do_extract, **kw)
        log(step='extract', seconds=secs, **{k: v for k, v in out.items() if k != 'seconds'})
    for d in order_domains(view.domain_sizes(), list(first)):
        def do_chunk(d=d):
            with exclusive(roots, 'chunk'):
                chunk_guard.pauses = []
                chunk_guard.check('chunk')
                return ingest.chunk_segment(roots, batch, d, batch_docs=5000)
        beat(f'chunk {d}')
        out, secs, _ = retrying(f'chunk {d}', do_chunk, **kw)
        log(step='chunk', domain=d, segment=out['name'], status=out['status'], rows=out.get('rows'), seconds=secs)
    chunked = view.chunked_segments()
    for d in order_domains({d: rows for d, (_, rows) in chunked.items()}, list(first)):
        name, rows = chunked[d]

        def do_embed(name=name):
            embed_guard.pauses = []
            with exclusive(roots, 'gpu'):
                return ingest.embed_segment(roots, name, memory=embed_guard, before_block=lambda i, name=name: (stop(i), beat(f'embed {name}', block=i)))
        beat(f'embed {name}', rows=rows)
        out, secs, attempt = retrying(f'embed {name}', do_embed, **kw)
        log(step='embed', segment=name, rows=rows, status=out['status'], seconds=secs, attempt=attempt,
            chunks_per_s=out.get('chunks_per_s_this_run'))
    log(step='ALL_DONE')


def launch_command(roots_repo: str, worktree: str, batch: str, first: str | None) -> str:
    """cmd.exe line for Win32_Process.Create: no parent process, env set inline, output to files."""
    run = r'out\runs\zh-full-ingest'
    tail = f' --first {first}' if first else ''
    return (f'cmd.exe /c "cd /d {roots_repo} && set PYTHONPATH={worktree}\\src&& set PYTHONIOENCODING=utf-8&& '
            rf'.venv\Scripts\python.exe -B -m r2ai.zh_full_ingest.driver run --batch {batch}{tail} '
            rf'>> {run}\driver-{batch}.stdout.log 2>> {run}\driver-{batch}.stderr.log"')


def launch(roots: Roots, batch: str, first: str | None) -> dict:
    repo = str(Path(roots.run).resolve().parents[2])
    worktree = str(Path(__file__).resolve().parents[3])
    cmd = launch_command(repo, worktree, batch, first)
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          "-Arguments @{CommandLine=$env:DRIVER_CMD; CurrentDirectory=$env:DRIVER_CWD}; "
          "Write-Output ($r.ReturnValue.ToString() + ' ' + $r.ProcessId.ToString())")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, check=True,
                         env={**os.environ, 'DRIVER_CMD': cmd, 'DRIVER_CWD': repo}).stdout.split()
    if out[0] != '0':
        raise RuntimeError(f'Win32_Process.Create failed: {out}')
    return {'pid': int(out[1]), 'command': cmd}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['run', 'launch', 'status'])
    ap.add_argument('--batch', required=True)
    ap.add_argument('--first', default=None, help='comma-separated domains to process before the rest (small-first)')
    ap.add_argument('--interval', type=float, default=30.0)
    a = ap.parse_args(argv)
    roots = Roots()
    if a.action == 'status':
        print(json.dumps(driver_status(roots, a.batch), indent=1, default=str))
        return 0
    if a.action == 'launch':
        print(json.dumps(launch(roots, a.batch, a.first), indent=1))
        return 0
    from . import ingest
    from .__main__ import _stop_on_alert
    p = paths(roots, a.batch)

    def log(**kw):
        line = json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False, default=str)
        print(line, flush=True)
        with open(roots.guard(p['log']), 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    with Heartbeat(roots, p['heartbeat'], a.interval) as hb, ExitRecord(roots, p['exit'], hb):
        t = time.time()
        roots.preflight()
        log(step='preflight', seconds=round(time.time() - t, 1), pid=os.getpid())
        run_batch(roots, a.batch, view=ManifestView(roots, a.batch), ingest=ingest,
                  first=[d for d in (a.first or '').split(',') if d], log=log, beat=hb.beat, stop=_stop_on_alert(roots),
                  extract_guard=MemoryGuard(roots, minimum_gib=2.5, max_pauses=8), chunk_guard=MemoryGuard(roots),
                  embed_guard=MemoryGuard(roots, minimum_gib=3.0, max_pauses=8))
    return 0


if __name__ == '__main__':
    sys.exit(main())
