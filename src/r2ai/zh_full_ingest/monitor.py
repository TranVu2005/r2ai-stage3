"""Read-only monitor while ingest runs: crawler ok/s (crawl.db mode=ro), RAM, VRAM, CPU.

Rule (approved): baseline = cnkang ok/s over the 30 minutes before the batch; when two consecutive
30-minute windows are both < 80% of baseline, write run/monitor_alert.json. The ingest CLI stops
gracefully at the next shard/block when that file exists. zhongyibaodian is logged only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time

from .common import Roots, atomic_json, log_event

WATCH, LOG_ONLY, WINDOW_S, RATIO = 'cnkang.com', ('zhongyibaodian.net',), 1800, 0.8


def ok_per_s(crawl_db, domain, lo, hi):
    c = sqlite3.connect('file:' + crawl_db.resolve().as_posix() + '?mode=ro', uri=True)
    try:
        n = c.execute("SELECT count(*) FROM urls WHERE domain=? AND status='ok' AND fetched_at>=? AND fetched_at<?",
                      (domain, lo, hi)).fetchone()[0]
    finally:
        c.close()
    return n / (hi - lo)


def resources():
    import psutil
    out = {'available_gib': psutil.virtual_memory().available / 2**30, 'cpu_pct': psutil.cpu_percent(interval=1)}
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True).stdout.strip().split(',')
        out.update(vram_mib=int(q[0]), gpu_util_pct=int(q[1]))
    except Exception as e:  # noqa: BLE001 - a monitor must not die on a probe
        out['gpu_error'] = str(e)
    return out


def evaluate(baseline, windows, ratio=RATIO):
    """True when the last two windows are both below ratio * baseline."""
    return baseline > 0 and len(windows) >= 2 and all(w < ratio * baseline for w in windows[-2:])


def run(roots: Roots, *, start=None, sample_s=300, clock=time.time, sleep=time.sleep, max_iterations=None):
    start = clock() if start is None else start
    baseline = ok_per_s(roots.crawl_db, WATCH, start - WINDOW_S, start)
    atomic_json(roots, roots.run / 'monitor_baseline.json', {'start': start, 'domain': WATCH, 'baseline_ok_per_s': baseline,
                'window_s': WINDOW_S, 'ratio': RATIO, 'log_only': {d: ok_per_s(roots.crawl_db, d, start - WINDOW_S, start)
                                                                for d in LOG_ONLY}})
    windows, next_window, i = [], start + WINDOW_S, 0
    while max_iterations is None or i < max_iterations:
        i += 1
        now = clock()
        row = {'kind': 'sample', **resources(),
               'ok_per_s_last_window': {d: ok_per_s(roots.crawl_db, d, now - WINDOW_S, now) for d in (WATCH, *LOG_ONLY)}}
        if now >= next_window:
            w = ok_per_s(roots.crawl_db, WATCH, next_window - WINDOW_S, next_window)
            windows.append(w)
            row.update(kind='window', window_end=next_window, watch_ok_per_s=w, baseline=baseline)
            next_window += WINDOW_S
            if evaluate(baseline, windows):
                atomic_json(roots, roots.run / 'monitor_alert.json', {'at': now, 'baseline': baseline, 'windows': windows[-2:],
                            'rule': f'two consecutive {WINDOW_S}s windows < {RATIO} x baseline ({WATCH})'})
        log_event(roots, 'monitor', row)
        sleep(sample_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=float, default=None)
    ap.add_argument('--sample-seconds', type=int, default=300)
    a = ap.parse_args(argv)
    roots = Roots()
    run(roots, start=a.start, sample_s=a.sample_seconds)


if __name__ == '__main__':
    main()
