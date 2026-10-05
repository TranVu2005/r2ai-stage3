"""Resume the authorized zh experiment; all state/output paths belong to zh-sample."""
from __future__ import annotations

from r2ai.paths import ROOT
from .common import RUN, CHUNKS, preflight, atomic_json, exclusive
import argparse
import asyncio
import json
import subprocess
import sys
import time


def run():
    from .sample import prepare
    from .crawl import run_crawl
    from .extract import extract_all
    from .index import chunk, embed, digest
    from .evaluate import retrieve
    from .variant import build_z, EXPECTED
    from .report import report
    preflight()
    with exclusive('pipeline'):
        state_path = RUN / 'pipeline_state.json'
        state = json.loads(state_path.read_text('utf-8')) if state_path.exists() else {'done': [], 'started_at': time.time()}
        def stage(name, fn, gpu=False):
            if name in state['done']:
                return
            state.update({'active': name, 'last_error': None, 'updated_at': time.time()})
            atomic_json(state_path, state)
            print(f'zh pipeline: {name}', flush=True)
            while True:
                try:
                    rc = fn()
                    if rc:
                        raise RuntimeError(f'{name}: exit {rc}')
                    break
                except RuntimeError as e:
                    if ((gpu and str(e).startswith('GPU_BUSY:'))
                            or (name == 'crawl' and (str(e) == 'crawl already running' or 'EGRESS_UNAVAILABLE:' in str(e)))):
                        state['last_error'] = str(e)
                        atomic_json(state_path, state)
                        print(f'{name} resource busy, waiting 30s; checkpoint retained', flush=True)
                        time.sleep(30)
                        continue
                    raise
            state['done'].append(name)
            state.update({'active': None, 'updated_at': time.time()})
            atomic_json(state_path, state)
        def full_crawl():
            gate = json.loads((RUN / 'smoke_gate.json').read_text('utf-8'))
            with exclusive('crawl'):
                return run_crawl(domains=set(gate['allowed_domains']))
        def smoke():
            if (RUN / 'smoke_gate.json').exists():
                return
            with exclusive('crawl'):
                rc = run_crawl(limit=50)
            if rc:
                return rc
            extract_all()
            stats = json.loads((RUN / 'extract_stats.json').read_text('utf-8'))
            atomic_json(RUN / 'smoke_gate.json', {'allowed_domains': sorted(d for d,s in stats.items() if s['ok'] and not s['unknown_lang']),
                        'smoke_extract_stats': stats, 'reason': 'nonempty ok Chinese text, inaccessible robots deferred'})
        def baseline():
            output = RUN / 'd50_replay/D50.json'
            if output.exists() and digest(output) == EXPECTED:
                return
            return subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'scripts/build_best_submission.py'),
                                   '--out', str(RUN / 'd50_replay/D50.zip')], cwd=ROOT).returncode
        try:
            stage('sample', prepare)
            stage('smoke', smoke)
            stage('crawl', full_crawl)
            stage('extract', extract_all)
            stage('chunk', lambda: None if (RUN / 'chunks_complete.json').exists() else chunk())
            atomic_json(RUN / 'chunks_complete.json', {'chunks_sha256': digest(CHUNKS / 'chunks_t256.parquet')})
            stage('embed', embed, gpu=True)
            stage('retrieve', retrieve, gpu=True)
            stage('baseline', baseline)
            stage('Z', build_z)
            state['complete'] = True
            atomic_json(state_path, state)
        except (Exception, KeyboardInterrupt) as e:
            state.update({'last_error': f'{type(e).__name__}: {e}', 'updated_at': time.time(), 'complete': False})
            atomic_json(state_path, state)
            raise
        finally:
            report()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', nargs='?', default='run', choices=['run', 'status'])
    a = ap.parse_args()
    if a.command == 'status':
        from .common import DB
        import sqlite3
        c = sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)
        try:
            print(c.execute('SELECT domain,status,count(*) FROM urls GROUP BY domain,status').fetchall())
        finally:
            c.close()
    else:
        run()
