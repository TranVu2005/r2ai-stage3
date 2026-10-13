"""Batch ingest of the v2 new-host crawl (state/zh_v2, data/raw_zh_v2) into the zh v2 index: the zh_full_ingest pipeline unchanged
(cut closed shards -> extract -> t256 chunker -> BGE-M3 embed, one segment per (batch, domain)), only the roots, the extractor
(r2ai.zh_v2.extract) and the segment-id base differ.

* Roots: raw/db/docs/state/run are zh-v2 private; segments land in data/index/zh_v2_segments, chunks in data/chunks_zh_v2.
* Segment ids start at 100 (zh_full uses 1..31) so global chunk ids never collide with the filtered zh_full segments.
* Chunker t256 and `min_body_tokens` 50 are the zh_full values, so the reranker threshold tau = 0.96 stays comparable.
* `adopt` registers finished segments in the v2 index manifest (all their docs are v2 by construction).

  python -m r2ai.zh_v2.ingest cut --batch v2n-001 [--domains a,b]
  python -m r2ai.zh_v2.ingest launch --batch v2n-001      # detached driver (extract -> chunk -> embed -> adopt)
  python -m r2ai.zh_v2.ingest status --batch v2n-001
  python -m r2ai.zh_v2.ingest adopt
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, STATE_DIR, WORK_DATA_DIR
from r2ai.zh_full_ingest import driver as drv
from r2ai.zh_full_ingest import ingest as base_ingest
from r2ai.zh_full_ingest.common import MemoryGuard, Roots

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import crawl as zcrawl
from . import extract as zextract
from . import index as zindex

FIRST_SEGMENT_ID = 100
RUN = RUNS_DIR / 'zh-v2-ingest'


@dataclass(frozen=True)
class V2Roots(Roots):
    raw: Path = zcrawl.RAW
    crawl_db: Path = zcrawl.DB
    docs: Path = WORK_DATA_DIR / 'docs_zh_v2'
    chunks: Path = zindex.CHUNKS_V2
    index: Path = zindex.ZH_V2_INDEX_ROOT
    run: Path = RUN
    state: Path = STATE_DIR / 'zh_v2_ingest'
    forbidden: tuple = field(default_factory=lambda: (
        WORK_DATA_DIR / 'raw_vi', WORK_DATA_DIR / 'docs_vi', WORK_DATA_DIR / 'chunks', WORK_DATA_DIR / 'index' / 't256',
        WORK_DATA_DIR / 'raw_zh_full', WORK_DATA_DIR / 'raw_zh_sample', WORK_DATA_DIR / 'docs_zh_sample',
        WORK_DATA_DIR / 'chunks_zh_sample', WORK_DATA_DIR / 'index' / 'zh_sample_t256',
        WORK_DATA_DIR / 'docs_zh_full', WORK_DATA_DIR / 'chunks_zh_full', WORK_DATA_DIR / 'index' / 'zh_full_segments',
        STATE_DIR / 'zh_full', STATE_DIR / 'zh_full_ingest', STATE_DIR / 'crawl.db', STATE_DIR / 'crawl_zh_sample.db',
        STATE_DIR / 'extract.db'))


def use_v2_extractor():
    """Route the shared ingest functions to the zh-v2 extractor (it delegates every other host to zh_full)."""
    base_ingest.zf = zextract


def init_manifest(roots: V2Roots):
    """Create the manifest and move its segment-id sequence to FIRST_SEGMENT_ID - 1 (only while it has no segments)."""
    c = base_ingest.connect(roots)
    try:
        if not c.execute('SELECT 1 FROM segments').fetchone():
            with c:
                c.execute("UPDATE sqlite_sequence SET seq=? WHERE name='segments'", (FIRST_SEGMENT_ID - 1,))
        seq = c.execute("SELECT seq FROM sqlite_sequence WHERE name='segments'").fetchone()[0]
    finally:
        c.close()
    return seq


def adopt(roots: V2Roots | None = None) -> dict:
    """Add every ACTIVE v2-ingest segment missing from the v2 index manifest. Returns the manifest rows added."""
    roots = roots or V2Roots()
    manifest = json.loads(zindex.MANIFEST.read_text('utf-8'))
    have = {r['name'] for r in manifest['segments']}
    added = []
    for seg in base_ingest.segment_rows(roots):
        if seg['name'] in have:
            continue
        docs = pl.read_parquet(roots.chunks / seg['name'] / 'docs.parquet', columns=['doc_id'])
        added.append({'name': seg['name'], 'segment_id': seg['segment_id'], 'batch': seg['batch'], 'domain': seg['domain'],
                      'rows_source': seg['rows'], 'rows_v2': seg['rows'], 'docs_v2': docs.height, 'source': 'zh_v2_ingest',
                      'global_start': base_ingest.global_start(seg['segment_id'])})
    if added:
        import numpy as np
        zindex.write_manifest(manifest['segments'] + added, np.array(zindex.scope.v2_zh()['id'].to_numpy()))
    return {'added': [a['name'] for a in added], 'rows': sum(a['rows_v2'] for a in added), 'docs': sum(a['docs_v2'] for a in added)}


def launch(batch: str, first: str | None = None) -> dict:
    repo = str(Path(__file__).resolve().parents[3])
    cwd = str(Path(RUNS_DIR).resolve().parents[1])
    run = str(RUN.resolve())
    tail = f' --first {first}' if first else ''
    cmd = (f'cmd.exe /c "cd /d {cwd} && set PYTHONPATH={repo}\\src&& set PYTHONIOENCODING=utf-8&& '
           rf'.venv\Scripts\python.exe -B -m r2ai.zh_v2.ingest run --batch {batch}{tail} '
           rf'>> {run}\driver-{batch}.stdout.log 2>> {run}\driver-{batch}.stderr.log"')
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          "-Arguments @{CommandLine=$env:DRIVER_CMD; CurrentDirectory=$env:DRIVER_CWD}; "
          "Write-Output ($r.ReturnValue.ToString() + ' ' + $r.ProcessId.ToString())")
    RUN.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, check=True,
                         env={**os.environ, 'DRIVER_CMD': cmd, 'DRIVER_CWD': cwd}).stdout.split()
    if out[0] != '0':
        raise RuntimeError(f'Win32_Process.Create failed: {out}')
    return {'pid': int(out[1]), 'command': cmd}


def run(batch: str, first: str | None, interval: float = 30.0):
    roots = V2Roots()
    use_v2_extractor()
    p = drv.paths(roots, batch)
    from r2ai.zh_full_ingest.__main__ import _stop_on_alert

    def log(**kw):
        line = json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, ensure_ascii=False, default=str)
        print(line, flush=True)
        with open(roots.guard(p['log']), 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    with drv.Heartbeat(roots, p['heartbeat'], interval) as hb, drv.ExitRecord(roots, p['exit'], hb):
        roots.preflight()
        init_manifest(roots)
        log(step='preflight', pid=os.getpid(), extractor=zextract.extractor_version()[:12])
        drv.run_batch(roots, batch, view=drv.ManifestView(roots, batch), ingest=base_ingest,
                      first=[d for d in (first or '').split(',') if d], log=log, beat=hb.beat, stop=_stop_on_alert(roots),
                      extract_guard=MemoryGuard(roots, minimum_gib=2.5, max_pauses=8), chunk_guard=MemoryGuard(roots),
                      embed_guard=MemoryGuard(roots, minimum_gib=3.0, max_pauses=8))
        log(step='adopt', **adopt(roots))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['cut', 'run', 'launch', 'status', 'adopt'])
    ap.add_argument('--batch')
    ap.add_argument('--domains')
    ap.add_argument('--first', default=None)
    a = ap.parse_args(argv)
    roots = V2Roots()
    if a.action == 'status':
        print(json.dumps(drv.driver_status(roots, a.batch), indent=1, default=str))
    elif a.action == 'launch':
        print(json.dumps(launch(a.batch, a.first), indent=1))
    elif a.action == 'run':
        run(a.batch, a.first)
    elif a.action == 'adopt':
        print(json.dumps(adopt(roots)))
    else:
        roots.preflight()
        init_manifest(roots)
        domains = a.domains.split(',') if a.domains else sorted(zcrawl.DOMAINS)
        print(json.dumps(base_ingest.cut_batch(roots, a.batch, domains), default=str, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
