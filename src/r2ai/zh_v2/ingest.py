"""Batch ingest of the v2 new-host crawl (state/zh_v2, data/raw_zh_v2) into the zh v2 index: the zh_full_ingest pipeline unchanged
(cut closed shards -> extract -> t256 chunker -> BGE-M3 embed, one segment per (batch, domain)), only the roots, the extractor
(r2ai.zh_v2.extract) and the segment-id base differ.

* Roots: raw/db/docs/state/run are zh-v2 private; segments land in data/index/zh_v2_segments, chunks in data/chunks_zh_v2.
* Segment ids start at 100 (zh_full uses 1..31) so global chunk ids never collide with the filtered zh_full segments.
* Chunker t256 and `min_body_tokens` 50 are the zh_full values, so the reranker threshold tau = 0.96 stays comparable.
* `adopt` registers finished segments in the v2 index manifest (all their docs are v2 by construction).

  python -m r2ai.zh_v2.ingest cut --batch v2n-001 [--domains a,b]
  python -m r2ai.zh_v2.ingest cut-zf --batch v2n-002 --domains cnkang.com,...   # zh_full rows never ingested, v2 ids only
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

from . import direct as zcrawl
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
    skipped = []
    seen = set()    # a doc already in the index (2% sample segment, earlier batch) must not be indexed twice
    for r in manifest['segments']:
        f = (zindex.CHUNKS_V2 / r['name'] / 'docs.parquet')
        if f.exists():
            seen.update(pl.read_parquet(f, columns=['doc_id'])['doc_id'].to_list())
    for seg in base_ingest.segment_rows(roots):
        if seg['name'] in have:
            continue
        docs = pl.read_parquet(roots.chunks / seg['name'] / 'docs.parquet', columns=['doc_id'])
        ids = set(docs['doc_id'].to_list())
        if ids & seen:
            if not ids <= seen:
                raise ValueError(f"{seg['name']}: {len(ids & seen)} of {len(ids)} docs already indexed; filter the batch")
            skipped.append(seg['name'])
            continue
        seen |= ids
        added.append({'name': seg['name'], 'segment_id': seg['segment_id'], 'batch': seg['batch'], 'domain': seg['domain'],
                      'rows_source': seg['rows'], 'rows_v2': seg['rows'], 'docs_v2': docs.height, 'source': 'zh_v2_ingest',
                      'global_start': base_ingest.global_start(seg['segment_id'])})
    if added:
        import numpy as np
        zindex.write_manifest(manifest['segments'] + added, np.array(zindex.scope.v2_zh()['id'].to_numpy()))
    return {'added': [a['name'] for a in added], 'skipped_duplicate': skipped, 'rows': sum(a['rows_v2'] for a in added), 'docs': sum(a['docs_v2'] for a in added)}


def launch(batch: str, first: str | None = None, crawl_db: str | Path | None = None) -> dict:
    roots = V2Roots(crawl_db=Path(crawl_db).resolve() if crawl_db else zcrawl.DB)
    roots.preflight()
    roots.guard(RUN)
    repo = str(Path(__file__).resolve().parents[3])
    cwd = str(Path(RUNS_DIR).resolve().parents[1])
    run = str(RUN.resolve())
    tail = f' --first {first}' if first else ''
    dbtail = f' --crawl-db {Path(crawl_db).resolve()}' if crawl_db else ''
    cmd = (f'cmd.exe /c "cd /d {cwd} && set PYTHONPATH={repo}\\src&& set PYTHONIOENCODING=utf-8&& '
           rf'.venv\Scripts\python.exe -B -m r2ai.zh_v2.ingest run --batch {batch}{tail}{dbtail} '
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


def _atomic_provenance(path: Path, rows: list[dict], roots: V2Roots):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ('doc_id', pa.int64()), ('doc_ids_v2', pa.list_(pa.int64())), ('url_norm', pa.string()),
        ('source', pa.string()), ('fetched_at', pa.float64()),
    ])
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression='zstd')
    with open(tmp, 'rb+') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_provenance(roots: V2Roots, batch: str) -> dict:
    """Write an additive provenance sidecar for every extracted doc and each indexed segment doc."""
    v2_ids = set(zindex.scope.v2_zh()['id'].to_list())
    direct_db = sqlite3.connect(Path(zcrawl.DB).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        fetched = {r[0]: (r[1], json.loads(r[2]), r[3]) for r in direct_db.execute(
            "SELECT url_norm,fetched_at,doc_ids,source FROM urls WHERE status IN ('ok','thin') AND source='direct'")}
    finally:
        direct_db.close()
    provenance, empty_docs, good_text_docs, extracted = [], 0, 0, 0
    for file in sorted((roots.docs / batch).glob('*/*.parquet')):
        rows = pl.read_parquet(file, columns=['doc_ids', 'url_norm', 'fetched_at', 'body', 'status', 'n_tokens_bge_m3']).to_dicts()
        for row in rows:
            ids = [int(i) for i in (row['doc_ids'] or [])]
            if not ids or not set(ids) <= v2_ids:
                raise ValueError(f'{batch}: extracted doc has an empty or out-of-v2 id group: {file} {ids[:5]}')
            if row['url_norm'] not in fetched:
                raise ValueError(f'{batch}: extracted URL has no direct provenance row: {row["url_norm"]}')
            fetched_at, source_ids, source = fetched[row['url_norm']]
            if source != 'direct' or source_ids != ids or fetched_at != row['fetched_at']:
                raise ValueError(f'{batch}: direct provenance disagrees with extracted doc {row["url_norm"]}')
            provenance.append({'doc_id': ids[0], 'doc_ids_v2': ids, 'url_norm': row['url_norm'],
                               'source': source, 'fetched_at': fetched_at})
            extracted += 1
            if not (row.get('body') or '').strip():
                empty_docs += 1
            if row.get('status') == 'ok' and int(row.get('n_tokens_bge_m3') or 0) >= base_ingest.MIN_BODY_TOKENS:
                good_text_docs += 1
    if len({r['doc_id'] for r in provenance}) != len(provenance):
        raise ValueError(f'{batch}: duplicate primary doc_id in provenance')
    batch_sidecar = roots.docs / batch / 'provenance.parquet'
    _atomic_provenance(batch_sidecar, provenance, roots)
    v2_doc_ids = {i for row in provenance for i in row['doc_ids_v2']}
    segments = []
    for seg in base_ingest.segment_rows(roots):
        if seg['batch'] != batch or seg['status'] == 'retired':
            continue
        docs_file = roots.chunks / seg['name'] / 'docs.parquet'
        docs = pl.read_parquet(docs_file, columns=['doc_id', 'doc_ids_group'])
        doc_ids = set(int(i) for i in docs['doc_id'].to_list())
        groups = {int(i) for group in docs['doc_ids_group'].to_list() for i in (group or [])}
        if not doc_ids <= v2_ids or not groups <= v2_ids:
            raise ValueError(f'{seg["name"]}: segment includes doc ids outside v2 provenance')
        sidecar = [r for r in provenance if r['doc_id'] in doc_ids]
        if len(sidecar) != len(doc_ids):
            raise ValueError(f'{seg["name"]}: provenance sidecar incomplete ({len(sidecar)} != {len(doc_ids)})')
        _atomic_provenance(roots.chunks / seg['name'] / 'provenance.parquet', sidecar, roots)
        segments.append({'name': seg['name'], 'docs': len(doc_ids), 'path': str(roots.chunks / seg['name']),
                         'docs_schema': pl.read_parquet_schema(docs_file).names()})
    return {'batch': batch, 'extracted_doc_groups': extracted, 'empty_docs': empty_docs,
            'good_text_docs': good_text_docs, 'v2_ids': len(v2_doc_ids),
            'provenance_schema': ['doc_id:int64', 'doc_ids_v2:list[int64]', 'url_norm:string',
                                  'source:string', 'fetched_at:float64'],
            'provenance': str(batch_sidecar), 'segments': segments}


def run(batch: str, first: str | None, interval: float = 30.0, crawl_db: str | Path | None = None):
    roots = V2Roots(crawl_db=Path(crawl_db).resolve() if crawl_db else zcrawl.DB)
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
        existing = base_ingest.connect(roots)
        try:
            has_batch = existing.execute('SELECT 1 FROM batches WHERE batch=?', (batch,)).fetchone()
        finally:
            existing.close()
        if not has_batch:
            log(step='cut', **base_ingest.cut_batch(roots, batch, list(zcrawl.HOSTS)))
        drv.run_batch(roots, batch, view=drv.ManifestView(roots, batch), ingest=base_ingest,
                      first=[d for d in (first or '').split(',') if d], log=log, beat=hb.beat, stop=_stop_on_alert(roots),
                      extract_guard=MemoryGuard(roots, minimum_gib=2.5, max_pauses=8), chunk_guard=MemoryGuard(roots),
                      embed_guard=MemoryGuard(roots, minimum_gib=3.0, max_pauses=8))
        provenance = write_provenance(roots, batch)
        log(step='provenance', **provenance)
        adopted = adopt(roots)
        log(step='adopt', **adopted)
        source_db = sqlite3.connect(Path(roots.crawl_db).resolve().as_uri() + '?mode=ro', uri=True)
        try:
            batch_url_count = source_db.execute('SELECT count(*) FROM urls').fetchone()[0]
        finally:
            source_db.close()
        direct_db = sqlite3.connect(Path(zcrawl.DB).resolve().as_uri() + '?mode=ro', uri=True)
        try:
            batch_v2_ids = sum(len(json.loads(row[0])) for row in direct_db.execute(
                'SELECT u.doc_ids FROM urls u JOIN ingest_urls i USING(url_norm) WHERE i.batch=?', (batch,)))
        finally:
            direct_db.close()
        report = {**provenance, 'source': 'direct', 'crawl_db': str(roots.crawl_db),
                  'batch_url_count': batch_url_count, 'batch_v2_id_count': batch_v2_ids,
                  'doc_count': provenance['extracted_doc_groups'], 'all_doc_ids_in_v2': True,
                  'empty_doc_count': provenance['empty_docs'], 'adopted': adopted,
                  'completed_at': time.time()}
        report_path = roots.guard(roots.run / f'handoff-{batch}.json')
        base_ingest.atomic_json(roots, report_path, report)
        log(step='handoff', report=str(report_path), doc_count=report['doc_count'],
            good_text_docs=report['good_text_docs'], empty_doc_count=report['empty_doc_count'],
            all_doc_ids_in_v2=report['all_doc_ids_in_v2'])
        return report


def cut_zh_full(roots: V2Roots, batch: str, domains: list[str], zf=None, v2_ids=None) -> dict:
    """Cut the zh_full crawl rows that are v2 zh docs and were never ingested (zh_full_ingest batch_urls), without touching zh_full.

    Reads state/zh_full/crawl.db (mode=ro) and the zh_full ingest manifest (mode=ro), writes a small filtered copy of the urls
    table under the zh-v2 ingest state and cuts from it against data/raw_zh_full: only rows with >= 1 v2 zh id are taken, so no
    out-of-v2 doc is chunked or embedded. Segments, docs and chunks land in the zh-v2 roots.
    """
    import dataclasses
    from r2ai.zh_full_ingest.common import Roots as ZfRoots
    zf = zf or ZfRoots()
    v2 = set(zindex.scope.v2_zh()['id'].to_list() if v2_ids is None else v2_ids)
    taken = set()
    m = base_ingest._ro(zf.manifest)
    try:
        taken = {r[0] for r in m.execute('SELECT url_norm FROM batch_urls')}
    finally:
        m.close()
    src = base_ingest._ro(zf.crawl_db)
    keep = []
    try:
        q = ','.join('?' * len(domains))
        for u, d, st, sh, ids in src.execute("SELECT url_norm, domain, status, shard, doc_ids FROM urls WHERE domain IN (%s) "
                                             "AND status IN ('ok','thin') AND shard IS NOT NULL" % q, domains):
            if u not in taken and any(i in v2 for i in json.loads(ids)):
                keep.append((u, d, st, sh, ids))
    finally:
        src.close()
    path = roots.guard(roots.state / f'cut_src_{batch}.db')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    c = sqlite3.connect(path)
    with c:
        c.execute('CREATE TABLE urls (url_norm TEXT PRIMARY KEY, domain TEXT, status TEXT, shard TEXT, doc_ids TEXT)')
        c.executemany('INSERT INTO urls VALUES (?,?,?,?,?)', keep)
    c.close()
    out = base_ingest.cut_batch(dataclasses.replace(roots, raw=zf.raw, crawl_db=path), batch, domains)
    out['rows_selected'] = len(keep)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['cut', 'cut-zf', 'run', 'launch', 'status', 'adopt'])
    ap.add_argument('--batch')
    ap.add_argument('--domains')
    ap.add_argument('--first', default=None)
    ap.add_argument('--crawl-db', default=None)
    a = ap.parse_args(argv)
    roots = V2Roots()
    if a.action == 'status':
        print(json.dumps(drv.driver_status(roots, a.batch), indent=1, default=str))
    elif a.action == 'launch':
        print(json.dumps(launch(a.batch, a.first, a.crawl_db), indent=1))
    elif a.action == 'run':
        print(json.dumps(run(a.batch, a.first, crawl_db=a.crawl_db), ensure_ascii=False, indent=1, default=str))
    elif a.action == 'adopt':
        print(json.dumps(adopt(roots)))
    elif a.action == 'cut-zf':
        roots.preflight()
        init_manifest(roots)
        print(json.dumps(cut_zh_full(roots, a.batch, a.domains.split(',')), default=str, indent=1))
    else:
        roots.preflight()
        init_manifest(roots)
        domains = a.domains.split(',') if a.domains else sorted(zcrawl.DOMAINS)
        print(json.dumps(base_ingest.cut_batch(roots, a.batch, domains), default=str, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
