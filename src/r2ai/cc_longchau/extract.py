"""Raw CC shards -> docs parquet with the vi pipeline's own extractor, plus a per-doc provenance table.

  python -m r2ai.cc_longchau.extract [--workers 3]

The extractor is whatever `configs/domains.yaml` assigns to nhathuoclongchau.com.vn through `Config.for_domain` (today:
`generic`, i.e. trafilatura with the vicrawl settings), run by `vicrawl.extract_pipeline.extract_records` — the same
function ExtractPipeline uses — so docs rows have the vi docs schema. A shard record is used only when fetch.db says that
shard holds the final fetch of its url_id (re-fetches never produce two docs). Ids must belong to the corpus and the URL
must equal the corpus URL; anything else is dropped and counted. Idempotent: a shard is re-extracted into the same file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.paths import CONFIG_DIR
from vicrawl.extract_pipeline import SCHEMA, extract_records
from vicrawl.settings import Config
from vicrawl.shards import list_shards, read_shard
from vicrawl.urlnorm import domain_of

from .common import DOMAIN, Roots, atomic_json, sha256_bytes
from .fetch import connect

PROV_SCHEMA = pa.schema([
    ('doc_id', pa.int64()), ('url', pa.string()), ('source', pa.string()), ('crawl', pa.string()), ('warc_filename', pa.string()),
    ('warc_offset', pa.int64()), ('warc_length', pa.int64()), ('warc_date', pa.string()), ('capture_time_utc', pa.string()),
    ('warc_record_id', pa.string()), ('warc_payload_digest', pa.string()), ('record_sha256', pa.string()), ('body_sha256', pa.string()),
    ('content_type', pa.string()), ('truncated', pa.string()), ('raw_shard', pa.string()), ('extractor', pa.string()),
    ('extractor_name', pa.string()), ('trafilatura', pa.string()), ('status', pa.string()), ('n_chars', pa.int32()),
    ('n_tokens_bge_m3', pa.int32()), ('text_sha256', pa.string())])


def extractor_name() -> str:
    return Config.load(CONFIG_DIR / 'domains.yaml').for_domain(DOMAIN).extractor


def _trafilatura_version() -> str:
    try:
        import trafilatura
        return trafilatura.__version__
    except Exception:  # noqa: BLE001
        return ''


def extract_shard(path: str, name: str, allowed: dict, corpus: dict) -> tuple[list[dict], list[dict], dict]:
    """-> (docs rows, provenance rows, drop counters) for the records of one shard that are the final fetch of their id."""
    shard = Path(path).name
    records, drops = [], {'superseded': 0, 'not_corpus': 0}
    for r in read_shard(Path(path)):
        uid = r['doc_ids'][0]
        if allowed.get(uid) != shard:
            drops['superseded'] += 1
        elif corpus.get(uid) != r['url'] or r['domain'] != DOMAIN:
            drops['not_corpus'] += 1
        else:
            records.append(r)
    rows = extract_records(records, {DOMAIN: name})
    by_id = {r['doc_ids'][0]: r for r in records}
    tv, prov = _trafilatura_version(), []
    for d in rows:
        r = by_id[d['doc_ids'][0]]
        c = r['cc']
        prov.append({'doc_id': d['doc_ids'][0], 'url': d['url'], 'source': 'common_crawl', 'crawl': c['crawl'], 'warc_filename': c['warc_filename'],
                     'warc_offset': c['warc_offset'], 'warc_length': c['warc_length'], 'warc_date': c['warc_date'],
                     'capture_time_utc': datetime.fromtimestamp(r['fetched_at'], timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                     'warc_record_id': c['warc_record_id'], 'warc_payload_digest': c['warc_payload_digest'], 'record_sha256': c['record_sha256'],
                     'body_sha256': c['body_sha256'], 'content_type': c['content_type'], 'truncated': c['truncated'], 'raw_shard': shard,
                     'extractor': d['extractor'], 'extractor_name': name, 'trafilatura': tv, 'status': d['status'], 'n_chars': len(d['body']),
                     'n_tokens_bge_m3': d['n_tokens_bge_m3'], 'text_sha256': sha256_bytes(d['body'].encode('utf-8'))})
    return rows, prov, drops


def _job(args):
    return args[0], extract_shard(*args)


def _write(roots: Roots, path: Path, table: pa.Table):
    path = roots.guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = roots.guard(path.with_name(path.name + '.tmp'))
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, path)


def run(roots: Roots, *, workers=3, only=None, force=False) -> dict:
    roots.preflight()
    t0 = time.time()
    db = connect(roots)
    allowed = dict(db.execute("SELECT url_id, shard FROM fetch WHERE status='ok'").fetchall())
    corpus = pd.read_parquet(roots.corpus)
    corpus = corpus[corpus.url.map(domain_of) == DOMAIN]
    corpus = dict(zip(corpus.id.tolist(), corpus.url.tolist()))
    name = extractor_name()
    shards = [p for p in list_shards(roots.raw, DOMAIN) if only is None or p.name in only]
    done = {p.name for p in roots.docs.glob('*.parquet')} if roots.docs.exists() and not force else set()
    shards = [p for p in shards if f"{DOMAIN}__{p.name.removesuffix('.jsonl.zst')}.parquet" not in done]
    jobs = [(str(p), name, allowed, corpus) for p in shards]
    totals = {'shards': len(shards), 'docs': 0, 'superseded': 0, 'not_corpus': 0, 'status': {}}
    with ProcessPoolExecutor(workers) as ex:
        for path, (rows, prov, drops) in (map(_job, jobs) if workers == 1 else ex.map(_job, jobs)):
            stem = Path(path).name.removesuffix('.jsonl.zst')
            if rows:
                _write(roots, roots.docs / f'{DOMAIN}__{stem}.parquet', pa.Table.from_pylist(rows, schema=SCHEMA))
                _write(roots, roots.run / 'provenance' / f'{stem}.parquet', pa.Table.from_pylist(prov, schema=PROV_SCHEMA))
            totals['docs'] += len(rows)
            totals['superseded'] += drops['superseded']
            totals['not_corpus'] += drops['not_corpus']
            for r in rows:
                totals['status'][r['status']] = totals['status'].get(r['status'], 0) + 1
            print(f'{totals["docs"]} docs ({time.time() - t0:.0f}s)', flush=True)
    totals.update(extractor=name, trafilatura=_trafilatura_version(), seconds=round(time.time() - t0, 1))
    atomic_json(roots, roots.run / 'extract_summary.json', totals)
    return totals


def consolidate_provenance(roots: Roots) -> dict:
    """provenance/*.parquet -> provenance.parquet (one row per doc, sorted by doc_id); every doc in docs must be present."""
    files = sorted((roots.run / 'provenance').glob('*.parquet'))
    t = pa.concat_tables([pq.read_table(f) for f in files]).to_pandas().sort_values('doc_id')
    if t.doc_id.duplicated().any():
        raise ValueError(f'{int(t.doc_id.duplicated().sum())} duplicate doc_id in provenance')
    docs = pa.concat_tables([pq.read_table(f, columns=['doc_ids']) for f in sorted(roots.docs.glob('*.parquet'))])
    ids = sorted(x[0] for x in docs.column('doc_ids').to_pylist())
    if ids != t.doc_id.tolist():
        raise ValueError('provenance doc ids differ from docs doc ids')
    _write(roots, roots.run / 'provenance.parquet', pa.Table.from_pandas(t, schema=PROV_SCHEMA, preserve_index=False))
    return {'docs': len(t), 'by_crawl': t.crawl.value_counts().to_dict()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--force', action='store_true', help='re-extract shards that already have a docs file')
    a = ap.parse_args(argv)
    roots = Roots()
    s = run(roots, workers=a.workers, force=a.force)
    s['provenance'] = consolidate_provenance(roots)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
