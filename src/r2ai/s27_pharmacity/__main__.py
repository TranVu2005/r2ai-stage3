"""CLI: python -m r2ai.s27_pharmacity {map,sample,crawl,match-report,extract,chunk,verify,embed,status}."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

os.environ.setdefault('HF_HUB_OFFLINE', '1')

from pathlib import Path

from .pipeline import (BATCH, USER_AGENT, Roots, atomic_json, chunk_batch, embed_batch, extract_batch,
                       crawl_halt_report, current_crawl_status, map_batch, sample_and_crawl,
                       title_match_report, verify_batch)


def _manifest(roots: Roots) -> dict:
    import hashlib

    roots.preflight()
    crawl_status, crawl_coverage = current_crawl_status(roots)
    halt_path = roots.reports / 'crawl_halted.json'
    crawl_path = roots.reports / 'crawl.json'
    value = {'batch': BATCH, 'contract': 'S15', 'created_or_updated_utc': datetime.now(timezone.utc).isoformat(
        timespec='seconds').replace('+00:00', 'Z'),
        'crawl': {'user_agent': USER_AGENT, 'minimum_request_interval_seconds': 1.05, 'proxy': False},
        'extract': {'body_selector': '#pmc-content-blog-detail', 'min_good_body_codepoints': 200},
        'chunk': {'target_tokens': 256, 'min_body_tokens_vi': 50},
        'crawl_status': crawl_status, 'crawl_coverage': crawl_coverage,
        'raw_pages': len(list(roots.raw.glob('*.html.gz'))) if roots.raw.exists() else 0,
        'outputs': {}, 'shared_index_written': False}
    for label, path in (('mapping', roots.mapping), ('docs', roots.docs / 'docs.parquet'),
                        ('provenance', roots.docs / 'provenance.parquet'),
                        ('provenance_all', roots.docs / 'provenance_all.parquet'),
                        ('chunk_docs', roots.chunks / 'docs.parquet'),
                        ('chunks_t256', roots.chunks / 'chunks_t256.parquet'),
                        ('vectors', roots.vectors / 'dense_doc_level.npz')):
        if path.exists():
            with open(roots.guard(path), 'rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            value['outputs'][label] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}
    for label, path in (('mapping_report', roots.reports / 'mapping.json'), ('sample', roots.reports / 'sample.json'),
                        ('crawl', crawl_path), ('crawl_halted', halt_path), ('title_match', roots.reports / 'title_match.json'),
                        ('extract', roots.reports / 'extract.json'), ('chunk', roots.reports / 'chunk.json'),
                        ('verify', roots.reports / 'verify.json'), ('embed', roots.reports / 'embed.json')):
        if path.exists():
            value['outputs'][label] = {'path': str(path)}
    atomic_json(roots, roots.root / 'batch_manifest.json', value)
    return value


def _status(roots: Roots) -> dict:
    import sqlite3
    if not roots.state.exists():
        crawl_status, coverage = current_crawl_status(roots)
        return {'batch': BATCH, 'state': crawl_status, 'crawl_coverage': coverage}
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        counts = dict(c.execute('SELECT status,count(*) FROM pages GROUP BY status'))
        n = c.execute('SELECT count(*) FROM pages').fetchone()[0]
    finally:
        c.close()
    crawl_status, coverage = current_crawl_status(roots)
    return {'batch': BATCH, 'state': crawl_status, 'records': n, 'status_counts': counts,
            'crawl_coverage': coverage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=['map', 'sample', 'crawl', 'halt-report', 'match-report', 'extract', 'chunk', 'verify', 'embed', 'manifest', 'status'])
    ap.add_argument('--corpus', type=Path, help='corpus_v2_ids.parquet; defaults to R2AI_DATA_DIR/corpus_v2_ids.parquet')
    ap.add_argument('--progress-every', type=int, default=100)
    ap.add_argument('--resume-after-block', action='store_true', help='explicitly retry a saved 403/429/CAPTCHA checkpoint')
    a = ap.parse_args(argv)
    roots = Roots()
    if a.step == 'map':
        result = map_batch(roots, a.corpus)
    elif a.step in ('sample', 'crawl'):
        result = sample_and_crawl(roots, sample_only=a.step == 'sample', corpus_path=a.corpus,
                                  progress_every=a.progress_every,
                                  resume_after_block=(a.step == 'crawl' and a.resume_after_block))
    elif a.step == 'halt-report':
        result = crawl_halt_report(roots)
    elif a.step == 'match-report':
        result = title_match_report(roots)
    elif a.step == 'extract':
        result = extract_batch(roots)
    elif a.step == 'chunk':
        result = chunk_batch(roots)
    elif a.step == 'verify':
        result = verify_batch(roots)
    elif a.step == 'embed':
        result = embed_batch(roots)
    elif a.step == 'manifest':
        result = _manifest(roots)
    else:
        result = _status(roots)
    # Windows terminals commonly use cp1252; keep the CLI stdout portable while reports stay UTF-8.
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
