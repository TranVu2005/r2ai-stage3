"""Ingest one more batch of fetched CC docs into its own segment and rebuild the A variants (S3 step 2).

  python -m r2ai.cc_longchau.batch --label cc-longchau-b2 --tag=-n2 [--min-new 10000] [--no-variants]

extract (new raw shards only) -> chunk (docs files no other segment used) -> embed (integrity checks per block) -> verify ->
first stage over ALL active segments -> rerank (logits of earlier runs reused, only new pairs hit the GPU) ->
VS-cc-A<tag> and VS-cc-A-cut115<tag>. Segment 20000 and the uploaded files are never touched. Every step is idempotent;
rerun after a stop. GPU busy / low RAM: the step is retried every 2 minutes (nothing is killed or preempted).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

from . import build, extract, retrieve
from .common import Roots, log_event

RETRY_S = 120


def retrying(fn, what, roots, *a, **kw):
    while True:
        try:
            return fn(*a, **kw)
        except RuntimeError as e:
            if not any(t in str(e) for t in ('GPU_BUSY', 'MEMORY_PAUSES')):
                raise
            log_event(roots, 'batch_wait', {'step': what, 'reason': str(e)[:200]})
            print(f'{what}: waiting ({str(e)[:80]})', flush=True)
            time.sleep(RETRY_S)


def new_ok(roots: Roots) -> int:
    """fetch.db ok rows whose shard has no docs file yet."""
    from vicrawl.shards import list_shards
    from .common import DOMAIN
    have = {p.name for p in roots.docs.glob('*.parquet')}
    shards = {p.name for p in list_shards(roots.raw, DOMAIN) if f"{DOMAIN}__{p.name.removesuffix('.jsonl.zst')}.parquet" not in have}
    c = sqlite3.connect(f'file:{roots.fetch_db.as_posix()}?mode=ro', uri=True)
    try:
        return sum(n for sh, n in c.execute("SELECT shard, COUNT(*) FROM fetch WHERE status='ok' GROUP BY shard") if sh in shards)
    finally:
        c.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--min-new', type=int, default=10000)
    ap.add_argument('--no-variants', action='store_true')
    a = ap.parse_args(argv)
    roots = Roots()
    n = new_ok(roots)
    zr = build.zroots(roots)
    try:
        build.ensure_segment(zr, a.label, create=False)
        started = True
    except ValueError:
        started = False
    if n < a.min_new and not started:
        print(f'only {n} new ok URLs (< {a.min_new}); nothing to do', flush=True)
        return 2
    t = {}
    t0 = time.time()
    s = extract.run(roots, workers=3)
    extract.consolidate_provenance(roots)
    t['extract_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    seg = build.ensure_segment(zr, a.label)
    if seg['status'] == 'allocated':
        build.chunk(roots, batch=a.label)
    t['chunk_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    if build.ensure_segment(zr, a.label, create=False)['status'] != 'active':
        retrying(build.embed, 'embed', roots, roots, batch=a.label)
    t['embed_s'] = round(time.time() - t0, 1)
    v = build.verify(roots, batch=a.label)
    if v['verbatim_bad'] or not v['chunks_sha256_ok'] or v.get('dense_nan') or v.get('sparse_empty_rows'):
        raise SystemExit(f'segment integrity failed: {json.dumps(v)[:400]}')
    t0 = time.time()
    run = retrieve.default_run(roots)
    if not (roots.run / run / 'first_stage_summary.json').exists():
        retrieve.first_stage(roots, run)
    t['first_stage_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    r = retrying(retrieve.rerank, 'rerank', roots, roots, run)
    t['rerank_s'] = round(time.time() - t0, 1)
    out = {'label': a.label, 'new_ok_urls': n, 'extract': {k: s[k] for k in ('docs', 'shards')}, 'verify': v, 'run': run, 'rerank': r, 'seconds': t}
    if not a.no_variants:
        from . import append_cc, cut_cc
        t0 = time.time()
        append_cc.main(['--variants', 'VS-cc-A', '--run', run, f'--tag={a.tag}'])
        cut_cc.main(['--target-mean', '115', '--run', run, f'--tag={a.tag}'])
        t['variants_s'] = round(time.time() - t0, 1)
    from .common import atomic_json
    atomic_json(roots, roots.run / f'batch_{a.label}.json', out)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
