"""CLI: python -m r2ai.zh_full_ingest <step> ...   (run from the zh-full-ingest worktree, PowerShell)

  convert-old-sample                      old sample index -> retired segment 1 (Z-append control)
  cut-sample --batch sample-v2            the finished 2% sample, zh_sample record rule
  cut --batch b001 --domains a,b          closed committed zh-full shards (crawl.db mode=ro)
  extract --batch b001                    one worker; resumable; RAM guard
  chunk --batch b001 --domain d           t256 verbatim chunker, --batch-docs 5000
  embed --segment NAME                    BGE-M3 fp16 batch 16, block 1000; resumable; RAM/GPU guard
  copy-sample --segment NAME --domain d   reuse old vectors after the docs + chunk data gates
  status                                  manifest summary
"""
from __future__ import annotations

import argparse
import json
import sys

from .common import MemoryGuard, Roots, exclusive


def _stop_on_alert(roots):
    def check(_where):
        if (roots.run / 'monitor_alert.json').exists():
            raise SystemExit('CRAWL_SLOWDOWN: monitor_alert.json present; stopping before the next unit')
    return check


def status(roots):
    from .ingest import connect
    c = connect(roots)
    try:
        return {'batches': [dict(zip(('batch', 'kind', 'snapshot_at'), r)) for r in c.execute('SELECT batch, kind, snapshot_at FROM batches')],
                'shards': [dict(zip(('batch', 'domain', 'status', 'n', 'docs'), r)) for r in
                           c.execute('SELECT batch, domain, status, count(*), sum(n_docs) FROM shards GROUP BY 1,2,3')],
                'segments': [dict(zip(('segment_id', 'name', 'status', 'rows', 'vectors'), r)) for r in
                             c.execute('SELECT segment_id, name, status, rows, vectors FROM segments ORDER BY segment_id')]}
    finally:
        c.close()


def memory_guard(roots, min_available_gib=None, max_pauses=3):
    kwargs = {} if min_available_gib is None else {'minimum_gib': min_available_gib}
    return MemoryGuard(roots, max_pauses=max_pauses, **kwargs)


def main(argv=None):
    from . import ingest
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['convert-old-sample', 'cut-sample', 'cut', 'extract', 'chunk', 'embed',
                                     'copy-sample', 'docs-gate', 'status'])
    ap.add_argument('--batch')
    ap.add_argument('--domains')
    ap.add_argument('--domain')
    ap.add_argument('--segment')
    ap.add_argument('--batch-docs', type=int, default=5000)
    ap.add_argument('--min-available-gib', type=float, default=None, help='RAM guard threshold (default MIN_AVAILABLE_GIB)')
    ap.add_argument('--max-pauses', type=int, default=3, help='RAM guard pauses allowed per 30 min before stopping')
    a = ap.parse_args(argv)
    roots = Roots()
    roots.preflight()
    stop = _stop_on_alert(roots)
    memory = memory_guard(roots, a.min_available_gib, a.max_pauses)
    if a.step == 'convert-old-sample':
        with exclusive(roots, 'segments'):
            out = ingest.convert_old_sample(roots)
    elif a.step == 'cut-sample':
        out = ingest.cut_sample(roots, a.batch)
    elif a.step == 'cut':
        out = ingest.cut_batch(roots, a.batch, a.domains.split(','))
    elif a.step == 'extract':
        with exclusive(roots, 'extract'):
            out = ingest.extract_batch(roots, a.batch, memory=memory, before_shard=stop)
    elif a.step == 'chunk':
        with exclusive(roots, 'chunk'):
            memory.check('chunk')
            out = ingest.chunk_segment(roots, a.batch, a.domain, batch_docs=a.batch_docs)
    elif a.step == 'embed':
        with exclusive(roots, 'gpu'):
            out = ingest.embed_segment(roots, a.segment, memory=memory, before_block=stop)
    elif a.step == 'copy-sample':
        with exclusive(roots, 'segments'):
            out = ingest.copy_sample_vectors(roots, a.segment, a.domain)
            out = {k: v for k, v in out.items() if k != 'old_rows'}
    elif a.step == 'docs-gate':
        out = ingest.compare_sample_docs(roots, a.batch, a.domain)
    else:
        out = status(roots)
    print(json.dumps(out, ensure_ascii=True, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
