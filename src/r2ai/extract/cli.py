"""Incremental extraction of crawled shards into data/docs_vi/*.parquet (safe to run beside the crawler).

  python extract.py [--watch] [--workers N]      # only shards not processed yet
  python extract.py qa <domain>                  # out/qa_extract/<domain>.md (also written by the crawler's QA gate)
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from vicrawl.settings import Config, paths_from_args, safe_name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', default='run', choices=['run', 'qa'])
    ap.add_argument('domain', nargs='?')
    ap.add_argument('--watch', action='store_true', help='keep polling for new shards')
    ap.add_argument('--interval', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    for name in ('state-dir', 'raw-dir', 'docs-dir', 'out-dir', 'config', 'corpus', 'log-dir'):
        ap.add_argument('--' + name)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('vicrawl').setLevel(logging.INFO)
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:  # noqa: BLE001
        pass
    paths = paths_from_args(args)
    for target in (paths.state_dir, paths.docs_dir, paths.out_dir, paths.log_dir):
        assert_writable(target)
    require_inputs(paths.config, paths.raw_dir)
    if not any(paths.raw_dir.rglob('*.jsonl.zst')):
        raise ValueError(f'Missing or empty shard input: {paths.raw_dir}')
    cfg = Config.load(paths.config)
    from vicrawl.extract_pipeline import ExtractPipeline, build_qa_report
    if args.command == 'qa':
        if not args.domain:
            ap.error('qa needs a domain')
        out = build_qa_report(args.domain, raw_dir=paths.raw_dir, out_path=paths.qa_dir / f'{safe_name(args.domain)}.md', cfg=cfg)
        print(out)
        return 0
    pipe = ExtractPipeline(paths.raw_dir, paths.docs_dir, paths.state_dir, cfg, workers=args.workers)
    while True:
        n = pipe.run_once()
        if n:
            logging.getLogger('vicrawl.extract').info('processed %d new shard(s)', n)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
