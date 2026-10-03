"""Resumable Vietnamese-domain crawler.  Fetch + shard only; run `python extract.py` for text.

  python crawl.py init                  # classify domains, de-duplicate urls, fill state/crawl.db
  python crawl.py run [--domains a,b] [--group vi] [--hours 3 | --until 23:30] [--keep-awake] [--limit N]
  python crawl.py status
  python crawl.py approve <domain>      # release a domain that stopped at the QA gate
  python crawl.py reset-errors <domain>
  python crawl.py report
"""
from __future__ import annotations

from r2ai.paths import RAW_WRITE_DIR, assert_writable, require_inputs

import argparse
import asyncio
import logging
import logging.handlers
import sys
from pathlib import Path

from vicrawl.settings import Config, paths_from_args

log = logging.getLogger('vicrawl')


def setup_logging(log_dir: Path, quiet: bool = False):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    root = logging.getLogger('vicrawl')
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(log_dir / 'crawl.log', maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not quiet:
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001
            pass
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('hpack').setLevel(logging.WARNING)


def add_path_args(p):
    for name in ('state-dir', 'raw-dir', 'docs-dir', 'out-dir', 'config', 'corpus', 'log-dir'):
        p.add_argument('--' + name)


def cmd_init(args):
    from vicrawl.domains import build_url_table, classify_domains, live_probe, load_probed_lang, read_vi_domains, write_domain_files
    from vicrawl.state import StateDB
    paths = paths_from_args(args)
    csv_path = paths.vi_domains_csv
    if args.reclassify or not csv_path.exists():
        probed = load_probed_lang(paths.out_dir)
        results = classify_domains(paths.corpus, probed_lang=probed, probe_fn=None if args.no_probe else live_probe)
        write_domain_files(results, paths.out_dir)
        log.info('wrote %s (%d vi domains)', csv_path, sum(r['is_vi'] and not r['deferred'] for r in results))
    domains = read_vi_domains(csv_path)
    groups = build_url_table(paths.corpus, domains)
    db = StateDB(paths.db)
    added = db.add_urls(groups)
    log.info('urls: %d groups for %d domains, %d new rows inserted', len(groups), len(domains), added)
    return 0


def cmd_run(args):
    from vicrawl.engine import Engine, RunOptions
    paths = paths_from_args(args)
    from vicrawl.state import StateDB
    if not paths.db.exists() or StateDB(paths.db, readonly=True).conn.execute('select count(*) from urls').fetchone()[0] == 0:
        log.info('state empty: initialising first')
        cmd_init(args)
    opts = RunOptions(
        domains={d.strip().lower().removeprefix('www.') for d in args.domains.split(',') if d.strip()} if args.domains else None,
        group=args.group, limit=args.limit, max_conns=args.max_conns, hours=args.hours, until=args.until,
        probe_url=args.probe_url or '', skip_report=args.no_report, flush_docs=args.flush_docs, flush_secs=args.flush_secs, qa_docs=args.qa_docs, pause_s=args.pause_s)
    if args.egress_url:
        opts.egress_url = args.egress_url
    cfg = Config.load(paths.config)
    eng = Engine(paths, cfg, opts)
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except ImportError:
        pass
    if args.keep_awake:
        try:
            from wakepy import keep
            with keep.running() as mode:
                log.info('keep-awake: %s', 'active' if getattr(mode, 'active', True) else 'NOT active (platform refused)')
                return asyncio.run(eng.run())
        except ImportError:
            log.warning('--keep-awake needs `pip install wakepy`; continuing without it')
    return asyncio.run(eng.run())


def cmd_status(args):
    from vicrawl.report import status_text
    print(status_text(paths_from_args(args).db))
    return 0


def cmd_report(args):
    from vicrawl.report import build_report
    p = paths_from_args(args)
    build_report(p.db, p.docs_dir, p.out_dir / 'crawl_vi_report.md')
    return 0


def cmd_approve(args):
    from vicrawl.state import StateDB
    db = StateDB(paths_from_args(args).db)
    if db.get_domain(args.domain) is None:
        print(f'unknown domain {args.domain}')
        return 2
    db.upsert_domain(args.domain, approved=1)
    db.conn.execute("UPDATE domains SET state='active', halt_reason=NULL WHERE domain=? AND state='awaiting_qa'", (args.domain,))
    print(f'{args.domain} approved. A running crawler resumes it within ~10s; otherwise start `python crawl.py run`.')
    return 0


def cmd_reset_errors(args):
    from vicrawl.state import StateDB
    db = StateDB(paths_from_args(args).db)
    n = db.reset_errors(args.domain)
    print(f'{args.domain}: {n} error URL(s) back to pending, halt cleared')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    add_path_args(p)
    p.add_argument('--reclassify', action='store_true', help='redo domain classification (fetches 3 urls for unprobed domains)')
    p.add_argument('--no-probe', action='store_true', help='skip live language probes of unprobed domains')
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser('run')
    add_path_args(p)
    p.add_argument('--domains')
    p.add_argument('--group')
    p.add_argument('--limit', type=int, help='crawl only the first N (pseudo-random order) urls of each domain')
    p.add_argument('--hours', type=float)
    p.add_argument('--until', help='HH:MM local time')
    p.add_argument('--keep-awake', action='store_true')
    p.add_argument('--max-conns', type=int, default=64)
    p.add_argument('--egress-url')
    p.add_argument('--probe-url')
    p.add_argument('--no-report', action='store_true')
    p.add_argument('--pause-s', type=float, default=60.0, help='domain pause after 429/503 (spec: 60)')
    p.add_argument('--qa-docs', type=int, default=300, help='QA gate threshold (ok docs) for domains with >=5000 urls')
    p.add_argument('--flush-docs', type=int, default=500, help='write a shard every N docs ...')
    p.add_argument('--flush-secs', type=float, default=30.0, help='... or every N seconds')
    p.add_argument('--reclassify', action='store_true')
    p.add_argument('--no-probe', action='store_true')
    p.set_defaults(fn=cmd_run)
    for name, fn in (('status', cmd_status), ('report', cmd_report)):
        p = sub.add_parser(name)
        add_path_args(p)
        p.set_defaults(fn=fn)
    for name, fn in (('approve', cmd_approve), ('reset-errors', cmd_reset_errors)):
        p = sub.add_parser(name)
        add_path_args(p)
        p.add_argument('domain')
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    if args.cmd != 'status':
        if not args.raw_dir:
            args.raw_dir = str(RAW_WRITE_DIR)
        p = paths_from_args(args)
        for target in (p.state_dir, p.raw_dir, p.docs_dir, p.out_dir, p.log_dir):
            assert_writable(target)
        if args.cmd in ('init', 'run'):
            require_inputs(p.config, p.corpus)
        elif args.cmd in ('approve', 'reset-errors', 'report'):
            require_inputs(p.db)
        setup_logging(p.log_dir, quiet=args.cmd == 'report')
    return args.fn(args) or 0


if __name__ == '__main__':
    sys.exit(main())
