#!/usr/bin/env python3
"""List the URLs that are not `ok` after the crawl so far (run next to crawler.py, on the VPS or locally with a copy of data/log.jsonl).

    .venv/bin/python remaining.py --urls urls.tsv --log data/log.jsonl

Writes:
  urls_remaining.tsv   header + one row per URL: url, doc_id, status   (status = failed | untried)
  queue_remaining.tsv  doc_id<TAB>url, the format `crawler.py run --urls` reads (use it as urls.tsv for a fresh crawl of the rest)
`gone` (404) is final and is not listed; `ok` is never listed. Prints the counts.
"""
import argparse
import json
from pathlib import Path

from crawler import read_log, read_queue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urls', default='urls.tsv')
    ap.add_argument('--log', default='data/log.jsonl')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    queue = read_queue(Path(a.urls))
    latest: dict[str, dict] = {}
    for r in read_log(Path(a.log)):
        latest[r['doc_id']] = r
    rows, counts = [], {'ok': 0, 'gone': 0, 'failed': 0, 'untried': 0}
    for doc_id, url in queue:
        r = latest.get(doc_id)
        st = r['status'] if r else 'untried'
        counts[st] = counts.get(st, 0) + 1
        if st in ('failed', 'untried'):
            rows.append((url, doc_id, st))
    out = Path(a.out)
    (out / 'urls_remaining.tsv').write_text('url\tdoc_id\tstatus\n' + ''.join(f'{u}\t{d}\t{s}\n' for u, d, s in rows), encoding='utf-8')
    (out / 'queue_remaining.tsv').write_text(''.join(f'{d}\t{u}\n' for u, d, s in rows), encoding='utf-8')
    print(json.dumps({'queue': len(queue), 'remaining_rows': len(rows), **counts}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
