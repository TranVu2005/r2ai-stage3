"""Export the S37 URL list: S21 queue (75,410 pending 120ask URLs) minus doc_ids that already have text in canonical.

    python -m r2ai.s37.urls [--out out/runs/s37/urls.tsv]

urls.tsv columns: doc_id, url, doc_ids. `doc_id` is the smallest id of the URL group, `doc_ids` all ids (comma separated).
A URL is dropped only when every id of its group has text in canonical (which already holds the 2,018 webMedQA docs).
"""
from __future__ import annotations

from r2ai.paths import DATA_DIR, RUNS_DIR, assert_writable

import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl

TARGETS = RUNS_DIR / 'zh-v2-direct' / 'targets.jsonl'
CANONICAL = DATA_DIR / 'canonical-20261018' / 'docs.parquet'
OUT = RUNS_DIR / 's37' / 'urls.tsv'
HOST = '120ask.com'


def build(targets: Path = TARGETS, canonical: Path = CANONICAL) -> tuple[list[tuple[int, str, list[int]]], dict]:
    have = pl.read_parquet(canonical, columns=['doc_id', 'doc_text']).filter(pl.col('doc_text').fill_null('') != '')
    have_ids = set(have['doc_id'].to_list())
    rows, stats = [], {'target_urls': 0, 'target_ids': 0, 'dropped_urls_with_text': 0, 'dropped_ids_with_text': 0}
    for line in targets.read_text('utf-8').splitlines():
        t = json.loads(line)
        if t['domain'] != HOST:
            continue
        ids = sorted(t['doc_ids'])
        stats['target_urls'] += 1
        stats['target_ids'] += len(ids)
        got = [i for i in ids if i in have_ids]
        stats['dropped_ids_with_text'] += len(got)
        if len(got) == len(ids):
            stats['dropped_urls_with_text'] += 1
            continue
        rows.append((ids[0] if ids[0] not in have_ids else next(i for i in ids if i not in have_ids), t['url'], ids))
    stats['urls_out'] = len(rows)
    return rows, stats


def write(rows, out: Path) -> str:
    out = assert_writable(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ''.join(f"{d}\t{u}\t{','.join(map(str, ids))}\n" for d, u, ids in rows)
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_bytes(body.encode('utf-8'))
    os.replace(tmp, out)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(argv)
    rows, stats = build()
    stats['sha256'] = write(rows, a.out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
