"""S9 step 6: top hosts per language of corpus v2 as markdown tables (stdout).

    python scripts/s9_top_hosts.py [--top 30]
"""
from __future__ import annotations

from r2ai import corpus_v2 as C

import argparse

import polars as pl


def table(df: pl.DataFrame, lang: str, top: int) -> str:
    sub = df.filter(pl.col('lang') == lang)
    g = sub.group_by('host').agg(pl.len().alias('n'), pl.col('url').first().alias('example')).sort(['n', 'host'], descending=[True, False])
    shown = g.head(top)
    lines = [f'### {lang}: {sub.height} URL, {g.height} host (hiển thị {shown.height}, chiếm {shown["n"].sum()} URL)', '',
             '| # | host | URL | % của lang | ví dụ |', '|--:|:--|--:|--:|:--|']
    for i, (h, n, ex) in enumerate(shown.iter_rows(), 1):
        lines.append(f'| {i} | {h} | {n} | {100 * n / sub.height:.2f} | {ex} |')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=30)
    args = ap.parse_args(argv)
    df = pl.read_parquet(C.IDS_PATH)
    print('\n\n'.join(table(df, lang, args.top) for lang in ('en', 'zh')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
