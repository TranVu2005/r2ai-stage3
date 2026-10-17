"""Which crawl.db rows belong to the zh part of corpus v2, per-host status table and the out-of-scope defer plan.

A crawl.db row is one normalised URL carrying `doc_ids` (all corpus ids that normalise to it). A row is in scope when at
least one of its ids is a v2 zh id; a mixed row (v2 and non-v2 ids) is still needed for the v2 doc, so it stays.
Read-only on crawl.db; writes only the parquet/CSV/JSON it is told to.

    python -m r2ai.zh_v2.scope table [--out DIR]
    python -m r2ai.zh_v2.scope plan  --out DIR      # rows to defer (pending/network_error with no v2 id)
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, STATE_DIR, WORK_DATA_DIR, assert_writable

import argparse
import json
import sqlite3
from pathlib import Path

import polars as pl

CRAWL_DB = STATE_DIR / 'zh_full' / 'crawl.db'
MANIFEST = STATE_DIR / 'zh_full_ingest' / 'manifest.db'
CHUNKS = WORK_DATA_DIR / 'chunks_zh_full'
IDS_PATH = WORK_DATA_DIR / 'corpus_v2_ids.parquet'
RUN = RUNS_DIR / 'zh-v2'
OPEN = ('pending', 'network_error')
OTHER = ('network_error', 'soft404_or_home', 'dead_origin', 'blocked_4xx', 'in_progress')
HELD = {'qihuangzhishu.com': 'D5', 'wujue.com': 'hoãn', 'zydcd.com': 'hoãn'}


def ro(path=CRAWL_DB):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def v2_zh(path=IDS_PATH) -> pl.DataFrame:
    return pl.read_parquet(path, columns=['id', 'host', 'lang']).filter(pl.col('lang') == 'zh').drop('lang')


def crawl_rows(conn) -> pl.DataFrame:
    rows = conn.execute('SELECT url_norm, domain, status, doc_ids, next_try_at IS NOT NULL FROM urls').fetchall()
    return pl.DataFrame(rows, schema=['url_norm', 'domain', 'status', 'doc_ids', 'retry'], orient='row')


def explode_ids(rows: pl.DataFrame) -> pl.DataFrame:
    return (rows.with_columns(pl.col('doc_ids').str.strip_chars('[]').str.split(','))
            .explode('doc_ids', empty_as_null=True).with_columns(pl.col('doc_ids').cast(pl.Int64).alias('id')).drop('doc_ids'))


def v2_flags(rows: pl.DataFrame, zh: pl.DataFrame) -> pl.DataFrame:
    """One line per crawl row with n_ids / n_v2 (how many of its ids are v2 zh ids)."""
    ids = explode_ids(rows).join(zh.select('id').with_columns(pl.lit(1).alias('v2')), on='id', how='left')
    n = ids.group_by('url_norm').agg(pl.len().alias('n_ids'), pl.col('v2').fill_null(0).sum().alias('n_v2'))
    return rows.join(n, on='url_norm', how='left')


def status_class(col='status') -> pl.Expr:
    s = pl.col(col)
    return (pl.when(s == 'ok').then(pl.lit('ok')).when(s == 'thin').then(pl.lit('thin'))
            .when(s == 'pending').then(pl.lit('pending')).when(s == 'deferred_select').then(pl.lit('deferred'))
            .when(s == 'absent').then(pl.lit('absent')).otherwise(pl.lit('other')))


def active_segments(manifest=MANIFEST) -> list[dict]:
    c = sqlite3.connect(Path(manifest).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        return [dict(zip(('segment_id', 'name', 'batch', 'domain', 'rows'), r)) for r in c.execute(
            "SELECT segment_id, name, batch, domain, rows FROM segments WHERE status='active' ORDER BY segment_id")]
    finally:
        c.close()


def indexed_docs(segments=None, chunks=CHUNKS) -> pl.DataFrame:
    """(segment, doc_id, id): every id of every doc_ids_group of the docs chunked into an ACTIVE segment."""
    import pyarrow.parquet as pq
    parts = []
    for seg in segments if segments is not None else active_segments():
        t = pl.from_arrow(pq.read_table(Path(chunks) / seg['name'] / 'docs.parquet', columns=['doc_id', 'doc_ids_group']))
        parts.append(t.explode('doc_ids_group').rename({'doc_ids_group': 'id'}).with_columns(pl.lit(seg['name']).alias('segment')))
    return pl.concat(parts)


def host_table(zh: pl.DataFrame, rows: pl.DataFrame, indexed: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per v2 host: v2 ids by crawl status (an id = one v2 URL), plus the crawl rows (fetch units) still open."""
    ids = explode_ids(rows).select('id', 'url_norm', 'domain', 'status')
    j = zh.join(ids, on='id', how='left').with_columns(pl.col('status').fill_null('absent'))
    j = j.with_columns(status_class().alias('cls'))
    t = j.group_by('host', 'cls').len().pivot(on='cls', index='host', values='len').fill_null(0)
    for c in ('ok', 'thin', 'pending', 'deferred', 'absent', 'other'):
        if c not in t.columns:
            t = t.with_columns(pl.lit(0, dtype=pl.UInt32).alias(c))
    t = t.with_columns(pl.sum_horizontal('ok', 'thin', 'pending', 'deferred', 'absent', 'other').alias('total'))
    domain = j.filter(pl.col('domain').is_not_null()).group_by('host').agg(pl.col('domain').first().alias('crawl_domain'))
    open_rows = (j.filter(pl.col('status').is_in(OPEN)).group_by('host').agg(pl.col('url_norm').n_unique().alias('open_rows')))
    held = pl.DataFrame({'host': list(HELD), 'note': list(HELD.values())})
    if indexed is not None:
        cover = zh.join(indexed.select('id').unique(), on='id', how='inner').group_by('host').len().rename({'len': 'in_index'})
        t = t.join(cover, on='host', how='left').with_columns(pl.col('in_index').fill_null(0))
    t = (t.join(domain, on='host', how='left').join(open_rows, on='host', how='left').join(held, on='host', how='left')
         .with_columns(pl.col('open_rows').fill_null(0), pl.col('note').fill_null('')))
    cols = ['host', 'crawl_domain', 'total', 'ok', 'thin', 'pending', 'deferred', 'absent', 'other', 'open_rows']
    return t.select(*cols, *(['in_index'] if indexed is not None else []), 'note').sort('total', descending=True)


def defer_plan(rows: pl.DataFrame, zh: pl.DataFrame) -> pl.DataFrame:
    """Open rows (pending, or network_error awaiting retry) with no v2 id: the rows to defer."""
    flagged = v2_flags(rows, zh)
    return (flagged.filter(pl.col('status').is_in(OPEN) & (pl.col('n_v2') == 0))
            .select('url_norm', 'domain', 'status').sort('url_norm'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=('table', 'plan'))
    ap.add_argument('--out', default=str(RUN))
    ap.add_argument('--db', default=str(CRAWL_DB))
    a = ap.parse_args(argv)
    out = Path(a.out)
    zh = v2_zh()
    conn = ro(a.db)
    try:
        rows = crawl_rows(conn)
    finally:
        conn.close()
    out.mkdir(parents=True, exist_ok=True)
    if a.step == 'table':
        t = host_table(zh, rows, indexed_docs())
        t.write_csv(assert_writable(out / 'host_table.csv'))
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_cols(20)
        pl.Config.set_tbl_width_chars(220)
        print(t)
        print(json.dumps({'v2_zh_ids': zh.height, **{c: int(t[c].sum()) for c in
                          ('ok', 'thin', 'pending', 'deferred', 'absent', 'other', 'open_rows', 'in_index')}}))
    else:
        p = defer_plan(rows, zh)
        p.write_parquet(assert_writable(out / 'defer_plan.parquet'), compression='zstd')
        print(p.group_by('domain', 'status').len().sort('domain', 'status'))
        print(json.dumps({'defer_rows': p.height}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
