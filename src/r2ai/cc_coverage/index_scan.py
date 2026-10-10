"""Locate and read the rows of one SURT prefix in the CC columnar index (parquet) with as few bytes as possible.

If the files of a crawl are sorted by url_surtkey (checked from row-group min/max statistics, not assumed), a binary search over
file footers finds the files holding the prefix, and only the row groups whose statistics overlap the prefix are read, with
only the needed columns. Every byte goes through the metered Client (ledger caps, backoff).
"""
from __future__ import annotations

import gzip

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .net import Client, InvalidResponse
from .rangefile import RangeFile

BASE = 'https://data.commoncrawl.org/'
COLUMNS = ('url_surtkey', 'fetch_time', 'fetch_status', 'content_mime_type', 'warc_filename', 'warc_record_offset',
           'warc_record_length')


class TooBig(Exception):
    """The row groups to read exceed the per-read byte cap: stop and ask."""


def list_parquet_paths(client: Client, crawl: str) -> list[str]:
    r = client.get(f'{BASE}crawl-data/{crawl}/cc-index-table.paths.gz')
    if r.status_code == 404:
        raise InvalidResponse(f'{crawl}: cc-index-table.paths.gz not found')
    lines = gzip.decompress(r.content).decode('utf-8').split()
    return sorted(p for p in lines if 'subset=warc/' in p and p.endswith('.parquet'))


def open_parquet(client: Client, path: str, **kw):
    rf = RangeFile(client, BASE + path, **kw)
    return pq.ParquetFile(pa.PythonFile(rf, mode='r')), rf


def key_ranges(pf: pq.ParquetFile, column: str = 'url_surtkey'):
    """Per row group (min, max) of `column`, None where statistics are missing."""
    idx = pf.schema_arrow.get_field_index(column)
    out = []
    for g in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(g).column(idx).statistics
        out.append((st.min, st.max) if st is not None and st.has_min_max else None)
    return out


def file_range(ranges):
    """(min, max) over the row groups, None if any group lacks statistics."""
    if not ranges or any(r is None for r in ranges):
        return None
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def upper_bound(prefix: str) -> str:
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


def footer_info(pf: pq.ParquetFile, columns=COLUMNS) -> dict:
    md = pf.metadata
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)] if md.num_row_groups else []
    return {'footer_bytes': md.serialized_size, 'row_groups': md.num_row_groups, 'rows': md.num_rows,
            'missing_columns': [c for c in columns if c not in names],
            'row_group_rows': [md.row_group(g).num_rows for g in range(md.num_row_groups)]}


def row_group_cost(pf: pq.ParquetFile, g: int, columns=COLUMNS) -> int:
    rg = pf.metadata.row_group(g)
    return sum(rg.column(i).total_compressed_size for i in range(rg.num_columns) if rg.column(i).path_in_schema in columns)


def bisect_files(n_files: int, probe, lo_key: str, hi_key: str, verify=None):
    """probe(i) -> (min, max) of the file's url_surtkey, or None when statistics are missing. Returns (candidate file indices,
    probed {i: range}, sorted_ok). Binary search for the first file whose max >= lo_key, then walk forward while min <= hi_key.
    sorted_ok = the probed files (the search path plus `verify`, default first / quartiles / last) are ordered by key."""
    seen: dict[int, tuple | None] = {}

    def rng(i):
        if i not in seen:
            seen[i] = probe(i)
        return seen[i]

    lo, hi = 0, n_files - 1
    while lo < hi:
        mid = (lo + hi) // 2
        r = rng(mid)
        if r is None:
            return [], seen, False
        if r[1] >= lo_key:
            hi = mid
        else:
            lo = mid + 1
    cands = []
    i = lo
    while i < n_files:
        r = rng(i)
        if r is None:
            return [], seen, False
        if r[0] > hi_key:
            break
        if r[1] >= lo_key:
            cands.append(i)
        i += 1
    for i in (verify if verify is not None else sorted({0, n_files // 4, n_files // 2, (3 * n_files) // 4, n_files - 1})):
        if rng(i) is None:
            return [], seen, False
    order = sorted(k for k in seen)
    ok = all(seen[a][1] <= seen[b][0] for a, b in zip(order, order[1:]))
    return cands, seen, ok


def read_prefix(pf: pq.ParquetFile, prefix: str, *, columns=COLUMNS, max_bytes: int) -> tuple[pa.Table, dict]:
    """Rows whose url_surtkey starts with prefix, from the row groups whose statistics overlap [prefix, upper_bound)."""
    ranges = key_ranges(pf)
    hi = upper_bound(prefix)
    groups = [g for g, r in enumerate(ranges) if r is None or (r[0] < hi and r[1] >= prefix)]
    cost = sum(row_group_cost(pf, g, columns) for g in groups)
    if cost > max_bytes:
        raise TooBig(f'row groups {groups} need {cost} bytes > cap {max_bytes}')
    tables = []
    for g in groups:
        t = pf.read_row_group(g, columns=list(columns), use_threads=False)
        tables.append(t.filter(pc.starts_with(t['url_surtkey'], prefix)))
    table = pa.concat_tables(tables) if tables else pa.table({c: [] for c in columns})
    return table, {'row_groups_read': groups, 'compressed_bytes_planned': cost, 'rows_matched': table.num_rows}


def footer_map(client: Client, paths: list[str], prefix: str, known: dict | None = None, columns=COLUMNS) -> dict:
    """Read the footer of every file (one 64 KiB tail request each; `known` = {index: (ParquetFile, file_bytes)} already open) and
    return, per file, its key range, row-group ranges and the exact compressed byte cost of `columns` per row group; plus which row
    groups may hold `prefix` and what reading them would cost. No row group is read."""
    known = known or {}
    hi = upper_bound(prefix)
    files, hits = [], []
    for i, path in enumerate(paths):
        if i in known:
            pf, size = known[i]
        else:
            pf, rf = open_parquet(client, path)
            size = rf.size
        ranges = key_ranges(pf)
        rec = {'i': i, 'file_bytes': size, 'footer_bytes': pf.metadata.serialized_size, 'rows': pf.metadata.num_rows,
               'row_groups': pf.metadata.num_row_groups, 'range': file_range(ranges)}
        files.append(rec)
        for g, r in enumerate(ranges):
            if r is None or (r[0] < hi and r[1] >= prefix):
                hits.append({'file': i, 'row_group': g, 'rows': pf.metadata.row_group(g).num_rows,
                             'cost_columns_bytes': row_group_cost(pf, g, columns), 'cost_surtkey_only_bytes': row_group_cost(pf, g, ('url_surtkey',)),
                             'rg_total_bytes': pf.metadata.row_group(g).total_byte_size, 'range': r})
    ranged = sorted((f['range'][0], f['range'][1], f['i']) for f in files if f['range'])
    overlaps = sum(1 for a, b in zip(ranged, ranged[1:]) if a[1] > b[0])
    fb = [f['footer_bytes'] for f in files]
    return {'files': len(files), 'file_bytes_total': sum(f['file_bytes'] for f in files), 'rows_total': sum(f['rows'] for f in files),
            'footer_bytes': {'min': min(fb), 'max': max(fb), 'mean': round(sum(fb) / len(fb))},
            'file_ranges_missing_stats': sum(1 for f in files if f['range'] is None), 'adjacent_range_overlaps_when_sorted_by_min': overlaps,
            'row_group_hits': hits, 'hits_cost_columns_bytes': sum(h['cost_columns_bytes'] for h in hits),
            'hits_cost_surtkey_only_bytes': sum(h['cost_surtkey_only_bytes'] for h in hits), 'file_list': files}
