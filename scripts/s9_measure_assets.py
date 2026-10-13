"""S9 step 4: how much of each existing asset is still valid under corpus v2 (read-only; writes only out/runs/s9).

    python scripts/s9_measure_assets.py [--out out/runs/s9/assets.json]

Docs carry `doc_ids` (all corpus ids of one URL group); a doc is 'any' valid when at least one id is in v2 and 'all'
valid when every id is. Chunks are valid on the primary `doc_id` (what the submission validator checks).
"""
from __future__ import annotations

from r2ai import paths
from r2ai import corpus_v2 as C
from r2ai.paths import OUT_DIR, STATE_DIR, WORK_DATA_DIR, assert_writable

import argparse
import glob
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RUNS = OUT_DIR / 'runs'


def list_stats(col) -> dict:
    """col: arrow list<int64> column. Doc-level and id-level validity."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(arr)
    lens = np.diff(arr.offsets.to_numpy())
    flat = arr.flatten().to_numpy()
    ok = C.member_mask(flat)
    row = np.repeat(np.arange(n), lens)
    hit = np.bincount(row, weights=ok, minlength=n)
    return {'docs': n, 'docs_any_in_v2': int((hit > 0).sum()), 'docs_all_in_v2': int((hit == lens).sum()),
            'ids': int(flat.size), 'ids_in_v2': int(ok.sum())}


def add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in {*a, *b}}


def pct(d: dict) -> dict:
    out = dict(d)
    for num, den in (('docs_any_in_v2', 'docs'), ('docs_all_in_v2', 'docs'), ('ids_in_v2', 'ids'), ('chunks_in_v2', 'chunks'),
                     ('chunks_group_any_in_v2', 'chunks'), ('pairs_in_v2', 'pairs'), ('distinct_docs_in_v2', 'distinct_docs'),
                     ('in_v2', 'total')):
        if num in d and d.get(den):
            out[num + '_pct'] = round(100 * d[num] / d[den], 3)
    return out


def id_stats(ids) -> dict:
    ids = np.asarray(ids, dtype=np.int64)
    uniq = np.unique(ids)
    return {'pairs': int(ids.size), 'pairs_in_v2': int(C.member_mask(ids).sum()),
            'distinct_docs': int(uniq.size), 'distinct_docs_in_v2': int(C.member_mask(uniq).sum())}


def chunk_file_stats(path) -> dict:
    t = pq.read_table(path, columns=['doc_id', 'doc_ids_group'])
    prim = C.member_mask(t['doc_id'].to_numpy())
    g = t['doc_ids_group'].combine_chunks()
    lens = np.diff(g.offsets.to_numpy())
    gok = C.member_mask(g.flatten().to_numpy())
    hit = np.bincount(np.repeat(np.arange(len(g)), lens), weights=gok, minlength=len(g))
    return {'chunks': t.num_rows, 'chunks_in_v2': int(prim.sum()), 'chunks_group_any_in_v2': int(((hit > 0) | prim).sum())}


def doc_dir_stats(files) -> dict:
    tot: dict = {}
    for f in files:
        tot = add(tot, list_stats(pq.read_table(f, columns=['doc_ids'])['doc_ids']))
    return tot


def ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)


def measure() -> dict:
    res: dict = {'measured_at': time.strftime('%Y-%m-%d %H:%M:%S%z'), 'v2_ids': int(C.load_ids().size)}

    # --- vi docs / chunks / index
    vi_docs = doc_dir_stats(sorted(glob.glob(str(WORK_DATA_DIR / 'docs_vi' / '*.parquet'))))
    res['vi_docs_extracted'] = pct(vi_docs)
    res['vi_index_docs'] = pct(list_stats(pq.read_table(WORK_DATA_DIR / 'chunks' / 'docs.parquet', columns=['doc_ids_group'])['doc_ids_group']))
    res['vi_index_chunks_t256'] = pct(chunk_file_stats(WORK_DATA_DIR / 'chunks' / 'chunks_t256.parquet'))

    # --- vi caches: rerank pair caches (query, doc, chunk, score)
    res['vi_rerank_cache'] = {}
    for name, rel in (('deep-rerank-100', 'deep-rerank-2026-10-05/full'), ('rerank200', 'rerank200/full'), ('pool300', 'pool300-2026-10-06/full')):
        t = pq.read_table(RUNS / rel / 'vi_k100_pairs.parquet', columns=['doc_id'])
        res['vi_rerank_cache'][name] = pct(id_stats(t['doc_id'].to_numpy()))

    # --- zh: docs by batch and domain, chunk segments
    zh = {}
    root = WORK_DATA_DIR / 'docs_zh_full'
    for batch in sorted(p.name for p in root.iterdir() if p.is_dir()):
        per_dom = {}
        for dom in sorted(p.name for p in (root / batch).iterdir() if p.is_dir()):
            per_dom[dom] = pct(doc_dir_stats(sorted(glob.glob(str(root / batch / dom / '*.parquet')))))
        total = {}
        for d in per_dom.values():
            total = add(total, {k: v for k, v in d.items() if not k.endswith('_pct')})
        zh[batch] = {'total': pct(total), 'by_domain': per_dom}
    res['zh_docs'] = zh
    seg = {}
    for d in sorted((WORK_DATA_DIR / 'chunks_zh_full').iterdir()):
        s: dict = {}
        for f in sorted(d.glob('*.parquet')):
            if 'doc_id' in pq.ParquetFile(f).schema_arrow.names:
                s = add(s, chunk_file_stats(f))
        seg[d.name] = pct(s)
    res['zh_chunk_segments'] = seg
    tot = {}
    for d in seg.values():
        tot = add(tot, {k: v for k, v in d.items() if not k.endswith('_pct')})
    res['zh_chunks_total'] = pct(tot)

    # --- zh rerank cache (retrieval-b001) and the ZG output
    zroot = RUNS / 'zh-full-ingest'
    cache = {}
    for name in ('dense_merged', 'hybrid_merged'):
        d = json.loads((zroot / 'retrieval-b001' / 'rerank' / f'{name}.json').read_text('utf-8'))
        ids = [int(p[0]) for v in d.values() for p in v]
        cache[name] = pct(id_stats(ids))
    res['zh_rerank_cache'] = cache
    qa = json.loads((zroot / 'ZF' / 'ZG' / 'query_append.json').read_text('utf-8'))
    app = [int(i) for v in qa.values() for i in v['appended']]
    prim = [int(i) for v in qa.values() for i in v['docs']]
    keep = {q: [i for i in v['appended'] if C.is_member(int(i))] for q, v in qa.items()}
    res['zg_output'] = {
        'queries': len(qa), 'appended_ids': pct({'ids': len(app), 'ids_in_v2': int(C.member_mask(app).sum())}),
        'primary_docs': pct({'ids': len(prim), 'ids_in_v2': int(C.member_mask(prim).sum())}),
        'queries_with_append_before': sum(1 for v in qa.values() if v['appended']),
        'queries_with_append_after_v2_filter': sum(1 for v in keep.values() if v),
        'appended_per_query_after_v2_filter_mean': round(sum(len(v) for v in keep.values()) / len(qa), 4),
    }

    # --- CC (S3, longchau)
    cov = pq.read_table(RUNS / 'vi-longchau' / 'cc-coverage' / 'coverage.parquet', columns=['url_id', 'crawl']).to_pandas()
    targets = cov[cov.crawl.notna()].url_id.to_numpy()
    db = ro(STATE_DIR / 'cc_longchau' / 'fetch.db')
    final = ('ok', 'bad_http_status', 'url_mismatch', 'parse_error', 'range_mismatch', 'not_html')
    done = np.array([r[0] for r in db.execute(f"SELECT url_id FROM fetch WHERE status IN ({','.join('?' * len(final))})", final)], dtype=np.int64)
    okids = np.array([r[0] for r in db.execute("SELECT url_id FROM fetch WHERE status='ok'")], dtype=np.int64)
    queue = np.setdiff1d(targets, done)
    f = lambda a: {'total': int(a.size), 'in_v2': int(C.member_mask(a).sum())}
    res['cc_s3'] = {'targets_with_capture': pct(f(targets)), 'fetched_ok': pct(f(okids)), 'queue_remaining': pct(f(queue)),
                    'fetch_db_status': dict(db.execute('SELECT status, count(*) FROM fetch GROUP BY 1').fetchall())}
    res['cc_s3']['docs'] = pct(doc_dir_stats(sorted(glob.glob(str(WORK_DATA_DIR / 'docs_cc_longchau' / '*.parquet')))))
    ccseg = {}
    for d in sorted((WORK_DATA_DIR / 'chunks_cc_longchau').iterdir()):
        s = {}
        for fp in sorted(d.glob('*.parquet')):
            if 'doc_id' in pq.ParquetFile(fp).schema_arrow.names:
                s = add(s, chunk_file_stats(fp))
        ccseg[d.name] = pct(s)
    res['cc_s3']['chunk_segments'] = ccseg

    # --- b002 (120ask + cnkang): the crawl queue behind it (state/zh_full/crawl.db, read-only snapshot)
    cdb = ro(STATE_DIR / 'zh_full' / 'crawl.db')
    q: dict = {}
    for dom, status, ids in cdb.execute("SELECT domain, status, doc_ids FROM urls WHERE domain IN ('120ask.com','cnkang.com')"):
        ids = json.loads(ids)
        s = q.setdefault(dom, {}).setdefault(status, {'urls': 0, 'urls_any_in_v2': 0, 'urls_all_in_v2': 0})
        m = C.member_mask(ids)
        s['urls'] += 1
        s['urls_any_in_v2'] += int(m.any())
        s['urls_all_in_v2'] += int(m.all())
    res['b002_crawl_queue'] = q
    res['b002_crawl_snapshot'] = time.strftime('%Y-%m-%d %H:%M:%S%z')
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(RUNS / 's9' / 'assets.json'))
    args = ap.parse_args(argv)
    out = assert_writable(args.out)
    res = measure()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k not in ('zh_docs', 'zh_chunk_segments', 'b002_crawl_queue')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
