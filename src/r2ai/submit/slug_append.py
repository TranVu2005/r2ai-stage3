"""slug_append: append the top-k vi doc ids ranked by BM25 over URL slugs to relevant_docs (VS-append), chunks untouched.

The candidates are vi URLs that have no text: crawl.db rows with a failed / pending status, plus the nhathuoclongchau.com.vn
corpus URLs that are not in crawl.db. Score = BM25 of the query tokens against the de-accented slug tokens (IDF and average
length over every vi URL). The first doc id of a candidate URL is appended when it is not already in relevant_docs.

    python -m r2ai.submit.slug_append pin --crawl-db state/crawl.db --corpus data/raw/links_corpus.parquet --out PIN.parquet
    python -m r2ai.submit.slug_append apply --base RD150.json --pin PIN --pin-sha256 SHA --queries query.parquet --k 40 --out X.zip

The pin file freezes the URL universe (all URLs, in crawl.db row order, then the corpus longchau URLs), their status and the
first doc id of each candidate; best.py checks its SHA256 (postprocess.slug_append.pin_sha256) and never reads crawl.db, so a
replay does not change when the crawl state does. k, k1, b and scan_depth come from the config.
Row order matters: it fixes the vocabulary indices and so the float summation order of tied scores.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, resolve_path

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from r2ai.selective.bm25 import slug_toks, toks
from r2ai.zh_sample.append import _addition, _rows

BAD_STATUS = {'blocked_4xx', 'bot_challenge', 'soft404_or_home', 'dead_origin', 'network_error', 'pending'}
EXTRA_HOST = 'nhathuoclongchau.com.vn'          # regex, as r2ai.selective.step3_vs (corpus urls absent from crawl.db)
NOT_IN_CRAWL = 'not_in_crawl'
QUERY_BATCH = 100


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_pin(crawl_db, corpus_file, out) -> dict:
    """Freeze the slug universe from a crawl.db (read only) and the corpus into `out`; returns rows / pool / sha256."""
    db = sqlite3.connect(f'file:{Path(crawl_db).resolve().as_posix()}?mode=ro', uri=True)
    try:
        rows = db.execute('select url,doc_ids,domain,status from urls').fetchall()
    finally:
        db.close()
    corp = pq.read_table(corpus_file, columns=['id', 'url']).to_pylist()
    extra = [r for r in corp if re.search(EXTRA_HOST, r['url'])]
    url = [r[0] for r in rows] + [r['url'] for r in extra]
    domain = [r[2] for r in rows] + [EXTRA_HOST] * len(extra)
    status = [r[3] for r in rows] + [NOT_IN_CRAWL] * len(extra)
    in_pool = [s in BAD_STATUS or s == NOT_IN_CRAWL for s in status]
    first = [(json.loads(r[1]) or [-1])[0] for r in rows] + [int(r['id']) for r in extra]
    doc_id = [d if p else -1 for d, p in zip(first, in_pool)]
    out = assert_writable(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'url': url, 'domain': domain, 'status': status, 'in_pool': in_pool,
                             'doc_id': pa.array(doc_id, pa.int64())}), out)
    return {'rows': len(url), 'pool': int(sum(in_pool)), 'crawl_rows': len(rows), 'corpus_extra_rows': len(extra),
            'sha256': sha256(out), 'path': str(out)}


def bm25_matrix(urls, k1: float, b: float):
    """Term x URL BM25 weights (CSR, shape [vocab, n]) and the vocabulary; same arithmetic as r2ai.selective.bm25.Slug."""
    docs = [slug_toks(u) for u in urls]
    vocab, indptr, indices, tf = {}, [0], [], []
    for d in docs:
        for t, f in Counter(d).items():
            indices.append(vocab.setdefault(t, len(vocab)))
            tf.append(f)
        indptr.append(len(indices))
    n = len(docs)
    dl = np.array([len(d) for d in docs], float)
    X = sp.csr_matrix((np.array(tf, float), indices, indptr), shape=(n, len(vocab)))
    df = np.bincount(X.indices, minlength=len(vocab))
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * dl / dl.mean())
    X.data = X.data * (k1 + 1) / (X.data + np.repeat(norm, np.diff(X.indptr))) * idf[X.indices]
    return X.T.tocsr(), vocab


def query_matrix(texts, vocab):
    Q = sp.lil_matrix((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        for t in set(toks(text)):
            j = vocab.get(t)
            if j is not None:
                Q[i, j] = 1.0
    return Q.tocsr()


def append_slug_ids(base_json, out_json, *, pin, pin_sha256: str, queries, k: int, k1: float, b: float, scan_depth: int,
                    exclude_ids=None) -> dict:
    """Write out_json = base_json with up to k slug-ranked doc ids appended to every relevant_docs; returns stats.

    exclude_ids (default None = off, VS-append unchanged): doc ids never appended, e.g. longchau docs that now have CC text."""
    pin = resolve_path(pin)
    if sha256(pin) != pin_sha256:
        raise ValueError(f'slug pin SHA256 mismatch: {pin} is {sha256(pin)}, config says {pin_sha256}')
    t = pq.read_table(pin).to_pydict()
    urls, in_pool, doc_id = t['url'], np.array(t['in_pool'], bool), t['doc_id']
    pool = np.flatnonzero(in_pool)
    WT, vocab = bm25_matrix(urls, k1, b)
    WTp = WT[:, pool]
    q = pq.read_table(queries).to_pylist()
    Q = query_matrix([r['query'] for r in q], vocab)
    base = Path(base_json).read_text(encoding='utf-8')
    ref = list(_rows(base))
    if [r['id'] for r, _ in ref] != [r['id'] for r in q]:
        raise ValueError('base JSON query ids differ from the query file')
    picks = []
    for a in range(0, len(q), QUERY_BATCH):
        S = (Q[a:a + QUERY_BATCH] @ WTp).toarray()
        for i in range(S.shape[0]):
            have = set(ref[a + i][0]['relevant_docs'])
            order = np.lexsort((np.arange(len(pool)), -S[i]))
            pick = []
            for j in order[:scan_depth]:
                if S[i][j] <= 0 or len(pick) >= k:
                    break
                d = doc_id[pool[j]]
                if d >= 0 and d not in have and d not in pick and (exclude_ids is None or d not in exclude_ids):
                    pick.append(d)
            picks.append(pick)
    pieces, cursor = [], 0
    for (row, spans), pick in zip(ref, picks):
        end = spans['relevant_docs'][1]
        pieces.append(base[cursor:end - 1])
        pieces.append(_addition(row['relevant_docs'], pick))
        cursor = end - 1
    pieces.append(base[cursor:])
    result = ''.join(pieces)
    chunk_diff = docs_diff = 0
    for (old, _), (new, _), pick in zip(ref, _rows(result), picks):
        chunk_diff += old['relevant_chunks'] != new['relevant_chunks']
        docs_diff += new['relevant_docs'] != old['relevant_docs'] + pick
    if chunk_diff or docs_diff:
        raise ValueError(f'slug_append altered more than relevant_docs: chunks {chunk_diff}, docs {docs_diff}')
    out_json = assert_writable(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(result, encoding='utf-8', newline='')
    n = np.array([len(p) for p in picks])
    return {'queries': len(q), 'pool_urls': int(len(pool)), 'universe_urls': len(urls), 'appended_total': int(n.sum()),
            'appended_mean': float(n.mean()) if len(n) else 0.0, 'queries_below_k': int((n < k).sum()), 'k': k,
            'pin_sha256': pin_sha256, 'query_picks': picks}


def slug_append_submission(base_json, out_zip, **kw) -> dict:
    """append_slug_ids into <out_zip stem>.json and zip it (deflate 9, one member) as the other post-processing steps do."""
    out_zip = Path(out_zip)
    out_json = out_zip.with_suffix('.json')
    st = append_slug_ids(base_json, out_json, **kw)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out_json, out_json.name)
    return st | {'json_sha256': sha256(out_json), 'json_bytes': out_json.stat().st_size,
                 'zip_sha256': sha256(out_zip), 'zip_bytes': out_zip.stat().st_size}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pin', help='freeze the slug universe (reads crawl.db with mode=ro)')
    p.add_argument('--crawl-db', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('apply', help='append k slug ids to a submission JSON (best.py does this for the configured k)')
    p.add_argument('--base', required=True, help='submission JSON (e.g. RD150)')
    p.add_argument('--pin', required=True)
    p.add_argument('--pin-sha256', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--k1', type=float, default=1.2)
    p.add_argument('--b', type=float, default=0.75)
    p.add_argument('--scan-depth', type=int, default=400)
    p.add_argument('--out', required=True, help='output zip; the JSON is written next to it')
    a = ap.parse_args(argv)
    if a.cmd == 'pin':
        print(json.dumps(build_pin(a.crawl_db, a.corpus, a.out), indent=1))
        return 0
    st = slug_append_submission(a.base, a.out, pin=a.pin, pin_sha256=a.pin_sha256, queries=a.queries, k=a.k, k1=a.k1, b=a.b,
                                scan_depth=a.scan_depth)
    st.pop('query_picks')
    print(json.dumps(st, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
