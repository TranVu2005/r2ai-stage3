"""Batch ingest of CLOSED zh-full shards: cut -> extract -> chunk -> embed, one segment per (batch, domain).

Closed shard: the crawler writes `<name>.jsonl.zst.tmp`, fsyncs, renames it to its final name and only
then commits `urls.shard` (r2ai.zh_full.crawl.write_raw/flush). A cut therefore snapshots crawl.db
(mode=ro) FIRST and takes a shard only if its final-name file exists, is referenced by the snapshot and
is older than the snapshot. Only the record each snapshot URL points to is used (a shard also holds
superseded attempts). Every step is idempotent and resumable; the manifest (state/zh_full_ingest/
manifest.db) records shards (size, mtime, sha256, docs, extractor version) and segments (stable id).

Chunk ids are global int64: 2**48 + segment_id * 2**32 + local row. Segment 1 is the old sample index
(retired); ids are allocated by AUTOINCREMENT and never reused; a replaced segment is retired, not renumbered.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import (Roots, MemoryGuard, atomic_json, exclusive, global_start, log_event, require_gpu_idle,
                     sha256_file)
from . import extract as zf

TARGET = 256
MIN_BODY_TOKENS = 50
MODEL = {'model': 'BAAI/bge-m3', 'max_len': 512, 'fp16': True, 'batch_size': 16, 'block': 1000,
         'dense': 'CLS, L2-normalised, fp16', 'sparse': 'relu(sparse_linear) max per token id'}
OLD_SAMPLE_ID = 1
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches(batch TEXT PRIMARY KEY, kind TEXT NOT NULL, snapshot_at REAL NOT NULL,
    created_at REAL NOT NULL, domains TEXT NOT NULL, source_counts TEXT NOT NULL, skipped TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, n_urls INTEGER NOT NULL,
    status TEXT NOT NULL, n_records INTEGER, n_docs INTEGER, docs_file TEXT, extractor_version TEXT, done_at REAL);
CREATE TABLE IF NOT EXISTS batch_urls(url_norm TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    shard TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs(url_norm TEXT PRIMARY KEY, batch TEXT NOT NULL, domain TEXT NOT NULL,
    doc_id INTEGER NOT NULL, status TEXT NOT NULL, docs_file TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS segments(segment_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
    batch TEXT NOT NULL, domain TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, rows INTEGER,
    docs_count INTEGER, global_start INTEGER, chunks_sha256 TEXT, extractor_version TEXT, chunker TEXT,
    model TEXT, vectors TEXT, note TEXT, stats TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS embed_blocks(segment TEXT NOT NULL, block INTEGER NOT NULL, rows INTEGER NOT NULL,
    dense_sha256 TEXT NOT NULL, sparse_sha256 TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(segment, block));
"""


def connect(roots: Roots) -> sqlite3.Connection:
    path = roots.guard(roots.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    c = sqlite3.connect(path)
    c.execute('PRAGMA journal_mode=WAL')
    c.executescript(SCHEMA_SQL)
    if fresh or not c.execute("SELECT 1 FROM sqlite_sequence WHERE name='segments'").fetchone():
        # Reserve segment_id 1 for the old sample index, whether or not it is converted yet.
        if not c.execute('SELECT 1 FROM segments').fetchone():
            c.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('segments', ?)", (OLD_SAMPLE_ID,))
    c.commit()
    return c


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect('file:' + Path(path).resolve().as_posix() + '?mode=ro', uri=True)


def _insert_batch(c, batch, kind, snapshot, domains, counts, skipped):
    if c.execute('SELECT 1 FROM batches WHERE batch=?', (batch,)).fetchone():
        raise ValueError(f'batch {batch} exists')
    c.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)', (batch, kind, snapshot, time.time(), json.dumps(domains),
              json.dumps(counts), json.dumps(skipped)))


def _record_shards(c, batch, by_shard):
    """by_shard: {Path: (domain, [url_norm])}. Stat + hash now; extraction re-verifies both."""
    for path, (domain, urls) in sorted(by_shard.items()):
        st = path.stat()
        c.execute('INSERT INTO shards(path,batch,domain,size,mtime_ns,sha256,n_urls,status) VALUES (?,?,?,?,?,?,?,?)',
                  (str(path), batch, domain, st.st_size, st.st_mtime_ns, sha256_file(path), len(urls), 'pending'))
        c.executemany('INSERT INTO batch_urls VALUES (?,?,?,?)', [(u, batch, domain, str(path)) for u in urls])


def cut_batch(roots: Roots, batch: str, domains: list[str], *, clock=time.time) -> dict:
    """Snapshot crawl.db, then select closed committed shards of `domains` not taken by an earlier batch."""
    snapshot = clock()
    ro = _ro(roots.crawl_db)
    try:
        q = ','.join('?' * len(domains))
        rows = ro.execute(f"SELECT url_norm, domain, status, shard FROM urls WHERE domain IN ({q}) "
                          "AND status IN ('ok','thin') AND shard IS NOT NULL", domains).fetchall()
        counts = defaultdict(dict)
        for d, s, n in ro.execute(f'SELECT domain, status, count(*) FROM urls WHERE domain IN ({q}) GROUP BY 1,2', domains):
            counts[d][s] = n
    finally:
        ro.close()
    c = connect(roots)
    try:
        if c.execute('SELECT 1 FROM batches WHERE batch=?', (batch,)).fetchone():
            raise ValueError(f'batch {batch} exists')
        taken = {r[0] for r in c.execute('SELECT url_norm FROM batch_urls')}
        files = {}
        for d in domains:
            directory = roots.raw / d
            files[d] = {f.name: f for f in directory.iterdir()} if directory.is_dir() else {}
        skipped, by_shard = Counter(), {}
        for u, d, status, shard in rows:
            if u in taken:
                skipped['previous_batch'] += 1
                continue
            f = files[d].get(shard)
            if f is None or not shard.endswith('.jsonl.zst'):
                skipped['not_final_name_or_open' if (shard + '.tmp') in files[d] else 'missing_file'] += 1
            elif f.stat().st_mtime >= snapshot:
                skipped['newer_than_snapshot'] += 1
            else:
                by_shard.setdefault(f.resolve(), (d, []))[1].append(u)
        skipped = {k: v for k, v in sorted(skipped.items()) if v}
        _insert_batch(c, batch, 'full', snapshot, domains, counts, skipped)
        _record_shards(c, batch, by_shard)
        c.commit()
    finally:
        c.close()
    out = {'batch': batch, 'snapshot_at': snapshot, 'shards': len(by_shard),
           'urls': sum(len(v[1]) for v in by_shard.values()), 'skipped': skipped, 'source_counts': counts}
    log_event(roots, 'cut', out)
    return out


def cut_sample(roots: Roots, batch: str, domains: list[str] | None = None) -> dict:
    """The finished 2% sample: same record rule as r2ai.zh_sample.extract (latest fetched ok/thin copy per URL,
    shards in that module's parquet name order, strict > keeps the earlier copy on ties)."""
    from vicrawl.shards import list_shards, read_shard
    snapshot = time.time()
    shards = sorted(list_shards(roots.sample_raw), key=lambda p: p.parent.name + '__' + p.stem)
    if any(p.name.endswith('.tmp') for p in roots.sample_raw.rglob('*.tmp')):
        raise ValueError('sample raw has open .tmp shards')
    best = {}
    for p in shards:
        if domains is not None and p.parent.name not in domains:
            continue
        for r in read_shard(p):
            if r['status'] not in ('ok', 'thin') or not r.get('html'):
                continue
            u = r['url_norm']
            if u not in best or r['fetched_at'] > best[u][0]:
                best[u] = (r['fetched_at'], p.resolve(), r['domain'])
    by_shard = {}
    for u, (_, p, d) in best.items():
        by_shard.setdefault(p, (d, []))[1].append(u)
    c = connect(roots)
    try:
        counts = dict(Counter(d for _, _, d in best.values()))
        _insert_batch(c, batch, 'sample', snapshot, sorted(set(counts)), counts, {})
        _record_shards(c, batch, by_shard)
        c.commit()
    finally:
        c.close()
    out = {'batch': batch, 'shards': len(by_shard), 'urls': len(best), 'by_domain': counts}
    log_event(roots, 'cut', out)
    return out


def _doc_row(r, doc, domain, tok):
    body = doc.body
    return {'doc_ids': r['doc_ids'], 'url': r['url'], 'final_url': r.get('final_url'), 'domain': domain,
            'title': doc.title, 'description': doc.description, 'question': doc.question, 'answer': doc.answer,
            'body': body, 'paragraphs': doc.paragraphs, 'headings': doc.headings,
            'lang': 'zh' if sum('一' <= ch <= '鿿' for ch in body) > .1 * len(body) else 'unknown',
            'n_tokens_bge_m3': tok(body) if body else 0, 'text_sha1': hashlib.sha1(body.encode()).hexdigest(),
            'fetched_at': r['fetched_at'], 'url_norm': r['url_norm'], 'status': zf.doc_status(body, domain),
            'extractor': doc.extractor}


def _write_parquet(roots: Roots, rows, out: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vicrawl.extract_pipeline import SCHEMA
    out = roots.guard(out)
    tmp = roots.guard(out.with_name(out.name + '.partial'))
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression='zstd')
    with open(tmp, 'rb+') as f:
        os.fsync(f.fileno())
    os.replace(tmp, out)


def extract_batch(roots: Roots, batch: str, *, tok=None, memory: MemoryGuard | None = None, before_shard=None) -> dict:
    """One worker. Each shard commits its parquet and manifest rows atomically; reruns skip done shards."""
    from vicrawl.shards import read_shard
    if tok is None:
        from vicrawl.extract_pipeline import default_tokenizer
        tok = default_tokenizer()
    version = zf.extractor_version()
    c = connect(roots)
    done = docs = 0
    t0 = time.perf_counter()
    try:
        pending = c.execute("SELECT path, domain, size, sha256 FROM shards WHERE batch=? AND status='pending' ORDER BY path",
                            (batch,)).fetchall()
        for stray in (roots.docs / batch).rglob('*.partial') if (roots.docs / batch).exists() else []:
            roots.guard(stray).unlink()
        for path, domain, size, sha in pending:
            if before_shard is not None:
                before_shard(path)
            if memory is not None:
                memory.check('extract')
            p = Path(path)
            if p.stat().st_size != size or sha256_file(p) != sha:
                raise ValueError(f'shard changed since cut: {path}')
            urls = {r[0] for r in c.execute('SELECT url_norm FROM batch_urls WHERE shard=?', (path,))}
            chosen, n_records = {}, 0
            for r in read_shard(p):
                n_records += 1
                if r['url_norm'] in urls:
                    chosen[r['url_norm']] = r      # last occurrence within the shard
            if set(chosen) != urls:
                raise ValueError(f'pointer URLs missing from shard {path}: {sorted(urls - set(chosen))[:3]}')
            rows = []
            for u in sorted(chosen):
                r = chosen[u]
                if r['status'] not in ('ok', 'thin') or not r.get('html'):
                    raise ValueError(f'pointer record is not ok/thin with html: {u}')
                rows.append(_doc_row(r, zf.extract(r['html'], domain), domain, tok))
            out = roots.docs / batch / domain / (p.name.removesuffix('.jsonl.zst') + '.parquet')
            if rows:
                _write_parquet(roots, rows, out)
            with c:
                c.executemany('INSERT INTO docs VALUES (?,?,?,?,?,?)',
                              [(r['url_norm'], batch, domain, r['doc_ids'][0], r['status'], str(out)) for r in rows])
                c.execute("UPDATE shards SET status='done', n_records=?, n_docs=?, docs_file=?, extractor_version=?, "
                          "done_at=? WHERE path=?", (n_records, len(rows), str(out) if rows else None, version,
                                                     time.time(), path))
            done += 1
            docs += len(rows)
            if done % 500 == 0:
                log_event(roots, 'extract', {'batch': batch, 'shards_done': done, 'docs': docs,
                                             'seconds': time.perf_counter() - t0})
    finally:
        c.close()
    out = {'batch': batch, 'shards_done': done, 'docs': docs, 'seconds': time.perf_counter() - t0,
           'extractor_version': version}
    log_event(roots, 'extract', out)
    return out


def allocate_segment(roots: Roots, batch: str, domain: str, *, kind='full') -> dict:
    c = connect(roots)
    try:
        now = time.time()
        with c:
            cur = c.execute("INSERT INTO segments(name,batch,domain,kind,status,created_at,updated_at) "
                            "VALUES (?,?,?,?,'allocated',?,?)", (f'pending-{batch}-{domain}-{now}', batch, domain, kind, now, now))
            sid = cur.lastrowid
            name = f'seg{sid:05d}-{batch}-{domain}'
            c.execute('UPDATE segments SET name=?, global_start=? WHERE segment_id=?', (name, global_start(sid), sid))
        return {'segment_id': sid, 'name': name}
    finally:
        c.close()


def _segment(c, name):
    row = c.execute('SELECT segment_id, name, batch, domain, kind, status, rows FROM segments WHERE name=?', (name,)).fetchone()
    if row is None:
        raise ValueError(f'unknown segment {name}')
    return dict(zip(('segment_id', 'name', 'batch', 'domain', 'kind', 'status', 'rows'), row))


def _update_segment(roots, name, **fields):
    c = connect(roots)
    try:
        with c:
            sets = ', '.join(f'{k}=?' for k in fields)
            c.execute(f'UPDATE segments SET {sets}, updated_at=? WHERE name=?', (*fields.values(), time.time(), name))
    finally:
        c.close()


def retire_segment(roots: Roots, name: str, reason: str):
    _update_segment(roots, name, status='retired', note=reason)


def _rmtree(roots, path):
    path = roots.guard(path)
    if path.exists():
        shutil.rmtree(path)


def chunk_segment(roots: Roots, batch: str, domain: str, *, batch_docs=5000, kind=None) -> dict:
    """Existing verbatim t256 chunker on one (batch, domain) docs directory, staged then renamed."""
    import pyarrow.parquet as pq
    from r2ai.index.chunk import main as chunk_main
    c = connect(roots)
    try:
        row = c.execute("SELECT name FROM segments WHERE batch=? AND domain=? AND status!='retired'", (batch, domain)).fetchone()
        bkind = c.execute('SELECT kind FROM batches WHERE batch=?', (batch,)).fetchone()
        versions = {r[0] for r in c.execute("SELECT DISTINCT extractor_version FROM shards WHERE batch=? AND domain=?", (batch, domain))}
        pending = c.execute("SELECT count(*) FROM shards WHERE batch=? AND domain=? AND status!='done'", (batch, domain)).fetchone()[0]
    finally:
        c.close()
    if pending or len(versions) != 1:
        raise ValueError(f'{batch}/{domain}: extraction incomplete or mixed extractor versions {versions}')
    name = row[0] if row else allocate_segment(roots, batch, domain, kind=kind or (bkind[0] if bkind else 'full'))['name']
    c = connect(roots)
    try:
        seg = _segment(c, name)
    finally:
        c.close()
    if seg['status'] != 'allocated':
        return seg
    src = roots.docs / batch / domain
    files = sorted(src.glob('*.parquet'))
    ids = []
    for f in files:
        t = pq.read_table(f, columns=['doc_ids', 'status', 'n_tokens_bge_m3', 'body']).to_pylist()
        ids += [r['doc_ids'][0] for r in t if r['status'] == 'ok' and (r['n_tokens_bge_m3'] or 0) >= MIN_BODY_TOKENS and r['body']]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{batch}/{domain}: primary doc ids are not unique')
    out, staging = roots.guard(roots.chunks / name), roots.guard(roots.chunks / (name + '.partial'))
    if out.exists():
        raise ValueError(f'chunks already exist for allocated segment {name}')
    _rmtree(roots, staging)
    staging.mkdir(parents=True)
    t0 = time.perf_counter()
    chunk_main(['--docs-dir', str(src), '--out-dir', str(staging), '--targets', str(TARGET),
                '--min-body-tokens', str(MIN_BODY_TOKENS), '--batch-docs', str(batch_docs)])
    report = json.loads((staging / 'chunk_report.json').read_text(encoding='utf-8'))
    cfg = report['configs'][str(TARGET)]
    if cfg['verbatim_ok'] != f"{cfg['chunks_all']}/{cfg['chunks_all']}":
        raise ValueError(f'verbatim gate failed: {cfg["verbatim_ok"]}')
    os.replace(staging, out)
    cf = out / f'chunks_t{TARGET}.parquet'
    chunker = {'target': TARGET, 'min_body_tokens': MIN_BODY_TOKENS, 'batch_docs': batch_docs,
               'module': 'r2ai.index.chunk'}
    stats = {'chunk_seconds': time.perf_counter() - t0, 'docs': report['docs'], 'chunks': cfg['chunks_all']}
    _update_segment(roots, name, status='chunked', rows=cfg['chunks_all'], docs_count=report['docs'],
                    chunks_sha256=sha256_file(cf), extractor_version=versions.pop(), chunker=json.dumps(chunker),
                    model=json.dumps(MODEL), stats=json.dumps(stats))
    log_event(roots, 'chunk', {'segment': name, **stats})
    return {**seg, 'status': 'chunked', 'rows': cfg['chunks_all'], **stats}


def _exact_blocks(path, block, columns):
    """Consecutive row blocks of exactly `block` rows (last may be shorter), independent of row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pending, lo = [], 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=block, columns=columns):
        pending.append(b)
        while sum(len(x) for x in pending) >= block:
            t = pa.Table.from_batches(pending)
            yield lo, t.slice(0, block)
            lo += block
            rest = t.slice(block)
            pending = rest.to_batches() if len(rest) else []
    if pending:
        t = pa.Table.from_batches(pending)
        if len(t):
            yield lo, t


def _write_segment_from_blocks(roots, name, sid, n, vocabulary, cf, blocks_iter):
    """blocks_iter yields (dense fp16, csr, doc_id, texts); staged as <name>.assembling then renamed."""
    from r2ai.index.segments import SegmentWriter, load_segment
    staging = roots.guard(roots.index / (name + '.assembling'))
    final = roots.guard(roots.index / name)
    if final.exists():
        raise ValueError(f'segment {name} exists')
    _rmtree(roots, staging)
    roots.index.mkdir(parents=True, exist_ok=True)
    w = SegmentWriter(roots.index, staging.name, rows=n, dimension=1024, vocabulary=vocabulary, global_start=global_start(sid))
    try:
        for dense, csr, doc_id, texts in blocks_iter:
            w.append(dense, csr, doc_id, texts)
        w.close()
    except BaseException:
        for f in w.files.values():
            f.close()
        raise
    os.replace(staging, final)
    seg = load_segment(final)
    if len(seg) != n or seg.global_start != global_start(sid):
        raise ValueError('assembled segment shape/id mismatch')
    return seg


NORM_TOLERANCE = 0.02


def check_block(dense_path: Path, sparse_path: Path, rows: int, dimension: int = 1024) -> str | None:
    """None when the embedding block is intact, else the reason it must be recomputed."""
    from scipy.sparse import load_npz
    try:
        dense = np.load(dense_path, allow_pickle=False)
        if dense.shape != (rows, dimension) or dense.dtype.kind != 'f':
            return f'dense shape/dtype {dense.shape} {dense.dtype}, expected ({rows}, {dimension}) float'
        v = dense.astype(np.float32)
        if not np.isfinite(v).all():
            return 'dense has NaN/inf'
        norms = np.linalg.norm(v, axis=1)
        if (np.abs(1.0 - norms) > NORM_TOLERANCE).any():
            return f'dense norm outside 1 +/- {NORM_TOLERANCE}: min {norms.min():.4f} max {norms.max():.4f}'
        sparse = load_npz(sparse_path).tocsr()
        if sparse.shape[0] != rows:
            return f'sparse rows {sparse.shape[0]}, expected {rows}'
        if (np.diff(sparse.indptr) == 0).any():
            return 'sparse has an empty row'
    except Exception as e:  # noqa: BLE001 - any unreadable file means the block is not trusted
        return f'unreadable: {e!r}'
    return None


def _save_durably(final: Path, write):
    """write(file) to <final>.tmp-style sibling, fsync, then rename: a power loss never leaves a short block."""
    tmp = final.with_name(final.name.replace('.npy', '.tmp.npy').replace('.npz', '.tmp.npz'))
    with open(tmp, 'wb') as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)


def _record_block(c, segment, block, rows, dense_path, sparse_path):
    c.execute('INSERT OR REPLACE INTO embed_blocks VALUES (?,?,?,?,?,?)',
              (segment, block, rows, sha256_file(dense_path), sha256_file(sparse_path), time.time()))
    c.commit()


def _block_is_trusted(roots, c, segment, i, rows, dpath, spath) -> bool:
    """A finished block is skipped only if it passes the checks and matches the manifest SHA256 (adopted if none)."""
    reason = check_block(dpath, spath, rows)
    if reason is None:
        recorded = c.execute('SELECT rows, dense_sha256, sparse_sha256 FROM embed_blocks WHERE segment=? AND block=?',
                             (segment, i)).fetchone()
        if recorded is None:
            _record_block(c, segment, i, rows, dpath, spath)       # block from before hashes existed
            return True
        if recorded == (rows, sha256_file(dpath), sha256_file(spath)):
            return True
        reason = 'sha256 differs from the manifest'
    log_event(roots, 'embed_block_rejected', {'segment': segment, 'block': i, 'reason': reason})
    for f in (dpath, spath):
        f.unlink(missing_ok=True)
    c.execute('DELETE FROM embed_blocks WHERE segment=? AND block=?', (segment, i))
    c.commit()
    return False


def embed_segment(roots: Roots, name: str, *, encoder_factory=None, block=MODEL['block'], batch_size=MODEL['batch_size'],
                  gpu_check=None, before_block=None, memory: MemoryGuard | None = None) -> dict:
    """BGE-M3 per 1000-chunk block (batch 16, length-sorted inside the block, as the sample index);
    block checkpoints, then one immutable segment."""
    import pyarrow.parquet as pq
    from scipy.sparse import load_npz, save_npz
    from r2ai.index.build import encode, to_csr
    c = connect(roots)
    try:
        seg = _segment(c, name)
    finally:
        c.close()
    if seg['status'] == 'active':
        return seg
    if seg['status'] != 'chunked':
        raise ValueError(f'segment {name} is {seg["status"]}, not chunked')
    cf = roots.chunks / name / f'chunks_t{TARGET}.parquet'
    n = pq.ParquetFile(cf).metadata.num_rows
    blocks_dir = roots.guard(roots.index / (name + '.blocks'))
    blocks_dir.mkdir(parents=True, exist_ok=True)
    gpu_check = gpu_check or (lambda: require_gpu_idle(roots))
    enc, embedded, seconds, t0 = None, 0, 0.0, time.perf_counter()
    c = connect(roots)
    try:
        for i, (lo, t) in enumerate(_exact_blocks(cf, block, ['chunk_id', 'text'])):
            ids = t['chunk_id'].to_numpy()
            if not np.array_equal(ids, np.arange(lo, lo + len(ids))):
                raise ValueError('chunk ids must be 0..N-1 in file order')
            dpath, spath = blocks_dir / f'{i:06d}.dense.npy', blocks_dir / f'{i:06d}.sparse.npz'
            if dpath.exists() and spath.exists() and _block_is_trusted(roots, c, name, i, len(ids), dpath, spath):
                continue
            for stale in (dpath, spath):                                  # half a block (crash between the two renames)
                stale.unlink(missing_ok=True)
            if before_block is not None:
                before_block(i)
            if memory is not None:
                memory.check('embed')
            if enc is None:
                import torch
                gpu_check()
                np.random.seed(42)
                torch.manual_seed(42)
                if encoder_factory is None:
                    from r2ai.index.bge_m3 import M3Encoder as encoder_factory
                enc = encoder_factory(max_len=MODEL['max_len'])
            elif i % 25 == 0:
                gpu_check()
            ts = time.perf_counter()
            dense, sparse = encode(enc, t['text'].to_pylist(), batch_size)
            roots.guard(blocks_dir / f'{i:06d}.dense.tmp.npy')
            _save_durably(dpath, lambda f: np.save(f, dense))
            csr = to_csr(sparse, enc.vocab)
            roots.guard(blocks_dir / f'{i:06d}.sparse.tmp.npz')
            _save_durably(spath, lambda f: save_npz(f, csr))
            _record_block(c, name, i, len(dense), dpath, spath)
            dt = time.perf_counter() - ts
            seconds += dt
            embedded += len(dense)
            log_event(roots, 'embed_blocks', {'segment': name, 'block': i, 'chunks': len(dense), 'seconds': dt})
    finally:
        c.close()
        if enc is not None:
            import torch
            del enc
            torch.cuda.empty_cache()
    nblocks = (n + block - 1) // block
    vocabulary = load_npz(blocks_dir / f'{0:06d}.sparse.npz').shape[1]

    def blocks():
        for i, (lo, t) in enumerate(_exact_blocks(cf, block, ['doc_id', 'text'])):
            yield (np.load(blocks_dir / f'{i:06d}.dense.npy'), load_npz(blocks_dir / f'{i:06d}.sparse.npz').tocsr(),
                   t['doc_id'].to_numpy().astype(np.int64), t['text'].to_pylist())

    if len(list(blocks_dir.glob('*.dense.npy'))) != nblocks:
        raise ValueError('missing embedding blocks')
    _write_segment_from_blocks(roots, name, seg['segment_id'], n, vocabulary, cf, blocks())
    _rmtree(roots, blocks_dir)
    stats = {'embed_seconds_this_run': seconds, 'chunks_embedded_this_run': embedded,
             'chunks_per_s_this_run': embedded / seconds if seconds else None, 'wall_seconds': time.perf_counter() - t0}
    _update_segment(roots, name, status='active', vectors='embedded', stats=json.dumps(stats))
    log_event(roots, 'embed', {'segment': name, **stats})
    return {**seg, 'status': 'active', **stats}


def convert_old_sample(roots: Roots, *, writer_block=8192) -> dict:
    """Copy data/index/zh_sample_t256 byte-for-byte into retired segment 1 (control for Z-append)."""
    import pyarrow.parquet as pq
    from scipy.sparse import load_npz
    c = connect(roots)
    try:
        if c.execute('SELECT 1 FROM segments WHERE segment_id=?', (OLD_SAMPLE_ID,)).fetchone():
            raise ValueError('segment 1 (old sample) already exists')
    finally:
        c.close()
    name = f'seg{OLD_SAMPLE_ID:05d}-sample-old-zh_sample_t256'
    cf = roots.sample_chunks / 'chunks_t256.parquet'
    dense = np.load(roots.sample_index / 'dense.npy', mmap_mode='r')
    sp = load_npz(roots.sample_index / 'sparse.npz').tocsr()
    table = pq.read_table(cf, columns=['chunk_id', 'doc_id', 'text'])
    n = table.num_rows
    if not np.array_equal(table['chunk_id'].to_numpy(), np.arange(n)) or dense.shape[0] != n or sp.shape[0] != n:
        raise ValueError('old sample index/chunks rows differ')
    if dense.dtype != np.float16 or not sp.has_canonical_format:
        raise ValueError('old sample index must be fp16 dense and canonical CSR')
    doc = table['doc_id'].to_numpy().astype(np.int64)
    texts = table['text']

    def blocks():
        for lo in range(0, n, writer_block):
            hi = min(n, lo + writer_block)
            yield np.asarray(dense[lo:hi]), sp[lo:hi], doc[lo:hi], texts.slice(lo, hi - lo).to_pylist()

    seg = _write_segment_from_blocks(roots, name, OLD_SAMPLE_ID, n, sp.shape[1], cf, blocks())
    now = time.time()
    sources = {str(p): sha256_file(p) for p in (roots.sample_index / 'dense.npy', roots.sample_index / 'sparse.npz', cf)}
    c = connect(roots)
    try:
        with c:
            c.execute("INSERT INTO segments(segment_id,name,batch,domain,kind,status,rows,global_start,chunks_sha256,"
                      "extractor_version,chunker,model,vectors,note,created_at,updated_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (OLD_SAMPLE_ID, name, 'sample-old', '*', 'sample', 'retired', n, global_start(OLD_SAMPLE_ID),
                       sources[str(cf)], 'zh_sample:54b2b262f4bc9360aec56b557771c0526f6f77fd8f9d2122f2436a6e96f19640',
                       json.dumps({'target': TARGET, 'min_body_tokens': MIN_BODY_TOKENS}), json.dumps(MODEL),
                       'copied:zh_sample_t256', json.dumps({'reason': 'control for Z-append replay', 'sources_sha256': sources}),
                       now, now))
    finally:
        c.close()
    return {'segment_id': OLD_SAMPLE_ID, 'name': name, 'status': 'retired', 'rows': n, 'nnz': int(sp.nnz)}


CHUNK_KEY = ('doc_id', 'doc_ids_group', 'field', 'char_start', 'char_end', 'text')


def _chunk_tuples(rows):
    return [tuple(tuple(r[k]) if k == 'doc_ids_group' else r[k] for k in CHUNK_KEY) for r in rows]


def copy_sample_vectors(roots: Roots, name: str, domain: str, *, writer_block=8192) -> dict:
    """Reuse old-sample vectors for a domain whose new chunks are identical (data gate, not version)."""
    import pyarrow.parquet as pq
    from scipy.sparse import load_npz
    c = connect(roots)
    try:
        seg = _segment(c, name)
    finally:
        c.close()
    new = pq.read_table(roots.chunks / name / f'chunks_t{TARGET}.parquet').to_pylist()
    domains = dict(zip(*pq.read_table(roots.sample_chunks / 'docs.parquet', columns=['doc_id', 'domain']).to_pydict().values()))
    old_rows = pq.read_table(roots.sample_chunks / 'chunks_t256.parquet', columns=['chunk_id', *CHUNK_KEY]).to_pylist()
    old = [r for r in old_rows if domains.get(r['doc_id']) == domain]
    if _chunk_tuples(new) != _chunk_tuples(old):
        raise ValueError(f'{domain}: new chunks differ from the old sample; embed instead of copying')
    idx = np.array([r['chunk_id'] for r in old], dtype=np.int64)
    dense = np.load(roots.sample_index / 'dense.npy', mmap_mode='r')
    sp = load_npz(roots.sample_index / 'sparse.npz').tocsr()
    n = len(idx)
    doc = np.array([r['doc_id'] for r in new], dtype=np.int64)
    texts = [r['text'] for r in new]

    def blocks():
        for lo in range(0, n, writer_block):
            sel = idx[lo:lo + writer_block]
            yield np.asarray(dense[sel]), sp[sel], doc[lo:lo + writer_block], texts[lo:lo + writer_block]

    _write_segment_from_blocks(roots, name, seg['segment_id'], n, sp.shape[1], None, blocks())
    proof = hashlib.sha256(json.dumps(_chunk_tuples(new), ensure_ascii=False).encode()).hexdigest()
    note = {'output_identical_to': 'zh_sample extractor 54b2b262… (old sample index)', 'chunk_tuples_sha256': proof,
            'old_rows_first_last': [int(idx[0]), int(idx[-1])] if n else []}
    _update_segment(roots, name, status='active', vectors='copied:seg00001', note=json.dumps(note))
    return {'name': name, 'vectors': 'copied', 'rows': n, 'old_rows': idx.tolist(), 'chunk_tuples_sha256': proof}


def compare_sample_docs(roots: Roots, batch: str, domain: str) -> dict:
    """Docs gate for copied domains: doc_ids, title, body, status, n_tokens equal to the old sample bundle."""
    import pyarrow.parquet as pq
    keys = ('url_norm', 'doc_ids', 'title', 'body', 'status', 'n_tokens_bge_m3')

    def digest(rows):
        rows = sorted((tuple(tuple(r[k]) if k == 'doc_ids' else r[k] for k in keys) for r in rows))
        return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest(), len(rows)

    new = [r for f in sorted((roots.docs / batch / domain).glob('*.parquet')) for r in pq.read_table(f, columns=list(keys)).to_pylist()]
    old = pq.read_table(roots.sample_docs / f'{domain}.parquet', columns=list(keys)).to_pylist()
    (a, na), (b, nb) = digest(new), digest(old)
    return {'domain': domain, 'new_sha256': a, 'old_sha256': b, 'new_docs': na, 'old_docs': nb, 'equal': a == b}


def segment_rows(roots: Roots, *, include_retired=False):
    c = connect(roots)
    try:
        statuses = ('active', 'retired') if include_retired else ('active',)
        q = ','.join('?' * len(statuses))
        return [dict(zip(('segment_id', 'name', 'batch', 'domain', 'kind', 'status', 'rows'), r)) for r in
                c.execute(f'SELECT segment_id, name, batch, domain, kind, status, rows FROM segments WHERE status IN ({q}) '
                          'ORDER BY segment_id', statuses)]
    finally:
        c.close()


def active_segments(roots: Roots, *, include_retired=False, names=None):
    from r2ai.index.segments import load_segment
    rows = segment_rows(roots, include_retired=include_retired)
    if names is not None:
        rows = [r for r in rows if r['name'] in set(names)]
    return [load_segment(roots.index / r['name']) for r in rows]
