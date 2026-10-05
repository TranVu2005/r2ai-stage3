"""H1 corpus scan: exact + near duplicate clusters over the ok docs of data/docs_vi (CPU only, no model, streamed per file).

    python -m r2ai.dupes.scan features --docs-dir data/docs_vi --out-dir out/runs/H1 [--workers 3]
    python -m r2ai.dupes.scan cluster  --docs-dir data/docs_vi --out-dir out/runs/H1 [--workers 3]

features  one pass over the parquet files: per ok doc with a non-empty body the exact key (normalised body), the key of the
          whole-doc chunk text, the title key, text_sha1, n_tokens, and the MinHash signature of the first doc of every
          exact group with >= 50 tokens. Writes <out>/work/{features.parquet,sig_pos.npy,sigs.npy,files.json}.
cluster   exact groups; LSH over the signatures; MinHash estimate filter; TRUE Jaccard of the 5-word shingle sets
          (re-read from the parquet files in batches of connected components); union-find clusters.
          Writes <out>/clusters.parquet (doc_id, cluster_id, ...) for clusters of size >= 2 and <out>/cluster_stats.json.
Memory: one file per worker at a time; the main process keeps ~1 GiB (features + signatures) and a batch of shingle sets.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs, resolve_path

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.dupes import cluster as C
from r2ai.eval.scorer import normalize_metric

COLS = ['doc_ids', 'domain', 'title', 'description', 'answer', 'body', 'status', 'n_tokens_bge_m3', 'text_sha1']
BATCH_DOCS = 40_000


def _key_norm(norm: str) -> bytes | None:
    import hashlib
    return hashlib.blake2b(norm.encode('utf-8'), digest_size=16).digest() if norm else None


def scan_file(args):
    """-> (rows, sigs): rows = per ok doc with a body: (row, doc_id, n_ids, domain, ntok, ek, ck, tk, tsha, sig_row or -1)."""
    path, min_tokens = args
    t = pq.read_table(path, columns=COLS).to_pylist()
    mh = scan_file.mh = getattr(scan_file, 'mh', None) or C.MinHasher()
    rows, sigs = [], []
    for r, d in enumerate(t):
        if d['status'] != 'ok':
            continue
        body = d['body'] or ''
        norm = normalize_metric(body)
        if not norm:
            continue
        ek = _key_norm(norm)
        ans = (d['answer'] or '').strip()
        ck = ek if not (ans and ans in body) and body.strip() else C.chunk_key(body, d['answer'], d['title'], d['description'])
        ntok = d['n_tokens_bge_m3'] or 0
        srow = -1
        if ntok >= min_tokens:
            sh = C.shingle_hashes(C._WORD.findall(norm))
            sigs.append(mh.signature(sh))
            srow = len(sigs) - 1
        rows.append((r, int(d['doc_ids'][0]), len(d['doc_ids']), d['domain'], int(ntok), ek, ck, C.title_key(d['title']),
                     d['text_sha1'], srow))
    return rows, (np.stack(sigs) if sigs else np.empty((0, C.NUM_PERM), np.uint32))


def shingles_file(args):
    """-> {row: shingle array} of the requested rows of one parquet file."""
    path, want = args
    body = pq.read_table(path, columns=['body'])['body']
    return {r: C.shingle_hashes(C._WORD.findall(normalize_metric(body[r].as_py()))) for r in want}


def rss_gib(pool=None) -> float:
    import psutil
    p = psutil.Process()
    tot = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
    return tot / 2**30


def cmd_features(a) -> int:
    docs_dir, out = resolve_path(a.docs_dir), assert_writable(a.out_dir)
    require_inputs(docs_dir)
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in docs_dir.glob('*.parquet'))
    if a.limit_files:
        files = files[:a.limit_files]
    t0, peak = time.time(), 0.0
    cols = {k: [] for k in ('file', 'row', 'doc_id', 'n_ids', 'domain', 'ntok', 'ek', 'ck', 'tk', 'tsha')}
    seen: set[bytes] = set()
    sig_pos, sig_blocks = [], []
    n = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for fi, (rows, sigs) in enumerate(pool.map(scan_file, [(f, a.min_tokens) for f in files], chunksize=4)):
            keep = []
            for (r, doc, nid, dom, ntok, ek, ck, tk, tsha, srow) in rows:
                for k, v in zip(cols, (fi, r, doc, nid, dom, ntok, ek, ck, tk, tsha)):
                    cols[k].append(v)
                if srow >= 0 and ek not in seen:                       # signature only for the first doc of an exact group
                    keep.append(srow)
                    sig_pos.append(n)
                if ek not in seen:
                    seen.add(ek)
                n += 1
            if keep:
                sig_blocks.append(sigs[keep])
            if fi % 500 == 0:
                peak = max(peak, rss_gib())
                print(f'{fi}/{len(files)} files {n} docs {time.time() - t0:.0f}s rss {peak:.2f} GiB', flush=True)
    peak = max(peak, rss_gib())
    tbl = pa.table({k: pa.array(v, type=pa.binary() if k in ('ek', 'ck', 'tk') else None) for k, v in cols.items()})
    pq.write_table(tbl, work / 'features.parquet')
    np.save(work / 'sig_pos.npy', np.array(sig_pos, dtype=np.int64))
    np.save(work / 'sigs.npy', np.concatenate(sig_blocks) if sig_blocks else np.empty((0, C.NUM_PERM), np.uint32))
    (work / 'files.json').write_text(json.dumps(files), encoding='utf-8')
    meta = {'files': len(files), 'docs': n, 'sigs': len(sig_pos), 'elapsed_s': round(time.time() - t0, 1),
            'peak_rss_gib_main_plus_workers': round(peak, 3), 'workers': a.workers, 'min_tokens': a.min_tokens}
    (work / 'features.meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    return 0


def exact_groups(ek: list) -> np.ndarray:
    first: dict[bytes, int] = {}
    g = np.empty(len(ek), dtype=np.int64)
    for i, k in enumerate(ek):
        g[i] = first.setdefault(k, i)
    return g


def verify(files, rows_file, rows_row, pairs, pos_of_rep, workers: int, stats: dict, jaccard_min: float, batch_docs: int = BATCH_DOCS):
    """True Jaccard of candidate pairs (rep node indices); returns {(i, j): J} for J >= jaccard_min.
    Pairs are verified in batches of connected components so each batch loads only its own shingle sets."""
    nodes = sorted({x for p in pairs for x in p})
    comp = C.components(len(pos_of_rep), pairs)
    by: dict[int, list[int]] = {}
    for x in nodes:
        by.setdefault(int(comp[x]), []).append(x)
    pair_of: dict[int, list] = {}
    for p in pairs:
        pair_of.setdefault(int(comp[p[0]]), []).append(p)
    out, batch, load, nb = {}, [], 0, 0
    stats['verify_components'] = len(by)
    stats['verify_largest_component_nodes'] = max((len(v) for v in by.values()), default=0)

    def flush(cs):
        nonlocal nb
        need: dict[int, list[int]] = {}
        for c in cs:
            for x in by[c]:
                need.setdefault(int(rows_file[pos_of_rep[x]]), []).append(int(rows_row[pos_of_rep[x]]))
        sh: dict[tuple[int, int], np.ndarray] = {}
        fidx = sorted(need)
        with ProcessPoolExecutor(workers) as pool:
            for f, res in zip(fidx, pool.map(shingles_file, [(files[f], sorted(set(need[f]))) for f in fidx], chunksize=4)):
                for r, arr in res.items():
                    sh[(f, r)] = arr
        for c in cs:
            for i, j in pair_of[c]:
                a = sh[(int(rows_file[pos_of_rep[i]]), int(rows_row[pos_of_rep[i]]))]
                b = sh[(int(rows_file[pos_of_rep[j]]), int(rows_row[pos_of_rep[j]]))]
                jac = C.jaccard(a, b)
                if jac >= jaccard_min:
                    out[(i, j)] = jac
        nb += 1
        print(f'verify batch {nb}: {len(cs)} components {sum(len(by[c]) for c in cs)} docs, {len(out)} near links so far, rss {rss_gib():.2f} GiB', flush=True)

    for c in sorted(by, key=lambda c: len(by[c])):
        if batch and load + len(by[c]) > batch_docs:
            flush(batch)
            batch, load = [], 0
        batch.append(c)
        load += len(by[c])
    if batch:
        flush(batch)
    stats['verify_batches'] = nb
    return out


def cmd_cluster(a) -> int:
    out = assert_writable(a.out_dir)
    work = resolve_path(a.work_dir) if a.work_dir else out / 'work'
    out.mkdir(parents=True, exist_ok=True)
    require_inputs(work / 'features.parquet', work / 'sigs.npy')
    t0 = time.time()
    files = json.loads((work / 'files.json').read_text(encoding='utf-8'))
    f = pq.read_table(work / 'features.parquet').to_pydict()
    n = len(f['doc_id'])
    sig_pos = np.load(work / 'sig_pos.npy')
    sigs = np.load(work / 'sigs.npy')
    stats: dict = {'docs': n, 'min_tokens': a.min_tokens, 'jaccard_min': a.jaccard, 'num_perm': C.NUM_PERM, 'bands': C.BANDS,
                   'rows': C.ROWS, 'shingle_words': C.SHINGLE, 'seed': C.SEED, 'short_tokens': C.SHORT_TOKENS}
    group_of = exact_groups(f['ek'])
    stats['exact_groups_ge2'] = int((np.bincount(group_of, minlength=n)[np.unique(group_of)] >= 2).sum())
    stats |= {'run_cap': a.run_cap, 'est_floor': a.est_floor}
    cand = C.lsh_pairs(sigs, run_cap=a.run_cap, stats=stats)
    stats['lsh_candidate_pairs'] = len(cand)
    est = {p: C.est_jaccard(sigs, *p) for p in cand}
    pairs = sorted(p for p, e in est.items() if e >= a.est_floor)
    stats['pairs_after_estimate_filter'] = len(pairs)
    print(f'{n} docs, {len(sig_pos)} sigs, {len(cand)} LSH pairs, {len(pairs)} after estimate filter, {time.time() - t0:.0f}s', flush=True)
    rows_file, rows_row = np.array(f['file']), np.array(f['row'])
    near_node = verify(files, rows_file, rows_row, pairs, sig_pos, a.workers, stats, a.jaccard)       # sig index -> doc position
    near = {(int(sig_pos[i]), int(sig_pos[j])): jac for (i, j), jac in near_node.items()}
    stats['verified_near_pairs'] = len(near)
    cl = C.build_clusters(f['doc_id'], f['ntok'], f['tk'], f['domain'], group_of, near)
    cid = np.full(n, -1, dtype=np.int64)
    pos_of = {d: i for i, d in enumerate(f['doc_id'])}
    rows = []
    for c in cl:
        for d in c.doc_ids:
            cid[pos_of[d]] = c.cluster_id
    for c in cl:
        for d in c.doc_ids:
            i = pos_of[d]
            rows.append((d, c.cluster_id, c.size, c.kind, c.short, c.template, c.boilerplate, f['domain'][i], f['ntok'][i], int(f['n_ids'][i])))
    tbl = pa.table({k: pa.array([r[i] for r in rows]) for i, k in enumerate(
        ('doc_id', 'cluster_id', 'cluster_size', 'kind', 'short', 'template', 'boilerplate', 'domain', 'n_tokens', 'n_ids'))})
    pq.write_table(tbl, out / 'clusters.parquet')
    pq.write_table(pa.table({'cluster_id': [c.cluster_id for c in cl], 'size': [c.size for c in cl], 'kind': [c.kind for c in cl],
                             'exact_groups': [c.exact_groups for c in cl], 'domains': [c.domains for c in cl],
                             'short': [c.short for c in cl], 'template': [c.template for c in cl],
                             'min_jaccard': [c.min_jaccard for c in cl]}), out / 'cluster_table.parquet')
    stats |= {'clusters': len(cl), 'docs_in_clusters': len(rows), 'elapsed_s': round(time.time() - t0, 1), 'peak_rss_gib': round(rss_gib(), 3)}
    (out / 'cluster_stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('features', 'cluster'):
        p = sub.add_parser(name)
        p.add_argument('--docs-dir', required=True)
        p.add_argument('--out-dir', default='out/runs/H1')
        p.add_argument('--workers', type=int, default=3)
        p.add_argument('--min-tokens', type=int, default=C.MIN_TOKENS)
        if name == 'features':
            p.add_argument('--limit-files', type=int, default=0, help='first N files only (smoke test)')
        else:
            p.add_argument('--jaccard', type=float, default=C.JACCARD)
            p.add_argument('--run-cap', type=int, default=C.RUN_CAP, help='LSH bucket size above which docs pair with their next N neighbours only')
            p.add_argument('--est-floor', type=float, default=C.EST_FLOOR, help='MinHash estimate below which a candidate pair is not verified')
            p.add_argument('--work-dir', default=None, help='features dir (default <out-dir>/work); lets a sensitivity run write elsewhere')
    a = ap.parse_args(argv)
    return cmd_features(a) if a.cmd == 'features' else cmd_cluster(a)


if __name__ == '__main__':
    sys.exit(main())
