"""S37 local pipeline for 120ask batches pulled from the VPS (deploy/s37).

    python -m r2ai.s37.ingest pool                              # reference top-211 / pool13 cut per query (cached)
    python -m r2ai.s37.ingest batch out/runs/s37/pulled/batch-0001.tar [--device defer|cuda|cpu]   # default defer
    python -m r2ai.s37.ingest embed [--device cuda|cpu]         # segments waiting for the GPU (cuda refuses while it is busy)
    python -m r2ai.s37.ingest status                            # which segments wait for the GPU
    python -m r2ai.s37.ingest report                            # rebuild pairs_new over every embedded segment

A batch runs on CPU: verify sha256 -> extract (the existing 120ask extractor, keep body with >= 50 Han chars) -> new canonical-schema
segment `seg37NNN-s37-bNNNN-120ask.com` + its t256 chunks (canonical chunker, for C-max later) -> state `waiting_gpu`.
`embed` then runs BGE-M3 doc-level dense (doc_text, first 4096 chars, max_len 512; same settings as the
S26 full_doc_dense) -> pairs of new docs that would enter the cosine top-211 (or the S29 1.3 x K_q pool) of any of the 1,200 queries.

Nothing existing is modified: the canonical bundle, the S26 vectors and the S29 pool are read only. Chunks have no vectors (has_vector False). No scoring (4B / Jina / C-max) is run.
"""
from __future__ import annotations

from r2ai.paths import DATA_DIR, RUNS_DIR, assert_writable

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import polars as pl

RUN = RUNS_DIR / 's37'
STORE = DATA_DIR / 's37_120ask'
CANONICAL = DATA_DIR / 'canonical-20261018' / 'docs.parquet'
S26_VECTORS = RUNS_DIR / 's26-gold-structure' / 'full' / 'full_doc_dense.npz'
QUERY_VECTORS = RUNS_DIR / 's29-qwen-judge' / 'query_vectors.npz'
S29_POOL = RUNS_DIR / 's29-qwen-judge' / 'candidate_pool.parquet'
PHARMACITY_DOCS = RUNS_DIR / 's27-pharmacity' / 'handoff' / 'cc-s27-pharmacity' / 'lots'
PHARMACITY_VECTORS = RUNS_DIR / 's27-pharmacity' / 'vectors' / 'handoff'
URLS = RUN / 'urls.tsv'
POOL = RUN / 'pool.npz'
TOP = 211
KEEP = 400                      # cosine scores kept per query in the reference pool (>= max pool13 size)
MIN_HAN = 50
HOST = '120ask.com'
FIRST_SEGMENT = 37001
EMBED_CHARS, EMBED_LEN = 4096, 512
HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]')
WS = re.compile(r'\s+')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes):
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_json(path: Path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=1).encode('utf-8'))


def atomic_npz(path: Path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write(path, buf.getvalue())


def atomic_parquet(df: pl.DataFrame, path: Path):
    buf = io.BytesIO()
    df.write_parquet(buf, compression='zstd')
    atomic_write(path, buf.getvalue())


# ---- reference pool -------------------------------------------------------------------------------------------------

def load_doc_vectors() -> tuple[np.ndarray, np.ndarray]:
    """S26 canonical doc-level vectors plus the Pharmacity lots (the same set the S29/S31 pools were cut from)."""
    base = np.load(S26_VECTORS)
    ids, dense = [base['doc_id'].astype(np.int64)], [base['dense']]
    if PHARMACITY_DOCS.exists():
        for lot in sorted(p for p in PHARMACITY_DOCS.iterdir() if p.is_dir()):
            f = PHARMACITY_VECTORS / lot.name / 'dense_doc_level.npz'
            if f.exists():
                z = np.load(f)
                ids.append(z['doc_id'].astype(np.int64))
                dense.append(z['dense'])
    all_ids = np.concatenate(ids)
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError('duplicate doc_id across canonical and Pharmacity vectors')
    return all_ids, np.concatenate(dense)


def load_queries() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(QUERY_VECTORS)
    return z['q_ids'].astype(np.int64), z['q_dense']


def top_scores(q: np.ndarray, dense: np.ndarray, keep: int, block: int = 50_000):
    """Per query the `keep` largest cosines (descending) and their row index into `dense`."""
    nq = len(q)
    qf = q.astype(np.float32)
    best_s = np.full((nq, keep), -2.0, np.float32)
    best_i = np.full((nq, keep), -1, np.int64)
    for start in range(0, len(dense), block):
        s = qf @ dense[start:start + block].astype(np.float32).T
        cand_s = np.concatenate([best_s, s], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(np.arange(start, start + s.shape[1]), (nq, s.shape[1]))], axis=1)
        sel = np.argpartition(-cand_s, keep - 1, axis=1)[:, :keep]
        best_s = np.take_along_axis(cand_s, sel, 1)
        best_i = np.take_along_axis(cand_i, sel, 1)
    order = np.argsort(-best_s, axis=1, kind='stable')
    return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)


def build_pool(force: bool = False) -> dict:
    """Reference cut per query: cos at rank 211 and at the end of the S29 pool13 (1.3 x K_q). Cached in pool.npz."""
    if POOL.exists() and not force:
        z = np.load(POOL)
        return {k: z[k] for k in z.files}
    q_ids, q = load_queries()
    ids, dense = load_doc_vectors()
    scores, rows = top_scores(q, dense, KEEP)
    p13 = pl.read_parquet(S29_POOL, columns=['query_id', 'cosine_rank', 'cosine_score'])
    size13 = np.zeros(len(q_ids), np.int64)
    sizes = dict(p13.group_by('query_id').agg(pl.col('cosine_rank').max().alias('n')).iter_rows())
    for k, qid in enumerate(q_ids.tolist()):
        size13[k] = sizes.get(qid, 0)
    cut = np.maximum(TOP, size13)
    if cut.max() > KEEP:
        raise ValueError(f'pool13 size {cut.max()} exceeds KEEP={KEEP}')
    out = {'q_ids': q_ids, 'scores': scores, 'doc_ids': ids[rows], 'size13': size13, 'cut': cut,
           'thr211': scores[:, TOP - 1], 'thr': scores[np.arange(len(q_ids)), cut - 1]}
    atomic_npz(POOL, **out)
    return out


def parity(pool: dict) -> dict:
    """Compare the recomputed cosines with the S29 pool13 and the S31 pool_index (rank 211): documents the reference quality."""
    q_pos = {int(x): i for i, x in enumerate(pool['q_ids'])}
    p13 = pl.read_parquet(S29_POOL, columns=['query_id', 'cosine_rank', 'cosine_score'])
    qi = np.array([q_pos[x] for x in p13['query_id'].to_list()])
    mine = pool['scores'][qi, p13['cosine_rank'].to_numpy() - 1]
    out = {'s29_pool13_pairs': p13.height, 's29_frac_within_1e-3': float((np.abs(mine - p13['cosine_score'].to_numpy()) <= 1e-3).mean())}
    f = RUNS_DIR / 's31-strong-judge' / 'pool' / 'pool_index.parquet'
    if f.exists():
        r = pl.read_parquet(f, columns=['query_id', 'doc_id', 'cosine_rank', 'cosine_score']).filter(pl.col('cosine_rank') == TOP)
        ri = np.array([q_pos[x] for x in r['query_id'].to_list()])
        d = np.abs(pool['thr211'][ri] - r['cosine_score'].to_numpy())
        out.update({'s31_rank211_queries': r.height, 's31_rank211_max_abs_diff': float(d.max())})
    return out


# ---- extract ----------------------------------------------------------------------------------------------------------

def read_tar(tar: Path) -> tuple[dict, list[dict], list[dict]]:
    """-> manifest, log rows, shard records (last record wins per doc_id), after verifying every sha256."""
    side = tar.with_name(tar.name + '.sha256')
    if side.exists():
        want = side.read_text('utf-8').split()[0]
        if sha256_file(tar) != want:
            raise ValueError(f'{tar.name}: sha256 differs from {side.name}')
    records: dict[str, dict] = {}
    with tarfile.open(tar) as t:
        manifest = json.load(t.extractfile('MANIFEST.json'))
        log = [json.loads(x) for x in t.extractfile('log.jsonl').read().decode('utf-8').splitlines() if x.strip()]
        for sh in manifest['shards']:
            blob = t.extractfile(f"shards/{sh['name']}").read()
            if hashlib.sha256(blob).hexdigest() != sh['sha256']:
                raise ValueError(f"{sh['name']}: sha256 differs from MANIFEST")
            for line in gzip.decompress(blob).decode('utf-8').splitlines():
                if line.strip():
                    r = json.loads(line)
                    records[r['doc_id']] = r
    return manifest, log, list(records.values())


def url_groups(path: Path = URLS) -> dict[str, list[int]]:
    out = {}
    for line in path.read_text('utf-8').splitlines():
        parts = line.split('\t')
        if len(parts) >= 3:
            out[parts[0]] = [int(x) for x in parts[2].split(',')]
    return out


def extract_record(rec: dict) -> dict:
    from r2ai.index.chunk import doc_layout
    from r2ai.zh_full_ingest.extract import extract
    doc = extract(rec['html'], HOST)
    body = doc.body
    r = {'title': doc.title, 'question': doc.question, 'answer': doc.answer, 'body': body, 'paragraphs': doc.paragraphs}
    doc_text = doc_layout(r)[0]
    clean = WS.sub(' ', body).strip()
    return {'title': doc.title, 'doc_text': doc_text, 'body_cp': len(clean), 'han': len(HAN.findall(clean)),
            'extractor': doc.extractor, 'layout_src': r}


def existing_ids() -> set[int]:
    have = set(pl.read_parquet(CANONICAL, columns=['doc_id', 'doc_text']).filter(pl.col('doc_text').fill_null('') != '')['doc_id'].to_list())
    for f in STORE.glob('seg*/docs.parquet'):
        have.update(pl.read_parquet(f, columns=['doc_id'])['doc_id'].to_list())
    return have


def build_docs(records: list[dict], groups: dict[str, list[int]], have: set[int], segment: str, min_han: int = MIN_HAN):
    rows, srcs, drop = [], {}, {'no_group': 0, 'already_have': 0, 'thin': 0, 'suspect_thin': 0}
    for rec in records:
        ids = groups.get(rec['doc_id'])
        if not ids:
            drop['no_group'] += 1
            continue
        todo = [i for i in ids if i not in have]
        if not todo:
            drop['already_have'] += 1
            continue
        e = extract_record(rec)
        if e['han'] < min_han:
            drop['thin'] += 1
            continue
        sha = hashlib.sha256(e['doc_text'].encode('utf-8')).hexdigest()
        srcs[sha] = (todo, e['layout_src'])
        for i in todo:
            rows.append({'doc_id': i, 'lang': 'zh', 'host': HOST, 'url': rec['url'], 'phys_id': todo[0], 'doc_text': e['doc_text'],
                         'body_cp': e['body_cp'], 'han': e['han'], 'quality': 'good', 'rule': 's37', 'text_sha256': sha,
                         'segment': segment, 'source_path': rec['final_url']})
    return rows, drop, srcs


# ---- chunk t256 -------------------------------------------------------------------------------------------------------

CHUNK_TARGET = 256


def chunk_segment(srcs: dict, segment: str, target: int = CHUNK_TARGET) -> pl.DataFrame:
    """t256 chunks of the new docs with the canonical chunker (r2ai.index.chunk: doc_layout + chunk_paragraphs, same blob
    filter, verbatim check). One doc per distinct text; doc_id = first id of the group, doc_ids = whole group. CPU only."""
    from r2ai.gold_check.gold_check_common import load_tokenizer
    from r2ai.index.chunk import chunk_paragraphs, doc_layout, _token_counts
    tok = load_tokenizer()
    cache: dict[str, int] = {}

    def ntok(x: str) -> int:
        v = cache.get(x)
        if v is None:
            v = cache[x] = len(tok(x, add_special_tokens=False)['input_ids'])
        return v

    rows = []
    for ids, r in srcs.values():
        doc_text, spans, paras = doc_layout(r)
        ptoks = _token_counts(tok, [doc_text[s:e] for _, s, e in paras])
        for f, s, e in list(spans) + chunk_paragraphs(doc_text, paras, ptoks, target, ntok):
            rows.append((ids[0], ids, f, s, e, doc_text[s:e]))
    ntoks = _token_counts(tok, [x[5] for x in rows])
    keep = [not (nt > 1.5 * target and not any(ch.isspace() for ch in x[5])) for x, nt in zip(rows, ntoks)]
    rows = [x for x, k in zip(rows, keep) if k]
    ntoks = [nt for nt, k in zip(ntoks, keep) if k]
    texts = {ids[0]: layout[0] for ids, r in srcs.values() for layout in [doc_layout(r)]}
    bad = sum(texts[d][s:e] != t or not t.strip() for d, _, _, s, e, t in rows)
    if bad:
        raise AssertionError(f'{bad} chunks are not verbatim substrings')
    return pl.DataFrame({'chunk_id': list(range(len(rows))), 'doc_id': [x[0] for x in rows], 'doc_ids': [x[1] for x in rows],
                         'field': [x[2] for x in rows], 'char_start': [x[3] for x in rows], 'char_end': [x[4] for x in rows],
                         'text': [x[5] for x in rows], 'n_tokens': ntoks, 'segment': segment, 'has_vector': False},
                        schema={'chunk_id': pl.Int64, 'doc_id': pl.Int64, 'doc_ids': pl.List(pl.Int64), 'field': pl.String,
                                'char_start': pl.Int32, 'char_end': pl.Int32, 'text': pl.String, 'n_tokens': pl.Int32,
                                'segment': pl.String, 'has_vector': pl.Boolean})


# ---- embed ------------------------------------------------------------------------------------------------------------

def embed_texts(texts: list[str], device: str) -> np.ndarray:
    """Same recipe as S26 full_embed.py: first 4096 chars, length-sorted batches, dense CLS L2-normalised fp16."""
    from r2ai.index.bge_m3 import M3Encoder
    enc = M3Encoder(device=device, fp16=(device != 'cpu'), max_len=EMBED_LEN)
    texts = [t[:EMBED_CHARS] for t in texts]
    order = np.argsort([len(x) for x in texts], kind='stable')
    out = np.zeros((len(texts), 1024), np.float16)
    i = 0
    while i < len(order):
        bs = 64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16
        sel = order[i:i + bs]
        out[sel] = enc.encode_batch([texts[j] for j in sel])[0]
        i += bs
    return out


# ---- report -----------------------------------------------------------------------------------------------------------

def pair_rows(pool: dict, q: np.ndarray, doc_ids: np.ndarray, vecs: np.ndarray, segment: str, batch: str) -> pl.DataFrame:
    cos = vecs.astype(np.float32) @ q.astype(np.float32).T                      # [docs, queries]
    thr, thr211, ref = pool['thr'], pool['thr211'], pool['scores']
    hit = cos > thr[None, :]                                                    # beats the end of top-211 / pool13
    di, qi = np.nonzero(hit)
    if len(di) == 0:
        return pl.DataFrame(schema={'query_id': pl.Int64, 'doc_id': pl.Int64, 'cosine': pl.Float64, 'entry_rank': pl.Int64,
                                    'in_top211': pl.Boolean, 'in_pool13': pl.Boolean, 'segment': pl.String, 'batch': pl.String})
    c = cos[di, qi]
    entry = np.array([1 + int((ref[k] > v).sum()) for k, v in zip(qi, c)])      # rank among the old top list
    return pl.DataFrame({'query_id': pool['q_ids'][qi], 'doc_id': doc_ids[di], 'cosine': c.astype(np.float64), 'entry_rank': entry,
                         'in_top211': c > thr211[qi], 'in_pool13': entry <= pool['size13'][qi],
                         'segment': segment, 'batch': batch})


def report(pool: dict | None = None) -> dict:
    pool = pool or build_pool()
    q_ids, q = load_queries()
    if not np.array_equal(q_ids, pool['q_ids']):
        raise ValueError('query order differs from the cached pool')
    frames, docs, per_seg = [], [], {}
    for seg_dir in sorted(STORE.glob('seg*')):
        meta = seg_meta(seg_dir)
        if meta['state'] != 'embedded':
            continue
        z = np.load(seg_dir / 'vectors.npz')
        df = pair_rows(pool, q, z['doc_id'].astype(np.int64), z['dense'], seg_dir.name, meta['batch'])
        frames.append(df)
        per_seg[seg_dir.name] = {'docs': int(len(z['doc_id'])), 'pairs': df.height, 'docs_in_pairs': int(df['doc_id'].n_unique()),
                                 'queries_hit': int(df['query_id'].n_unique()), 'in_top211': int(df['in_top211'].sum())}
    pairs = pl.concat(frames) if frames else pl.DataFrame()
    summary = {'segments': per_seg, 'reference': {'top': TOP, 'parity': parity(pool),
                                                  'pool13_sizes': [int(pool['size13'].min()), int(pool['size13'].max())]}}
    if pairs.height:
        atomic_parquet(pairs.sort(['query_id', 'cosine'], descending=[False, True]), RUN / 'pairs_new.parquet')
        need = pairs.select('doc_id', 'segment').unique().sort('doc_id')
        atomic_parquet(need, RUN / 'docs_need_scoring.parquet')
        by_q = pairs.group_by('query_id').agg(pl.len().alias('pairs'), pl.col('in_top211').sum().alias('top211'),
                                              pl.col('cosine').max().alias('best_cosine')).sort('pairs', descending=True)
        atomic_parquet(by_q, RUN / 'pairs_new_by_query.parquet')
        summary.update({'pairs': pairs.height, 'pairs_top211': int(pairs['in_top211'].sum()), 'docs': need.height,
                        'queries_hit': by_q.height, 'top_queries': by_q.head(10).to_dicts()})
    else:
        summary.update({'pairs': 0, 'docs': 0, 'queries_hit': 0})
    atomic_json(RUN / 'report.json', summary)
    return summary


# ---- batch ------------------------------------------------------------------------------------------------------------

def next_segment(batch: str) -> str:
    n = FIRST_SEGMENT + len(list(STORE.glob('seg*')))
    return f'seg{n}-s37-{batch}-{HOST}'


def seg_meta(seg_dir: Path) -> dict:
    return json.loads((seg_dir / 'segment.json').read_text('utf-8'))


def gpu_free(min_free_mib: int = 3072, max_util: int = 10) -> tuple[bool, str]:
    """Same idea as the VI2 watcher: usable only when utilisation is low and enough VRAM is free."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
        used, total, util = (int(x) for x in out.split(','))
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return False, f'nvidia-smi failed: {e}'
    return (total - used >= min_free_mib and util <= max_util), f'used {used}/{total} MiB, util {util}%'


def run_batch(tar: Path, device: str, min_han: int = MIN_HAN, urls: Path = URLS) -> dict:
    """Phase 1 (CPU): verify -> extract -> segment docs + t256 chunks. device != 'defer' also runs phase 2 (embed + report)."""
    t0 = time.time()
    batch = re.sub(r'\.tar$', '', tar.name).replace('batch-', 'b')
    tar_sha = sha256_file(tar)
    for f in STORE.glob('seg*/segment.json'):
        if seg_meta(f.parent).get('tar_sha256') == tar_sha:
            return {'status': 'already_extracted', 'segment': f.parent.name, 'state': seg_meta(f.parent)['state']}
    manifest, log, records = read_tar(tar)
    segment = next_segment(batch)
    rows, drop, srcs = build_docs(records, url_groups(urls), existing_ids(), segment, min_han)
    stats = {'batch': batch, 'tar_sha256': tar_sha, 'urls_in_log': len(log), 'ok': manifest['ok'], 'suspect_in_log': manifest['suspect'],
             'failed_in_log': manifest['failed'], 'status_counts': dict(pl.Series([r['status'] for r in log]).value_counts().iter_rows()),
             'records': len(records), 'docs_kept': len(rows), 'dropped': drop, 'min_han': min_han}
    if not rows:
        atomic_json(RUN / 'batches' / f'{batch}.json', {**stats, 'status': 'no_docs'})
        return {**stats, 'status': 'no_docs'}
    df = pl.DataFrame(rows)
    chunks = chunk_segment(srcs, segment)
    seg_dir = STORE / segment
    atomic_parquet(df, seg_dir / 'docs.parquet')
    atomic_parquet(chunks, seg_dir / 'chunks_t256.parquet')
    stats.update({'segment': segment, 'han_p50': int(df['han'].median()), 'chunks_t256': chunks.height,
                  'chunk_docs': chunks['doc_id'].n_unique(), 'state': 'waiting_gpu',
                  'docs_sha256': sha256_file(seg_dir / 'docs.parquet'), 'chunks_sha256': sha256_file(seg_dir / 'chunks_t256.parquet'),
                  'extract_seconds': round(time.time() - t0, 1), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    atomic_json(seg_dir / 'segment.json', stats)
    atomic_json(RUN / 'batches' / f'{batch}.json', {**stats, 'status': 'extracted'})
    if device != 'defer':
        stats['embed'] = embed_pending(device)
    return {**stats, 'status': 'extracted'}


def embed_pending(device: str) -> dict:
    """Phase 2: BGE-M3 doc-level vectors + pair report for every segment still `waiting_gpu`. cuda waits for a free GPU."""
    todo = [d for d in sorted(STORE.glob('seg*')) if seg_meta(d)['state'] == 'waiting_gpu']
    if not todo:
        return {'status': 'nothing_waiting'}
    if device == 'cuda':
        ok, why = gpu_free()
        if not ok:
            return {'status': 'gpu_busy', 'detail': why, 'waiting': [d.name for d in todo]}
    pool = build_pool()
    done = []
    for seg_dir in todo:
        t0 = time.time()
        meta = seg_meta(seg_dir)
        df = pl.read_parquet(seg_dir / 'docs.parquet')
        uniq = df.unique('text_sha256', keep='first', maintain_order=True)       # ids that share a URL share one vector
        vec = embed_texts(uniq['doc_text'].to_list(), device)
        by_sha = {x: vec[i] for i, x in enumerate(uniq['text_sha256'].to_list())}
        dense = np.stack([by_sha[x] for x in df['text_sha256'].to_list()])
        atomic_npz(seg_dir / 'vectors.npz', doc_id=df['doc_id'].to_numpy().astype(np.int64), dense=dense)
        meta.update({'state': 'embedded', 'embed_device': device, 'embed_seconds': round(time.time() - t0, 1),
                     'vectors_sha256': sha256_file(seg_dir / 'vectors.npz')})
        atomic_json(seg_dir / 'segment.json', meta)
        done.append(seg_dir.name)
    summary = report(pool)
    for name in done:
        meta = seg_meta(STORE / name)
        meta['report'] = summary['segments'][name]
        atomic_json(STORE / name / 'segment.json', meta)
        atomic_json(RUN / 'batches' / f"{meta['batch']}.json", {**meta, 'status': 'embedded'})
    return {'status': 'embedded', 'segments': done, 'cumulative': {k: summary.get(k) for k in ('pairs', 'pairs_top211', 'docs', 'queries_hit')}}


def status() -> dict:
    segs = {d.name: {k: seg_meta(d).get(k) for k in ('state', 'batch', 'docs_kept', 'chunks_t256', 'embed_device', 'report')}
            for d in sorted(STORE.glob('seg*'))}
    ok, why = gpu_free()
    return {'segments': segs, 'waiting_gpu': [n for n, m in segs.items() if m['state'] == 'waiting_gpu'], 'gpu_free': ok, 'gpu': why}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pool')
    p.add_argument('--force', action='store_true')
    b = sub.add_parser('batch')
    b.add_argument('tar', type=Path)
    b.add_argument('--device', default='defer', choices=['defer', 'cuda', 'cpu'],
                   help='defer (default): extract + chunk only, segment waits for the GPU; cpu only to smoke-test the first batch')
    b.add_argument('--min-han', type=int, default=MIN_HAN)
    b.add_argument('--urls', type=Path, default=URLS)
    e = sub.add_parser('embed')
    e.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    sub.add_parser('report')
    sub.add_parser('status')
    a = ap.parse_args(argv)
    if a.cmd == 'pool':
        pool = build_pool(a.force)
        out = {'queries': len(pool['q_ids']), 'thr211_min_max': [float(pool['thr211'].min()), float(pool['thr211'].max())],
               'pool13_sizes': [int(pool['size13'].min()), int(pool['size13'].max())], 'parity': parity(pool)}
    elif a.cmd == 'batch':
        out = run_batch(a.tar, a.device, a.min_han, a.urls)
    elif a.cmd == 'embed':
        out = embed_pending(a.device)
    elif a.cmd == 'status':
        out = status()
    else:
        out = report()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
