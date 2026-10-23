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

from r2ai.s37 import qa

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
HOST = '120ask.com'
FIRST_SEGMENT = 37001
EMBED_CHARS, EMBED_LEN = 4096, 512


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
    e = qa.extract(rec['html'])
    r = {k: e[k] for k in ('title', 'question', 'answer', 'answers', 'body', 'paragraphs')}
    return {'title': e['title'], 'doc_text': doc_layout(r)[0], 'body_cp': e['body_cp'], 'han': e['han'], 'quality': e['quality'],
            'extractor': e['extractor'], 'layout_src': r}


def existing_ids() -> set[int]:
    have = set(pl.read_parquet(CANONICAL, columns=['doc_id', 'doc_text']).filter(pl.col('doc_text').fill_null('') != '')['doc_id'].to_list())
    for f in STORE.glob('seg*/docs.parquet'):
        have.update(pl.read_parquet(f, columns=['doc_id'])['doc_id'].to_list())
    return have


def build_docs(records: list[dict], groups: dict[str, list[int]], have: set[int], segment: str, shards: dict | None = None,
               tar_sha: str = ''):
    """-> docs rows, provenance rows, drop counts, layout sources by text sha. Thin docs (< 200 cp and < 50 Han) are kept."""
    rows, prov, srcs, drop = [], [], {}, {'no_group': 0, 'already_have': 0, 'thin_kept': 0, 'empty_body': 0}
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
        drop['thin_kept'] += e['quality'] == 'thin'
        drop['empty_body'] += e['body_cp'] == 0
        sha = hashlib.sha256(e['doc_text'].encode('utf-8')).hexdigest()
        srcs[sha] = (todo, e['layout_src'])
        for i in todo:
            rows.append({'doc_id': i, 'lang': 'zh', 'host': HOST, 'url': rec['url'], 'phys_id': todo[0], 'doc_text': e['doc_text'],
                         'body_cp': e['body_cp'], 'han': e['han'], 'quality': e['quality'], 'rule': 's37', 'text_sha256': sha,
                         'segment': segment, 'source_path': rec['final_url']})
            prov.append({'doc_id': i, 'phys_id': todo[0], 'url': rec['url'], 'final_url': rec['final_url'], 'http': rec['http'],
                         'fetched_at': float(rec['fetched_at']), 'encoding': rec['encoding'], 'extractor': e['extractor'],
                         'n_answers': len(e['layout_src']['answers']), 'quality': e['quality'], 'text_sha256': sha,
                         'raw_shard': (shards or {}).get(rec['doc_id']), 'tar_sha256': tar_sha, 'source': 's37-vps-sydney'})
    return rows, prov, drop, srcs


# ---- chunk t256 -------------------------------------------------------------------------------------------------------

CHUNK_TARGET = 256


CHUNK_SCHEMA = {'chunk_id': pl.Int64, 'doc_id': pl.Int64, 'doc_ids': pl.List(pl.Int64), 'field': pl.String,
                'char_start': pl.Int32, 'char_end': pl.Int32, 'text': pl.String, 'n_tokens': pl.Int32, 'segment': pl.String,
                'has_vector': pl.Boolean}


def chunk_segment(srcs: dict, segment: str, target: int = CHUNK_TARGET) -> tuple[pl.DataFrame, pl.DataFrame]:
    """-> (t256, struct). t256: the canonical chunker (r2ai.index.chunk: doc_layout + chunk_paragraphs, min body tokens 1, same
    blob filter, verbatim check). struct: question + one answer per chunk (r2ai.s37.qa.struct_chunks). One doc per distinct text;
    doc_id = first id of the group, doc_ids = whole group. CPU only."""
    from r2ai.gold_check.gold_check_common import load_tokenizer
    from r2ai.index.chunk import chunk_paragraphs, doc_layout, _token_counts
    tok = load_tokenizer()
    cache: dict[str, int] = {}

    def ntok(x: str) -> int:
        v = cache.get(x)
        if v is None:
            v = cache[x] = len(tok(x, add_special_tokens=False)['input_ids'])
        return v

    t_rows, s_rows, texts = [], [], {}
    for ids, r in srcs.values():
        doc_text, spans, paras = doc_layout(r)
        texts[ids[0]] = doc_text
        ptoks = _token_counts(tok, [doc_text[s:e] for _, s, e in paras])
        for f, s, e in list(spans) + chunk_paragraphs(doc_text, paras, ptoks, target, ntok):
            t_rows.append((ids[0], ids, f, s, e, doc_text[s:e], -1))
        for f, s, e, k in qa.struct_chunks(doc_text, r, paras, ntok, target):
            s_rows.append((ids[0], ids, f, s, e, doc_text[s:e], k))

    def frame(rows, extra=False):
        ntoks = _token_counts(tok, [x[5] for x in rows])
        keep = [not (nt > 1.5 * target and not any(ch.isspace() for ch in x[5])) for x, nt in zip(rows, ntoks)]
        rows = [x for x, k in zip(rows, keep) if k]
        ntoks = [nt for nt, k in zip(ntoks, keep) if k]
        bad = sum(texts[d][s:e] != t or not t.strip() for d, _, _, s, e, t, _ in rows)
        if bad:
            raise AssertionError(f'{bad} chunks are not verbatim substrings')
        cols = {'chunk_id': list(range(len(rows))), 'doc_id': [x[0] for x in rows], 'doc_ids': [x[1] for x in rows],
                'field': [x[2] for x in rows], 'char_start': [x[3] for x in rows], 'char_end': [x[4] for x in rows],
                'text': [x[5] for x in rows], 'n_tokens': ntoks, 'segment': segment, 'has_vector': False}
        schema = dict(CHUNK_SCHEMA)
        if extra:
            cols['answer_idx'] = [x[6] for x in rows]
            cols['n_chars'] = [x[4] - x[3] for x in rows]
            schema['answer_idx'] = pl.Int32
            schema['n_chars'] = pl.Int32
        return pl.DataFrame(cols, schema=schema)

    return frame(t_rows), frame(s_rows, extra=True)


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


PAIR_ID_BASE = 1_000_000          # S31 pair_ids are 0..433,731; S37 pairs start well above
# measured throughputs (docs/PROGRESS.md S29/S31/S35): 4B Kaggle T4 2.87 pair/s per T4, 4B local RTX 3050 ~1.7 pair/s,
# 0.6B local 10.9 pair/s, Jina v3.5 b10s0 17.1 s/query (~250 docs per query list)
RATES = {'4b_kaggle_t4': 2.87, '4b_local': 1.7, 'cmax_06b_local': 10.9, 'jina_s_per_query': 17.1}
CMAX_TOP = 3


def eta(pairs_union: int, queries_union: int) -> dict:
    h = lambda x: round(x / 3600, 2)
    return {'4b_pairs': pairs_union, '4b_local_hours': h(pairs_union / RATES['4b_local']),
            '4b_kaggle_hours_per_t4': h(pairs_union / RATES['4b_kaggle_t4']),
            '4b_local_ok_under_15k_pairs': pairs_union < 15_000,
            'jina_queries_to_rescore': queries_union, 'jina_hours': h(queries_union * RATES['jina_s_per_query']),
            'cmax_chunk_pairs': pairs_union * CMAX_TOP, 'cmax_hours': h(pairs_union * CMAX_TOP / RATES['cmax_06b_local']),
            'note': 'ETA from earlier measured rates, not re-measured on S37 pairs; no GPU job started'}


def assign_pair_ids(pairs: pl.DataFrame) -> pl.DataFrame:
    """Append-only (query_id, doc_id) -> pair_id registry, so ids stay stable when later batches add pairs."""
    path = RUN / 'pair_ids.parquet'
    reg = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema={'query_id': pl.Int64, 'doc_id': pl.Int64, 'pair_id': pl.Int64})
    new = (pairs.select('query_id', 'doc_id').join(reg.select('query_id', 'doc_id'), on=['query_id', 'doc_id'], how='anti')
           .unique().sort('query_id', 'doc_id'))
    if new.height:
        start = int(reg['pair_id'].max()) + 1 if reg.height else PAIR_ID_BASE
        reg = pl.concat([reg, new.with_columns(pl.Series('pair_id', range(start, start + new.height), dtype=pl.Int64))])
        atomic_parquet(reg, path)
    return pairs.join(reg, on=['query_id', 'doc_id'], how='left')


def report(pool: dict | None = None) -> dict:
    pool = pool or build_pool()
    q_ids, q = load_queries()
    if not np.array_equal(q_ids, pool['q_ids']):
        raise ValueError('query order differs from the cached pool')
    frames, per_seg = [], {}
    for seg_dir in sorted(STORE.glob('seg*')):
        meta = seg_meta(seg_dir)
        if meta['state'] != 'embedded':
            continue
        z = np.load(seg_dir / 'vectors.npz')
        df = pair_rows(pool, q, z['doc_id'].astype(np.int64), z['dense'], seg_dir.name, meta['batch'])
        frames.append(df)
        per_seg[seg_dir.name] = {'docs': int(len(z['doc_id'])), 'pairs': df.height, 'docs_in_pairs': int(df['doc_id'].n_unique()),
                                 'queries_hit': int(df['query_id'].n_unique()), 'in_top211': int(df['in_top211'].sum()),
                                 'in_pool13': int(df['in_pool13'].sum())}
    pairs = pl.concat(frames) if frames else pl.DataFrame()
    summary = {'segments': per_seg, 'reference': {'top': TOP, 'parity': parity(pool),
                                                  'pool13_sizes': [int(pool['size13'].min()), int(pool['size13'].max())]}}
    if pairs.height:
        # rank among the old list plus the new pairs that beat it (the new list a scorer will see)
        pairs = pairs.with_columns((pl.col('entry_rank') + pl.col('cosine').rank('ordinal', descending=True).over('query_id') - 1)
                                   .alias('cosine_rank'))
        pairs = assign_pair_ids(pairs).rename({'cosine': 'cosine_score'}).with_columns(
            pl.lit('zh').alias('lang'), pl.lit(None, dtype=pl.Int32).alias('shard'))
        pairs = pairs.select('query_id', 'doc_id', 'cosine_rank', 'cosine_score', 'in_pool13', 'lang', 'pair_id', 'shard',
                             'in_top211', 'entry_rank', 'segment', 'batch').sort('query_id', 'cosine_rank')
        atomic_parquet(pairs, RUN / 'pairs_new.parquet')
        need = pairs.select('doc_id', 'segment').unique().sort('doc_id')
        atomic_parquet(need, RUN / 'docs_need_scoring.parquet')
        by_q = pairs.group_by('query_id').agg(pl.len().alias('pairs'), pl.col('in_top211').sum().alias('top211'),
                                              pl.col('in_pool13').sum().alias('pool13'),
                                              pl.col('cosine_score').max().alias('best_cosine')).sort('pairs', descending=True)
        atomic_parquet(by_q, RUN / 'pairs_new_by_query.parquet')
        top, p13 = pairs.filter(pl.col('in_top211')), pairs.filter(pl.col('in_pool13'))
        summary.update({'docs': need.height, 'queries_hit': by_q.height, 'top_queries': by_q.head(10).to_dicts(),
                        'pools': {'top211': {'pairs': top.height, 'queries': top['query_id'].n_unique(), 'docs': top['doc_id'].n_unique()},
                                  'pool13': {'pairs': p13.height, 'queries': p13['query_id'].n_unique(), 'docs': p13['doc_id'].n_unique()},
                                  'union': {'pairs': pairs.height, 'queries': by_q.height, 'docs': need.height}},
                        'pairs': pairs.height, 'pairs_top211': top.height, 'eta': eta(pairs.height, by_q.height)})
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


SEGMENT_FILES = ('docs.parquet', 'chunks_t256.parquet', 'chunks_struct.parquet', 'provenance.parquet')


def run_batch(tar: Path, device: str, urls: Path = URLS) -> dict:
    """Phase 1 (CPU): verify -> extract -> segment (docs, provenance, t256 + structural chunks, manifest).
    device != 'defer' also runs phase 2 (embed + report); the default leaves the segment `waiting_gpu`."""
    t0 = time.time()
    batch = re.sub(r'\.tar$', '', tar.name).replace('batch-', 'b')
    tar_sha = sha256_file(tar)
    for f in STORE.glob('seg*/segment.json'):
        if seg_meta(f.parent).get('tar_sha256') == tar_sha:
            return {'status': 'already_extracted', 'segment': f.parent.name, 'state': seg_meta(f.parent)['state']}
    manifest, log, records = read_tar(tar)
    segment = next_segment(batch)
    shards = {r['doc_id']: r['shard'] for r in log if r.get('shard')}
    rows, prov, drop, srcs = build_docs(records, url_groups(urls), existing_ids(), segment, shards, tar_sha)
    stats = {'batch': batch, 'tar_sha256': tar_sha, 'urls_in_log': len(log), 'ok': manifest['ok'], 'suspect_in_log': manifest['suspect'],
             'failed_in_log': manifest['failed'], 'status_counts': dict(pl.Series([r['status'] for r in log]).value_counts().iter_rows()),
             'records': len(records), 'docs_written': len(rows), 'dropped': drop,
             'quality_rule': f'good = body >= {qa.GOOD_CP} code points OR >= {qa.GOOD_HAN} Han; else thin (kept)'}
    if not rows:
        atomic_json(RUN / 'batches' / f'{batch}.json', {**stats, 'status': 'no_docs'})
        return {**stats, 'status': 'no_docs'}
    df = pl.DataFrame(rows)
    t256, struct = chunk_segment(srcs, segment)
    seg_dir = STORE / segment
    atomic_parquet(df, seg_dir / 'docs.parquet')
    atomic_parquet(t256, seg_dir / 'chunks_t256.parquet')
    atomic_parquet(struct, seg_dir / 'chunks_struct.parquet')
    atomic_parquet(pl.DataFrame(prov), seg_dir / 'provenance.parquet')
    stats.update({'segment': segment, 'state': 'waiting_gpu', 'docs_good': int((df['quality'] == 'good').sum()),
                  'docs_thin': int((df['quality'] == 'thin').sum()), 'han_p50': int(df['han'].median()),
                  'chunks_t256': t256.height, 'chunks_struct': struct.height,
                  'struct_fields': dict(struct['field'].value_counts().iter_rows()),
                  'files_sha256': {n: sha256_file(seg_dir / n) for n in SEGMENT_FILES},
                  'extract_seconds': round(time.time() - t0, 1), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    atomic_json(seg_dir / 'manifest.json', {'schema': 'r2ai.s37.segment.v1', 'segment': segment, 'canonical_base': 'canonical-20261018 (unchanged)',
                                            'files_sha256': stats['files_sha256'], 'tar_sha256': tar_sha, 'chunks_has_vector': False})
    atomic_json(seg_dir / 'segment.json', stats)
    atomic_json(RUN / 'batches' / f'{batch}.json', {**stats, 'status': 'extracted'})
    if device != 'defer':
        stats['embed'] = embed_pending(device)
    return {**stats, 'status': 'extracted'}


def embed_chunks(seg_dir: Path, device: str) -> dict:
    """Chunk-level vectors (t256 for C-max; structural qa/answer/question/span_qa only for S38's chunk choice, never judged here;
    identical texts embedded once) -> chunk_vectors.npz."""
    out, info = {}, {}
    for kind, name in (('t256', 'chunks_t256.parquet'), ('struct', 'chunks_struct.parquet')):
        ch = pl.read_parquet(seg_dir / name, columns=['chunk_id', 'text'])
        uniq = ch['text'].unique(maintain_order=True).to_list()
        vec = embed_texts(uniq, device)
        pos = {t: i for i, t in enumerate(uniq)}
        out[f'{kind}_chunk_id'] = ch['chunk_id'].to_numpy().astype(np.int64)
        out[f'{kind}_dense'] = vec[[pos[t] for t in ch['text'].to_list()]]
        info[kind] = {'chunks': ch.height, 'unique_texts': len(uniq)}
    atomic_npz(seg_dir / 'chunk_vectors.npz', **out)
    return info


def cmax_top3(seg_dir: Path, pairs: pl.DataFrame, q_ids: np.ndarray, q: np.ndarray) -> int:
    """C-max as in S32: for every new pair of this segment the 3 t256 chunks of field body/answer with the highest cosine.
    Structural chunks (qa/answer/question/span_qa) are never used here: another chunk unit would skew the RRF scale."""
    seg_pairs = pairs.filter(pl.col('segment') == seg_dir.name)
    if not seg_pairs.height:
        return 0
    qpos = {int(x): i for i, x in enumerate(q_ids)}
    z = np.load(seg_dir / 'chunk_vectors.npz')
    rows = []
    for kind, name, fields in (('t256', 'chunks_t256.parquet', ('body', 'answer')),):
        ch = pl.read_parquet(seg_dir / name, columns=['chunk_id', 'doc_id', 'doc_ids', 'field'])
        dense = z[f'{kind}_dense'].astype(np.float32)
        by_doc = {}
        for i, (d, ids, f) in enumerate(zip(ch['doc_id'].to_list(), ch['doc_ids'].to_list(), ch['field'].to_list())):
            if f in fields:
                for x in ids:
                    by_doc.setdefault(x, []).append(i)
        for qid, did, pid in seg_pairs.select('query_id', 'doc_id', 'pair_id').iter_rows():
            idx = by_doc.get(did, [])
            if not idx:
                continue
            cos = dense[idx] @ q[qpos[qid]].astype(np.float32)
            for rank, j in enumerate(np.argsort(-cos, kind='stable')[:CMAX_TOP], 1):
                rows.append({'pair_id': pid, 'kind': kind, 'rank': rank, 'chunk_id': int(ch['chunk_id'][idx[j]]), 'chunk_cosine': float(cos[j])})
    atomic_parquet(pl.DataFrame(rows, schema={'pair_id': pl.Int64, 'kind': pl.String, 'rank': pl.Int32, 'chunk_id': pl.Int64,
                                              'chunk_cosine': pl.Float64}), seg_dir / 'cmax_top3.parquet')
    return len(rows)


def embed_pending(device: str) -> dict:
    """Phase 2: BGE-M3 doc + chunk vectors, pair report and C-max chunk picks for every segment still `waiting_gpu`.
    cuda refuses while the GPU is busy (ask Vu before taking it: S35/S38 use it)."""
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
        meta['chunk_vectors'] = embed_chunks(seg_dir, device)
        meta.update({'state': 'embedded', 'embed_device': device, 'embed_seconds': round(time.time() - t0, 1),
                     'vectors_sha256': sha256_file(seg_dir / 'vectors.npz'),
                     'chunk_vectors_sha256': sha256_file(seg_dir / 'chunk_vectors.npz')})
        atomic_json(seg_dir / 'segment.json', meta)
        done.append(seg_dir.name)
    summary = report(pool)
    q_ids, q = load_queries()
    pairs = pl.read_parquet(RUN / 'pairs_new.parquet') if summary.get('pairs') else None
    for name in done:
        meta = seg_meta(STORE / name)
        meta['report'] = summary['segments'][name]
        if pairs is not None:
            meta['cmax_rows'] = cmax_top3(STORE / name, pairs, q_ids, q)
        atomic_json(STORE / name / 'segment.json', meta)
        atomic_json(RUN / 'batches' / f"{meta['batch']}.json", {**meta, 'status': 'embedded'})
    return {'status': 'embedded', 'segments': done,
            'cumulative': {k: summary.get(k) for k in ('pairs', 'pairs_top211', 'docs', 'queries_hit', 'pools', 'eta')}}


def status() -> dict:
    segs = {d.name: {k: seg_meta(d).get(k) for k in ('state', 'batch', 'docs_written', 'docs_thin', 'chunks_t256', 'chunks_struct', 'embed_device', 'report')}
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
        out = run_batch(a.tar, a.device, a.urls)
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
