"""S25 phase 2 step 5: verify the canonical corpus (read-only on the canonical dir; reports go to --out).

    python -m r2ai.canonical.verify --canonical data/canonical-20261018 --zip P1=...P1.zip --zip P2=...P2.zip

Checks: (1) every doc_id is in corpus v2, lang vi/zh, exactly one row; (2) every chunk is verbatim in the canonical
doc_text of every doc_id it lists, and chunk IDs are unique; (3) vectors row count equals has_vector chunks and a
sample of rows equals the source index rows; (4) every relevant_doc and relevant_chunk of each ZIP is found in
canonical: chunk_text equal to the canonical doc_text, to a canonical chunk text, or (reported apart) a substring of
doc_text; everything else is listed; (5) coverage by (lang, host) vs the S20 audit CSV.
"""
from __future__ import annotations

from r2ai.paths import resolve_path

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from r2ai.canonical import select as S
from r2ai.canonical.guard import RUN_ROOT, guarded, write_parquet
from r2ai.compose.base_io import baseline

S20_CSV = Path('D:/GitHub/r2ai-stage3/.superpowers/worktrees/s21-120ask/reports/s20-coverage-audit.csv')
H = lambda s: hashlib.sha256(s.encode('utf-8')).digest()


def id_ranges(ids: np.ndarray, parts: int):
    qs = np.unique(np.quantile(ids, np.linspace(0, 1, parts + 1)).astype(np.int64))
    qs[-1] += 1
    return list(zip(qs[:-1], qs[1:]))


def check_docs(canon: Path, corpus: pl.DataFrame) -> dict:
    d = pl.read_parquet(canon / 'docs.parquet', columns=['doc_id', 'lang', 'quality', 'rule'])
    v2 = corpus.select(pl.col('id').alias('doc_id'), pl.col('lang').alias('v2_lang'))
    j = d.join(v2, on='doc_id', how='left')
    return {'docs': d.height, 'unique_doc_ids': d['doc_id'].n_unique(),
            'not_in_v2': int(j['v2_lang'].is_null().sum()), 'lang_mismatch': int((j['lang'] != j['v2_lang']).sum()),
            'en_rows': int((d['lang'] == 'en').sum()), 'pass': d.height == d['doc_id'].n_unique()
            and int(j['v2_lang'].is_null().sum()) == 0 and int((j['lang'] != j['v2_lang']).sum()) == 0}


def check_chunks(canon: Path, log) -> dict:
    docs = ds.dataset(canon / 'docs.parquet', format='parquet')
    chunks = ds.dataset(canon / 'chunks_t256.parquet', format='parquet')
    cid = chunks.to_table(columns=['chunk_id'])['chunk_id'].to_numpy()
    first = chunks.to_table(columns=['doc_id'])['doc_id'].to_numpy()
    total = bad = ids_checked = 0
    missing_doc = 0
    for lo, hi in id_ranges(first, 24):
        t = chunks.to_table(columns=['doc_id', 'doc_ids', 'char_start', 'char_end', 'text'],
                            filter=(ds.field('doc_id') >= int(lo)) & (ds.field('doc_id') < int(hi)))
        need = np.unique(np.concatenate([np.asarray(x, np.int64) for x in t['doc_ids'].to_pylist()])) if t.num_rows else []
        dt = docs.to_table(columns=['doc_id', 'doc_text'], filter=ds.field('doc_id').isin(pa.array(need, pa.int64())))
        text = dict(zip(dt['doc_id'].to_pylist(), dt['doc_text'].to_pylist()))
        for ids, s, e, tx in zip(t['doc_ids'].to_pylist(), t['char_start'].to_pylist(), t['char_end'].to_pylist(),
                                 t['text'].to_pylist()):
            total += 1
            for i in ids:
                ids_checked += 1
                if i not in text:
                    missing_doc += 1
                    bad += 1
                elif text[i][s:e] != tx:
                    bad += 1
        log(f'verbatim {total}/{len(first)}')
    return {'chunks': int(len(cid)), 'unique_chunk_ids': int(len(np.unique(cid))), 'doc_refs_checked': ids_checked,
            'not_verbatim_or_missing_doc': bad, 'chunk_doc_not_in_docs': missing_doc,
            'pass': bad == 0 and len(np.unique(cid)) == len(cid)}


def check_vectors(canon: Path, sample: int = 2000) -> dict:
    vj = json.loads((canon / 'vectors/vectors.json').read_text(encoding='utf-8'))
    has = pq.read_table(canon / 'chunks_t256.parquet', columns=['has_vector'])['has_vector'].to_numpy()
    dense = np.load(canon / 'vectors/dense.npy', mmap_mode='r')
    src_row = np.fromfile(canon / 'vectors/source_row.bin', dtype=np.int64)
    rng = np.random.default_rng(25)
    pos = np.sort(rng.choice(len(dense), size=min(sample, len(dense)), replace=False)) if len(dense) else np.zeros(0, int)
    bounds = np.cumsum([0] + [s['rows'] for s in vj['segments']])
    eq = 0
    for p in pos:
        k = int(np.searchsorted(bounds, p, side='right') - 1)
        idx = Path(vj['segments'][k]['index'])
        src = np.load(idx / 'dense.npy', mmap_mode='r')
        eq += int(np.array_equal(src[src_row[p]], dense[p]))
    return {'rows': int(len(dense)), 'has_vector_chunks': int(has.sum()), 'sampled': int(len(pos)), 'sample_equal': eq,
            'pass': len(dense) == int(has.sum()) == vj['rows'] and eq == len(pos)}


def check_zip(name: str, zip_path: Path, canon: Path, out: Path, log) -> dict:
    refs_docs, refs_chunks = set(), set()
    for row, _ in baseline(zip_path):
        refs_docs.update(int(d) for d in row['relevant_docs'])
        refs_chunks.update(int(c['doc_id']) for c in row['relevant_chunks'])
    need = pa.array(sorted(refs_docs | refs_chunks), pa.int64())
    dt = ds.dataset(canon / 'docs.parquet', format='parquet').to_table(columns=['doc_id', 'doc_text', 'lang', 'rule'],
                                                                      filter=ds.field('doc_id').isin(need))
    doc_text = dict(zip(dt['doc_id'].to_pylist(), dt['doc_text'].to_pylist()))
    ct = ds.dataset(canon / 'chunks_t256.parquet', format='parquet').to_table(
        columns=['doc_ids', 'text', 'field', 'char_start', 'char_end'],
        filter=ds.field('doc_id').isin(pa.array(sorted(refs_chunks), pa.int64())))
    chunk_sha, tail_start, ans_span = defaultdict(set), {}, {}
    for ids, tx, fld, st, en in zip(ct['doc_ids'].to_pylist(), ct['text'].to_pylist(), ct['field'].to_pylist(),
                                    ct['char_start'].to_pylist(), ct['char_end'].to_pylist()):
        for i in ids:
            chunk_sha[i].add(H(tx))
            if fld in ('answer', 'body'):
                tail_start[i] = min(st, tail_start.get(i, st))
            if fld == 'answer':
                a, b = ans_span.get(i, (st, en))
                ans_span[i] = (min(a, st), max(b, en))
    # chunks are keyed by physical doc_id; alias ids are covered through doc_ids, so also index by all listed ids
    lang = dict(zip(dt['doc_id'].to_pylist(), dt['lang'].to_pylist()))
    cnt, issues = Counter(), []
    for row, _ in baseline(zip_path):
        q = int(row['id'])
        for d in row['relevant_docs']:
            d = int(d)
            if d in doc_text:
                cnt['doc_found'] += 1
            else:
                cnt['doc_missing'] += 1
                issues.append({'zip': name, 'query_id': q, 'doc_id': d, 'kind': 'doc_missing', 'detail': ''})
        for c in row['relevant_chunks']:
            d, tx = int(c['doc_id']), c['chunk_text']
            if d not in doc_text:
                cnt['chunk_doc_missing'] += 1
                issues.append({'zip': name, 'query_id': q, 'doc_id': d, 'kind': 'chunk_doc_missing', 'detail': ''})
            elif tx == doc_text[d]:
                cnt['chunk_eq_doc_text'] += 1
            elif H(tx) in chunk_sha.get(d, ()):
                cnt['chunk_eq_canonical_chunk'] += 1
            elif d in ans_span and tx == doc_text[d][ans_span[d][0]:ans_span[d][1]]:
                # make_submission.load_full_texts "full text": the answer span when the page has one ...
                cnt['chunk_eq_answer_span'] += 1
            elif d in tail_start and tx == doc_text[d][tail_start[d]:]:
                # ... otherwise doc_text from its first body span (the whole body)
                cnt['chunk_eq_body_span'] += 1
            elif tx in doc_text[d]:
                cnt['chunk_substring_of_doc_text'] += 1
                issues.append({'zip': name, 'query_id': q, 'doc_id': d, 'kind': 'chunk_substring_only',
                               'detail': f'len={len(tx)}'})
            else:
                cnt['chunk_text_mismatch'] += 1
                issues.append({'zip': name, 'query_id': q, 'doc_id': d, 'kind': 'chunk_text_mismatch',
                               'detail': f'zip_len={len(tx)} canon_len={len(doc_text[d])} lang={lang.get(d)}'})
    log(f'{name}: {dict(cnt)}')
    return {'zip': str(zip_path), 'zip_sha256': sha_file(zip_path), 'counts': dict(cnt),
            'distinct_docs': len(refs_docs), 'distinct_chunk_docs': len(refs_chunks),
            'pass': cnt['doc_missing'] == cnt['chunk_doc_missing'] == cnt['chunk_text_mismatch']
            == cnt['chunk_substring_of_doc_text'] == 0}, issues


def sha_file(p: Path) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def coverage(canon: Path, corpus: pl.DataFrame) -> pl.DataFrame:
    d = pl.read_parquet(canon / 'docs.parquet', columns=['doc_id', 'quality'])
    j = corpus.select(pl.col('id').alias('doc_id'), 'lang', 'host').join(d, on='doc_id', how='left')
    cov = (j.group_by('lang', 'host').agg(pl.len().alias('v2_urls'),
                                          (pl.col('quality') == 'good').sum().alias('good'),
                                          (pl.col('quality') == 'thin').sum().alias('thin'),
                                          pl.col('quality').is_null().sum().alias('missing'))
           .sort('lang', 'host'))
    if S20_CSV.is_file():
        s20 = pl.read_csv(S20_CSV, columns=['lang', 'host', 'v2_urls', 'good_text_ge200', 'empty_or_extract_error',
                                            'missing_completely', 'awaiting_s3_s11'], infer_schema_length=0)
        s20 = s20.select('lang', 'host', *[pl.col(c).cast(pl.Int64).alias(f's20_{c}') for c in
                                           ('v2_urls', 'good_text_ge200', 'empty_or_extract_error',
                                            'missing_completely', 'awaiting_s3_s11')])
        cov = cov.join(s20, on=['lang', 'host'], how='left').with_columns(
            (pl.col('good') - pl.col('s20_good_text_ge200')).alias('delta_good_vs_s20'))
    return cov


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--canonical', default=str(S.DATA / 'canonical-20261018'))
    ap.add_argument('--out', default=str(RUN_ROOT / 'verify'))
    ap.add_argument('--zip', action='append', default=[], help='NAME=path, repeatable')
    ap.add_argument('--skip', default='', help='comma list of: docs,chunks,vectors,coverage')
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)
    canon, out = resolve_path(a.canonical), guarded(a.out)
    out.mkdir(parents=True)
    skip = set(filter(None, a.skip.split(',')))
    corpus = pl.read_parquet(S.DATA / 'corpus_v2_ids.parquet')
    report = {'canonical': str(canon), 'manifest_sha256': sha_file(canon / 'manifest.json')}
    if 'docs' not in skip:
        report['docs'] = check_docs(canon, corpus)
        log(f"docs {report['docs']}")
    if 'chunks' not in skip:
        report['chunks'] = check_chunks(canon, log)
        log(f"chunks {report['chunks']}")
    if 'vectors' not in skip and (canon / 'vectors/vectors.json').is_file():
        report['vectors'] = check_vectors(canon)
        log(f"vectors {report['vectors']}")
    issues = []
    report['zips'] = {}
    for spec in a.zip:
        name, path = spec.split('=', 1)
        r, iss = check_zip(name, resolve_path(path), canon, out, log)
        report['zips'][name] = r
        issues += iss
    write_parquet(pa.Table.from_pylist(issues) if issues else
                  pa.table({'zip': pa.array([], pa.string())}), out / 'zip-mismatches.parquet')
    if 'coverage' not in skip:
        cov = coverage(canon, corpus)
        write_parquet(cov.to_arrow(), out / 'coverage.parquet')
        cov.write_csv(guarded(out / 'coverage.csv'))
        report['coverage_totals'] = cov.group_by('lang').agg(pl.sum('v2_urls'), pl.sum('good'), pl.sum('thin'),
                                                             pl.sum('missing')).sort('lang').to_dicts()
    guarded(out / 'verify-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1:])
