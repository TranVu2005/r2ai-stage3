"""Validate a submission file before upload.

    python -m eval.validate out/submissions/sub01_vi_k5_c2.zip [--queries data/raw/query.parquet]

Checks (any failure -> exit 1, every problem listed):
  * ZIP holds exactly 1 file (when a .zip is given); the file is a JSON list
  * exactly the query ids of query.parquet (1,200), each once, every id an int (bool / float / str rejected)
  * relevant_docs: list of int, all present in links_corpus.parquet `id`
  * relevant_chunks: list of {doc_id: int, chunk_text: str, chunk_order: int >= 0 (optional)}; chunk_text non-empty
    and a verbatim substring (no normalisation at all) of the doc text of data/docs_vi, i.e. index.chunk.doc_layout:
    title, question (when not already inside the body) and body joined by a blank line. A chunk whose doc_id is not
    in data/docs_vi is reported as an error (its text cannot be checked).
Warnings (do not fail): chunk doc_id not in that query's relevant_docs, empty relevant_docs.
"""
from __future__ import annotations

from r2ai.paths import DOCS_DIR, RAW_DATA_DIR, ROOT

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.index.chunk import doc_layout  # noqa: E402

TEXT_FIELDS = ('title', 'question', 'answer', 'body', 'paragraphs')


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check_format(raw: str, n_expected: int) -> list[str]:
    """Layout check: '[' line, one JSON object per line (trailing ',' except last), ']' line."""
    errs = []
    lines = raw.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    if len(lines) != n_expected + 2:
        errs.append(f'{len(lines)} lines, expected {n_expected + 2} (incl. "[" and "]")')
    if not lines or lines[0] != '[' or lines[-1] != ']':
        errs.append('first line must be "[" and last line must be "]"')
        return errs
    mid = lines[1:-1]
    for k, ln in enumerate(mid):
        last = k == len(mid) - 1
        if last == ln.endswith(','):
            errs.append(f'line {k + 2}: {"must not" if last else "must"} end with ","')
        try:
            o = json.loads(ln[:-1] if ln.endswith(',') else ln)
        except ValueError as e:
            errs.append(f'line {k + 2}: not valid JSON ({e})')
            continue
        if not isinstance(o, dict) or 'id' not in o:
            errs.append(f'line {k + 2}: not an object with id')
        if len(errs) > 20:
            break
    if raw.startswith('\ufeff') or '\r' in raw:
        errs.append('BOM or CR found')
    return errs


def read_text(path: Path) -> str:
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            files = [n for n in z.namelist() if not n.endswith('/')]
            return z.read(files[0]).decode('utf-8') if files else ''
    return path.read_text(encoding='utf-8', newline='')


def load_raw(path: Path) -> tuple[object, list[str]]:
    errs = []
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != 1:
                errs.append(f'ZIP must contain exactly 1 file, found {len(names)}: {names[:5]}')
            files = [n for n in names if not n.endswith('/')]
            if not files:
                return None, errs + ['ZIP has no file']
            raw = z.read(files[0]).decode('utf-8')
    else:
        raw = path.read_text(encoding='utf-8')
    return json.loads(raw), errs


def doc_texts(doc_ids: set[int], docs_dir: Path) -> dict[int, str]:
    """doc_id -> doc text exactly as chunked (index.chunk.doc_layout), for the requested ids only."""
    out: dict[int, str] = {}
    for f in sorted(glob.glob(str(docs_dir / '*.parquet'))):
        t = pq.read_table(f, columns=['doc_ids', *TEXT_FIELDS])
        for r in t.to_pylist():
            hit = [d for d in r['doc_ids'] if d in doc_ids]
            if hit:
                text = doc_layout({**r, 'body': r['body'] or '', 'paragraphs': r['paragraphs'] or []})[0]
                for d in hit:
                    out[d] = text
    return out


def validate(sub, query_ids: list[int], corpus_ids: np.ndarray, docs_dir: Path) -> tuple[list[str], list[str], dict]:
    errs, warns = [], []
    if not isinstance(sub, list):
        return [f'top level must be a JSON list, got {type(sub).__name__}'], warns, {}
    seen = []
    chunk_refs = []
    all_docs = []
    for k, r in enumerate(sub):
        if not isinstance(r, dict):
            errs.append(f'row {k}: not an object')
            continue
        qid = r.get('id')
        if not _is_int(qid):
            errs.append(f'row {k}: id {qid!r} is not an int')
            continue
        seen.append(qid)
        docs = r.get('relevant_docs')
        if not isinstance(docs, list) or not all(_is_int(d) for d in docs):
            errs.append(f'id {qid}: relevant_docs must be a list of int')
            docs = [d for d in docs if _is_int(d)] if isinstance(docs, list) else []
        if not docs:
            warns.append(f'id {qid}: empty relevant_docs')
        all_docs += docs
        chunks = r.get('relevant_chunks')
        if not isinstance(chunks, list):
            errs.append(f'id {qid}: relevant_chunks must be a list')
            continue
        for j, c in enumerate(chunks):
            if not isinstance(c, dict) or not _is_int(c.get('doc_id')) or not isinstance(c.get('chunk_text'), str):
                errs.append(f'id {qid} chunk {j}: needs int doc_id and str chunk_text')
                continue
            order = c.get('chunk_order')
            if 'chunk_order' in c and (not _is_int(order) or order < 0):
                errs.append(f'id {qid} chunk {j}: chunk_order must be an int >= 0, got {order!r}')
                continue
            if not c['chunk_text'].strip():
                errs.append(f'id {qid} chunk {j}: empty chunk_text')
                continue
            if c['doc_id'] not in set(docs):
                warns.append(f'id {qid} chunk {j}: doc_id {c["doc_id"]} not in relevant_docs')
            chunk_refs.append((qid, j, c['doc_id'], c['chunk_text']))
    dup = len(seen) - len(set(seen))
    if dup:
        errs.append(f'{dup} duplicated id(s)')
    missing, extra = set(query_ids) - set(seen), set(seen) - set(query_ids)
    if missing:
        errs.append(f'{len(missing)} query id(s) missing, e.g. {sorted(missing)[:5]}')
    if extra:
        errs.append(f'{len(extra)} id(s) not in query.parquet, e.g. {sorted(extra)[:5]}')
    if len(sub) != len(query_ids):
        errs.append(f'{len(sub)} rows, expected {len(query_ids)}')
    if all_docs:
        arr = np.array(all_docs, dtype=np.int64)
        bad = arr[~np.isin(arr, corpus_ids)]
        if len(bad):
            errs.append(f'{len(bad)} relevant_docs id(s) not in links_corpus, e.g. {sorted(set(bad.tolist()))[:5]}')
    texts = doc_texts({d for _, _, d, _ in chunk_refs}, docs_dir)
    n_bad = 0
    for qid, j, d, t in chunk_refs:
        if d not in texts:
            errs.append(f'id {qid} chunk {j}: doc_id {d} not in {docs_dir}')
            n_bad += 1
        elif t not in texts[d]:
            errs.append(f'id {qid} chunk {j}: chunk_text is not a substring of doc {d}')
            n_bad += 1
    stats = {'rows': len(sub), 'relevant_docs_total': len(all_docs), 'chunks_total': len(chunk_refs),
             'chunks_bad': n_bad, 'warnings': len(warns)}
    return errs, warns, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--corpus', default=str(RAW_DATA_DIR / 'links_corpus.parquet'))
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    a = ap.parse_args(argv)
    sub, errs = load_raw(Path(a.path))
    raw = read_text(Path(a.path))
    qids = [int(x) for x in pq.read_table(a.queries, columns=['id'])['id'].to_pylist()]
    errs += check_format(raw, len(qids))
    corpus = pq.read_table(a.corpus, columns=['id'])['id'].to_numpy()
    e2, warns, stats = validate(sub, qids, corpus, Path(a.docs_dir))
    errs += e2
    for w in warns[:20]:
        print('WARN ', w)
    if len(warns) > 20:
        print(f'WARN  ... {len(warns) - 20} more')
    for e in errs[:50]:
        print('ERROR', e)
    if len(errs) > 50:
        print(f'ERROR ... {len(errs) - 50} more')
    print(json.dumps({**stats, 'errors': len(errs), 'ok': not errs}))
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
