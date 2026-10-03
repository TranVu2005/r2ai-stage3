"""Validate a submission zip/json. Any failure -> exit 1, problems listed.

    python -m scripts.validate_submission out/submissions/sub03_vi_k5_full.zip [--skip-near] [--skip-verbatim]

  * zip: exactly 1 entry, a .json, no sub-directory; layout (eval.validate.check_format)
  * schema: list of {id:int, relevant_docs:[int], relevant_chunks:[{doc_id:int, chunk_text:str}]}, no other keys
  * ids: exactly the 1,200 ids of query.parquet, each once
  * every relevant_docs id in links_corpus.parquet
  * every chunk.doc_id in the same row's relevant_docs; no empty / whitespace-only chunk_text
  * (--dedupe-scope doc, default: same doc_id only; query: whole query) no two chunks equal after normalisation (NFKC, html.unescape, lowercase, whitespace); unless
    --skip-near also none with LCS_tokens / len(shorter) >= 0.8 (scripts.make_submission.dedupe)
  * unless --skip-verbatim: chunk_text is a verbatim substring of the doc text (eval.validate)
"""
from __future__ import annotations

from r2ai.paths import DOCS_DIR, RAW_DATA_DIR, require_inputs, resolve_path

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


from r2ai.eval.scorer import Tokenizer, normalize_metric  # noqa: E402
from r2ai.eval.validate import check_format, doc_texts  # noqa: E402
from r2ai.submit.make_submission import dedupe  # noqa: E402


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--corpus', default=str(RAW_DATA_DIR / 'links_corpus.parquet'))
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--dedupe-scope', choices=('doc', 'query'), default='doc')
    ap.add_argument('--max-zip-mib', type=float, default=0, help='fail if the file is larger (0 = off)')
    ap.add_argument('--skip-near', action='store_true')
    ap.add_argument('--skip-verbatim', action='store_true')
    a = ap.parse_args(argv)
    a.path, a.queries, a.corpus, a.docs_dir = map(lambda x: str(resolve_path(x)), (a.path,a.queries,a.corpus,a.docs_dir))
    require_inputs(a.path, a.queries, a.corpus)
    if not a.skip_verbatim:
        require_inputs(a.docs_dir)
        if not any(Path(a.docs_dir).glob('*.parquet')):
            raise ValueError(f'Missing or empty parquet input: {a.docs_dir}')
    errs: list[str] = []
    p = Path(a.path)
    if a.max_zip_mib and p.stat().st_size / 2**20 > a.max_zip_mib:
        errs.append(f'{p.stat().st_size / 2**20:.2f} MiB exceeds --max-zip-mib {a.max_zip_mib}')
    if p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if len(names) != 1 or names[0].endswith('/') or '/' in names[0] or '\\' in names[0] or not names[0].endswith('.json'):
                errs.append(f'zip must hold exactly one top-level .json, got {names[:5]}')
            raw = z.read(names[0]).decode('utf-8') if names else ''
    else:
        with p.open(encoding='utf-8', newline='') as f:
            raw = f.read()
    try:
        sub = json.loads(raw)
    except ValueError as e:
        print(f'ERROR not valid JSON: {e}')
        return 1
    qids = [int(x) for x in pq.read_table(a.queries, columns=['id'])['id'].to_pylist()]
    errs += check_format(raw, len(qids))
    if not isinstance(sub, list):
        print('ERROR top level must be a list')
        return 1
    corpus = pq.read_table(a.corpus, columns=['id'])['id'].to_numpy()

    seen, refs, tok = [], [], Tokenizer()
    n_near = 0
    for k, r in enumerate(sub):
        if not isinstance(r, dict) or set(r) != {'id', 'relevant_docs', 'relevant_chunks'}:
            errs.append(f'row {k}: must be an object with exactly id, relevant_docs, relevant_chunks')
            continue
        qid = r['id']
        if not _is_int(qid):
            errs.append(f'row {k}: id {qid!r} is not an int')
            continue
        seen.append(qid)
        docs, chunks = r['relevant_docs'], r['relevant_chunks']
        if not isinstance(docs, list) or not all(_is_int(d) for d in docs):
            errs.append(f'id {qid}: relevant_docs must be a list of int')
            continue
        if len(set(docs)) != len(docs):
            errs.append(f'id {qid}: duplicated ids in relevant_docs')
        bad = np.array(docs, dtype=np.int64)[~np.isin(np.array(docs, dtype=np.int64), corpus)] if docs else []
        if len(bad):
            errs.append(f'id {qid}: {len(bad)} relevant_docs not in links_corpus, e.g. {bad[:3].tolist()}')
        if not isinstance(chunks, list):
            errs.append(f'id {qid}: relevant_chunks must be a list')
            continue
        dset, ok_texts, keys = set(docs), [], {}
        for j, c in enumerate(chunks):
            if not isinstance(c, dict) or set(c) != {'doc_id', 'chunk_text'} or not _is_int(c['doc_id']) or not isinstance(c['chunk_text'], str):
                errs.append(f'id {qid} chunk {j}: must be {{doc_id:int, chunk_text:str}} and nothing else')
                continue
            if not c['chunk_text'].strip():
                errs.append(f'id {qid} chunk {j}: empty chunk_text')
                continue
            if c['doc_id'] not in dset:
                errs.append(f'id {qid} chunk {j}: doc_id {c["doc_id"]} not in relevant_docs')
            key = normalize_metric(c['chunk_text']) if a.dedupe_scope == 'query' else (c['doc_id'], normalize_metric(c['chunk_text']))
            if key in keys:
                errs.append(f'id {qid} chunk {j}: equal to chunk {keys[key]} after normalisation')
                continue
            keys[key] = j
            ok_texts.append(c['chunk_text'])
            refs.append((qid, j, c['doc_id'], c['chunk_text']))
        if not a.skip_near and len(ok_texts) > 1:
            nn = 0
            if a.dedupe_scope == 'query':
                nn = dedupe(ok_texts, tok)[2]
            else:
                byd = {}
                for (_, _, d, t) in refs[len(refs) - len(ok_texts):]:
                    byd.setdefault(d, []).append(t)
                nn = sum(dedupe(v, tok)[2] for v in byd.values() if len(v) > 1)
            if nn:
                n_near += nn
                errs.append(f'id {qid}: {nn} chunk(s) near-duplicate (LCS/shorter >= 0.8) of an earlier chunk')
    if len(seen) != len(set(seen)):
        errs.append(f'{len(seen) - len(set(seen))} duplicated id(s)')
    if set(seen) != set(qids) or len(sub) != len(qids):
        errs.append(f'ids differ from query.parquet: missing {len(set(qids) - set(seen))}, extra {len(set(seen) - set(qids))}, rows {len(sub)}')
    if not a.skip_verbatim:
        texts = doc_texts({d for _, _, d, _ in refs}, Path(a.docs_dir))
        for qid, j, d, t in refs:
            if d not in texts:
                errs.append(f'id {qid} chunk {j}: doc_id {d} not in docs_vi, cannot check text')
            elif t not in texts[d]:
                errs.append(f'id {qid} chunk {j}: chunk_text is not a verbatim substring of doc {d}')
    for e in errs[:50]:
        print('ERROR', e)
    if len(errs) > 50:
        print(f'ERROR ... {len(errs) - 50} more')
    print(json.dumps({'rows': len(sub), 'chunks': len(refs), 'errors': len(errs), 'ok': not errs}))
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
