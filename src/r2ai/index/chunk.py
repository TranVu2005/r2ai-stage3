"""Paragraph-preserving chunking of data/docs_vi.

    python -m index.chunk [--targets 128,256,400] [--min-body-tokens 50] [--out-dir data/chunks]

Input: data/docs_vi/*.parquet, rows with status == "ok" and n_tokens_bge_m3 >= --min-body-tokens.
extract.py already merges http/https / duplicate URL variants by url_norm: one row = one text, `doc_ids` = the group.
doc_id = doc_ids[0], doc_ids_group = doc_ids.

doc_text = "\\n\\n".join(non-empty [title, question (only when not already inside body), body]).
Every chunk satisfies doc_text[char_start:char_end] == text (asserted for 100 % of chunks).

Chunks per doc:
  * field "title":    the title, one chunk (for query-question matching, never submitted as chunk_text)
  * field "question": the reader question of Q&A pages (same use, never submitted)
  * field "answer" / "body": body paragraphs, grouped per field segment (a Q&A body is split into the question span,
    the answer span and the rest; paragraphs are labelled by the span they overlap).
Grouping: consecutive paragraphs are added until the next one would exceed target_tokens; the next chunk starts
with the last paragraph of the previous chunk (1-paragraph overlap) when that chunk had >= 2 paragraphs.
A paragraph > 1.5 * target is split at sentence ends into pieces <= target tokens; a single sentence that is still
> 1.5 * target is split at whitespace (the only case where a sentence is cut). Pieces of a split paragraph do not
overlap. Whitespace-free pieces > 1.5 * target tokens (base64 / inline-image junk) are dropped. Token counts use the BAAI/bge-m3 tokenizer without special tokens, on the raw text.

Outputs:  data/chunks/docs.parquet (doc_id, doc_ids_group, domain, url, title, doc_text)
          data/chunks/chunks_t{target}.parquet (chunk_id, doc_id, doc_ids_group, field, char_start, char_end, text, n_tokens)

Memory: docs are bucketed by doc_id range into a temporary directory inside --out-dir, then processed one bucket
(--batch-docs docs) at a time in doc_id order, so peak RAM does not grow with the corpus. Output rows and order are
the same as processing all docs at once; outputs are written as *.partial and renamed only after every target is done.
"""
from __future__ import annotations

from r2ai.paths import CHUNKS_DIR, DOCS_DIR, assert_writable, require_inputs, resolve_path

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SUBMITTABLE = ('answer', 'body')
SENT_END = re.compile(r'(?<=[.!?…;:])\s+')
DOC_COLS = ['doc_ids', 'url', 'domain', 'title', 'question', 'answer', 'body', 'paragraphs', 'status', 'n_tokens_bge_m3']
TOKENIZE_BATCH = 20000
FLUSH_ROWS = 5000
DOCS_SCHEMA = pa.schema([('doc_id', pa.int64()), ('doc_ids_group', pa.list_(pa.int64())), ('domain', pa.string()),
                         ('url', pa.string()), ('title', pa.string()), ('doc_text', pa.string())])
CHUNKS_SCHEMA = pa.schema([('chunk_id', pa.int64()), ('doc_id', pa.int64()), ('doc_ids_group', pa.list_(pa.int64())),
                           ('field', pa.string()), ('char_start', pa.int32()), ('char_end', pa.int32()),
                           ('text', pa.string()), ('n_tokens', pa.int32())])


def load_docs(docs_dir: Path, min_body_tokens: int) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(docs_dir / '*.parquet'))):
        for r in pq.read_table(f, columns=DOC_COLS).to_pylist():
            if r['status'] == 'ok' and (r['n_tokens_bge_m3'] or 0) >= min_body_tokens and r['body']:
                rows.append(r)
    rows.sort(key=lambda r: r['doc_ids'][0])
    return rows


def _keep_mask(t: pa.Table, min_body_tokens: int) -> pa.Array:
    """Arrow form of load_docs' filter: status == 'ok' and (n_tokens_bge_m3 or 0) >= min and body non-empty."""
    ok = pc.equal(t['status'], 'ok')
    long_enough = pc.greater_equal(pc.fill_null(t['n_tokens_bge_m3'], 0), min_body_tokens)
    has_body = pc.greater(pc.utf8_length(t['body']), 0)
    return pc.fill_null(pc.and_(pc.and_(ok, long_enough), has_body), False)


def _first_ids(t: pa.Table) -> np.ndarray:
    return pc.list_element(t['doc_ids'], 0).to_numpy(zero_copy_only=False)


def _bucket_docs(files: list[str], min_body_tokens: int, batch_docs: int, tmp_dir: Path) -> tuple[list[Path], int]:
    """Write the kept rows of every file into doc_id-range buckets of about batch_docs rows -> (bucket files, rows)."""
    schema = pa.unify_schemas([pq.read_schema(f) for f in files], promote_options='permissive')
    schema = pa.schema([schema.field(c) for c in DOC_COLS])
    ids = []
    for f in files:
        t = pq.read_table(f, columns=['doc_ids', 'status', 'n_tokens_bge_m3', 'body'])
        ids.append(_first_ids(t.filter(_keep_mask(t, min_body_tokens))))
    ids = np.sort(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)
    if not len(ids):
        return [], 0
    bounds = ids[batch_docs::batch_docs]                    # bucket b holds bounds[b-1] <= doc_id < bounds[b]
    tmp_dir.mkdir(parents=True)
    paths = [tmp_dir / f'bucket_{b:05d}.parquet' for b in range(len(bounds) + 1)]
    writers: dict[int, pq.ParquetWriter] = {}
    pending: dict[int, list[pa.Table]] = {}

    def flush(b: int) -> None:
        if b not in writers:
            writers[b] = pq.ParquetWriter(paths[b], schema)
        writers[b].write_table(pa.concat_tables(pending.pop(b)))

    try:
        for f in files:
            t = pq.read_table(f, columns=DOC_COLS)
            t = t.filter(_keep_mask(t, min_body_tokens)).select(DOC_COLS).cast(schema)
            if not t.num_rows:
                continue
            bucket = np.searchsorted(bounds, _first_ids(t), side='right')
            for b in np.unique(bucket):
                pending.setdefault(int(b), []).append(t.filter(pa.array(bucket == b)))
                if sum(x.num_rows for x in pending[int(b)]) >= FLUSH_ROWS:
                    flush(int(b))
        for b in list(pending):
            flush(b)
    finally:
        for w in writers.values():
            w.close()
    return [p for p in paths if p.exists()], len(ids)


def _token_counts(tok, texts: list[str]) -> list[int]:
    out = []
    for b in range(0, len(texts), TOKENIZE_BATCH):
        out += [len(x) for x in tok(texts[b:b + TOKENIZE_BATCH], add_special_tokens=False)['input_ids']]
    return out


def doc_layout(r: dict):
    """-> doc_text, list of (field, start, end) spans in doc_text: title/question singles + body paragraphs."""
    title = (r['title'] or '').strip()
    q = (r['question'] or '').strip()
    body = r['body']
    parts, spans = [], []
    pos = 0

    def add(text):
        nonlocal pos
        if parts:
            pos += 2
        start = pos
        parts.append(text)
        pos += len(text)
        return start

    if title:
        s = add(title)
        spans.append(('title', s, s + len(title)))
    q_in_body = bool(q) and q in body
    if q and not q_in_body:
        s = add(q)
        spans.append(('question', s, s + len(q)))
    b0 = add(body)
    doc_text = '\n\n'.join(parts)
    # field segments inside body
    seg = []
    a = (r['answer'] or '').strip()
    if a and a in body:
        i = body.find(a)
        seg.append(('answer', i, i + len(a)))
    if q_in_body:
        i = body.find(q)
        seg.append(('question', i, i + len(q)))
    paras = []
    off = 0
    for p in r['paragraphs']:
        i = body.find(p, off)
        assert i >= 0, 'paragraph not found in body'
        field = 'body'
        for f, s, e in seg:
            if i < e and i + len(p) > s:
                field = f
                break
        paras.append((field, b0 + i, b0 + i + len(p)))
        off = i + len(p)
    return doc_text, spans, paras


def _split_long(text: str, start: int, ntok, target: int) -> list[tuple[int, int]]:
    """Split one over-long paragraph at sentence ends (or whitespace for over-long sentences) into <= target pieces."""
    units = []                                # (s, e) char spans relative to text
    last = 0
    for m in SENT_END.finditer(text):
        units.append((last, m.start()))
        last = m.end()
    units.append((last, len(text)))
    fine = []
    for s, e in units:
        if ntok(text[s:e]) > 1.5 * target:    # cut inside the sentence, at whitespace
            words = [(m.start() + s, m.end() + s) for m in re.finditer(r'\S+', text[s:e])]
            cur = None
            for ws, we in words:
                if cur is None:
                    cur = [ws, we]
                elif ntok(text[cur[0]:we]) > target:
                    fine.append(tuple(cur))
                    cur = [ws, we]
                else:
                    cur[1] = we
            if cur:
                fine.append(tuple(cur))
        else:
            fine.append((s, e))
    pieces, cur = [], None
    for s, e in fine:
        if cur is None:
            cur = [s, e]
        elif ntok(text[cur[0]:e]) > target:
            pieces.append(tuple(cur))
            cur = [s, e]
        else:
            cur[1] = e
    if cur:
        pieces.append(tuple(cur))
    return [(start + s, start + e) for s, e in pieces]


def chunk_paragraphs(doc_text: str, paras: list[tuple[str, int, int]], ptoks: list[int], target: int, ntok):
    """-> list of (field, start, end). Groups never cross a field change."""
    out = []
    i, n = 0, len(paras)
    while i < n:
        field = paras[i][0]
        j = i
        while j < n and paras[j][0] == field:
            j += 1
        k = i                                  # segment [i, j)
        while k < j:
            f, s, e = paras[k]
            if ptoks[k] > 1.5 * target:
                out += [(f, a, b) for a, b in _split_long(doc_text[s:e], s, ntok, target)]
                k += 1
                continue
            tot, m = ptoks[k], k + 1
            while m < j and ptoks[m] <= 1.5 * target and tot + ptoks[m] <= target:
                tot += ptoks[m]
                m += 1
            out.append((f, s, paras[m - 1][2]))
            if m >= j:
                break
            # 1-paragraph overlap, unless the overlap paragraph cannot share a chunk with the next one
            # (that chunk would be a strict subset of the previous one)
            overlap = m - k >= 2 and ptoks[m] <= 1.5 * target and ptoks[m - 1] + ptoks[m] <= target
            k = m - 1 if overlap else m
        i = j
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--out-dir', default=str(CHUNKS_DIR))
    ap.add_argument('--targets', default='128,256,400')
    ap.add_argument('--min-body-tokens', type=int, default=50)
    ap.add_argument('--batch-docs', type=int, default=20000, help='docs held in memory at once (memory only, same output)')
    a = ap.parse_args(argv)
    if a.batch_docs < 1:
        ap.error('--batch-docs must be >= 1')
    a.docs_dir = str(resolve_path(a.docs_dir))
    a.out_dir = str(assert_writable(a.out_dir))
    require_inputs(a.docs_dir)
    files = sorted(glob.glob(str(Path(a.docs_dir) / '*.parquet')))
    if not files:
        raise ValueError(f'Missing or empty parquet input: {a.docs_dir}')
    from r2ai.gold_check.gold_check_common import load_tokenizer
    tok = load_tokenizer()
    targets = [int(x) for x in a.targets.split(',')]
    out_dir = Path(a.out_dir)
    tmp_dir = out_dir / '.chunk_tmp'
    t0 = time.time()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)                                  # leftover of an interrupted run
    try:
        buckets, n_docs = _bucket_docs(files, a.min_body_tokens, a.batch_docs, tmp_dir)
        if not n_docs:
            raise ValueError(f'Missing or empty documents after filtering: {a.docs_dir}')
        print(f'{n_docs} docs bucketed into {len(buckets)} batch(es) ({time.time() - t0:.0f}s)', flush=True)
        report = _chunk_buckets(buckets, tok, targets, out_dir, t0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    report = {'docs': report['docs'], 'paragraphs': report['paragraphs'], 'min_body_tokens': a.min_body_tokens,
              'configs': report['configs']}
    (out_dir / 'chunk_report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0


def _chunk_buckets(buckets: list[Path], tok, targets: list[int], out_dir: Path, t0: float) -> dict:
    """Chunk bucket after bucket in doc_id order; same rows, chunk_ids and report as one pass over all docs."""
    final = {'docs': out_dir / 'docs.parquet', **{t: out_dir / f'chunks_t{t}.parquet' for t in targets}}
    partial = {k: p.with_name(p.name + '.partial') for k, p in final.items()}
    writers = {'docs': pq.ParquetWriter(partial['docs'], DOCS_SCHEMA, compression='zstd')}
    writers.update({t: pq.ParquetWriter(partial[t], CHUNKS_SCHEMA, compression='zstd') for t in targets})
    stats = {t: {'ntoks': [], 'fields': [], 'n_blob': 0} for t in targets}
    n_docs = n_paras = 0
    try:
        for bucket in buckets:
            t = pq.read_table(bucket)
            docs = t.take(np.argsort(_first_ids(t), kind='stable')).to_pylist()
            del t
            layouts = [doc_layout(r) for r in docs]
            # token count of every paragraph (raw text), batched calls
            pt = _token_counts(tok, [lt[0][s:e] for lt in layouts for _, s, e in lt[2]])
            n_docs += len(docs)
            n_paras += len(pt)
            cache: dict[str, int] = {}

            def ntok(s: str) -> int:
                v = cache.get(s)
                if v is None:
                    v = cache[s] = len(tok(s, add_special_tokens=False)['input_ids'])
                return v

            writers['docs'].write_table(pa.Table.from_pylist(
                [{'doc_id': r['doc_ids'][0], 'doc_ids_group': r['doc_ids'], 'domain': r['domain'], 'url': r['url'],
                  'title': r['title'], 'doc_text': lt[0]} for r, lt in zip(docs, layouts)], schema=DOCS_SCHEMA))
            for target in targets:
                rows = []
                p0 = 0
                for r, (doc_text, spans, paras) in zip(docs, layouts):
                    ptoks = pt[p0:p0 + len(paras)]
                    p0 += len(paras)
                    pieces = [(f, s, e) for f, s, e in spans] + chunk_paragraphs(doc_text, paras, ptoks, target, ntok)
                    for f, s, e in pieces:
                        rows.append((r['doc_ids'][0], r['doc_ids'], f, s, e, doc_text[s:e]))
                ntoks = _token_counts(tok, [x[5] for x in rows])
                # drop unsplittable blobs: no whitespace at all and > 1.5 * target tokens (base64 / inline-image junk)
                keep = [not (nt > 1.5 * target and not any(ch.isspace() for ch in x[5])) for x, nt in zip(rows, ntoks)]
                st = stats[target]
                st['n_blob'] += len(rows) - sum(keep)
                rows = [x for x, k in zip(rows, keep) if k]
                ntoks = [nt for nt, k in zip(ntoks, keep) if k]
                texts = [x[5] for x in rows]
                # 100 % verbatim check
                dt = {r['doc_ids'][0]: lt[0] for r, lt in zip(docs, layouts)}
                bad = sum(dt[d][s:e] != t or not t.strip() for d, _, _, s, e, t in rows)
                assert bad == 0, f'{bad} chunks are not verbatim substrings'
                c0 = len(st['ntoks'])
                writers[target].write_table(pa.Table.from_pydict({
                    'chunk_id': list(range(c0, c0 + len(rows))), 'doc_id': [x[0] for x in rows],
                    'doc_ids_group': [x[1] for x in rows], 'field': [x[2] for x in rows],
                    'char_start': [x[3] for x in rows], 'char_end': [x[4] for x in rows],
                    'text': texts, 'n_tokens': ntoks}, schema=CHUNKS_SCHEMA), row_group_size=50000)
                st['ntoks'] += ntoks
                st['fields'] += [x[2] for x in rows]
            print(f'{n_docs} docs / {n_paras} paragraphs chunked ({time.time() - t0:.0f}s)', flush=True)
    except BaseException:
        for w in writers.values():
            w.close()
        for p in partial.values():
            p.unlink(missing_ok=True)
        raise
    for w in writers.values():
        w.close()
    for k in final:
        os.replace(partial[k], final[k])

    configs = {}
    for target in targets:
        st = stats[target]
        nt = np.array(st['ntoks'])
        fld = np.array(st['fields'])
        sub = np.isin(fld, SUBMITTABLE)
        configs[target] = {
            'chunks_all': len(nt), 'chunks_answer_body': int(sub.sum()),
            'by_field': {f: int((fld == f).sum()) for f in ('title', 'question', 'answer', 'body')},
            'tokens_answer_body': {'median': float(np.median(nt[sub])), 'p95': float(np.percentile(nt[sub], 95)),
                                   'max': int(nt[sub].max()), 'pct_over_target': round(100 * float((nt[sub] > target).mean()), 2),
                                   'pct_over_1_5x': round(100 * float((nt[sub] > 1.5 * target).mean()), 3)},
            'tokens_all': {'median': float(np.median(nt)), 'p95': float(np.percentile(nt, 95))},
            'dropped_whitespace_free_blobs': st['n_blob'], 'verbatim_ok': f'{len(nt)}/{len(nt)}'}
        print(f'target {target}: {len(nt)} chunks ({time.time() - t0:.0f}s)', flush=True)
    return {'docs': n_docs, 'paragraphs': n_paras, 'configs': configs}


if __name__ == '__main__':
    sys.exit(main())
