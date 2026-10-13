"""Compare r2ai.index.chunk.chunk_paragraphs (t256, the index chunker) with vi_quality.chunker.structured_chunks on a doc sample.

    python -m r2ai.vi_quality.compare_chunkers --per-host 300 --target 256 --out reports/s12_vi_chunker_compare.json

Sample: first `--per-host` ok docs (n_tokens_bge_m3 >= 50) per domain in shard order, seed-free. Both chunkers see the same
doc_layout paragraphs and the real BGE-M3 tokenizer. Only answer/body chunks are compared (title / question singles are shared).
"""
from __future__ import annotations

from r2ai.paths import DOCS_DIR, assert_writable, resolve_path

import argparse
import glob
import json
import lzma
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.index.chunk import chunk_paragraphs, doc_layout
from r2ai.vi_quality.chunker import classify, structured_chunks, _norm

COLS = ['doc_ids', 'domain', 'title', 'question', 'answer', 'body', 'paragraphs', 'headings', 'status', 'n_tokens_bge_m3']


def sample_docs(docs_dir, per_host: int) -> list[dict]:
    got: Counter = Counter()
    rows = []
    for f in sorted(glob.glob(f'{docs_dir}/*.parquet')):
        host = Path(f).name.split('__')[0]
        if got[host] >= per_host:
            continue
        for r in pq.read_table(f, columns=COLS).to_pylist():
            if got[host] >= per_host:
                break
            if r['status'] == 'ok' and (r['n_tokens_bge_m3'] or 0) >= 50 and r['body']:
                rows.append(r)
                got[host] += 1
    return rows


def zip_size(texts: list[str]) -> dict:
    raw = json.dumps(texts, ensure_ascii=False).encode('utf-8')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr('x.json', raw)
    return {'json_bytes': len(raw), 'zip_deflate9': buf.tell(), 'lzma': len(lzma.compress(raw, preset=6))}


def stats(name, chunks, tok_counts, target):
    t = np.array(tok_counts)
    return {'chunks': len(chunks), 'tokens_mean': float(t.mean()), 'tokens_p50': float(np.percentile(t, 50)),
            'tokens_p95': float(np.percentile(t, 95)), 'tokens_max': int(t.max()),
            'over_1.5x_target': float((t > 1.5 * target).mean()), 'under_32': float((t < 32).mean()), **zip_size(chunks)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--per-host', type=int, default=300)
    ap.add_argument('--target', type=int, default=256)
    ap.add_argument('--out', default='reports/s12_vi_chunker_compare.json')
    a = ap.parse_args(argv)
    out = assert_writable(a.out)
    from r2ai.gold_check.gold_check_common import load_tokenizer
    tok = load_tokenizer()
    cache: dict[str, int] = {}

    def ntok(s):
        v = cache.get(s)
        if v is None:
            v = cache[s] = len(tok(s, add_special_tokens=False)['input_ids'])
        return v

    rows = sample_docs(resolve_path(a.docs_dir), a.per_host)
    res = {'old': {'texts': [], 'tok': []}, 'new': {'texts': [], 'tok': []}}
    struct = {k: Counter() for k in res}
    cover = {'old': 0, 'new': 0}
    body_chars = 0
    for r in rows:
        doc_text, spans, paras = doc_layout(r)
        ptoks = [ntok(doc_text[s:e]) for _, s, e in paras]
        heads = {_norm(h) for h in (r['headings'] or []) if h}
        body_chars += sum(e - s for _, s, e in paras)
        for name, ch in (('old', chunk_paragraphs(doc_text, paras, ptoks, a.target, ntok)),
                         ('new', structured_chunks(doc_text, paras, ptoks, a.target, ntok, headings=r['headings'] or ()))):
            for f, s, e in ch:
                res[name]['texts'].append(doc_text[s:e])
                res[name]['tok'].append(ntok(doc_text[s:e]))
            cover[name] += sum(e - s for _, s, e in ch)
            c = struct[name]
            last_para = {e: i for i, (_, s, e) in enumerate(paras)}
            for f, s, e in ch:
                i = last_para.get(e)
                if i is not None and classify(doc_text[paras[i][1]:paras[i][2]], heads) == 'heading' and paras[i][0] != 'question':
                    c['chunks_ending_in_heading'] += 1
            qa = [i for i, p in enumerate(paras) if p[0] == 'question' and i + 1 < len(paras) and paras[i + 1][0] == 'answer']
            for i in qa:
                c['qa_pairs'] += 1
                if any(s <= paras[i][1] and e >= paras[i + 1][2] for _, s, e in ch):
                    c['qa_pairs_same_chunk'] += 1
            i = 0
            while i < len(paras):
                if classify(doc_text[paras[i][1]:paras[i][2]], heads) in ('item', 'row'):
                    j = i
                    k = classify(doc_text[paras[i][1]:paras[i][2]], heads)
                    while j + 1 < len(paras) and classify(doc_text[paras[j + 1][1]:paras[j + 1][2]], heads) == k:
                        j += 1
                    if j > i:
                        c['list_runs'] += 1
                        if not any(s <= paras[i][1] and e >= paras[j][2] for _, s, e in ch):
                            c['list_runs_split'] += 1
                    i = j + 1
                else:
                    i += 1
    report = {'docs': len(rows), 'hosts': len({r['domain'] for r in rows}), 'target': a.target, 'body_chars': body_chars,
              'chars_in_chunks': cover, 'struct': {k: dict(v) for k, v in struct.items()},
              **{k: stats(k, v['texts'], v['tok'], a.target) for k, v in res.items()}}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
