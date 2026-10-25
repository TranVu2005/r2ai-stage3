"""S26 P-A4: can P1-vi+10's chunk text ("whole body") be rebuilt from canonical-20261018?

Rule (body_span): vi doc with an `answer` field -> doc_text[min char_start : max char_end] of its answer chunks;
other vi doc -> same span over its `body` chunks; zh doc -> whole doc_text; doc without chunks -> whole doc_text.
Samples N corpus IDs that P1-vi+10 ships a chunk for (seed fixed), rebuilds the text from canonical and reports byte
equality, and for mismatches whether the rebuilt text is a substring of the P1 chunk (P1 longer) and the host.
Usage: body_check.py <pa2_dir> [N=200] [seed=26]
"""
import json, os, sys
from collections import Counter
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
CAN = 'data/canonical-20261018'


def body_span(doc_text, lang, spans):
    """spans: {field: (start, end)} over doc_text, from chunks_t256 of that doc"""
    t = doc_text or ''
    if lang == 'zh' or not spans:
        return t
    a, b = spans.get('answer') or spans.get('body') or (0, len(t))
    return t[a:b]


def load_spans(doc_ids):
    ch = pq.read_table(f'{CAN}/chunks_t256.parquet', columns=['doc_id', 'field', 'char_start', 'char_end'],
                       filters=[('doc_id', 'in', sorted(doc_ids))]).to_pandas()
    g = ch.groupby(['doc_id', 'field']).agg(a=('char_start', 'min'), b=('char_end', 'max')).reset_index()
    out = {}
    for d, f, a, b in g.itertuples(index=False):
        out.setdefault(int(d), {})[f] = (int(a), int(b))
    return out


if __name__ == '__main__':
    pa2 = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 200; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 26
    c = pd.read_parquet(pa2 / 'pa2_p1_chunks.parquet')
    p = pd.read_parquet(pa2 / 'pa2_p1_docs.parquet').dropna(subset=['doc_id'])
    c['doc_id'] = c.corpus_id.map(dict(zip(p.corpus_id.astype(int), p.doc_id.astype(int))))
    s = c.dropna(subset=['doc_id']).sample(N, random_state=seed); s['doc_id'] = s.doc_id.astype(int)
    d = pq.read_table(f'{CAN}/docs.parquet', columns=['doc_id', 'lang', 'host', 'doc_text'],
                      filters=[('doc_id', 'in', sorted(set(s.doc_id)))]).to_pandas().set_index('doc_id')
    sp = load_spans(set(s.doc_id))
    res, miss = Counter(), Counter()
    for r in s.itertuples():
        if r.doc_id not in d.index:
            res['no_canonical_doc'] += 1; continue
        x = d.loc[r.doc_id]; b = body_span(x.doc_text, x.lang, sp.get(r.doc_id, {}))
        if b == r.chunk_text:
            res[f'equal_{x.lang}'] += 1
        else:
            res[f'differ_{x.lang}'] += 1
            miss[(x.host, 'rebuilt_is_substring_of_p1' if b in r.chunk_text else 'other')] += 1
    eq = sum(v for k, v in res.items() if k.startswith('equal'))
    print(json.dumps({'n': N, 'seed': seed, 'result': dict(res), 'equal_share': eq / N,
                      'mismatch_by_host': {f'{h} {k}': v for (h, k), v in miss.items()}}, indent=1, ensure_ascii=False))
