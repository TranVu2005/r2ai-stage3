r"""S26 H2/H6: which canonical docs contain long verbatim spans of a test query?

Normalisation = metric (NFKC -> html.unescape -> lower -> collapse whitespace -> strip); tokens = regex \w+ words
(Vietnamese syllables; not BGE-M3 subwords). n = 8 words (whole query when shorter). For every doc (all langs),
count distinct query n-grams present; keep pairs with >= 1 match. Also flags exact normalised title == query.
"""
import os
import html, re, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

N = 8
DOCS = 'data/canonical-20261018/docs.parquet'
W = re.compile(r'\w+')


def norm(s):
    return re.sub(r'\s+', ' ', html.unescape(unicodedata.normalize('NFKC', s)).lower()).strip()


def grams(toks, n):
    return {hash(tuple(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def load_q():
    q = pd.read_parquet('data/raw/query.parquet')
    qg, qn, qnorm = defaultdict(list), {}, {}
    for i, t in zip(q.id, q['query']):
        nt = norm(t); toks = W.findall(nt); n = min(N, len(toks))
        g = grams(toks, n); qn[i] = (len(g), n); qnorm[i] = ' '.join(toks)
        for h in g:
            qg[(n, h)].append(i)
    return qg, qn, qnorm


QG, QN, QNORM = load_q()
NS = sorted({n for n, _ in QG})
QBYTEXT = {v: k for k, v in QNORM.items()}


def work(rg):
    t = pq.ParquetFile(DOCS).read_row_group(rg, columns=['doc_id', 'lang', 'doc_text']).to_pandas()
    out = []
    for d, lang, txt in zip(t.doc_id, t.lang, t.doc_text):
        if lang != 'vi':
            continue
        nt = norm(txt); toks = W.findall(nt)
        title = ' '.join(W.findall(norm(txt.split('\n\n', 1)[0])))
        hit = defaultdict(int)
        for n in NS:
            for h in grams(toks, n):
                for qi in QG.get((n, h), ()):
                    hit[qi] += 1
        for qi, c in hit.items():
            out.append((qi, int(d), c, QN[qi][0], title == QNORM[qi]))
        if title in QBYTEXT and QBYTEXT[title] not in hit:
            out.append((QBYTEXT[title], int(d), 0, QN[QBYTEXT[title]][0], True))
    return out


if __name__ == '__main__':
    n = pq.ParquetFile(DOCS).metadata.num_row_groups
    rows = []
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as p:
        for k, r in enumerate(p.imap_unordered(work, range(n), chunksize=4)):
            rows += r
            if k % 100 == 0:
                print(k, n, len(rows), flush=True)
    df = pd.DataFrame(rows, columns=['query_id', 'doc_id', 'matched', 'q_grams', 'title_eq'])
    df['cover'] = df.matched / df.q_grams
    df.to_parquet(Path(sys.argv[1]) / 'ngram_hits.parquet', index=False)
    print('pairs', len(df))
