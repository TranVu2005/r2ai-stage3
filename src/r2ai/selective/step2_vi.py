"""Step 2 (vi cross-check): BM25 over de-accented URL slugs of all crawled vi URLs, recall vs RD150 doc sets.

Proxy (RD150 is a retrieval output, not gold). Global ranking and within-domain ranking; random = N%.
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from r2ai.paths import QUERY_FILE, RUNS_DIR
from r2ai.selective.common import OUT, VI_DB, alpha_tokens, ro, strip_marks

RD150 = RUNS_DIR / 'rrf-docs-2026-10-06/RD150/sub_RD150.json'
NS = (0.01, 0.05, 0.10)
K1, B = 1.2, 0.75


def toks(s):
    return [t for t in re.findall(r'[a-z0-9]+', strip_marks(s).lower()) if t.isalpha() and len(t) >= 2]


def main():
    c = ro(VI_DB)
    rows = c.execute('select url,doc_ids,domain,status from urls').fetchall()
    urls = [r[0] for r in rows]
    dom = np.array([r[2] for r in rows])
    status = np.array([r[3] for r in rows])
    from urllib.parse import unquote, urlsplit
    docs = [toks(unquote(urlsplit(u).path + ' ' + urlsplit(u).query)) for u in urls]
    N = len(docs)
    vocab, indptr, indices, tf = {}, [0], [], []
    for d in docs:
        for t, f in Counter(d).items():
            indices.append(vocab.setdefault(t, len(vocab))); tf.append(f)
        indptr.append(len(indices))
    dl = np.array([len(d) for d in docs], float)
    X = sp.csr_matrix((np.array(tf, float), indices, indptr), shape=(N, len(vocab)))
    df = np.bincount(X.indices, minlength=len(vocab))
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    norm = K1 * (1 - B + B * dl / dl.mean())
    X.data = X.data * (K1 + 1) / (X.data + np.repeat(norm, np.diff(X.indptr))) * idf[X.indices]
    WT = X.T.tocsr()  # V x N
    q = pd.read_parquet(QUERY_FILE)
    qv = np.zeros(len(q), dtype=object)
    Q = sp.lil_matrix((len(q), len(vocab)))
    for i, text in enumerate(q['query']):
        for t in set(toks(text)):
            j = vocab.get(t)
            if j is not None:
                Q[i, j] = 1.0
    Q = Q.tocsr()
    # refs: RD150 docs -> url rows
    row_of = {}
    for r, rr in enumerate(rows):
        for i in json.loads(rr[1]):
            row_of[i] = r
    rd = json.load(open(RD150, encoding='utf-8'))
    refs = [np.array(sorted({row_of[i] for i in e['relevant_docs'] if i in row_of}), int) for e in rd]
    qids = [int(e['id']) for e in rd]
    assert qids == q['id'].tolist()
    print('refs/query (vi rows):', np.mean([len(r) for r in refs]), 'rows', N)
    rng = np.random.default_rng(42)
    doms = sorted(set(dom))
    didx = {d: np.where(dom == d)[0] for d in doms}
    glob = {n: [0, 0] for n in NS}
    perdom = {d: {n: [0, 0] for n in NS} for d in doms}
    CH = 20
    for a in range(0, len(q), CH):
        S = (Q[a:a + CH] @ WT).toarray().astype(np.float32)
        S += rng.random(S.shape, dtype=np.float32) * 1e-6
        for k in range(S.shape[0]):
            r = refs[a + k]
            if not len(r):
                continue
            s = S[k]
            for n in NS:
                m = int(round(n * N))
                thr = np.partition(s, N - m)[N - m]
                glob[n][0] += int((s[r] >= thr).sum()); glob[n][1] += len(r)
            rdom = dom[r]
            for d in set(rdom):
                rr = r[rdom == d]
                ix = didx[d]
                sub = s[ix]
                for n in NS:
                    m = max(1, int(round(n * len(ix))))
                    thr = np.partition(sub, len(ix) - m)[len(ix) - m]
                    perdom[d][n][0] += int((s[rr] >= thr).sum()); perdom[d][n][1] += len(rr)
        print('chunk', a, flush=True)
    out = [{'scope': 'GLOBAL', 'domain': 'ALL', 'n_urls': N, 'refs': glob[NS[0]][1], **{f'R@{int(n*100)}%': glob[n][0] / glob[n][1] for n in NS}}]
    for d in doms:
        if perdom[d][NS[0]][1] >= 20:
            out.append({'scope': 'within_domain', 'domain': d, 'n_urls': len(didx[d]), 'refs': perdom[d][NS[0]][1],
                        **{f'R@{int(n*100)}%': perdom[d][n][0] / perdom[d][n][1] for n in NS}})
    res = pd.DataFrame(out)
    res.to_csv(OUT / 'vi_slug_bm25_recall.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 100)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
