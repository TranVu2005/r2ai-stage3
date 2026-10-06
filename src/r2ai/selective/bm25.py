"""Sparse BM25 over de-accented URL slug tokens."""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import unquote, urlsplit

import numpy as np
import scipy.sparse as sp

from r2ai.selective.common import strip_marks

K1, B = 1.2, 0.75


def toks(s):
    return [t for t in re.findall(r'[a-z0-9]+', strip_marks(s).lower()) if t.isalpha() and len(t) >= 2]


def slug_toks(url):
    p = urlsplit(url)
    return toks(unquote(p.path + ' ' + p.query))


class Slug:
    def __init__(self, urls):
        docs = [slug_toks(u) for u in urls]
        self.vocab = {}
        indptr, indices, tf = [0], [], []
        for d in docs:
            for t, f in Counter(d).items():
                indices.append(self.vocab.setdefault(t, len(self.vocab))); tf.append(f)
            indptr.append(len(indices))
        n = len(docs)
        dl = np.array([len(d) for d in docs], float)
        X = sp.csr_matrix((np.array(tf, float), indices, indptr), shape=(n, len(self.vocab)))
        df = np.bincount(X.indices, minlength=len(self.vocab))
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = K1 * (1 - B + B * dl / dl.mean())
        X.data = X.data * (K1 + 1) / (X.data + np.repeat(norm, np.diff(X.indptr))) * idf[X.indices]
        self.WT = X.T.tocsr()
        self.n = n

    def queries(self, texts):
        Q = sp.lil_matrix((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for t in set(toks(text)):
                j = self.vocab.get(t)
                if j is not None:
                    Q[i, j] = 1.0
        return Q.tocsr()
