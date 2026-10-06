"""Step 2 (zh, 2% sample): recall of pre-download rankings vs content+rerank reference. Proxy, not gold.

Reference per query = top-K (K=10/30) of the cached rerank-ordered `hybrid` list in retrieval/q*.json.
Signals: (a) dense BGE-M3 (cached vi query vec <-> embedded slug text), only where the slug has words/CJK;
(b) structural priors (path prefix, numeric-ID bin) learned on the sample with 5-fold CV.
Ranking is within each domain (the crawl is per domain). Random baseline = N%.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import defaultdict
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

from r2ai.paths import CORPUS_FILE, QUERY_FILE, RUNS_DIR
from r2ai.selective.common import OUT

ZS = RUNS_DIR / 'zh-sample'
NS = (0.01, 0.05, 0.10)
KS = (10, 30)


def load_refs(docs):
    q = pd.read_parquet(QUERY_FILE)
    assert q.id.tolist() == list(range(1, 1201)), 'query order'
    known = set(docs.doc_id)
    refs = {K: {} for K in KS}
    for qi in q.id:
        d = json.load(open(ZS / 'retrieval' / f'q{qi}.json', encoding='utf-8'))
        lst = [i for i, _ in d['hybrid']]
        for K in KS:
            refs[K][qi] = [i for i in lst[:K] if i in known]
    return refs


def path_feats(url):
    p = urlsplit(url)
    segs = [unquote(s) for s in p.path.split('/') if s]
    prefix = '/'.join(segs[:-1][:2]) or '/'
    nums = re.findall(r'\d+', segs[-1]) if segs else []
    return prefix, (int(nums[-1]) if nums else None)


def cv_prior(df, hits, kind, m=5.0, folds=5, seed=42):
    """Out-of-fold predicted hit rate for each doc from `kind` buckets."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(df))
    if kind == 'prefix':
        key = df.prefix.to_numpy()
    else:  # 20 quantile bins of numeric ID (docs w/o ID -> -1)
        idv = df.idnum.to_numpy(dtype=float)
        ok = ~np.isnan(idv)
        key = np.full(len(df), -1)
        if ok.sum() > 40:
            qs = np.quantile(idv[ok], np.linspace(0, 1, 21)[1:-1])
            key[ok] = np.searchsorted(qs, idv[ok])
    score = np.zeros(len(df))
    for f in range(folds):
        tr, te = fold != f, fold == f
        g = hits[tr].mean()
        s = pd.DataFrame({'k': key[tr], 'h': hits[tr]}).groupby('k').h.agg(['sum', 'count'])
        rate = ((s['sum'] + m * g) / (s['count'] + m)).to_dict()
        score[te] = [rate.get(k, g) for k in key[te]]
    return score


def recall_static(score, hits, n_frac, rng):
    """Micro recall of hit mass in top n% (ties broken randomly)."""
    n = max(1, int(round(n_frac * len(score))))
    order = np.lexsort((rng.random(len(score)), -score))
    return hits[order[:n]].sum() / max(hits.sum(), 1)


def recall_dense(sim, ref_pos, n_frac):
    """sim [Q,D], ref_pos: list per query of doc column indices. Micro recall over queries with refs."""
    D = sim.shape[1]
    n = max(1, int(round(n_frac * D)))
    num = den = 0
    for qi, pos in enumerate(ref_pos):
        if not len(pos):
            continue
        thr = np.partition(sim[qi], D - n)[D - n]
        num += int((sim[qi][pos] >= thr).sum())
        den += len(pos)
    return num / max(den, 1)


def embed_texts(texts, cache):
    if cache.exists():
        return np.load(cache)
    from r2ai.index.bge_m3 import M3Encoder
    enc = M3Encoder(device='cpu', fp16=False, max_len=64)
    out = []
    for i in range(0, len(texts), 32):
        out.append(enc.encode_batch(texts[i:i + 32])[0].astype(np.float32))
        print('embed', i, len(texts), file=sys.stderr, flush=True)
    arr = np.concatenate(out)
    np.save(cache, arr)
    return arr


def slug_text_for(domain, url):
    segs = [unquote(s) for s in urlsplit(url).path.split('/') if s]
    if domain == 'a-hospital.com':
        return segs[-1] if segs else ''
    if domain == 'msdmanuals.cn':
        return ' '.join(s.replace('-', ' ') for s in segs[-2:])
    return ''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs = pd.read_parquet('data/chunks_zh_sample/docs.parquet', columns=['doc_id', 'domain', 'url', 'title'])
    refs = load_refs(docs)
    hit = {K: defaultdict(int) for K in KS}
    for K in KS:
        for qi, lst in refs[K].items():
            for i in lst:
                hit[K][i] += 1
    for K in KS:
        docs[f'h{K}'] = docs.doc_id.map(hit[K]).fillna(0).astype(int)
    pf = docs.url.map(path_feats)
    docs['prefix'] = [a for a, _ in pf]
    docs['idnum'] = [b if b is not None else np.nan for _, b in pf]
    rng = np.random.default_rng(42)
    rows = []
    for dom, d in docs.groupby('domain'):
        d = d.reset_index(drop=True)
        base = {'domain': dom, 'n_docs': len(d), 'hits10': int(d.h10.sum()), 'hits30': int(d.h30.sum())}
        if d.h30.sum() < 20:
            continue
        for kind in ('prefix', 'idbin'):
            for K in KS:
                h = d[f'h{K}'].to_numpy(float)
                sc = cv_prior(d, h, kind)
                for n in NS:
                    rows.append({**base, 'signal': f'prior_{kind}', 'K': K, 'N': n,
                                 'recall': recall_static(sc, h, n, rng)})
    # dense on slug/title where text exists
    qd = pickle.load(open(ZS / 'retrieval' / 'queries.pkl', 'rb'))[0].astype(np.float32)
    for dom in ('a-hospital.com', 'msdmanuals.cn'):
        d = docs[docs.domain == dom].reset_index(drop=True)
        pos = {i: k for k, i in enumerate(d.doc_id)}
        for field, texts in (('slug', [slug_text_for(dom, u) for u in d.url]), ('title_after_fetch', d.title.fillna('').tolist())):
            E = embed_texts(texts, OUT / f'emb_{dom}_{field}.npy')
            sim = qd @ E.T
            for K in KS:
                rp = [[pos[i] for i in refs[K][qi] if i in pos] for qi in range(1, 1201)]
                for n in NS:
                    rows.append({'domain': dom, 'n_docs': len(d), 'hits10': int(d.h10.sum()), 'hits30': int(d.h30.sum()),
                                 'signal': f'dense_{field}', 'K': K, 'N': n, 'recall': recall_dense(sim, rp, n)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'zh_recall.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 300)
    pv = res[res.K == 30].pivot_table(index=['domain', 'n_docs', 'hits30', 'signal'], columns='N', values='recall').round(3)
    print(pv)


if __name__ == '__main__':
    main()
