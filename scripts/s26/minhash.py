"""S26 H3: near-duplicate clusters over canonical docs (vi+zh) — MinHash/LSH, Jaccard >= 0.7 on shingles.

Shingles: vi = 5 consecutive words (\w+ after metric normalisation); zh = 5 consecutive non-space characters
(Chinese has no word boundaries). 64 permutations, LSH 16 bands x 4 rows, candidate pairs verified by signature
agreement >= 0.7; buckets larger than 2,000 docs are skipped and counted. Union-find gives clusters.
Run with PYTHONHASHSEED=0 so worker shingle hashes agree.
"""
import os, html, re, sys, unicodedata
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

assert os.environ.get('PYTHONHASHSEED') == '0'
DOCS = 'data/canonical-20261018/docs.parquet'
P = 64; BANDS, ROWS = 16, 4; MAXB = 2000; THR = 0.7
rng = np.random.RandomState(42)
PR = np.uint64((1 << 61) - 1)
A = rng.randint(1, 1 << 31, P).astype(np.uint64); B = rng.randint(0, 1 << 31, P).astype(np.uint64)
W = re.compile(r'\w+')
norm = lambda s: re.sub(r'\s+', ' ', html.unescape(unicodedata.normalize('NFKC', s)).lower()).strip()


def sig(text, lang):
    t = norm(text)
    if lang == 'zh':
        s = t.replace(' ', ''); sh = {s[i:i + 5] for i in range(max(1, len(s) - 4))}
    else:
        w = W.findall(t); sh = {' '.join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
    h = np.fromiter((hash(x) & 0xFFFFFFFF for x in sh), np.uint64, len(sh))
    return ((h[:, None] * A + B) % PR).min(0).astype(np.uint32)


def work(rg):
    t = pq.ParquetFile(DOCS).read_row_group(rg, columns=['doc_id', 'lang', 'doc_text']).to_pandas()
    return t.doc_id.values, t.lang.values, np.stack([sig(x, l) for x, l in zip(t.doc_text, t.lang)])


if __name__ == '__main__':
    S = Path(sys.argv[1]); n = pq.ParquetFile(DOCS).metadata.num_row_groups
    ids, langs, sigs = [], [], []
    with Pool(int(sys.argv[2])) as p:
        for k, (i, l, s) in enumerate(p.imap(work, range(n), chunksize=4)):
            ids.append(i); langs.append(l); sigs.append(s)
            if k % 100 == 0:
                print('sig', k, n, flush=True)
    ids = np.concatenate(ids); langs = np.concatenate(langs); sig_ = np.concatenate(sigs)
    np.savez(S / 'minhash_sig.npz', ids=ids, langs=langs, sig=sig_)
    parent = np.arange(len(ids))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    skipped = 0; pairs = 0
    for b in range(BANDS):
        key = sig_[:, b * ROWS:(b + 1) * ROWS]
        kv = key.view(np.dtype((np.void, key.dtype.itemsize * ROWS))).ravel()
        order = np.argsort(kv, kind='stable'); ks = kv[order]
        cut = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        for g in np.split(order, cut):
            if len(g) < 2:
                continue
            if len(g) > MAXB:
                skipped += len(g); continue
            sub = sig_[g]
            for a in range(len(g)):
                agree = (sub[a + 1:] == sub[a]).mean(1)
                for j in np.flatnonzero(agree >= THR):
                    ra, rb = find(g[a]), find(g[a + 1 + j])
                    if ra != rb:
                        parent[ra] = rb; pairs += 1
        print('band', b, 'unions', pairs, 'skipped', skipped, flush=True)
    root = np.array([find(x) for x in range(len(ids))])
    pd.DataFrame({'doc_id': ids, 'lang': langs, 'cluster': ids[root]}).to_parquet(S / 'minhash_clusters.parquet', index=False)
    print('done unions', pairs, 'skipped bucket docs', skipped)
