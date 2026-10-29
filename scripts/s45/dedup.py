"""S45 dedup hypothesis: does BTC's zh pool selection drop near-duplicates (one copy kept per cluster)?

Data: the S26 pre-v2 2% zh sample (29,385 docs, kept = any id of the doc group in corpus v2, 2,413 kept), its dt512 CLS
min-rank (out/runs/s45/fp/dt512.cls.npy), and the official zh corpus = canonical-20261018 lang zh (159,470 docs, all v2).
The rest of v1 zh (about 98%) was never crawled, so "whole zh corpus" here = canonical zh + the sample itself.

Twin of a sample doc i = another doc j (canonical or sample) that is near-identical, by two independent measures:
  a. Jaccard of 5-character shingles over the normalised body (title paragraph dropped, NFKC, lower case, only \\w
     characters kept), boilerplate removed: shingles found in more than DF_MAX = 100 docs of the universe are dropped (site
     templates such as the a-hospital formula index made raw Jaccard >= 0.5 link one page to 973 others), docs with fewer
     than 20 content shingles take no part. MinHash 126 permutations, LSH 42 bands x 3 rows (recall about 0.996 at J = 0.5), candidates with
     signature agreement >= 0.35 checked with the exact Jaccard of the shingle sets; thresholds 0.5 / 0.7 / 0.9.
  b. Cosine of the dt512 CLS doc vectors (S26 recipe: BGE-M3, doc_text[:4096], max_len 512) >= 0.95 / 0.98.
The kept copy of i itself (a canonical doc whose id is in i's doc_ids_group) is not a twin. A twin is KEPT when it is a
canonical doc (in v2) or a kept sample doc.

  dedup.py enc       sample dt512 CLS vectors -> sample_dense.npy (checked against the fp score matrix)
  dedup.py sig       MinHash signatures of sample + canonical zh -> sig.npz   (PYTHONHASHSEED=0)
  dedup.py pairs [content|raw]   LSH candidates + exact Jaccard -> pairs_jac.parquet (content) / pairs_jac_raw.parquet
  dedup.py dense     top-20 cosine neighbours of every sample doc -> pairs_cos.parquet
  dedup.py report    P1 / P2 / P3 tables and the kept-dropped pair comparison -> report.json
"""
import json, os, re, sys, time, unicodedata
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and out/ of the main checkout
sys.path.insert(0, 'src')

SAMPLE = 'D:/_r2ai_archive/data/chunks_zh_sample/docs.parquet'
CANON = 'data/canonical-20261018/docs.parquet'
CANON_VEC = 'out/runs/s26-gold-structure/full/full_doc_dense.npz'
FP = Path('out/runs/s45/fp')
OUT = Path('out/runs/s45/dedup'); OUT.mkdir(parents=True, exist_ok=True)
P, BANDS, ROWS = 126, 42, 3
MAXB = 5000                    # LSH buckets larger than this are skipped (counted)
SIG_MIN = 0.35                 # signature agreement needed before the exact Jaccard
DF_MAX = 100                   # content shingles: found in at most this many docs (boilerplate filter)
MIN_SH = 20                    # docs with fewer content shingles take no part in the Jaccard measure
COS_K = 20
rng = np.random.RandomState(42)
PR = np.uint64((1 << 61) - 1)
A = rng.randint(1, 1 << 31, P).astype(np.uint64); B = rng.randint(0, 1 << 31, P).astype(np.uint64)
NW = re.compile(r'\W+')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def body(text, title=None):
    """drop the title paragraph: the sample's own title, else the first paragraph of a canonical doc_text"""
    if title is not None:
        return text[len(title) + 2:] if title and text.startswith(title + '\n\n') else text
    head, sep, rest = text.partition('\n\n')
    return rest if sep and len(head) <= 200 else text


def shingles(b):
    s = NW.sub('', unicodedata.normalize('NFKC', b).lower())
    return {s[i:i + 5] for i in range(max(1, len(s) - 4))} if s else set()


def sig(b):
    sh = shingles(b)
    if not sh:
        return np.full(P, 0xFFFFFFFF, np.uint32), np.zeros(0, np.uint32)
    h = np.unique(np.fromiter((hash(x) & 0xFFFFFFFF for x in sh), np.uint64, len(sh)))
    return ((h[:, None] * A + B) % PR).min(0).astype(np.uint32), h.astype(np.uint32)


def sig_many(bodies):
    out = [sig(b) for b in bodies]
    return np.stack([o[0] for o in out]), [o[1] for o in out]


def load_sample():
    s = pd.read_parquet(SAMPLE, columns=['doc_id', 'doc_ids_group', 'domain', 'url', 'title', 'doc_text'])
    s['title'] = s.title.fillna(''); s['doc_text'] = s.doc_text.fillna('')
    s['group'] = s.doc_ids_group.map(lambda g: [int(x) for x in (json.loads(g) if isinstance(g, str) else g)])
    s['y'] = np.load(FP / 'y.npy').astype(int)
    return s


def load_canon(cols=('doc_id', 'host', 'url', 'doc_text')):
    return pd.read_parquet(CANON, columns=list(cols), filters=[('lang', '==', 'zh')])


def universe():
    """sample docs first (row i = sample row i), then canonical zh: (bodies, source, doc_id)"""
    s, c = load_sample(), load_canon()
    bodies = [body(x, t) for x, t in zip(s.doc_text, s.title)] + [body(x) for x in c.doc_text]
    src = np.r_[np.zeros(len(s), np.int8), np.ones(len(c), np.int8)]
    return bodies, src, np.r_[s.doc_id.values, c.doc_id.values].astype(np.int64)


def cmd_enc():
    import torch
    from r2ai.index.bge_m3 import M3Encoder
    s = load_sample()
    texts = [x[:4096] for x in s.doc_text]
    enc = M3Encoder(max_len=512)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 1024), np.float16)
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]
        out[idx] = enc.encode_batch([texts[i] for i in idx])[0]
        if k % 3200 == 0:
            log('enc', k, len(order))
    np.save(OUT / 'sample_dense.npy', out)
    q = np.load(FP / 'q_cls.npy').astype(np.float32)
    ref = np.load(FP / 'dt512.cls.npy', mmap_mode='r')
    cols = np.random.RandomState(0).choice(len(texts), 500, replace=False)
    diff = np.abs(q @ out[cols].astype(np.float32).T - ref[:, cols].astype(np.float32)).max()
    log('max |score - fp dt512.cls| on 500 docs', float(diff))
    del torch


def cmd_sig():
    assert os.environ.get('PYTHONHASHSEED') == '0'
    bodies, src, ids = universe()
    parts = [bodies[i:i + 2000] for i in range(0, len(bodies), 2000)]
    del bodies
    sigs, hs = [], []
    with Pool(6) as p:
        for k, (sg, h) in enumerate(p.imap(sig_many, parts)):
            sigs.append(sg); hs.extend(h)
            if k % 10 == 0:
                log('sig', k, len(parts))
    nsh = np.array([len(h) for h in hs], np.int64)
    np.savez(OUT / 'sig.npz', sig=np.concatenate(sigs), nsh=nsh, src=src, ids=ids)
    np.save(OUT / 'shingles.npy', np.concatenate(hs))          # sorted unique 32-bit shingle hashes per doc, offsets = cumsum(nsh)
    log('sig done', len(nsh), 'shingle hashes', int(nsh.sum()))


def content_shingles(H, nsh):
    """drop boilerplate: shingle hashes found in more than DF_MAX docs of the universe (site templates, sidebars)"""
    uniq, cnt = np.unique(H, return_counts=True)
    keep = cnt[np.searchsorted(uniq, H)] <= DF_MAX
    del uniq, cnt
    doc = np.repeat(np.arange(len(nsh)), nsh)
    nf = np.bincount(doc, weights=keep, minlength=len(nsh)).astype(np.int64)
    return np.asarray(H)[keep], nf


def minhash_rows(H, nsh):
    off = np.r_[0, np.cumsum(nsh)]
    S = np.full((len(nsh), P), 0xFFFFFFFF, np.uint32)
    for d in range(len(nsh)):
        if nsh[d]:
            h = H[off[d]:off[d + 1]].astype(np.uint64)
            S[d] = ((h[:, None] * A + B) % PR).min(0)
    return S


def cmd_pairs(mode='content'):
    """mode content: Jaccard over shingles with DF <= DF_MAX, docs with fewer than MIN_SH content shingles left out;
    mode raw: all shingles (site templates inflate it, kept for the record)"""
    z = np.load(OUT / 'sig.npz'); S, src, nsh = z['sig'], z['src'], z['nsh']
    ns = int((src == 0).sum())
    H = np.load(OUT / 'shingles.npy')
    if mode == 'content':
        H, nsh = content_shingles(H, nsh)
        log('content shingles', len(H), 'docs with <', MIN_SH, int((nsh < MIN_SH).sum()))
        nsh_ok = nsh >= MIN_SH
        S = minhash_rows(H, nsh)
        log('content signatures done')
    else:
        nsh_ok = nsh > 0
    off = np.r_[0, np.cumsum(nsh)]
    cand, skipped = set(), 0
    for b in range(BANDS):
        key = np.ascontiguousarray(S[:, b * ROWS:(b + 1) * ROWS])
        kv = key.view(np.dtype((np.void, key.dtype.itemsize * ROWS))).ravel()
        order = np.argsort(kv, kind='stable'); ks = kv[order]
        cut = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        for g in np.split(order, cut):
            g = g[nsh_ok[g]]
            if len(g) < 2:
                continue
            smp = g[g < ns]
            if not len(smp):
                continue
            if len(g) > MAXB:
                skipped += 1
                continue
            for i in smp:
                for j in g:
                    if j != i and not (j < ns and j < i):
                        cand.add((int(i), int(j)))
        log('band', b, 'candidates', len(cand), 'skipped buckets', skipped)
    cand = np.array(sorted(cand), np.int64).reshape(-1, 2)
    agree = (S[cand[:, 0]] == S[cand[:, 1]]).mean(1)
    keep = agree >= SIG_MIN
    cand, agree = cand[keep], agree[keep]
    log('pairs for exact Jaccard', len(cand))
    jac = np.empty(len(cand), np.float64)
    for k, (i, j) in enumerate(cand):
        a, b = H[off[i]:off[i + 1]], H[off[j]:off[j + 1]]
        n = len(np.intersect1d(a, b, assume_unique=True))
        jac[k] = n / (len(a) + len(b) - n)
        if k % 200000 == 0:
            log('exact', k, len(cand))
    name = 'pairs_jac' if mode == 'content' else 'pairs_jac_raw'
    pd.DataFrame({'i': cand[:, 0], 'j': cand[:, 1], 'sig_agree': agree.astype(np.float32), 'jaccard': jac.astype(np.float32)})         .to_parquet(OUT / f'{name}.parquet', index=False)
    if mode == 'content':
        np.save(OUT / 'nsh_content.npy', nsh)
    (OUT / f'{name}.json').write_text(json.dumps({'mode': mode, 'df_max': DF_MAX, 'min_sh': MIN_SH, 'candidates_after_sig': int(len(cand)),
                                                  'skipped_buckets': skipped, 'perm': P, 'bands': BANDS, 'rows': ROWS}), encoding='utf-8')
    log('pairs done', mode, len(cand), 'J>=0.5', int((jac >= 0.5).sum()))


def cmd_dense():
    import torch
    s = load_sample(); c = load_canon(('doc_id',))
    sd = np.load(OUT / 'sample_dense.npy')
    z = np.load(CANON_VEC)
    pos = pd.Series(np.arange(len(z['doc_id'])), index=z['doc_id'])
    cd = z['dense'][pos.loc[c.doc_id.values].values]
    del z
    T = torch.from_numpy(np.concatenate([sd, cd])).cuda()          # rows: sample, then canonical zh
    Q = torch.from_numpy(sd).cuda()
    vals, idx = [], []
    for k in range(0, len(Q), 512):
        sim = Q[k:k + 512] @ T.T
        v, i = sim.float().topk(COS_K + 1, dim=1)
        vals.append(v.cpu().numpy()); idx.append(i.cpu().numpy())
    vals, idx = np.concatenate(vals), np.concatenate(idx)
    ii = np.repeat(np.arange(len(sd)), COS_K + 1)
    df = pd.DataFrame({'i': ii, 'j': idx.ravel(), 'cos': vals.ravel()})
    df = df[df.i != df.j]
    df.to_parquet(OUT / 'pairs_cos.parquet', index=False)
    log('dense done', len(df), 'cos>=0.95', int((df.cos >= 0.95).sum()))


def min_rank():
    import torch
    S = torch.from_numpy(np.load(FP / 'dt512.cls.npy')).cuda().float()
    mr = (S.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    return mr


def cmd_report():
    from sklearn.metrics import roc_auc_score
    s = load_sample(); c = load_canon(('doc_id', 'host', 'url', 'doc_text'))
    ns = len(s)
    s['mr'] = min_rank()
    ids = np.r_[s.doc_id.values, c.doc_id.values].astype(np.int64)
    kept_u = np.r_[s.y.values, np.ones(len(c), int)]                   # every canonical doc is in v2
    host_u = np.r_[s.domain.values, c.host.values]
    url_u = np.r_[s.url.values, c.url.values]
    len_u = np.r_[[len(body(x, t)) for x, t in zip(s.doc_text, s.title)], [len(body(x)) for x in c.doc_text]]
    grp = [set(g) for g in s.group]
    jp = pd.read_parquet(OUT / 'pairs_jac.parquet'); cp = pd.read_parquet(OUT / 'pairs_cos.parquet')
    rp = pd.read_parquet(OUT / 'pairs_jac_raw.parquet')
    # both directions for sample-sample pairs, then drop self copies (canonical j with id in i's own group)
    def sym(df):
        r = df[df.j < ns].rename(columns={'i': 'j', 'j': 'i'})
        return pd.concat([df, r], ignore_index=True)
    jp, rp = sym(jp), sym(rp[rp.jaccard >= 0.7])
    def not_self(df):
        return np.array([ids[j] not in grp[i] for i, j in zip(df.i.values, df.j.values)], bool)
    jp = jp[not_self(jp)]; cp = cp[not_self(cp)]; rp = rp[not_self(rp)]
    for df in (jp, cp, rp):
        df['j_kept'] = kept_u[df.j.values]; df['j_canon'] = df.j.values >= ns
    measures = {f'J>={t}': jp[jp.jaccard >= t] for t in (0.5, 0.7, 0.9)}
    measures.update({f'cos>={t}': cp[cp.cos >= t] for t in (0.95, 0.98)})
    measures['rawJ>=0.7'] = rp                                          # all shingles, templates included
    nshc = np.load(OUT / 'nsh_content.npy')[:ns]
    rep_short = {'sample_docs_under_min_sh': int((nshc < MIN_SH).sum()),
                 'kept_under_min_sh': int(((nshc < MIN_SH) & (s.y.values == 1)).sum())}
    def flags(df):
        anyt = np.zeros(ns, bool); keptt = np.zeros(ns, bool); dropt = np.zeros(ns, bool)
        anyt[df.i.values] = True
        keptt[df.i.values[df.j_kept.values == 1]] = True
        dropt[df.i.values[df.j_kept.values == 0]] = True
        return anyt, keptt, dropt
    groups = {'kept': s.y == 1, 'dropped': s.y == 0,
              'kept mr<=3': (s.y == 1) & (s.mr <= 3), 'dropped mr<=3': (s.y == 0) & (s.mr <= 3),
              'kept mr=1': (s.y == 1) & (s.mr == 1), 'dropped mr=1': (s.y == 0) & (s.mr == 1),
              'dropped mr>20': (s.y == 0) & (s.mr > 20)}
    rep = {'n': {g: int(m.sum()) for g, m in groups.items()}, 'content_filter': rep_short, 'step1': {}, 'P2': {}, 'P3': {}}
    base_auc = roc_auc_score(s.y, -s.mr); base_k1 = float(s.y[s.mr == 1].mean())
    rep['base'] = {'auc_minrank': round(base_auc, 4), 'kept_at_mr1': round(base_k1, 4), 'kept_at_mr<=3': round(float(s.y[s.mr <= 3].mean()), 4)}
    for name, df in measures.items():
        anyt, keptt, dropt = flags(df)
        rep['step1'][name] = {g: {'any_twin': round(float(anyt[m].mean()), 4), 'kept_twin': round(float(keptt[m].mean()), 4),
                                  'dropped_twin': round(float(dropt[m].mean()), 4)} for g, m in groups.items()}
        drop = (s.y.values == 0) & keptt
        r = s[~drop]
        rep['P2'][name] = {'removed_dropped_with_kept_twin': int(drop.sum()),
                           'removed_mr<=3': int((drop & (s.mr.values <= 3)).sum()),
                           'kept_at_mr1': round(float(r.y[r.mr == 1].mean()), 4),
                           'kept_at_mr<=3': round(float(r.y[r.mr <= 3].mean()), 4),
                           'auc_minrank': round(roc_auc_score(r.y, -r.mr), 4)}
        top = s.mr.values <= 3
        t = pd.DataFrame({'domain': s.domain.values, 'y': s.y.values, 'twin': anyt})[top]
        tab = t.groupby(['domain', 'twin']).y.agg(['size', 'mean']).unstack('twin')
        rep['P3'][name] = {d: {('twin' if tw else 'no_twin'): [int(row[('size', tw)]) if not np.isnan(row[('size', tw)]) else 0,
                                                              None if np.isnan(row[('mean', tw)]) else round(float(row[('mean', tw)]), 3)]
                               for tw in (False, True) if ('size', tw) in row.index}
                           for d, row in tab.iterrows()}
    # rule: dropped (mr<=3) vs its best kept twin, at J>=0.7
    df = measures['J>=0.7']
    d = df[(s.y.values[df.i.values] == 0) & (s.mr.values[df.i.values] <= 3) & (df.j_kept.values == 1)]
    d = d.sort_values('jaccard', ascending=False).drop_duplicates('i')
    cmp = pd.DataFrame({'i': d.i.values, 'j': d.j.values, 'jaccard': d.jaccard.values,
                        'len_i': len_u[d.i.values], 'len_j': len_u[d.j.values],
                        'id_i': ids[d.i.values], 'id_j': ids[d.j.values],
                        'host_i': host_u[d.i.values], 'host_j': host_u[d.j.values],
                        'url_i': url_u[d.i.values], 'url_j': url_u[d.j.values], 'j_canon': d.j_canon.values})
    cmp.to_parquet(OUT / 'pairs_kept_dropped.parquet', index=False)
    if len(cmp):
        path = lambda u: re.sub(r'^https?://(www\.)?', '', str(u)).rstrip('/')
        rep['rule'] = {'pairs': int(len(cmp)),
                       'kept_longer': round(float((cmp.len_j > cmp.len_i).mean()), 4),
                       'kept_equal_len': round(float((cmp.len_j == cmp.len_i).mean()), 4),
                       'kept_smaller_id': round(float((cmp.id_j < cmp.id_i).mean()), 4),
                       'same_host': round(float((cmp.host_i == cmp.host_j).mean()), 4),
                       'same_url_path': round(float((cmp.url_i.map(path) == cmp.url_j.map(path)).mean()), 4),
                       'kept_is_canonical': round(float(cmp.j_canon.mean()), 4),
                       'host_pairs': {f'{a} -> {b}': int(n) for (a, b), n in
                                      cmp.groupby(['host_i', 'host_j']).size().sort_values(ascending=False).head(15).items()}}
    (OUT / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    {'enc': cmd_enc, 'sig': cmd_sig, 'pairs': cmd_pairs, 'dense': cmd_dense, 'report': cmd_report}[sys.argv[1]](*sys.argv[2:])
