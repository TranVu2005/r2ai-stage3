import pandas as pd, numpy as np, zipfile, json, torch, sys
S = sys.argv[1]; S26 = __import__('os').environ['S26_SCRATCH'].rstrip('/') + '/'  # q_dense.npy, q_ids.npy of S26
r = pd.read_parquet(f'{S}/srcdoc_cands.parquet')
j4 = {int(x['id']): [int(d) for d in x['relevant_docs']] for x in json.loads(zipfile.ZipFile('out/runs/s31-strong-judge/build-topk/J4-K60.zip').read('J4-K60.json'))}
k90 = {int(x['id']): [int(d) for d in x['relevant_docs']] for x in json.loads(zipfile.ZipFile('out/runs/s26-gold-structure/base/K90-ph-nc140.zip').read(zipfile.ZipFile('out/runs/s26-gold-structure/base/K90-ph-nc140.zip').namelist()[0]))}
pool = pd.read_parquet('out/runs/s31-strong-judge/pharmacity-ext/pool13_index.parquet')[['query_id', 'doc_id', 'cosine_rank']]
ex = pd.read_parquet('out/runs/s31-strong-judge/pool-extra/extra_pairs.parquet')[['query_id', 'doc_id']]
inpool = set(zip(pool.query_id, pool.doc_id)) | set(zip(ex.query_id, ex.doc_id))
z = np.load('out/runs/s26-gold-structure/full/full_doc_dense.npz'); did = z['doc_id']; D = torch.from_numpy(z['dense']).cuda(); pos = {d: i for i, d in enumerate(did)}
Q = np.load(S26 + 'q_dense.npy'); qids = list(np.load(S26 + 'q_ids.npy')); Qt = torch.from_numpy(Q.astype(np.float16)).cuda()
corpus = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id)
out = []
for th in [0.7, 0.8]:
    best = r[r.r_max >= th].sort_values('r_max', ascending=False)
    for qid, g in best.groupby('q'):
        ids = sorted({int(d) for l in g.doc_ids for d in l} | set(g.owner))
        ids_c = [d for d in ids if d in corpus]
        rk = [j4[qid].index(d) + 1 for d in ids_c if d in j4[qid]]
        sc = (D @ Qt[qids.index(qid)]).float(); srt = torch.sort(sc, descending=True).values
        gr = [int(torch.searchsorted(-srt, -sc[pos[d]]).item()) + 1 for d in ids if d in pos]
        out.append({'th': th, 'q': qid, 'r_max': g.r_max.max(), 'n_src': len(ids_c), 'j4_best': min(rk) if rk else np.nan,
                    'in_pool': any((qid, d) in inpool for d in ids_c), 'in_k90': any(d in k90[qid] for d in ids_c), 'cos_grank_best': min(gr) if gr else np.nan})
o = pd.DataFrame(out); o.to_parquet(f'{S}/srcrank.parquet')
for th, g in o.groupby('th'):
    print(f'== r_max>={th}: queries {len(g)}; src docs/query mean {g.n_src.mean():.2f}')
    print('  J4 best rank quantiles', g.j4_best.quantile([.25, .5, .75]).tolist(), 'rank1', int((g.j4_best == 1).sum()), '<=5', int((g.j4_best <= 5).sum()), '<=20', int((g.j4_best <= 20).sum()), '<=60', int(g.j4_best.notna().sum()))
    print('  in pool13/extra', int(g.in_pool.sum()), ' in K90-ph-nc140', int(g.in_k90.sum()), ' cosine global rank quantiles', g.cos_grank_best.quantile([.5, .9, 1]).tolist())
    print('  outside J4 top-60:', g[g.j4_best.isna()][['q', 'r_max', 'cos_grank_best', 'in_pool']].to_string(index=False))
