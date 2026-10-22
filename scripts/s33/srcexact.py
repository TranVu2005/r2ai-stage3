import pandas as pd, numpy as np, zipfile, json, sys
S = sys.argv[1]
r = pd.read_parquet(f'{S}/srcdoc_cands.parquet')
dd = pd.read_parquet('data/canonical-20261018/docs.parquet', columns=['doc_id', 'host']).set_index('doc_id').host
j4 = {int(x['id']): [int(d) for d in x['relevant_docs']] for x in json.loads(zipfile.ZipFile('out/runs/s31-strong-judge/build-topk/J4-K60.zip').read('J4-K60.json'))}
corpus = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id)
for th in [0.9, 0.95]:
    b = r[r.r_max >= th]; qs = sorted(b.q.unique())
    ranks, hosts, nsrc = [], [], []
    for q in qs:
        g = b[b.q == q]; ids = {int(d) for l in g.doc_ids for d in l if int(d) in corpus}; nsrc.append(len(ids))
        rk = [j4[q].index(d) + 1 for d in ids if d in j4[q]]; ranks.append(min(rk) if rk else None)
        hosts += [dd.get(int(o), '?') for o in g.owner]
    ranks = pd.Series(ranks, dtype=float)
    print(f'r_max>={th}: {len(qs)} queries; src docs/query {np.mean(nsrc):.2f}; J4 best rank: =1 {int((ranks==1).sum())}, 2-5 {int(((ranks>1)&(ranks<=5)).sum())}, 6-60 {int((ranks>5).sum())}, >60 {int(ranks.isna().sum())}; hosts', pd.Series(hosts).value_counts().to_dict())
    print('   ranks', ranks.tolist())
# vinmec share among all vi titles vs among exact matches; how many vinmec 'bai-viet' docs whose title is a question
