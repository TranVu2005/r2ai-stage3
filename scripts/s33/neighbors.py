"""Item 3: from J4 top-5, take same-host row neighbours +-w; their global cosine rank, J4 rank, 4B score vs random same-host docs."""
import pandas as pd, numpy as np, zipfile, json, torch
S = __import__('sys').argv[1]; S26 = __import__('os').environ['S26_SCRATCH'].rstrip('/') + '/'  # q_dense.npy, q_ids.npy of S26
c = pd.read_parquet(f'{S}/corpus_meta.parquet'); row_id = c.id.values; host = c.host.values; id2row = dict(zip(c.id.values, c.row.values))
z = np.load('out/runs/s26-gold-structure/full/full_doc_dense.npz'); did = z['doc_id']; D = torch.from_numpy(z['dense']).cuda()
pos = {d: i for i, d in enumerate(did)}
Q = np.load(S26 + 'q_dense.npy'); qids = np.load(S26 + 'q_ids.npy'); Qt = torch.from_numpy(Q.astype(np.float16)).cuda()
j4 = {int(r['id']): [int(x) for x in r['relevant_docs']] for r in json.loads(zipfile.ZipFile('out/runs/s31-strong-judge/build-topk/J4-K60.zip').read('J4-K60.json'))}
s4 = pd.concat([pd.read_parquet('out/runs/s31-strong-judge/pharmacity-ext/qwen3-reranker-4b-scores-pool13.parquet')[['query_id', 'doc_id', 'score']],
                pd.read_parquet('out/runs/s31-strong-judge/gpu/qwen3-reranker-4b/scores-extra.parquet')[['query_id', 'doc_id', 'score']]]).drop_duplicates(['query_id', 'doc_id'])
s4 = {(int(a), int(b)): float(s) for a, b, s in zip(s4.query_id, s4.doc_id, s4.score)}
by_host = pd.Series(np.arange(len(c))).groupby(host).apply(np.array).to_dict()
rng = np.random.default_rng(0); W = [1, 2, 3, 5, 10]; out = []
for qi, q in enumerate(qids):
    q = int(q); top5 = j4[q][:5]; rank_j4 = {d: i + 1 for i, d in enumerate(j4[q])}
    sc = (D @ Qt[qi]).float(); srt = torch.sort(sc, descending=True).values
    def grank(d):
        if d not in pos: return np.nan
        return int(torch.searchsorted(-srt, -sc[pos[d]], right=False).item()) + 1
    seen = set(top5)
    for d in top5:
        r = id2row.get(d)
        if r is None: continue
        for w in W:
            for nr in (r - w, r + w):
                if 0 <= nr < len(c) and host[nr] == host[r]:
                    n = int(row_id[nr])
                    if n in seen: continue
                    out.append({'q': q, 'seed': d, 'w': w, 'kind': 'neighbor', 'doc': n, 'grank': grank(n), 'j4rank': rank_j4.get(n, np.nan), 'p4b': s4.get((q, n), np.nan)})
            # random same-host control (2 per seed per w)
            for nr in rng.choice(by_host[host[r]], 2):
                n = int(row_id[nr])
                out.append({'q': q, 'seed': d, 'w': w, 'kind': 'random_host', 'doc': n, 'grank': grank(n), 'j4rank': rank_j4.get(n, np.nan), 'p4b': s4.get((q, n), np.nan)})
o = pd.DataFrame(out); o.to_parquet(f'{S}/neighbors.parquet')
g = o.groupby(['w', 'kind'])
t = pd.DataFrame({'n': g.size(), 'grank_median': g.grank.median(), 'grank<=100': g.grank.apply(lambda s: (s <= 100).mean()),
                  'grank<=1000': g.grank.apply(lambda s: (s <= 1000).mean()), 'in_J4_top40': g.j4rank.apply(lambda s: (s <= 40).mean()),
                  'in_J4_top60': g.j4rank.apply(lambda s: (s <= 60).mean()), 'has_4B': g.p4b.apply(lambda s: s.notna().mean()),
                  'p4b_mean_if_scored': g.p4b.mean(), 'p4b>=0.5_if_scored': g.p4b.apply(lambda s: (s.dropna() >= .5).mean())})
pd.set_option('display.width', 250); print(t.round(4).to_string())
# reference: 4B of J4 ranks 6-40
ref = [s4.get((q, d)) for q in j4 for d in j4[q][5:40]]; ref = np.array([x for x in ref if x is not None])
print('J4 ranks 6-40: mean P4B', ref.mean().round(4), 'share>=0.5', (ref >= .5).mean().round(4))
nb = o[(o.kind == 'neighbor') & (o.w == 1)]
new = nb[nb.j4rank.isna() | (nb.j4rank > 40)]
print('w=1 neighbours outside J4 top-40:', len(new), 'grank quantiles', new.grank.quantile([.1, .25, .5]).tolist(), 'scored 4B', int(new.p4b.notna().sum()), 'mean', round(new.p4b.mean(), 4))
