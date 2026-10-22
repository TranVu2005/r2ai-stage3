"""Item 2b: what are the adjacent-id pairs inside J4 top-20? duplicates (alias/phys) or distinct docs?"""
import pandas as pd, numpy as np, zipfile, json, difflib
S = __import__('sys').argv[1]
c = pd.read_parquet(f'{S}/corpus_meta.parquet').set_index('id')
dd = pd.read_parquet('data/canonical-20261018/docs.parquet', columns=['doc_id', 'phys_id', 'rule', 'quality', 'doc_text']).set_index('doc_id')
print('canonical docs', len(dd), 'rules', dd.rule.value_counts().to_dict())
rows = json.loads(zipfile.ZipFile('out/runs/s31-strong-judge/build-topk/J4-K40.zip').read('J4-K40.json'))
pairs = []
for r in rows:
    ids = [int(x) for x in r['relevant_docs']][:20]; pos = {d: i + 1 for i, d in enumerate(ids)}
    rs = sorted(ids, key=lambda d: c.row[d])
    for a, b in zip(rs, rs[1:]):
        if c.row[b] - c.row[a] <= 5: pairs.append((r['id'], a, b, int(c.row[b] - c.row[a]), pos[a], pos[b]))
p = pd.DataFrame(pairs, columns=['q', 'a', 'b', 'gap', 'rank_a', 'rank_b']); print('pairs gap<=5', len(p), 'gap==1', int((p.gap == 1).sum()))
def t(d): return dd.doc_text.get(d, None)
p['in_canon'] = p.a.isin(dd.index) & p.b.isin(dd.index)
p['same_phys'] = [dd.phys_id.get(a) is not None and dd.phys_id.get(a) == dd.phys_id.get(b) for a, b in zip(p.a, p.b)]
p['text_eq'] = [t(a) is not None and t(a) == t(b) for a, b in zip(p.a, p.b)]
p['text_sim'] = [difflib.SequenceMatcher(None, (t(a) or '')[:1500], (t(b) or '')[:1500], autojunk=False).ratio() for a, b in zip(p.a, p.b)]
p['host'] = c.host.reindex(p.a).values
p['url_a'] = c.url.reindex(p.a).values; p['url_b'] = c.url.reindex(p.b).values
p.to_parquet(f'{S}/adjpairs.parquet')
print(p[['in_canon', 'same_phys', 'text_eq']].mean().round(3).to_dict())
print('text_sim quantiles', p.text_sim.quantile([.1, .25, .5, .75, .9]).round(3).tolist(), 'share sim>=0.9', round((p.text_sim >= .9).mean(), 3))
print(p.host.value_counts().head(12).to_dict())
print('rank pairs: both in top5', int(((p.rank_a <= 5) & (p.rank_b <= 5)).sum()))
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 110)
print(p.sample(25, random_state=1)[['q', 'gap', 'rank_a', 'rank_b', 'same_phys', 'text_sim', 'url_a', 'url_b']].to_string())
