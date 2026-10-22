import pandas as pd, sys
S = sys.argv[1]
r = pd.read_parquet(f'{S}/srcdoc_cands.parquet'); q = pd.read_parquet('data/raw/query.parquet').set_index('id')['query']
t = pd.read_parquet('data/canonical-20261018/chunks_t256.parquet', columns=['doc_id', 'field', 'text'], filters=[('field', '==', 'title')]).set_index('doc_id').text
dd = pd.read_parquet('data/canonical-20261018/docs.parquet', columns=['doc_id', 'url', 'quality']).set_index('doc_id')
o = pd.read_parquet(f'{S}/srcrank.parquet'); o = o[o.th == 0.8]
for _, x in o.sort_values('j4_best', na_position='first').head(40).iterrows():
    b = r[(r.q == x.q)].sort_values('r_max', ascending=False).iloc[0]
    print(f"q{x.q} r={b.r_max:.2f} J4={x.j4_best} cos={x.cos_grank_best} | Q: {q[x.q][:110]}\n      T: {t.get(b.owner, '')[:110]} | {dd.url.get(b.owner, '')[:80]} {dd.quality.get(b.owner, '')}")
