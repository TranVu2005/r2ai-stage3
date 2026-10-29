"""S45 step 2: hosts of the (rebuilt) public CH-rrf docsel, with P4B / p_cal / cosine rank per host and the S26 host retention.

  host_docsel.py <apply_dir> <name> <prior_json> [<other_name>]
With <other_name> (e.g. the P-host probe), also the docs added / dropped per query against <name>, by host.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_rebuild import host_lang  # noqa: E402

A, name, prior = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
other = sys.argv[4] if len(sys.argv) > 4 else None
LOW = 0.30          # "low retention" = shrunk top-10 retention <= 0.30 (S26 mean 0.349)


def load(n):
    return pd.read_parquet(A / f'docsel-{n}.parquet'), pd.read_parquet(A / f'pcal-{n}.parquet')


sel, pc = load(name)
sels = [sel] + ([load(other)[0]] if other else [])
hl = host_lang({int(d) for f in sels + [pc] for d in f.doc_id})


def tag(df):
    df = df.copy()
    df['host'] = [hl[int(d)][0] for d in df.doc_id]; df['lang'] = [hl[int(d)][1] for d in df.doc_id]
    df['r_host'] = [prior[h]['shrunk'] if (l == 'zh' and h in prior and not h.startswith('_')) else np.nan for h, l in zip(df.host, df.lang)]
    return df


sel, pc = tag(sel), tag(pc)
Q = sel.query_id.nunique()
print('pool pairs / query by lang', (pc.groupby('lang').size() / Q).round(1).to_dict())
print('docsel docs / query by lang', (sel.groupby('lang').size() / Q).round(2).to_dict())
z, zp = sel[sel.lang == 'zh'], pc[pc.lang == 'zh']
t = z.groupby('host').agg(docs=('doc_id', 'size'), p4b=('p4b', 'mean'), p_cal=('p_cal', 'mean'), cos_rank=('cosine_rank', 'mean'),
                          r_host=('r_host', 'first'))
t['per_query'] = t.docs / Q
t['pool_pairs'] = zp.groupby('host').size(); t['sel_rate'] = t.docs / t.pool_pairs; t['pool_p_cal'] = zp.groupby('host').p_cal.mean()
print(t.sort_values('docs', ascending=False).round(3).to_string())
v = sel[sel.lang == 'vi']
print(f'vi: docs/query {len(v) / Q:.1f}, P4B {v.p4b.mean():.3f}, p_cal {v.p_cal.mean():.3f}')
x = z[z.r_host <= LOW]
qq = x.groupby('query_id').size().reindex(sel.query_id.unique(), fill_value=0)
print(f'low-retention zh hosts (shrunk <= {LOW}): docs/query {len(x) / Q:.2f}, share of docsel {len(x) / len(sel):.3f}, '
      f'p_cal {x.p_cal.mean():.3f}, P4B {x.p4b.mean():.3f}; per query P10/P50/P90 {qq.quantile([.1, .5, .9]).tolist()}')
# same-rank comparison: p_cal of low vs other zh hosts inside cosine-rank bins of the pool
zp = zp.merge(pc[['query_id', 'doc_id']].assign(), on=['query_id', 'doc_id'])
pr = pd.read_parquet(A / f'docsel-{name}.parquet', columns=['query_id', 'doc_id'])
if other:
    o = tag(load(other)[0])
    a = set(zip(sel.query_id, sel.doc_id)); b = set(zip(o.query_id, o.doc_id))
    add = o[[k not in a for k in zip(o.query_id, o.doc_id)]]; drop = sel[[k not in b for k in zip(sel.query_id, sel.doc_id)]]
    print(f'{other} vs {name}: added {len(add) / Q:.2f} / dropped {len(drop) / Q:.2f} docs per query; docs/query {len(o) / Q:.1f} vs {len(sel) / Q:.1f}')
    tab = pd.DataFrame({'added': add.groupby(['lang', 'host']).size(), 'dropped': drop.groupby(['lang', 'host']).size()}).fillna(0)
    tab = tab.assign(net=tab.added - tab.dropped).sort_values('net')
    print((tab / Q).round(3).to_string())
