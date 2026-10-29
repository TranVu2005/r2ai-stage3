"""S45 probe pool depth (Vu 29/10): how deep in the cosine order the R0 docsel reaches, and what a pool cut to the top-D cosine changes.

  pool_depth.py bands <apply_dir> [name]            R0 docsel by cosine-rank band: docs / query, mean p_cal, P4B, C-max, carriers / query
  pool_depth.py diff <apply_dir> <base> <name> ...   per probe vs base: docs dropped / added per query, carrier set changed / query

The probes themselves: ``public_rebuild.py --top ... --depth D`` (pool = cosine top-D, everything else as R0), then S38 compose.
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd

BANDS = [(1, 100), (101, 150), (151, 200), (201, 250), (251, 300)]


def bands(apply_dir, name='R0'):
    sel = pd.read_parquet(apply_dir / f'docsel-{name}.parquet')
    car = pd.read_parquet(apply_dir / f'carriers-{name}.parquet').assign(carrier=True)
    sel = sel.merge(car, on=['query_id', 'doc_id'], how='left').fillna({'carrier': False})
    nq = sel.query_id.nunique(); out = []
    for lo, hi in BANDS:
        b = sel[sel.cosine_rank.between(lo, hi)]
        out.append({'band': f'{lo}-{hi}', 'docs_per_query': len(b) / nq, 'queries_with_doc': int(b.query_id.nunique()),
                    'p_cal_mean': float(b.p_cal.mean()), 'p4b_mean': float(b.p4b.mean()), 'cmax_mean': float(b.cmax.mean()),
                    'sum_p_cal_per_query': float(b.p_cal.sum() / nq), 'carriers_per_query': float(b.carrier.sum() / nq)})
    res = {'name': name, 'queries': nq, 'docs': int(len(sel)), 'bands': out,
           'max_cosine_rank_p50_p90_p99': [float(x) for x in sel.groupby('query_id').cosine_rank.max().quantile([.5, .9, .99])]}
    print(json.dumps(res, indent=1)); return res


def diff(apply_dir, base, names):
    def load(n):
        s = pd.read_parquet(apply_dir / f'docsel-{n}.parquet'); c = pd.read_parquet(apply_dir / f'carriers-{n}.parquet')
        return ({int(q): set(g.doc_id.astype(int)) for q, g in s.groupby('query_id')},
                {int(q): set(g.doc_id.astype(int)) for q, g in c.groupby('query_id')}, s)
    bd, bc, bs = load(base); res = {}
    for n in names:
        d, c, s = load(n); qs = sorted(bd)
        drop = np.array([len(bd[q] - d.get(q, set())) for q in qs]); add = np.array([len(d.get(q, set()) - bd[q]) for q in qs])
        cd = np.array([len(bc.get(q, set()) - c.get(q, set())) for q in qs]); ca = np.array([len(c.get(q, set()) - bc.get(q, set())) for q in qs])
        res[n] = {'docs': int(len(s)), 'docs_per_query': len(s) / len(qs), 'carriers': int(sum(map(len, c.values()))),
                  'docs_dropped_per_query': float(drop.mean()), 'docs_added_per_query': float(add.mean()),
                  'queries_doc_set_changed': int(((drop + add) > 0).sum()), 'dropped_p50_p90_p99_max': [float(x) for x in np.quantile(drop, [.5, .9, .99, 1])],
                  'carriers_dropped_per_query': float(cd.mean()), 'carriers_added_per_query': float(ca.mean()),
                  'queries_carrier_set_changed': int(((cd + ca) > 0).sum())}
    print(json.dumps(res, indent=1)); return res


if __name__ == '__main__':
    cmd, apply_dir = sys.argv[1], Path(sys.argv[2])
    if cmd == 'bands':
        bands(apply_dir, *sys.argv[3:4])
    else:
        diff(apply_dir, sys.argv[3], sys.argv[4:])
