import json
import numpy as np, pandas as pd
R = "D:/GitHub/r2ai-stage3/out/runs/"; C = R + "s38-chunk-metric-v2/compose/"; S = R + "s36-fuse4/"
s_free = 1.7405181516323944
pc = pd.read_parquet(S + "cal/pcal-free.parquet")
G = (pc.groupby("query_id").p_cal.sum() * s_free * 1.1414)   # + ~12.4% mean tail share (diagnostic)
ds = pd.read_parquet(S + "docsel/docsel-EF2-free.parquet", columns=["query_id", "doc_id", "p4b", "chunk"]).merge(pc[["query_id", "doc_id", "p_cal"]], on=["query_id", "doc_id"], how="left")
assert ds.p_cal.notna().all()
cache = pd.read_parquet(C + "EF2-free-full+w768.cache.parquet", columns=["query_id", "doc_id", "rank", "body_tokens"])
bt = cache[cache["rank"] == 1].set_index(["query_id", "doc_id"]).body_tokens
ds["g"] = ds.query_id.map(G); k = ds.groupby("query_id").size(); ds["k"] = ds.query_id.map(k)
ds["bt"] = [bt.get((q, d), np.nan) for q, d in zip(ds.query_id, ds.doc_id)]
N = int(ds.chunk.sum()); print("carriers in docsel", N)
def macro_f2(sel, wins):
    d = ds[sel].copy(); d["w"] = wins.reindex(d.index).fillna(0) if wins is not None else 0
    g = d.groupby("query_id")
    tp = g.p_cal.sum(); n = g.size() + g.w.sum()
    allq = pd.Index(range(1, 1201))
    tp, n = tp.reindex(allq, fill_value=0), n.reindex(allq, fill_value=0)
    gq = G.reindex(allq)
    f2 = 5 * tp / (4 * gq + n); r = tp / gq; p = tp / n.replace(0, np.nan)
    return float(f2.mean()), float(r.mean()), float(p.mean()), int(n.sum())
cur = ds.chunk.values
res = {"current": macro_f2(cur, None)}
# A: global budget N, marginal gain p/(4G+k) per query, no per-query threshold
ds["score_A"] = ds.p_cal / (4 * ds.g + ds.k)
selA = ds.score_A >= ds.score_A.nlargest(N).min()
res["A_marginal_gain_same_N"] = macro_f2(selA.values, None)
# A2: absolute p_cal threshold equal to current t_c in p_cal terms (same as current)
# B: window allocation by p_cal (docs with bt > 1907), 9561 windows
long_ = ds.bt > 1907
res["windows_candidates_bt>1907_among_carriers"] = int((long_ & ds.chunk).sum())
cur_w = ds.chunk & (ds.bt > 2002)
b_idx = ds[ds.chunk & long_].p_cal.nlargest(9561).index
for label, idx in (("windows_current_T2002", ds.index[cur_w]), ("windows_top_p_cal_9561", b_idx)):
    w = pd.Series(1.0, index=idx)
    res[label] = {"sum_p_cal": float(ds.loc[idx].p_cal.sum()), "n": len(idx), "macro_f2_carriers_plus_windows_as_chunks": macro_f2(cur, w)[0]}
res["A_vs_current_recall_pct"] = (res["A_marginal_gain_same_N"][1] / res["current"][1] - 1) * 100
res["A_vs_current_f2_pct"] = (res["A_marginal_gain_same_N"][0] / res["current"][0] - 1) * 100
res["A_carriers_overlap_with_current"] = float((selA.values & cur).sum() / N)
# per query carriers after A
cA = ds[selA].groupby("query_id").size().reindex(range(1, 1201), fill_value=0); c0 = ds[ds.chunk].groupby("query_id").size().reindex(range(1, 1201), fill_value=0)
res["carriers_per_query_p10_p50_p90"] = {"current": [float(np.quantile(c0, x)) for x in (.1, .5, .9)], "A": [float(np.quantile(cA, x)) for x in (.1, .5, .9)]}
json.dump(res, open(S + "cal/chunkrule-proxy.json", "w"), indent=1); print(json.dumps(res, indent=1))
