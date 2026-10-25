import json, math, lzma, zipfile
import numpy as np, pandas as pd
R = "D:/GitHub/r2ai-stage3/out/runs/"
C = R + "s38-chunk-metric-v2/compose/"
S = R + "s36-fuse4/"
a, b = 0.5298075151496865, -1.40969757276664
def pcal(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
cols = ["query_id", "doc_id", "p4b", "rank"]
ds_f = pd.read_parquet(S + "docsel/docsel-EF2-free.parquet", columns=cols)
ds_r = pd.read_parquet(S + "docsel/docsel-EF2-cut-p4b.parquet", columns=cols)
p4b = {}
for d in (ds_r, ds_f):
    p4b.update(zip(zip(d.query_id, d.doc_id), d.p4b))
docs_f = set(zip(ds_f.query_id, ds_f.doc_id)); docs_r = set(zip(ds_r.query_id, ds_r.doc_id))
cf = pd.read_parquet(C + "EF2-free-full+w768.cache.parquet", columns=["query_id", "doc_id", "rank", "body_tokens"])
cr = pd.read_parquet(C + "EF2-cut-p4b-full+w768.cache.parquet", columns=["query_id", "doc_id", "rank", "body_tokens"])
car = {}
for n, c in (("free", cf), ("ref", cr)):
    r1 = c[c["rank"] == 1]
    car[n] = {"set": set(zip(r1.query_id, r1.doc_id)), "bt": dict(zip(zip(r1.query_id, r1.doc_id), r1.body_tokens))}
    r2 = c[c["rank"] == 2]
    car[n]["sec"] = set(zip(r2.query_id, r2.doc_id))
    car[n]["sec_bt"] = dict(zip(zip(r2.query_id, r2.doc_id), r2.body_tokens))
F, M = car["free"]["set"], car["ref"]["set"]
out = {}
def stat(keys):
    keys = list(keys)
    if not keys: return {"n": 0}
    p = pcal([p4b[k] for k in keys])
    return {"n": len(keys), "p_cal_mean": float(p.mean()), "p_cal_sum": float(p.sum()), "p4b_mean": float(np.mean([p4b[k] for k in keys]))}
com, onlyF, onlyM = F & M, F - M, M - F
out["carriers"] = {"free": len(F), "ref": len(M), "common": stat(com), "only_free": stat(onlyF), "only_ref": stat(onlyM),
                   "all_free": stat(F), "all_ref": stat(M)}
out["only_ref_split"] = {"doc_selected_in_free_but_no_chunk": stat(k for k in onlyM if k in docs_f),
                         "doc_not_in_free": stat(k for k in onlyM if k not in docs_f)}
out["only_free_split"] = {"doc_in_ref_docs_without_chunk": stat(k for k in onlyF if k in docs_r),
                          "doc_not_in_ref": stat(k for k in onlyF if k not in docs_r)}
# per query concentration
def per_q(S_):
    s = pd.Series([q for q, _ in S_]).value_counts()
    v = s.reindex(range(1, 1201), fill_value=0) if s.index.max() <= 1200 else s
    return v
qf, qr = per_q(F), per_q(M)
out["carriers_per_query"] = {"free_p10_p50_p90": [float(np.quantile(qf, x)) for x in (.1, .5, .9)], "ref": [float(np.quantile(qr, x)) for x in (.1, .5, .9)],
                             "free_top100_queries_share": float(qf.sort_values(ascending=False).head(100).sum() / qf.sum()),
                             "ref_top100_queries_share": float(qr.sort_values(ascending=False).head(100).sum() / qr.sum())}
# expected chunk metrics proxy: sum p_cal over carriers per query
def macro_sum(S_):
    d = pd.DataFrame({"q": [q for q, _ in S_], "p": pcal([p4b[k] for k in S_])})
    return d.groupby("q").p.sum()
sf, sr = macro_sum(list(F)), macro_sum(list(M))
out["expected_relevant_carriers"] = {"sum_free": float(sf.sum()), "sum_ref": float(sr.sum()), "ratio_free_over_ref": float(sf.sum() / sr.sum()),
        "macro_mean_free": float(sf.mean()), "macro_mean_ref": float(sr.reindex(sf.index, fill_value=0).mean()),
        "mean_p_per_carrier_free": float(sf.sum() / len(F)), "mean_p_per_carrier_ref": float(sr.sum() / len(M)), "lb_chunk_recall_ratio": 0.4554 / 0.4769,
        "lb_chunk_precision_ratio": 0.2236 / 0.2213}
# second chunk
sec_f, sec_r = car["free"]["sec"], car["ref"]["sec"]
Tf, Tr = 2002, 1907
bt_f, bt_r = car["free"]["bt"], car["ref"]["bt"]
sec_f_T = {k for k, v in car["free"]["sec_bt"].items() if v > Tf}; sec_r_T = {k for k, v in car["ref"]["sec_bt"].items() if v > Tr}
out["second"] = {"free": len(sec_f_T), "ref": len(sec_r_T),
  "free_carriers_with_body_tokens_in_(1907,2002]": int(sum(1 for k, v in bt_f.items() if Tr < v <= Tf)),
  "ref_carriers_with_body_tokens_in_(1907,2002]": int(sum(1 for k, v in bt_r.items() if Tr < v <= Tf)),
  "free_second_if_T1907": int(sum(1 for k, v in bt_f.items() if v > Tr)),
  "ref_second_if_T2002": int(sum(1 for k, v in bt_r.items() if v > Tf)),
  "free_carriers_body_gt_T2002": int(sum(1 for v in bt_f.values() if v > Tf)),
  "common_carriers_second_in_both": len([k for k in com if k in sec_f_T and k in sec_r_T]),
  "common_second_only_ref": len([k for k in com if k in sec_r_T and k not in sec_f_T]),
  "common_second_only_free": len([k for k in com if k in sec_f_T and k not in sec_r_T]),
  "second_lost_p_cal_sum_(1907-2002 window docs, free)": float(pcal([p4b[k] for k, v in bt_f.items() if Tr < v <= Tf]).sum())}
def qdist(v):
    v = np.array(v); return {"n": len(v), "p10_p50_p90": [float(np.quantile(v, x)) for x in (.1, .5, .9)] if len(v) else None}
out["body_tokens_all_carriers"] = {"free": qdist(list(bt_f.values())), "ref": qdist(list(bt_r.values()))}
out["body_tokens_only_free"] = qdist([bt_f[k] for k in onlyF]); out["body_tokens_only_ref"] = qdist([bt_r[k] for k in onlyM])
bins = [0, 256, 512, 1024, 1500, 1907, 2002, 3000, 5000, 1e9]
for n, bt, sec in (("free", bt_f, sec_f_T), ("ref", bt_r, sec_r_T)):
    h = np.histogram(list(bt.values()), bins=bins)[0].tolist()
    out[f"body_hist_{n}"] = dict(zip([f"<= {int(x)}" for x in bins[1:]], h))
json.dump(out, open(S + "cal/chunkdiff.json", "w"), indent=1)
print(json.dumps(out, indent=1))
