"""d211 analysis (a)(b)(c) (S35, 25/10). CPU only; writes cal/d211-analysis.json. Run: PYTHONPATH=src python scripts/s36/d211_analysis.py"""
import json
import math
import sys

import numpy as np
import pandas as pd

from r2ai.s29.qwen_judge import _load_submission  # noqa: E402
from r2ai.s31 import build as s31_build  # noqa: E402
from r2ai.s36 import cal, fuse  # noqa: E402

RUN = fuse.run_root()
fitted = json.loads((cal.cal_root() / "cal.json").read_text(encoding="utf-8"))["variants"]["p4b"]
A, B = fitted["fit"]["a"], fitted["fit"]["b"]
S_FREE = float(json.loads((cal.cal_root() / "ef2-free.json").read_text(encoding="utf-8"))["s_free"])
T_FREE, T_C = s31_build.T_FREE, 0.2030746281894702
ranked, union, free_pools, _, _, _ = fuse.load_frames(deepen=True)
jina, _ = fuse.load_jina(RUN / "extra" / "jina-scores.parquet")
cmax, _ = fuse.load_cmax(RUN / "extra" / "cmax-scores.parquet")
qids = sorted(union)
n_q = {q: len(union[q]) for q in qids}
out = {"a_b": [A, B], "s_free": S_FREE, "t_free": T_FREE}

# ---- (a) truncated pools and the docs the cut loses
contig = all([int(r["cosine_rank"]) for r in union[q]] == list(range(1, n_q[q] + 1)) for q in qids)
out["union_cosine_ranks_contiguous"] = bool(contig)
bottom = {q: float(union[q][cal.FREE_G_DEPTH - 1]["judge_score"]) for q in qids}
out["a_queries_211th_doc_p4b_at_or_above"] = {str(t): int(sum(v >= t for v in bottom.values())) for t in (T_FREE, 0.1, 0.15, T_C)}
out["a_scored_depth"] = {"p10_p50_p90_max": [float(np.quantile(list(n_q.values()), x)) for x in (.1, .5, .9)] + [max(n_q.values())]}
docs = pd.read_parquet("D:/GitHub/r2ai-stage3/data/canonical-20261018/docs.parquet", columns=["doc_id", "lang"])
lang_of = dict(zip(docs.doc_id.astype(int), docs.lang))
free_sel = pd.read_parquet(RUN / "docsel" / "docsel-EF2-free.parquet", columns=["query_id", "doc_id", "cosine_rank", "p4b"])
d211_sel = pd.read_parquet(RUN / "docsel" / "docsel-EF2-free-d211.parquet", columns=["query_id", "doc_id"])
m = free_sel.merge(d211_sel.assign(kept=1), on=["query_id", "doc_id"], how="left")
lost = m[m.kept.isna()].copy()
lost["lang"] = lost.doc_id.map(lambda d: lang_of.get(int(d), "vi"))
lost["beyond_211"] = lost.cosine_rank > cal.FREE_G_DEPTH
free_sel["lang"] = free_sel.doc_id.map(lambda d: lang_of.get(int(d), "vi"))
gained = d211_sel.merge(free_sel[["query_id", "doc_id"]].assign(inf=1), on=["query_id", "doc_id"], how="left")
gained = gained[gained.inf.isna()]
gained["lang"] = gained.doc_id.map(lambda d: lang_of.get(int(d), "vi"))
out["a_lost_vs_ef2_free"] = {
    "docs_ef2_free": int(len(free_sel)), "docs_d211": int(len(d211_sel)), "lost_total": int(len(lost)),
    "lost_by_lang": lost.lang.value_counts().to_dict(), "lost_beyond_cosine_211_by_lang": lost[lost.beyond_211].lang.value_counts().to_dict(),
    "lost_within_211": int((~lost.beyond_211).sum()), "lost_queries": int(lost.query_id.nunique()),
    "lost_p4b_mean": float(lost.p4b.mean()), "lost_p4b_median": float(lost.p4b.median()),
    "ef2_free_by_lang": free_sel.lang.value_counts().to_dict(), "gained_total": int(len(gained)), "gained_by_lang": gained.lang.value_counts().to_dict()}
p_lost = 1.0 / (1.0 + np.exp(-(A * cal.logit(lost.p4b.to_numpy()) + B)))
out["a_lost_vs_ef2_free"]["expected_gold_in_lost_docs"] = float(p_lost.sum())
out["a_lost_vs_ef2_free"]["lost_p_cal_mean"] = float(p_lost.mean())


# ---- pools cut at a cosine depth, model on them
def cut_pools(depth_of):
    pools = {}
    for q in qids:
        m_q = depth_of(q)
        pools[q] = s31_build._with_judge_rank([dict(r) for r in union[q][:m_q]])
    return pools


def predict(depth_of):
    pools = cut_pools(depth_of)
    frame = cal.Frame(fuse.FOUR_KEYS, "free", data={"pools": pools, "union": union, "jina": jina, "cmax": cmax}, allow_missing=True)
    model = cal.Calibrated(frame, "p4b", A, B, S_FREE, use_tail=True, g_depth=cal.FREE_G_DEPTH)
    keep = cal.ef2_keep(model)
    stuck = int(sum(1 for q in qids if pools[q][-1]["judge_score"] >= T_FREE))
    res = model.expected(keep)
    res.update({"docs": int(sum(keep.values())), "keep_p10_p50_p90": [float(np.quantile(list(keep.values()), x)) for x in (.1, .5, .9)],
                "queries_bottom_doc_p4b_at_or_above_t_free": stuck, "missing": frame.missing,
                "queries_at_depth": int(sum(1 for q in qids if keep[q] == len(frame.order[q])))})
    return res, frame, model


# ---- (b) fixed depth D for every query
def need(depth):
    new_4b = sum(max(0, depth - n_q[q]) for q in qids)
    miss_j = miss_c = 0
    for q in qids:
        for r in union[q][:depth]:
            key = (q, int(r["doc_id"]))
            miss_j += key not in jina
            miss_c += key not in cmax
    return {"depth": depth, "queries_needing_4b": int(sum(1 for q in qids if n_q[q] < depth)),
            "pairs_4b": int(new_4b), "pairs_jina": int(miss_j + new_4b), "pairs_cmax": int(miss_c + new_4b),
            "pairs_jina_within_scored": int(miss_j), "pairs_cmax_within_scored": int(miss_c)}


out["b"] = {}
for name, fn in (("211", lambda q: min(211, n_q[q])), ("300", lambda q: min(300, n_q[q])), ("400", lambda q: min(400, n_q[q])),
                 ("all_scored", lambda q: n_q[q])):
    res, _, _ = predict(fn)
    out["b"][name] = {"predicted": res}
for d in (300, 400):
    out["b"][str(d)]["need"] = need(d)
out["b"]["adaptive_ef2_free_reference"] = json.loads((cal.cal_root() / "ef2-free.json").read_text(encoding="utf-8"))["predicted"]

# ---- (c) LB-known doc sets predicted with the d211 G (s_free over the cosine top-211 + tail)
_, frame211, model211 = predict(lambda q: min(211, n_q[q]))
g211 = model211.g
p4b_of = dict(frame211.p4b_of)
for q in qids:
    for r in union[q]:
        p4b_of[(q, int(r["doc_id"]))] = float(r["judge_score"])
for q, rows in ranked.items():
    for r in rows:
        p4b_of.setdefault((q, int(r["doc_id"])), float(r["judge_score"]))
sets = {"FUSE4-free": (RUN / "build-docread" / "FUSE4-free-docread.zip", 0.4104), "EF2-free": (RUN / "build-docread" / "EF2-free-docread.zip", 0.4184),
        "EF2-cut-p4b": (RUN / "build-docread" / "EF2-cut-p4b-docread.zip", 0.4209)}
for n, path in cal.submitted_zips().items():
    if n in cal.LB_DOC_F2:
        sets[n] = (path, cal.LB_DOC_F2[n])
res_c = {}
for name, (path, lb) in sets.items():
    prec, rec, f2, miss = [], [], [], 0
    for row in _load_submission(path):
        q, ids = int(row["id"]), row["relevant_docs"]
        ps = []
        for d in ids:
            p4 = p4b_of.get((q, int(d)))
            if p4 is None:
                miss += 1
                continue
            ps.append(1.0 / (1.0 + math.exp(-(A * float(cal.logit(np.array([p4]))[0]) + B))))
        tp = sum(ps)
        prec.append(tp / len(ids))
        rec.append(tp / g211[q])
        f2.append(5 * tp / (4 * g211[q] + len(ids)))
    pred = float(np.mean(f2))
    res_c[name] = {"lb_doc_f2": lb, "predicted_f2": pred, "error": pred - lb, "docs_without_p4b": miss, "predicted_precision": float(np.mean(prec)),
                   "predicted_recall": float(np.mean(rec))}
out["c"] = res_c
out["c_pairwise_vs_lb"] = {f"{a}-{b}": {"lb": res_c[a]["lb_doc_f2"] - res_c[b]["lb_doc_f2"], "model": res_c[a]["predicted_f2"] - res_c[b]["predicted_f2"]}
                           for a, b in (("EF2-cut-p4b", "EF2-free"), ("EF2-free", "FUSE4-free"), ("EF2-cut-p4b", "FUSE4-free"), ("J3-thrC", "K4-thr-C"))
                           if a in res_c and b in res_c}
path = cal.cal_root() / "d211-analysis.json"
path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
