import json, zipfile, lzma, gc
import numpy as np, pandas as pd
R = "D:/GitHub/r2ai-stage3/out/runs/"; C = R + "s38-chunk-metric-v2/compose/"; S = R + "s36-fuse4/"
a, b, s_free = 0.5298075151496865, -1.40969757276664, 1.7405181516323944
pc = pd.read_parquet(S + "cal/pcal-free.parquet")
g_pool = pc.groupby("query_id").p_cal.sum() * s_free   # diagnostic G_q: scored-pool sum (same for both bundles)
def pcal(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
ds = pd.concat([pd.read_parquet(S + f"docsel/docsel-{n}.parquet", columns=["query_id", "doc_id", "p4b"]) for n in ("EF2-cut-p4b", "EF2-free")]).drop_duplicates(["query_id", "doc_id"])
p4b = dict(zip(zip(ds.query_id, ds.doc_id), ds.p4b))
out = {}
for n, name, T in (("ref", "EF2-cut-p4b-full+w768", 1907), ("free", "EF2-free-full+w768", 2002)):
    c = pd.read_parquet(C + f"{name}.cache.parquet", columns=["query_id", "doc_id", "rank", "body_tokens"])
    r1 = c[c["rank"] == 1].copy(); r1["p"] = pcal([p4b[k] for k in zip(r1.query_id, r1.doc_id)])
    r1["sec"] = r1.body_tokens > T
    g = r1.query_id.map(g_pool)
    per_q = r1.assign(w=r1.p / g).groupby("query_id").w.sum()
    per_q2 = r1[r1.sec].assign(w=lambda d: d.p / d.query_id.map(g_pool)).groupby("query_id").w.sum()
    out[n] = {"recall_proxy_carriers_macro": float(per_q.reindex(range(1, 1201), fill_value=0).mean()),
              "recall_proxy_second_chunks_macro": float(per_q2.reindex(range(1, 1201), fill_value=0).mean()),
              "second_p_cal_sum": float(r1[r1.sec].p.sum()), "carriers": len(r1),
              "n_chunks": len(r1) + int(r1.sec.sum()),
              "precision_proxy_per_chunk": float((r1.p.sum()) / (len(r1) + r1.sec.sum()))}
out["ratio_recall_proxy_free_over_ref"] = out["free"]["recall_proxy_carriers_macro"] / out["ref"]["recall_proxy_carriers_macro"]
out["ratio_second_proxy_free_over_ref"] = out["free"]["recall_proxy_second_chunks_macro"] / out["ref"]["recall_proxy_second_chunks_macro"]
out["lb_ratios"] = {"chunk_recall": 0.4554 / 0.4769, "chunk_precision": 0.2236 / 0.2213}
# bytes
def pieces(path):
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read(z.namelist()[0]))
    docs = json.dumps([r["relevant_docs"] for r in data], separators=(",", ":")).encode()
    texts = sum(len(c["chunk_text"].encode("utf-8")) for r in data for c in r["relevant_chunks"])
    n_chunks = sum(len(r["relevant_chunks"]) for r in data)
    del data; gc.collect()
    return {"relevant_docs_raw_bytes": len(docs), "relevant_docs_xz_bytes": len(lzma.compress(docs, preset=9)), "chunk_text_raw_bytes": texts, "chunks": n_chunks}
for n, zp in (("ref_final", C + "EF2-cut-p4b-full+w768-t1907.zip"), ("free_final", C + "EF2-free-full+w768-t2002.zip"),
              ("ref_base", S + "build-docread/EF2-cut-p4b-docread.zip"), ("free_base", S + "build-docread/EF2-free-docread.zip")):
    import os
    out.setdefault("bytes", {})[n] = {**pieces(zp), "zip_bytes": os.path.getsize(zp)}
json.dump(out, open(S + "cal/chunkdiff2.json", "w"), indent=1); print(json.dumps(out, indent=1))
