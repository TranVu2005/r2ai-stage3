"""S40 rehearsal check (reads public files ON PURPOSE, never an input of r2ai.newq): docsel of the rehearsal build against the docsel of
120A-free-v2 for the same queries (original ids via id-map-rh100.parquet), with the per-query cause of every difference:

* pool       docs of the free-v2 top-300 not in the new top-300 (query vector drift or a cosine tie block straddling the cut);
* 4b         per pair |P4B new - P4B free-v2| on the pairs both pools hold (Kaggle numerics);
* cmax/jina  same for the C-max and Jina scores (Jina is listwise: free-v2 scored other pools, so its scores are not comparable 1:1);
* cut        k (docs kept) new vs free-v2.

Usage: compare_docsel.py <run> <build name> [out.json]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import paths  # noqa: E402

run, name = Path(sys.argv[1]), sys.argv[2]
m = pd.read_parquet(paths.RUNS_DIR / "s40" / "rehearsal-input" / "id-map-rh100.parquet")
to_new = dict(zip(m.orig_id.astype(int), m.id.astype(int)))
new = pd.read_parquet(run / "build" / name / "apply" / f"docsel-{name}.parquet")
old = pd.read_parquet(paths.RUNS_DIR / "release" / "120A-free-v2" / "apply" / "docsel-120A-free-v2.parquet")
old = old[old.query_id.isin(to_new)].assign(query_id=lambda f: f.query_id.map(to_new))
pcal_old = pd.read_parquet(paths.RUNS_DIR / "release" / "120A-free-v2" / "apply" / "pcal-120A-free-v2.parquet")
pcal_old = pcal_old[pcal_old.query_id.isin(to_new)].assign(query_id=lambda f: f.query_id.map(to_new))
pool_new = pd.read_parquet(run / "top.parquet", columns=["query_id", "doc_id"])
pcal_new = pd.read_parquet(run / "build" / name / "apply" / f"pcal-{name}.parquet")
cols = ["query_id", "doc_id", "p4b", "cmax", "jina"]
both = new[cols].merge(old[cols], on=["query_id", "doc_id"], suffixes=("_new", "_old"))
rows = []
for q in sorted(new.query_id.unique()):
    a, b = set(new.doc_id[new.query_id == q]), set(old.doc_id[old.query_id == q])
    pn, po = set(pool_new.doc_id[pool_new.query_id == q]), set(pcal_old.doc_id[pcal_old.query_id == q])
    rows.append({"query_id": int(q), "k_new": len(a), "k_old": len(b), "only_new": len(a - b), "only_old": len(b - a),
                 "pool_only_old": len(po - pn), "pool_only_new": len(pn - po), "same_order": new.doc_id[new.query_id == q].tolist() == old.doc_id[old.query_id == q].tolist()})
r = pd.DataFrame(rows)
r["cause"] = np.where((r.only_new == 0) & (r.only_old == 0), "same", np.where(r.pool_only_old > 0, "pool", "scores"))
d = {k: (both[f"{k}_new"] - both[f"{k}_old"]).abs() for k in ("p4b", "cmax", "jina")}
res = {"queries": len(r), "queries_identical_set": int((r.cause == "same").sum()), "queries_identical_order": int(r.same_order.sum()),
       "queries_differ_pool": int((r.cause == "pool").sum()), "queries_differ_scores_only": int((r.cause == "scores").sum()),
       "docs_new": int(r.k_new.sum()), "docs_old": int(r.k_old.sum()), "docs_only_new": int(r.only_new.sum()), "docs_only_old": int(r.only_old.sum()),
       "common_selected_pairs": int(len(both)),
       "abs_diff_selected": {k: {"max": float(v.max()), "p99": float(v.quantile(.99)), "mean": float(v.mean())} for k, v in d.items()}}
print(json.dumps(res, indent=1))
r.to_parquet(run / "build" / name / "check-docsel-vs-free-v2.parquet", index=False)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
