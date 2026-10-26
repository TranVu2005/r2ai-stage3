"""S40 rehearsal check (reads public files ON PURPOSE, never an input of r2ai.newq): the new top-300 pool of each rehearsal query against the
free-v2 pool of its original id (r2ai.s36.d300.merged_top on out/runs/s37/pool.npz + the S29 query vectors)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import paths  # noqa: E402
from r2ai.s36 import d300  # noqa: E402

run = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.RUNS_DIR / "s40" / "rh100"
m = pd.read_parquet(paths.RUNS_DIR / "s40" / "rehearsal-input" / "id-map-rh100.parquet")
new = pd.read_parquet(run / "top.parquet")
old = d300.merged_top(300)
old = old[old.query_id.isin(m.orig_id)]
to_new = dict(zip(m.orig_id, m.id))
old["query_id"] = old.query_id.map(to_new)
rows, diff_q = [], 0
for q, g in new.groupby("query_id"):
    o = old[old.query_id == q]
    a, b = set(g.doc_id), set(o.doc_id)
    same_order = list(g.doc_id) == list(o.doc_id)
    rows.append({"query_id": int(q), "only_new": len(a - b), "only_old": len(b - a), "same_order": same_order,
                 "new_cos_300": float(g.cosine_score.iloc[-1]), "old_cos_300": float(o.cosine_score.iloc[-1])})
r = pd.DataFrame(rows)
print(json.dumps({"queries": len(r), "queries_set_differs": int((r.only_new > 0).sum()), "docs_only_new_total": int(r.only_new.sum()),
                  "queries_order_differs": int((~r.same_order).sum()), "max_only_new": int(r.only_new.max())}, indent=1))
r.to_parquet(run / "check-pool-vs-free-v2.parquet", index=False)
