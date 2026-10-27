"""S41 rehearsal check: the rerun export of the 100 rehearsal queries must reproduce out/runs/s40/rh100 (top-300 pairs, 4B pairs, C-max
chunk rows with the same ids), so the rh100 4B / C-max scores can be imported. Usage: rehearsal_same_pairs.py <new run> <old run>"""
import json
import sys
from pathlib import Path

import pandas as pd

new, old = Path(sys.argv[1]), Path(sys.argv[2])


def one(run: Path, pattern: str) -> Path:
    found = sorted(run.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"{run}: {len(found)} files for {pattern}")
    return found[0]


checks = {
    "top": (["query_id", "doc_id", "pair_id", "cosine_rank"], "top.parquet"),
    "4b_pairs": (["pair_id", "query_id", "doc_id"], "4b/kaggle-dataset-*/*/extra_pairs.parquet"),
    "cmax_rows": (["pair_id", "query_id", "doc_id", "chunk_id"], "cmax/chunk_pairs.parquet"),
}
res = {}
for name, (cols, pattern) in checks.items():
    a = pd.read_parquet(one(new, pattern), columns=cols).sort_values(cols).reset_index(drop=True)
    b = pd.read_parquet(one(old, pattern), columns=cols).sort_values(cols).reset_index(drop=True)
    res[name] = {"rows_new": len(a), "rows_old": len(b), "equal": bool(a.equals(b))}
print(json.dumps(res, indent=1))
(new / "same-pairs.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
raise SystemExit(0 if all(r["equal"] for r in res.values()) else 1)
