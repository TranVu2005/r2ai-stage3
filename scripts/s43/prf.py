"""S43 pseudo-relevance feedback cosine on the pool (offline signal, CPU).

q' = normalise(q + alpha * mean of the doc vectors of the query's top-m pool docs by P4B); cos' = cos(q', d) for every pool doc.
Writes ``<out>`` with query_id, doc_id, prf_<m>_<alpha> columns.

    python scripts/s43/prf.py --base out/runs/s43/base --out out/runs/s43/base/prf.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals  # noqa: E402

GRID = [(5, 0.5), (10, 0.5), (10, 1.0), (20, 1.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pool = pd.read_parquet(args.base / "pool.parquet", columns=["query_id", "doc_id", "p4b", "pos"])
    with np.load(signals.ingest.QUERY_VECTORS) as z:
        q_index, qv = pd.Index(z["q_ids"].astype(np.int64)), z["q_dense"].astype(np.float32)
    docs = np.unique(pool.doc_id.to_numpy(np.int64))
    dv = signals.doc_vectors(docs).astype(np.float16)
    pool["di"] = pd.Index(docs).get_indexer(pool.doc_id)
    out = pool[["query_id", "doc_id"]].copy()
    for m, alpha in GRID:
        col = np.zeros(len(pool), np.float32)
        for q, g in pool.groupby("query_id"):
            fb = g.sort_values(["p4b", "pos"], ascending=[False, True]).di.to_numpy()[:m]
            v = qv[q_index.get_loc(q)] + alpha * dv[fb].astype(np.float32).mean(axis=0)
            v /= np.linalg.norm(v)
            col[g.index.to_numpy()] = dv[g.di.to_numpy()].astype(np.float32) @ v
        out[f"prf_{m}_{alpha}"] = col
        print(f"prf m={m} alpha={alpha} done", flush=True)
    out.to_parquet(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
