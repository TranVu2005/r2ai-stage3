"""S43 offline: cosine weight in the RRF (w_cos / (60 + cosine_rank) + 1 / (60 + judge_rank) + 1 / (60 + cmax_rank)).

(a) carriers only: docs = CH-rrf docs, per-query carrier count of CH-rrf, carriers = top n_q of the weighted order inside the docsel;
(b) docs: top k_q (k_q of CH-rrf) of the weighted order over the 300 pool, carriers as (a) on the new docs.
Reports swaps and the scores of the docs moved in / out, plus the mean cosine rank of the carrier set (LB pattern S41: lower is better).

    python scripts/s43/cosweight.py --base out/runs/s43/base
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze

COLS = ("cosine_rank", "p4b", "cmax", "p_cal", "jina", "bgr_doc", "bgr_chk")


def weighted(df: pd.DataFrame, w: float, mask: pd.Series | None = None) -> pd.Series:
    s = w / (60 + df.cosine_rank) + 1 / (60 + df.judge_rank) + 1 / (60 + df.cmax_rank) - 1e-12 * df.pos
    if mask is not None:
        s = s.where(mask, -np.inf)
    return s.groupby(df.query_id).rank(ascending=False, method="first")


def stats(x: pd.DataFrame) -> dict:
    return {c: round(float(x[c].mean()), 4) for c in COLS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.base / "pool.parquet")
    for name, f, col in (("jina", analyze.RUNS / "s35-part-jina/jina/scores-all.parquet", "score"),
                         ("jina_x", analyze.RUNS / "s36-fuse4/extra/jina-scores.parquet", "score"),
                         ("bgr_doc", analyze.RUNS / "s33-btc-baseline/gpu-rdoc/bge-reranker-v2-m3/scores-all.parquet", "score")):
        s = pd.read_parquet(f, columns=["query_id", "doc_id", col]).drop_duplicates(["query_id", "doc_id"]).rename(columns={col: name})
        df = df.merge(s, on=["query_id", "doc_id"], how="left")
    df["jina"] = df.jina.fillna(df.jina_x)
    bc = pd.read_parquet(analyze.RUNS / "s33-btc-baseline/gpu-rchk/bge-reranker-v2-m3/scores-all.parquet", columns=["query_id", "doc_id", "score"])
    df = df.merge(bc.groupby(["query_id", "doc_id"]).score.max().rename("bgr_chk").reset_index(), on=["query_id", "doc_id"], how="left")
    keep = df[df.chosen].groupby("query_id").size()
    ncar = df[df.carrier].groupby("query_id").size().reindex(keep.index, fill_value=0)
    n = len(keep)
    res = {"base_carriers": stats(df[df.carrier]), "base_docs": stats(df[df.chosen]), "carriers_only": {}, "docs": {}}
    chk = weighted(df, 1.0, df.chosen) <= df.query_id.map(ncar)
    res["check_w1_carriers_equal_ch_rrf"] = bool((chk == df.carrier).all())
    for w in (1.5, 2.0, 3.0, 5.0, 100.0):
        car = weighted(df, w, df.chosen) <= df.query_id.map(ncar)
        res["carriers_only"][str(w)] = {"swapped_per_query": round(float((car & ~df.carrier).sum()) / n, 2), "set": stats(df[car]),
                                        "in": stats(df[car & ~df.carrier]), "out": stats(df[df.carrier & ~car])}
        pos = weighted(df, w)
        docs = pos <= df.query_id.map(keep)
        car2 = weighted(df, w, docs) <= df.query_id.map(ncar)
        res["docs"][str(w)] = {"docs_swapped_per_query": round(float((docs & ~df.chosen).sum()) / n, 2), "docs_in": stats(df[docs & ~df.chosen]),
                               "docs_out": stats(df[df.chosen & ~docs]), "carriers_swapped_per_query": round(float((car2 & ~df.carrier).sum()) / n, 2),
                               "carrier_set": stats(df[car2])}
    (args.base / "cosweight.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
