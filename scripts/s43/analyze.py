"""S43 offline analysis of query-set signals against the judges (no LB, no gold).

1. Per-query Spearman of each first-stage signal with P4B and C-max (mean over queries), against raw cosine.
2. RRF variants: cosine source replaced by / added with a signal; docs = top k_q (k_q of CH-rrf), carriers = top ceil(rho k_q) of the new order
   (rho = CH-rrf carriers / docs). Reports docs / carriers swapped and the judges' scores of the docs swapped in and out.

    python scripts/s43/analyze.py --base out/runs/s43/base
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RRF_K = 60
HELD = ("jina", "bgr_doc", "bgr_chk")    # judges outside CH-rrf (NaN where unscored; mean over scored)
RUNS = Path("D:/GitHub/r2ai-stage3/out/runs")


def per_query_spearman(df: pd.DataFrame, a: str, b: str) -> float:
    return float(np.nanmean([spearmanr(g[a], g[b])[0] for _, g in df.groupby("query_id")]))


def rank_in_query(df: pd.DataFrame, col: str, ascending=False) -> pd.Series:
    return df.groupby("query_id")[col].rank(ascending=ascending, method="first")


def variant(df: pd.DataFrame, rank_cols: list[str], keep: pd.Series, ncar: pd.Series) -> pd.DataFrame:
    """New RRF order; docs = top k_q, carriers = top n_q (k_q, n_q of CH-rrf per query). Ties: old RRF position."""
    rrf = sum(1.0 / (RRF_K + df[c]) for c in rank_cols)
    d = df.assign(new_rrf=rrf - 1e-12 * df.pos)
    d["new_pos"] = d.groupby("query_id").new_rrf.rank(ascending=False, method="first")
    d["new_chosen"] = d.new_pos <= d.query_id.map(keep)
    d["new_carrier"] = d.new_pos <= d.query_id.map(ncar)
    return d


def swap_report(d: pd.DataFrame, flag: str, new_flag: str) -> dict:
    inn, out = d[d[new_flag] & ~d[flag]], d[d[flag] & ~d[new_flag]]
    n = len(d.query_id.unique())
    f = lambda x: {c: round(float(x[c].mean()), 4) for c in ("p4b", "cmax", "p_cal", "cosine") + HELD} if len(x) else {}
    return {"swapped_per_query": round(len(inn) / n, 2), "in": f(inn), "out": f(out),
            "delta_sum_p_cal_per_query": round(float((inn.p_cal.sum() - out.p_cal.sum()) / n), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    pool = pd.read_parquet(args.base / "pool.parquet")
    sig = pd.read_parquet(args.base / "signals.parquet").drop(columns=["cosine"])
    df = pool.merge(sig, on=["query_id", "doc_id"], validate="1:1")
    res: dict = {"spearman": {}}
    jina = pd.concat([pd.read_parquet(RUNS / "s35-part-jina/jina/scores-all.parquet", columns=["query_id", "doc_id", "score"]),
                      pd.read_parquet(RUNS / "s36-fuse4/extra/jina-scores.parquet", columns=["query_id", "doc_id", "score"])])
    df = df.merge(jina.drop_duplicates(["query_id", "doc_id"]).rename(columns={"score": "jina"}), on=["query_id", "doc_id"], how="left")
    bd = pd.read_parquet(RUNS / "s33-btc-baseline/gpu-rdoc/bge-reranker-v2-m3/scores-all.parquet", columns=["query_id", "doc_id", "score"])
    df = df.merge(bd.drop_duplicates(["query_id", "doc_id"]).rename(columns={"score": "bgr_doc"}), on=["query_id", "doc_id"], how="left")
    bc = pd.read_parquet(RUNS / "s33-btc-baseline/gpu-rchk/bge-reranker-v2-m3/scores-all.parquet", columns=["query_id", "doc_id", "score"])
    df = df.merge(bc.groupby(["query_id", "doc_id"]).score.max().rename("bgr_chk").reset_index(), on=["query_id", "doc_id"], how="left")
    res["coverage"] = {h: round(float(df[h].notna().mean()), 4) for h in HELD}
    print("coverage", res["coverage"], flush=True)
    df["neg_rr"] = -df.rr
    prf_cols = []
    if (args.base / "prf.parquet").exists():
        prf = pd.read_parquet(args.base / "prf.parquet")
        prf_cols = [c for c in prf.columns if c.startswith("prf_")]
        df = df.merge(prf, on=["query_id", "doc_id"], validate="1:1")
    df["neg_hub"] = -df.hub300
    df["cos_div_hub"] = df.cosine - 0.01 * np.log(df.hub300)
    for s in ("cosine", "csls10", "csls20", "neg_rr", "neg_hub", "p4b", "cmax", *prf_cols):
        res["spearman"][s] = {j: round(per_query_spearman(df.dropna(subset=[j]), s, j), 4) for j in ("p4b", "cmax") + HELD}
        print(s, res["spearman"][s], flush=True)
    keep = df[df.chosen].groupby("query_id").size()
    ncar = df[df.carrier].groupby("query_id").size().reindex(keep.index, fill_value=0)
    df["random_rank"] = np.random.default_rng(43).permutation(len(df)) % 300 + 1
    for s in ("csls5", "csls10", "csls20", "neg_rr", *prf_cols):
        df[f"{s}_rank"] = rank_in_query(df, s)
    extra = {f"replace_cos_{c}": [f"{c}_rank", "judge_rank", "cmax_rank"] for c in prf_cols}
    base = ["cosine_rank", "judge_rank", "cmax_rank"]
    chk = variant(df, base, keep, ncar)
    res["check_base"] = {"docs_equal": bool((chk.new_chosen == df.chosen).all()), "carriers_equal": bool((chk.new_carrier == df.carrier).all())}
    print("check", res["check_base"], flush=True)
    res["variants"] = {}
    for name, cols in {"replace_cos_csls10": ["csls10_rank", "judge_rank", "cmax_rank"],
                       "replace_cos_csls5": ["csls5_rank", "judge_rank", "cmax_rank"],
                       "replace_cos_csls20": ["csls20_rank", "judge_rank", "cmax_rank"],
                       "add_csls10": base + ["csls10_rank"],
                       "add_rr": base + ["neg_rr_rank"],
                       "replace_cos_rr": ["neg_rr_rank", "judge_rank", "cmax_rank"],
                       "control_replace_cos_random": ["random_rank", "judge_rank", "cmax_rank"],
                       "control_drop_cos": ["judge_rank", "cmax_rank"], **extra}.items():
        d = variant(df, cols, keep, ncar)
        res["variants"][name] = {"docs": swap_report(d, "chosen", "new_chosen"), "carriers": swap_report(d, "carrier", "new_carrier")}
        print(name, json.dumps(res["variants"][name]), flush=True)
    (args.base / "analyze.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
