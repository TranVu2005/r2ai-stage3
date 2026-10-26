"""S40 (26/10): refit the global doc model (a, b, s_free) of the free frame on every aggregate LB Doc number of the free-frame submissions.

    python scripts/s40/refit_pcal.py [--lb lb.json] [--out out/runs/s40/recal/refit.json]

Model (unchanged, r2ai.s36.cal): p_cal = sigmoid(a logit(P4B) + b); G_q = s (sum p_cal over the cosine top-211 + geometric tail);
per query tp = sum p_cal over the submitted docs, P = tp / k, R = tp / G_q, F2 = 5 tp / (4 G_q + k); macro mean over the 1,200 queries.
G_q is a property of the query, so ONE G is used for every submission: the cosine top-211 of the current corpus (canonical + Pharmacity +
120ask, ``r2ai.s36.d300.merged_top``) with the P4B of the FREE-noJina inputs (pcal-ABL-noJina-free-v2: p_fused = P4B over the top-300 pools).
The submitted docs carry their own P4B (docsel ``p4b``). Only aggregate LB numbers enter the fit (no per-query label or score).
Least squares over every LB number given (Doc F2, and Doc P / R where known), equal weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s36 import cal, d300

RUNS = paths.RUNS_DIR
DOCSEL = {
    "EF2-free": RUNS / "s36-fuse4" / "docsel" / "docsel-EF2-free.parquet",          # CR-fair has this docsel too (same Doc numbers)
    "FUSE4-free": RUNS / "s36-fuse4" / "docsel" / "docsel-FUSE4-free.parquet",
    "120A-free-v0": RUNS / "s36-fuse4" / "s37" / "docsel-120A-p4b.parquet",        # 120A-free-v0 was named 120A-p4b when built
    "120A-free-v2": RUNS / "release" / "120A-free-v2" / "apply" / "docsel-120A-free-v2.parquet",
    "FREE-noJina": RUNS / "s40" / "abl" / "apply" / "docsel-ABL-noJina-free-v2.parquet",
    "FREE-noCmax": RUNS / "s40" / "abl" / "apply" / "docsel-ABL-noCmax-free-v2.parquet",
}
G_SOURCE = RUNS / "s40" / "abl" / "apply" / "pcal-ABL-noJina-free-v2.parquet"
# aggregate LB Doc numbers reported by Vu (docs/s36-fuse4.md, docs/s40-private.md); None = not reported
LB = {
    "EF2-free": {"f2": 0.4184, "p": 0.226, "r": 0.6076},
    "FUSE4-free": {"f2": 0.4104},
    "120A-free-v0": {"f2": 0.4344},
    "120A-free-v2": {"f2": 0.4402},
    "FREE-noCmax": {"f2": 0.4370},
}
OLD = {"a": 0.5298075151496865, "b": -1.40969757276664, "s": 1.7405181516323944}


def g_base() -> tuple[list[int], list[np.ndarray]]:
    """Per query, logit(P4B) of the cosine top-211 of the corpus, in cosine order."""
    top = d300.merged_top(cal.FREE_G_DEPTH)
    pc = pd.read_parquet(G_SOURCE, columns=["query_id", "doc_id", "p_fused"])
    m = top.merge(pc, on=["query_id", "doc_id"], how="left")
    if m.p_fused.isna().any():
        raise SystemExit(f"{int(m.p_fused.isna().sum())} top-211 pairs without P4B")
    m = m.sort_values(["query_id", "cosine_rank"])
    qids = sorted(m.query_id.unique().tolist())
    return qids, [cal.logit(g.p_fused.to_numpy()) for _, g in m.groupby("query_id", sort=True)]


def load_sets(qids: list[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pos = {q: i for i, q in enumerate(qids)}
    out = {}
    for name, f in DOCSEL.items():
        d = pd.read_parquet(f, columns=["query_id", "doc_id", "p4b"])
        if d.p4b.isna().any() or set(d.query_id) != set(qids):
            raise SystemExit(f"{name}: P4B missing or query set differs")
        out[name] = (d.query_id.map(pos).to_numpy(), cal.logit(d.p4b.to_numpy()))
    return out


def predict(theta, base, sets) -> dict[str, dict[str, float]]:
    a, b, s = theta
    g = np.array([s * (p.sum() + cal.tail_mass(p)) for p in (cal.sigmoid(a * x + b) for x in base)])
    out = {}
    for name, (qi, x) in sets.items():
        tp = np.bincount(qi, weights=cal.sigmoid(a * x + b), minlength=len(g))
        k = np.bincount(qi, minlength=len(g))
        out[name] = {"p": float(np.mean(tp / k)), "r": float(np.mean(tp / g)), "f2": float(np.mean(5 * tp / (4 * g + k)))}
    return out


def residuals(theta, base, sets, lb) -> np.ndarray:
    pred = predict(theta, base, sets)
    return np.array([pred[n][m] - v for n, ms in lb.items() for m, v in ms.items() if v is not None])


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb", type=Path, help="JSON {name: {f2, p, r}} replacing the built-in LB table")
    ap.add_argument("--out", type=Path, default=RUNS / "s40" / "recal" / "refit.json")
    args = ap.parse_args()
    lb = json.loads(args.lb.read_text(encoding="utf-8")) if args.lb else LB
    qids, base = g_base()
    sets = load_sets(qids)
    theta0 = np.array([OLD["a"], OLD["b"], OLD["s"]])
    fits = [least_squares(residuals, np.array(t), args=(base, sets, lb), bounds=([0.05, -10, 0.3], [20, 10, 10]))
            for t in (theta0, (1.0, 0.0, 1.5), (0.3, -3.0, 2.0), (2.0, 2.0, 1.0))]
    best = min(fits, key=lambda r: r.cost)
    n_obs = int(sum(v is not None for ms in lb.values() for v in ms.values()))
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (2 * best.cost / max(n_obs - 3, 1))
        se = np.sqrt(np.diag(cov)).tolist()
    except np.linalg.LinAlgError:
        se = None
    rows = {}
    for tag, theta in (("old", theta0), ("new", best.x)):
        pred = predict(theta, base, sets)
        rows[tag] = {n: {**{m: round(v, 5) for m, v in pred[n].items()},
                         **{f"err_{m}": round(pred[n][m] - v, 5) for m, v in lb.get(n, {}).items() if v is not None}} for n in pred}
        r = residuals(theta, base, sets, lb)
        rows[tag + "_rmse"], rows[tag + "_max_abs"] = float(math.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r)))
    result = {"lb": lb, "n_obs": n_obs, "old": OLD, "new": dict(zip(("a", "b", "s"), map(float, best.x))), "se_new": se,
              "jacobian_rank": int(np.linalg.matrix_rank(J)), "fits": rows,
              "inputs": {**{n: _sha256(f) for n, f in DOCSEL.items()}, "g_source": _sha256(G_SOURCE)}}
    paths.assert_writable(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out, result)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, ensure_ascii=False))
