"""S41 / S42 hypothesis (Vu 27/10): split the number of docs between queries by a learned K_q instead of the EF2 cut; all else CH-rrf.

    python -m r2ai.s41.kfit fit   --out out/runs/s41/kfit          # features, 5-fold CV (ridge, small GBM), out-of-fold K_hat
    python -m r2ai.s41.kfit build --out out/runs/s41/kfit          # FREE-Kfit docsel + carriers (refused when the best CV R2 < 0.2)
    python -m r2ai.s41.kfit report --out out/runs/s41/kfit         # per-fold R2 / Spearman, held-out permutation importance

Target: K_q = len(relevant_docs) of P1-vi+10 for the 1,200 public queries (reads public files ON PURPOSE; the private run would use the
fitted model on its own free frame). Features come only from the free frame a new query set also has (the FREE-noJina pool top-300 in
RRF order: P4B, cosine, C-max, p_cal; zh share of the pool; query length; the EF2 keep). CV: 5 folds over queries (KFold, shuffled,
seed 41); R2 / Pearson / Spearman of the out-of-fold predictions against K_q.

FREE-Kfit: k_q = clip(round(c K_hat_q), 1, |pool_q|) with the out-of-fold K_hat (no query predicts itself) and c solved so the total equals
CH-rrf's 220,436 docs (183.7 per query); docs = top k_q of the RRF order (``newq.apply_selection`` with ``keep_fn``), carriers CR-fair in
RRF order with the CH-rrf total 175,548 (config ``configs/s41-private-final.json``), then S38 compose full+w768 (``scripts/s41/run_kfit.sh``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import newq, paths

ROOT = Path(__file__).resolve().parents[3]
P1_ZIP = paths.RUNS_DIR / "reranker-trial" / "release" / "P1-vi+10.zip"
CH_RRF_DOCS = 220_436
SEED = 41
NAME = "FREE-Kfit"
P_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
RANKS = (10, 25, 50, 100, 150, 200, 250, 300)


def _regress_module():
    spec = importlib.util.spec_from_file_location("s41_regress_public", ROOT / "scripts" / "s41" / "regress_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_frame() -> dict[str, Any]:
    """``newq.select`` with the frozen config on the FREE-noJina pools / 4B / C-max (1,200 public queries), and the source spec."""
    reg = _regress_module()
    spec = {**reg.s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
    pools, cmax = reg.public_sources(spec)
    cfg = newq.load_config(None)
    keys = ("cosine_rank", "judge_rank", "cmax_rank")
    return {"pools": pools, "cmax": cmax, "cfg": cfg, "keys": keys, "res": newq.select(pools, {}, cmax, cfg, keys, True)}


def doc_lang(doc_ids: set[int]) -> dict[int, str]:
    """Canonical lang; a doc outside canonical is a 120ask segment doc (zh) or Pharmacity (vi)."""
    from r2ai.s37 import ingest

    canon = pd.read_parquet(paths.DATA_DIR / "canonical-20261018" / "docs.parquet", columns=["doc_id", "lang"])
    canon = canon[canon.doc_id.isin(doc_ids)]
    out = dict(zip(canon.doc_id.astype(int), canon.lang.astype(str)))
    ask = set()
    for seg in sorted(p for p in ingest.STORE.iterdir() if p.is_dir() and (p / "docs.parquet").exists()):
        ask |= set(pd.read_parquet(seg / "docs.parquet", columns=["doc_id"]).doc_id.astype(int)) & doc_ids
    for d in doc_ids - set(out):
        out[d] = "zh" if d in ask else "vi"
    return out


def _at(values: np.ndarray, ranks: Sequence[int]) -> list[float]:
    return [float(values[min(r, len(values)) - 1]) for r in ranks]


def query_features(rows: list[dict[str, Any]], p: np.ndarray, keep: int, lang: dict[int, str], text: str) -> dict[str, float]:
    p4b = np.array([r["judge_score"] for r in rows], float)
    cos = np.array([r["cosine_score"] for r in rows], float)
    cmx = np.array([r.get("cmax_score", np.nan) for r in rows], float)
    cmx = np.where(np.isfinite(cmx), cmx, 0.0)
    zh = np.array([lang[int(r["doc_id"])] == "zh" for r in rows], float)
    f: dict[str, float] = {"ef2_keep": keep, "pool": len(rows), "query_chars": len(text), "query_words": len(text.split()),
                           "zh_top50": zh[:50].mean(), "zh_top100": zh[:100].mean(), "zh_pool": zh.mean(),
                           "pcal_sum": p.sum(), "pcal_sum100": p[:100].sum(), "pcal_mean": p.mean()}
    for name, v in (("pcal", np.sort(p)[::-1]), ("p4b", np.sort(p4b)[::-1]), ("cos", np.sort(cos)[::-1]), ("cmax", np.sort(cmx)[::-1])):
        for r, x in zip(RANKS, _at(v, RANKS)):
            f[f"{name}_at{r}"] = x
        f[f"{name}_slope"] = _at(v, (25,))[0] - _at(v, (150,))[0]
    for t in P_THRESHOLDS:
        f[f"n_pcal_ge{t}"] = float((p >= t).sum())
        f[f"n_p4b_ge{t}"] = float((p4b >= t).sum())
        f[f"n_cmax_ge{t}"] = float((cmx >= t).sum())
    f["cos_gap"] = float(cos.max() - cos.mean())
    return f


def features(ff: dict[str, Any]) -> pd.DataFrame:
    res = ff["res"]
    frame, calibrated, keep = res["frame"], res["calibrated"], res["keep"]
    texts = pd.read_parquet(paths.QUERY_FILE, columns=["id", "query"])
    text = dict(zip(texts.id.astype(int), texts["query"].astype(str)))
    lang = doc_lang({int(r["doc_id"]) for q in frame.qids for r in frame.order[q]})
    rows = [{"query_id": q, **query_features(frame.order[q], np.asarray(calibrated.p[q], float), int(keep[q]), lang, text[q])}
            for q in frame.qids]
    return pd.DataFrame(rows)


def p1_k() -> pd.Series:
    from r2ai.s38.sources import read_bundle

    return pd.Series({int(r["id"]): len(r["relevant_docs"]) for r in read_bundle(P1_ZIP)}, name="k_q")


def models() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {"ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
            "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=20, random_state=SEED)}


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    return {"r2": float(r2_score(y, pred)), "pearson": float(np.corrcoef(y, pred)[0, 1]), "spearman": float(spearmanr(y, pred).statistic)}


def cross_validate(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from sklearn.model_selection import KFold

    folds = list(KFold(5, shuffle=True, random_state=SEED).split(x))
    report, oof = {}, {}
    for name, make in models().items():
        pred = np.zeros(len(y))
        per_fold = []
        for train, test in folds:
            m = make().fit(x[train], y[train])
            pred[test] = m.predict(x[test])
            per_fold.append(scores(y[test], pred[test])["r2"])
        oof[name] = pred
        report[name] = {**scores(y, pred), "r2_per_fold": per_fold}
    return report, oof


def fit(out: Path) -> dict[str, Any]:
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    ff = free_frame()
    feat = features(ff)
    k = p1_k()
    feat = feat[feat.query_id.isin(k.index)].reset_index(drop=True)
    y = k.loc[feat.query_id].to_numpy(float)
    cols = [c for c in feat.columns if c != "query_id"]
    x = feat[cols].to_numpy(float)
    report, oof = cross_validate(x, y)
    best = max(report, key=lambda n: report[n]["r2"])
    model = models()[best]().fit(x, y)
    feat["k_q"], feat["k_hat_oof"] = y, oof[best]
    for name, pred in oof.items():
        feat[f"oof_{name}"] = pred
    newq.atomic_parquet(feat, out / "features.parquet")
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "columns": cols, "kind": best}, f)
    res = {"queries": len(y), "features": len(cols), "target": f"len(relevant_docs) of {P1_ZIP.name}",
           "k_q_mean_p10_p50_p90": [float(y.mean())] + [float(np.quantile(y, q)) for q in (.1, .5, .9)],
           "cv": report, "best": best, "baseline_ef2_keep": scores(y, feat.ef2_keep.to_numpy(float)),
           "go": report[best]["r2"] >= 0.2, "model_sha256": newq.sha256(out / "model.pkl"), "features_sha256": newq.sha256(out / "features.parquet")}
    if best == "ridge":
        coef = model[-1].coef_
        res["ridge_top_coef"] = dict(sorted(zip(cols, map(float, coef)), key=lambda kv: -abs(kv[1]))[:10])
        res["ridge_alpha"] = float(model[-1].alpha_)
    newq.atomic_json(out / "fit.json", res)
    return res


def report(out: Path) -> dict[str, Any]:
    """Per-fold R2 / Spearman of the out-of-fold K_hat (same folds as ``fit``) and the feature importance measured on the held-out fold
    (permutation, 5 repeats, mean over folds), for both models."""
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    feat = pd.read_parquet(out / "features.parquet")
    with open(out / "model.pkl", "rb") as f:
        cols = pickle.load(f)["columns"]
    x, y = feat[cols].to_numpy(float), feat.k_q.to_numpy(float)
    res: dict[str, Any] = {}
    for name, make in models().items():
        folds, imp = [], np.zeros(len(cols))
        for train, test in KFold(5, shuffle=True, random_state=SEED).split(x):
            m = make().fit(x[train], y[train])
            pred = m.predict(x[test])
            folds.append({"r2": float(r2_score(y[test], pred)), "spearman": float(spearmanr(y[test], pred).statistic)})
            imp += permutation_importance(m, x[test], y[test], n_repeats=5, random_state=SEED).importances_mean / 5
        top = sorted(zip(cols, imp), key=lambda kv: -kv[1])[:10]
        res[name] = {"folds": folds, "r2_mean": float(np.mean([f["r2"] for f in folds])), "spearman_mean": float(np.mean([f["spearman"] for f in folds])),
                     "oof": scores(y, feat[f"oof_{name}"].to_numpy(float)),
                     "importance_top10": [{"feature": c, "r2_drop": float(v), "corr_k_q": float(np.corrcoef(feat[c], y)[0, 1])} for c, v in top]}
    newq.atomic_json(out / "report.json", res)
    return res


def solve_c(k_hat: dict[int, float], cap: dict[int, int], total: int) -> tuple[float, dict[int, int]]:
    """c whose clip(round(c k_hat), 1, cap) sums closest to ``total`` (the sum grows with c; bisection)."""
    keep = lambda c: {q: int(min(cap[q], max(1, round(c * v)))) for q, v in k_hat.items()}     # noqa: E731
    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(keep(mid).values()) < total else (lo, mid)
    c = min((lo, hi), key=lambda v: abs(sum(keep(v).values()) - total))
    return c, keep(c)


def build(out: Path, name: str, force: bool) -> dict[str, Any]:
    fitted = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    if not fitted["go"] and not force:
        raise SystemExit(f"best CV R2 {fitted['cv'][fitted['best']]['r2']:.3f} < 0.2: no build (Vu 27/10)")
    feat = pd.read_parquet(out / "features.parquet")
    k_hat = dict(zip(feat.query_id.astype(int), feat.k_hat_oof.astype(float)))
    ff = free_frame()
    info: dict[str, Any] = {}

    def keep_fn(res):
        cap = {q: len(res["frame"].order[q]) for q in res["frame"].qids}
        c, keep = solve_c(k_hat, cap, CH_RRF_DOCS)
        info.update({"c": c, "ef2_expected": res["calibrated"].expected(res["keep"]), "kfit_expected": res["calibrated"].expected(keep)})
        return keep

    apply_dir = out / "apply"
    apply_dir.mkdir(parents=True, exist_ok=True)
    got = newq.apply_selection(ff["pools"], {}, ff["cmax"], ff["cfg"], ff["keys"], True, name, apply_dir, keep_fn=keep_fn)
    from r2ai.s36 import cal

    keep = got["keep"]
    ef2 = dict(zip(feat.query_id.astype(int), feat.ef2_keep.astype(int)))
    res = {"name": name, "model": fitted["best"], "cv_r2": fitted["cv"][fitted["best"]]["r2"], **info, "docs_per_query": cal.keep_stats(keep),
           "vs_ef2_keep": {"pearson": float(np.corrcoef([keep[q] for q in ef2], [ef2[q] for q in ef2])[0, 1]),
                           "mean_abs_diff": float(np.mean([abs(keep[q] - ef2[q]) for q in ef2]))},
           "carriers": {**got["carrier_info"], "count": int(len(got["carriers"]))},
           "outputs": {p.name: newq.sha256(p) for p in sorted(apply_dir.glob(f"*{name}*")) if p.is_file()}}
    newq.atomic_json(out / f"build-{name}.json", res)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("fit", "build", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--force", action="store_true", help="build even when the CV R2 < 0.2 (not without Vu)")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    res = fit(out) if args.step == "fit" else report(out) if args.step == "report" else build(out, args.name, args.force)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
