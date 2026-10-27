"""S36 doc side: global probability calibration, EF2-cut and LANG-thr on the J3-thrC frame (CPU).

``python -m r2ai.s36.cal run``  fits p_cal for P_fused = P4B and P_fused = mean(P4B, Jina mapped to the P4B quantiles), selects
``EF2-cut-p4b``, ``EF2-cut-avg``, ``LANG-thr-avg`` (and the degenerate ``LANG-thr-p4b``), writes ``cal/cal.json``,
``cal/keep-<name>.json`` (docs per query), ``cal/pcal-<name>.parquet`` and ``docsel/docsel-J3-thrC.parquet``.

Frame = J3-thrC: pool13 of K4-thr-C (round(1.3 K_q) after Pharmacity 979), order RRF k=60 over cosine, 4B, Jina.

* ``P_fused``: ``p4b`` = the Qwen3-Reranker-4B P(yes); ``avg`` = (P4B + Jina') / 2, Jina' = the Jina score mapped to the P4B value of
  the same global quantile over the pool13 pairs (monotone, so ranks are unchanged).
* ``p_cal = sigmoid(a logit(P_fused) + b)``. (a, b) minimise the squared error between the macro mean of p_cal (mean over the
  1,200 queries of the per-query mean over the submitted docs) and the LB Doc P of J4-K40 (0.376), K4-thr-C (0.2346), J3-thrC (0.2357).
* ``G_q`` (expected gold docs of a query) = sum of p_cal over the pool + a tail beyond the pool depth: the mean p_cal of the last 10 pool
  docs in cosine order decays geometrically, with the decay rate fitted on the last 60 docs and bounded below (tail <= 50 docs of
  the last mean).
* ``G_q`` is also scaled by a global factor s fitted so that the predicted macro Doc F2 of J3-thrC equals its LB value (0.4161): the
  pool misses gold docs (LB Doc R about 0.51 against 0.79 predicted from the pool). The K4-thr-C prediction (LB 0.4139) checks s.
  ``anchored`` (s fitted) builds the bundles; ``literal`` (s = 1) is reported.
* EF2-cut: top-k of the RRF order maximising 5 sum_{i<=k} p_cal / (4 G_q + k), k in [min(20, depth), depth].
* LANG-thr: docs per query = clip(#(P4B >= t_lang), ceil(0.5 K_q), floor(1.3 K_q)) taken in RRF order, as J3-thrC; t_vi and t_zh are
  the P4B values where the marginal precision (mean p_cal of the language's pool docs within +-5% of the language's P4B quantile)
  falls to the same c, and c is the fixed point c = F2 / 5 of the predicted macro F2 of the resulting selection (the F2-optimal
  cut-off probability is F2 / 5).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _inputs, _load_submission
from r2ai.s31 import build as s31_build
from r2ai.s31.select import RRF_K, doc_set_diff, select_rrf
from r2ai.s36 import fuse

J3_KEYS = ("cosine_rank", "judge_rank", "jina_rank")
EPS = 1e-6
FLOOR_K = 20
TAIL_LAST = 10
TAIL_FIT = 60
TAIL_MAX_DOCS = 50.0
BAND = 0.05
LB_DOC_P = {"J4-K40": 0.376, "K4-thr-C": 0.2346, "J3-thrC": 0.2357}
LB_DOC_F2 = {"K4-thr-C": 0.4139, "J3-thrC": 0.4161}  # S35 / BTC report of 23/10
VARIANTS = ("p4b", "avg")


def cal_root() -> Path:
    return fuse.run_root() / "cal"


def submitted_zips() -> dict[str, Path]:
    runs = paths.RUNS_DIR
    return {"J4-K40": runs / "s31-strong-judge" / "build-topk" / "J4-K40.zip",
            "K4-thr-C": runs / "s31-strong-judge" / "build-k4b" / "K4-thr-C.zip",
            "J3-thrC": runs / "s35-part-jina" / "build-j3" / "J3-thrC.zip"}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Frame:
    """pool13 rows of every query in J3 RRF order, with P4B, Jina, language and the lookups of the scored union."""

    def __init__(self, keys: Sequence[str] = J3_KEYS, frame: str = "thrC", data: dict[str, Any] | None = None,
                 allow_missing: bool = False) -> None:
        """``frame`` thrC: the pool13 rows; free: the K4-free pools (cosine top-211, deepened); free211: the same pools cut at the
        cosine top-211 for every query (no deepening). ``keys``: the RRF rank sources. ``data`` (S37 apply) replaces the loaded S36
        inputs: ``pools`` (query -> rows with cosine_rank / judge_score), ``union``, ``jina`` and ``cmax`` lookups.
        ``allow_missing``: a pair without a Jina / C-max score takes the last rank of its pool in that source (counted in ``self.missing``)."""
        if data is None:
            ranked, union, free, _, _, _ = fuse.load_frames(deepen=frame != "free211")
            jina, _ = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
            cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
            pools = ranked if frame == "thrC" else free
        else:
            ranked, union, jina, cmax, pools = data.get("ranked", {}), data["union"], data["jina"], data["cmax"], data["pools"]
        self.keys, self.frame_name = tuple(keys), frame
        self.ranked = ranked
        self.p4b_of = {(q, int(r["doc_id"])): float(r["judge_score"]) for q, rows in union.items() for r in rows}
        self.jina, self.cmax = jina, cmax
        self.missing = {"jina": 0, "cmax": 0, "queries_jina": 0, "queries_cmax": 0}
        for q, rows in pools.items():
            for source, lookup, score_key, rank_key in (("jina", self.jina, "jina_score", "jina_rank"), ("cmax", self.cmax, "cmax_score", "cmax_rank")):
                n = fuse.rank_missing_last(rows, q, lookup, score_key, rank_key)
                if n and not allow_missing:
                    raise ValueError(f"query {q}: pair without {'Jina' if source == 'jina' else 'C-max'} score")
                self.missing[source] += n
                self.missing[f"queries_{source}"] += bool(n)
        self.pools = pools
        inputs = _inputs()
        self.base_k = {int(r["id"]): len(r["relevant_docs"]) for r in _load_submission(inputs["baseline"])}
        docs = pq.read_table(inputs["canonical"] / "docs.parquet", columns=["doc_id", "lang"]).to_pandas()
        self.lang = dict(zip(docs.doc_id.astype(int), docs.lang))
        self.order = {q: select_rrf(rows, len(rows), self.keys, RRF_K) for q, rows in pools.items()}  # full RRF order
        self.qids = sorted(pools)
        pairs = [(q, r) for q in self.qids for r in self.order[q]]
        self.p4b13 = np.array([float(r["judge_score"]) for _, r in pairs])
        self.jina13 = np.array([float(r["jina_score"]) if r["jina_score"] is not None else np.nan for _, r in pairs])
        scored = ~np.isnan(self.jina13)
        self.sorted_jina, self.sorted_p4b = np.sort(self.jina13[scored]), np.sort(self.p4b13[scored])

    def jina_prime(self, jina: np.ndarray) -> np.ndarray:
        """Jina score -> P4B value of the same global quantile (pool13 pairs); monotone."""
        return np.interp(jina, self.sorted_jina, self.sorted_p4b)

    def fused(self, variant: str, p4b: np.ndarray, jina: np.ndarray) -> np.ndarray:
        return p4b if variant == "p4b" else 0.5 * (p4b + self.jina_prime(jina))

    def doc_lang(self, doc_id: int) -> str:
        return self.lang.get(doc_id, "vi")  # Pharmacity docs are not in docs.parquet


def set_arrays(frame: Frame, docs_by_query: dict[int, list[int]]) -> dict[str, np.ndarray]:
    q, p4b, jina = [], [], []
    for qid, ids in docs_by_query.items():
        for d in ids:
            q.append(qid)
            p4b.append(frame.p4b_of[(qid, d)])
            jina.append(frame.jina[(qid, d)])
    return {"q": np.array(q), "p4b": np.array(p4b), "jina": np.array(jina)}


def macro_mean(q: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """(mean over queries of the per-query mean, mean over all docs)."""
    _, inverse = np.unique(q, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    return float((sums / counts).mean()), float(values.mean())


def fit_calibration(frame: Frame, variant: str, sets: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    x = {name: logit(frame.fused(variant, s["p4b"], s["jina"])) for name, s in sets.items()}

    def residuals(theta: np.ndarray) -> np.ndarray:
        a, b = theta
        return np.array([macro_mean(sets[n]["q"], sigmoid(a * x[n] + b))[0] - LB_DOC_P[n] for n in sets])

    best = min((least_squares(residuals, np.array([a0, b0]), bounds=([0.05, -10.0], [20.0, 10.0])) for a0 in (0.5, 1.0, 2.0) for b0 in (-2.0, 0.0, 2.0)),
               key=lambda r: r.cost)
    a, b = map(float, best.x)
    fits = {}
    for n in sets:
        macro, micro = macro_mean(sets[n]["q"], sigmoid(a * x[n] + b))
        fits[n] = {"lb_doc_p": LB_DOC_P[n], "predicted_macro": macro, "error": macro - LB_DOC_P[n], "predicted_micro": micro, "docs": int(len(x[n]))}
    return {"variant": variant, "a": a, "b": b, "max_abs_error": max(abs(f["error"]) for f in fits.values()),
            "rmse": float(math.sqrt(np.mean([f["error"] ** 2 for f in fits.values()]))), "sets": fits}


def tail_mass(p_cos_order: np.ndarray) -> float:
    """Expected gold docs beyond the pool depth, from the decay of p_cal over the last pool docs (cosine order)."""
    n = len(p_cos_order)
    if n < TAIL_FIT:
        return 0.0
    blocks = p_cos_order[n - TAIL_FIT:].reshape(-1, TAIL_LAST).mean(axis=1) + 1e-9
    slope = np.polyfit(np.arange(len(blocks)), np.log(blocks), 1)[0]  # per block of TAIL_LAST docs
    lam = max(-slope / TAIL_LAST, math.log(1 + 1 / TAIL_MAX_DOCS))
    return float(blocks[-1] * math.exp(-(TAIL_LAST / 2 - 0.5) * lam) / math.expm1(lam))


class Calibrated:
    """p_cal of every pool13 pair for one P_fused variant, G_q and the per-query orders."""

    def __init__(self, frame: Frame, variant: str, a: float, b: float, scale: float = 1.0, use_tail: bool = True,
                 g_depth: int | None = None) -> None:
        """``g_depth``: G_q (sum and tail) is taken over the top ``g_depth`` docs in cosine order only, so it does not depend on how
        deep the pool was scored (free frame: 211)."""
        self.frame, self.variant, self.a, self.b, self.scale = frame, variant, a, b, scale
        self.p: dict[int, np.ndarray] = {}      # p_cal in RRF order
        self.fused: dict[int, np.ndarray] = {}
        self.g_pool: dict[int, float] = {}
        self.g: dict[int, float] = {}
        self.tail: dict[int, float] = {}
        for q in frame.qids:
            rows = frame.order[q]
            p4b = np.array([float(r["judge_score"]) for r in rows])
            jina = np.array([float(r["jina_score"]) if r["jina_score"] is not None else np.nan for r in rows])
            fused = frame.fused(variant, p4b, jina)
            p = sigmoid(a * logit(fused) + b)
            by_cos = p[np.argsort([int(r["cosine_rank"]) for r in rows], kind="stable")]
            self.fused[q], self.p[q] = fused, p
            if g_depth is not None:
                by_cos = by_cos[:g_depth]
            self.g_pool[q] = float(p.sum() if g_depth is None else by_cos.sum())
            self.tail[q] = tail_mass(by_cos)
            self.g[q] = scale * (self.g_pool[q] + (self.tail[q] if use_tail else 0.0))
        self.p_of = {(q, int(r["doc_id"])): float(p) for q in frame.qids for r, p in zip(frame.order[q], self.p[q])}

    def expected(self, keep: dict[int, int], tail: bool = True) -> dict[str, float]:
        """Macro predicted Doc P / R / F2 of selecting the top ``keep[q]`` docs of the RRF order."""
        precision, recall, f2 = [], [], []
        for q in self.frame.qids:
            k = keep[q]
            tp = float(self.p[q][:k].sum())
            g = self.g[q] if tail else self.g_pool[q]
            precision.append(tp / k if k else 0.0)
            recall.append(tp / g)
            f2.append(5 * tp / (4 * g + k))
        return {"precision": float(np.mean(precision)), "recall": float(np.mean(recall)), "f2": float(np.mean(f2))}


    def expected_docs(self, docs_by_query: dict[int, list[int]]) -> dict[str, float]:
        """Same as ``expected`` for an arbitrary submitted set (docs of the pool13)."""
        precision, recall, f2 = [], [], []
        for q in self.frame.qids:
            ids = docs_by_query[q]
            tp = sum(self.p_of[(q, d)] for d in ids)
            precision.append(tp / len(ids))
            recall.append(tp / self.g[q])
            f2.append(5 * tp / (4 * self.g[q] + len(ids)))
        return {"precision": float(np.mean(precision)), "recall": float(np.mean(recall)), "f2": float(np.mean(f2))}


def fit_scale(frame: Frame, variant: str, a: float, b: float, base_keep: dict[int, int]) -> float:
    """Global factor s on G_q so that the predicted macro F2 of J3-thrC equals its LB Doc F2 (the pool13 and the tail model miss
    gold docs outside the pool: the Doc R of the LB is about 0.51, while sum p_cal over the pool predicts about 0.79)."""
    lo, hi = 0.5, 5.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        f2 = Calibrated(frame, variant, a, b, mid).expected(base_keep)["f2"]
        lo, hi = (lo, mid) if f2 < LB_DOC_F2["J3-thrC"] else (mid, hi)
    return math.sqrt(lo * hi)


def ef2_keep(cal: Calibrated, tail: bool = True) -> dict[int, int]:
    keep = {}
    for q in cal.frame.qids:
        p, depth = cal.p[q], len(cal.p[q])
        g = cal.g[q] if tail else cal.g_pool[q]
        k = np.arange(1, depth + 1)
        value = 5 * np.cumsum(p) / (4 * g + k)
        lo = min(FLOOR_K, depth)
        keep[q] = int(lo + np.argmax(value[lo - 1:]))  # argmax returns the first (smallest k) on ties; +1 -1
        keep[q] = max(lo, keep[q])
    return keep


def ef2_keep_scaled(cal: Calibrated, lam: float, tail: bool = True) -> dict[int, int]:
    """EF2 cut with the marginal threshold scaled by a global ``lam`` (S40 probes FREE-cut150 / -cut130). The EF2 optimum k* of a query
    also maximises sum_{i<=k} p_i - (F2*_q / 5) k (F2*_q = the query's EF2 value; Dinkelbach), so ``lam`` = 1 gives ``ef2_keep`` and
    ``lam`` > 1 cuts earlier: k = argmax_k sum_{i<=k} p_i - lam (F2*_q / 5) k over [min(20, depth), depth], first k on ties."""
    keep = {}
    for q in cal.frame.qids:
        p, depth = cal.p[q], len(cal.p[q])
        g = cal.g[q] if tail else cal.g_pool[q]
        k = np.arange(1, depth + 1)
        cum = np.cumsum(p)
        lo = min(FLOOR_K, depth)
        f_star = float(np.max((5 * cum / (4 * g + k))[lo - 1:]))
        value = cum - lam * f_star / 5 * k
        keep[q] = int(lo + np.argmax(value[lo - 1:]))
    return keep


def solve_lambda(cal: Calibrated, mean_docs: float, tail: bool = True) -> tuple[float, dict[int, int]]:
    """Global ``lam`` whose ``ef2_keep_scaled`` mean docs per query is closest to ``mean_docs`` (bisection on log lam, 60 steps)."""
    lo, hi = 0.05, 50.0
    mean = lambda lam: float(np.mean(list(ef2_keep_scaled(cal, lam, tail).values())))  # noqa: E731  decreasing in lam
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if mean(mid) > mean_docs else (lo, mid)
    lam = min((lo, hi), key=lambda x: abs(mean(x) - mean_docs))
    return lam, ef2_keep_scaled(cal, lam, tail)


def thr_keep(frame: Frame, threshold_of_doc) -> dict[int, int]:
    """clip(#docs of the pool with P4B >= their threshold, ceil(0.5 K_q), floor(1.3 K_q))."""
    keep = {}
    for q in frame.qids:
        k = frame.base_k[q]
        n = sum(1 for r in frame.order[q] if float(r["judge_score"]) >= threshold_of_doc(int(r["doc_id"])))
        keep[q] = int(np.clip(n, math.ceil(0.5 * k - 1e-9), math.floor(1.3 * k + 1e-9)))
    return keep


def marginal_curve(p4b: np.ndarray, p_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P4B descending and, for each position, the mean p_cal of the docs within +-BAND of the language's size around it."""
    order = np.argsort(-p4b, kind="stable")
    sorted_p4b, sorted_p = p4b[order], p_cal[order]
    n, h = len(p4b), max(1, int(round(BAND * len(p4b))))
    csum = np.concatenate([[0.0], np.cumsum(sorted_p)])
    pos = np.arange(n)
    lo, hi = np.clip(pos - h, 0, n), np.clip(pos + h + 1, 0, n)
    return sorted_p4b, (csum[hi] - csum[lo]) / (hi - lo)


def threshold_at(curve: tuple[np.ndarray, np.ndarray], c: float) -> float:
    """P4B where the marginal precision first drops to c along the descending P4B list."""
    sorted_p4b, marginal = curve
    below = np.flatnonzero(marginal <= c)
    return float(sorted_p4b[below[0]]) if len(below) else float(sorted_p4b[-1])


def lang_thr(cal: Calibrated) -> dict[str, Any]:
    frame = cal.frame
    by_lang: dict[str, tuple[list[float], list[float]]] = {}
    for q in frame.qids:
        for r, p in zip(frame.order[q], cal.p[q]):
            lst = by_lang.setdefault(frame.doc_lang(int(r["doc_id"])), ([], []))
            lst[0].append(float(r["judge_score"]))
            lst[1].append(float(p))
    curves = {lang: marginal_curve(np.array(v[0]), np.array(v[1])) for lang, v in by_lang.items()}

    def selection(c: float) -> tuple[dict[str, float], dict[int, int]]:
        t = {lang: threshold_at(curve, c) for lang, curve in curves.items()}
        return t, thr_keep(frame, lambda d: t[frame.doc_lang(d)])

    def gap(c: float) -> float:
        return cal.expected(selection(c)[1])["f2"] / 5 - c

    lo, hi = 0.01, 0.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    c = 0.5 * (lo + hi)
    thresholds, keep = selection(c)
    return {"c": c, "thresholds": thresholds, "keep": keep, "langs": {lang: len(v[0]) for lang, v in by_lang.items()},
            "f2_over_5": cal.expected(keep)["f2"] / 5}


def keep_stats(keep: dict[int, int]) -> dict[str, float]:
    values = np.array([keep[q] for q in sorted(keep)], dtype=np.float64)
    return {"total": int(values.sum()), "p10": float(np.quantile(values, .1)), "p50": float(np.quantile(values, .5)),
            "p90": float(np.quantile(values, .9)), "min": float(values.min()), "max": float(values.max())}


def chosen_rows(frame: Frame, keep: dict[int, int]) -> dict[int, list[dict[str, Any]]]:
    return {q: frame.order[q][: keep[q]] for q in frame.qids}


def run() -> dict[str, Any]:
    frame = Frame()
    out = cal_root()
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name, path in submitted_zips().items():
        docs = {r["id"]: r["relevant_docs"] for r in _load_submission(path)}
        sets[name] = set_arrays(frame, docs)
    base_keep, _, _ = s31_build.variable_keep(frame.ranked, frame.base_k, "thr")
    j3_ids = {q: [int(r["doc_id"]) for r in rows[: base_keep[q]]] for q, rows in frame.order.items()}
    zip_j3 = {r["id"]: r["relevant_docs"] for r in _load_submission(submitted_zips()["J3-thrC"])}
    result: dict[str, Any] = {"j3_thrC_docs_equal_submitted_zip": all(j3_ids[q] == zip_j3[q] for q in j3_ids),
                              "lb_doc_p": LB_DOC_P, "lb_doc_f2": LB_DOC_F2,
                              "jina_to_p4b": "monotone quantile map on the pool13 pairs", "variants": {}}
    k4_docs = {r["id"]: r["relevant_docs"] for r in _load_submission(submitted_zips()["K4-thr-C"])}
    for variant in VARIANTS:
        fit = fit_calibration(frame, variant, sets)
        scale = fit_scale(frame, variant, fit["a"], fit["b"], base_keep)
        entry: dict[str, Any] = {"fit": fit, "g_scale": scale, "modes": {}}
        for mode, factor in (("anchored", scale), ("literal", 1.0)):
            cal = Calibrated(frame, variant, fit["a"], fit["b"], factor)
            tails = np.array([cal.tail[q] / (cal.g_pool[q] + cal.tail[q]) for q in frame.qids])
            sub: dict[str, Any] = {"scale": factor,
                                   "g": {"mean": float(np.mean(list(cal.g.values()))), "mean_pool": float(np.mean(list(cal.g_pool.values()))),
                                         "tail_share_mean": float(tails.mean()), "tail_share_p90": float(np.quantile(tails, .9))},
                                   "predicted_lb_check": {"J3-thrC": cal.expected_docs(zip_j3), "K4-thr-C": cal.expected_docs(k4_docs),
                                                          "lb_doc_f2": LB_DOC_F2},
                                   "selections": {}}
            selections = {"J3-thrC": base_keep, "EF2-cut": ef2_keep(cal)}
            lang = lang_thr(cal)
            selections["LANG-thr"] = lang["keep"]
            sub["lang_thr"] = {k: lang[k] for k in ("c", "thresholds", "langs", "f2_over_5")}
            for tag, keep in selections.items():
                ids = {q: [int(r["doc_id"]) for r in rows] for q, rows in chosen_rows(frame, keep).items()}
                sub["selections"][tag] = {"docs_per_query": keep_stats(keep), "predicted": cal.expected(keep),
                                          "vs_j3_thrC": doc_set_diff(ids, j3_ids)}
            entry["modes"][mode] = sub
            if mode == "anchored":
                for tag, name in (("EF2-cut", f"EF2-cut-{variant}"), ("LANG-thr", f"LANG-thr-{variant}")):
                    _atomic_json(out / f"keep-{name}.json", {str(q): int(k) for q, k in selections[tag].items()})
                rows = [{"query_id": q, "doc_id": int(r["doc_id"]), "p_fused": float(f), "p_cal": float(p)}
                        for q in frame.qids for r, f, p in zip(frame.order[q], cal.fused[q], cal.p[q])]
                pd.DataFrame(rows).to_parquet(out / f"pcal-{variant}.parquet", index=False)
        result["variants"][variant] = entry
    _atomic_json(out / "cal.json", result)
    return result


def ef2_fuse() -> dict[str, Any]:
    """EF2-FUSE4 (pool13, RRF of the four sources) and EF2-free (K4-free pools, RRF of the four sources, no K_q anywhere), with the
    P4B calibration (a, b) and the scale s of ``run`` kept as they are. Writes ``keep-<name>.json`` and ``pcal-<tag>.parquet``."""
    out = cal_root()
    fitted = json.loads((out / "cal.json").read_text(encoding="utf-8"))["variants"]["p4b"]
    a, b, s = fitted["fit"]["a"], fitted["fit"]["b"], fitted["g_scale"]
    result: dict[str, Any] = {"a": a, "b": b, "s": s, "lb_doc_f2": {**LB_DOC_F2, "FUSE4-thrC": 0.4178, "FUSE4-free": 0.4104, "EF2-cut-p4b": 0.4209}, "frames": {}}
    zips = {"thrC": {"J3-thrC": submitted_zips()["J3-thrC"], "K4-thr-C": submitted_zips()["K4-thr-C"],
                     "FUSE4-thrC": fuse.run_root() / "build-docread" / "FUSE4-thrC-docread.zip",
                     "FUSE4w-thrC": fuse.run_root() / "build-docread" / "FUSE4w-thrC-docread.zip",
                     "EF2-cut-p4b": fuse.run_root() / "build-docread" / "EF2-cut-p4b-docread.zip",
                     "EF2-cut-avg": fuse.run_root() / "build-docread" / "EF2-cut-avg-docread.zip",
                     "LANG-thr-avg": fuse.run_root() / "build-docread" / "LANG-thr-avg-docread.zip"},
            "free": {"FUSE4-free": fuse.run_root() / "build-docread" / "FUSE4-free-docread.zip",
                     "K4-free": paths.RUNS_DIR / "s31-strong-judge" / "build-ext" / "K4-free.zip"}}
    ref_ids = {r["id"]: r["relevant_docs"] for r in _load_submission(zips["thrC"]["EF2-cut-p4b"])}
    for frame_name, name, tail in (("thrC", "EF2-FUSE4", True),):  # EF2-free: ``ef2_free`` (S35, 24/10)
        frame = Frame(fuse.FOUR_KEYS, frame_name)
        cal = Calibrated(frame, "p4b", a, b, s, use_tail=tail)
        keep = ef2_keep(cal)
        ids = {q: [int(r["doc_id"]) for r in rows] for q, rows in chosen_rows(frame, keep).items()}
        entry: dict[str, Any] = {"frame": frame_name, "g_rule": "s (sum p_cal over the pool + geometric tail beyond the pool13 depth)" if tail
                                 else "s sum p_cal over the scored pool (no tail)",
                                 "depth": keep_stats({q: len(frame.order[q]) for q in frame.qids}),
                                 "g_mean": float(np.mean(list(cal.g.values()))), "docs_per_query": keep_stats(keep),
                                 "predicted": cal.expected(keep), "vs_EF2-cut-p4b": doc_set_diff(ids, ref_ids),
                                 "keep_at_floor": int(sum(1 for q in frame.qids if keep[q] == min(FLOOR_K, len(frame.order[q])))),
                                 "keep_at_depth": int(sum(1 for q in frame.qids if keep[q] == len(frame.order[q]))),
                                 "predicted_submitted": {}}
        for zname, zpath in zips[frame_name].items():
            docs = {r["id"]: r["relevant_docs"] for r in _load_submission(zpath)}
            entry["predicted_submitted"][zname] = cal.expected_docs(docs)
        result["frames"][name] = entry
        _atomic_json(out / f"keep-{name}.json", {str(q): int(k) for q, k in keep.items()})
        tag = "p4b" if frame_name == "thrC" else "free"
        if frame_name == "free":
            rows = [{"query_id": q, "doc_id": int(r["doc_id"]), "p_fused": float(f), "p_cal": float(p)}
                    for q in frame.qids for r, f, p in zip(frame.order[q], cal.fused[q], cal.p[q])]
            pd.DataFrame(rows).to_parquet(out / f"pcal-{tag}.parquet", index=False)
        del frame, cal
    _atomic_json(out / "ef2-fuse.json", result)
    return result


FREE_G_DEPTH = 211  # P_FREE: G_q is the sum of p_cal over the cosine top-211 plus a geometric tail beyond it
LB_DOC_F2_FREE = {"FUSE4-free": 0.4104}


def ef2_free(deepen: bool = True) -> dict[str, Any]:
    """EF2-free (S35, 24/10): free frame (cosine top-211 + adaptive deepening over the scored union), RRF of the four sources, EF2 cut
    with floor 20 and ceiling = scored pool depth; no K_q anywhere. (a, b) of ``run`` unchanged (fitted on three aggregate LB Doc P
    values, no per-query label or LB score). G_q = s_free (sum p_cal over the cosine top-211 + tail beyond 211), s_free fitted once so
    that the predicted macro Doc F2 of the submitted FUSE4-free (aggregate LB 0.4104) is equal to it.

    ``deepen`` False (EF2-free-d211): the pool is cut at the cosine top-211 for every query; (a, b) and s_free are read from
    ``cal/ef2-free.json`` and not refitted (G_q is the same sum over the cosine top-211 plus tail, so only the pool differs)."""
    out = cal_root()
    fitted = json.loads((out / "cal.json").read_text(encoding="utf-8"))["variants"]["p4b"]
    a, b = fitted["fit"]["a"], fitted["fit"]["b"]
    build_dir = fuse.run_root() / "build-docread"
    name = "EF2-free" if deepen else "EF2-free-d211"
    tag = "free" if deepen else "free-d211"
    frame = Frame(fuse.FOUR_KEYS, "free" if deepen else "free211")
    docs_of = lambda path: {r["id"]: r["relevant_docs"] for r in _load_submission(path)}  # noqa: E731
    ref_ids = docs_of(build_dir / "EF2-cut-p4b-docread.zip")
    if deepen:
        fuse4_free = docs_of(build_dir / "FUSE4-free-docread.zip")
        lo, hi = 0.5, 5.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            f2 = Calibrated(frame, "p4b", a, b, mid, use_tail=True, g_depth=FREE_G_DEPTH).expected_docs(fuse4_free)["f2"]
            lo, hi = (lo, mid) if f2 < LB_DOC_F2_FREE["FUSE4-free"] else (mid, hi)
        s_free = math.sqrt(lo * hi)
    else:
        s_free = float(json.loads((out / "ef2-free.json").read_text(encoding="utf-8"))["s_free"])
    cal = Calibrated(frame, "p4b", a, b, s_free, use_tail=True, g_depth=FREE_G_DEPTH)
    keep = ef2_keep(cal)
    ids = {q: [int(r["doc_id"]) for r in rows] for q, rows in chosen_rows(frame, keep).items()}
    tails = np.array([cal.tail[q] / (cal.g_pool[q] + cal.tail[q]) for q in frame.qids])
    result: dict[str, Any] = {
        "name": name, "a": a, "b": b, "s_free": s_free, "s_thrC": fitted["g_scale"],
        "fit_data": {"lb_doc_f2_free": LB_DOC_F2_FREE, "calibration": "ab from cal.json (three aggregate LB Doc P)",
                     "s_free": "fitted on the aggregate LB Doc F2 of FUSE4-free" if deepen else "read from cal/ef2-free.json, not refitted"},
        "g_rule": f"s_free (sum p_cal over the cosine top-{FREE_G_DEPTH} + geometric tail beyond {FREE_G_DEPTH})",
        "depth": keep_stats({q: len(frame.order[q]) for q in frame.qids}),
        "g": {"mean": float(np.mean(list(cal.g.values()))), "tail_share_mean": float(tails.mean()), "tail_share_p90": float(np.quantile(tails, .9))},
        "docs_per_query": keep_stats(keep), "predicted": cal.expected(keep),
        "vs_EF2-cut-p4b": doc_set_diff(ids, ref_ids),
        "keep_at_floor": int(sum(1 for q in frame.qids if keep[q] == min(FLOOR_K, len(frame.order[q])))),
        "keep_at_depth": int(sum(1 for q in frame.qids if keep[q] == len(frame.order[q]))),
    }
    if deepen:
        result["predicted_submitted"] = {n: cal.expected_docs(docs_of(path)) for n, path in (
            ("FUSE4-free", build_dir / "FUSE4-free-docread.zip"), ("K4-free", paths.RUNS_DIR / "s31-strong-judge" / "build-ext" / "K4-free.zip"))}
        result["vs_FUSE4-free"] = doc_set_diff(ids, fuse4_free)
        result["queries_deeper_than_211"] = int(sum(1 for q in frame.qids if len(frame.order[q]) > FREE_G_DEPTH))
    else:
        result["deep_queries"] = d211_loss(ids)
    _atomic_json(out / f"keep-{name}.json", {str(q): int(k) for q, k in keep.items()})
    rows = [{"query_id": q, "doc_id": int(r["doc_id"]), "p_fused": float(f), "p_cal": float(p)}
            for q in frame.qids for r, f, p in zip(frame.order[q], cal.fused[q], cal.p[q])]
    pd.DataFrame(rows).to_parquet(out / f"pcal-{tag}.parquet", index=False)
    _atomic_json(out / f"{name.lower()}.json", result)
    return result


def d211_loss(ids: dict[int, list[int]]) -> dict[str, Any]:
    """Docs of EF2-free (``docsel-EF2-free``) that the cosine top-211 cut loses on the queries whose free pool was deeper than 211
    (pool rows counted in ``pcal-free.parquet``), and the docs it takes in their place."""
    depth = pd.read_parquet(cal_root() / "pcal-free.parquet", columns=["query_id"]).groupby("query_id").size()
    deep = sorted(int(q) for q in depth[depth > FREE_G_DEPTH].index)
    free = pd.read_parquet(fuse.run_root() / "docsel" / "docsel-EF2-free.parquet", columns=["query_id", "doc_id", "cosine_rank"])
    free = free[free.query_id.isin(deep)]
    old = {int(q): set(g.doc_id.astype(int)) for q, g in free.groupby("query_id")}
    beyond = free[free.cosine_rank > FREE_G_DEPTH].groupby("query_id").size().reindex(deep, fill_value=0)
    lost = np.array([len(old[q] - set(ids[q])) for q in deep])
    gained = np.array([len(set(ids[q]) - old[q]) for q in deep])
    old_n, new_n = np.array([len(old[q]) for q in deep]), np.array([len(ids[q]) for q in deep])
    return {"queries": len(deep), "docs_ef2_free": int(old_n.sum()), "docs_d211": int(new_n.sum()),
            "docs_lost_total": int(lost.sum()), "docs_lost_per_query_mean": float(lost.mean()),
            "docs_lost_per_query_p10_p50_p90": [float(np.quantile(lost, q)) for q in (.1, .5, .9)],
            "docs_lost_beyond_cosine_211": int(beyond.sum()), "docs_gained_total": int(gained.sum()),
            "docs_gained_per_query_mean": float(gained.mean()), "queries_identical_docs": int(((lost == 0) & (gained == 0)).sum())}


def docsel_j3() -> dict[str, Any]:
    """``docsel/docsel-J3-thrC.parquet``: the docs of the submitted J3-thrC (same scheme as the FUSE4 docsel; ``chunk`` = the doc
    carries a chunk in the submitted ZIP, built before the Pharmacity fix, so no Pharmacity doc has one)."""
    from r2ai.s36.build import docsel_writer

    frame = Frame()
    keep, _, _ = s31_build.variable_keep(frame.ranked, frame.base_k, "thr")
    rows = _load_submission(submitted_zips()["J3-thrC"], with_chunks=True)
    chunk_docs = {r["id"]: [int(c["doc_id"]) for c in r["relevant_chunks"]] for r in rows}
    sidecar = json.loads((submitted_zips()["J3-thrC"].with_name("J3-thrC.s31.json")).read_text(encoding="utf-8"))
    t_c = float(sidecar["chunk_threshold"]["t_c"])
    out = fuse.run_root() / "docsel"
    out.mkdir(parents=True, exist_ok=True)
    pcal = {"J3-thrC": pd.read_parquet(cal_root() / "pcal-p4b.parquet").rename(columns={"p_fused": "p_fused_p4b", "p_cal": "p_cal_p4b"})
            .merge(pd.read_parquet(cal_root() / "pcal-avg.parquet").rename(columns={"p_fused": "p_fused_avg", "p_cal": "p_cal_avg"}), on=["query_id", "doc_id"])}
    docsel_writer(out, "", t_c, pcal)("J3-thrC", chosen_rows(frame, keep), chunk_docs, {"t_c": t_c})
    return {"path": str(out / "docsel-J3-thrC.parquet"), "t_c": t_c, "docs": int(sum(keep.values())), "chunks": int(sum(len(v) for v in chunk_docs.values()))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "docsel-j3", "ef2-fuse", "ef2-free", "ef2-free-d211"))
    args = parser.parse_args(argv)
    result = {"docsel-j3": docsel_j3, "ef2-fuse": ef2_fuse, "ef2-free": ef2_free, "ef2-free-d211": lambda: ef2_free(deepen=False), "run": run}[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
