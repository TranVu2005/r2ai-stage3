"""S41 (27/10): chunk-carrier probes on a fixed docsel (FREE-noJina docs unchanged).

    python -m r2ai.s41.carriers --docsel docsel.parquet --perq perq.parquet --rule fair --order rank --target 175548 --name CH-rrf --out DIR

Which selected docs carry a chunk. CR-fair (``r2ai.s36.carriers``) gives every query its top ``ceil(rho |D_q|)`` docs by P4B. LB 27/10:
FREE-cut150 (EF2 cut in RRF order, nearly every doc a carrier) swapped 23k of FREE-noJina's 175,548 carriers for docs with a lower P4B
(median 0,147 against 0,255) but a much higher C-max (0,915 against 0,562), and Chunk F2 went up (+0,0026: P +0,004, R -0,003).
C-max is the chunk-level judge (best of the top-3 t256 chunks), so it may rank carriers better than the doc-level P4B.

* ``--order``: ``p4b`` (CR-fair as S36), ``rank`` (docsel RRF rank), ``cmax`` (C-max score; missing C-max last), ties by rank.
* ``--rule fair``: top ``ceil(rho |D_q|)`` in that order, ``rho`` solved for ``--target`` (``r2ai.s36.carriers.solve_rho``).
* ``--rule ef2``: in that order, query q takes the n maximising the expected chunk F2 ``5 sum_{i<=n} p_i / (4 s g_q + n)`` (p = p_cal of the
  docsel, g_q = the apply's G_q, s one global scale of the gold chunk count), n in [1, |D_q|]; s solved so that the total is closest to
  ``--target`` (the ZIP budget of FREE-noJina). Expected F2 under P = c sum p / n, R = c' sum p / G: only s = c / c' matters for n.

Writes ``<out>/carriers-<name>.parquet`` (``query_id, doc_id``) and ``<out>/carriers-<name>.json``; ``r2ai.s38.compose --carriers`` takes it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s36.carriers import describe, solve_rho

ORDERS = ("p4b", "rank", "cmax")


def ordered(docsel: pd.DataFrame, order: str) -> pd.DataFrame:
    """Docsel rows sorted within each query by ``order`` (descending score, ascending rank), ``pos`` = position in the query."""
    if order not in ORDERS:
        raise ValueError(f"order {order} not in {ORDERS}")
    frame = docsel.copy()
    if order == "rank":
        frame = frame.sort_values(["query_id", "rank"], kind="stable")
    else:
        frame["_key"] = frame[order].fillna(-np.inf)
        frame = frame.sort_values(["query_id", "_key", "rank"], ascending=[True, False, True], kind="stable").drop(columns="_key")
    frame["pos"] = frame.groupby("query_id").cumcount()
    return frame


def fair(frame: pd.DataFrame, target: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    sizes = frame.groupby("query_id").size()
    rho = solve_rho(sizes.to_numpy(), target)
    take = np.ceil(rho * frame.query_id.map(sizes).to_numpy() - 1e-12)
    return frame[frame.pos.to_numpy() < take], {"rho": rho}


def ef2_counts(p: dict[int, np.ndarray], g: dict[int, float], s: float) -> dict[int, int]:
    """n maximising sum_{i<=n} p_i / (4 s g_q + n) per query (first n on ties), n >= 1."""
    out = {}
    for q, pq in p.items():
        n = np.arange(1, len(pq) + 1)
        out[q] = int(1 + np.argmax(np.cumsum(pq) / (4 * s * g[q] + n)))
    return out


def solve_s(p: dict[int, np.ndarray], g: dict[int, float], target: int) -> tuple[float, dict[int, int]]:
    """Global s whose total ef2_counts is closest to ``target`` (the total grows with s; bisection on log s)."""
    total = lambda s: sum(ef2_counts(p, g, s).values())          # noqa: E731
    lo, hi = 1e-4, 1e4
    if not total(lo) <= target <= total(hi):
        raise ValueError(f"target {target} outside [{total(lo)}, {total(hi)}]")
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if total(mid) < target else (lo, mid)
    s = min((lo, hi), key=lambda x: (abs(total(x) - target), x))
    return s, ef2_counts(p, g, s)


def ef2(frame: pd.DataFrame, g: dict[int, float], target: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = {int(q): grp.p_cal.to_numpy(dtype=float) for q, grp in frame.groupby("query_id", sort=True)}
    s, n = solve_s(p, g, target)
    take = frame.query_id.map(n).to_numpy()
    sp = {q: float(p[q][: n[q]].sum()) for q in p}
    expected = float(np.mean([5 * sp[q] / (4 * s * g[q] + n[q]) for q in p]))
    return frame[frame.pos.to_numpy() < take], {"s": s, "expected_chunk_f2_unscaled": expected}


def select(docsel: pd.DataFrame, rule: str, order: str, target: int, g: dict[int, float] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = ordered(docsel, order)
    if rule == "fair":
        chosen, info = fair(frame, target)
    elif rule == "ef2":
        if g is None:
            raise ValueError("ef2 needs g")
        chosen, info = ef2(frame, g, target)
    else:
        raise ValueError(f"unknown rule {rule}")
    return chosen[["query_id", "doc_id"]].reset_index(drop=True), {"rule": rule, "order": order, "target": target, **info}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docsel", type=Path, required=True)
    parser.add_argument("--perq", type=Path, default=None, help="apply perq parquet (query_id, g): needed by --rule ef2")
    parser.add_argument("--rule", choices=("fair", "ef2"), required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None, help="carriers parquet to compare with (e.g. the apply's CR-fair)")
    args = parser.parse_args(argv)
    docsel = pd.read_parquet(args.docsel, columns=["query_id", "doc_id", "rank", "p4b", "cmax", "p_cal"])
    g = None
    if args.perq is not None:
        perq = pd.read_parquet(args.perq, columns=["query_id", "g"])
        g = dict(zip(perq.query_id.astype(int), perq.g.astype(float)))
    carriers, info = select(docsel, args.rule, args.order, args.target, g)
    out = args.out.resolve()
    target = out / f"carriers-{args.name}.parquet"
    paths.assert_writable(target)
    out.mkdir(parents=True, exist_ok=True)
    carriers.to_parquet(target, index=False)
    sel = carriers.merge(docsel, on=["query_id", "doc_id"])
    report = {"name": args.name, "docsel": str(args.docsel), "docsel_sha256": _sha256(args.docsel), **info, **describe(carriers),
              "carriers_parquet": str(target), "carriers_sha256": _sha256(target),
              "carrier_p4b_median": float(sel.p4b.median()), "carrier_cmax_median": float(sel.cmax.median()),
              "carrier_pcal_sum": float(sel.p_cal.sum())}
    if args.compare is not None:
        old = pd.read_parquet(args.compare, columns=["query_id", "doc_id"])
        a, b = set(zip(carriers.query_id, carriers.doc_id)), set(zip(old.query_id, old.doc_id))
        report["vs_compare"] = {"compare": str(args.compare), "both": len(a & b), "only_new": len(a - b), "only_compare": len(b - a)}
    _atomic_json(out / f"carriers-{args.name}.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
