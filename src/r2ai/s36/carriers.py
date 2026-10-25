"""S36 chunk-carrier rules on a docsel (24/10): which selected docs carry a chunk. Parameters live in a config (JSON) or on the CLI.

``python -m r2ai.s36.carriers --docsel docsel-EF2-free.parquet --rule fair --target 174535 --name EF2-free-CRfair``
``python -m r2ai.s36.carriers --docsel docsel-X.parquet --rule fair --name X-CRfair``      (default chunk rule from 24/10: ``rho`` = ``DEFAULT_RHO``)
``python -m r2ai.s36.carriers --docsel docsel-EF2-free.parquet --rule swap --drop 4400 --name EF2-free-CRswap``
``python -m r2ai.s36.carriers --config chunk-rule.json --docsel ...``      (``{"rule": "fair", "target": 174535, "name": "..."}``)

Rules (``p`` = P4B of the docsel; docs sorted by p descending, ties by docsel rank):

* ``threshold``  the docsel's own carriers (``chunk`` flag: global P4B >= t_c), unchanged;
* ``fair``       CR-fair: every query carries its top ``ceil(rho * |D_q|)`` docs by p. ``rho`` is one global constant: ``--rho``
                 (default ``DEFAULT_RHO``, the value that gave 174,535 carriers on the EF2-free docsel and won on the LB: chunk 0,3537),
                 or, with ``--target``, solved so that the total number of carriers is closest to ``target`` (smallest ``rho`` on a tie);
* ``swap``       CR-swap: the global threshold carriers minus the ``drop`` carriers with the lowest p (ties: later query, later rank).

Writes ``<out>/carriers-<name>.parquet`` (``query_id, doc_id``) and ``<out>/carriers-<name>.json`` (counts, carriers per query
P10/P50/P90, share of the 100 queries with most carriers, rho). ``r2ai.s38.compose --carriers`` takes the parquet.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s36 import fuse

RULES = ("threshold", "fair", "swap")
DEFAULT_RHO = 0.7876447876447916   # CR-fair on EF2-free (174,535 carriers), LB chunk 0,3537 (+0,0053 over the threshold rule): the default


def order_by_p(docsel: pd.DataFrame) -> pd.DataFrame:
    """Docsel rows with ``order`` = position of the doc in its query when sorted by p4b descending (ties by rank)."""
    ranked = docsel.sort_values(["query_id", "p4b", "rank"], ascending=[True, False, True], kind="stable").copy()
    ranked["order"] = ranked.groupby("query_id").cumcount()
    return ranked


def fair_total(sizes: np.ndarray, rho: float) -> int:
    return int(np.ceil(rho * sizes - 1e-12).clip(min=0).sum())


def solve_rho(sizes: np.ndarray, target: int) -> float:
    """Global ``rho`` in (0, 1] whose total ``sum ceil(rho * |D_q|)`` is closest to ``target`` (smallest rho on a tie)."""
    if not 0 < target <= int(sizes.sum()):
        raise ValueError(f"target {target} outside (0, {int(sizes.sum())}]")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fair_total(sizes, mid) < target else (lo, mid)
    best = min((lo, hi), key=lambda r: (abs(fair_total(sizes, r) - target), r) if r > 0 else (float("inf"), r))
    return float(best)


def carriers_fair(docsel: pd.DataFrame, target: int | None = None, rho: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top ``ceil(rho * |D_q|)`` docs by p in every query; ``rho`` given, else solved for ``target`` carriers."""
    ranked = order_by_p(docsel)
    sizes = ranked.groupby("query_id").size()
    if rho is None:
        rho = solve_rho(sizes.to_numpy(), target)
    take = np.ceil(rho * ranked.query_id.map(sizes).to_numpy() - 1e-12)
    chosen = ranked[ranked.order.to_numpy() < take]
    return chosen[["query_id", "doc_id"]].reset_index(drop=True), {"rho": rho, "target": target}


def carriers_swap(docsel: pd.DataFrame, drop: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    car = docsel[docsel.chunk]
    if drop >= len(car):
        raise ValueError(f"drop {drop} >= carriers {len(car)}")
    worst = car.sort_values(["p4b", "query_id", "rank"], ascending=[True, False, False], kind="stable").head(drop)
    kept = car.drop(worst.index)
    return kept[["query_id", "doc_id"]].reset_index(drop=True), {
        "drop": drop, "t_c_before": float(car.p4b.min()), "t_c_after": float(kept.p4b.min()), "dropped_p4b_max": float(worst.p4b.max())}


def select_carriers(docsel: pd.DataFrame, rule: str, target: int | None = None, drop: int | None = None,
                    rho: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rule == "threshold":
        return docsel[docsel.chunk][["query_id", "doc_id"]].reset_index(drop=True), {}
    if rule == "fair":
        return carriers_fair(docsel, target, rho if rho is not None or target is not None else DEFAULT_RHO)
    if rule == "swap":
        if drop is None:
            raise ValueError("swap needs drop")
        return carriers_swap(docsel, drop)
    raise ValueError(f"unknown rule {rule}; one of {RULES}")


def describe(carriers: pd.DataFrame, queries: int = 1200) -> dict[str, Any]:
    per_query = carriers.groupby("query_id").size().reindex(range(1, queries + 1), fill_value=0)
    return {"carriers": int(len(carriers)), "queries": int(queries),
            "per_query_p10_p50_p90": [float(np.quantile(per_query, q)) for q in (.1, .5, .9)],
            "top100_queries_share": float(per_query.sort_values(ascending=False).head(100).sum() / max(1, per_query.sum()))}


def run(docsel_path: Path, rule: str, name: str, out: Path, target: int | None = None, drop: int | None = None,
        rho: float | None = None) -> dict[str, Any]:
    docsel = pd.read_parquet(docsel_path, columns=["query_id", "doc_id", "rank", "p4b", "chunk"])
    carriers, info = select_carriers(docsel, rule, target, drop, rho)
    out.mkdir(parents=True, exist_ok=True)
    target_path = out / f"carriers-{name}.parquet"
    paths.assert_writable(target_path)
    carriers.to_parquet(target_path, index=False)
    report = {"name": name, "rule": rule, "docsel": str(docsel_path), "docsel_sha256": _sha256(docsel_path), "docs": int(len(docsel)),
              "carriers_parquet": str(target_path), "carriers_sha256": _sha256(target_path), **info, **describe(carriers),
              "threshold_rule": describe(docsel[docsel.chunk][["query_id", "doc_id"]])}
    _atomic_json(out / f"carriers-{name}.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="JSON with rule / target / drop / name (CLI flags override)")
    parser.add_argument("--docsel", type=Path, required=True)
    parser.add_argument("--rule", choices=RULES)
    parser.add_argument("--target", type=int)
    parser.add_argument("--drop", type=int)
    parser.add_argument("--rho", type=float, help="fair: the global rho (default DEFAULT_RHO unless --target)")
    parser.add_argument("--name")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    rule = args.rule or config.get("rule")
    name = args.name or config.get("name")
    if not rule or not name:
        parser.error("rule and name are required (flags or config)")
    out = paths.resolve_path(args.out) if args.out else fuse.run_root() / "carriers"
    report = run(paths.resolve_path(args.docsel), rule, name, out, args.target if args.target is not None else config.get("target"),
                 args.drop if args.drop is not None else config.get("drop"), args.rho if args.rho is not None else config.get("rho"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
