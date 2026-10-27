"""S41 (Vu 27/10): measure the private fallbacks on the 1,200 public queries (reads the public FREE-noJina inputs ON PURPOSE), frozen config
CH-rrf, through the same newq functions as ``newq build`` (``pools_from``, ``cosine_pcal_map``, ``apply_selection``):

    FULL   all 4B, cosine / 4B / C-max (control: must give the CH-rrf docsel and carriers)
    B0     ``--no-cmax`` (Jina already off): RRF cosine + 4B, C-max left out
    IMP30  30% of the pool pairs lose their 4B score (``numpy.random.default_rng(2026)``, as ``--test-drop-4b 0.3 --test-drop-seed 2026``),
           ``--impute-4b``
    COS    every 4B score gone, ``--impute-4b``; C-max kept. ``cosine_pcal_map`` needs >= 1,000 scored pairs, so the cosine -> p_cal map is
           the one fitted on the full public 4B (two constants; on private they would have to be pinned in the config)

    python scripts/s41/fallback_public.py --variant B0 --out out/runs/s41/fallback [--compose]

Writes ``<out>/<variant>/apply`` (docsel, carriers, ...), ``<out>/<variant>/fallback.json``: docs, the variant's own predicted Doc F2 and the
Doc F2 predicted for its docs by the FULL model (true 4B p_cal, the same G_q), docs / carriers shared with CH-rrf; with ``--compose`` the S38
compose full+w768 (as scripts/s41/run_ch.sh) and its ZIP / SHA256 / validator."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "s41"))
from r2ai import newq, paths  # noqa: E402
import regress_public as reg  # noqa: E402

VARIANTS = {"FULL": (1.0, True, 0.0), "B0": (0.0, False, 0.0), "IMP30": (0.0, True, 0.3), "COS": (0.0, True, 1.0)}   # (unused, C-max, drop share)
SEED = 2026


def top_frame(pools: dict) -> tuple[pd.DataFrame, dict[int, float]]:
    rows = [(int(r["pair_id"]), q, int(r["doc_id"]), int(r["cosine_rank"]), float(r["cosine_score"]), float(r["judge_score"]))
            for q, rs in pools.items() for r in rs]
    top = pd.DataFrame(rows, columns=["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "p4b"])
    if top.pair_id.duplicated().any():
        raise SystemExit("duplicate pair_id in the public pools")
    top = top.sort_values(["query_id", "cosine_rank"], kind="stable").reset_index(drop=True)
    return top.drop(columns="p4b"), dict(zip(top.pair_id, top.p4b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    name = f"FB-{args.variant}"
    work = args.out.resolve() / args.variant
    apply_dir = work / "apply"
    paths.assert_writable(work)
    apply_dir.mkdir(parents=True, exist_ok=True)
    _, use_cmax, drop = VARIANTS[args.variant]
    cfg = newq.load_config(None)
    a, b = float(cfg["a"]), float(cfg["b"])
    spec = {**reg.s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
    pools_pub, cmax = reg.public_sources(spec)
    top, scores = top_frame(pools_pub)
    full_map = {**newq.cosine_pcal_map(top, scores, a, b), "a": a, "b": b}
    res: dict = {"variant": args.variant, "name": name, "config_sha256": newq.sha256(newq.DEFAULT_CONFIG), "pool_pairs": int(len(top)),
                 "drop_share": drop, "drop_seed": SEED if drop else None, "cmax": use_cmax}
    kept = scores
    if drop:
        ids = top.pair_id.astype(int).to_numpy()
        gone = set(np.random.default_rng(SEED).choice(ids, size=int(round(drop * len(ids))), replace=False).tolist())
        kept = {k: v for k, v in scores.items() if k not in gone}
        res["dropped_pairs"] = len(gone)
    impute = None
    if drop:
        impute = full_map if len(kept) < 1000 else {**newq.cosine_pcal_map(top, kept, a, b), "a": a, "b": b}
        res["impute_map"] = {**impute, "source": "full public 4B (no scored pair left)" if len(kept) < 1000 else "the kept 4B pairs"}
    keys = ("cosine_rank", "judge_rank", "cmax_rank") if use_cmax else ("cosine_rank", "judge_rank")
    t0 = time.monotonic()
    got = newq.apply_selection(newq.pools_from(top, kept, impute), {}, cmax if use_cmax else {}, cfg, keys, True, name, apply_dir)
    res["apply_seconds"] = round(time.monotonic() - t0, 1)
    full = newq.select(newq.pools_from(top, scores), {}, cmax, cfg, ("cosine_rank", "judge_rank", "cmax_rank"), True)
    chosen = {q: [int(r["doc_id"]) for r in rows] for q, rows in got["chosen"].items()}
    from r2ai.s36 import cal

    res["docs"] = cal.keep_stats(got["keep"])
    res["predicted_own"] = got["calibrated"].expected(got["keep"])
    res["predicted_full_model"] = full["calibrated"].expected_docs(chosen)
    res["predicted_ch_rrf"] = full["calibrated"].expected(full["keep"])
    ref = pd.read_parquet(reg.REF_DOCSEL, columns=["query_id", "doc_id"])
    new = pd.read_parquet(apply_dir / f"docsel-{name}.parquet", columns=["query_id", "doc_id"])
    a_set, b_set = set(zip(new.query_id, new.doc_id)), set(zip(ref.query_id, ref.doc_id))
    car = pd.read_parquet(apply_dir / f"carriers-{name}.parquet")
    car_ref = pd.read_parquet(reg.REF_CARRIERS)
    c_new, c_ref = set(zip(car.query_id, car.doc_id)), set(zip(car_ref.query_id, car_ref.doc_id))
    res["vs_ch_rrf"] = {"docs_shared": len(a_set & b_set), "docs_only_new": len(a_set - b_set), "docs_only_ch_rrf": len(b_set - a_set),
                        "carriers": len(c_new), "carriers_shared": len(c_new & c_ref)}
    res["carrier_info"] = got["carrier_info"]
    newq.atomic_json(work / "fallback.json", res)
    print(json.dumps(res, indent=1, default=str), flush=True)
    if not args.compose:
        return 0
    compose_dir = work / "compose"
    compose_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "r2ai.s38.compose", "run", "--name", f"{name}-full+w768", "--docsel", str(apply_dir / f"docsel-{name}.parquet"),
           "--rule", "full+w768", "--carriers", str(apply_dir / f"carriers-{name}.parquet"), "--fit-carriers", "--rounds", str(args.rounds),
           "--out", str(compose_dir)]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    t0 = time.monotonic()
    with open(work / "compose.log", "ab") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(compose_dir.glob(f"{name}-full+w768-t*.compose.json"))]
    ok = [r for r in found if r.get("under_ceiling") and r.get("validator", {}).get("ok")]
    best = min(ok, key=lambda r: r["threshold"]) if ok else {}
    res["compose"] = {"returncode": rc, "seconds": round(time.monotonic() - t0, 1),
                      **{k: best.get(k) for k in ("zip", "zip_bytes", "zip_sha256", "threshold", "docs", "chunks", "second_chunks", "dropped_carriers")},
                      "validator_ok": bool(best.get("validator", {}).get("ok"))}
    newq.atomic_json(work / "fallback.json", res)
    print(json.dumps(res["compose"], indent=1), flush=True)
    return 0 if res["compose"]["validator_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
