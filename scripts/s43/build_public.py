"""S43 public probes on the CH-rrf inputs: ``r2ai.newq.apply_selection`` with ``--w-cos-doc`` / ``--w-cos-carrier`` on the FREE-noJina pools /
4B / C-max (same inputs as scripts/s41/regress_public.py), then (``--compose``) the S38 compose of CH-rrf (full+w768, --fit-carriers).

    python scripts/s43/build_public.py --name P1-carrier-w2 --w-cos-carrier 2 --out out/runs/s43/probes [--compose]

Checks written to ``<out>/<name>/check.json``: docsel against CH-rrf (rows, order), carriers against CH-rrf (total, swapped per query),
mean / median cosine rank, P4B, C-max of the carrier set; with ``--compose`` the ZIP (SHA256, size, T, validator).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "s41"))
from r2ai import newq, paths  # noqa: E402
from r2ai.s36 import s37  # noqa: E402
import regress_public  # noqa: E402

KEYS = ("cosine_rank", "judge_rank", "cmax_rank")


def carrier_stats(car: pd.DataFrame, docsel: pd.DataFrame) -> dict:
    m = car.merge(docsel, on=["query_id", "doc_id"], how="left", validate="1:1")
    return {"carriers": len(m), "cosine_rank_mean": round(float(m.cosine_rank.mean()), 2), "cosine_rank_median": float(m.cosine_rank.median()),
            "p4b_mean": round(float(m.p4b.mean()), 4), "cmax_mean": round(float(m.cmax.mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--w-cos-doc", type=float, default=1.0)
    ap.add_argument("--w-cos-carrier", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    work = args.out.resolve() / args.name
    apply_dir, compose_dir = work / "apply", work / "compose"
    paths.assert_writable(work)
    apply_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"name": args.name, "w_cos_doc": args.w_cos_doc, "w_cos_carrier": args.w_cos_carrier, "config_sha256": newq.sha256(newq.DEFAULT_CONFIG)}
    if not (apply_dir / f"carriers-{args.name}.parquet").exists():
        cfg = newq.load_config(None)
        spec = {**s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
        pools, cmax = regress_public.public_sources(spec)
        got = newq.apply_selection(pools, {}, cmax, cfg, KEYS, True, args.name, apply_dir, w_cos_doc=args.w_cos_doc, w_cos_carrier=args.w_cos_carrier)
        res["carrier_info"] = got["carrier_info"]
    docsel = pd.read_parquet(apply_dir / f"docsel-{args.name}.parquet")
    ref_docsel = pd.read_parquet(regress_public.REF_DOCSEL)
    cols = ["query_id", "doc_id", "rank"]
    res["docsel"] = {"rows": len(docsel), "rows_ch_rrf": len(ref_docsel),
                     "equal_ch_rrf_docs_and_order": bool(docsel[cols].reset_index(drop=True).equals(ref_docsel[cols].reset_index(drop=True))),
                     "same_doc_set": set(zip(docsel.query_id, docsel.doc_id)) == set(zip(ref_docsel.query_id, ref_docsel.doc_id)),
                     "sha256": newq.sha256(apply_dir / f"docsel-{args.name}.parquet")}
    car = pd.read_parquet(apply_dir / f"carriers-{args.name}.parquet")
    ref = pd.read_parquet(regress_public.REF_CARRIERS)
    new_k, ref_k = set(zip(car.query_id, car.doc_id)), set(zip(ref.query_id, ref.doc_id))
    nq = int(ref.query_id.nunique())
    res["carriers"] = {"total": len(new_k), "total_ch_rrf": len(ref_k), "swapped": len(new_k - ref_k),
                       "swapped_per_query": round(len(new_k - ref_k) / nq, 2),
                       "per_query_count_equal_ch_rrf": bool(car.groupby("query_id").size().equals(ref.groupby("query_id").size())),
                       "set": carrier_stats(car, docsel), "set_ch_rrf": carrier_stats(ref, ref_docsel),
                       "sha256": newq.sha256(apply_dir / f"carriers-{args.name}.parquet")}
    newq.atomic_json(work / "check.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if args.compose:
        compose_dir.mkdir(parents=True, exist_ok=True)
        base = f"{args.name}-full+w768"
        cmd = [sys.executable, "-u", "-m", "r2ai.s38.compose", "run", "--name", base, "--docsel", str(apply_dir / f"docsel-{args.name}.parquet"),
               "--rule", "full+w768", "--carriers", str(apply_dir / f"carriers-{args.name}.parquet"), "--fit-carriers", "--rounds", str(args.rounds),
               "--out", str(compose_dir)]
        env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        t0 = time.monotonic()
        with open(work / "compose.log", "ab") as f:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(compose_dir.glob(f"{base}-t*.compose.json"))]
        ok = [r for r in found if r.get("under_ceiling") and r.get("validator", {}).get("ok") and not str(r["threshold"]).startswith("1000000000")]
        best = min(ok, key=lambda r: r["threshold"]) if ok else {}
        res["compose"] = {"returncode": rc, "seconds": round(time.monotonic() - t0, 1),
                          **{k: best.get(k) for k in ("zip", "zip_bytes", "zip_sha256", "threshold", "dropped_carriers", "docs", "chunks")},
                          "validator_ok": bool(best.get("validator", {}).get("ok"))}
        newq.atomic_json(work / "check.json", res)
        print(json.dumps(res["compose"], indent=1), flush=True)
        return 0 if res["compose"]["validator_ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
