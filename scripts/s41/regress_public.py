"""S41 regression (reads the public FREE-noJina inputs ON PURPOSE; never an input of r2ai.newq): ``r2ai.newq.apply_selection`` with the
frozen private config (configs/s41-private-final.json) on the exact pools / 4B / C-max that r2ai.s36.s37.apply built for FREE-noJina
(configs/s40-abl-free-v2-noJina.json, 1,200 public queries) must give the FREE-noJina docsel and the CH-rrf carriers; with ``--compose``,
the S38 compose of that docsel + carriers (same command as scripts/s41/run_ch.sh) must give the CH-rrf ZIP byte for byte (SHA256 b12b8591...).

    python scripts/s41/regress_public.py --out out/runs/s41/regress [--compose]

Writes ``<out>/apply/`` (newq outputs), ``<out>/compose/`` and ``<out>/regress.json`` (every check, pass / fail)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r2ai import newq, paths  # noqa: E402
from r2ai.s36 import d300, fuse, s37  # noqa: E402

NAME = "REG-CH-rrf"
RUNS = paths.RUNS_DIR
REF_DOCSEL = RUNS / "s40" / "abl" / "apply" / "docsel-ABL-noJina-free-v2.parquet"
REF_CARRIERS = RUNS / "s41" / "carriers" / "carriers-CH-rrf.parquet"
REF_ZIP_SHA = "b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8"
COLUMNS = ["query_id", "doc_id", "rank", "rrf_score", "cosine", "p4b", "cmax", "cosine_rank", "judge_rank", "cmax_rank", "p4b_ge_tc", "t_c",
           "p_fused", "p_cal"]          # every docsel column except the Jina ones (Jina is not an RRF source here) and the carrier flag


def public_sources(spec: dict) -> tuple[dict, dict]:
    """The pool / C-max part of s37.apply for ``spec`` (pool fixed 300), unchanged."""
    _, union, _, _, _, _ = fuse.load_frames()
    scores_4b = s37.load_scores(spec["scores_4b"], "pair_id", "score")
    pairs = s37.read_pairs(spec["pairs_new"], scores_4b)
    union = s37.merge_union(union, pairs, scores_4b, float(spec["min_4b_coverage"]))
    pairs = pairs[pairs.pair_id.astype(int).isin(scores_4b)]
    pools, _ = s37.fixed_pools(union, int(spec["depth"]), d300.merged_top(int(spec["depth"])))
    cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
    by_pair = dict(zip(pairs.pair_id.astype(int), zip(pairs.query_id.astype(int), pairs.doc_id.astype(int))))
    keyed = [f for f in spec["cmax"] if "pair_id" not in pq.read_schema(f).names]
    for pid, value in s37.load_scores([f for f in spec["cmax"] if f not in keyed], "pair_id", "cmax").items():
        if pid in by_pair:
            cmax[by_pair[pid]] = value
    for f in keyed:
        fr = pd.read_parquet(f, columns=["query_id", "doc_id", "cmax"])
        cmax.update(zip(zip(fr.query_id.astype(int), fr.doc_id.astype(int)), fr.cmax.astype(float)))
    return pools, cmax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    out = args.out.resolve()
    apply_dir, compose_dir = out / "apply", out / "compose"
    paths.assert_writable(out)
    apply_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"config": str(newq.DEFAULT_CONFIG), "config_sha256": newq.sha256(newq.DEFAULT_CONFIG)}
    for p, want in ((REF_DOCSEL, "cea56122cab3d7921cc162ced30affa80c8b964cd6bf05970cbafe8e013fe309"),
                    (REF_CARRIERS, "f2d737c09b6d535b28f346fb8f560031722a09c985c9d76fb3cc9da67f7b3380")):
        if newq.sha256(p) != want:
            raise SystemExit(f"{p}: SHA256 differs from {want}")
    cfg = newq.load_config(None)
    spec = {**s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
    t0 = time.monotonic()
    pools, cmax = public_sources(spec)
    res["load_seconds"] = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    got = newq.apply_selection(pools, {}, cmax, cfg, ("cosine_rank", "judge_rank", "cmax_rank"), True, NAME, apply_dir)
    res["apply_seconds"] = round(time.monotonic() - t0, 1)
    res["carrier_info"] = got["carrier_info"]
    new = pd.read_parquet(apply_dir / f"docsel-{NAME}.parquet")
    ref = pd.read_parquet(REF_DOCSEL)
    a, b = new[COLUMNS].reset_index(drop=True), ref[COLUMNS].reset_index(drop=True)
    res["docsel"] = {"rows_new": len(a), "rows_ref": len(b), "equal_except_jina_and_chunk": bool(a.equals(b))}
    if not res["docsel"]["equal_except_jina_and_chunk"] and len(a) == len(b):
        res["docsel"]["columns_differing"] = [c for c in COLUMNS if not a[c].equals(b[c])]
    car_new = pd.read_parquet(apply_dir / f"carriers-{NAME}.parquet")
    car_ref = pd.read_parquet(REF_CARRIERS)
    s_new, s_ref = set(zip(car_new.query_id, car_new.doc_id)), set(zip(car_ref.query_id, car_ref.doc_id))
    res["carriers"] = {"new": len(s_new), "ref": len(s_ref), "only_new": len(s_new - s_ref), "only_ref": len(s_ref - s_new),
                       "equal": s_new == s_ref, "chunk_flag_equal": set(zip(new[new.chunk].query_id, new[new.chunk].doc_id)) == s_ref}
    res["pass_apply"] = res["docsel"]["equal_except_jina_and_chunk"] and res["carriers"]["equal"] and res["carriers"]["chunk_flag_equal"]
    newq.atomic_json(out / "regress.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if args.compose and res["pass_apply"]:
        compose_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "r2ai.s38.compose", "run", "--name", f"{NAME}-full+w768", "--docsel", str(apply_dir / f"docsel-{NAME}.parquet"),
               "--rule", "full+w768", "--carriers", str(apply_dir / f"carriers-{NAME}.parquet"), "--fit-carriers", "--rounds", str(args.rounds),
               "--out", str(compose_dir)]
        env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        t0 = time.monotonic()
        with open(out / "compose.log", "ab") as f:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(compose_dir.glob(f"{NAME}-full+w768-t*.compose.json"))]
        ok = [r for r in found if r.get("under_ceiling") and r.get("validator", {}).get("ok") and not str(r["threshold"]).startswith("1000000000")]
        best = min(ok, key=lambda r: r["threshold"]) if ok else {}
        res["compose"] = {"returncode": rc, "seconds": round(time.monotonic() - t0, 1), "zip": best.get("zip"), "zip_bytes": best.get("zip_bytes"),
                          "threshold": best.get("threshold"), "zip_sha256": best.get("zip_sha256"), "zip_equal_ch_rrf": best.get("zip_sha256") == REF_ZIP_SHA}
        newq.atomic_json(out / "regress.json", res)
        print(json.dumps(res["compose"], indent=1), flush=True)
        return 0 if res["compose"]["zip_equal_ch_rrf"] else 1
    return 0 if res["pass_apply"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
