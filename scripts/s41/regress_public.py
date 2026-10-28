"""S41 regression (reads the public FREE-noJina inputs ON PURPOSE; never an input of r2ai.newq): ``r2ai.newq.apply_selection`` with the
frozen private config (configs/s41-private-final.json) on the exact pools / 4B / C-max that r2ai.s36.s37.apply built for FREE-noJina
(configs/s40-abl-free-v2-noJina.json, 1,200 public queries) must give the FREE-noJina docsel and the CH-rrf carriers; with ``--compose``,
the S38 compose of that docsel + carriers (same command as scripts/s41/run_ch.sh) must give the CH-rrf ZIP byte for byte (SHA256 b12b8591...).

``--variant COS`` (Vu 27/10, fallback "no 4B at all"): the ``newq build --cos-only`` path (``load_config(COS_CONFIG)``, ``cos_only_map``,
``pools_from`` with no 4B score) on the same public pools / C-max must give the docsel / carriers of the fallback probe FB-COS
(scripts/s41/fallback_public.py) and, composed, its ZIP byte for byte (SHA256 1c5600bb...).

``--variant COSK150`` (Vu 28/10, replaces COS in the fallback chain): the ``newq build --cos-k 150`` path (``cos_only_map`` of
configs/s41-private-fallback-cos-k150.json, ``cos_k_keep``) must give FB-COSK150 (docsel SHA256 equal, ZIP fd331252...).

    python scripts/s41/regress_public.py --out out/runs/s41/regress [--variant CH-rrf|COS|COSK150] [--compose]

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

RUNS = paths.RUNS_DIR
# variant: (name = the reference's, so the JSON inside the ZIP has the same file name; docsel, its SHA256; carriers, its SHA256; ZIP SHA256)
REFS = {
    "CH-rrf": ("CH-rrf", RUNS / "s40" / "abl" / "apply" / "docsel-ABL-noJina-free-v2.parquet",
               "cea56122cab3d7921cc162ced30affa80c8b964cd6bf05970cbafe8e013fe309",
               RUNS / "s41" / "carriers" / "carriers-CH-rrf.parquet", "f2d737c09b6d535b28f346fb8f560031722a09c985c9d76fb3cc9da67f7b3380",
               "b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8"),
    "COS": ("FB-COS", RUNS / "s41" / "fallback" / "COS" / "apply" / "docsel-FB-COS.parquet",
            "a0df57cfd44c0a2004b47c8258b1ee74da5a80ea13c7d044d3948681b09f0660",
            RUNS / "s41" / "fallback" / "COS" / "apply" / "carriers-FB-COS.parquet", "9d095c902b08594d47e4893b053dbb6bd120d561da1b61407aa42104536904a1",
            "1c5600bbd54b80fc2095927d583cc37635b0920940503b172feda0b270790fc1"),
    "COSK150": ("FB-COSK150", RUNS / "s41" / "fallback" / "COSK150" / "apply" / "docsel-FB-COSK150.parquet",
                "e7a5891780f6332a64b417545fa4f0e40a0b7b4aa52f0b9a224873c02afca69c",
                RUNS / "s41" / "fallback" / "COSK150" / "apply" / "carriers-FB-COSK150.parquet",
                "30fc4bd2e623d7b478d0d2a910feb408e75d0fb9609a6d2e4bb770bc1eee22d1",
                "fd33125209bc508eadc459fa569a23780b3d58e8f5dc833f9b04758a3d93c2fb"),
}
COS_K = {"COSK150": 150}
REF_DOCSEL, REF_CARRIERS, REF_ZIP_SHA = REFS["CH-rrf"][1], REFS["CH-rrf"][3], REFS["CH-rrf"][5]   # read by fallback_public.py
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
    parser.add_argument("--variant", choices=sorted(REFS), default="CH-rrf")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    out = args.out.resolve()
    apply_dir, compose_dir = out / "apply", out / "compose"
    paths.assert_writable(out)
    apply_dir.mkdir(parents=True, exist_ok=True)
    name, ref_docsel, docsel_sha, ref_carriers, carriers_sha, ref_zip_sha = REFS[args.variant]
    cfg_path = (Path(str(newq.COS_K_CONFIG).format(k=COS_K[args.variant])) if args.variant in COS_K
                else newq.COS_CONFIG if args.variant == "COS" else newq.DEFAULT_CONFIG)
    res: dict = {"variant": args.variant, "config": str(cfg_path), "config_sha256": newq.sha256(cfg_path)}
    for p, want in ((ref_docsel, docsel_sha), (ref_carriers, carriers_sha)):
        if newq.sha256(p) != want:
            raise SystemExit(f"{p}: SHA256 differs from {want}")
    cfg = newq.load_config(cfg_path)
    spec = {**s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
    t0 = time.monotonic()
    pools, cmax = public_sources(spec)
    keep_fn = None
    if args.variant != "CH-rrf":                # what build --cos-only / --cos-k does: no 4B score, every pair from the pinned map
        from fallback_public import top_frame
        impute = newq.cos_only_map(cfg)
        res["map"] = impute
        pools = newq.pools_from(top_frame(pools)[0], {}, impute)
        if args.variant in COS_K:
            keep_fn = newq.cos_k_keep(cfg, COS_K[args.variant])
    res["load_seconds"] = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    got = newq.apply_selection(pools, {}, cmax, cfg, ("cosine_rank", "judge_rank", "cmax_rank"), True, name, apply_dir, keep_fn)
    res["apply_seconds"] = round(time.monotonic() - t0, 1)
    res["carrier_info"] = got["carrier_info"]
    new = pd.read_parquet(apply_dir / f"docsel-{name}.parquet")
    ref = pd.read_parquet(ref_docsel)
    a, b = new[COLUMNS].reset_index(drop=True), ref[COLUMNS].reset_index(drop=True)
    res["docsel"] = {"rows_new": len(a), "rows_ref": len(b), "equal_except_jina_and_chunk": bool(a.equals(b))}
    if not res["docsel"]["equal_except_jina_and_chunk"] and len(a) == len(b):
        res["docsel"]["columns_differing"] = [c for c in COLUMNS if not a[c].equals(b[c])]
    if args.variant != "CH-rrf":
        res["docsel"]["sha256_equal"] = newq.sha256(apply_dir / f"docsel-{name}.parquet") == docsel_sha
    car_new = pd.read_parquet(apply_dir / f"carriers-{name}.parquet")
    car_ref = pd.read_parquet(ref_carriers)
    s_new, s_ref = set(zip(car_new.query_id, car_new.doc_id)), set(zip(car_ref.query_id, car_ref.doc_id))
    res["carriers"] = {"new": len(s_new), "ref": len(s_ref), "only_new": len(s_new - s_ref), "only_ref": len(s_ref - s_new),
                       "equal": s_new == s_ref, "chunk_flag_equal": set(zip(new[new.chunk].query_id, new[new.chunk].doc_id)) == s_ref}
    res["pass_apply"] = res["docsel"]["equal_except_jina_and_chunk"] and res["carriers"]["equal"] and res["carriers"]["chunk_flag_equal"]
    newq.atomic_json(out / "regress.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if args.compose and res["pass_apply"]:
        compose_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "r2ai.s38.compose", "run", "--name", f"{name}-full+w768", "--docsel", str(apply_dir / f"docsel-{name}.parquet"),
               "--rule", "full+w768", "--carriers", str(apply_dir / f"carriers-{name}.parquet"), "--fit-carriers", "--rounds", str(args.rounds),
               "--out", str(compose_dir)]
        env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        t0 = time.monotonic()
        with open(out / "compose.log", "ab") as f:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(compose_dir.glob(f"{name}-full+w768-t*.compose.json"))]
        ok = [r for r in found if r.get("under_ceiling") and r.get("validator", {}).get("ok") and not str(r["threshold"]).startswith("1000000000")]
        best = min(ok, key=lambda r: r["threshold"]) if ok else {}
        res["compose"] = {"returncode": rc, "seconds": round(time.monotonic() - t0, 1), "zip": best.get("zip"), "zip_bytes": best.get("zip_bytes"),
                          "threshold": best.get("threshold"), "zip_sha256": best.get("zip_sha256"), "zip_equal_ref": best.get("zip_sha256") == ref_zip_sha}
        newq.atomic_json(out / "regress.json", res)
        print(json.dumps(res["compose"], indent=1), flush=True)
        return 0 if res["compose"]["zip_equal_ref"] else 1
    return 0 if res["pass_apply"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
