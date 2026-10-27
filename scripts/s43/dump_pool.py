"""S43 offline base: the full public pool of CH-rrf (1,200 queries x 300 docs) as one table.

Runs ``r2ai.newq.apply_selection`` with the frozen private config on the FREE-noJina pools / 4B / C-max (the same inputs as
scripts/s41/regress_public.py), checks the carriers equal CH-rrf, and writes ``<out>/pool.parquet``: query_id, doc_id, pos (1-based RRF
position), cosine, p4b, cmax, cosine_rank, judge_rank, cmax_rank, rrf_score, p_cal, chosen, carrier.

    python scripts/s43/dump_pool.py --out out/runs/s43/base
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "s41"))
from r2ai import newq, paths  # noqa: E402
from r2ai.s36 import s37  # noqa: E402
import regress_public  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    paths.assert_writable(out)
    (out / "apply").mkdir(parents=True, exist_ok=True)
    cfg = newq.load_config(None)
    spec = {**s37.default_inputs(), **json.loads((ROOT / "configs" / "s40-abl-free-v2-noJina.json").read_text(encoding="utf-8"))}
    pools, cmax = regress_public.public_sources(spec)
    got = newq.apply_selection(pools, {}, cmax, cfg, ("cosine_rank", "judge_rank", "cmax_rank"), True, "S43-base", out / "apply")
    frame, cal, keep = got["frame"], got["calibrated"], got["keep"]
    car = {(int(q), int(d)) for q, d in zip(got["carriers"].query_id, got["carriers"].doc_id)}
    ref = pd.read_parquet(regress_public.REF_CARRIERS)
    same = car == set(zip(ref.query_id.astype(int), ref.doc_id.astype(int)))
    rows = []
    for q in frame.qids:
        for i, (r, p) in enumerate(zip(frame.order[q], cal.p[q]), 1):
            d = int(r["doc_id"])
            rows.append((q, d, i, float(r["cosine_score"]), float(r["judge_score"]), float(r.get("cmax_score", float("nan"))),
                         int(r["cosine_rank"]), int(r["judge_rank"]), int(r["cmax_rank"]), float(p), i <= keep[q], (q, d) in car))
    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "pos", "cosine", "p4b", "cmax", "cosine_rank", "judge_rank", "cmax_rank",
                                     "p_cal", "chosen", "carrier"])
    df.to_parquet(out / "pool.parquet", index=False)
    info = {"rows": len(df), "chosen": int(df.chosen.sum()), "carriers": int(df.carrier.sum()), "carriers_equal_ch_rrf": same,
            "g_sum": float(sum(cal.g[q] for q in frame.qids))}
    (out / "pool.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
