"""S41 regression of EN-hedge (Vu 28/10, hedge copy 5 of the private: ``newq build --en-hedge 3``): on the 1,200 public queries,
``r2ai.s43.enhedge`` applied to the CH-rrf ZIP ``base`` (SHA256 b12b8591...; pass the ZIP regression a just rebuilt) with n = 3 and
min cosine 0.5226 must give the ZIP S43 delivered (EN-hedge3-c5226, SHA256 ce0fabc0...8742) byte for byte.

Two selections are checked against the delivered ``hedge.parquet``: the top-20 lane file the S43 build read (``enhedge.LANE``) and
``enhedge.lane_pairs`` recomputed from the vectors, which is what ``newq build --en-hedge`` uses on a new query set.

    python scripts/s41/regress_enhedge.py --base <CH-rrf ZIP> --out out/runs/s41/regress-28c/EN-hedge

Writes ``<out>/EN-hedge3-c5226/`` (ZIP, validator, enhedge report) and ``<out>/regress.json``."""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r2ai import newq, paths  # noqa: E402
from r2ai.s36 import enlane  # noqa: E402
from r2ai.s43 import enhedge  # noqa: E402

NAME, N, MIN_COS = "EN-hedge3-c5226", 3, 0.5226
REF_HEDGE = paths.RUNS_DIR / "s43" / "probes" / NAME / "hedge.parquet"
REF_HEDGE_SHA = "15ee1fb3ebae92d67af6101524d3b47cee24ebcd4708f17e1393620294c0350d"
REF_ZIP_SHA = "ce0fabc01cd017ff2372351355711a31affa0152940f77c4f945acebc6188742"
KEYS = ["query_id", "doc_id", "en_rank"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    work = out / NAME
    paths.assert_writable(work)
    work.mkdir(parents=True, exist_ok=True)
    if newq.sha256(REF_HEDGE) != REF_HEDGE_SHA:
        raise SystemExit(f"{REF_HEDGE}: SHA256 differs from {REF_HEDGE_SHA}")
    ref = pd.read_parquet(REF_HEDGE)
    res: dict = {"base": str(args.base), "n": N, "min_cos": MIN_COS}
    t0 = time.monotonic()
    hedge = enhedge.choose(pd.read_parquet(enhedge.LANE), N, MIN_COS)
    q_ids, q_dense = enlane.load_queries()
    recomputed = enhedge.choose(enhedge.lane_pairs(q_ids, q_dense, N), N, MIN_COS)
    res["hedge"] = {"pairs": len(hedge), "lane_file_equal_ref": bool(hedge[KEYS].equals(ref[KEYS])),
                    "recomputed_equal_ref": bool(recomputed[KEYS].reset_index(drop=True).equals(ref[KEYS])),
                    "seconds": round(time.monotonic() - t0, 1)}
    hedge.to_parquet(work / "hedge.parquet", index=False)
    rep = enhedge.run(args.base, hedge, work, f"{NAME}-full+w768-t1536", enhedge.CH_RRF_SHA)
    res["zip"] = {k: rep.get(k) for k in ("zip", "zip_bytes", "zip_sha256", "docs", "chunks", "base_docs_prefix_kept")}
    res["zip"]["validator_ok"] = rep["validator"].get("ok")
    res["zip_equal_ref"] = rep["zip_sha256"] == REF_ZIP_SHA
    res["pass"] = res["zip_equal_ref"] and res["hedge"]["lane_file_equal_ref"] and res["hedge"]["recomputed_equal_ref"]
    newq.atomic_json(out / "regress.json", res)
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
