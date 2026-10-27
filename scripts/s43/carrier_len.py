"""S43 offline: body length of carriers (ideas #3 / #4) on the CH-rrf pool.

Body tokens per doc from the S38 compose caches already built (``body_tokens``, BGE-M3 tokens of the canonical body; a doc's body does not
depend on the query). For the carrier swaps whose Chunk F2 is on the LB (CH-rrf / CH-ef2 / CH-cmax against the P4B order) and for the
S43 probes, the body length of the carriers moved in / out: share <= 768, 769..1536 (one full-body chunk), > 1536 (body + 768 window).

    python scripts/s43/carrier_len.py --base out/runs/s43/base --probes out/runs/s43/probes
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("D:/GitHub/r2ai-stage3/out/runs")
CACHES = ["s41/ch/compose/CH-rrf-full+w768.cache.parquet", "s41/ch/compose/CH-cmax-full+w768.cache.parquet",
          "s41/ch/compose/CH-ef2-full+w768.cache.parquet", "s40/abl/compose/ABL-noJina-free-v2-full+w768.cache.parquet",
          "s40/cut/compose/FREE-cut130-full+w768.cache.parquet", "s40/cut/compose/FREE-cut150-full+w768.cache.parquet",
          "s41/d400/compose/FREE-noJina-D400-full+w768.cache.parquet"]


def body_tokens() -> pd.Series:
    parts = [pd.read_parquet(RUNS / f, columns=["doc_id", "body_tokens"]) for f in CACHES if (RUNS / f).exists()]
    bt = pd.concat(parts).query("body_tokens >= 0").drop_duplicates("doc_id")
    return bt.set_index("doc_id").body_tokens


def profile(x: pd.DataFrame) -> dict:
    b = x.body_tokens.dropna()
    return {"n_per_query": round(len(x) / 1200, 2), "covered": round(float(x.body_tokens.notna().mean()), 3),
            "median": float(b.median()) if len(b) else None, "le768": round(float((b <= 768).mean()), 3),
            "769_1536": round(float(((b > 768) & (b <= 1536)).mean()), 3), "gt1536": round(float((b > 1536).mean()), 3),
            "cosine_rank": round(float(x.cosine_rank.mean()), 1), "p4b": round(float(x.p4b.mean()), 3), "cmax": round(float(x.cmax.mean()), 3)}


def keys(f: Path) -> set:
    c = pd.read_parquet(f, columns=["query_id", "doc_id"])
    return set(zip(c.query_id.astype(int), c.doc_id.astype(int)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    args = ap.parse_args()
    pool = pd.read_parquet(args.base / "pool.parquet")
    pool["body_tokens"] = pool.doc_id.map(body_tokens())
    key = pd.Series(list(zip(pool.query_id, pool.doc_id)))
    sets = {"P4B-order": RUNS / "s40/abl/apply/carriers-ABL-noJina-free-v2.parquet", "CH-rrf": RUNS / "s41/carriers/carriers-CH-rrf.parquet",
            "CH-ef2": RUNS / "s41/carriers/carriers-CH-ef2.parquet", "CH-cmax": RUNS / "s41/carriers/carriers-CH-cmax.parquet"}
    for f in sorted(glob.glob(str(args.probes / "*" / "apply" / "carriers-*.parquet"))):
        sets[Path(f).stem.replace("carriers-", "")] = Path(f)
    member = {n: key.isin(keys(f)).to_numpy() for n, f in sets.items()}
    res = {"docs_all": profile(pool[pool.chosen]), "sets": {n: profile(pool[m]) for n, m in member.items()}, "swaps": {}}
    for new, old in (("CH-rrf", "P4B-order"), ("CH-ef2", "P4B-order"), ("CH-cmax", "P4B-order"),
                     *((n, "CH-rrf") for n in sets if n not in ("P4B-order", "CH-rrf", "CH-ef2", "CH-cmax"))):
        a, b = member[new], member[old]
        res["swaps"][f"{new} vs {old}"] = {"in": profile(pool[a & ~b]), "out": profile(pool[b & ~a])}
    # carrier order cos + C-max (no 4B), idea #3, simulated as in scripts/s43/cosweight.py
    ncar = pool[pool.carrier].groupby("query_id").size()
    s = (1 / (60 + pool.cosine_rank) + 1 / (60 + pool.cmax_rank) - 1e-12 * pool.pos).where(pool.chosen, -np.inf)
    nc = (s.groupby(pool.query_id).rank(ascending=False, method="first") <= pool.query_id.map(ncar)).to_numpy()
    res["swaps"]["sim cos+cmax vs CH-rrf"] = {"in": profile(pool[nc & ~pool.carrier.to_numpy()]), "out": profile(pool[pool.carrier.to_numpy() & ~nc])}
    (args.probes / "carrier_len.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
