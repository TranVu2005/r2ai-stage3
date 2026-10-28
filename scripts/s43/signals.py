"""S43 query-set signals for every pool pair (CPU, doc vectors read by memmap in blocks).

For each pool doc d: cosines to every query of the set (BGE-M3 doc-level vectors, the same as the pool cosine). Per pair (q, d):
  rr      reverse rank: 1 + #{q' : cos(q', d) > cos(q, d)} (1 = q is d's nearest query)
  csls_k  cos(q, d) - mean of d's k largest query cosines (k = 5, 10, 20; hubness correction, query term dropped: constant per query)
Per doc: hub300 = number of queries whose top-300 pool holds d.

    python scripts/s43/signals.py --pool out/runs/s43/base/pool.parquet --out out/runs/s43/base/signals.parquet
"""
import argparse
import glob
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r2ai.s37 import ingest  # noqa: E402

KS = (5, 10, 20)


def npz_memmap(path: Path, member: str) -> np.ndarray:
    """Memmap one stored (uncompressed) member of an .npz."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member + ".npy")
        if info.compress_type != 0:
            raise ValueError(f"{path}:{member} is compressed")
        off = info.header_offset
    with open(path, "rb") as f:
        f.seek(off)
        head = f.read(30)
        start = off + 30 + int.from_bytes(head[26:28], "little") + int.from_bytes(head[28:30], "little")
        f.seek(start)
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        data = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data, shape=shape, order="F" if fortran else "C")


def doc_vectors(doc_ids: np.ndarray) -> np.ndarray:
    """Rows of the S26 matrix, Pharmacity lots and S37 120ask segments for ``doc_ids`` (float32, same order)."""
    want = pd.Index(doc_ids)
    out = np.full((len(doc_ids), 1024), np.nan, np.float32)
    base_ids = npz_memmap(ingest.S26_VECTORS, "doc_id")
    dense = npz_memmap(ingest.S26_VECTORS, "dense")
    pos = want.get_indexer(np.asarray(base_ids))
    rows = np.flatnonzero(pos >= 0)
    for b in range(0, len(rows), 50_000):
        r = rows[b:b + 50_000]
        out[pos[r]] = dense[r].astype(np.float32)
    extra = [ingest.PHARMACITY_VECTORS / lot.name / "dense_doc_level.npz" for lot in sorted(ingest.PHARMACITY_DOCS.iterdir()) if lot.is_dir()]
    extra += [Path(p) for p in sorted(glob.glob(str(ingest.STORE / "*" / "vectors.npz")))]
    for f in extra:
        if not f.exists():
            continue
        with np.load(f) as z:
            p = want.get_indexer(z["doc_id"].astype(np.int64))
            m = p >= 0
            fill = m & np.isnan(out[np.where(m, p, 0), 0])
            out[p[fill]] = z["dense"][fill].astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=ingest.QUERY_VECTORS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block", type=int, default=20_000)
    args = ap.parse_args()
    pool = pd.read_parquet(args.pool, columns=["query_id", "doc_id", "cosine"])
    with np.load(args.queries) as z:
        q_ids, qv = z["q_ids"].astype(np.int64), z["q_dense"].astype(np.float32)
    q_index = pd.Index(q_ids)
    docs = np.unique(pool.doc_id.to_numpy(np.int64))
    dv = doc_vectors(docs)
    missing = int(np.isnan(dv[:, 0]).sum())
    print(f"docs {len(docs)}, without vector {missing}", flush=True)
    hub = pool.groupby("doc_id").size()
    pool = pool.assign(qi=q_index.get_indexer(pool.query_id), di=pd.Index(docs).get_indexer(pool.doc_id)).sort_values("di")
    res = {k: np.full(len(pool), np.nan) for k in ["cos_check", "rr"] + [f"top{k}" for k in KS]}
    di_all, qi_all = pool.di.to_numpy(), pool.qi.to_numpy()
    for b in range(0, len(docs), args.block):
        S = dv[b:b + args.block] @ qv.T                       # docs x queries
        top = -np.sort(-S, axis=1)[:, : max(KS)]
        lo, hi = np.searchsorted(di_all, [b, b + args.block])
        di, qi = di_all[lo:hi] - b, qi_all[lo:hi]
        c = S[di, qi]
        res["cos_check"][lo:hi] = c
        res["rr"][lo:hi] = 1 + (S[di] > c[:, None]).sum(axis=1)
        for k in KS:
            res[f"top{k}"][lo:hi] = top[di, :k].mean(axis=1)
        print(f"block {b // args.block + 1}/{-(-len(docs) // args.block)}", flush=True)
    for k, v in res.items():
        pool[k] = v
    for k in KS:
        pool[f"csls{k}"] = pool.cos_check - pool[f"top{k}"]
    pool["hub300"] = pool.doc_id.map(hub).astype(int)
    pool = pool.drop(columns=["qi", "di"]).sort_values(["query_id", "doc_id"]).reset_index(drop=True)
    pool.to_parquet(args.out, index=False)
    print(f"max |cos - pool cosine| = {np.nanmax(np.abs(pool.cos_check - pool.cosine)):.2e}; rows {len(pool)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
