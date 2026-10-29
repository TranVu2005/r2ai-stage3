"""S45: the exact public cosine order of CH-rrf (``r2ai.s36.d300.merged_top(300)``) after S44 deleted its inputs.

S44 deleted out/runs/s29-qwen-judge/query_vectors.npz and out/runs/s37/pool.npz. The S29 query vectors were a byte copy of the S26
``q_dense.npy`` / ``q_ids.npy`` (S29 prepare.json, query_vectors.meta.json in D:/_r2ai_archive/out-runs-meta.tar.xz), and those two files
are still in the S26 session scratchpad with the recorded SHA256. ``np.savez`` of them gives the S29 npz byte for byte (SHA ``eed0e104...``,
the value in the S31 pool manifest). pool.npz is then ``r2ai.s37.ingest.top_scores`` of these vectors over the S26 doc vectors + Pharmacity
(KEEP 400), and merged_top adds the 120ask segment vectors, unchanged.

  exact_top.py <s26_dir> <out_dir>
writes <out_dir>/query_vectors.npz (checked against the S31 SHA), <out_dir>/pool_scores.npz (q_ids, scores, doc_ids), <out_dir>/top300.parquet.
"""
import hashlib, io, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np  # noqa: E402
from r2ai.s36 import d300  # noqa: E402
from r2ai.s37 import ingest  # noqa: E402

S29_SHA = 'eed0e1046ffce6925987d21fdf832b99d548a7837c592ef21cf75e48cb02edca'
S26_SHA = {'q_dense.npy': '0da5df8a0823f6fd4341a6c66415bfb41ed3a13697ed1ed24cc7492c6825d72b',
           'q_ids.npy': '8e31f02595b1b2eab14ed6bc1d4138431c47bd6af1fde098e7786e0b3929d75d'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, want in S26_SHA.items():
        if sha((src / name).read_bytes()) != want:
            raise ValueError(f'{name}: SHA differs from the one S29 recorded')
    q_ids = np.load(src / 'q_ids.npy').astype(np.int64)
    q_dense = np.load(src / 'q_dense.npy')
    buf = io.BytesIO()
    np.savez(buf, q_ids=np.asarray(q_ids.tolist(), dtype=np.int64), q_dense=q_dense)   # as r2ai.s29.qwen_judge._load_query_vectors
    if sha(buf.getvalue()) != S29_SHA:
        raise ValueError('rebuilt S29 query_vectors.npz differs from the S31 manifest SHA')
    (out / 'query_vectors.npz').write_bytes(buf.getvalue())
    t0 = time.monotonic()
    ids, dense = ingest.load_doc_vectors()
    scores, rows = ingest.top_scores(q_dense, dense, ingest.KEEP)
    pool = {'q_ids': q_ids, 'scores': scores, 'doc_ids': ids[rows]}
    np.savez(out / 'pool_scores.npz', **pool)
    t1 = time.monotonic()
    top = d300.merged_top(300, pool, q_dense)
    top.to_parquet(out / 'top300.parquet', index=False)
    rep = {'query_vectors_sha256': S29_SHA, 'doc_vectors': int(len(ids)), 'pool_seconds': round(t1 - t0, 1),
           'merged_seconds': round(time.monotonic() - t1, 1), 'top_pairs': int(len(top)), 'ask_pairs': int(top.is_ask.sum())}
    (out / 'exact_top.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
