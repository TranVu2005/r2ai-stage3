"""Build a submission for the 1,200 test queries with one retrieval config (see retrieve/run.py).

    python -m submission.build --name sub01_vi_k5_c2 --w 0.7 --agg max --rerank --k 5 --c 2

Per query:
  relevant_docs   = the top-K docs (after rerank when --rerank), each expanded to its whole doc_ids_group
                    (http/https variants of the same page), in rank order, de-duplicated.
  relevant_chunks = for each of the top-K docs, its top-C chunks of field answer/body, ranked by reranker score
                    (or hybrid score without --rerank; chunks never retrieved then rank last, in doc order);
                    {"doc_id": primary doc_id (= doc_ids_group[0]), "chunk_text": verbatim chunk text}.
Writes out/submissions/<name>.json and <name>.zip (that single file), then runs eval.validate on the ZIP.
"""
from __future__ import annotations

from r2ai.paths import CHUNKS_DIR, OUT_DIR, RAW_DATA_DIR, ROOT, auxiliary_disabled

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')

from r2ai.submit.writer import write_submission  # noqa: E402


def main(argv=None):
    auxiliary_disabled()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--target', type=int, default=256)
    ap.add_argument('--w', type=float, required=True)
    ap.add_argument('--agg', choices=('max', 'sum3'), required=True)
    ap.add_argument('--rerank', action='store_true')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--c', type=int, default=2)
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    a = ap.parse_args(argv)
    import torch
    from r2ai.index.bge_m3 import M3Encoder, Reranker
    from r2ai.retrieve.run import Index, rank_docs, rerank_order, ws

    t_start = time.time()
    qs_rows = pq.read_table(a.queries).to_pylist()
    index = Index(a.target)
    groups = {}
    for r in pq.read_table(CHUNKS_DIR / 'docs.parquet', columns=['doc_id', 'doc_ids_group']).to_pylist():
        groups[r['doc_id']] = r['doc_ids_group']

    enc = M3Encoder(max_len=512)
    qd, qsp = [], []
    for b in range(0, len(qs_rows), 32):
        d, s = enc.encode_batch([ws(r['query']) for r in qs_rows[b:b + 32]])
        qd.append(d)
        qsp += s
    qd = np.concatenate(qd)
    del enc
    torch.cuda.empty_cache()
    rr = Reranker(max_len=512) if a.rerank else None

    out, t_q = [], []
    n_docs, n_chunks = [], []
    for i, r in enumerate(qs_rows):
        t0 = time.perf_counter()
        q = ws(r['query'])
        ch, hs, ranked, _ = rank_docs(index, *index.candidates(qd[i], qsp[i]), a.w, a.agg)
        hyb = dict(zip(ch.tolist(), hs.tolist()))
        if rr is not None:
            top = set(ranked)
            pairs = [int(c) for c in ch if int(index.doc_id[c]) in top]
            s = dict(zip(pairs, rr.score(q, index.texts(pairs)).tolist()))
            ranked = rerank_order(index, ch, ranked, s)
        top_docs = ranked[:a.k]
        chunks = []
        for d in top_docs:
            cs = [int(c) for c in index.chunks_of(d) if index.submittable[c]]
            if rr is not None:
                new = [c for c in cs if c not in s]
                s.update(zip(new, rr.score(q, index.texts(new)).tolist() if new else []))
                sc = {c: s[c] for c in cs}
            else:
                sc = {c: hyb.get(c, -1e9) for c in cs}
            cs.sort(key=lambda c: -sc[c])
            chunks += [{'doc_id': int(d), 'chunk_text': t} for t in index.texts(cs[:a.c])]
        docs = list(dict.fromkeys(int(x) for d in top_docs for x in groups[d]))
        out.append({'id': int(r['id']), 'relevant_docs': docs, 'relevant_chunks': chunks})
        t_q.append(time.perf_counter() - t0)
        n_docs.append(len(docs))
        n_chunks.append(len(chunks))
        if i % 100 == 0:
            print(f'{i}/{len(qs_rows)} {t_q[-1]:.2f}s', flush=True)

    sub_dir = OUT_DIR / 'submissions'
    sub_dir.mkdir(parents=True, exist_ok=True)
    js = sub_dir / f'{a.name}.json'
    write_submission(out, js)
    zp = sub_dir / f'{a.name}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(js, js.name)
    stats = {'name': a.name, 'target': a.target, 'w': a.w, 'agg': a.agg, 'rerank': a.rerank, 'k': a.k, 'c': a.c,
             'queries': len(out), 'seconds_total': round(time.time() - t_start),
             'seconds_per_query_mean': round(float(np.mean(t_q)), 3),
             'relevant_docs_per_query': {int(k): int(v) for k, v in zip(*np.unique(n_docs, return_counts=True))},
             'relevant_chunks_per_query': {int(k): int(v) for k, v in zip(*np.unique(n_chunks, return_counts=True))},
             'index_meta': index.meta, 'n_docs_in_index': len(groups)}
    (sub_dir / f'{a.name}.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))
    rc = subprocess.call([sys.executable, '-m', 'r2ai.eval.validate', str(zp)], cwd=ROOT)
    print('validator exit code', rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
