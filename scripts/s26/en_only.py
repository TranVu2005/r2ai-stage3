"""S26 H7: EN-only probe. Top-50 PubMed docs/query by BGE-M3 dense (max chunk), one best chunk per doc.

Reads the frozen en index (data/index/en_pubmed) and query dense vectors encoded with r2ai.index.bge_m3.M3Encoder
(max_len 512, CLS, L2). Writes a ZIP_LZMA dict256 submission plus a candidate parquet and a stats sidecar under --out.
No vi/zh rows. Read-only on data/.
"""
import os
import argparse, hashlib, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

sys.path.insert(0, 'src')
from r2ai.compose.base_io import lzma_dictionary  # noqa: E402

DATE_TIME = (2026, 10, 18, 0, 0, 0)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qdense', required=True)
    ap.add_argument('--qids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='EN-only')
    ap.add_argument('--top-docs', type=int, default=50)
    ap.add_argument('--top-chunks', type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zpath = out / f'{a.name}.zip'
    if zpath.exists():
        raise SystemExit(f'{zpath} exists')
    q = torch.from_numpy(np.load(a.qdense).astype(np.float16)).cuda()
    qids = np.load(a.qids)
    dense = np.load('data/index/en_pubmed/dense.npy', mmap_mode='r')
    ch = pd.read_parquet('data/pubmed_en/chunks.parquet', columns=['chunk_id', 'doc_id', 'doc_ids_group', 'chunk_order', 'text', 'title_only'])
    assert (ch.chunk_id.values == np.arange(len(ch))).all() and len(ch) == len(dense)
    t0 = time.time()
    K = a.top_chunks
    best_s = torch.full((len(q), K), -9.0, device='cuda'); best_i = torch.zeros((len(q), K), dtype=torch.long, device='cuda')
    for s in range(0, len(dense), 100_000):
        blk = torch.from_numpy(np.ascontiguousarray(dense[s:s + 100_000])).cuda()
        sc = (q @ blk.T).float()
        cs = torch.cat([best_s, sc], 1)
        ci = torch.cat([best_i, torch.arange(s, s + len(blk), device='cuda').expand(len(q), -1)], 1)
        best_s, idx = cs.topk(K, dim=1)
        best_i = ci.gather(1, idx)
    best_s, best_i = best_s.cpu().numpy(), best_i.cpu().numpy()
    print(f'dense scan {time.time() - t0:.1f}s', flush=True)
    doc_of = ch.doc_id.values
    rows, cand = [], []
    short = 0
    for qi, qid in enumerate(qids):
        seen = {}
        for s_, ci in zip(best_s[qi], best_i[qi]):   # descending score, first hit per doc = max chunk
            d = int(doc_of[ci])
            if d not in seen:
                seen[d] = (float(s_), int(ci))
                if len(seen) == a.top_docs:
                    break
        short += len(seen) < a.top_docs
        docs, chunks = [], []
        for rank, (d, (s_, ci)) in enumerate(seen.items(), 1):
            grp = [int(x) for x in ch.doc_ids_group.iat[ci]] or [d]
            docs.extend(x for x in grp if x not in docs)
            chunks.append({'doc_id': d, 'chunk_text': ch.text.iat[ci]})
            cand.append((int(qid), rank, d, ci, int(ch.chunk_order.iat[ci]), bool(ch.title_only.iat[ci]), s_))
        rows.append({'id': int(qid), 'relevant_docs': docs, 'relevant_chunks': chunks})
    pd.DataFrame(cand, columns=['query_id', 'rank', 'doc_id', 'chunk_id', 'chunk_order', 'title_only', 'dense']).to_parquet(out / f'{a.name}.cand.parquet', index=False)
    payload = ('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n]\n').encode('utf-8')
    jsha = hashlib.sha256(payload).hexdigest()
    tmp = zpath.with_suffix('.zip.tmp')
    with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
        info = zipfile.ZipInfo(f'{a.name}.json', date_time=DATE_TIME); info.compress_type = zipfile.ZIP_LZMA
        with z.open(info, 'w', force_zip64=True) as f:
            f.write(payload)
    tmp.rename(zpath)
    stats = {'name': a.name, 'queries': len(rows), 'docs_per_query': a.top_docs, 'queries_short': short,
             'docs_total': sum(len(r['relevant_docs']) for r in rows), 'chunks_total': sum(len(r['relevant_chunks']) for r in rows),
             'retriever': 'BGE-M3 dense CLS fp16, exact inner product over data/index/en_pubmed (581102 chunks), doc = max chunk',
             'query_text': 'raw Vietnamese query, M3Encoder max_len 512',
             'json_bytes': len(payload), 'json_sha256': jsha, 'zip_bytes': zpath.stat().st_size, 'zip_sha256': sha(zpath),
             'index_dense_sha256_meta': json.load(open('data/index/en_pubmed/meta.json'))['dense_sha256']}
    (out / f'{a.name}.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
