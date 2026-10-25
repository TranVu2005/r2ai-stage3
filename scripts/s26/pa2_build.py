"""S26 P-A2 step 3: compare P1-vi+10 ordering with BGE-M3 doc-level ordering, and build P1-docrank (not submitted).

P1-docrank keeps every row's relevant_docs of P1-vi+10 unchanged (same order, same IDs => Doc P/R unchanged) and keeps
the per-query chunk count, but gives the chunks to the docs ranked highest by BGE-M3 dense doc-level cosine
(canonical doc_text, 512 tokens; pa2_embed.py) instead of P1's own order. Chunk text is reused byte for byte from a chunk
P1-vi+10 already ships for the same doc_id (P1 passed the native verbatim check), one chunk per doc_id, so the
validator's doc-scope dedupe/near-duplicate rules hold. Docs without a P1 chunk text or without an embedding are skipped.
Any LB change is therefore attributable to which docs get chunks: a direct test of doc-level precision.
"""
import hashlib, json, os, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.compose.base_io import lzma_dictionary  # noqa: E402
from r2ai.compose.archive import compression_profile  # noqa: E402

ZP = 'out/runs/reranker-trial/release/P1-vi+10.zip'; CEIL = 102_600_000; DATE_TIME = (2026, 10, 19, 0, 0, 0)
out, S = Path(sys.argv[1]), Path(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else 'P1-docrank'
zpath = out / f'{name}.zip'
if zpath.exists():
    raise SystemExit(f'{zpath} exists')
rows = []
with zipfile.ZipFile(ZP).open('P1-vi+10.json') as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if line.startswith(b'{'):
            rows.append(json.loads(line))
text = {}
for r in rows:
    for c in r['relevant_chunks']:
        text.setdefault(c['doc_id'], c['chunk_text'])
p1 = pd.read_parquet(out / 'pa2_p1_docs.parquet')
m = p1.dropna(subset=['doc_id']); cid2doc = dict(zip(m.corpus_id.astype(int), m.doc_id.astype(int)))
E = np.load(out / 'pa2_doc_dense.npz'); emb = dict(zip(E['doc_id'].tolist(), range(len(E['doc_id']))))
Dm = E['dense'].astype(np.float32); Q = np.load(S / 'q_dense.npy').astype(np.float32); qids = np.load(S / 'q_ids.npy')
qpos = {int(q): i for i, q in enumerate(qids)}

stats, new_rows = [], []
for r in rows:
    qi = qpos[r['id']]
    seen, order_p1 = set(), []                         # P1 order over canonical docs
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None and d not in seen:
            seen.add(d); order_p1.append(d)
    scored = [d for d in order_p1 if d in emb]
    s = Dm[[emb[d] for d in scored]] @ Q[qi] if scored else np.zeros(0)
    order_dl = [scored[i] for i in np.argsort(-s, kind='stable')]
    top50_p1 = set([d for d in order_p1 if d in emb][:50]); top50_dl = set(order_dl[:50])
    p1_chunk_docs = {cid2doc[c['doc_id']] for c in r['relevant_chunks'] if c['doc_id'] in cid2doc}
    n_c = len(r['relevant_chunks'])
    # one corpus ID per canonical doc: the ID P1 chunked in this row, else the first relevant_docs ID that has P1 text
    own = {cid2doc[c['doc_id']]: c['doc_id'] for c in r['relevant_chunks'] if c['doc_id'] in cid2doc}
    pick = {}
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None and d not in pick and cid in text:
            pick[d] = cid
    pick.update({d: c for d, c in own.items() if c in text})
    chunks, used = [], set()
    for d in order_dl:
        if len(chunks) == n_c:
            break
        if d in pick and d not in used:
            chunks.append({'doc_id': int(pick[d]), 'chunk_text': text[pick[d]]}); used.add(d)
    new_rows.append({'id': r['id'], 'relevant_docs': r['relevant_docs'], 'relevant_chunks': chunks})
    stats.append({'query_id': r['id'], 'docs_canon': len(order_p1), 'docs_emb': len(scored), 'overlap_top50': len(top50_p1 & top50_dl),
                  'n_chunks_p1': n_c, 'n_chunks_new': len(chunks), 'chunk_doc_overlap': len(p1_chunk_docs & used),
                  'p1_top1_dl_rank': (order_dl.index(order_p1[0]) + 1) if order_p1 and order_p1[0] in emb else None})
st = pd.DataFrame(stats); st.to_parquet(out / f'{name}.query_stats.parquet', index=False)
payload = ('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in new_rows) + '\n]\n').encode('utf-8')
tmp = zpath.with_suffix('.zip.tmp')
with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
    info = zipfile.ZipInfo(f'{name}.json', date_time=DATE_TIME); info.compress_type = zipfile.ZIP_LZMA
    with z.open(info, 'w', force_zip64=True) as f:
        f.write(payload)
size = tmp.stat().st_size
if size >= CEIL:
    raise SystemExit(f'ZIP {size} >= ceiling {CEIL}; not published ({tmp})')
tmp.rename(zpath)
# checks: same ids/docs as P1, every chunk (doc_id, text) pair shipped by P1, chunk doc in relevant_docs, 1 chunk per doc_id
pairs = {(c['doc_id'], hashlib.sha256(c['chunk_text'].encode()).hexdigest()) for r in rows for c in r['relevant_chunks']}
err = {'docs_changed': 0, 'chunk_pair_not_in_p1': 0, 'chunk_doc_not_in_docs': 0, 'dup_chunk_doc': 0}
for a, b in zip(rows, new_rows):
    err['docs_changed'] += a['relevant_docs'] != b['relevant_docs'] or a['id'] != b['id']
    ids = [c['doc_id'] for c in b['relevant_chunks']]; err['dup_chunk_doc'] += len(ids) - len(set(ids))
    for c in b['relevant_chunks']:
        err['chunk_pair_not_in_p1'] += (c['doc_id'], hashlib.sha256(c['chunk_text'].encode()).hexdigest()) not in pairs
        err['chunk_doc_not_in_docs'] += c['doc_id'] not in set(b['relevant_docs'])
cp = compression_profile(zpath)
rep = {'name': name, 'base': ZP, 'queries': len(new_rows), 'zip_bytes': size, 'zip_sha256': hashlib.sha256(zpath.read_bytes()).hexdigest(),
       'json_bytes': len(payload), 'json_sha256': hashlib.sha256(payload).hexdigest(), 'ceiling': CEIL, 'lzma_props': cp['properties_hex'],
       'chunks_p1': int(st.n_chunks_p1.sum()), 'chunks_new': int(st.n_chunks_new.sum()), 'errors': err,
       'overlap_top50_mean': float(st.overlap_top50.mean()), 'overlap_top50_quantiles': st.overlap_top50.quantile([.1, .25, .5, .75, .9]).to_dict(),
       'chunk_doc_overlap_share': float(st.chunk_doc_overlap.sum() / st.n_chunks_p1.sum()),
       'p1_top1_doclevel_rank_median': float(st.p1_top1_dl_rank.median()), 'docs_with_embedding_share': float(st.docs_emb.sum() / st.docs_canon.sum())}
(out / f'{name}.build.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
print(json.dumps(rep, indent=1))
