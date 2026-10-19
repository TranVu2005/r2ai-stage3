"""S26 P-A3 / P-A3r / P-A4: choose docs by BGE-M3 doc-level cosine (canonical doc_text, 512 tokens), not by P1's order.

Per query, K_q = len(relevant_docs) of the same row of P1-vi+10.
  pa3   top-K_q canonical docs by cosine(query, doc) over every doc of the given doc-vector file
        (pa2_doc_dense.npz = the 125,605 docs of P1-vi+10; full_doc_dense.npz = all 887,151 => P-A4).
  pa3r  top-K_q by RRF(k=60) of the P1 row order (canonical docs, first occurrence) and the doc-level order.
No zh gate (tau on BGE chunk-level) and no language quota: doc-level cosine decides for vi and zh alike.
One corpus ID per canonical doc: the ID P1-vi+10 used for it (this row first, then any row), else the canonical doc_id
when it is in its own alias group, else the first alias. P1 IDs without a canonical doc (no vector) are not ranked.
Chunks: exactly 1 per chosen doc. If P1-vi+10 ships a chunk for that corpus ID (this row first, then any row), that text
byte for byte; otherwise the canonical chunks_t256 chunk of that doc with the highest dense cosine to the query
(canonical chunk vectors, BGE-M3 dense CLS, vectors.json alignment); docs with no vector chunk take their first chunk.
With a 6th argument `nc`, only the first n_c chosen docs (variant order) get a chunk, n_c = len(relevant_chunks) of the
same P1-vi+10 row, so the chunk count per query equals P1-vi+10 (same text rule, 1 chunk per doc).
Usage: pa3_build.py <out_dir> <scratch_dir> <doc_vectors.npz> <name> {pa3|pa3r} [nc]
"""
import hashlib, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.compose.base_io import lzma_dictionary  # noqa: E402

ZP = 'out/runs/reranker-trial/release/P1-vi+10.zip'; CEIL = 102_600_000; DATE_TIME = (2026, 10, 19, 0, 0, 0); RRF_K = 60
CAN = 'data/canonical-20261018'
out, S, vec, name, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
NC = len(sys.argv) > 6 and sys.argv[6] == 'nc'
out.mkdir(parents=True, exist_ok=True); zpath = out / f'{name}.zip'
if zpath.exists():
    raise SystemExit(f'{zpath} exists')
t0 = time.time()
rows = []
with zipfile.ZipFile(ZP).open('P1-vi+10.json') as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if line.startswith(b'{'):
            rows.append(json.loads(line))
gtext = {}                                              # corpus ID -> first chunk text P1 ships for it
for r in rows:
    for c in r['relevant_chunks']:
        gtext.setdefault(c['doc_id'], c['chunk_text'])
ct = pq.read_table(f'{CAN}/chunks_t256.parquet', columns=['doc_id', 'has_vector', 'doc_ids'])
ch_doc, ch_has = ct['doc_id'].to_numpy(), ct['has_vector'].to_numpy()
_, first = np.unique(ch_doc, return_index=True)
grp = dict(zip(ch_doc[first].tolist(), ct['doc_ids'].take(first).to_pylist())); del ct
cid2doc = {int(a): int(d) for d, ids in grp.items() for a in ids}
doc_lang = pq.read_table(f'{CAN}/docs.parquet', columns=['doc_id', 'lang']).to_pandas().set_index('doc_id').lang
p1_any = {}                                             # canonical doc -> corpus ID P1 used for it in any row
for r in rows:
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None:
            p1_any.setdefault(d, cid)


def corpus_id(d, row_ids):
    if d in row_ids:
        return row_ids[d]
    if d in p1_any:
        return p1_any[d]
    g = list(grp.get(d, [d]))
    return d if d in g else int(g[0])


E = np.load(vec); ids = E['doc_id'].astype(np.int64); Dh = E['dense']
pos = {int(d): i for i, d in enumerate(ids)}
Q = np.load(S / 'q_dense.npy').astype(np.float32); qpos = {int(q): i for i, q in enumerate(np.load(S / 'q_ids.npy'))}
M = 3 * max(len(r['relevant_docs']) for r in rows)      # per query: doc-level top-M (index, score), blockwise fp32
top_i = np.zeros((len(Q), 0), np.int64); top_s = np.zeros((len(Q), 0), np.float32)
for b in range(0, len(ids), 50_000):
    sc = Q @ Dh[b:b + 50_000].astype(np.float32).T
    m = min(M, sc.shape[1]); part = np.argpartition(-sc, m - 1, axis=1)[:, :m]
    top_i = np.c_[top_i, part + b]; top_s = np.c_[top_s, np.take_along_axis(sc, part, 1)]
    keep = np.argsort(-top_s, axis=1, kind='stable')[:, :M]
    top_i, top_s = np.take_along_axis(top_i, keep, 1), np.take_along_axis(top_s, keep, 1)
del Dh, E
print(f'loaded {len(rows)} rows, {len(ids)} doc vectors, top-{M} done {time.time() - t0:.0f}s', flush=True)

picked = []                                             # per row: list of (canonical doc, corpus ID)
stats = []
for r in rows:
    qi, K = qpos[r['id']], len(r['relevant_docs'])
    row_ids, order_p1 = {}, []
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None and d not in row_ids:
            row_ids[d] = cid; order_p1.append(d)
    if mode == 'pa3':
        chosen = [int(ids[i]) for i in top_i[qi, :K]]
    else:                                               # RRF over P1 order and doc-level order (doc-level top 3*max K)
        sc = {}
        for rk, i in enumerate(top_i[qi], 1):
            sc[int(ids[i])] = sc.get(int(ids[i]), 0) + 1 / (RRF_K + rk)
        for rk, d in enumerate(order_p1, 1):
            if d in pos:                                # rankable docs only (same rule as pa3)
                sc[d] = sc.get(d, 0) + 1 / (RRF_K + rk)
        chosen = sorted(sc, key=lambda d: -sc[d])[:K]
    picked.append([(d, corpus_id(d, row_ids)) for d in chosen])
    p1set = set(order_p1)
    stats.append({'query_id': r['id'], 'K': K, 'p1_canon': len(order_p1), 'new_vs_p1': sum(d not in p1set for d in chosen),
                  'zh_new': sum(doc_lang.get(d) == 'zh' for d in chosen), 'vi_new': sum(doc_lang.get(d) == 'vi' for d in chosen),
                  'zh_p1': sum(doc_lang.get(d) == 'zh' for d in order_p1), 'vi_p1': sum(doc_lang.get(d) == 'vi' for d in order_p1)})
print(f'docs chosen {time.time() - t0:.0f}s', flush=True)

# chunk texts: P1 text (this row, then any row) else best t256 chunk of the canonical doc by dense cosine
need = {}                                               # canonical doc -> list of row indexes needing a t256 chunk
for k, (r, pk) in enumerate(zip(rows, picked)):
    own = {c['doc_id']: c['chunk_text'] for c in r['relevant_chunks']}
    for d, cid in (pk[:len(r['relevant_chunks'])] if NC else pk):
        if cid not in own and cid not in gtext:
            need.setdefault(d, []).append(k)
vrow_all = np.where(ch_has, np.cumsum(ch_has) - 1, -1)
r_need = np.flatnonzero(np.isin(ch_doc, np.fromiter(need, np.int64)))
sub = pd.DataFrame({'index': r_need, 'doc_id': ch_doc[r_need], 'vrow': vrow_all[r_need]})   # index = chunk row
V = np.load(f'{CAN}/vectors/dense.npy', mmap_mode='r')
vr = sub.vrow.values; ok = vr >= 0
CV = np.zeros((len(sub), V.shape[1]), np.float16)
o = np.argsort(vr[ok]); rows_ok = np.flatnonzero(ok)[o]
for b in range(0, len(rows_ok), 200_000):
    sel = rows_ok[b:b + 200_000]; CV[sel] = V[vr[sel]]
print(f'chunk vectors {len(sub)} ({ok.sum()} with vector) {time.time() - t0:.0f}s', flush=True)
best_row = {}                                           # (row k, canonical doc) -> chunk row
for d, g in sub.groupby('doc_id'):
    gi = g.index.values; has = ok[gi]          # gi = positions in sub (RangeIndex)
    for k in need[d]:
        if has.any():
            sc = CV[gi[has]].astype(np.float32) @ Q[qpos[rows[k]['id']]]; best_row[(k, d)] = int(g['index'].values[has][np.argmax(sc)])
        else:
            best_row[(k, d)] = int(g['index'].values[0])
want = np.array(sorted(set(best_row.values())), np.int64); ctext = {}
pf = pq.ParquetFile(f'{CAN}/chunks_t256.parquet'); lo = 0
for g in range(pf.num_row_groups):                      # read only row groups holding wanted chunk rows
    n = pf.metadata.row_group(g).num_rows; w = want[(want >= lo) & (want < lo + n)]
    if len(w):
        ctext.update(zip(w.tolist(), pf.read_row_group(g, columns=['text'])['text'].take(w - lo).to_pylist()))
    lo += n
print(f't256 chunks {len(want)} {time.time() - t0:.0f}s', flush=True)

new_rows, src = [], {'p1_row': 0, 'p1_any': 0, 't256': 0, 'none': 0}
for k, (r, pk) in enumerate(zip(rows, picked)):
    own = {c['doc_id']: c['chunk_text'] for c in r['relevant_chunks']}
    chunks = []
    for d, cid in (pk[:len(r['relevant_chunks'])] if NC else pk):
        if cid in own:
            t = own[cid]; src['p1_row'] += 1
        elif cid in gtext:
            t = gtext[cid]; src['p1_any'] += 1
        elif (k, d) in best_row:
            t = ctext[best_row[(k, d)]]; src['t256'] += 1
        else:
            src['none'] += 1; continue
        chunks.append({'doc_id': int(cid), 'chunk_text': t})
    new_rows.append({'id': r['id'], 'relevant_docs': [int(c) for _, c in pk], 'relevant_chunks': chunks})
st = pd.DataFrame(stats); st.to_parquet(out / f'{name}.query_stats.parquet', index=False)
payload = ('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in new_rows) + '\n]\n').encode('utf-8')
tmp = zpath.with_suffix('.zip.tmp')
with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
    info = zipfile.ZipInfo(f'{name}.json', date_time=DATE_TIME); info.compress_type = zipfile.ZIP_LZMA
    with z.open(info, 'w', force_zip64=True) as f:
        f.write(payload)
size = tmp.stat().st_size
rep = {'name': name, 'mode': mode, 'chunks_rule': 'n_c of P1 row' if NC else '1 per chosen doc', 'doc_vectors': str(vec), 'n_doc_vectors': len(ids), 'base': ZP, 'queries': len(new_rows),
       'zip_bytes': size, 'json_bytes': len(payload), 'json_sha256': hashlib.sha256(payload).hexdigest(), 'ceiling': CEIL,
       'docs_total': int(st.K.sum()), 'chunks_total': sum(len(r['relevant_chunks']) for r in new_rows), 'chunk_source': src,
       'new_vs_p1_mean': float(st.new_vs_p1.mean()), 'new_vs_p1_quantiles': st.new_vs_p1.quantile([.1, .5, .9]).to_dict(),
       'zh_per_query': float(st.zh_new.mean()), 'vi_per_query': float(st.vi_new.mean()),
       'zh_per_query_p1': float(st.zh_p1.mean()), 'vi_per_query_p1': float(st.vi_p1.mean()),
       'zh_share': float(st.zh_new.sum() / st.K.sum()), 'zh_share_p1': float(st.zh_p1.sum() / (st.zh_p1 + st.vi_p1).sum())}
if size >= CEIL:
    (out / f'{name}.build.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    raise SystemExit(f'ZIP {size} >= ceiling {CEIL}; not published ({tmp})\n' + json.dumps(rep, indent=1))
tmp.rename(zpath)
rep['zip_sha256'] = hashlib.sha256(zpath.read_bytes()).hexdigest()
(out / f'{name}.build.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
print(json.dumps(rep, indent=1), flush=True)
