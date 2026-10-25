"""S26 P-A4 step 1: BGE-M3 dense doc-level embedding of the whole canonical corpus (887,151 docs).

Same encoder/settings as pa2_embed.py (canonical doc_text as-is, first 4096 chars, max_len 512, dense CLS L2 fp16).
The 125,605 vectors of pa2_doc_dense.npz are reused. Remaining docs are ordered: zh first, then vi by host priority
(tamanhhospital.vn, tiemchunglongchau.com.vn, medlatec.vn, vinmec.com, hellobacsi.com, then the other hosts by the share
of their canonical docs that P1-vi+10 lists for at least one query, descending), and cut into blocks of 5,000.
The order is frozen in <out>/order.parquet on the first run; each block is saved to <out>/full_emb/<k>.npz, so a restart
resumes at the first missing block. Final: <out>/full_doc_dense.npz (doc_id int64, dense fp16 [887151, 1024]).
Usage: full_embed.py <out> <pa2_dir>
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

out, pa2 = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
BLOCK, L, DOCS = 5000, 512, 'data/canonical-20261018/docs.parquet'
FIRST_HOSTS = ['tamanhhospital.vn', 'tiemchunglongchau.com.vn', 'medlatec.vn', 'vinmec.com', 'hellobacsi.com']
part = out / 'full_emb'; part.mkdir(exist_ok=True)
E = np.load(pa2 / 'pa2_doc_dense.npz'); have = set(E['doc_id'].tolist())

if not (out / 'order.parquet').exists():
    meta = pq.read_table(DOCS, columns=['doc_id', 'lang', 'host', 'body_cp']).to_pandas()
    p1 = pd.read_parquet(pa2 / 'pa2_p1_docs.parquet').dropna(subset=['doc_id'])
    meta['in_p1'] = meta.doc_id.isin(set(p1.doc_id.astype('int64')))
    share = meta[meta.lang == 'vi'].groupby('host').in_p1.mean().sort_values(ascending=False, kind='stable')
    hosts = FIRST_HOSTS + [h for h in share.index if h not in FIRST_HOSTS]
    hrank = {h: i for i, h in enumerate(hosts)}
    meta['prio'] = np.where(meta.lang == 'zh', -1, meta.host.map(hrank).fillna(len(hosts)).astype(int))
    todo = meta[~meta.doc_id.isin(have)].sort_values(['prio', 'doc_id'], kind='stable').reset_index(drop=True)
    todo['block'] = np.arange(len(todo)) // BLOCK
    todo[['doc_id', 'lang', 'host', 'prio', 'block', 'body_cp']].to_parquet(out / 'order.parquet', index=False)
    share.rename('p1_share').reset_index().to_parquet(out / 'host_p1_share.parquet', index=False)
todo = pd.read_parquet(out / 'order.parquet')
blocks = todo.block.max() + 1
missing = [k for k in range(blocks) if not (part / f'{k:04d}.npz').exists()]
print(f'reused {len(have)}; to embed {len(todo)} in {blocks} blocks; missing {len(missing)}', flush=True)

if missing:
    need = set(todo[todo.block.isin(missing)].doc_id.tolist())
    pf, ids, txt = pq.ParquetFile(DOCS), [], []
    for g in range(pf.num_row_groups):                 # only the needed texts, sliced, kept as Arrow (compact)
        t = pf.read_row_group(g, columns=['doc_id', 'doc_text'])
        t = t.filter(pc.is_in(t['doc_id'], pa.array(list(need), pa.int64())))
        ids.append(t['doc_id']); txt.append(pc.utf8_slice_codeunits(pc.fill_null(t['doc_text'], ''), 0, L * 8).cast(pa.large_string()))
    ids = pa.chunked_array(ids).to_numpy(); txt = pa.chunked_array(txt).combine_chunks()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    print(f'texts loaded: {len(ids)}', flush=True)
    enc = M3Encoder(max_len=L); t0, done = time.time(), 0
    for n, k in enumerate(missing):
        blk = todo[todo.block == k]
        texts = txt.take(pa.array(pos.loc[blk.doc_id.values].values)).to_pylist()
        order = np.argsort([len(x) for x in texts], kind='stable'); D = np.zeros((len(blk), 1024), np.float16)
        i = 0
        while i < len(order):
            bs = 64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16
            sel = order[i:i + bs]; D[sel] = enc.encode_batch([texts[j] for j in sel])[0]; i += bs
        np.savez(part / f'{k:04d}.npz', doc_id=blk.doc_id.values.astype(np.int64), dense=D)
        done += len(blk); el = time.time() - t0; left = int(todo.block.isin(missing[n + 1:]).sum())
        print(f'block {k + 1}/{blocks} lang={blk.lang.iloc[0]} host={blk.host.iloc[0]} done {done} '
              f'{el:.0f}s {done / el:.1f} doc/s eta {left / (done / el) / 3600:.2f}h', flush=True)

parts = [np.load(f) for f in sorted(part.glob('*.npz'))]
ids = np.concatenate([E['doc_id']] + [p['doc_id'] for p in parts]); D = np.concatenate([E['dense']] + [p['dense'] for p in parts])
assert len(ids) == len(set(ids.tolist())) == len(have) + len(todo), (len(ids), len(have) + len(todo))
np.savez(out / 'full_doc_dense.npz', doc_id=ids, dense=D)
print('done', len(ids), flush=True)
