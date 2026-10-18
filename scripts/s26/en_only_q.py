"""S26 H7: EN-only-Q — the top-k (default 20) of EN-only's 50 docs/query by Qwen3-Reranker-0.6B yes-probability.

Reads <dir>/EN-only.cand.parquet and <dir>/qwen_scores.parquet; writes <dir>/EN-only-Q.zip (ZIP_LZMA dict256) + stats.
Docs are a subset of EN-only; each kept doc keeps the same chunk (best dense chunk).
"""
import argparse, hashlib, json, os, sys, zipfile
from pathlib import Path
import pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

sys.path.insert(0, 'src')
from r2ai.compose.base_io import lzma_dictionary  # noqa: E402

DATE_TIME = (2026, 10, 18, 0, 0, 0)
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--name', default='EN-only-Q'); ap.add_argument('--top', type=int, default=20)
a = ap.parse_args()
d = Path(a.dir); zpath = d / f'{a.name}.zip'
if zpath.exists():
    raise SystemExit(f'{zpath} exists')
cand = pd.read_parquet(d / 'EN-only.cand.parquet').merge(pd.read_parquet(d / 'qwen_scores.parquet'), on=['query_id', 'chunk_id'], how='left')
assert cand.qwen.notna().all() and len(cand) == 60000
ch = pd.read_parquet('data/pubmed_en/chunks.parquet', columns=['doc_ids_group', 'text'])
rows = []
for qid, g in cand.sort_values(['query_id', 'qwen', 'rank'], ascending=[True, False, True]).groupby('query_id', sort=True):
    g = g.head(a.top)
    docs = []
    for ci, dd in zip(g.chunk_id, g.doc_id):
        docs.extend(int(x) for x in (list(ch.doc_ids_group.iat[ci]) or [dd]) if int(x) not in docs)
    rows.append({'id': int(qid), 'relevant_docs': docs,
                 'relevant_chunks': [{'doc_id': int(dd), 'chunk_text': ch.text.iat[ci]} for ci, dd in zip(g.chunk_id, g.doc_id)]})
payload = ('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n]\n').encode('utf-8')
tmp = zpath.with_suffix('.zip.tmp')
with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
    info = zipfile.ZipInfo(f'{a.name}.json', date_time=DATE_TIME); info.compress_type = zipfile.ZIP_LZMA
    with z.open(info, 'w', force_zip64=True) as f:
        f.write(payload)
tmp.rename(zpath)
h = hashlib.sha256(zpath.read_bytes()).hexdigest()
top1_changed = int((cand.loc[cand.groupby('query_id').qwen.idxmax(), 'rank'] != 1).sum())
stats = {'name': a.name, 'queries': len(rows), 'reranker': 'Qwen/Qwen3-Reranker-0.6B e61197ed, BF16, yes-prob, instruction names English passage',
         'docs_per_query': a.top, 'subset_of': 'EN-only (top by Qwen yes-prob, tie by dense rank)', 'queries_top1_changed': top1_changed, 'qwen_yes_mean': float(cand.qwen.mean()),
         'qwen_yes_ge_0.5': int((cand.qwen >= 0.5).sum()), 'json_bytes': len(payload), 'json_sha256': hashlib.sha256(payload).hexdigest(),
         'zip_bytes': zpath.stat().st_size, 'zip_sha256': h}
(d / f'{a.name}.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
print(json.dumps(stats, indent=1))
