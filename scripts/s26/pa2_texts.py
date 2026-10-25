"""S26 P-A2 step 1: texts for the docs of P1-vi+10.

Maps every relevant_docs ID of P1-vi+10 to its canonical doc (alias groups via chunks_t256.doc_ids), reads canonical
doc_text, and collects the chunk_text P1 already ships for each doc_id (natively validated). Reports how often canonical
doc_text is byte-identical to the P1 chunk text. Writes <out>/pa2_docs.parquet and <out>/pa2_p1_chunks.parquet.
"""
import json, os, sys, zipfile
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ZP = 'out/runs/reranker-trial/release/P1-vi+10.zip'
rows, chunks = [], {}
with zipfile.ZipFile(ZP).open('P1-vi+10.json') as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if not line.startswith(b'{'):
            continue
        r = json.loads(line)
        rows += [(r['id'], k, d) for k, d in enumerate(r['relevant_docs'], 1)]
        for c in r['relevant_chunks']:
            chunks.setdefault(c['doc_id'], c['chunk_text'])
p1 = pd.DataFrame(rows, columns=['query_id', 'rank', 'corpus_id'])
ch = pq.read_table('data/canonical-20261018/chunks_t256.parquet', columns=['doc_id', 'doc_ids']).to_pandas().drop_duplicates('doc_id')
al = ch.explode('doc_ids'); cid2doc = dict(zip(al.doc_ids.astype('int64'), al.doc_id))
p1['doc_id'] = p1.corpus_id.map(cid2doc)
p1.to_parquet(out / 'pa2_p1_docs.parquet', index=False)
need = sorted(set(p1.doc_id.dropna().astype('int64')))
t = pq.read_table('data/canonical-20261018/docs.parquet', columns=['doc_id', 'lang', 'host', 'doc_text'],
                  filters=[('doc_id', 'in', need)]).to_pandas()
t.to_parquet(out / 'pa2_docs.parquet', index=False)
pd.DataFrame([(k, v) for k, v in chunks.items()], columns=['corpus_id', 'chunk_text']).to_parquet(out / 'pa2_p1_chunks.parquet', index=False)
txt = dict(zip(t.doc_id, t.doc_text))
eq = [(txt.get(cid2doc.get(cid)) == v) for cid, v in chunks.items() if cid2doc.get(cid) in txt]
print(json.dumps({'p1_rows': len(p1), 'p1_unique_ids': int(p1.corpus_id.nunique()), 'unmapped_ids': int(p1.doc_id.isna().sum()),
                  'canonical_docs': len(t), 'lang': t.lang.value_counts().to_dict(), 'p1_chunk_ids': len(chunks),
                  'chunk_vs_canonical_compared': len(eq), 'byte_identical': int(sum(eq))}, indent=1))
