"""S26: flatten a submission ZIP into (query_id, rank, doc_id) and (query_id, chunk_rank, doc_id, chunk_chars) parquet."""
import os, json, sys, zipfile
import pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

zp, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zp); name = z.namelist()[0]
D, C = [], []
with z.open(name) as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if not line.startswith(b'{'):
            continue
        r = json.loads(line)
        D += [(r['id'], k, d) for k, d in enumerate(r['relevant_docs'], 1)]
        C += [(r['id'], k, c['doc_id'], len(c['chunk_text'])) for k, c in enumerate(r['relevant_chunks'], 1)]
pd.DataFrame(D, columns=['query_id', 'rank', 'doc_id']).to_parquet(out + '.docs.parquet', index=False)
pd.DataFrame(C, columns=['query_id', 'rank', 'doc_id', 'chars']).to_parquet(out + '.chunks.parquet', index=False)
print(name, len(D), len(C))
