"""S26 H7: independent validator for EN-only ZIPs (native validator only checks verbatim against data/docs_vi)."""
import os
import json, sys, zipfile, struct
from collections import Counter
import pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

sys.path.insert(0, 'src')
from r2ai.compose.archive import compression_profile  # noqa: E402


def record_text(title, sections):   # mirror of r2ai.en_pubmed.build.record_text (en-pubmed worktree)
    pieces = []
    def append(t, sep=''):
        if pieces:
            pieces.append(sep)
        pieces.append(t)
    if title:
        append(str(title))
    for s in sections:
        label, text = s.get('label'), str(s.get('text') or '')
        if label:
            append(str(label), '\n\n')
            if text:
                append(text, '\n')
        elif text:
            append(text, '\n\n')
    return ''.join(pieces)


zp = sys.argv[1]
err = Counter()
z = zipfile.ZipFile(zp)
names = z.namelist()
err['zip_not_single_file'] += len(names) != 1
cp = compression_profile(zp)
err['not_dict256'] += cp['dictionary_bytes'] != 256 << 20
rows = json.loads(z.read(names[0]))
q = pd.read_parquet('data/raw/query.parquet')
corpus = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id)
rec = pd.read_parquet('data/pubmed_en/records.parquet', columns=['doc_id', 'title', 'abstract_sections'])
text = {int(d): record_text(t, list(s)) for d, t, s in zip(rec.doc_id, rec.title, rec.abstract_sections)}
ids = [r['id'] for r in rows]
err['ids_mismatch'] += sorted(ids) != sorted(q.id.tolist()) or len(set(ids)) != len(ids)
nd, nc = [], []
for r in rows:
    err['id_not_int'] += type(r['id']) is not int
    docs = r['relevant_docs']
    err['doc_not_int'] += sum(type(d) is not int for d in docs)
    err['doc_not_in_corpus'] += sum(d not in corpus for d in docs)
    err['doc_dup'] += len(docs) - len(set(docs))
    ds = set(docs)
    for c in r['relevant_chunks']:
        err['chunk_bad_keys'] += set(c) != {'doc_id', 'chunk_text'}
        err['chunk_doc_not_in_docs'] += c['doc_id'] not in ds
        t = text.get(c['doc_id'])
        err['chunk_doc_no_text'] += t is None
        err['chunk_not_verbatim'] += t is not None and c['chunk_text'] not in t
        err['chunk_empty'] += not c['chunk_text']
    nd.append(len(docs)); nc.append(len(r['relevant_chunks']))
print(json.dumps({'zip': zp, 'rows': len(rows), 'docs_per_query': dict(Counter(nd)), 'chunks_per_query': dict(Counter(nc)),
                  'compression': cp, 'errors': {k: v for k, v in err.items() if v}, 'ok': not any(err.values())}, indent=1, default=str))
