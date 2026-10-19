"""S26 P-A3/P-A4: validator for doc-level ZIPs.

Format: single JSON file in a ZIP_LZMA dict 256 MiB under 102,600,000 B; exactly the 1,200 query ids, ints, once each;
relevant_docs ints in links_corpus_vi_en_zh, no duplicates; chunks {doc_id, chunk_text}, non-empty, doc_id in the row's
relevant_docs, at most 1 chunk per doc_id per row.
Verbatim: a chunk passes if its (doc_id, chunk_text) pair is shipped by P1-vi+10 (natively validated when P1 passed),
or if chunk_text is a substring of the canonical doc_text of its canonical doc (canonical-20261018, S25-verified).
Usage: validate_pa3.py <zip>
"""
import hashlib, json, os, sys, zipfile
from collections import Counter
import pandas as pd, pyarrow.parquet as pq, numpy as np
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.compose.archive import compression_profile  # noqa: E402

CEIL, ZP, CAN = 102_600_000, 'out/runs/reranker-trial/release/P1-vi+10.zip', 'data/canonical-20261018'
zp = sys.argv[1]; err = Counter()
z = zipfile.ZipFile(zp); names = z.namelist()
err['zip_not_single_file'] += len(names) != 1
cp = compression_profile(zp)
err['not_dict256'] += cp['dictionary_bytes'] != 256 << 20
err['zip_over_ceiling'] += os.path.getsize(zp) >= CEIL
head = z.open(names[0]).read(2); err['not_json_list'] += head != b'[\n'


def stream_rows():                                      # one row per line (compose format), parsed lazily
    with z.open(names[0]) as f:
        for line in f:
            line = line.strip().rstrip(b',')
            if line.startswith(b'{'):
                yield json.loads(line)


pairs = set()
with zipfile.ZipFile(ZP).open('P1-vi+10.json') as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if line.startswith(b'{'):
            for c in json.loads(line)['relevant_chunks']:
                pairs.add((c['doc_id'], hashlib.sha256(c['chunk_text'].encode()).digest()))
q = pd.read_parquet('data/raw/query.parquet')
corpus = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id.tolist())
ct = pq.read_table(f'{CAN}/chunks_t256.parquet', columns=['doc_id', 'doc_ids'])
d_all = ct['doc_id'].to_numpy(); _, first = np.unique(d_all, return_index=True)
cid2doc = {int(a): int(d) for d, ids in zip(d_all[first].tolist(), ct['doc_ids'].take(first).to_pylist()) for a in ids}; del ct
ids, nd, nc, todo, src = [], [], [], [], Counter()
for r in stream_rows():
    ids.append(r['id']); nd.append(len(r['relevant_docs'])); nc.append(len(r['relevant_chunks']))
    err['id_not_int'] += type(r['id']) is not int
    docs = r['relevant_docs']; ds = set(docs)
    err['doc_not_int'] += sum(type(d) is not int for d in docs)
    err['doc_not_in_corpus'] += sum(d not in corpus for d in docs)
    err['doc_dup'] += len(docs) - len(ds)
    cids = [c['doc_id'] for c in r['relevant_chunks']]
    err['chunk_dup_doc'] += len(cids) - len(set(cids))
    for c in r['relevant_chunks']:
        err['chunk_bad_keys'] += set(c) != {'doc_id', 'chunk_text'}
        err['chunk_doc_not_int'] += type(c['doc_id']) is not int
        err['chunk_doc_not_in_docs'] += c['doc_id'] not in ds
        err['chunk_empty'] += not c['chunk_text'].strip()
        if (c['doc_id'], hashlib.sha256(c['chunk_text'].encode()).digest()) in pairs:
            src['p1_pair'] += 1
        else:
            todo.append((cid2doc.get(c['doc_id']), c['chunk_text']))
need = sorted({d for d, _ in todo if d is not None})
txt = dict(zip(*[pq.read_table(f'{CAN}/docs.parquet', columns=['doc_id', 'doc_text'], filters=[('doc_id', 'in', need)])[k].to_pylist()
                 for k in ('doc_id', 'doc_text')]))
for d, t in todo:
    if d is None or d not in txt:
        err['chunk_no_canonical_doc'] += 1
    elif t in (txt[d] or ''):
        src['canonical_substring'] += 1
    else:
        err['chunk_not_verbatim'] += 1
err['ids_mismatch'] += sorted(ids) != sorted(q.id.tolist()) or len(set(ids)) != len(ids)
rep = {'zip': zp, 'zip_bytes': os.path.getsize(zp), 'zip_sha256': hashlib.sha256(open(zp, 'rb').read()).hexdigest(),
       'lzma_props': cp['properties_hex'], 'rows': len(ids), 'docs_total': sum(nd), 'chunks_total': sum(nc),
       'docs_per_query_mean': float(np.mean(nd)), 'chunks_per_query_mean': float(np.mean(nc)), 'chunk_check': dict(src),
       'errors': {k: v for k, v in err.items()}, 'ok': sum(err.values()) == 0}
print(json.dumps(rep, indent=1))
sys.exit(0 if rep['ok'] else 1)
