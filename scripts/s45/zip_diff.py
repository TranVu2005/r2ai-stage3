"""S45: compare two submission ZIPs query by query (relevant_docs order, chunk owners, (doc_id, chunk_text) pairs).

  zip_diff.py <a.zip> <b.zip>
"""
import json, sys, zipfile


def rows(p):
    with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            line = line.strip().rstrip(b',')
            if line.startswith(b'{'):
                yield json.loads(line)


n = docs_eq = docs_set_eq = chunks_eq = owners_eq = ca_n = cb_n = same = doc_only_a = doc_only_b = 0
for a, b in zip(rows(sys.argv[1]), rows(sys.argv[2])):
    n += 1
    assert a['id'] == b['id']
    docs_eq += a['relevant_docs'] == b['relevant_docs']; docs_set_eq += set(a['relevant_docs']) == set(b['relevant_docs'])
    doc_only_a += len(set(a['relevant_docs']) - set(b['relevant_docs'])); doc_only_b += len(set(b['relevant_docs']) - set(a['relevant_docs']))
    ca = {(c['doc_id'], c['chunk_text']) for c in a['relevant_chunks']}; cb = {(c['doc_id'], c['chunk_text']) for c in b['relevant_chunks']}
    chunks_eq += ca == cb; ca_n += len(ca); cb_n += len(cb); same += len(ca & cb)
    owners_eq += {d for d, _ in ca} == {d for d, _ in cb}
print(json.dumps({'queries': n, 'docs_equal_order': docs_eq, 'docs_equal_set': docs_set_eq, 'docs_only_a': doc_only_a, 'docs_only_b': doc_only_b,
                  'chunk_owner_sets_equal': owners_eq, 'chunks_equal': chunks_eq, 'chunks_a': ca_n, 'chunks_b': cb_n, 'chunks_same_doc_text': same}))
