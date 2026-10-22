"""Item 4: docs whose title (or question field) is a near-verbatim copy of the test query (char LCS)."""
import pandas as pd, numpy as np, re, unicodedata, zipfile, json, sys
from sklearn.feature_extraction.text import CountVectorizer
S = sys.argv[1]
def norm(s):
    s = unicodedata.normalize('NFC', s).lower(); s = re.sub(r'[^\w\s]', ' ', s); return re.sub(r'\s+', ' ', s).strip()
def lcs(a, b):
    if len(a) > len(b): a, b = b, a
    M = {}
    for i, ch in enumerate(a): M[ch] = M.get(ch, 0) | (1 << i)
    mask = (1 << len(a)) - 1; V = mask
    for ch in b:
        U = V & M.get(ch, 0); V = ((V + U) | (V - U)) & mask
    return len(a) - bin(V).count('1')
assert lcs('abcbdab', 'bdcaba') == 4
ch = pd.read_parquet('data/canonical-20261018/chunks_t256.parquet', columns=['doc_id', 'doc_ids', 'field', 'text'], filters=[('field', 'in', ['title', 'question'])])
ch = ch[ch.doc_id < 766442].reset_index(drop=True)  # vi owners (queries are Vietnamese)
ch['n'] = ch.text.map(norm); print('vi title/question texts', len(ch), ch.field.value_counts().to_dict())
q = pd.read_parquet('data/raw/query.parquet').sort_values('id').reset_index(drop=True); q['n'] = q['query'].map(norm)
cv = CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\b\w+\b', max_df=20000, dtype=np.int32)
T = cv.fit_transform(ch.n); Qm = cv.transform(q.n); print('bigram vocab', len(cv.vocabulary_))
SH = (Qm @ T.T).tocsr()
qlen_bg = np.asarray(Qm.sum(1)).ravel()
res = []
for i in range(len(q)):
    row = SH.getrow(i); idx, val = row.indices, row.data
    if len(idx) == 0: continue
    top = idx[np.argsort(-val)[:100]]
    for j in top:
        a, b = q.n[i], ch.n[j]; L = lcs(a, b)
        res.append((int(q.id[i]), int(ch.doc_id[j]), list(ch.doc_ids[j]), ch.field[j], L / max(len(a), len(b)), L / len(a), L / max(1, len(b)), len(a), len(b)))
r = pd.DataFrame(res, columns=['q', 'owner', 'doc_ids', 'field', 'r_max', 'r_q', 'r_t', 'len_q', 'len_t'])
r.to_parquet(f'{S}/srcdoc_cands.parquet'); print('candidate pairs', len(r))
best = r.sort_values('r_max', ascending=False).drop_duplicates('q')
for th in [0.6, 0.7, 0.8, 0.9]:
    print(f'queries with r_max>={th}:', int((best.r_max >= th).sum()), '/ 1200', ' by field', r[r.r_max >= th].drop_duplicates('q').field.value_counts().to_dict())
bq = r.sort_values('r_q', ascending=False).drop_duplicates('q')
for th in [0.8, 0.9]: print(f'queries with LCS/len(query)>={th} (title covers query):', int((bq.r_q >= th).sum()))
