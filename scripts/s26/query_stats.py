"""S26 H6: query distribution — length, script, question form, near-duplicate queries (word 3-gram Jaccard)."""
import os
import html, re, sys, unicodedata
from collections import Counter
import numpy as np, pandas as pd
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

W = re.compile(r'\w+')
norm = lambda s: re.sub(r'\s+', ' ', html.unescape(unicodedata.normalize('NFKC', s)).lower()).strip()
q = pd.read_parquet('data/raw/query.parquet').sort_values('id')
q['n'] = q['query'].map(lambda s: len(W.findall(norm(s))))
print('words: ', q.n.describe(percentiles=[.1, .25, .5, .75, .9, .95]).round(1).to_dict())
print('len buckets', pd.cut(q.n, [0, 10, 15, 20, 30, 50, 100, 400]).value_counts().sort_index().to_dict())
han = q['query'].str.contains(r'[一-鿿]'); latin_only = ~q['query'].str.contains(r'[À-ỹđĐ]')
print('has Han', int(han.sum()), 'no Vietnamese diacritics', int(latin_only.sum()))
print(q[latin_only]['query'].head(5).tolist())
qm = q['query'].str.strip().str.endswith('?'); print('ends with ?', int(qm.sum()))
greet = q['query'].str.lower().str.contains(r'^(chào|xin chào|thưa|kính|bác sĩ ơi|bs ơi|em chào|cháu chào|chao)')
print('greeting start', int(greet.sum()))
multi_q = q['query'].str.count(r'\?') >= 2; print('>=2 question marks', int(multi_q.sum()))
first = Counter(' '.join(W.findall(norm(s))[:2]) for s in q['query']); print('top first-2 words', first.most_common(15))
last = Counter(' '.join(W.findall(norm(s))[-2:]) for s in q['query']); print('top last-2 words', last.most_common(15))
person = q['query'].str.lower().str.contains(r'\b(em|tôi|cháu|mình|con tôi|bé nhà|vợ tôi|chồng tôi|mẹ tôi|bố tôi)\b')
print('first-person/patient narrative', int(person.sum()))
age = q['query'].str.contains(r'\d+\s*(tuổi|tháng|tuần)'); print('mentions age/weeks', int(age.sum()))
# near duplicates
sh = [set(zip(*(W.findall(norm(s))[i:] for i in range(3)))) for s in q['query']]
ids = q.id.tolist(); pairs = []
for i in range(len(sh)):
    for j in range(i + 1, len(sh)):
        if sh[i] and sh[j]:
            jac = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
            if jac >= 0.5:
                pairs.append((ids[i], ids[j], round(jac, 3)))
print('near-dup pairs J>=0.5:', len(pairs), pairs[:20])
ex = q['query'].map(norm).duplicated(keep=False); print('exact dup (normalised)', int(ex.sum()))
q.to_parquet(sys.argv[1] + '/query_stats.parquet', index=False)
