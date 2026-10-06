"""Step 3: VS-append = RD150 (bytes unchanged) + top-k vi doc ids by slug BM25 from non-crawlable vi URLs. Not submitted."""
from __future__ import annotations

import json
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai.paths import CORPUS_FILE, DOCS_DIR, QUERY_FILE, assert_writable
from r2ai.selective.bm25 import Slug
from r2ai.selective.common import OUT, VI_DB, ro
from r2ai.zh_sample.append import BASE, ZIP_CEILING, _addition, _rows, digest, pinned_base, verify_gate

K = 20
BAD_STATUS = {'blocked_4xx', 'bot_challenge', 'soft404_or_home', 'dead_origin', 'network_error', 'pending'}
EXTRA_HOSTS = ('nhathuoclongchau.com.vn',)


def main():
    out = OUT / 'VS-append'
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ro(VI_DB).execute('select url,doc_ids,domain,status from urls').fetchall()
    corp = pq.read_table(CORPUS_FILE).to_pandas()
    corp = corp[corp.url.str.contains('nhathuoclongchau.com.vn')]
    urls = [r[0] for r in rows] + corp.url.tolist()
    ids = [json.loads(r[1]) for r in rows] + [[int(i)] for i in corp.id]
    dom = [r[2] for r in rows] + ['nhathuoclongchau.com.vn'] * len(corp)
    st = [r[3] for r in rows] + ['not_in_crawl'] * len(corp)
    pool = np.array([i for i, s in enumerate(st) if s in BAD_STATUS or s == 'not_in_crawl'])
    print('pool urls', len(pool), pd.Series([dom[i] for i in pool]).value_counts().head(8).to_dict(), flush=True)
    bm = Slug(urls)  # IDF over all vi URLs
    q = pd.read_parquet(QUERY_FILE)
    Q = bm.queries(q['query'].tolist())
    WTp = bm.WT[:, pool]
    base = pinned_base(BASE)
    sel, comp = {}, {}
    ref = list(_rows(base))
    assert [r['id'] for r, _ in ref] == q['id'].tolist()
    for a in range(0, len(q), 100):
        S = (Q[a:a + 100] @ WTp).toarray()
        for k in range(S.shape[0]):
            qid = int(q.id.iloc[a + k]); have = set(ref[a + k][0]['relevant_docs'])
            order = np.lexsort((np.arange(len(pool)), -S[k]))
            pick = []
            for j in order[:400]:
                if S[k][j] <= 0 or len(pick) >= K:
                    break
                for d in ids[pool[j]][:1]:
                    if d not in have and d not in pick:
                        pick.append(d); comp[dom[pool[j]]] = comp.get(dom[pool[j]], 0) + 1
            sel[qid] = {'available': int((S[k] > 0).sum()), 'appended': pick}
    pieces, cursor = [], 0
    for row, spans in ref:
        end = spans['relevant_docs'][1]
        pieces.append(base[cursor:end - 1]); pieces.append(_addition(row['relevant_docs'], sel[row['id']]['appended']))
        cursor = end - 1
    pieces.append(base[cursor:])
    result = ''.join(pieces)
    gate = verify_gate(base, result, sel)
    jpath, zpath = out / 'VS_append.json', out / 'VS_append.zip'
    jpath.write_text(result, encoding='utf-8', newline='')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(jpath, arcname=jpath.name)
    assert zpath.stat().st_size <= ZIP_CEILING, zpath.stat().st_size
    from r2ai.submit.validate_submission import main as validate
    import contextlib, io
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        rc = validate([str(zpath), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE), '--docs-dir', str(DOCS_DIR), '--max-zip-mib', '100'])
    (out / 'validator.log').write_text(cap.getvalue(), encoding='utf-8')
    app = [len(v['appended']) for v in sel.values()]
    stats = {'diagnostic_only': True, 'LB': 'chưa đo', 'k': K, 'gate': gate, 'validator_exit': rc, 'validator': cap.getvalue()[:600],
             'json_bytes': jpath.stat().st_size, 'json_sha256': digest(jpath), 'zip_bytes': zpath.stat().st_size, 'zip_sha256': digest(zpath),
             'pool_urls': int(len(pool)), 'appended_total': int(sum(app)), 'appended_mean': float(np.mean(app)), 'queries_below_k': int(sum(a < K for a in app)),
             'appended_by_domain': comp}
    (out / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'query_append.json').write_text(json.dumps(sel), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
