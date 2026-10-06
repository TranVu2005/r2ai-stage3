"""VS-longchau = RD150 + only the nhathuoclongchau ids of VS-append (same order, same per-query ids). Not submitted."""
from __future__ import annotations

import contextlib
import io
import json
import zipfile

import numpy as np
import pyarrow.parquet as pq

from r2ai.paths import CORPUS_FILE, QUERY_FILE, ROOT
from r2ai.selective.common import OUT
from r2ai.zh_sample.append import BASE, ZIP_CEILING, _addition, _rows, digest, pinned_base, verify_gate


def main():
    out = OUT / 'VS-longchau'
    out.mkdir(parents=True, exist_ok=True)
    corp = pq.read_table(CORPUS_FILE).to_pandas()
    lc = set(corp[corp.url.str.contains('nhathuoclongchau.com.vn')].id.tolist())
    full = json.loads((OUT / 'VS-append' / 'query_append.json').read_text(encoding='utf-8'))
    sel = {int(q): {'appended': [d for d in v['appended'] if d in lc]} for q, v in full.items()}
    base = pinned_base(BASE)
    pieces, cursor = [], 0
    for row, spans in _rows(base):
        end = spans['relevant_docs'][1]
        pieces.append(base[cursor:end - 1]); pieces.append(_addition(row['relevant_docs'], sel[row['id']]['appended']))
        cursor = end - 1
    pieces.append(base[cursor:])
    result = ''.join(pieces)
    gate = verify_gate(base, result, sel)
    jpath, zpath = out / 'VS_longchau.json', out / 'VS_longchau.zip'
    jpath.write_text(result, encoding='utf-8', newline='')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(jpath, arcname=jpath.name)
    assert zpath.stat().st_size <= ZIP_CEILING, zpath.stat().st_size
    from r2ai.submit.validate_submission import main as validate
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        rc = validate([str(zpath), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                       '--docs-dir', str(ROOT / 'data/docs_vi'), '--max-zip-mib', '100'])
    (out / 'validator.log').write_text(cap.getvalue(), encoding='utf-8')
    n = np.array([len(v['appended']) for v in sel.values()])
    rest = {q: [d for d in full[str(q)]['appended'] if d not in lc] for q in sel}
    assert all(not set(sel[q]['appended']) & set(rest[q]) for q in sel)     # disjoint with the non-longchau remainder
    stats = {'diagnostic_only': True, 'LB': 'chưa đo', 'gate': gate, 'validator_exit': rc, 'validator': cap.getvalue().strip()[:400],
             'json_bytes': jpath.stat().st_size, 'json_sha256': digest(jpath), 'zip_bytes': zpath.stat().st_size, 'zip_sha256': digest(zpath),
             'ids_per_query': {'mean': float(n.mean()), 'min': int(n.min()), 'max': int(n.max()), 'total': int(n.sum())},
             'remainder_ids_total_vs_append_minus_longchau': int(sum(len(v) for v in rest.values()))}
    (out / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'query_append.json').write_text(json.dumps(sel), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
