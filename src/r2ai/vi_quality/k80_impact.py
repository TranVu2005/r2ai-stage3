"""Measure what the vi_quality rules would change in a submission JSON (one query per line, r2ai.submit.writer format).

    python -m r2ai.vi_quality.k80_impact --json SUB.json --out-json CLEAN.json --report reports/s12_vi_k80_impact.json

Rules applied to the submission: a relevant_chunks entry whose doc is in the drop list is removed; a relevant_docs id that
belongs to a dropped doc (its doc_ids_group) is removed; the chunk_text of a doc with a strip rule is cut at the boilerplate tail
(chunk_text is the whole body or answer, so the tail pattern still anchors). Doc ids outside the index (Common Crawl docs) are
left alone. The report counts docs / chunks / UTF-8 bytes; the ZIP effect is measured separately by compressing both JSON files.
"""
from __future__ import annotations

from r2ai.paths import CHUNKS_DIR, assert_writable, resolve_path

import argparse
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from r2ai.vi_quality.rules import drop_reason, strip_tail


def index_info(docs_file):
    """doc id (any id of a group) -> (primary doc_id, domain)."""
    pf = pq.ParquetFile(docs_file)
    out = {}
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=['doc_id', 'doc_ids_group', 'domain']).to_pydict()
        for d, grp, dom in zip(t['doc_id'], t['doc_ids_group'], t['domain']):
            for i in grp or [d]:
                out[i] = (d, dom)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out-json')
    ap.add_argument('--report', required=True)
    ap.add_argument('--docs', default=str(CHUNKS_DIR / 'docs.parquet'))
    ap.add_argument('--drop-tsv', default='reports/s12_vi_drop_doc_ids.tsv')
    a = ap.parse_args(argv)
    drop = {}
    for line in resolve_path(a.drop_tsv).read_text(encoding='utf-8').splitlines()[1:]:
        d, dom, reason, _ = line.split('\t', 3)
        drop[int(d)] = (dom, reason)
    info = index_info(resolve_path(a.docs))
    dropped_ids = {i for i, (p, _) in info.items() if p in drop}
    rep = defaultdict(Counter)
    host = defaultdict(Counter)
    kept = []
    for line in open(resolve_path(a.json), encoding='utf-8'):
        s = line.strip().rstrip(',')
        if s in ('[', ']', ''):
            continue
        row = json.loads(s)
        docs = row['relevant_docs']
        new_docs = [d for d in docs if d not in dropped_ids]
        rep['docs']['total'] += len(docs)
        rep['docs']['removed'] += len(docs) - len(new_docs)
        rep['docs']['outside_index'] += sum(d not in info for d in docs)
        for d in docs:
            if d in dropped_ids:
                host[info[d][1]]['docs_removed'] += 1
        chunks = []
        for c in row['relevant_chunks']:
            did, text = c['doc_id'], c['chunk_text']
            nb = len(text.encode('utf-8'))
            rep['chunks']['total'] += 1
            rep['bytes']['total'] += nb
            if did not in info:
                rep['chunks']['outside_index'] += 1
                rep['bytes']['outside_index'] += nb
                chunks.append(c)
                continue
            prim, dom = info[did]
            if prim in drop:
                rep['chunks']['removed'] += 1
                rep['bytes']['removed'] += nb
                host[dom]['chunks_removed'] += 1
                host[dom]['bytes_removed'] += nb
                continue
            new, rule, _ = strip_tail(dom, text)
            if rule:
                cut = nb - len(new.encode('utf-8'))
                rep['chunks']['stripped'] += 1
                rep['bytes']['stripped'] += cut
                host[dom]['chunks_stripped'] += 1
                host[dom]['bytes_stripped'] += cut
                c = {**c, 'chunk_text': new}
            chunks.append(c)
        kept.append({**row, 'relevant_docs': new_docs, 'relevant_chunks': chunks})
    if a.out_json:
        out = assert_writable(a.out_json)
        lines = [json.dumps(r, ensure_ascii=False, separators=(', ', ': ')) for r in kept]
        out.write_text('[\n' + ',\n'.join(lines) + '\n]\n', encoding='utf-8', newline='\n')
    rep['queries']['n'] = len(kept)
    report = {k: dict(v) for k, v in rep.items()}
    report['host'] = {h: dict(v) for h, v in sorted(host.items(), key=lambda x: -sum(x[1].values()))}
    out = assert_writable(a.report)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
