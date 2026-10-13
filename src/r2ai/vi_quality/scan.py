"""Scan the indexed vi docs with r2ai.vi_quality.rules and report per host. Read only on data/, writes under --out-dir.

    python -m r2ai.vi_quality.scan --out-dir reports

Outputs (prefix s12_vi_): scan.json (per host counts, chunk impact), drop_doc_ids.tsv (doc_id, domain, reason, url),
strip_doc_ids.tsv (doc_id, domain, rule, cut_start, removed_chars), dup_map.tsv (drop_id, keep_id; exact same text after
strip, different doc_id). Chunk impact is measured on data/chunks/chunks_t256.parquet: chunks whose span lies at or after
the cut are removed, the one straddling the cut is truncated.
"""
from __future__ import annotations

from r2ai.paths import CHUNKS_DIR, assert_writable, chunks_file, require_inputs, resolve_path

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from r2ai.vi_quality.rules import drop_reason, strip_tail


def scan_docs(docs_file: Path) -> dict:
    pf = pq.ParquetFile(docs_file)
    host, drop, strip, seen = {}, {}, {}, {}
    dup = {}
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=['doc_id', 'domain', 'url', 'doc_text']).to_pydict()
        for did, dom, url, text in zip(t['doc_id'], t['domain'], t['url'], t['doc_text']):
            h = host.setdefault(dom, Counter())
            h['docs'] += 1
            h['chars'] += len(text)
            reason = drop_reason(dom, text)
            if reason:
                drop[did] = (dom, reason, url)
                h['drop_' + reason] += 1
                continue
            new, rule, cut = strip_tail(dom, text)
            if rule:
                strip[did] = (dom, rule, cut, len(text) - len(new))
                h['strip_' + rule] += 1
                h['strip_chars'] += len(text) - len(new)
            key = hashlib.sha1(new.encode('utf-8')).hexdigest()
            if key in seen:
                dup[did] = seen[key]
                h['dup_text'] += 1
            else:
                seen[key] = did
    return {'host': host, 'drop': drop, 'strip': strip, 'dup': dup}


def chunk_impact(chunks: Path, drop: dict, strip: dict, dup: dict) -> dict:
    cut = {d: v[2] for d, v in strip.items()}
    gone = set(drop) | set(dup)
    per = defaultdict(Counter)
    pf = pq.ParquetFile(chunks)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=['doc_id', 'field', 'char_start', 'char_end', 'text'])
        nbytes = pc.binary_length(t['text']).to_numpy()
        for did, field, s, e, nb in zip(t['doc_id'].to_pylist(), t['field'].to_pylist(), t['char_start'].to_pylist(),
                                        t['char_end'].to_pylist(), nbytes):
            if did in gone:
                kind = 'removed'
            elif did in cut:
                kind = 'removed' if s >= cut[did] else ('truncated' if e > cut[did] else None)
            else:
                continue
            if kind is None:
                continue
            c = per[kind]
            c['chunks'] += 1
            c['bytes'] += int(nb)
            if field in ('answer', 'body'):
                c['submittable_chunks'] += 1
                c['submittable_bytes'] += int(nb)
    return {k: dict(v) for k, v in per.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default=str(CHUNKS_DIR / 'docs.parquet'))
    ap.add_argument('--chunks', default=str(chunks_file(256)))
    ap.add_argument('--out-dir', default='reports')
    a = ap.parse_args(argv)
    docs, chunks, out = resolve_path(a.docs), resolve_path(a.chunks), assert_writable(a.out_dir)
    require_inputs(docs, chunks)
    r = scan_docs(docs)
    impact = chunk_impact(chunks, r['drop'], r['strip'], r['dup'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 's12_vi_drop_doc_ids.tsv').write_text(
        'doc_id\tdomain\treason\turl\n' + ''.join(f'{d}\t{v[0]}\t{v[1]}\t{v[2]}\n' for d, v in sorted(r['drop'].items())),
        encoding='utf-8', newline='\n')
    (out / 's12_vi_strip_doc_ids.tsv').write_text(
        'doc_id\tdomain\trule\tcut_start\tremoved_chars\n' + ''.join(
            f'{d}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\n' for d, v in sorted(r['strip'].items())), encoding='utf-8', newline='\n')
    (out / 's12_vi_dup_map.tsv').write_text(
        'drop_id\tkeep_id\n' + ''.join(f'{d}\t{k}\n' for d, k in sorted(r['dup'].items())), encoding='utf-8', newline='\n')
    summary = {'docs': sum(h['docs'] for h in r['host'].values()), 'dropped': len(r['drop']), 'stripped': len(r['strip']),
               'dup': len(r['dup']), 'chunk_impact': impact,
               'host': {h: dict(c) for h, c in sorted(r['host'].items(), key=lambda x: -x[1]['docs'])}}
    (out / 's12_vi_scan.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'host'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
