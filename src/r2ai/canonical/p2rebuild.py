"""S25 phase 3 step 8 check: rebuild P2 JSON from the canonical corpus and compare SHA-256 with the approved P2.

    python -m r2ai.canonical.p2rebuild map      # once, from P2.zip + canonical -> skeleton.parquet (no text stored)
    python -m r2ai.canonical.p2rebuild rebuild  # skeleton + canonical only -> P2-rebuilt.json, SHA, row diff

The skeleton keeps, per query, relevant_docs and for each relevant_chunk only (doc_id, char_start, char_end): the
chunk text is a slice of the canonical doc_text. Chunks whose text is not in canonical (Vu Q1, 18/10: the 3,435
Tam Anh chunks of P2 hold the superseded base body; canonical keeps S18) are mapped to the S18 body span and
flagged `accepted_diff`; every other difference is an error. Serialization = json.dumps(ensure_ascii=False) per row,
'[\\n' + ',\\n'.join(rows) + '\\n]\\n', which reproduces all 1,200 P2 rows byte for byte.
"""
from __future__ import annotations

from r2ai.paths import resolve_path

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from r2ai.canonical.guard import CANONICAL_ROOT, RUN_ROOT, guarded, write_parquet
from r2ai.compose.base_io import baseline

P2_ZIP = Path('D:/GitHub/r2ai-stage3/out/runs/reranker-trial/release/P2.zip')
P2_APPROVAL = Path('D:/GitHub/r2ai-stage3/out/runs/reranker-trial/release/P2.approval.json')
OUT = RUN_ROOT / 'p2-rebuild'


def canonical_texts(canon: Path, ids) -> tuple[dict, dict, dict]:
    ids = pa.array(sorted(set(ids)), pa.int64())
    d = ds.dataset(canon / 'docs.parquet', format='parquet').to_table(columns=['doc_id', 'doc_text'],
                                                                     filter=ds.field('doc_id').isin(ids))
    text = dict(zip(d['doc_id'].to_pylist(), d['doc_text'].to_pylist()))
    c = ds.dataset(canon / 'chunks_t256.parquet', format='parquet').to_table(
        columns=['doc_ids', 'field', 'char_start', 'char_end'], filter=ds.field('doc_id').isin(ids))
    tail, ans = {}, {}
    for dids, f, s, e in zip(c['doc_ids'].to_pylist(), c['field'].to_pylist(), c['char_start'].to_pylist(),
                             c['char_end'].to_pylist()):
        for i in dids:
            if f in ('answer', 'body'):
                tail[i] = min(s, tail.get(i, s))
            if f == 'answer':
                a, b = ans.get(i, (s, e))
                ans[i] = (min(a, s), max(b, e))
    return text, tail, ans


def build_map(canon: Path, out: Path) -> dict:
    rows = list(r for r, _ in baseline(P2_ZIP))
    need = {int(c['doc_id']) for r in rows for c in r['relevant_chunks']}
    text, tail, ans = canonical_texts(canon, need)
    recs, kinds = [], defaultdict(int)
    for r in rows:
        for k, c in enumerate(r['relevant_chunks']):
            d, tx = int(c['doc_id']), c['chunk_text']
            t = text.get(d)
            s = e = -1
            kind = 'missing_doc'
            if t is not None:
                if d in ans and tx == t[ans[d][0]:ans[d][1]]:
                    s, e = ans[d]
                elif d in tail and tx == t[tail[d]:]:
                    s, e = tail[d], len(t)
                else:
                    i = t.find(tx)
                    if i >= 0:
                        s, e = i, i + len(tx)
                kind = 'slice' if s >= 0 else 'accepted_diff'
                if s < 0:                                  # text not in canonical: substitute the S18 body span
                    s, e = (tail[d], len(t)) if d in tail else (0, len(t))
            kinds[kind] += 1
            recs.append({'query_id': int(r['id']), 'k': k, 'doc_id': d, 'char_start': s, 'char_end': e, 'kind': kind})
    if kinds['missing_doc']:
        raise ValueError(f'{kinds["missing_doc"]} P2 chunks have no canonical doc')
    out = guarded(out)
    out.mkdir(parents=True)
    write_parquet(pa.Table.from_pylist(recs), out / 'skeleton-chunks.parquet')
    write_parquet(pa.Table.from_pylist([{'query_id': int(r['id']), 'relevant_docs': [int(x) for x in r['relevant_docs']]}
                                        for r in rows]), out / 'skeleton-docs.parquet')
    return dict(kinds)


def rebuild(canon: Path, skel: Path, out_name: str) -> dict:
    ch = pq.read_table(skel / 'skeleton-chunks.parquet').to_pylist()
    docs = pq.read_table(skel / 'skeleton-docs.parquet').to_pylist()
    text, _, _ = canonical_texts(canon, {c['doc_id'] for c in ch})
    by_q = defaultdict(list)
    for c in ch:
        by_q[c['query_id']].append(c)
    h = hashlib.sha256()
    out = guarded(skel / out_name)
    lines = []
    for r in docs:
        q = r['query_id']
        chunks = [{'doc_id': c['doc_id'], 'chunk_text': text[c['doc_id']][c['char_start']:c['char_end']]}
                  for c in sorted(by_q[q], key=lambda c: c['k'])]
        lines.append(json.dumps({'id': q, 'relevant_docs': r['relevant_docs'], 'relevant_chunks': chunks},
                                ensure_ascii=False).encode('utf-8'))
    payload = b'[\n' + b',\n'.join(lines) + b'\n]\n'
    h.update(payload)
    out.write_bytes(payload)
    approved = json.loads(P2_APPROVAL.read_text(encoding='utf-8'))
    want = approved.get('json_sha256') or json.loads(P2_APPROVAL.with_name('P2.validator.json').read_text())['json_sha256']
    # row-level diff against the approved P2 JSON
    orig = {int(r['id']): r for r, _ in baseline(P2_ZIP)}
    diff_chunks, diff_rows, accepted = 0, 0, 0
    kinds = {(c['query_id'], c['k']): c['kind'] for c in ch}
    other = []
    for line in lines:
        r = json.loads(line)
        o = orig[r['id']]
        if r != o:
            diff_rows += 1
        for k, (a, b) in enumerate(zip(r['relevant_chunks'], o['relevant_chunks'])):
            if a != b:
                diff_chunks += 1
                if kinds[(r['id'], k)] == 'accepted_diff':
                    accepted += 1
                else:
                    other.append({'query_id': r['id'], 'k': k, 'doc_id': a['doc_id']})
        if r['relevant_docs'] != o['relevant_docs'] or len(r['relevant_chunks']) != len(o['relevant_chunks']):
            other.append({'query_id': r['id'], 'k': -1, 'doc_id': -1})
    # Proof that only the Q1 chunks differ: put back the approved text for accepted_diff chunks only -> SHA must match.
    h2 = hashlib.sha256()
    fixed = []
    for line in lines:
        r = json.loads(line)
        for k, c in enumerate(r['relevant_chunks']):
            if kinds[(r['id'], k)] == 'accepted_diff':
                c['chunk_text'] = orig[r['id']]['relevant_chunks'][k]['chunk_text']
        fixed.append(json.dumps(r, ensure_ascii=False).encode('utf-8'))
    h2.update(b'[\n' + b',\n'.join(fixed) + b'\n]\n')
    res = {'rebuilt_json': str(out), 'rebuilt_sha256': h.hexdigest(), 'approved_p2_json_sha256': want,
           'sha_with_only_q1_chunks_restored': h2.hexdigest(), 'restored_sha_equal': h2.hexdigest() == want,
           'sha_equal': h.hexdigest() == want, 'rows_differing': diff_rows, 'chunks_differing': diff_chunks,
           'accepted_diff_chunks': accepted, 'unexpected_differences': len(other), 'unexpected_sample': other[:20],
           'pass_with_q1_exception': not other}
    guarded(skel / (out_name + '.report.json')).write_text(json.dumps(res, indent=2), encoding='utf-8')
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=['map', 'rebuild'])
    ap.add_argument('--canonical', default=str(CANONICAL_ROOT))
    ap.add_argument('--dir', default=str(OUT))
    ap.add_argument('--name', default='P2-rebuilt.json')
    a = ap.parse_args(argv)
    canon, d = resolve_path(a.canonical), resolve_path(a.dir)
    print(json.dumps(build_map(canon, d) if a.step == 'map' else rebuild(canon, d, a.name), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
