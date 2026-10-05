"""Post-process a built submission with H1-expand: append the ids of duplicate-cluster mates to relevant_docs.

Used by r2ai.submit.best when the config enables postprocess.expand_clusters. No new logic: the doc lists, clusters and
mates come from r2ai.dupes.variants (load_lists / d50_docs / load_clusters / load_groups / expand_lists), so the output equals
`python -m r2ai.dupes.variants build --mode expand` byte for byte when the builder output equals its identity mode.
The base JSON is read one query per line; relevant_docs must equal the doc_ids_group expansion of the rebuilt doc list
(else ValueError: the cache paths do not match the builder run); relevant_chunks are copied unchanged.
"""
from __future__ import annotations

from r2ai.paths import assert_writable, require_inputs, resolve_path

import hashlib
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from r2ai.dupes import variants as V


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def expand_submission(base_json, out_zip, *, runs_dir, doc_ranking, queries, chunks_dir, clusters, scope: str = 'content',
                      k_cache: int = 100, k_total: int = 150) -> dict:
    """base_json (builder output) -> out_zip + its .json with cluster mates appended to relevant_docs; returns stats."""
    base_json, out_zip = Path(base_json), assert_writable(out_zip)
    out_json = assert_writable(out_zip.with_suffix('.json'))
    require_inputs(base_json, clusters, resolve_path(chunks_dir) / 'docs.parquet')
    qids = sorted(int(x) for x in pq.read_table(resolve_path(queries), columns=['id'])['id'].to_pylist())
    top, pool = V.load_lists(runs_dir, doc_ranking, qids, k_cache)
    docs = V.d50_docs(top, pool, k_total)
    groups = V.load_groups(chunks_dir)
    of, members = V.load_clusters(clusters, scope)
    extra = V.expand_lists(docs, of, members, groups)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    n, added, changed = 0, 0, 0
    with open(base_json, 'r', encoding='utf-8', newline='') as fi, open(out_json, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write('[\n')
        for line in fi:
            line = line.rstrip('\n')
            if line in ('[', ']'):
                continue
            row = json.loads(line.rstrip(','))
            q = row['id']
            if row['relevant_docs'] != list(dict.fromkeys(int(x) for d in docs[q] for x in groups[d])):
                raise ValueError(f'query {q}: relevant_docs of {base_json} differ from the doc list rebuilt from {runs_dir}')
            row['relevant_docs'] = row['relevant_docs'] + extra[q]
            fo.write(('' if n == 0 else ',\n') + json.dumps(row, ensure_ascii=False, separators=(', ', ': ')))
            n += 1
            added += len(extra[q])
            changed += bool(extra[q])
        fo.write('\n]\n')
    if n != len(qids):
        raise ValueError(f'{base_json}: {n} queries, expected {len(qids)}')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(out_json, out_json.name)
    return {'queries': n, 'queries_changed': changed, 'ids_added_total': added, 'scope': scope, 'clusters': str(clusters),
            'json': str(out_json), 'json_bytes': out_json.stat().st_size, 'json_sha256': _sha256(out_json),
            'zip_bytes': out_zip.stat().st_size, 'zip_sha256': _sha256(out_zip)}
