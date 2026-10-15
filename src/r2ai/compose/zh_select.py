"""zh v2 candidate projection, doc/chunk selection and source-text loading (S8 zg_pair + prepare_sources, unchanged rules)."""
from __future__ import annotations

import collections
import math
from pathlib import Path

import pyarrow.parquet as pq

from r2ai.paths import resolve_path

from .base_io import id_list, sha
from .batch3 import VARIANT, check_candidate_live


def select_docs(rows, present, tau, cap):
    """Same primary-present skip and stable rank order as S11 select_zf; keep v2 aliases only."""
    have = set(present)
    ids = []
    docs = []
    for r in sorted(rows, key=lambda x: x['rank']):
        if r['logit'] < tau or r['doc_id'] in have:
            continue
        if len(docs) >= cap:
            break
        aliases = set(r['doc_ids_v2'])
        new = [d for d in r['doc_ids_group'] if d in aliases and d not in have]
        ids.extend(new)
        have.update(new)
        docs.append(r)
    return {'ids': ids, 'docs': docs}


def select_chunks(rows, cap, *, tau=.96):
    return [r for r in sorted(rows, key=lambda r: r['rank']) if r['logit'] >= tau][:cap]


def project_candidate_v2(rows, membership):
    """Drop alias ids outside corpus v2; a candidate without any v2 zh id is an error."""
    result = []
    dropped = 0
    for r in rows:
        group = id_list(r['doc_ids_group'], 'group')
        v = id_list(r['doc_ids_v2'], 'v2')
        if not set(v) <= set(group):
            raise ValueError('candidate aliases outside group')
        kept = [d for d in v if membership.get(d) == 'zh']
        if not kept:
            raise ValueError('candidate doc has no v2 zh ID')
        dropped += len(v) - len(kept)
        result.append({**r, 'doc_ids_v2': kept})
    return result, dropped


def load_source_texts(selections, manifest, chunks_root, corpus_path):
    """Whole doc_text of the selected zh docs from the pinned segment parquet files -> ({(segment, doc_id): row}, {segment: file pin})."""
    segments = {s['name']: s for s in manifest['segments']}
    if manifest.get('s8_variant') == VARIANT:
        selections = list(selections)
        check_candidate_live(selections, manifest)
    wanted = collections.defaultdict(dict)
    for r in selections:
        name = r['segment']
        s = segments.get(name)
        if not s or not (s.get('verified') is True or s.get('s8_doc_source_verified') is True):
            raise ValueError('source lacks index or CPU doc verification')
        wanted[name][r['doc_id']] = r
    result = {}
    sources = {}
    chunks_root = resolve_path(chunks_root)
    member = {r['id']: r['lang'] for r in pq.read_table(resolve_path(corpus_path), columns=['id', 'lang']).to_pylist()}
    for name, docs in wanted.items():
        p = resolve_path(chunks_root / name / 'docs.parquet')
        if not p.is_relative_to(chunks_root):
            raise ValueError('segment escapes source root')
        digest = sha(p)
        if digest != segments[name]['s8_docs_file_sha256']:
            raise ValueError('CPU verified source SHA changed')
        found = {}
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512, columns=['doc_id', 'doc_ids_group', 'doc_ids_v2', 'doc_text', 'url']):
            for row in batch.to_pylist():
                if row['doc_id'] in docs:
                    r = docs[row['doc_id']]
                    row['doc_ids_v2'] = row.get('doc_ids_v2', [d for d in row['doc_ids_group'] if member.get(d) == 'zh'])
                    if (row['doc_id'] in found or set(row['doc_ids_group']) != set(r['doc_ids_group'])
                            or set(row['doc_ids_v2']) != set(r['doc_ids_v2']) or not row['doc_text']):
                        raise ValueError('candidate/source mismatch')
                    found[row['doc_id']] = row
        if set(found) != set(docs) or sha(p) != digest:
            raise ValueError('source missing/mutated during read')
        for d, row in found.items():
            result[(name, d)] = row
        sources[name] = {'path': str(p), 'sha256': digest, 'bytes': p.stat().st_size}
    return result, sources


def selections_for(by, present, *, tau, cap):
    """Per query: selected docs (absolute tau, cap) and the ZL chunk rows (logit >= .96, top 3, only docs already selected)."""
    sels = {q: select_docs(rs, present[q], tau, cap) for q, rs in by.items()}
    body = {}
    for q, rs in by.items():
        have = set(present[q]) | set(sels[q]['ids'])
        body[q] = select_chunks([r for r in rs if set(r['doc_ids_v2']) <= have], 3, tau=.96)
    return sels, body
