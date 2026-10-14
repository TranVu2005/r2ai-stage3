"""Native validator worker for a compose ZIP: the repo validator with a read-only doc-text source adapter (S8 validate_compose).

    python -m r2ai.compose.native --manifest configs/best-c1.yaml --worker OUT.zip

vi/CC doc texts come from the pinned `vi_docs_dir`; zh texts from the pinned zh v2 docs snapshot (candidate + manifest). Every check
of `r2ai.submit.validate_submission` stays active (schema, v2 membership, protection, dedupe, near-duplicate, verbatim, ZIP size).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from r2ai.paths import resolve_path

from . import build as B
from . import manifest as M
from .archive import check_pins
from .base_io import sha


def validate(zip_path, manifest_path, *, derived=False) -> int:
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import pyarrow.parquet as pq
    from r2ai.eval.validate import TEXT_FIELDS
    from r2ai.index.chunk import doc_layout
    from r2ai.submit import validate_submission as V
    path = resolve_path(zip_path)
    job = B.load_job(manifest_path, scope=(*M.COMPOSE_PINS, *M.NATIVE_PINS))
    m, p = job['manifest'], job['paths']
    plan_path = resolve_path(M.expand(m['output_dir']) / (m['name'] + '.plan.json'))
    plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else None
    if not derived:
        proof = json.loads(path.with_suffix('.approval.json').read_text(encoding='utf-8'))
        if proof['compose_config_sha256'] != job['manifest_sha256']:
            raise ValueError('manifest SHA changed since the build')
        if sha(plan_path) != proof['plan_sha256']:
            raise ValueError('plan SHA changed')
        check_pins(plan['pins'])
        expected = {'candidate_path': str(p['candidate']), 'candidate_sha256': M._entries(m, 'candidate')[0]['sha256'],
                    'manifest_path': str(p['zh_manifest']), 'manifest_sha256': M._entries(m, 'zh_manifest')[0]['sha256'],
                    'ceiling_bytes': M.POLICY['ceiling_bytes'], 'source_doc_files': plan['source_doc_files']}
        B.check_resume(proof, expected)
        if any(r['protected_ids'] or r['removed_vi_ids'] for r in proof['audit']):
            raise ValueError('native base/doc policy differs')
    zh_manifest = json.loads(p['zh_manifest'].read_text(encoding='utf-8'))
    member = {r['id']: r['lang'] for r in pq.read_table(p['corpus_v2_ids'], columns=['id', 'lang']).to_pylist()}
    zhids = {d for d, lang in member.items() if lang == 'zh'}
    chunk_root = M.expand(m['zh_chunk_root'])

    def adapter(ids, docs_dir):
        want_vi = ids - zhids
        result = {}
        files = sorted(Path(docs_dir).glob('*.parquet'))
        for i, f in enumerate(files):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=['doc_ids', *TEXT_FIELDS]):
                for r in batch.to_pylist():
                    hits = want_vi.intersection(r['doc_ids'])
                    if hits:
                        text = doc_layout({**r, 'body': r['body'] or '', 'paragraphs': r['paragraphs'] or []})[0]
                        for d in hits:
                            result[d] = text
            if (i + 1) % 1500 == 0:
                print(f'SOURCE vi/CC {i + 1}/{len(files)} files; {len(result)}/{len(want_vi)} IDs', flush=True)
        want_zh = ids & zhids
        if want_zh:
            from .zh_select import load_source_texts, project_candidate_v2
            rows, _ = project_candidate_v2(pq.read_table(p['candidate']).to_pylist(), member)
            selected = [r for r in rows if want_zh.intersection(r['doc_ids_v2'])]
            source, _ = load_source_texts(selected, zh_manifest, chunk_root, p['corpus_v2_ids'])
            for r in selected:
                for d in want_zh.intersection(r['doc_ids_v2']):
                    result[d] = source[(r['segment'], r['doc_id'])]['doc_text']
        return result

    V.doc_texts = adapter
    code = V.main([str(path), '--queries', str(p['query']), '--corpus', str(p['corpus_v2_ids']),
                   '--docs-dir', str(p['vi_docs_dir']), '--max-zip-mib', str(M.POLICY['ceiling_bytes'] / (1 << 20))])
    if plan is not None and not derived:
        check_pins(plan['pins'])
    return code


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--worker', required=True, help='ZIP to validate')
    ap.add_argument('--derived', action='store_true', help='ZIP derived from a built compose (e.g. host filter): no proof/plan identity checks')
    a = ap.parse_args(argv)
    return validate(a.worker, a.manifest, derived=a.derived)


if __name__ == '__main__':
    sys.exit(main())
