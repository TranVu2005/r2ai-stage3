"""Z: D50 document universe plus zh, direct reranker scores; full 50 chunk slots."""
from __future__ import annotations

from r2ai.paths import ROOT, CORPUS_FILE, QUERY_FILE, require_inputs
from .common import RUN, DOCS, CHUNKS, preflight, atomic_json, exclusive
from .evaluate import vi_rankings
from .index import digest
import json
import zipfile
from collections import defaultdict

CEILING = 104_857_600
EXPECTED = '2f10b78ac17f4eafdd5fe1b0b207260b062f17b29f4fcc3ee1a4b364b3c36544'


def build_z():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.submit.make_submission import load_full_texts
    from r2ai.submit.writer import write_submission
    from r2ai.submit.validate_submission import main as validate
    from vicrawl.extract_pipeline import _atomic_parquet
    preflight()
    baseline = RUN / 'd50_replay/D50.json'
    require_inputs(baseline, RUN / 'yield.json')
    if digest(baseline) != EXPECTED:
        raise ValueError('D50 replay JSON hash mismatch')
    with exclusive('variant'):
        metrics = json.loads((RUN / 'yield.json').read_text('utf-8'))
        branch = max(('hybrid', 'dense'), key=lambda b: metrics[b]['zh_hits'])
        # This branch choice uses a proxy; it does not establish relevance or leaderboard improvement.
        merged = json.loads((RUN / 'retrieval' / (branch + '_merged.json')).read_text('utf-8'))
        vi, _ = vi_rankings()
        source = json.loads(baseline.read_text('utf-8'))
        group_for = {}
        for path in (ROOT / 'data/chunks/docs.parquet', CHUNKS / 'docs.parquet'):
            for r in pq.read_table(path, columns=['doc_id', 'doc_ids_group']).to_pylist():
                group_for[r['doc_id']] = r['doc_ids_group']
        want = {int(d) for rows in merged.values() for d, _ in rows[:50]}
        vi_text, _, _ = load_full_texts(want, ROOT / 'data/docs_vi')
        zh_text, _, _ = load_full_texts(want, DOCS / 'bundle')
        full = vi_text | zh_text
        out, changed_docs, changed_slots, vi_reordered = [], 0, 0, 0
        for row in source:
            q = row['id']
            ranking = [(int(d), float(s)) for d, s in merged[str(q)]]
            primary = [d for d, _ in ranking]
            ds = list(dict.fromkeys(g for d in primary for g in group_for[d]))
            cs = [{'doc_id': d, 'chunk_text': full[d]} for d in primary[:50] if d in full and full[d].strip()]
            changed_docs += ds != row['relevant_docs']
            changed_slots += cs != row['relevant_chunks']
            existing = [d for d in primary if d in {v for v, _ in vi[q]}]
            vi_reordered += existing != [d for d, _ in vi[q] if d in set(existing)]
            out.append({'id': q, 'relevant_docs': ds, 'relevant_chunks': cs})
        destination = RUN / 'Z'
        destination.mkdir(parents=True, exist_ok=True)
        js, zp = destination / 'Z.json', destination / 'Z.zip'
        write_submission(out, js)
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(js, js.name)
        stat = {'branch': branch, 'branch_selection': 'max zh hit proxy, hybrid on tie',
                'json_bytes': js.stat().st_size, 'json_sha256': digest(js), 'zip_bytes': zp.stat().st_size,
                'zip_ceiling_bytes': CEILING, 'changed_doc_queries': changed_docs,
                'changed_chunk_slot_queries': changed_slots,
                'queries_vi_relative_order_changed': vi_reordered,
                'vi_order_note': 'membership uses direct logits; retained vi relative order is preserved across D50 tier boundaries',
                'chunk_rule': 'same D50 full answer if verbatim in body, else full body; 50 slots, no truncation',
                'LB': 'chưa đo', 'validator_errors': 'chưa đo'}
        atomic_json(destination / 'stats.json', stat)
        if zp.stat().st_size > CEILING:
            raise ValueError(f'Z ZIP {zp.stat().st_size} > {CEILING}; keep measured artifact, do not silently change D50 chunk rule')
        # Explicit union for validator: copy only the requested source rows into this run's private validation input.
        validate_dir = destination / 'validation_docs'
        for label, docs_dir in (('vi', ROOT / 'data/docs_vi'), ('zh', DOCS / 'bundle')):
            for path in sorted(docs_dir.glob('*.parquet')):
                rows = [r for r in pq.read_table(path).to_pylist() if any(d in want for d in r['doc_ids'])]
                if rows:
                    _atomic_parquet(rows, validate_dir / (label + '_' + path.name))
        rc = validate([str(zp), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                       '--docs-dir', str(validate_dir), '--max-zip-mib', '100'])
        stat['validator_errors'] = 0 if rc == 0 else 'see validation output'
        stat['validator_exit'] = rc
        atomic_json(destination / 'stats.json', stat)
        if rc:
            raise ValueError('Z validator failed')
        print(json.dumps(stat), flush=True)


if __name__ == '__main__':
    build_z()
