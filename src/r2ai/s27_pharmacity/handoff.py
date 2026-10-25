"""Create immutable, incremental S15 exports for the isolated Pharmacity batch."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import pipeline as P


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write_parquet(roots: P.Roots, path: Path, table: pa.Table) -> None:
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, path)


def _prepare_chunk_input(roots: P.Roots, lot_root: Path, docs_file: Path) -> Path:
    """Keep provenance in the S15 docs area but give the chunker a one-file input directory."""
    chunk_input = roots.guard(lot_root / 'chunk_input')
    chunk_input.mkdir(parents=True, exist_ok=True)
    P.atomic_bytes(roots, roots.guard(chunk_input / 'docs.parquet'), roots.guard(docs_file).read_bytes())
    return chunk_input


def _resumable_lot(roots: P.Roots, lots_root: Path, expected_ids: list[int]) -> Path | None:
    """Return an uncommitted lot directory only if its existing extraction IDs match exactly."""
    lots_root = roots.guard(lots_root)
    if not lots_root.exists():
        return None
    expected = sorted(int(x) for x in expected_ids)
    for lot_root in sorted(lots_root.glob('lot-*')):
        roots.guard(lot_root)
        if roots.guard(lot_root / 'lot_manifest.json').exists():
            continue
        docs_file = roots.guard(lot_root / 'docs' / 'docs.parquet')
        if not docs_file.is_file():
            continue
        try:
            docs = pq.read_table(docs_file, columns=['doc_ids']).column(0).to_pylist()
            found = sorted(int(doc_id) for cell in docs for doc_id in (cell or []))
        except (OSError, ValueError, pa.ArrowException):
            continue
        if found == expected:
            return lot_root
    return None


def _atomic_json(roots: P.Roots, path: Path, value: dict) -> None:
    P.atomic_json(roots, path, value)


def _read_lots(roots: P.Roots, handoff_root: Path) -> list[dict]:
    lots_root = roots.guard(handoff_root / 'lots')
    if not lots_root.exists():
        return []
    lots = []
    for manifest_path in sorted(lots_root.glob('lot-*/lot_manifest.json')):
        roots.guard(manifest_path)
        lot = json.loads(manifest_path.read_text(encoding='utf-8'))
        if lot.get('batch') != P.BATCH or lot.get('contract') != 'S15':
            raise ValueError(f'invalid handoff lot manifest: {manifest_path}')
        lots.append(lot)
    return lots


def _publish_manifest(roots: P.Roots, handoff_root: Path, lots: list[dict]) -> dict:
    lots = sorted(lots, key=lambda lot: lot['lot_id'])
    all_ids = [int(doc_id) for lot in lots for doc_id in lot['doc_ids']]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('duplicate doc_id across S27 handoff lots')
    manifest = {
        'batch': P.BATCH,
        'contract': 'S15',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'target_chunk_tokens': P.TARGET_CHUNK_TOKENS,
        'min_body_tokens_vi': P.MIN_BODY_TOKENS,
        'lots': lots,
        'docs': len(all_ids),
        'chunks': sum(int(lot['chunks']) for lot in lots),
        'shared_index_written': False,
        'vectors_written': False,
    }
    _atomic_json(roots, roots.guard(handoff_root / 'manifest.json'), manifest)
    return manifest


def handoff_lot(roots: P.Roots, *, limit: int = 1000) -> dict:
    """Export the next accepted docs without waiting for whole-crawl completion.

    Raw input state is read-only. Each lot has its own S15 docs/provenance/chunks and
    immutable manifest; the cumulative manifest is replaced atomically after verification.
    """
    roots.preflight()
    if not roots.mapping.is_file() or not roots.state.is_file():
        raise FileNotFoundError('S27 mapping and crawl state are required before handoff')
    handoff_root = roots.guard(roots.root / 'handoff' / P.BATCH)
    handoff_root.mkdir(parents=True, exist_ok=True)
    existing_lots = _read_lots(roots, handoff_root)
    previous_ids = {int(doc_id) for lot in existing_lots for doc_id in lot['doc_ids']}

    # Use a read-only connection: handoff does not mutate or checkpoint the crawl database.
    state_uri = roots.state.resolve().as_uri() + '?mode=ro'
    c = sqlite3.connect(state_uri, uri=True)
    try:
        candidates = c.execute("""SELECT doc_id,original_url,mapped_url,final_url,status,http_status,title,
                                       title_match_ratio,title_match_count,title_token_count,body_chars,fetched_at,
                                       raw_path,raw_sha256
                                FROM pages WHERE status='ok' AND http_status=200 AND body_chars>=?
                                ORDER BY doc_id""", (P.MIN_BODY_CP,)).fetchall()
    finally:
        c.close()
    selected_ids = P._next_handoff_ids((r[0] for r in candidates), previous_ids, limit=limit)
    if not selected_ids:
        manifest = _publish_manifest(roots, handoff_root, existing_lots)
        return {'batch': P.BATCH, 'status': 'no_new_docs', 'docs': 0, 'chunks': 0,
                'manifest': str(handoff_root / 'manifest.json'), 'cumulative_docs': manifest['docs'],
                'cumulative_chunks': manifest['chunks'], 'shared_index_written': False}

    by_id = {int(r[0]): r for r in candidates}
    rows_by_id = []
    prov_by_id = []
    from vicrawl.extract_pipeline import SCHEMA, default_tokenizer, detect_lang
    from vicrawl.extractors.base import norm
    from vicrawl.urlnorm import normalize_url

    tok = default_tokenizer()
    for doc_id in selected_ids:
        (row_id, original, mapped, final, status, code, db_title, ratio, match_n, term_n, body_chars, fetched_at,
         raw_path, raw_sha) = by_id[doc_id]
        raw_file = roots.guard(Path(raw_path))
        with gzip.open(raw_file, 'rb') as f:
            raw = f.read()
        if hashlib.sha256(raw).hexdigest() != raw_sha:
            raise ValueError(f'raw hash mismatch for doc_id={doc_id}')
        doc = P.selector_document(raw.decode('utf-8', errors='replace'))
        body = norm(doc['body'])
        if len(body) < P.MIN_BODY_CP:
            raise ValueError(f'S27 accepted page fell below the 200-codepoint body gate: doc_id={doc_id}')
        rows_by_id.append({
            'doc_ids': [doc_id], 'url': mapped, 'final_url': final, 'domain': P.DOMAIN,
            'title': doc['title'] or db_title or '', 'description': doc['description'] or '',
            'question': '', 'answer': '', 'body': body, 'paragraphs': doc['paragraphs'],
            'lang': detect_lang(body), 'n_tokens_bge_m3': tok(body) if body else 0,
            'text_sha1': hashlib.sha1(body.encode('utf-8')).hexdigest(),
            'fetched_at': datetime.fromisoformat(fetched_at.replace('Z', '+00:00')).timestamp(),
            'url_norm': normalize_url(mapped), 'status': 'ok', 'extractor': P.BODY_SELECTOR,
            'headings': doc['headings'],
        })
        prov_by_id.append({
            'doc_id': doc_id, 'original_url': original, 'mapped_url': mapped, 'final_url': final,
            'fetched_at': fetched_at, 'http_status': code, 'body_selector': P.BODY_SELECTOR,
            'body_chars': len(body), 'raw_sha256': raw_sha, 'title_match_ratio': ratio or 0.0,
            'title_match_count': match_n or 0, 'title_token_count': term_n or 0,
            'fetch_status': status, 'reason': '',
        })

    lots_root = roots.guard(handoff_root / 'lots')
    lots_root.mkdir(parents=True, exist_ok=True)
    lot_root = _resumable_lot(roots, lots_root, selected_ids)
    if lot_root is None:
        lot_number = 1
        while roots.guard(lots_root / f'lot-{lot_number:04d}').exists():
            lot_number += 1
        lot_id = f'lot-{lot_number:04d}'
        lot_root = roots.guard(lots_root / lot_id)
        lot_root.mkdir(parents=True, exist_ok=False)
    else:
        lot_id = lot_root.name
    docs_dir = roots.guard(lot_root / 'docs')
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_file = roots.guard(docs_dir / 'docs.parquet')
    provenance_file = roots.guard(docs_dir / 'provenance.parquet')
    _write_parquet(roots, docs_file, pa.Table.from_pylist(rows_by_id, schema=SCHEMA))
    _write_parquet(roots, provenance_file, pa.Table.from_pylist(prov_by_id, schema=P.PROVENANCE_SCHEMA))

    chunks_dir = roots.guard(lot_root / 'chunks')
    chunk_input = _prepare_chunk_input(roots, lot_root, docs_file)
    from r2ai.index.chunk import main as chunk_main
    chunk_main(['--docs-dir', str(chunk_input), '--out-dir', str(chunks_dir), '--targets', str(P.TARGET_CHUNK_TOKENS),
                '--min-body-tokens', str(P.MIN_BODY_TOKENS), '--batch-docs', '5000'])
    chunk_report_path = roots.guard(chunks_dir / 'chunk_report.json')
    chunk_report = json.loads(chunk_report_path.read_text(encoding='utf-8'))
    chunk_cfg = chunk_report['configs'][str(P.TARGET_CHUNK_TOKENS)]
    if chunk_cfg['verbatim_ok'] != f"{chunk_cfg['chunks_all']}/{chunk_cfg['chunks_all']}":
        raise ValueError(f"S27 lot chunk verbatim gate failed: {chunk_cfg['verbatim_ok']}")
    handoff_docs = pq.read_table(roots.guard(chunks_dir / 'docs.parquet'), columns=['doc_id']).column(0).to_pylist()
    chunks = pq.read_table(roots.guard(chunks_dir / f'chunks_t{P.TARGET_CHUNK_TOKENS}.parquet'), columns=['doc_id','text']).to_pylist()
    if len(handoff_docs) != len(set(handoff_docs)) or not set(handoff_docs).issubset(selected_ids):
        raise ValueError('S27 lot chunk documents do not match the selected extraction documents')
    if any(not chunk['text'] for chunk in chunks):
        raise ValueError('S27 lot contains an empty chunk')

    artifact_paths = [docs_file, provenance_file, roots.guard(chunks_dir / 'docs.parquet'),
                      roots.guard(chunks_dir / f'chunks_t{P.TARGET_CHUNK_TOKENS}.parquet'), chunk_report_path]
    lot_manifest = {
        'batch': P.BATCH, 'contract': 'S15', 'lot_id': lot_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'doc_ids': selected_ids, 'docs': len(rows_by_id), 'chunk_docs': len(handoff_docs), 'chunks': len(chunks),
        'body_ge_200_codepoints': len(rows_by_id), 'target_chunk_tokens': P.TARGET_CHUNK_TOKENS,
        'min_body_tokens_vi': P.MIN_BODY_TOKENS, 'verbatim_ok': chunk_cfg['verbatim_ok'],
        'artifacts': {str(p.relative_to(lot_root)): {'bytes': p.stat().st_size, 'sha256': _sha256(p)} for p in artifact_paths},
        'shared_index_written': False, 'vectors_written': False,
    }
    _atomic_json(roots, roots.guard(lot_root / 'lot_manifest.json'), lot_manifest)
    updated_lots = _read_lots(roots, handoff_root)
    manifest = _publish_manifest(roots, handoff_root, updated_lots)
    return {'batch': P.BATCH, 'status': 'ok', 'lot_id': lot_id, 'docs': lot_manifest['docs'],
            'chunk_docs': lot_manifest['chunk_docs'], 'chunks': lot_manifest['chunks'],
            'verbatim_ok': lot_manifest['verbatim_ok'], 'lot_manifest': str(lot_root / 'lot_manifest.json'),
            'cumulative_docs': manifest['docs'], 'cumulative_chunks': manifest['chunks'],
            'shared_index_written': False, 'vectors_written': False}
