"""Offline S20 failure audit and CC-contract staging; never writes an index."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.paths import CONFIG_DIR, ROOT, WORK_DATA_DIR, assert_writable, data_label
from r2ai.raw_recovery import core
from vicrawl.extractors import extract_doc
from vicrawl.settings import Config
from vicrawl.shards import read_shard


S20_CSV = ROOT / 'reports' / 's20-coverage-audit.csv'
S20_SOURCE_COMMIT = '2d3f44c'
STAGE_ROOT = WORK_DATA_DIR / 's22_reextract'
REPORT_MD = ROOT / 'reports' / 's22_raw_recovery_audit.md'
SAMPLES_CSV = ROOT / 'reports' / 's22_sample_diagnosis.csv'
MISSING_CSV = ROOT / 'reports' / 's22_raw_missing_ids.csv'
VI_HOSTS = {
    'bachmai.gov.vn', 'baodanang.vn', 'baonghean.vn', 'laodong.vn',
    'suckhoecongdongonline.vn', 'suckhoedoisong.vn', 'thanhnien.vn',
}
SCAN_COLUMNS = ['doc_ids', 'url', 'body', 'fetched_at', 'url_norm', 'status']
DOC_SCHEMA = __import__('vicrawl.extract_pipeline', fromlist=['SCHEMA']).SCHEMA


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_s20_rows() -> list[dict]:
    if S20_CSV.is_file():
        return _read_csv(S20_CSV)
    # The current worktree switched to feat/s21-120ask mid-task and no longer has reports/.
    # Read the exact S20 input blob from its local commit; this is offline and read-only.
    result = subprocess.run(
        ['git', 'show', f'{S20_SOURCE_COMMIT}:reports/s20-coverage-audit.csv'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _load_v2() -> tuple[dict[str, set[int]], dict[int, str]]:
    table = pq.read_table(WORK_DATA_DIR / 'corpus_v2_ids.parquet', columns=['id', 'host'])
    by_host: dict[str, set[int]] = defaultdict(set)
    host_by_id = {}
    for doc_id, host in zip(table['id'].to_pylist(), table['host'].to_pylist()):
        doc_id, host = int(doc_id), str(host)
        by_host[host].add(doc_id)
        host_by_id[doc_id] = host
    return dict(by_host), host_by_id


def _target_hosts() -> list[dict]:
    rows = [r for r in _read_s20_rows()
            if r['lang'] in ('vi', 'zh') and int(r['empty_or_extract_error'] or 0) >= 50]
    return sorted(rows, key=lambda r: (r['lang'], r['host']))


def _source_roots(host: str) -> list[tuple[Path, Path, str]]:
    if host in VI_HOSTS:
        return [(WORK_DATA_DIR / 'docs_vi', WORK_DATA_DIR / 'raw_vi', 'vi')]
    roots = []
    for docs_name, raw_name in (
        ('docs_zh_full', 'raw_zh_full'), ('docs_zh_v2', 'raw_zh_v2'),
        ('docs_zh_v2_redo', 'raw_zh_v2_redo'), ('docs_zh_v2_redo2', 'raw_zh_v2_redo2'),
    ):
        docs, raw = WORK_DATA_DIR / docs_name, WORK_DATA_DIR / raw_name
        if docs.exists():
            roots.append((docs, raw, 'zh'))
    return roots


def _doc_files(host: str, docs_root: Path, lang: str) -> list[Path]:
    if not docs_root.exists():
        return []
    if lang == 'vi':
        return sorted(docs_root.glob(f'{host}__*.parquet'))
    return sorted(docs_root.rglob(f'{host}/*.parquet'))


def _raw_path(doc_file: Path, docs_root: Path, raw_root: Path, host: str, lang: str) -> Path:
    if lang == 'vi':
        prefix = host + '__'
        name = doc_file.name.removesuffix('.parquet')
        if not name.startswith(prefix):
            raise ValueError(f'Unexpected VI docs filename: {doc_file}')
        raw_name = name[len(prefix):] + '.jsonl.zst'
    else:
        raw_name = doc_file.name.removesuffix('.parquet') + '.jsonl.zst'
    return raw_root / host / raw_name


def _raw_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return data_label(path)


def _path_from_label(label: str) -> Path:
    if label.startswith('data/'):
        return WORK_DATA_DIR / label[5:]
    return ROOT / label


def _scan_host(host: str, lang: str, v2_ids: set[int]) -> tuple[dict[int, dict], dict[int, list[Path]]]:
    latest: dict[int, dict] = {}
    raw_candidates: dict[int, list[Path]] = defaultdict(list)
    roots = _source_roots(host)
    for docs_root, raw_root, root_lang in roots:
        if root_lang != lang:
            continue
        files = _doc_files(host, docs_root, lang)
        if len(files) > 500:
            import polars as pl
            frame = pl.scan_parquet([str(p) for p in files], include_file_paths='_source_file',
                                    missing_columns='insert', extra_columns='ignore')
            rows = frame.select(SCAN_COLUMNS + ['_source_file']).collect(engine='streaming').to_dicts()
        else:
            rows = []
            for path in files:
                try:
                    table = pq.read_table(path, columns=SCAN_COLUMNS)
                except (OSError, pa.ArrowInvalid, KeyError):
                    continue
                for row in table.to_pylist():
                    rows.append({**row, '_source_file': str(path)})
        for row in rows:
            path = Path(row.pop('_source_file'))
            try:
                raw_file = _raw_path(path, docs_root, raw_root, host, lang)
            except ValueError:
                raw_file = None
            doc_id = core.primary_doc_id(row)
            if doc_id is None or doc_id not in v2_ids:
                continue
            candidate = {**row, '_source_file': str(path), '_raw_path': str(raw_file) if raw_file else ''}
            old = latest.get(doc_id)
            current_key = (float(row.get('fetched_at') or 0), str(path))
            old_key = ((float(old.get('fetched_at') or 0), old.get('_source_file', '')) if old else None)
            if old is None or current_key > old_key:
                latest[doc_id] = candidate
            if raw_file is not None:
                value = str(raw_file)
                if value not in {str(p) for p in raw_candidates[doc_id]}:
                    raw_candidates[doc_id].append(Path(value))
    return latest, raw_candidates


def _failed_rows(host: str, latest: dict[int, dict], v2_ids: set[int]) -> list[dict]:
    rows = [r for r in latest.values() if core.is_s20_failure(r, host, v2_ids)]
    return sorted(rows, key=lambda r: int(r['doc_ids'][0]))


def _raw_record_index(failures: list[dict], raw_candidates: dict[int, list[Path]]) -> tuple[dict[int, dict], dict[int, Path]]:
    paths: dict[str, set[int]] = defaultdict(set)
    expected_urls = {int(row['doc_ids'][0]): row.get('url_norm') for row in failures}
    for row in failures:
        doc_id = int(row['doc_ids'][0])
        for path in raw_candidates.get(doc_id, []):
            if path.is_file():
                paths[str(path)].add(doc_id)
    found: dict[int, dict] = {}
    found_path: dict[int, Path] = {}
    for raw_name, wanted in sorted(paths.items()):
        path = Path(raw_name)
        for record in read_shard(path):
            doc_id = core.primary_doc_id(record)
            if (doc_id not in wanted or record.get('url_norm') != expected_urls.get(doc_id)
                    or not (record.get('html') or '').strip()):
                continue
            previous = found.get(doc_id)
            if previous is None or float(record.get('fetched_at') or 0) > float(previous.get('fetched_at') or 0):
                found[doc_id], found_path[doc_id] = record, path
    return found, found_path


def _sample_record_index(samples: list[dict], raw_candidates: dict[int, list[Path]]) -> tuple[dict[int, dict], dict[int, Path]]:
    wanted = {int(r.get('doc_id', core.primary_doc_id(r))): r.get('url_norm') for r in samples}
    cache, found, found_path = {}, {}, {}
    for sample in samples:
        doc_id = int(sample.get('doc_id', core.primary_doc_id(sample)))
        for path in raw_candidates.get(doc_id, []):
            if not path.is_file():
                continue
            key = str(path)
            if key not in cache:
                cache[key] = read_shard(path)
            record = next((r for r in cache[key]
                           if core.primary_doc_id(r) == doc_id and r.get('url_norm') == wanted[doc_id]
                           and (r.get('html') or '').strip()), None)
            if record is not None:
                found[doc_id], found_path[doc_id] = record, path
                break
    return found, found_path


def _tokenizer():
    from r2ai.gold_check.gold_check_common import load_tokenizer
    return load_tokenizer(download=False)


def _extractor(host: str, lang: str, config: Config):
    if lang == 'vi':
        name = config.for_domain(host).extractor
        return name, lambda html: extract_doc(html, name, host), 'vicrawl-site-v1'
    if host == 'ask.39.net' or host == 'familydoctor.com.cn':
        from r2ai.zh_v2 import extract as extractor
        name = 'zh_v2'
    else:
        from r2ai.zh_full_ingest import extract as extractor
        name = 'zh_full'
    return name, lambda html: extractor.extract(html, host), extractor.extractor_version()


_NON_ARTICLE_LINES = (
    re.compile(r'^\s*(?:Thứ\s+\w+,?\s+\d{1,2}/\d{1,2}/\d{4}.*(?:GMT|UTC).*)$', re.I),
    re.compile(r'^\s*(?:Bình luận:?|Bạn nghĩ gì về nội dung này\??|Bạn cần đăng nhập để gửi bình luận\.?|Gửi bình luận)\s*$', re.I),
    re.compile(r'^\s*(?:评论|我要评论|我要留言|分享至|分享到|相关阅读|相关文章)\s*[:：]?.*$', re.I),
)


def _generic_candidate(html: str, host: str, lang: str):
    if lang == 'vi':
        generic = extract_doc(html, 'generic', host)
    else:
        from r2ai.zh_sample.extract import extract_zh
        generic = extract_zh(html, host)
    generic.paragraphs = [p for p in generic.paragraphs
                          if p and p != generic.title and not any(pattern.fullmatch(p) for pattern in _NON_ARTICLE_LINES)]
    generic.headings = [h for h in generic.headings if h in set(generic.paragraphs)]
    return generic


def _extract_recovery_doc(html: str, host: str, lang: str, extract_one):
    html = core.strip_null_bytes(html)
    primary = extract_one(html)
    if core.good_text(primary.body) or host == 'ask.39.net' or primary.question or primary.answer:
        return primary
    generic = _generic_candidate(html, host, lang)
    if core.good_text(generic.body):
        generic.extractor = 'generic_recovery'
        return generic
    return primary


def _diagnose(host: str, failed: dict, raw: dict | None, old_row: dict | None,
              lang: str, extract_one) -> dict:
    doc_id = int(failed['doc_ids'][0])
    base = {
        'host': host, 'doc_id': doc_id, 'url': failed.get('url', ''),
        'old_chars': core.cleaned_length((old_row or {}).get('body')),
        'raw_shard': failed.get('_raw_path', ''), 'raw_http_status': '', 'raw_chars': 0,
        'new_chars': 0, 'cause': 'không raw', 'extractor': '',
    }
    if raw is None:
        return base
    raw_html = raw.get('html') or ''
    status = raw.get('http_status')
    base['raw_http_status'] = status if status is not None else ''
    base['raw_chars'] = len(raw_html)
    had_null_bytes = '\x00' in raw_html
    if core.encoding_suspect(raw_html):
        base['cause'] = 'encoding'
        return base
    html = core.strip_null_bytes(raw_html)
    if core.looks_blocked(html, status):
        base['cause'] = 'chặn/lỗi'
        return base
    primary = extract_one(html)
    generic = _generic_candidate(html, host, lang)
    doc = _extract_recovery_doc(html, host, lang, extract_one)
    body = doc.body or ''
    base['new_chars'] = core.cleaned_length(body)
    base['extractor'] = doc.extractor
    if had_null_bytes:
        base['cause'] = 'encoding'
    elif core.good_text(body) and not core.good_text((old_row or {}).get('body')):
        base['cause'] = 'selector sai'
    elif not core.good_text(generic.body):
        base['cause'] = 'thật sự rỗng'
    elif not core.good_text(primary.body):
        base['cause'] = 'selector sai'
    else:
        base['cause'] = 'selector sai'
    return base


def _doc_row(record: dict, doc, lang: str, v2_ids: set[int], tokenizer) -> dict:
    ids = [int(i) for i in (record.get('doc_ids') or []) if int(i) in v2_ids]
    primary = core.primary_doc_id(record)
    if primary is None or primary not in ids:
        raise ValueError('raw record primary doc_id is not a v2 ID')
    body = doc.body or ''
    tokens = len(tokenizer.encode(body, add_special_tokens=False)) if body else 0
    title = doc.title or ''
    return {
        'doc_ids': ids, 'url': record.get('url') or '', 'final_url': record.get('final_url') or record.get('url') or '',
        'domain': record.get('domain') or '', 'title': title, 'description': doc.description or '',
        'question': doc.question or '', 'answer': doc.answer or '', 'body': body, 'paragraphs': doc.paragraphs or [],
        'lang': lang, 'n_tokens_bge_m3': tokens, 'text_sha1': hashlib.sha1(body.encode('utf-8')).hexdigest(),
        'fetched_at': float(record.get('fetched_at') or 0), 'url_norm': record.get('url_norm') or '',
        'status': 'ok' if core.good_text(body) else 'thin', 'extractor': doc.extractor or '',
        'headings': doc.headings or [],
    }


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = assert_writable(path.with_name(path.name + '.partial'))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression='zstd')
    os.replace(tmp, path)


def _write_manifest(stage: Path, hosts: list[dict], payloads: dict[str, dict]) -> None:
    manifest = assert_writable(stage / 'manifest.db')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(manifest)
    try:
        connection.executescript(core.CC_MANIFEST_SCHEMA_SQL)
        now = time.time()
        for sid, host_row in enumerate(hosts, 30000):
            host, batch = host_row['host'], core.batch_label(host_row['host'])
            payload = payloads[host]
            connection.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)', (
                batch, 'cc', now, now, json.dumps([host]), json.dumps({'extracted': payload['raw_docs'], 'recovered': payload['recovered']}), '{}'))
            by_raw: dict[str, list[dict]] = defaultdict(list)
            for entry in payload['entries']:
                by_raw[entry['raw_shard']].append(entry)
            for raw_name, entries in sorted(by_raw.items()):
                path = entries[0]['raw_path_obj']
                stat = path.stat()
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                connection.execute('INSERT INTO shards(path,batch,domain,size,mtime_ns,sha256,n_urls,status,n_records,n_docs,docs_file,extractor_version,done_at) '
                                   'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                    raw_name, batch, host, stat.st_size, stat.st_mtime_ns, digest, len(entries), 'done', len(entries), len(entries),
                    str(payload['docs_path']), payload['extractor_version'], now))
                connection.executemany('INSERT INTO batch_urls VALUES (?,?,?,?)', [
                    (e['url_norm'], batch, host, raw_name) for e in entries])
            for entry in payload['entries']:
                connection.execute('INSERT INTO docs VALUES (?,?,?,?,?,?)', (
                    entry['url_norm'], batch, host, entry['doc_id'], entry['status'], str(payload['docs_path'])))
            name = f'{batch}-{host}'
            chunks_file = payload['segment_path'] / 'chunks_t256.parquet'
            if chunks_file.exists():
                with chunks_file.open('rb') as handle:
                    chunk_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            else:
                chunk_hash = None
            chunk_report = payload['chunk_report']
            rows = int(chunk_report.get('configs', {}).get('256', {}).get('chunks_all', 0))
            docs_count = int(chunk_report.get('docs', 0))
            chunker = {'target': core.CC_TARGET, 'min_body_tokens': core.CC_MIN_BODY_TOKENS,
                       'module': 'r2ai.index.chunk', 'provisional_segment_id': sid}
            stats = {'rows': rows, 'docs': docs_count, 'embed_pending': True, 'source': 'reextract-raw'}
            connection.execute(
                'INSERT INTO segments(segment_id,name,batch,domain,kind,status,rows,docs_count,global_start,chunks_sha256,extractor_version,chunker,model,vectors,note,stats,created_at,updated_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (sid, name, batch, host, 'cc', 'chunked', rows, docs_count, None, chunk_hash,
                 payload['extractor_version'], json.dumps(chunker), json.dumps(core.CC_MODEL), None,
                 'Staged; segment_id provisional. S15 must allocate a production ID and embed before indexing.',
                 json.dumps(stats), now, now))
        connection.commit()
    finally:
        connection.close()


def run() -> dict:
    hosts = _target_hosts()
    v2_by_host, _ = _load_v2()
    config = Config.load(CONFIG_DIR / 'domains.yaml')
    # Fail before writing if the required local tokenizer is absent. Its loader uses local_files_only=True.
    tokenizer = _tokenizer()
    assert_writable(STAGE_ROOT)
    if (STAGE_ROOT / 'manifest.db').exists():
        raise FileExistsError(f'Stage manifest already exists; refusing to overwrite: {STAGE_ROOT / "manifest.db"}')

    audits, sample_rows, missing_rows, payloads = [], [], [], {}
    stage_hosts = []
    for audit in hosts:
        host, lang = audit['host'], audit['lang']
        v2_ids = v2_by_host.get(host, set())
        latest, raw_candidates = _scan_host(host, lang, v2_ids)
        failures = _failed_rows(host, latest, v2_ids)
        extract_name, extract_one, extractor_version = _extractor(host, lang, config)
        recovered_rows, entries, sample_diagnoses = [], [], []
        by_id_failed = {int(r['doc_ids'][0]): r for r in failures}
        selected = core.select_sample(host, [{'doc_id': i, **r} for i, r in by_id_failed.items()], limit=20)
        batch_root = STAGE_ROOT / core.batch_label(host)
        docs_path = batch_root / 'docs' / f'{host}.parquet'
        prov_path = batch_root / 'provenance.parquet'
        segment_path = batch_root / 'segment'
        stage_hosts.append(audit)

        completed = all(p.is_file() for p in (
            docs_path, prov_path, segment_path / 'docs.parquet', segment_path / 'chunks_t256.parquet',
            segment_path / 'chunk_report.json'))
        if completed:
            recovered_rows = pq.read_table(docs_path).to_pylist()
            prov_rows = pq.read_table(prov_path).to_pylist()
            if pq.read_schema(docs_path).remove_metadata() != DOC_SCHEMA or pq.read_schema(prov_path).remove_metadata() != core.PROV_SCHEMA:
                raise ValueError(f'{host}: staged parquet schema does not match the CC contract')
            if len(recovered_rows) != len(prov_rows):
                raise ValueError(f'{host}: staged provenance/doc row counts differ')
            if any(not set(r['doc_ids']) <= v2_ids for r in recovered_rows):
                raise ValueError(f'{host}: staged docs contain non-v2 ids')
            prov_by_id = {int(r['doc_id']): r for r in prov_rows}
            entries = []
            for row in recovered_rows:
                doc_id = int(row['doc_ids'][0])
                p = prov_by_id[doc_id]
                entries.append({
                    'doc_id': doc_id, 'url_norm': row['url_norm'], 'status': row['status'],
                    'raw_shard': p['raw_shard'], 'raw_path_obj': _path_from_label(p['raw_shard']),
                    'body': row['body'], 'tokens': row['n_tokens_bge_m3'], 'extractor': row['extractor'], 'url': row['url'],
                })
            sample_raw, sample_paths = _sample_record_index(selected, raw_candidates)
            for sample in selected:
                doc_id = int(sample['doc_id'])
                diagnosis = _diagnose(host, sample, sample_raw.get(doc_id), latest.get(doc_id), lang, extract_one)
                if doc_id in sample_paths:
                    diagnosis['raw_shard'] = _raw_label(sample_paths[doc_id])
                sample_diagnoses.append(diagnosis)
            sample_rows.extend(sample_diagnoses)
            raw_records = {int(r['doc_ids'][0]): {} for r in recovered_rows}
            raw_paths = {doc_id: _path_from_label(p['raw_shard']) for doc_id, p in prov_by_id.items()}
            no_raw_ids = [doc_id for doc_id in by_id_failed if doc_id not in raw_records]
            for doc_id in no_raw_ids:
                row = by_id_failed[doc_id]
                candidate_paths = raw_candidates.get(doc_id, [])
                missing_rows.append({
                    'lang': lang, 'host': host, 'doc_id': doc_id, 'url': row.get('url', ''),
                    'raw_candidate': ';'.join(_raw_label(p) for p in candidate_paths),
                    'reason': 'raw shard missing' if not any(p.is_file() for p in candidate_paths) else 'doc_id not found in candidate shard',
                })
            chunk_report = json.loads((segment_path / 'chunk_report.json').read_text(encoding='utf-8'))
            raw_count = len(raw_records)
            recovered = sum(core.good_text(r['body']) for r in recovered_rows)
            still_empty_local = len(failures) - recovered
            payloads[host] = {
                'entries': entries, 'raw_docs': len(recovered_rows), 'recovered': recovered, 'docs_path': docs_path,
                'segment_path': segment_path, 'chunk_report': chunk_report, 'extractor_version': extractor_version,
            }
            audits.append({
                'lang': lang, 'host': host, 's20_initial': int(audit['empty_or_extract_error'] or 0),
                'local_failures': len(failures), 'raw': raw_count, 'no_raw': len(failures) - raw_count,
                'recovered': recovered, 'still_empty_local': still_empty_local,
                'snapshot_delta': len(failures) - int(audit['empty_or_extract_error'] or 0),
                'batch': core.batch_label(host), 'docs_path': str(docs_path), 'provenance_path': str(prov_path),
                'segment_path': str(segment_path), 'docs_staged': len(recovered_rows),
                'segment_docs': int(chunk_report.get('docs', 0)), 'v2_verified_docs': len(recovered_rows),
                'extractor': extract_name, 'causes': Counter(r['cause'] for r in sample_diagnoses),
            })
            print(f'{host}: resumed staged batch; s20={audit["empty_or_extract_error"]} local={len(failures)} raw={raw_count} recovered={recovered}', flush=True)
            continue

        raw_records, raw_paths = _raw_record_index(failures, raw_candidates)
        for sample in selected:
            doc_id = int(sample['doc_id'])
            rec = raw_records.get(doc_id)
            old = latest.get(doc_id)
            diagnosis = _diagnose(host, sample, rec, old, lang, extract_one)
            if doc_id in raw_paths:
                diagnosis['raw_shard'] = _raw_label(raw_paths[doc_id])
            sample_diagnoses.append(diagnosis)
        sample_rows.extend(sample_diagnoses)

        raw_docs = 0
        for doc_id, record in sorted(raw_records.items()):
            if doc_id not in by_id_failed:
                continue
            html = record.get('html') or ''
            doc = _extract_recovery_doc(html, host, lang, extract_one)
            row = _doc_row(record, doc, lang, v2_ids, tokenizer)
            raw_path = raw_paths[doc_id]
            raw_rel = _raw_label(raw_path)
            recovered_rows.append(row)
            entries.append({
                'doc_id': doc_id, 'url_norm': row['url_norm'], 'status': row['status'], 'raw_shard': raw_rel,
                'raw_path_obj': raw_path,
                'body': row['body'], 'tokens': row['n_tokens_bge_m3'], 'extractor': row['extractor'], 'url': row['url'],
                'record': record,
            })
            raw_docs += 1

        if len({r['doc_ids'][0] for r in recovered_rows}) != len(recovered_rows):
            raise ValueError(f'{host}: primary doc_id duplicated after source selection')
        urls = [r['url_norm'] for r in recovered_rows]
        if any(not url for url in urls) or len(set(urls)) != len(urls):
            raise ValueError(f'{host}: output url_norm is empty or duplicated')
        if any(not set(r['doc_ids']) <= v2_ids for r in recovered_rows):
            raise ValueError(f'{host}: output doc_ids contain a non-v2 id')
        for row in recovered_rows:
            if row['status'] == 'ok':
                continue
        if recovered_rows:
            _write_parquet(docs_path, recovered_rows, DOC_SCHEMA)
        else:
            _write_parquet(docs_path, [], DOC_SCHEMA)

        prov_rows = []
        try:
            import trafilatura
            trafilatura_version = getattr(trafilatura, '__version__', '')
        except Exception:  # noqa: BLE001
            trafilatura_version = ''
        for row, entry in zip(recovered_rows, entries):
            prov_rows.append(core.provenance_row(
                doc_id=row['doc_ids'][0], url=row['url'], raw_shard=entry['raw_shard'], status=row['status'], body=row['body'],
                extractor=row['extractor'], extractor_name=extract_name, fetched_at=row['fetched_at'],
                n_tokens=row['n_tokens_bge_m3'], trafilatura=trafilatura_version))
        _write_parquet(prov_path, prov_rows, core.PROV_SCHEMA)

        chunk_report = {'docs': 0, 'configs': {}}
        if any(r['status'] == 'ok' and r['n_tokens_bge_m3'] >= 50 and r['body'] for r in recovered_rows):
            from r2ai.index.chunk import main as chunk_main
            assert_writable(segment_path)
            segment_path.mkdir(parents=True, exist_ok=False)
            chunk_main(['--docs-dir', str(docs_path.parent), '--out-dir', str(segment_path), '--targets', '256', '--min-body-tokens', '50'])
            chunk_report = json.loads((segment_path / 'chunk_report.json').read_text(encoding='utf-8'))
            # The chunker sees only recovered docs in this batch and has an independent 100% verbatim gate.
            chunked_docs = pq.read_table(segment_path / 'docs.parquet', columns=['doc_id', 'doc_ids_group']).to_pylist()
            chunk_ids = {int(r['doc_id']) for r in chunked_docs}
            chunk_groups = {int(i) for r in chunked_docs for i in r['doc_ids_group']}
            if not chunk_ids <= v2_ids or not chunk_groups <= v2_ids:
                raise ValueError(f'{host}: chunked segment contains a non-v2 doc_id')
        elif not segment_path.exists():
            assert_writable(segment_path)
            segment_path.mkdir(parents=True)
            _write_parquet(segment_path / 'docs.parquet', [], __import__('r2ai.index.chunk', fromlist=['DOCS_SCHEMA']).DOCS_SCHEMA)
            _write_parquet(segment_path / 'chunks_t256.parquet', [], __import__('r2ai.index.chunk', fromlist=['CHUNKS_SCHEMA']).CHUNKS_SCHEMA)
            (segment_path / 'chunk_report.json').write_text(json.dumps(chunk_report), encoding='utf-8')

        raw_count = len(raw_records)
        no_raw_ids = [int(r['doc_ids'][0]) for r in failures if int(r['doc_ids'][0]) not in raw_records]
        for doc_id in no_raw_ids:
            row = by_id_failed[doc_id]
            candidate_paths = raw_candidates.get(doc_id, [])
            missing_rows.append({
                'lang': lang, 'host': host, 'doc_id': doc_id, 'url': row.get('url', ''),
                'raw_candidate': ';'.join(_raw_label(p) for p in candidate_paths),
                'reason': 'raw shard missing' if not any(p.is_file() for p in candidate_paths) else 'doc_id not found in candidate shard',
            })
        recovered = sum(core.good_text(r['body']) for r in recovered_rows)
        still_empty_local = len(failures) - recovered
        s20_initial = int(audit['empty_or_extract_error'] or 0)
        output_prefix = STAGE_ROOT / core.batch_label(host)
        for entry in entries:
            # Use relative paths so the bundle remains relocatable within this repo.
            entry['raw_shard'] = entry['raw_shard']
        payloads[host] = {
            'entries': entries, 'raw_docs': raw_docs, 'recovered': recovered, 'docs_path': docs_path,
            'segment_path': segment_path, 'chunk_report': chunk_report, 'extractor_version': extractor_version,
        }
        audits.append({
            'lang': lang, 'host': host, 's20_initial': s20_initial, 'local_failures': len(failures),
            'raw': raw_count, 'no_raw': len(failures) - raw_count, 'recovered': recovered,
            'still_empty_local': still_empty_local, 'snapshot_delta': len(failures) - s20_initial,
            'batch': core.batch_label(host), 'docs_path': str(docs_path), 'provenance_path': str(prov_path),
            'segment_path': str(segment_path), 'docs_staged': len(recovered_rows),
            'segment_docs': int(chunk_report.get('docs', 0)), 'v2_verified_docs': len(recovered_rows), 'extractor': extract_name,
            'causes': Counter(r['cause'] for r in sample_diagnoses),
        })
        print(f'{host}: s20={s20_initial} local={len(failures)} raw={raw_count} recovered={recovered} empty={still_empty_local}', flush=True)

    _write_manifest(STAGE_ROOT, stage_hosts, payloads)
    _write_csv(SAMPLES_CSV, sample_rows,
               ['host', 'doc_id', 'url', 'old_chars', 'raw_shard', 'raw_http_status', 'raw_chars', 'new_chars', 'cause', 'extractor'])
    _write_csv(MISSING_CSV, missing_rows, ['lang', 'host', 'doc_id', 'url', 'raw_candidate', 'reason'])
    _write_report(audits, len(hosts), len(sample_rows), len(missing_rows))
    return {'stage_root': str(STAGE_ROOT), 'manifest': str(STAGE_ROOT / 'manifest.db'), 'hosts': audits,
            'sample_rows': len(sample_rows), 'missing_rows': len(missing_rows)}


def _write_report(audits: list[dict], n_hosts: int, n_samples: int, n_missing: int) -> None:
    assert_writable(REPORT_MD)
    lines = [
        '# S22 — Khôi phục text từ raw HTML cục bộ', '',
        '- Chỉ đọc raw đã lưu trong repo; không có request mạng.',
        '- Tập host: S20 `empty_or_extract_error >= 50`, VI/ZH; loại PubMed EN theo D17.',
        f'- S20 input: `{S20_CSV}` nếu có; checkout hiện chuyển sang `feat/s21-120ask` không có `reports/`, nên fallback đọc blob local `{S20_SOURCE_COMMIT}:reports/s20-coverage-audit.csv`.',
        '- Text tốt: ≥200 Unicode code point sau gộp whitespace. 120ask dùng `status != ok` để chọn lỗi theo proxy S20, rồi đo lại theo ngưỡng 200.',
        '- Một batch/domain cho mỗi host theo CC contract S15. Bundle tách riêng; không ghi `data/index`, `state/cc_longchau`, hay index chung.',
        '',
        '| Lang | Host | Lỗi S20 | Lỗi local | Raw có | Raw thiếu | Thu hồi ≥200 | Còn rỗng local | Docs staged | Docs segment | V2 | Lệch snapshot | Nguyên nhân mẫu |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in audits:
        causes = ', '.join(f'{k} {v}' for k, v in sorted(r['causes'].items())) or 'không có mẫu'
        lines.append(f"| {r['lang']} | `{r['host']}` | {r['s20_initial']} | {r['local_failures']} | {r['raw']} | {r['no_raw']} | {r['recovered']} | {r['still_empty_local']} | {r['docs_staged']} | {r['segment_docs']} | 100% ({r['v2_verified_docs']}) | {r['snapshot_delta']:+d} | {causes} |")
    lines += [
        '', '## Bundle S15', '',
        f'- Root: `{STAGE_ROOT}`; manifest: `{STAGE_ROOT / "manifest.db"}` theo `r2ai.zh_full_ingest.ingest.SCHEMA_SQL`.',
        '- Mỗi host có batch `cc-s22-{host}`, `docs/<host>.parquet` theo `vicrawl.extract_pipeline.SCHEMA`, `provenance.parquet` theo S15 `PROV_SCHEMA`, `source=reextract-raw`, `raw_shard` là đường dẫn repo-relative.',
        '- Mỗi batch có segment chunked t256 (`docs.parquet`, `chunks_t256.parquet`, `chunk_report.json`). Segment ID trong bundle là provisional; S15 cấp ID production và embed trước khi index.',
        '- Membership được kiểm trên doc ID chính và toàn bộ `doc_ids` group cho docs; chunk docs được kiểm độc lập trong v2.',
        '- Segment artifact không chứa vector và không được đăng ký vào index chung.',
        '', '### Đường dẫn từng batch', '',
        *[f"- `{r['batch']}`: `{STAGE_ROOT / r['batch']}` — docs={r['docs_staged']}, segment docs={r['segment_docs']}, v2=100% ({r['v2_verified_docs']})." for r in audits],
        '', '## Chẩn đoán và raw thiếu', '',
        f'- Mẫu chẩn đoán: {n_samples} dòng trong `reports/s22_sample_diagnosis.csv` (tối đa 20 lỗi/host).',
        f'- ID không đọc được raw: {n_missing} dòng trong `reports/s22_raw_missing_ids.csv`.',
        '- Lệch snapshot được giữ riêng ở cột “Lệch snapshot”: aggregate S20 không chứa danh sách ID; local failure set được chọn lại từ docs parquet hiện có.',
        '',
    ]
    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')


def audit_samples() -> dict:
    """Read-only first pass: count raw paths and classify a deterministic 20-doc sample per host."""
    hosts = _target_hosts()
    v2_by_host, _ = _load_v2()
    config = Config.load(CONFIG_DIR / 'domains.yaml')
    result = []
    for audit in hosts:
        host, lang = audit['host'], audit['lang']
        latest, candidates = _scan_host(host, lang, v2_by_host.get(host, set()))
        failures = _failed_rows(host, latest, v2_by_host.get(host, set()))
        existing = {int(r['doc_ids'][0]): [p for p in candidates.get(int(r['doc_ids'][0]), []) if p.is_file()]
                    for r in failures}
        extract_name, extract_one, _ = _extractor(host, lang, config)
        diagnoses = []
        records_by_path = {}
        samples = core.select_sample(host, [{'doc_id': int(r['doc_ids'][0]), **r} for r in failures], 20)
        for failed in samples:
            doc_id = int(failed['doc_id'])
            raw = None
            for path in existing.get(doc_id, []):
                key = str(path)
                if key not in records_by_path:
                    records_by_path[key] = read_shard(path)
                records = records_by_path[key]
                for record in records:
                    if core.primary_doc_id(record) == doc_id:
                        raw = record
                        failed['_raw_path'] = str(path)
                        break
                if raw is not None:
                    break
            diagnosis = _diagnose(host, failed, raw, latest.get(doc_id), lang, extract_one)
            if raw is not None:
                diagnosis['raw_shard'] = _raw_label(Path(failed['_raw_path']))
            diagnoses.append(diagnosis)
        causes = Counter(d['cause'] for d in diagnoses)
        item = {
            'lang': lang, 'host': host, 's20_initial': int(audit['empty_or_extract_error'] or 0),
            'local_failures': len(failures), 'raw_path_exists': sum(bool(existing.get(int(r['doc_ids'][0]))) for r in failures),
            'no_raw_path': sum(not existing.get(int(r['doc_ids'][0])) for r in failures),
            'sample': len(diagnoses), 'causes': dict(causes), 'extractor': extract_name,
        }
        result.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return {'hosts': result, 'sample_rows': sum(x['sample'] for x in result)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'run'], nargs='?', default='run')
    args = parser.parse_args()
    result = audit_samples() if args.command == 'audit' else run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
