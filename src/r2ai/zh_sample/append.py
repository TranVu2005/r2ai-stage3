"""CPU-only diagnostic: append cached dense zh IDs to the byte-pinned RD150."""
from __future__ import annotations

from r2ai.paths import ROOT, QUERY_FILE, CORPUS_FILE, require_inputs, resolve_path
from .common import RUN, CHUNKS, INDEX, guard_owned, preflight, atomic_json, exclusive

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

RD150_SHA256 = '136c259145c6fa134bcd5d4a877575088e28346ef43636afc87d2e283508e950'
ZIP_CEILING = 104_857_600
BASE = ROOT / 'out/runs/rrf-docs-2026-10-06/RD150/sub_RD150.json'
_DECODER = json.JSONDecoder()


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pinned_base(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RD150_SHA256:
        raise ValueError('RD150 SHA256 mismatch; refusing mutable baseline')
    return raw.decode('utf-8')


def guarded_output(path):
    p = resolve_path(path)
    if not p.is_relative_to(RUN):
        raise ValueError('Append outputs must remain under out/runs/zh-sample')
    return guard_owned(path)


def _skip(text, pos):
    while pos < len(text) and text[pos] in ' \t\r\n':
        pos += 1
    return pos


def _expect(text, pos, char):
    pos = _skip(text, pos)
    if text[pos:pos + 1] != char:
        raise ValueError(f'Expected {char!r} at offset {pos}')
    return pos + 1


def _rows(text):
    """Decode one row at a time, retaining literal field spans (including escapes)."""
    pos = _expect(text, 0, '[')
    seen = set()
    if text[_skip(text, pos):_skip(text, pos) + 1] == ']':
        pos = _expect(text, pos, ']')
    else:
        while True:
            pos = _expect(text, pos, '{')
            values, spans = {}, {}
            while True:
                key, pos = _DECODER.raw_decode(text, _skip(text, pos))
                if key not in ('id', 'relevant_docs', 'relevant_chunks') or key in values:
                    raise ValueError(f'Unexpected/duplicate row key: {key!r}')
                pos = _expect(text, pos, ':')
                start = _skip(text, pos)
                value, pos = _DECODER.raw_decode(text, start)
                values[key], spans[key] = value, (start, pos)
                pos = _skip(text, pos)
                if text[pos:pos + 1] == '}':
                    pos += 1
                    break
                pos = _expect(text, pos, ',')
            if set(values) != {'id', 'relevant_docs', 'relevant_chunks'}:
                raise ValueError('Incomplete row schema')
            qid, docs = values['id'], values['relevant_docs']
            if type(qid) is not int or qid in seen:
                raise ValueError('Invalid/duplicate query id')
            if (not isinstance(docs, list) or any(type(d) is not int for d in docs)
                    or len(set(docs)) != len(docs) or not isinstance(values['relevant_chunks'], list)):
                raise ValueError('Invalid docs/chunks schema')
            seen.add(qid)
            yield values, spans
            pos = _skip(text, pos)
            if text[pos:pos + 1] == ']':
                pos += 1
                break
            pos = _expect(text, pos, ',')
    if _skip(text, pos) != len(text):
        raise ValueError('Trailing JSON data')


def select_docs(pairs, known_zh_ids, vi_ids, limit=100):
    """Same max chunk-logit -> primary doc rule as Z; stable ties, no expansion."""
    if type(limit) is not int or not 0 <= limit <= 100:
        raise ValueError('Append limit must be 0..100')
    scores = {}
    vi = set(vi_ids)
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('Invalid cached score pair')
        doc, score = pair
        if (type(doc) is not int or doc not in known_zh_ids
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Unknown zh primary ID or nonfinite/invalid score')
        if doc not in vi:
            scores[doc] = max(scores.get(doc, -math.inf), score)
    ordered = sorted(scores, key=lambda d: -scores[d])
    return ordered[:limit], len(ordered)


def _addition(docs, selected):
    return ((', ' if docs else '') + ', '.join(map(str, selected))) if selected else ''


def render_append(base, cache, known_zh_ids, write):
    cursor, result = 0, {}
    for row, spans in _rows(base):
        qid = row['id']
        if qid not in cache:
            raise ValueError(f'Missing cached query {qid}')
        selected, available = select_docs(cache[qid], known_zh_ids, row['relevant_docs'])
        end = spans['relevant_docs'][1]
        write(base[cursor:end - 1])
        write(_addition(row['relevant_docs'], selected))
        cursor = end - 1
        result[qid] = {'available': available, 'appended': selected}
    if set(result) != set(cache):
        raise ValueError('Extra cached query ids')
    write(base[cursor:])
    return result


def verify_gate(base, result, selections):
    # Compare every byte outside docs and the reconstructed VI docs literal.
    original = iter(_rows(base))
    cursor_base = cursor_result = count = 0
    for row, spans in _rows(result):
        before = next(original, None)
        if before is None or before[0]['id'] != row['id']:
            raise ValueError('Gate: query order/count changed')
        vi, old_spans = before
        qid = vi['id']
        selected = selections[qid]['appended']
        if row['relevant_docs'] != vi['relevant_docs'] + selected:
            raise ValueError(f'Gate: VI docs order/prefix changed, query {qid}')
        old_lo, old_hi = old_spans['relevant_docs']
        lo, hi = spans['relevant_docs']
        expected = base[old_lo:old_hi - 1] + _addition(vi['relevant_docs'], selected) + ']'
        if result[lo:hi] != expected or base[cursor_base:old_lo] != result[cursor_result:lo]:
            raise ValueError(f'Gate: raw VI docs or other bytes changed, query {qid}')
        ca, cb = old_spans['relevant_chunks'], spans['relevant_chunks']
        if base[ca[0]:ca[1]] != result[cb[0]:cb[1]]:
            raise ValueError(f'Gate: chunk bytes changed, query {qid}')
        cursor_base, cursor_result = old_hi, hi
        count += 1
    if next(original, None) is not None or base[cursor_base:] != result[cursor_result:]:
        raise ValueError('Gate: remaining query/chunk bytes changed')
    if count != len(selections):
        raise ValueError('Gate: selection query count mismatch')
    return {'queries': count, 'vi_docs_diff': 0, 'chunks_byte_diff': 0}


def _distribution(values):
    import numpy as np
    return {'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)),
            'p5': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95)),
            'min': min(values), 'max': max(values)}


def _load_cache(query_ids):
    config_path = RUN / 'retrieval/config.json'
    cfg = json.loads(config_path.read_text('utf-8'))
    for key, path in [('query_sha256', QUERY_FILE), ('chunk_sha256', CHUNKS / 'chunks_t256.parquet'),
                      ('index_manifest_sha256', INDEX / 'input_manifest.json')]:
        if cfg[key] != digest(path):
            raise ValueError(f'Cache input changed: {key}')
    manifest = json.loads((INDEX / 'input_manifest.json').read_text('utf-8'))
    if (manifest['extract_manifest_sha256'] != digest(RUN / 'extract_manifest.json')
            or manifest['chunks_sha256'] != cfg['chunk_sha256'] or cfg['seed'] != 42
            or cfg['vi']['reranker'] != 'BAAI/bge-reranker-v2-m3'):
        raise ValueError('Cache bundle/reranker/seed mismatch')
    cache, fingerprint = {}, hashlib.sha256()
    paths = list((RUN / 'retrieval').glob('q*.json'))
    if len(paths) != len(query_ids):
        raise ValueError('Cached query file count mismatch')
    for qid in sorted(query_ids):
        path = RUN / f'retrieval/q{qid}.json'
        raw = path.read_bytes()
        row = json.loads(raw)
        if type(row['query_id']) is not int or row['query_id'] != qid:
            raise ValueError(f'Cache query mismatch: {path.name}')
        cache[qid] = row['dense']
        fingerprint.update(path.name.encode() + b'\0' + raw)
    return cache, cfg, {'config_sha256': digest(config_path), 'cache_sha256': fingerprint.hexdigest()}


def validate_stage(path, validate, arguments, log):
    """Validate a staging ZIP; failed validation never publishes final artifacts."""
    path, log = guarded_output(path), guarded_output(log)
    if path.suffix != '.zip':
        raise ValueError('Validator staging artifact must have .zip suffix')
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exit_code = validate([str(path), *arguments])
    validator_text = capture.getvalue()
    log.write_text(validator_text, encoding='utf-8')
    if exit_code != 0:
        raise ValueError(f'Validator failed ({exit_code}); see {log}')
    result = json.loads(validator_text.strip())
    if result.get('errors') != 0 or result.get('ok') is not True or result.get('rows') != 1200:
        raise ValueError('Validator did not confirm 1200 rows / 0 errors')
    return result


def publish_artifacts(tmp_json, tmp_zip, final_json, final_zip, stats, selections=None):
    paths = [guarded_output(p) for p in (tmp_json, tmp_zip, final_json, final_zip)]
    stats_path = guarded_output(paths[2].parent / 'stats.json')
    # Two renames cannot be atomic together. Invalidate old success metadata first;
    # interruption leaves explicit incomplete status, never a stale successful gate.
    atomic_json(stats_path, {'complete': False, 'status': 'publishing validated staging artifacts'})
    if selections is not None:
        atomic_json(guarded_output(paths[2].parent / 'query_append.json'), selections)
    os.replace(paths[0], paths[2])
    os.replace(paths[1], paths[3])
    atomic_json(stats_path, {**stats, 'complete': True})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default=str(BASE))
    ap.add_argument('--out-dir', default=str(RUN / 'Z-append'))
    ap.add_argument('--docs-dir', required=True)
    ap.add_argument('--chunks-dir', required=True)
    args = ap.parse_args(argv)
    # Everything remains local, including the validator's cached CPU tokenizer.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    started = time.perf_counter()
    base_path = resolve_path(args.base)
    base = pinned_base(base_path)
    docs_dir, chunks_dir = resolve_path(args.docs_dir), resolve_path(args.chunks_dir)
    if docs_dir != ROOT / 'data/docs_vi' or chunks_dir != ROOT / 'data/chunks':
        raise ValueError('Explicit VI docs/chunks paths must select the NEW bundle')
    require_inputs(docs_dir, chunks_dir, QUERY_FILE, CORPUS_FILE, CHUNKS / 'docs.parquet')
    out = guarded_output(args.out_dir)
    preflight()
    import pyarrow.parquet as pq
    query_ids = pq.read_table(QUERY_FILE, columns=['id']).column('id').to_pylist()
    if len(query_ids) != 1200 or len(set(query_ids)) != 1200:
        raise ValueError('Need 1200 unique query ids')
    cache, cfg, cache_hashes = _load_cache(query_ids)
    zh_ids = set(pq.read_table(CHUNKS / 'docs.parquet', columns=['doc_id']).column('doc_id').to_pylist())
    with exclusive('append'):
        out.mkdir(parents=True, exist_ok=True)
        final_json, final_zip = guarded_output(out / 'Z_append.json'), guarded_output(out / 'Z_append.zip')
        tmp_json, tmp_zip = guarded_output(out / 'Z_append.json.tmp'), guarded_output(out / 'Z_append.pending.zip')
        with tmp_json.open('w', encoding='utf-8', newline='') as f:
            selections = render_append(base, cache, zh_ids, f.write)
            f.flush()
            os.fsync(f.fileno())
        with tmp_json.open(encoding='utf-8', newline='') as f:
            gate = verify_gate(base, f.read(), selections)
        if gate['queries'] != 1200 or set(selections) != set(query_ids):
            raise ValueError('Need exact 1200-query byte gate')
        # Free the large baseline before the independent validator loads JSON.
        del base
        with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(tmp_json, arcname=final_json.name)
        if tmp_zip.stat().st_size > ZIP_CEILING:
            raise ValueError(f'ZIP exceeds {ZIP_CEILING} bytes')
        build_seconds = time.perf_counter() - started
        from r2ai.submit.validate_submission import main as validate
        validator_started = time.perf_counter()
        validator = validate_stage(tmp_zip, validate,
                                   ['--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                                    '--docs-dir', str(docs_dir), '--max-zip-mib', '100'], out / 'validator.log')
        stats = {'diagnostic_only': True, 'LB': 'chưa đo', 'base': str(base_path), 'base_sha256': RD150_SHA256,
                 'baseline_Doc_R_user_provided': .3412, 'branch': 'dense', 'seed': 42,
                 'aggregation': 'max cached reranker chunk logit per primary doc; no expansion',
                 'cache_config': cfg, **cache_hashes, 'gate': gate, 'validator': validator,
                 'json_bytes': tmp_json.stat().st_size, 'json_sha256': digest(tmp_json),
                 'zip_bytes': tmp_zip.stat().st_size, 'zip_sha256': digest(tmp_zip),
                 'zip_ceiling_bytes': ZIP_CEILING, 'build_seconds': build_seconds,
                 'validator_seconds': time.perf_counter() - validator_started,
                 'available_zh': _distribution([r['available'] for r in selections.values()]),
                 'appended_zh': _distribution([len(r['appended']) for r in selections.values()]),
                 'queries_below_100': sum(r['available'] < 100 for r in selections.values()),
                 'appended_total': sum(len(r['appended']) for r in selections.values()),
                 'new_model_calls': 0, 'new_gpu_seconds': 0}
        publish_artifacts(tmp_json, tmp_zip, final_json, final_zip, stats, selections)
        # PowerShell pipes may expose cp1252 stdout; metadata must print safely.
        print(json.dumps(stats, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
