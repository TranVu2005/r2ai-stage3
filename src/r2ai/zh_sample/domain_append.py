"""CPU diagnostics: retain one domain from the pinned Z-append ID lists."""
from __future__ import annotations

from r2ai.paths import ROOT, OLD_ROOT, RUNS_DIR, QUERY_FILE, CORPUS_FILE, assert_writable, require_inputs, resolve_path
from .append import BASE, RD150_SHA256, ZIP_CEILING, _addition, _distribution, _rows, digest, pinned_base, verify_gate
from .common import CHUNKS, RUN

import argparse
import contextlib
import ctypes
import io
import json
import os
from pathlib import Path
import time
import zipfile

DIAGNOSTICS = RUNS_DIR / 'zh-full/diagnostics'
SOURCE_SHA256 = 'b461ea2bc53284eb86e634f5b5af25e128529799ff37667f75e3019202ee03f6'
SOURCE_DIR = RUN / 'Z-append'
DOMAIN_NAMES = {'120ask.com': 'ZA-120ask', 'cnkang.com': 'ZA-cnkang'}


def guard_output(path):
    """Resolve every concrete write target; descendants must stay in diagnostics."""
    result = resolve_path(path)
    if OLD_ROOT is not None and result.is_relative_to(OLD_ROOT):
        raise ValueError('Write to OLD is forbidden')
    if not result.is_relative_to(DIAGNOSTICS):
        raise ValueError('Domain diagnostics outputs must remain under out/runs/zh-full/diagnostics')
    return result


def _preflight(out):
    assert_writable(out)  # Recursive OLD check once, before any output I/O.
    pending = [out] if out.is_dir() else []
    seen = set()
    while pending:
        directory = pending.pop()
        if directory in seen:
            continue
        seen.add(directory)
        for child in directory.iterdir():
            target = guard_output(child)
            if target.is_dir():
                pending.append(target)


def _atomic_json(path, value):
    path = guard_output(path)
    tmp = guard_output(path.with_name(path.name + '.tmp'))
    with tmp.open('w', encoding='utf-8', newline='') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def _exclusive(out):
    import msvcrt
    path = guard_output(out / 'build.lock')
    with path.open('a+b') as f:
        f.seek(0, os.SEEK_END)
        if not f.tell():
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError('Domain diagnostic already running') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def filter_selections(source, domains, domain):
    """Filter the existing ordered <=100 primary IDs; never rerank or top up."""
    result = {}
    for qid, row in source.items():
        if type(qid) is not int or qid in result:
            raise ValueError('Invalid/duplicate source query id')
        ids = row.get('appended')
        if (not isinstance(ids, list) or len(ids) > 100
                or any(type(d) is not int or d not in domains for d in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Invalid/unknown source primary IDs or >100 source count')
        selected = [d for d in ids if domains[d] == domain]
        result[qid] = {'available': len(selected), 'appended': selected}
    return result


def render_selected(base, selections, write):
    cursor, seen = 0, set()
    for row, spans in _rows(base):
        qid = row['id']
        if qid not in selections:
            raise ValueError(f'Missing source query {qid}')
        ids = selections[qid]['appended']
        if (len(ids) > 100 or len(set(ids)) != len(ids)
                or any(type(d) is not int for d in ids) or set(ids) & set(row['relevant_docs'])):
            raise ValueError('Invalid append count/IDs or VI overlap')
        end = spans['relevant_docs'][1]
        write(base[cursor:end - 1])
        write(_addition(row['relevant_docs'], ids))
        cursor = end - 1
        seen.add(qid)
    if seen != set(selections):
        raise ValueError('Extra source query ids')
    write(base[cursor:])


def verify_source(base, source, selections):
    return verify_gate(base, source, selections)


def _available_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_phys', 'avail_phys', 'total_pagefile', 'avail_pagefile', 'total_virtual', 'avail_virtual', 'avail_extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Unable to read Windows available physical memory')
    return status.avail_phys


def require_validation_memory(available=None):
    available = _available_memory() if available is None else available
    if available < 4 * 1024**3:
        raise RuntimeError(f'Validation needs >=4 GiB free physical memory; available={available} bytes')
    return available


def _load_source(base):
    source_path, selections_path, stats_path = [SOURCE_DIR / n for n in
                                               ('Z_append.json', 'query_append.json', 'stats.json')]
    require_inputs(source_path, selections_path, stats_path, CHUNKS / 'docs.parquet')
    if digest(source_path) != SOURCE_SHA256:
        raise ValueError('Z-append SHA256 mismatch')
    stats = json.loads(stats_path.read_text('utf-8'))
    if (stats.get('complete') is not True or stats.get('base_sha256') != RD150_SHA256
            or stats.get('json_sha256') != SOURCE_SHA256):
        raise ValueError('Z-append source stats/pins are incomplete or changed')
    raw = json.loads(selections_path.read_text('utf-8'))
    selections = {int(qid): row for qid, row in raw.items()}
    if len(selections) != len(raw):
        raise ValueError('Source query ids collide')
    with source_path.open(encoding='utf-8', newline='') as f:
        gate = verify_source(base, f.read(), selections)
    if gate['queries'] != 1200:
        raise ValueError('Source needs exactly 1200 queries')
    import pyarrow.parquet as pq
    table = pq.read_table(CHUNKS / 'docs.parquet', columns=['doc_id', 'domain']).to_pylist()
    domains = {row['doc_id']: row['domain'] for row in table}
    if len(domains) != len(table):
        raise ValueError('ZH primary doc IDs must be unique')
    query_ids = pq.read_table(QUERY_FILE, columns=['id']).column('id').to_pylist()
    if len(query_ids) != 1200 or set(query_ids) != set(selections):
        raise ValueError('Source query set differs from query.parquet')
    metadata = {'source_json_sha256': SOURCE_SHA256, 'source_json_path': str(source_path),
                'source_selections_sha256': digest(selections_path), 'source_stats_sha256': digest(stats_path),
                'primary_domain_map_sha256': digest(CHUNKS / 'docs.parquet'),
                'query_sha256': digest(QUERY_FILE), 'corpus_sha256': digest(CORPUS_FILE),
                'config_sha256': stats['config_sha256'], 'cache_sha256': stats['cache_sha256'],
                'cache_config': stats['cache_config'], 'source_gate': gate}
    return selections, domains, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True, choices=DOMAIN_NAMES)
    parser.add_argument('--out-dir')
    parser.add_argument('--docs-dir', required=True)
    parser.add_argument('--chunks-dir', required=True)
    args = parser.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    started = time.perf_counter()
    docs, chunks = resolve_path(args.docs_dir), resolve_path(args.chunks_dir)
    if docs != ROOT / 'data/docs_vi' or chunks != ROOT / 'data/chunks':
        raise ValueError('Explicit VI docs/chunks paths must select the NEW bundle')
    require_inputs(BASE, QUERY_FILE, CORPUS_FILE, docs, chunks)
    name = DOMAIN_NAMES[args.domain]
    out = guard_output(args.out_dir or DIAGNOSTICS / name)
    _preflight(out)
    base = pinned_base(BASE)
    source, domains, metadata = _load_source(base)
    selections = filter_selections(source, domains, args.domain)
    out.mkdir(parents=True, exist_ok=True)
    with _exclusive(out):
        final_json, final_zip = [guard_output(out / (name + suffix)) for suffix in ('.json', '.zip')]
        tmp_json, tmp_zip = [guard_output(out / (name + suffix)) for suffix in ('.json.tmp', '.pending.zip')]
        with tmp_json.open('w', encoding='utf-8', newline='') as f:
            render_selected(base, selections, f.write)
            f.flush()
            os.fsync(f.fileno())
        with tmp_json.open(encoding='utf-8', newline='') as f:
            gate = verify_gate(base, f.read(), selections)
        del base
        if gate['queries'] != 1200:
            raise ValueError('Need exact 1200-query byte gate')
        with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(tmp_json, arcname=final_json.name)
        if tmp_zip.stat().st_size > ZIP_CEILING:
            raise ValueError(f'ZIP exceeds {ZIP_CEILING} bytes')
        build_seconds = time.perf_counter() - started
        available = require_validation_memory()
        print(json.dumps({'phase': 'validate', 'domain': args.domain, 'free_memory_bytes': available}), flush=True)
        from r2ai.submit.validate_submission import main as validate
        validator_started = time.perf_counter()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = validate([str(tmp_zip), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                             '--docs-dir', str(docs), '--max-zip-mib', '100'])
        guard_output(out / 'validator.log').write_text(capture.getvalue(), encoding='utf-8')
        validator = json.loads(capture.getvalue().strip())
        if code != 0 or validator.get('errors') != 0 or validator.get('rows') != 1200 or validator.get('ok') is not True:
            raise ValueError(f'Validator failed; see {out / "validator.log"}')
        counts = [len(row['appended']) for row in selections.values()]
        stats = {'diagnostic_only': True, 'LB': 'chưa đo', 'domain': args.domain,
                 'base': str(BASE), 'base_sha256': RD150_SHA256, **metadata,
                 'selection': 'stable domain filter of original Z-append <=100 primary IDs; no topup, ranking or alias expansion',
                 'gate': gate, 'validator': validator, 'json_bytes': tmp_json.stat().st_size,
                 'json_sha256': digest(tmp_json), 'zip_bytes': tmp_zip.stat().st_size,
                 'zip_sha256': digest(tmp_zip), 'zip_ceiling_bytes': ZIP_CEILING,
                 'appended_zh': _distribution(counts), 'appended_total': sum(counts),
                 'queries_with_domain': sum(n > 0 for n in counts), 'queries_without_domain': sum(n == 0 for n in counts),
                 'queries_below_100': sum(n < 100 for n in counts), 'build_seconds': build_seconds,
                 'validator_seconds': time.perf_counter() - validator_started,
                 'available_memory_before_validation_bytes': available, 'new_model_calls': 0, 'new_gpu_seconds': 0}
        _atomic_json(out / 'stats.json', {'complete': False, 'status': 'publishing validated staging artifacts'})
        _atomic_json(out / 'query_append.json', selections)
        os.replace(guard_output(tmp_json), guard_output(final_json))
        os.replace(guard_output(tmp_zip), guard_output(final_zip))
        _atomic_json(out / 'stats.json', {**stats, 'complete': True})
        print(json.dumps(stats, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
