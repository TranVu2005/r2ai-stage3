"""Row assembly and ZIP_LZMA writer of the C1 compose: base rows kept byte for byte, zh docs/chunks appended (S8 build_E)."""
from __future__ import annotations

import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path

from r2ai.paths import assert_writable, resolve_path

from .base_io import base_sha, baseline, lzma_dictionary, sha

DATE_TIME = (2026, 10, 14, 0, 0, 0)


class OverBudget(ValueError):
    def __init__(self, size, cap, json_sha, zip_sha):
        super().__init__(f'ZIP {size} > {cap}')
        self.size = size
        self.cap = cap
        self.json_sha = json_sha
        self.zip_sha = zip_sha


def trim_vi(row, k, membership):
    if type(k) != int or k < 0:
        raise ValueError('invalid vi K')
    kept = []
    removed = []
    n = 0
    for ch in row['relevant_chunks']:
        if ch['doc_id'] not in membership:
            raise ValueError('unknown chunk language')
        if membership[ch['doc_id']] == 'vi':
            n += 1
            if n > k:
                removed.append(ch)
                continue
        kept.append(ch)
    return {**row, 'relevant_docs': list(row['relevant_docs']), 'relevant_chunks': kept}, removed


def extend_many(base, selection, rows, texts):
    got = {**base, 'relevant_docs': base['relevant_docs'] + selection['ids'], 'relevant_chunks': list(base['relevant_chunks'])}
    protected = []
    chunks = []
    for r in rows:
        text = texts[(r['segment'], r['doc_id'])]
        if not isinstance(text, str) or not text:
            raise ValueError('missing whole doc_text')
        for d in r['doc_ids_v2']:
            if d not in got['relevant_docs']:
                got['relevant_docs'].append(d)
                protected.append(d)
            ch = {'doc_id': d, 'chunk_text': text}
            if ch not in got['relevant_chunks']:
                got['relevant_chunks'].append(ch)
                chunks.append(d)
    return got, {'protected_ids': protected, 'chunk_ids': chunks}


def shape(row, membership):
    if set(row) != {'id', 'relevant_docs', 'relevant_chunks'} or type(row['id']) != int:
        raise ValueError('bad row schema')
    ids = row['relevant_docs']
    if any(type(d) != int or d not in membership for d in ids) or len(ids) != len(set(ids)):
        raise ValueError('v2/duplicate doc ID')
    for ch in row['relevant_chunks']:
        if (set(ch) != {'doc_id', 'chunk_text'} or type(ch['doc_id']) != int or ch['doc_id'] not in ids
                or not isinstance(ch['chunk_text'], str) or not ch['chunk_text'].strip()):
            raise ValueError('invalid/protection/empty chunk')


def compression_profile(path):
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) != 1:
            raise ValueError('base ZIP must contain one JSON')
        i = z.infolist()[0]
        if i.compress_type != zipfile.ZIP_LZMA:
            raise ValueError('unsupported base compression; need Vu compression handoff')
        if i.flag_bits & 1:
            raise ValueError('encrypted base unsupported')
        with Path(path).open('rb') as f:
            f.seek(i.header_offset)
            header = f.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError('bad base ZIP local header')
            namesize, extrasize = struct.unpack_from('<HH', header, 26)
            f.seek(namesize + extrasize, 1)
            version_major, version_minor, props_size = struct.unpack('<BBH', f.read(4))
            props = f.read(props_size)
            if props_size != 5:
                raise ValueError('unexpected ZIP LZMA properties')
            dictionary = struct.unpack_from('<I', props, 1)[0]
            if dictionary % (1 << 20) or dictionary >> 20 not in (8, 16, 32, 64, 128, 256):
                raise ValueError('unsupported base LZMA dictionary')
    return {'method': 'ZIP_LZMA', 'dictionary_mib': dictionary >> 20, 'dictionary_bytes': dictionary, 'lc': props[0] % 9,
            'lp': (props[0] // 9) % 5, 'pb': props[0] // 45, 'zip_lzma_version': [version_major, version_minor],
            'properties_hex': props.hex()}


def render_rows(*, base, selections, chunk_rows, texts, membership, vi_k, expected_queries=1200):
    """Yield (bytes, audit) of the output JSON in order: '[' line, one entry per base row, ']' line (audit None for brackets)."""
    seen = set()
    yield b'[\n', None
    for i, (original, raw) in enumerate(baseline(base)):
        shape(original, membership)
        q = original['id']
        if q in seen or q not in selections or q not in chunk_rows:
            raise ValueError('query mismatch/duplicate')
        seen.add(q)
        trimmed, removed = trim_vi(original, vi_k, membership)
        new, extra = extend_many(trimmed, selections[q], chunk_rows[q], texts)
        shape(new, membership)
        if new['relevant_docs'][:len(original['relevant_docs'])] != original['relevant_docs']:
            raise ValueError('base doc order changed')
        for ch in new['relevant_chunks'][len(trimmed['relevant_chunks']):]:
            if membership[ch['doc_id']] != 'zh':
                raise ValueError('added non-zh chunk')
        payload = raw.strip().removesuffix(b',') if new == original else json.dumps(new, ensure_ascii=False).encode('utf-8')
        yield payload + (b',\n' if i < expected_queries - 1 else b'\n'), {
            'query_id': q, 'k': len(original['relevant_docs']), 'c': len(original['relevant_chunks']),
            'retained_c': len(trimmed['relevant_chunks']), 'removed_vi_ids': [ch['doc_id'] for ch in removed],
            'new_ids': len(selections[q]['ids']), 'selected_docs': len(selections[q]['docs']),
            'protected_ids': extra['protected_ids'], 'new_chunks': len(extra['chunk_ids']),
            'chunk_ids': extra['chunk_ids'], 'chunk_doc_count': len(chunk_rows[q])}
    if len(seen) != expected_queries or set(selections) != seen or set(chunk_rows) != seen:
        raise ValueError('incomplete query set')
    yield b']\n', None


def json_sha256(**kw):
    """SHA256 and audit of the JSON `write_archive` would write, without the ZIP (fast replay gate)."""
    h = hashlib.sha256()
    audit = []
    for payload, row_audit in render_rows(**kw):
        h.update(payload)
        if row_audit is not None:
            audit.append(row_audit)
    return h.hexdigest(), audit


def write_archive(*, base, dst, selections, chunk_rows, texts, membership, vi_k, ceiling, expected_queries=1200,
                  postflight=None, proof_path=None, metadata=None):
    base = resolve_path(base)
    dst = assert_writable(dst)
    tmp = assert_writable(Path(str(dst) + '.tmp'))
    assert_writable(dst.parent).mkdir(parents=True, exist_ok=True)
    if dst.exists() or tmp.exists() or dst == base or tmp == base:
        raise ValueError('output exists/overlaps input')
    proof = assert_writable(proof_path) if proof_path else None
    proof_tmp = assert_writable(Path(str(proof) + '.tmp')) if proof else None
    if proof and (proof.exists() or proof_tmp.exists() or proof in (base, dst, tmp) or proof_tmp in (base, dst, tmp)):
        raise ValueError('proof exists/overlaps input or ZIP')
    published = False
    h = hashlib.sha256()
    audit = []
    seen = set()
    start = time.monotonic()
    last = start
    try:
        with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
            info = zipfile.ZipInfo(dst.stem + '.json', date_time=DATE_TIME)
            info.compress_type = zipfile.ZIP_LZMA
            with z.open(info, 'w', force_zip64=True) as f:
                def write(b):
                    f.write(b)
                    h.update(b)

                for payload, row_audit in render_rows(base=base, selections=selections, chunk_rows=chunk_rows, texts=texts,
                                                      membership=membership, vi_k=vi_k, expected_queries=expected_queries):
                    write(payload)
                    if row_audit is None:
                        continue
                    audit.append(row_audit)
                    if time.monotonic() - last >= 30:
                        print(f'{dst.stem} K{vi_k}: {len(audit)}/{expected_queries} elapsed={time.monotonic() - start:.0f}s', flush=True)
                        last = time.monotonic()
        cp = compression_profile(tmp)
        if cp['dictionary_bytes'] != 256 << 20 or cp['properties_hex'] != '5d00000010':
            raise ValueError('dict256 header mismatch')
        size = tmp.stat().st_size
        zip_sha = sha(tmp)
        print('HEADER ' + dst.stem + ' ' + json.dumps(cp) + ' ZIP_BYTES ' + str(size), flush=True)
        if size > ceiling:
            raise OverBudget(size, ceiling, h.hexdigest(), zip_sha)
        if base_sha(tmp) != h.hexdigest():
            raise ValueError('JSON SHA readback mismatch')
        count = 0
        for (old, _), (new, _) in zip(baseline(base), baseline(tmp), strict=True):
            q = old['id']
            trimmed, _ = trim_vi(old, vi_k, membership)
            if (new['id'] != q or new['relevant_docs'][:len(old['relevant_docs'])] != old['relevant_docs']
                    or new['relevant_chunks'][:len(trimmed['relevant_chunks'])] != trimmed['relevant_chunks']):
                raise ValueError('readback base-prefix/order mismatch')
            expected, _ = extend_many(trimmed, selections[q], chunk_rows[q], texts)
            if new != expected:
                raise ValueError('readback full text/source mismatch')
            shape(new, membership)
            count += 1
        if count != expected_queries:
            raise ValueError('readback missing queries')
        result = {'zip_bytes': size, 'zip_sha256': zip_sha, 'json_sha256': h.hexdigest(), 'header': cp, 'rows': count, 'K': vi_k,
                  'new_ids': sum(r['new_ids'] for r in audit), 'new_doc_physical': sum(r['selected_docs'] for r in audit),
                  'new_chunks': sum(r['new_chunks'] for r in audit), 'removed_vi_chunks': sum(len(r['removed_vi_ids']) for r in audit),
                  'protected_ids_added': sum(len(r['protected_ids']) for r in audit), 'audit': audit}
        result.update(metadata or {})
        if postflight:
            postflight(result)
        if proof:
            proof_tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        tmp.replace(dst)
        published = True
        if proof:
            proof_tmp.replace(proof)
        return result
    except BaseException:
        if tmp.exists():
            assert_writable(tmp).unlink()
        if proof_tmp and proof_tmp.exists():
            assert_writable(proof_tmp).unlink()
        if published and dst.exists():
            assert_writable(dst).unlink()
        raise


def check_pins(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('input/source mutated: ' + path)


def require_native_success(meta, validation):
    if meta.get('native_validator_pending') is not False or meta.get('rows') != 1200 or not isinstance(validation, dict):
        raise ValueError('native validator pending/missing')
    if (validation.get('ok') is not True or validation.get('errors') != 0 or validation.get('rows') != 1200
            or validation.get('v2') is not True or validation.get('cap_bytes') != meta.get('ceiling_bytes')):
        raise ValueError('native validator failed/incomplete')
    if meta.get('native_validator') != validation:
        raise ValueError('native result differs from proof')
    for key in ('zip_sha256', 'json_sha256'):
        if not meta.get(key) or validation.get(key) != meta[key]:
            raise ValueError('native validator SHA mismatch')


def verify_ready_file(path, meta, validation):
    require_native_success(meta, validation)
    if sha(path) != meta['zip_sha256'] or base_sha(path) != meta['json_sha256']:
        raise ValueError('current ZIP/JSON SHA mismatch')
    if Path(path).stat().st_size > meta['ceiling_bytes']:
        raise ValueError('current ZIP overcap')
