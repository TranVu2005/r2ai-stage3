"""Submission ZIP writer with a selectable compression method.

    write_submission_zip(json_path, zip_path, method='deflate' | 'lzma')

`deflate` (default) is what every builder writes (level 9, member mtime = file mtime). `lzma` uses the zipfile ZIP_LZMA default
preset (Python reads it back with zipfile; Info-ZIP 6.00 cannot). One top-level member, no sub-directory. Nothing here is
called by an existing code path: callers opt in with a `--zip-method` flag.
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

METHODS = ('deflate', 'lzma')


def _compression(method: str):
    if method == 'deflate':
        return zipfile.ZIP_DEFLATED, {'compresslevel': 9}
    if method == 'lzma':
        return zipfile.ZIP_LZMA, {}
    raise ValueError(f'unknown zip method {method!r}; choose from {METHODS}')


def write_submission_zip(json_path, zip_path, *, arcname: str | None = None, method: str = 'deflate', date_time=None,
                         lzma_dict_mib: int | None = None) -> Path:
    """json_path -> zip_path with one member `arcname` (default: the JSON file name); returns zip_path.

    lzma_dict_mib (lzma only): raise the LZMA dictionary above the stdlib 8 MiB default (preset 9 extreme, as r2ai.submit.corpus_filter);
    the decoder takes the size from the stream, but needs about that much RAM."""
    if lzma_dict_mib is not None:
        if method != 'lzma':
            raise ValueError('lzma_dict_mib needs method lzma')
        from r2ai.submit.corpus_filter import _lzma_dict
        with _lzma_dict(lzma_dict_mib):
            return write_submission_zip(json_path, zip_path, arcname=arcname, method=method, date_time=date_time)
    json_path, zip_path = Path(json_path), Path(zip_path)
    compression, kw = _compression(method)
    name = arcname or json_path.name
    if '/' in name or '\\' in name:
        raise ValueError(f'arcname must be a top-level file name: {name!r}')
    if date_time is None:
        with zipfile.ZipFile(zip_path, 'w', compression, **kw) as z:
            z.write(json_path, name)
        return zip_path
    info = zipfile.ZipInfo.from_file(json_path, name)
    info.date_time = tuple(date_time)
    info.compress_type = compression
    with zipfile.ZipFile(zip_path, 'w', compression, **kw) as z, open(json_path, 'rb') as src, z.open(info, 'w') as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    return zip_path


def _sha256_stream(f) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
    return h.hexdigest()


def verify_submission_zip(zip_path, json_sha256: str, *, expected_member: str | None = None) -> dict:
    """Open with Python zipfile, stream the single member and compare its SHA256 with `json_sha256` (CRC is checked on close)."""
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if len(infos) != 1:
            raise ValueError(f'{zip_path}: expected exactly one member, found {len(infos)}')
        info = infos[0]
        if '/' in info.filename or '\\' in info.filename or not info.filename.endswith('.json'):
            raise ValueError(f'{zip_path}: member must be a top-level .json, got {info.filename!r}')
        if expected_member is not None and info.filename != expected_member:
            raise ValueError(f'{zip_path}: member {info.filename!r} != {expected_member!r}')
        with z.open(info) as f:
            digest = _sha256_stream(f)
        method = {zipfile.ZIP_DEFLATED: 'deflate', zipfile.ZIP_LZMA: 'lzma'}.get(info.compress_type, str(info.compress_type))
        return {'member': info.filename, 'method': method, 'member_bytes': info.file_size, 'compressed_bytes': info.compress_size,
                'member_sha256': digest, 'json_sha256_match': digest == json_sha256}
