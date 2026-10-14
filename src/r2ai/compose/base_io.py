"""Streaming helpers for the one-row-per-line submission JSON/ZIP and the ZIP_LZMA dictionary writer (S8 generate_probe)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import lzma
import struct
import zipfile
from pathlib import Path

import pyarrow as pa

CANDIDATE_SCHEMA = {'query_id': 'int64', 'rank': 'int32', 'doc_id': 'int64', 'logit': 'double',
                    'domain': 'string', 'segment': 'string', 'chunk_id_v2': 'int64',
                    'doc_ids_group': 'list', 'doc_ids_v2': 'list'}


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def base_stream(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != 1 or not names[0].endswith('.json'):
                raise ValueError('base must contain one JSON')
            with z.open(names[0]) as f:
                yield f
    else:
        with Path(path).open('rb') as f:
            yield f


def baseline(path):
    """One JSON object per line; retain raw bytes for unchanged query rows."""
    with base_stream(path) as f:
        for line in f:
            s = line.strip()
            if s in (b'[', b']', b''):
                continue
            if s.endswith(b','):
                s = s[:-1]
            row = json.loads(s)
            if not isinstance(row, dict):
                raise ValueError('base requires one query object per line')
            yield row, line


def base_sha(path) -> str:
    h = hashlib.sha256()
    with base_stream(path) as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def lzma_dictionary(mib, *, encoder_preset=None):
    previous = zipfile.LZMACompressor

    class Compressor:
        def __init__(self):
            self.comp = None

        def compress(self, data):
            header = b''
            if self.comp is None:
                flt = {'id': lzma.FILTER_LZMA1, 'dict_size': mib << 20, 'preset': 9 | lzma.PRESET_EXTREME}
                if encoder_preset is not None:
                    flt = {'id': lzma.FILTER_LZMA1, 'preset': encoder_preset, 'dict_size': mib << 20, 'lc': 3, 'lp': 0, 'pb': 2}
                props = lzma._encode_filter_properties(flt)
                encoder_filter = flt if encoder_preset is not None else lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)
                self.comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[encoder_filter])
                header = struct.pack('<BBH', 9, 4, len(props)) + props
            return header + self.comp.compress(data)

        def flush(self):
            if self.comp is None:
                self.compress(b'')
            return self.comp.flush()

    zipfile.LZMACompressor = Compressor
    try:
        yield
    finally:
        zipfile.LZMACompressor = previous


def check_schema(table) -> None:
    for name, kind in CANDIDATE_SCHEMA.items():
        if name not in table.column_names:
            raise ValueError('missing candidate column ' + name)
        if table[name].null_count:
            raise ValueError('null candidate column ' + name)
        t = table.schema.field(name).type
        valid = str(t) == kind
        if kind == 'string':
            valid = pa.types.is_string(t) or pa.types.is_large_string(t)
        if kind == 'list':
            valid = (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_int64(t.value_type)
        if not valid:
            raise ValueError('wrong candidate schema ' + name + ': ' + str(t))


def id_list(x, label):
    if not isinstance(x, list) or not x or any(type(v) != int for v in x) or len(x) != len(set(x)):
        raise ValueError('invalid ' + label)
    return x
