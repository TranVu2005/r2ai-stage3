"""Filter a finished submission to corpus v2 and repack it as ZIP_LZMA (decision D7).

Drops every relevant_docs id and every relevant_chunks entry whose doc_id is not in v2; nothing else changes (row
order, field order, chunk text and the writer's layout stay byte-identical for untouched rows). `--drop-host H`
additionally removes every doc and chunk of host H (in v2 but unwanted), counted apart from the outside-v2 ones.

    python -m r2ai.submit.corpus_filter SRC.(zip|json) DST_v2.zip [--max-zip-bytes 98999999]
        [--lzma-dict-mib N] [--drop-host HOST ...]
"""
from __future__ import annotations

from r2ai.paths import assert_writable

import argparse
import hashlib
import io
import json
import lzma
import os
import struct
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from r2ai import corpus_v2 as C

DEFAULT_MAX_ZIP = 98_999_999   # "< 99 MB"


def _open_src(path: Path):
    if path.suffix.lower() != '.zip':
        return path.open('rb'), None
    z = zipfile.ZipFile(path)
    files = [i for i in z.infolist() if not i.is_dir()]
    if len(files) != 1 or len(z.infolist()) != 1:
        z.close()
        raise ValueError(f'{path}: ZIP must hold exactly one file')
    return z.open(files[0]), z


@contextmanager
def _lzma_dict(mib):
    """stdlib ZIP_LZMA fixes the 8 MiB default dictionary; optionally raise it (readers take it from the stream props)."""
    if mib is None:
        yield
        return

    class Comp(zipfile.LZMACompressor):
        def _init(self):
            f = {'id': lzma.FILTER_LZMA1, 'dict_size': mib << 20, 'preset': 9 | lzma.PRESET_EXTREME}
            props = lzma._encode_filter_properties(f)
            self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)])
            return struct.pack('<BBH', 9, 4, len(props)) + props

    orig = zipfile.LZMACompressor
    zipfile.LZMACompressor = Comp
    try:
        yield
    finally:
        zipfile.LZMACompressor = orig


def _dumps(row) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(', ', ': '))


def filter_submission(src, dst, ids_path=None, max_zip_bytes=DEFAULT_MAX_ZIP, lzma_dict_mib=None, drop_hosts=()) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.suffix.lower() != '.zip':
        raise ValueError('dst must end with .zip')
    dst = assert_writable(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = assert_writable(dst.with_name(dst.name + '.tmp'))
    inner = dst.stem + '.json'
    st = dict.fromkeys(('rows', 'rows_changed', 'rows_empty_docs', 'docs', 'docs_outside', 'docs_kept', 'chunks',
                        'chunks_outside', 'chunks_kept', 'docs_dropped_host', 'chunks_dropped_host', 'src_json_bytes', 'json_bytes'), 0)
    dropped = C.host_ids(drop_hosts, ids_path) if drop_hosts else None   # members that are removed on purpose
    out_docs: set[int] = set()
    out_chunk_docs: set[int] = set()
    sha = hashlib.sha256()
    fin, zin = _open_src(src)
    try:
        lines = io.TextIOWrapper(fin, encoding='utf-8', newline='\n')
        with _lzma_dict(lzma_dict_mib), zipfile.ZipFile(tmp, 'w', zipfile.ZIP_LZMA) as zout, zout.open(inner, 'w', force_zip64=True) as fout:
            def emit(text: str):
                b = text.encode('utf-8')
                sha.update(b)
                st['json_bytes'] += len(b)
                fout.write(b)

            first = lines.readline()
            if first != '[\n':
                raise ValueError('unexpected layout: first line must be "["')
            st['src_json_bytes'] += len(first)
            emit(first)
            pending = None   # one row of lookahead so the last row loses its trailing comma
            closed = False
            for line in lines:
                st['src_json_bytes'] += len(line.encode('utf-8'))
                if line == ']\n':
                    closed = True
                    continue
                if closed or not line.endswith('\n'):
                    raise ValueError('unexpected layout: rows must be one per line before "]"')
                if pending is not None:
                    emit(pending + ',\n')
                body = line[:-2] if line.endswith(',\n') else line[:-1]
                row = json.loads(body)
                docs, chunks = row['relevant_docs'], row['relevant_chunks']
                dm, cm = C.member_mask(docs, ids_path), C.member_mask([c['doc_id'] for c in chunks], ids_path)
                st['rows'] += 1
                st['docs'] += len(docs)
                st['chunks'] += len(chunks)
                out_docs.update(d for d, ok in zip(docs, dm) if not ok)
                out_chunk_docs.update(c['doc_id'] for c, ok in zip(chunks, cm) if not ok)
                dd = np.isin(np.asarray(docs, dtype=np.int64), dropped) if dropped is not None else np.zeros(len(docs), bool)
                cd = np.isin(np.asarray([c['doc_id'] for c in chunks], dtype=np.int64), dropped) if dropped is not None else np.zeros(len(chunks), bool)
                if dm.all() and cm.all() and not dd.any() and not cd.any():
                    pending = body
                    st['docs_kept'] += len(docs)
                    st['chunks_kept'] += len(chunks)
                    continue
                row['relevant_docs'] = [d for d, ok, x in zip(docs, dm, dd) if ok and not x]
                row['relevant_chunks'] = [c for c, ok, x in zip(chunks, cm, cd) if ok and not x]
                st['rows_changed'] += 1
                st['rows_empty_docs'] += not row['relevant_docs']
                st['docs_outside'] += int((~dm).sum())
                st['chunks_outside'] += int((~cm).sum())
                st['docs_dropped_host'] += int((dm & dd).sum())
                st['chunks_dropped_host'] += int((cm & cd).sum())
                st['docs_kept'] += len(row['relevant_docs'])
                st['chunks_kept'] += len(row['relevant_chunks'])
                pending = _dumps(row)
            if not closed:
                raise ValueError('unexpected layout: missing closing "]"')
            if pending is not None:
                emit(pending + '\n')
            emit(']\n')
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        fin.close()
        if zin is not None:
            zin.close()
    size = tmp.stat().st_size
    if size > max_zip_bytes:
        tmp.unlink()
        raise ValueError(f'ZIP is {size} bytes, over the limit {max_zip_bytes}')
    os.replace(tmp, dst)
    st.update(docs_distinct_outside=len(out_docs), chunks_distinct_docs_outside=len(out_chunk_docs),
              zip_bytes=size, json_sha256=sha.hexdigest())
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--max-zip-bytes', type=int, default=DEFAULT_MAX_ZIP)
    ap.add_argument('--drop-host', action='append', default=[], help='also drop every doc/chunk of this host (repeatable)')
    ap.add_argument('--lzma-dict-mib', type=int, default=None, help='raise the LZMA dictionary (stdlib default 8 MiB)')
    args = ap.parse_args(argv)
    print(json.dumps(filter_submission(args.src, args.dst, max_zip_bytes=args.max_zip_bytes, lzma_dict_mib=args.lzma_dict_mib, drop_hosts=args.drop_host), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
