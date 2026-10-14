"""Parse one Common Crawl WARC response record fetched by HTTP Range (a single gzip member)."""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

_CHUNK = re.compile(rb'^([0-9a-fA-F]+)[^\r\n]*\r\n')


class WarcError(Exception):
    def __init__(self, reason: str, detail: str = ''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason


@dataclass
class WarcResponse:
    warc: dict = field(default_factory=dict)
    status: int = 0
    http: dict = field(default_factory=dict)
    body: bytes = b''

    @property
    def target_uri(self) -> str:
        return self.warc.get('warc-target-uri', '')

    @property
    def truncated(self) -> str:
        return self.warc.get('warc-truncated', '')


def gunzip_member(data: bytes) -> bytes:
    d = zlib.decompressobj(wbits=31)
    try:
        out = d.decompress(data) + d.flush()
    except zlib.error as e:
        raise WarcError('gzip_error', str(e)) from e
    if not d.eof:
        raise WarcError('gzip_truncated', f'{len(data)} compressed bytes end inside the member')
    if d.unused_data:
        raise WarcError('gzip_extra_member', f'{len(d.unused_data)} bytes after the first member')
    return out


def _headers(block: bytes) -> tuple[str, dict]:
    first, _, rest = block.partition(b'\r\n')
    out: dict = {}
    for line in rest.split(b'\r\n'):
        if b':' in line:
            k, _, v = line.partition(b':')
            out.setdefault(k.strip().lower().decode('latin-1'), v.strip().decode('latin-1'))
    return first.decode('latin-1'), out


def dechunk(body: bytes) -> bytes:
    out, rest = [], body
    while rest:
        m = _CHUNK.match(rest)
        if not m:
            raise WarcError('bad_chunked')
        size = int(m.group(1), 16)
        rest = rest[m.end():]
        if size == 0:
            return b''.join(out)
        if len(rest) < size:
            raise WarcError('bad_chunked', 'short chunk')
        out.append(rest[:size])
        rest = rest[size + 2:]
    raise WarcError('bad_chunked', 'no terminating chunk')


def decode_content(body: bytes, encoding: str) -> bytes:
    enc = (encoding or '').lower().strip()
    if enc in ('', 'identity'):
        return body
    if enc in ('gzip', 'x-gzip'):
        return zlib.decompress(body, 31)
    if enc == 'deflate':
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -15)
    if enc == 'br':
        import brotli
        return brotli.decompress(body)
    raise WarcError('unsupported_content_encoding', enc)


def parse_response(record: bytes) -> WarcResponse:
    """`record` = the decompressed record: WARC headers, blank line, HTTP response (status line, headers, body)."""
    head, sep, rest = record.partition(b'\r\n\r\n')
    if not sep or not head.startswith(b'WARC/'):
        raise WarcError('not_warc')
    _, warc = _headers(head)
    if warc.get('warc-type', '').lower() != 'response':
        raise WarcError('not_response', warc.get('warc-type', ''))
    n = warc.get('content-length')
    if n and n.isdigit():
        if len(rest) < int(n):
            raise WarcError('short_block', f'{len(rest)} < {n}')
        block = rest[:int(n)]
    else:
        block = rest
    hhead, sep, body = block.partition(b'\r\n\r\n')
    if not sep:
        raise WarcError('no_http_headers')
    status_line, http = _headers(hhead)
    m = re.match(r'HTTP/\S+\s+(\d{3})', status_line)
    if not m:
        raise WarcError('bad_status_line', status_line[:60])
    try:
        if 'chunked' in http.get('transfer-encoding', '').lower():
            body = dechunk(body)
        body = decode_content(body, http.get('content-encoding', ''))
    except WarcError:
        raise
    except Exception as e:  # noqa: BLE001
        raise WarcError('body_decode_error', repr(e)[:100]) from e
    return WarcResponse(warc=warc, status=int(m.group(1)), http=http, body=body)
