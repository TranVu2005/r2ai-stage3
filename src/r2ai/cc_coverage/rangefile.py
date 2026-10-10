"""Read-only file object over HTTP Range requests (through the metered Client) for pyarrow.PythonFile / ParquetFile."""
from __future__ import annotations

import io

from .net import Client, InvalidResponse, content_range_total

TAIL = 64 * 1024
MIN_BLOCK = 256 * 1024
KEEP_BLOCKS = 12


class RangeFile(io.RawIOBase):
    def __init__(self, client: Client, url: str, *, tail: int = TAIL, min_block: int = MIN_BLOCK):
        super().__init__()
        self.client, self.url, self.min_block = client, url, min_block
        r = client.get(url, byte_range=(-tail, None))
        if r.status_code == 404:
            raise FileNotFoundError(url)
        total = content_range_total(r)
        if total is None:
            if r.status_code != 200:
                raise InvalidResponse(f'{url}: no Content-Range on a {r.status_code} answer')
            total = len(r.content)
        self.size, self.pos = total, 0
        self.blocks: list[tuple[int, bytes]] = [(total - len(r.content), r.content)]
        self.tail_start = total - len(r.content)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _find(self, pos: int):
        for start, data in self.blocks:
            if start <= pos < start + len(data):
                return start, data
        return None

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = []
        while n > 0:
            hit = self._find(self.pos)
            if hit is None:
                want = min(self.size - self.pos, max(n, self.min_block))
                r = self.client.get(self.url, byte_range=(self.pos, self.pos + want - 1), planned_bytes=want)
                if r.status_code not in (200, 206) or not r.content:
                    raise InvalidResponse(f'{self.url}: empty / bad answer for range at {self.pos}')
                self.blocks.append((self.pos, r.content))
                if len(self.blocks) > KEEP_BLOCKS:                              # keep the tail block (index 0), drop the oldest other
                    del self.blocks[1]
                continue
            start, data = hit
            off = self.pos - start
            take = min(n, len(data) - off)
            out.append(data[off:off + take])
            self.pos += take
            n -= take
        return b''.join(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
