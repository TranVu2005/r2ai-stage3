"""Atomic raw-HTML shards: `<domain>/<shard>.jsonl.zst`, one complete zstd frame per file."""
from __future__ import annotations

from r2ai.paths import ROOT

import itertools
import json
import os
import secrets
import time
from pathlib import Path

import zstandard

_seq = itertools.count()


def _replace(src: Path, dst: Path):
    # Windows: antivirus / indexers can briefly hold the fresh file.
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def write_shard(root: Path, domain: str, records: list[dict], level: int = 6) -> Path:
    d = Path(root) / domain.replace(':', '_')
    d.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}-{next(_seq):05d}-{secrets.token_hex(2)}.jsonl.zst"
    final, tmp = d / name, d / (name + '.tmp')
    payload = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in records).encode('utf-8')
    frame = zstandard.ZstdCompressor(level=level).compress(payload)  # single closed frame
    try:
        with open(tmp, 'wb') as f:
            f.write(frame)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, final)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return final


def read_shard(path: Path) -> list[dict]:
    data = zstandard.ZstdDecompressor().decompress(Path(path).read_bytes(), max_output_size=1 << 31)
    return [json.loads(line) for line in data.decode('utf-8').split('\n') if line]


def cleanup_tmp(root: Path) -> int:
    n = 0
    root = Path(root)
    if not root.exists():
        return 0
    for p in root.rglob('*.tmp'):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def list_shards(root: Path, domain: str | None = None) -> list[Path]:
    root = Path(root)
    base = root / domain.replace(':', '_') if domain else root
    return sorted(base.rglob('*.jsonl.zst')) if base.exists() else []
