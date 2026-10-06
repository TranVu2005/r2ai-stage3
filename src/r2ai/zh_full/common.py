from __future__ import annotations

from r2ai.paths import RUNS_DIR, WORK_DATA_DIR, STATE_DIR, OLD_ROOT, resolve_path, assert_writable
from pathlib import Path
from contextlib import contextmanager
import json
import os

RUN = RUNS_DIR / 'zh-full'
RAW = WORK_DATA_DIR / 'raw_zh_full'
STATE = STATE_DIR / 'zh_full'
DB = STATE / 'crawl.db'
OWNED = (RUN, RAW, STATE)


def guard(path):
    resolved = resolve_path(path)
    if not any(resolved.is_relative_to(root) for root in OWNED):
        raise ValueError(f'Not a private full-zh output: {resolved}')
    if OLD_ROOT is not None and resolved.is_relative_to(OLD_ROOT):
        raise ValueError(f'Write to OLD is forbidden: {resolved}')
    # A directory check here must stay O(1): preflight owns recursive scanning.
    return resolved if resolved.is_dir() else assert_writable(path)


def atomic_json(path, value):
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.tmp'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def preflight():
    for root in OWNED:
        assert_writable(guard(root))
        # Scan redirects once per CLI, never once per shard.
        if root.exists():
            for child in root.rglob('*'):
                guard(child)
    for suffix in ('', '-wal', '-shm', '-journal'):
        guard(DB.with_name(DB.name + suffix))


@contextmanager
def exclusive(name):
    import msvcrt
    path = guard(RUN / f'{name}.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(f'Full-zh {name} already running') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
