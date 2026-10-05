from __future__ import annotations

from r2ai.paths import ROOT, WORK_DATA_DIR, RUNS_DIR, STATE_DIR, assert_writable, resolve_path
import json
import os
from contextlib import contextmanager
from pathlib import Path

RUN = RUNS_DIR / 'zh-sample'
RAW = WORK_DATA_DIR / 'raw_zh_sample'
DOCS = WORK_DATA_DIR / 'docs_zh_sample'
CHUNKS = WORK_DATA_DIR / 'chunks_zh_sample'
INDEX = WORK_DATA_DIR / 'index/zh_sample_t256'
DB = STATE_DIR / 'crawl_zh_sample.db'
EXTRACT_STATE = STATE_DIR / 'zh_sample_extract'
OWNED = (RUN, RAW, DOCS, CHUNKS, INDEX, EXTRACT_STATE)


def guard_owned(path: str | Path) -> Path:
    p = resolve_path(path)
    # Compare to lexical owned roots: resolving an owned-root junction first would bless a redirect to vi.
    forbidden = (ROOT / 'data/raw_vi', ROOT / 'data/docs_vi', ROOT / 'data/chunks', ROOT / 'data/index/t256')
    if any(p.is_relative_to(r) for r in forbidden):
        raise ValueError(f'Vietnamese output is forbidden: {p}')
    if not (any(p.is_relative_to(r) for r in OWNED)
            or p == DB or (p.parent == DB.parent and p.name in (DB.name + '-wal', DB.name + '-shm'))):
        raise ValueError(f'Not a zh-sample output: {p}')
    return assert_writable(path)


def preflight():
    for p in (*OWNED, DB):
        guard_owned(p)
    # The central guard protects OLD; additionally reject redirects into vi or another agent's outputs.
    pending = [p for p in OWNED if p.is_dir()]
    seen = set()
    while pending:
        directory = pending.pop()
        resolved = resolve_path(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        for child in directory.iterdir():
            guard_owned(child)
            if child.is_dir():
                pending.append(child)


def atomic_json(path: Path, value):
    path = guard_owned(path)
    tmp = guard_owned(path.with_name(path.name + '.tmp'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def exclusive(name: str):
    """OS lock releases on Ctrl+C/crash; stale file does not prevent resume."""
    import msvcrt
    path = guard_owned(RUN / f'{name}.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        try:
            # A competing Windows byte lock forbids even reading byte zero.
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(b'0')
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(f'{name} already running') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
