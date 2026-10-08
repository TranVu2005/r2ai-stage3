"""Output boundary for the authorized selection run and zh-full state only."""
from r2ai.paths import RUNS_DIR, STATE_DIR, OLD_ROOT, resolve_path, assert_writable
import json
import os
import sqlite3

RUN = RUNS_DIR / 'zh-select-2026-10-08'
STATE = STATE_DIR / 'zh_full'
DB = STATE / 'crawl.db'


def guard(path):
    p = resolve_path(path)
    if not any(p.is_relative_to(root) for root in (RUN, STATE)):
        raise ValueError(f'Not a selection output: {p}')
    if OLD_ROOT is not None and p.is_relative_to(OLD_ROOT):
        raise ValueError(f'Write to OLD is forbidden: {p}')
    return p if p.is_dir() else assert_writable(p)


def preflight():
    for root in (RUN, STATE):
        assert_writable(guard(root))
        if root.exists():
            for child in root.rglob('*'):
                guard(child)
    for suffix in ('', '-wal', '-shm', '-journal'):
        guard(DB.with_name(DB.name + suffix))


def ro(path=DB, immutable=False):
    return sqlite3.connect(resolve_path(path).as_uri() + '?mode=ro' + ('&immutable=1' if immutable else ''), uri=True)


def writer(path):
    path = guard(path)
    for suffix in ('-wal','-shm','-journal'):
        guard(path.with_name(path.name+suffix))
    conn = sqlite3.connect(str(path), isolation_level=None)
    # Avoid unguarded SQLite sort files outside the authorized output boundary.
    conn.execute('PRAGMA temp_store=MEMORY')
    conn.execute('PRAGMA cache_size=-16384')
    return conn


def atomic_json(path, value):
    path = guard(path)
    tmp = guard(path.with_name(path.name + '.tmp'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
