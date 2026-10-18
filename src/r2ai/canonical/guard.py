"""Output guards for S25: writes stay below the S25 roots, never overwrite, never touch OLD."""
from __future__ import annotations

from r2ai.paths import OUT_DIR, WORK_DATA_DIR, assert_writable, resolve_path

import os
from pathlib import Path

CANONICAL_ROOT = WORK_DATA_DIR / 'canonical-20261018'
RUN_ROOT = OUT_DIR / 'runs' / 's25-canonical'
ALLOWED_ROOTS = (CANONICAL_ROOT, RUN_ROOT)


def guarded(path: str | Path, *, roots=ALLOWED_ROOTS, allow_existing: bool = False) -> Path:
    """Resolve a destination; refuse anything outside the S25 roots, inside OLD, or already present."""
    p = resolve_path(path)
    if not any(p.is_relative_to(resolve_path(r)) for r in roots):
        raise ValueError(f'S25 output must stay below {[str(r) for r in roots]}: {p}')
    assert_writable(p.parent if p.parent.exists() else p)
    if p.exists() and not allow_existing:
        raise FileExistsError(f'S25 never overwrites: {p}')
    return p


def atomic_write_bytes(path: str | Path, data: bytes, **kw) -> Path:
    p = guarded(path, **kw)
    tmp = guarded(p.with_name(p.name + '.partial'))
    p.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def write_parquet(table, path: str | Path, **kw) -> Path:
    import pyarrow.parquet as pq

    p = guarded(path, **kw)
    tmp = guarded(p.with_name(p.name + '.partial'))
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, p)
    return p
