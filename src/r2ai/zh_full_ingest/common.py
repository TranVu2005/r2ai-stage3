"""Roots, write guards, locks and resource guards for zh-full ingest.

Inputs (crawl.db, raw shards, the zh sample bundle) are read-only. Every write goes
through `guard`, which compares fully resolved paths (junctions/symlinks followed,
case-insensitive on Windows) against the owned roots and the forbidden roots.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from r2ai.paths import OLD_ROOT, RUNS_DIR, STATE_DIR, WORK_DATA_DIR, resolve_path

GLOBAL_ID_BASE = 1 << 48           # zh chunk ids never collide with vi ids (0..3.909.585)
SEGMENT_STRIDE = 1 << 32
MIN_AVAILABLE_GIB = 4.0


def global_start(segment_id: int) -> int:
    if type(segment_id) is not int or not 1 <= segment_id < (1 << 15):
        raise ValueError('segment_id must be an int in 1..32767')
    return GLOBAL_ID_BASE + segment_id * SEGMENT_STRIDE


def split_global_id(gid: int) -> tuple[int, int]:
    gid = int(gid)
    if gid < GLOBAL_ID_BASE + SEGMENT_STRIDE:
        raise ValueError(f'not a zh segment chunk id: {gid}')
    seg, local = divmod(gid - GLOBAL_ID_BASE, SEGMENT_STRIDE)
    return seg, local


def _key(path: Path) -> str:
    return os.path.normcase(str(resolve_path(path)))


def _under(path: Path, root: Path) -> bool:
    p, r = _key(path), _key(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


@dataclass(frozen=True)
class Roots:
    raw: Path = WORK_DATA_DIR / 'raw_zh_full'
    crawl_db: Path = STATE_DIR / 'zh_full' / 'crawl.db'
    sample_raw: Path = WORK_DATA_DIR / 'raw_zh_sample'
    sample_db: Path = STATE_DIR / 'crawl_zh_sample.db'
    sample_docs: Path = WORK_DATA_DIR / 'docs_zh_sample' / 'bundle'
    sample_chunks: Path = WORK_DATA_DIR / 'chunks_zh_sample'
    sample_index: Path = WORK_DATA_DIR / 'index' / 'zh_sample_t256'
    docs: Path = WORK_DATA_DIR / 'docs_zh_full'
    chunks: Path = WORK_DATA_DIR / 'chunks_zh_full'
    index: Path = WORK_DATA_DIR / 'index' / 'zh_full_segments'
    run: Path = RUNS_DIR / 'zh-full-ingest'
    state: Path = STATE_DIR / 'zh_full_ingest'
    forbidden: tuple = field(default_factory=lambda: (
        WORK_DATA_DIR / 'raw_vi', WORK_DATA_DIR / 'docs_vi', WORK_DATA_DIR / 'chunks', WORK_DATA_DIR / 'index' / 't256',
        WORK_DATA_DIR / 'raw_zh_full', WORK_DATA_DIR / 'raw_zh_sample', WORK_DATA_DIR / 'docs_zh_sample',
        WORK_DATA_DIR / 'chunks_zh_sample', WORK_DATA_DIR / 'index' / 'zh_sample_t256',
        STATE_DIR / 'zh_full', STATE_DIR / 'crawl.db', STATE_DIR / 'crawl_zh_sample.db', STATE_DIR / 'extract.db'))

    @property
    def owned(self):
        return (self.docs, self.chunks, self.index, self.run, self.state)

    @property
    def manifest(self) -> Path:
        return self.state / 'manifest.db'

    def guard(self, path) -> Path:
        """Resolve and refuse any write outside the owned roots or inside an input/vi/OLD root."""
        p = resolve_path(path)
        if OLD_ROOT is not None and _under(p, OLD_ROOT):
            raise ValueError(f'Write to OLD is forbidden: {p}')
        for root in self.forbidden:
            if _under(p, root):
                raise ValueError(f'Write to a protected input is forbidden: {p}')
        if not any(_under(p, root) for root in self.owned):
            raise ValueError(f'Not a zh-full-ingest output: {p}')
        return p

    def preflight(self):
        """Once per CLI: owned roots and every existing child must resolve inside the owned roots."""
        for root in self.owned:
            self.guard(root)
            if root.exists():
                for child in root.rglob('*'):
                    self.guard(child)


def sha256_file(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_bytes(roots: Roots, path, data: bytes):
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(roots: Roots, path, value):
    atomic_bytes(roots, path, json.dumps(value, ensure_ascii=False, indent=1).encode('utf-8'))


def log_event(roots: Roots, name: str, value: dict):
    path = roots.guard(roots.run / 'logs' / f'{name}.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'at': time.time(), **value}, ensure_ascii=False) + '\n')


@contextmanager
def exclusive(roots: Roots, name: str):
    """Windows byte lock: a second runner of the same step fails fast; a crash releases it."""
    import msvcrt
    path = roots.guard(roots.run / f'{name}.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(f'zh-full-ingest {name} already running') from e
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class MemoryGuard:
    """Wait while available RAM < minimum; too many pauses in a window stops the run for a human decision."""

    def __init__(self, roots: Roots | None = None, minimum_gib=MIN_AVAILABLE_GIB, wait_s=60, max_pauses=3,
                 window_s=1800, available=None, sleep=time.sleep, clock=time.time):
        import psutil
        self.roots, self.minimum, self.wait_s = roots, minimum_gib, wait_s
        self.max_pauses, self.window_s = max_pauses, window_s
        self.available = available or (lambda: psutil.virtual_memory().available / 2**30)
        self.sleep, self.clock, self.pauses = sleep, clock, []

    def check(self, where=''):
        while True:
            gib = self.available()
            if gib >= self.minimum:
                return gib
            now = self.clock()
            self.pauses = [t for t in self.pauses if now - t < self.window_s] + [now]
            if self.roots is not None:
                log_event(self.roots, 'memory', {'available_gib': gib, 'where': where, 'pauses_in_window': len(self.pauses)})
            if len(self.pauses) > self.max_pauses:
                raise RuntimeError(f'MEMORY_PAUSES: {len(self.pauses)} pauses in {self.window_s}s (available {gib:.2f} GiB); ask the user')
            self.sleep(self.wait_s)


def gpu_compute_processes(snapshot: str, own_pid: int):
    from r2ai.zh_sample.index import compute_processes
    return compute_processes(snapshot, own_pid)


def require_gpu_idle(roots: Roots):
    """Refuse the GPU when another CUDA compute process is present (WDDM desktop C+G excluded)."""
    import subprocess
    text = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
    if 'Processes:' not in text:
        raise RuntimeError('Cannot parse nvidia-smi process table; refusing GPU')
    active = gpu_compute_processes(text, os.getpid())
    if active:
        raise RuntimeError('GPU_BUSY: ' + json.dumps(active))
    return text
