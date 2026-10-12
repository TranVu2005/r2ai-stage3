"""Roots and write guard for the Common Crawl longchau shard (D1: text of corpus URLs only, with provenance).

Every write goes through `Roots.guard`: resolved paths must stay inside the owned roots and never inside OLD.
Inputs (links_corpus.parquet, cc-coverage/coverage.parquet) are read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from r2ai.paths import CORPUS_FILE, OLD_ROOT, RUNS_DIR, STATE_DIR, WORK_DATA_DIR, resolve_path

DOMAIN = 'nhathuoclongchau.com.vn'
CC_BASE = 'https://data.commoncrawl.org/'
UA = 'R2AI-Stage3-CCFetch/1.0 (research; HTTP Range GET of WARC records for URLs in the task corpus; <=4 req/s)'
COVERAGE = RUNS_DIR / 'vi-longchau' / 'cc-coverage' / 'coverage.parquet'
SEGMENT_ID = 20000                      # global chunk ids 2**48 + 20000 * 2**32 + row: no clash with vi (< 2**32) or zh segments 1..n


def _key(path) -> str:
    return os.path.normcase(str(resolve_path(path)))


def _under(path, root) -> bool:
    p, r = _key(path), _key(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


@dataclass(frozen=True)
class Roots:
    raw: Path = WORK_DATA_DIR / 'raw_cc_longchau'
    docs: Path = WORK_DATA_DIR / 'docs_cc_longchau'
    chunks: Path = WORK_DATA_DIR / 'chunks_cc_longchau'
    index: Path = WORK_DATA_DIR / 'index' / 'zh_full_segments' / 'cc_longchau'   # segments.guarded_segment_output only allows this tree
    run: Path = RUNS_DIR / 'cc-longchau'
    state: Path = STATE_DIR / 'cc_longchau'
    corpus: Path = CORPUS_FILE
    coverage: Path = COVERAGE

    @property
    def owned(self):
        return (self.raw, self.docs, self.chunks, self.index, self.run, self.state)

    @property
    def fetch_db(self) -> Path:
        return self.state / 'fetch.db'

    def guard(self, path) -> Path:
        p = resolve_path(path)
        if OLD_ROOT is not None and _under(p, OLD_ROOT):
            raise ValueError(f'Write to OLD is forbidden: {p}')
        if not any(_under(p, root) for root in self.owned):
            raise ValueError(f'Not a cc-longchau output: {p}')
        return p

    def preflight(self):
        for root in self.owned:
            self.guard(root)
            if root.exists():
                for child in root.rglob('*'):
                    self.guard(child)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(roots: Roots, path, value):
    path = roots.guard(path)
    tmp = roots.guard(path.with_name(path.name + '.partial'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=1).encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_event(roots: Roots, name: str, value: dict):
    path = roots.guard(roots.run / 'logs' / f'{name}.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'at': time.time(), **value}, ensure_ascii=False) + '\n')
