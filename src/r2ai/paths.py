"""Central read/write roots. CLI > process env > ROOT/.env > defaults.

Import before ML/HF libraries; importing this module never creates files.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'{path.name}:{number}: invalid environment assignment')
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'{path.name}:{number}: unmatched quote')
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv(ROOT / '.env')


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def _root(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is not None and not value.strip():
        raise ValueError(f'{name}: empty root is not allowed')
    return resolve_path(default if value is None else value)


DATA_DIR = _root('R2AI_DATA_DIR', ROOT / 'data')
WORK_DATA_DIR = _root('R2AI_WORK_DATA_DIR', ROOT / 'data')
if not DATA_DIR.is_relative_to(ROOT) and 'R2AI_OLD_ROOT' not in os.environ:
    raise ValueError('R2AI_OLD_ROOT is required when R2AI_DATA_DIR is outside this repository')
OLD_ROOT = _root('R2AI_OLD_ROOT', ROOT) if 'R2AI_OLD_ROOT' in os.environ else None
RAW_DATA_DIR = DATA_DIR / 'raw'
RAW_DOWNLOAD_DIR = WORK_DATA_DIR / 'raw'
RAW_DIR = _root('R2AI_RAW_DIR', DATA_DIR / 'raw_vi')
RAW_WRITE_DIR = WORK_DATA_DIR / 'raw_vi'
DOCS_DIR = DATA_DIR / 'docs_vi'
DOCS_WRITE_DIR = WORK_DATA_DIR / 'docs_vi'
LEGACY_CHUNKS_DIR, CHUNKS_DIR = DATA_DIR / 'chunks', WORK_DATA_DIR / 'chunks'
LEGACY_INDEX_DIR, INDEX_DIR = DATA_DIR / 'index', WORK_DATA_DIR / 'index'
LEGACY_DEV_DIR, DEV_DIR = DATA_DIR / 'dev', WORK_DATA_DIR / 'dev'
LEGACY_PROFILE_DIR, PROFILE_DIR = DATA_DIR / 'profile', WORK_DATA_DIR / 'profile'
OUT_DIR = _root('R2AI_OUT_DIR', ROOT / 'out')
LEGACY_OUT_DIR = _root('R2AI_LEGACY_OUT_DIR', OUT_DIR)
STATE_DIR = _root('R2AI_STATE_DIR', ROOT / 'state')
LEGACY_STATE_DIR = _root('R2AI_LEGACY_STATE_DIR', STATE_DIR)
LOG_DIR = _root('R2AI_LOG_DIR', ROOT / 'logs')
RUNS_DIR, LEGACY_RUNS_DIR = OUT_DIR / 'runs', DATA_DIR / 'runs'
CONFIG_DIR = ROOT / 'configs'
FIXTURES_DIR = ROOT / 'tests/fixtures'
QUERY_FILE, CORPUS_FILE = RAW_DATA_DIR / 'query.parquet', RAW_DATA_DIR / 'links_corpus.parquet'
SUBMISSIONS_LOG = ROOT / 'submissions/LOG.md'


def assert_writable(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if OLD_ROOT is not None and resolved.is_relative_to(OLD_ROOT):
        raise ValueError(f'Write to OLD is forbidden: {resolved}')
    return resolved


def require_inputs(*paths: str | Path) -> None:
    for value in paths:
        path = resolve_path(value)
        if not path.exists() or (path.is_file() and path.stat().st_size == 0) or (path.is_dir() and not any(path.iterdir())):
            raise ValueError(f'Missing or empty input: {path}')


def index_dir(target: int, *, base: Path | None = None) -> Path:
    return (INDEX_DIR if base is None else resolve_path(base)) / f't{target}'


def chunks_file(target: int, *, base: Path | None = None) -> Path:
    return (CHUNKS_DIR if base is None else resolve_path(base)) / f'chunks_t{target}.parquet'


def run_dir(run_id: str) -> Path:
    path = resolve_path(RUNS_DIR / run_id)
    if not path.is_relative_to(RUNS_DIR):
        raise ValueError('run_id must stay below RUNS_DIR')
    return path


def resolve_legacy_artifact(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == 'out':
        return LEGACY_OUT_DIR.joinpath(*p.parts[1:])
    if p.parts and p.parts[0] == 'data':
        return DATA_DIR.joinpath(*p.parts[1:])
    return ROOT / p


def data_label(path: Path) -> str:
    p = resolve_path(path)
    for root in (WORK_DATA_DIR, DATA_DIR):
        if p.is_relative_to(root):
            return 'data/' + p.relative_to(root).as_posix()
    return str(p)


def cli_script(name: str) -> Path:
    return ROOT / 'scripts' / name


def auxiliary_disabled() -> None:
    raise SystemExit('This auxiliary writer is disabled until its output guards are migrated. Core retrieval/submission remain available.')
