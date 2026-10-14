"""Pin manifest of a compose build (configs/best-c1.yaml): every input with its path and SHA256; a missing pin stops the build.

Paths may start with `{OUT_DIR}`, `{DATA_DIR}`, `{WORK_DATA_DIR}` (r2ai.paths roots). A file pin is `{path, sha256}`; a
directory pin is `{path, tree_sha256, files}` where tree_sha256 = SHA256 of the sorted lines `relpath<TAB>size<TAB>file sha256`.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from r2ai import paths
from r2ai.paths import resolve_path

from .base_io import sha

SHA = re.compile(r'[0-9a-f]{64}')
# inputs without which no C1 build may start (compose layer) and the S6-f replay layer
COMPOSE_PINS = ('base_stats', 'base_json', 'base_zip', 'candidate', 'zh_manifest', 'corpus_v2_ids', 'query')
NATIVE_PINS = ('vi_docs_dir',)
S6F_PINS = ('cc_a_cut115_json', 'pool_b_cache')
POLICY = {'tau_doc': -0.87255859375, 'doc_cap': 30, 'candidate_batch': 2, 'chunk_tau': 0.96, 'chunk_doc_cap': 3,
          'dict_mib': 256, 'ceiling_bytes': 102_600_000}


class PinError(ValueError):
    """A required pin is missing, malformed or does not match the file on disk."""


def expand(value) -> Path:
    text = str(value)
    for key in ('OUT_DIR', 'DATA_DIR', 'WORK_DATA_DIR'):
        text = text.replace('{' + key + '}', str(getattr(paths, key)))
    if '{' in text:
        raise PinError(f'unknown path token in {value!r}')
    return resolve_path(text)


def tree_sha256(directory, *, progress=None) -> tuple[str, int]:
    root = Path(directory)
    lines = []
    files = sorted(p for p in root.rglob('*') if p.is_file())
    for i, p in enumerate(files):
        lines.append(f'{p.relative_to(root).as_posix()}\t{p.stat().st_size}\t{sha(p)}\n')
        if progress and (i + 1) % 3000 == 0:
            progress(i + 1, len(files))
    return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest(), len(files)


def load(path) -> dict:
    manifest = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('inputs'), dict):
        raise PinError('manifest needs an inputs section')
    return manifest


def _entries(manifest, key):
    entry = manifest['inputs'].get(key)
    if entry is None:
        raise PinError(f'missing pin: {key}')
    return entry if isinstance(entry, list) else [entry]


def check_declared(manifest, keys) -> None:
    """Static check (no hashing): every key present, every entry with a path and a 64-hex digest."""
    for key in keys:
        for e in _entries(manifest, key):
            if not isinstance(e, dict) or not e.get('path'):
                raise PinError(f'pin {key}: path missing')
            digest = e.get('tree_sha256') if 'tree_sha256' in e else e.get('sha256')
            if not isinstance(digest, str) or not SHA.fullmatch(digest):
                raise PinError(f'pin {key}: sha256 missing or malformed for {e["path"]}')


def verify(manifest, keys, *, progress=None) -> dict[str, list[Path]]:
    """Hash every pinned input on disk; returns the resolved paths. Raises PinError on any miss or mismatch."""
    check_declared(manifest, keys)
    out = {}
    for key in keys:
        out[key] = []
        for e in _entries(manifest, key):
            p = expand(e['path'])
            if 'tree_sha256' in e:
                if not p.is_dir():
                    raise PinError(f'pin {key}: not a directory {p}')
                digest, n = tree_sha256(p, progress=progress)
                if digest != e['tree_sha256'] or n != e.get('files'):
                    raise PinError(f'pin {key}: tree differs {p} ({n} files, {digest})')
            else:
                if not p.is_file():
                    raise PinError(f'pin {key}: missing file {p}')
                if sha(p) != e['sha256']:
                    raise PinError(f'pin {key}: SHA256 differs {p}')
            out[key].append(p)
    return out


def check_policy(manifest) -> None:
    """The approved selection policy is fixed in code; the manifest may only restate it."""
    got = manifest.get('policy', {})
    for key, value in POLICY.items():
        if got.get(key) != value:
            raise PinError(f'policy {key} must be {value!r}, manifest has {got.get(key)!r}')
