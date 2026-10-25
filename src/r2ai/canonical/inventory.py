"""S25 phase 1: read-only inventory of every doc/chunk/vector/score/snapshot/ZIP artifact.

    python -m r2ai.canonical.inventory --out out/runs/s25-canonical/inventory

Walks the repo's data/, out/, state/ (plus untracked root dirs) and every Codex worktree. Files of one kind in
one directory are aggregated into one row when there are more than GROUP_MIN of them (crawl shards, extract
shards). Per row: path, files, bytes, mtime range, git-tracked count, category, session tag, rows and distinct
doc_id count where cheaply readable (Parquet ID columns, segment doc_id.bin, SQLite row counts). SQLite is opened
with immutable=1 (no -shm/-wal created); ZIPs are listed, not decompressed. Nothing is written outside --out.
"""
from __future__ import annotations

from r2ai.paths import ROOT as WORKTREE_ROOT

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.canonical.guard import RUN_ROOT, guarded, write_parquet

MAIN_REPO = Path('D:/GitHub/r2ai-stage3')
CODEX_WORKTREES = Path('C:/Users/Admin/.codex/worktrees')
OLD = Path('D:/GitHub/r2ai-stage3-old')
SKIP_DIRS = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache', 'hf_model_cache'}
GROUP_MIN = 20
TZ = dt.timezone(dt.timedelta(hours=7))
ID_COLUMNS = ('doc_ids', 'doc_id', 'id')


def main_roots() -> list[tuple[str, Path]]:
    roots = [('main', MAIN_REPO / d) for d in ('data', 'out', 'state', 'artifacts', 'tmp', '.scratch', 'submissions')]
    roots.append(('main', MAIN_REPO / 'data' / 'hf_model_cache'))     # listed as one aggregate, never walked deeply
    for wt in sorted(CODEX_WORKTREES.glob('*/r2ai-stage3')):
        roots.append((f'codex:{wt.parent.name}', wt))
    return roots


def extra_roots() -> list[tuple[str, Path]]:
    """Sibling and .superpowers worktrees (read-only listing; OLD and this S25 worktree excluded)."""
    roots = [(f'wt:{p.name}', p) for p in sorted(MAIN_REPO.parent.glob('r2ai-stage3-*'))
             if p.is_dir() and p.resolve() != OLD.resolve()]
    roots += [(f'wt:superpowers/{p.name}', p) for p in sorted((MAIN_REPO / '.superpowers' / 'worktrees').glob('*'))
              if p.is_dir() and p.resolve() != WORKTREE_ROOT.resolve()]
    return roots


def git_tracked(repo: Path) -> set[str]:
    try:
        out = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z'], capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {os.path.normcase(str((repo / p).resolve())) for p in out.decode('utf-8', 'replace').split('\0') if p}


def category(path: str) -> str:
    p = path.replace('\\', '/').lower()
    name = p.rsplit('/', 1)[-1]
    if 'hf_model_cache' in p:
        return 'model-cache'
    if re.search(r'/raw(_[a-z_]+)?/', p) or name.endswith(('.jsonl.zst', '.warc.gz', '.warc', '.html')) or '/cc_aws/' in p:
        return 'raw'
    if '/s21/' in p and ('queue' in name or 'heartbeat' in name or 'pending' in name):
        return 's21-queue'
    if name.endswith('.log') or name in ('requests.jsonl',) or 'request' in name and name.endswith('.jsonl'):
        return 'log'
    if 'provenance' in name or 'provenance' in p.rsplit('/', 2)[-2]:
        return 'provenance'
    if name.endswith('.zip') or name.endswith('.tmp.zip'):
        return 'zip'
    if name in ('dense.npy', 'sparse.npz', 'data.bin', 'indices.bin', 'indptr.bin') or '/sparse_mmap/' in p \
            or '.blocks/' in p or name.endswith(('.npy', '.npz')):
        return 'vectors'
    if name.startswith('chunks_t') or name == 'text.arrow' or name == 'doc_id.bin':
        return 'chunks'
    if 'score' in name or 'pair_cache' in name or 'rerank' in name or name.endswith('.pkl'):
        return 'scores'
    if name.endswith(('.db', '.sqlite')):
        return 'db'
    if name.endswith('.parquet'):
        return 'parquet'
    if name.endswith('.json') or name.endswith('.yaml'):
        return 'meta'
    return 'other'


SESSION_RE = re.compile(r'(?<![a-z0-9])(s\d{1,2}[a-z]?)(?![0-9])')
SEG_RE = re.compile(r'seg(\d{5})')


def session_tag(path: str) -> str:
    p = path.replace('\\', '/').lower()
    tags = []
    seg = SEG_RE.search(p)
    if seg:
        tags.append(f'seg{seg.group(1)}')
    tags += [m for m in SESSION_RE.findall(p) if m not in tags][:2]
    for key, tag in (('zh_full', 'zh-full'), ('zh-full', 'zh-full'), ('zh_v2', 'zh-v2'), ('zh-v2', 'zh-v2'),
                     ('zh_sample', 'zh-sample'), ('cc_longchau', 'cc-longchau'), ('cc-longchau', 'cc-longchau'),
                     ('docs_vi', 'vi-extract'), ('raw_vi', 'vi-crawl'), ('/chunks/', 'vi-base'),
                     ('/index/t256', 'vi-base'), ('pubmed', 'en-pubmed'), ('reranker-trial', 'reranker-trial')):
        if key in p and tag not in tags:
            tags.append(tag)
    return ','.join(tags)


def _ids_from_parquet(path: Path):
    """-> (rows, ids ndarray|None, columns, has_text)."""
    f = pq.ParquetFile(path)
    names = f.schema_arrow.names
    rows = f.metadata.num_rows
    has_text = 'doc_text' in names or 'body' in names or 'text' in names or 'chunk_text' in names
    col = next((c for c in ID_COLUMNS if c in names), None)
    if col == 'id' and 'url' not in names:
        col = None
    if col is None or rows == 0:
        return rows, None, names, has_text
    t = f.read(columns=[col]).column(0)
    if pa.types.is_list(t.type) or pa.types.is_large_list(t.type):
        t = pa.chunked_array([c.flatten() for c in t.chunks], type=t.type.value_type)
    try:
        arr = np.asarray(t.to_numpy(), dtype=np.int64)
    except (pa.ArrowInvalid, ValueError, TypeError):
        return rows, None, names, has_text
    return rows, arr, names, has_text


def probe_file(path: Path) -> dict:
    """Cheap per-file facts: rows, doc ids, schema; never decompresses ZIP payloads."""
    name = path.name.lower()
    info = {'rows': None, 'ids': None, 'note': ''}
    try:
        if name.endswith('.parquet'):
            rows, ids, names, has_text = _ids_from_parquet(path)
            info.update(rows=rows, ids=ids, note=('text;' if has_text else '') + 'cols=' + '|'.join(names[:12]))
        elif name == 'doc_id.bin':
            ids = np.fromfile(path, dtype=np.int64)
            info.update(rows=len(ids), ids=ids)
        elif name.endswith('.npy'):
            arr = np.load(path, mmap_mode='r')                 # header only; no data pages read
            info.update(rows=arr.shape[0] if arr.shape else 0, note=f'shape={arr.shape} dtype={arr.dtype}')
            del arr
        elif name.endswith('.arrow'):
            with pa.memory_map(str(path)) as src:
                try:
                    reader = pa.ipc.open_file(src)
                    info.update(rows=sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches)))
                except pa.ArrowInvalid:
                    reader = pa.ipc.open_stream(src)
                    info.update(rows=sum(b.num_rows for b in reader))
        elif name.endswith(('.db', '.sqlite')) and path.stat().st_size > 0:
            conn = sqlite3.connect(f'file:{path.as_posix()}?mode=ro&immutable=1', uri=True)
            try:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                counts = []
                for t in tables[:12]:
                    try:
                        counts.append(f'{t}={conn.execute(f"SELECT count(*) FROM [{t}]").fetchone()[0]}')
                    except sqlite3.DatabaseError:
                        counts.append(f'{t}=?')
                info.update(note='tables:' + ','.join(counts))
            finally:
                conn.close()
        elif name.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                members = z.infolist()
                info.update(rows=len(members),
                            note='members:' + ','.join(f'{m.filename}({m.file_size})' for m in members[:5]))
    except Exception as e:                                   # inventory must not stop on one bad file
        info['note'] = f'probe_error={type(e).__name__}: {str(e)[:120]}'
    return info


def kind_key(path: Path) -> str:
    n = path.name.lower()
    for ext in ('.jsonl.zst', '.tmp.zip', '.parquet', '.zst', '.json', '.npy', '.bin', '.arrow', '.zip', '.db',
                '.sqlite', '.html', '.gz', '.log', '.pkl', '.npz', '.csv', '.jsonl', '.md', '.yaml', '.py'):
        if n.endswith(ext):
            return ext
    return path.suffix.lower() or '(none)'


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Junctions/symlinks are not followed: a worktree's data/ link would double count the main repo.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not os.path.islink(os.path.join(dirpath, d)) and not os.path.isjunction(os.path.join(dirpath, d))]
        yield Path(dirpath), filenames


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, TZ).strftime('%Y-%m-%d %H:%M')


def build(out: Path, v2_ids: np.ndarray, progress_every: int = 200, roots=None) -> dict:
    rows = []
    v2_sorted = np.sort(v2_ids)
    tracked = {}
    roots = main_roots() if roots is None else roots
    for repo in [MAIN_REPO, *sorted(CODEX_WORKTREES.glob('*/r2ai-stage3')), *[r for o, r in roots if o.startswith('wt:')]]:
        tracked.update({p: str(repo) for p in git_tracked(repo)})
    started = time.time()
    nfiles = 0
    for owner, root in roots:
        if not root.exists():
            continue
        if root.name == 'hf_model_cache':
            sizes = [(p.stat().st_size, p.stat().st_mtime) for p in root.rglob('*') if p.is_file()]
            rows.append(dict(owner=owner, path=str(root), files=len(sizes), bytes=sum(s for s, _ in sizes),
                             mtime_min=iso(min(m for _, m in sizes)) if sizes else '',
                             mtime_max=iso(max(m for _, m in sizes)) if sizes else '',
                             git_tracked=0, category='model-cache', session='', rows=None, doc_ids=None,
                             doc_ids_v2=None, note='aggregate'))
            continue
        for d, filenames in walk(root):
            if owner == 'main' and 'hf_model_cache' in d.parts:
                continue
            if owner.startswith(('codex:', 'wt:')) and d == root:
                filenames = [f for f in filenames if not f.endswith(('.md', '.toml', '.ini'))]
            groups = defaultdict(list)
            for fn in filenames:
                p = d / fn
                groups[(kind_key(p), category(str(p)))].append(p)
            for (ext, cat), paths in groups.items():
                if owner.startswith(('codex:', 'wt:')) and any(part in ('src', 'tests', 'docs', 'configs', 'scripts')
                                                       for part in d.relative_to(root).parts[:1]):
                    continue                                  # worktree code is git history, not data
                stats = [p.stat() for p in paths]
                ntracked = sum(os.path.normcase(str(p.resolve())) in tracked for p in paths)
                aggregate = len(paths) > GROUP_MIN
                probe_targets = paths if cat in ('parquet', 'chunks', 'provenance', 'scores', 'vectors', 'db', 'zip', 's21-queue') \
                    or ext in ('.parquet', '.bin', '.npy', '.arrow', '.zip', '.db', '.sqlite') else []
                if cat == 'raw':
                    probe_targets = []                        # raw shards: count/size only
                if aggregate:
                    ids_acc, nrows, notes = [], 0, set()
                    for p in probe_targets:
                        info = probe_file(p)
                        nrows += info['rows'] or 0
                        if info['ids'] is not None:
                            ids_acc.append(np.unique(info['ids']))
                        if info['note'] and not notes:
                            notes.add(info['note'][:200])
                    ids = np.unique(np.concatenate(ids_acc)) if ids_acc else None
                    rows.append(dict(owner=owner, path=str(d / f'*{ext}'), files=len(paths),
                                     bytes=sum(s.st_size for s in stats),
                                     mtime_min=iso(min(s.st_mtime for s in stats)),
                                     mtime_max=iso(max(s.st_mtime for s in stats)), git_tracked=ntracked,
                                     category=cat, session=session_tag(str(d)), rows=nrows if probe_targets else None,
                                     doc_ids=None if ids is None else len(ids),
                                     doc_ids_v2=None if ids is None else int(np.isin(ids, v2_sorted, assume_unique=True).sum()),
                                     note='aggregate;' + ';'.join(notes)))
                else:
                    for p, s in zip(paths, stats):
                        info = probe_file(p) if p in probe_targets else {'rows': None, 'ids': None, 'note': ''}
                        ids = None if info['ids'] is None else np.unique(info['ids'])
                        rows.append(dict(owner=owner, path=str(p), files=1, bytes=s.st_size,
                                         mtime_min=iso(s.st_mtime), mtime_max=iso(s.st_mtime),
                                         git_tracked=int(os.path.normcase(str(p.resolve())) in tracked),
                                         category=cat, session=session_tag(str(p)), rows=info['rows'],
                                         doc_ids=None if ids is None else len(ids),
                                         doc_ids_v2=None if ids is None else int(np.isin(ids, v2_sorted, assume_unique=True).sum()),
                                         note=info['note'][:300]))
                nfiles += len(paths)
            if progress_every and nfiles and len(rows) % progress_every == 0:
                print(f'[{time.time()-started:7.0f}s] files={nfiles} rows={len(rows)} at {d}', flush=True)
    table = pa.Table.from_pylist(rows)
    write_parquet(table, out / 'inventory.parquet')
    import pyarrow.csv as pacsv
    csv_tmp = guarded(out / 'inventory.csv')
    pacsv.write_csv(table, csv_tmp)
    summary = {'rows': len(rows), 'files': nfiles, 'bytes': int(sum(r['bytes'] for r in rows)),
               'seconds': round(time.time() - started, 1), 'worktree_code_root': str(WORKTREE_ROOT)}
    (guarded(out / 'inventory-summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RUN_ROOT / 'inventory'))
    ap.add_argument('--extra', action='store_true', help='sibling/.superpowers worktrees instead of the main scope')
    ap.add_argument('--corpus', default=str(MAIN_REPO / 'data' / 'corpus_v2_ids.parquet'))
    a = ap.parse_args(argv)
    out = guarded(a.out)
    out.mkdir(parents=True)
    v2 = pq.read_table(a.corpus, columns=['id'])['id'].to_numpy()
    print(json.dumps(build(out, v2, roots=extra_roots() if a.extra else None), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
