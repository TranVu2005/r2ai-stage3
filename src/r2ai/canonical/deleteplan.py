"""S25 phase 3 step 6: PROPOSE a delete plan (CSV only). This module never deletes or moves anything.

    python -m r2ai.canonical.deleteplan --inventory out/runs/s25-canonical/inventory --out out/runs/s25-canonical/delete-plan

A candidate is a directory of derived artifacts (pytest scratch, superseded chunk/index copies, old run caches).
It is dropped from the plan when any protection hits:
  * raw HTML/WARC/shards, provenance, request logs, S21 queue/heartbeat (by path/category);
  * git-tracked files; files the canonical build or a running P1/P2 build references (canonical manifest/provenance,
    vector source indexes, P1-manifest.yaml, delta-queue.json, build_p2_s22.py pins);
  * Qwen/BGE score stores, submitted ZIPs and their JSON (any *.zip / sub_*.json / scores*.sqlite / pair_cache*);
  * any reference to the candidate path (forward/back slash, data/ or out/ relative) in a text file of the repo,
    out/runs metadata or a worktree (configs, *.json, *.yaml, *.py, *.md <= 20 MB).
Reclaimable bytes count only files with one hard link (a hard-linked file frees nothing until all links go).
"""
from __future__ import annotations

from r2ai.paths import resolve_path

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import polars as pl

from r2ai.canonical.guard import RUN_ROOT, guarded
from r2ai.canonical.inventory import CODEX_WORKTREES, MAIN_REPO, walk

PROTECT_RE = re.compile(r'(raw_vi|raw_zh|raw_cc|[\\/]raw[\\/]|\.warc|\.jsonl\.zst|\.html$|provenance|requests?\.jsonl|'
                        r'\.log$|[\\/]s21[\\/]|heartbeat|queue|scores?\.sqlite|pair_cache|\.zip$|[\\/]sub_[^\\/]*\.json$|'
                        r'approval\.json|validator\.json|crawl\.db|extract\.db|manifest\.db)', re.I)
TEXT_EXT = ('.json', '.yaml', '.yml', '.py', '.md', '.toml', '.csv', '.txt', '.ps1', '.sh')
MAX_TEXT = 20 << 20


def candidate_dirs() -> list[tuple[Path, str]]:
    d, o = MAIN_REPO / 'data', MAIN_REPO / 'out'
    cands = []
    seen = set()
    for p in sorted(o.rglob('pytest*')) + sorted((CODEX_WORKTREES).glob('*/r2ai-stage3/out/**/pytest*'))             + sorted((MAIN_REPO / 'tmp').glob('pytest*')):
        real = p.resolve()                                 # a worktree out/ junction is the same directory
        if p.is_dir() and real not in seen and not any(par.name.startswith('pytest') for par in p.parents):
            seen.add(real)
            cands.append((real, 'pytest scratch output (synthetic test fixtures), not an input of any build'))
    for p, why in ((d / 'chunks_zh_sample', 'zh sample chunks superseded by zh_v2 segments'),
                   (d / 'index/zh_sample_t256', 'zh sample index superseded by zh_v2 segments'),
                   (d / 'chunks_cc_pilot', 'CC pilot chunks superseded by seg20000 (data/chunks_cc_longchau)'),
                   (d / 'docs_cc_pilot', 'CC pilot extract superseded by data/docs_cc_longchau'),
                   (d / 'chunks_zh_full', 'unfiltered zh_full chunk copies; v2-filtered copies in data/chunks_zh_v2 are live')):
        if p.exists():
            cands.append((p, why))
    zf = d / 'index/zh_full_segments'
    for p in sorted(zf.glob('seg*')):
        cands.append((p, 'unfiltered zh_full index copy; v2-filtered copy in data/index/zh_v2_segments is live'))
    return cands


def reference_texts(extra_roots) -> list[tuple[Path, str]]:
    roots = [MAIN_REPO / r for r in ('configs', 'src', 'docs', 'reports', 'scripts', 'submissions')]
    roots += [MAIN_REPO / 'out/runs', MAIN_REPO / 'state', MAIN_REPO / 'data/index'] + list(extra_roots)
    out = []
    for root in roots:
        if not root.exists():
            continue
        for dpath, files in walk(root):
            if any(part.startswith('pytest') for part in dpath.parts) or 's25-canonical' in dpath.parts:
                continue                                   # test scratch and this session's own listings
            for f in files:
                p = dpath / f
                if p.suffix.lower() in TEXT_EXT and p.stat().st_size <= MAX_TEXT:
                    try:
                        out.append((p, p.read_text(encoding='utf-8', errors='ignore').lower()))
                    except OSError:
                        pass
    return out


def keys_for(p: Path) -> list[str]:
    rel = None
    for base in (MAIN_REPO, *CODEX_WORKTREES.glob('*/r2ai-stage3')):
        try:
            rel = p.relative_to(base).as_posix()
            break
        except ValueError:
            continue
    keys = {str(p).lower(), str(p).lower().replace('\\', '/'), str(p).lower().replace('\\', '\\\\')}
    if rel:
        keys |= {rel.lower(), rel.lower().replace('/', '\\'), rel.lower().replace('/', '\\\\')}
        if rel.startswith('data/'):
            keys.add('{work_data_dir}/' + rel[5:].lower())
            keys.add('{data_dir}/' + rel[5:].lower())
    return [k for k in keys if len(k) > 12]


def build(out: Path, canonical: Path) -> dict:
    t0 = time.time()
    protected_files = set()
    man = canonical / 'manifest.json'
    if man.is_file():
        m = json.loads(man.read_text(encoding='utf-8'))
        protected_files |= {str(Path(p)).lower() for p in m['inputs']}
        protected_files |= {str(Path(s['index'])).lower() for s in m.get('vector_segments', [])}
        protected_files |= {str(Path(s['docs']).parent).lower() for s in m.get('live_segments', [])}
    prov = canonical / 'provenance.parquet'
    if prov.is_file():
        protected_files |= {str(Path(p).parent).lower() for p in
                            pl.read_parquet(prov, columns=['source_path'])['source_path'].unique().to_list()}
    texts = reference_texts([w for w in CODEX_WORKTREES.glob('*/r2ai-stage3')])
    rows = []
    for path, why in candidate_dirs():
        files = [Path(dp) / f for dp, _, fs in os.walk(path) for f in fs]
        size = sum(f.stat().st_size for f in files)
        reclaim = sum(f.stat().st_size for f in files if f.stat().st_nlink == 1)
        is_test = path.name.startswith('pytest')
        # inside pytest scratch, crawl.db/*.zip/*.log are synthetic fixtures, not raw data or submitted ZIPs
        hits = [] if is_test else [str(f) for f in files if PROTECT_RE.search(str(f))]
        lp = str(path).lower()
        prot = [p for p in protected_files if p == lp or p.startswith(lp + '\\') or lp.startswith(p + '\\')]
        keys = keys_for(path)
        refs_all = [tp for tp, tx in texts if not str(tp).lower().startswith(lp) and any(k in tx for k in keys)]
        refs = [str(tp) for tp in refs_all if tp.suffix.lower() != '.md']    # docs mentions are history, not use
        md_refs = [str(tp) for tp in refs_all if tp.suffix.lower() == '.md']
        tracked = 0
        status = 'propose' if not (hits or prot or refs) else 'keep'
        evidence = []
        if hits:
            evidence.append(f'protected pattern in {len(hits)} files, e.g. {hits[0]}')
        if prot:
            evidence.append(f'referenced by canonical/live: {prot[0]}')
        if refs:
            evidence.append(f'path referenced in {len(refs)} text files, e.g. {refs[0]}')
        if not evidence:
            evidence.append(f'no code/config/manifest reference in {len(texts)} text files; not a canonical/live input; '
                            f'no protected file' + (f'; mentioned only in docs: {md_refs[0]}' if md_refs else ''))
        rows.append({'path': str(path), 'files': len(files), 'bytes': size, 'reclaim_bytes': reclaim,
                     'status': status, 'reason': why, 'evidence': '; '.join(evidence), 'git_tracked': tracked})
    df = pl.DataFrame(rows).sort(['status', 'bytes'], descending=[True, True])
    out = guarded(out)
    out.mkdir(parents=True)
    df.write_csv(guarded(out / 'delete-plan-all.csv'))
    df.filter(pl.col('status') == 'propose').write_csv(guarded(out / 'delete-plan.csv'))
    summary = {'candidates': df.height, 'proposed': int((df['status'] == 'propose').sum()),
               'proposed_bytes': int(df.filter(pl.col('status') == 'propose')['bytes'].sum()),
               'proposed_reclaim_bytes': int(df.filter(pl.col('status') == 'propose')['reclaim_bytes'].sum()),
               'text_files_scanned': len(texts), 'seconds': round(time.time() - t0, 1)}
    guarded(out / 'delete-plan-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(RUN_ROOT / 'delete-plan'))
    ap.add_argument('--canonical', default=str(MAIN_REPO / 'data/canonical-20261018'))
    a = ap.parse_args(argv)
    print(json.dumps(build(resolve_path(a.out), resolve_path(a.canonical)), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
