"""S25 phase 3 step 8: MOVE (never delete) approved delete-plan items to D:\\_to_delete_r2ai\\<original path>.

    python -m r2ai.canonical.relocate --check                       # gates + per-item recheck, moves nothing
    python -m r2ai.canonical.relocate --run --s15-queue-empty       # only after S15 reports an empty GPU queue

Gates (all must hold, else nothing moves):
  * the plan file is byte-identical to the one Vu approved on 2026-10-18 (APPROVED_PLAN_SHA256);
  * --s15-queue-empty is passed AND delta-queue.json has no entry/status still queued, running or blocked;
  * per item: path exists, file count and bytes equal the plan, no file tracked by git, and the reference scan of
    deleteplan still finds no code/config/manifest reference and no canonical/live use.
Items on another drive than D: are skipped (a cross-drive move is copy + delete) and listed for Vu.
Each move is a same-volume os.rename; a JSONL ledger records intent before and result after every rename.
After the moves, P2 is rebuilt from canonical (p2rebuild) and must differ from the approved JSON only in the
3,435 Q1 chunks, with the SHA restored when those are put back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from r2ai.canonical import deleteplan as DP
from r2ai.canonical import p2rebuild as PR
from r2ai.canonical.guard import RUN_ROOT, guarded

PLAN = RUN_ROOT / 'delete-plan' / 'delete-plan.csv'
APPROVED_PLAN_SHA256 = 'c032c340f90ef028c096584fcb0e3af48f8d465897d368857590bbbaff6abbf4'
QUEUE = Path('D:/GitHub/r2ai-stage3/out/runs/reranker-trial/delta-queue.json')
TRASH = Path('D:/_to_delete_r2ai')
BLOCKING = ('queued', 'running', 'not_running', 'blocked', 'pending', 'waiting')


def sha_file(p: Path) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def queue_blockers(queue: dict) -> list[str]:
    """Every status string in the S15 queue file that still names pending GPU work."""
    found = []

    def walk(x, where):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == 'status' and isinstance(v, str) and any(b in v.lower() for b in BLOCKING):
                    found.append(f'{where}: {v}')
                walk(v, f'{where}.{k}')
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f'{where}[{i}]')
    walk(queue, 'queue')
    return found


def destination(src: Path) -> Path:
    p = Path(os.path.abspath(src))
    return TRASH / p.drive.rstrip(':') / Path(*p.parts[1:])


def recheck(row: dict, texts, protected) -> list[str]:
    p = Path(row['path'])
    problems = []
    if not p.is_dir():
        return [f'missing: {p}']
    files = [Path(dp) / f for dp, _, fs in os.walk(p) for f in fs]
    if len(files) != row['files'] or sum(f.stat().st_size for f in files) != row['bytes']:
        problems.append('files/bytes changed since plan')
    top = next((par for par in p.parents if (par / '.git').exists()), None)
    if top is not None:
        tracked = subprocess.run(['git', '-C', str(top), 'ls-files', '--', str(p)], capture_output=True, text=True).stdout
        if tracked.strip():
            problems.append('git-tracked files inside')
    lp = str(p).lower()
    if any(x == lp or x.startswith(lp + '\\') or lp.startswith(x + '\\') for x in protected):
        problems.append('now used by canonical/live')
    keys = DP.keys_for(p)
    refs = [str(tp) for tp, tx in texts if tp.suffix.lower() != '.md' and not str(tp).lower().startswith(lp)
            and any(k in tx for k in keys)]
    if refs:
        problems.append(f'new reference: {refs[0]}')
    return problems


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--run', action='store_true')
    ap.add_argument('--s15-queue-empty', action='store_true', help='S15 has reported an empty GPU queue')
    a = ap.parse_args(argv)
    out = guarded(RUN_ROOT / f'relocate-{time.strftime("%Y%m%dT%H%M%S")}')
    out.mkdir(parents=True)
    report = {'mode': 'run' if a.run else 'check', 'plan': str(PLAN)}
    report['plan_sha256'] = sha_file(PLAN)
    report['plan_approved'] = report['plan_sha256'] == APPROVED_PLAN_SHA256
    blockers = queue_blockers(json.loads(QUEUE.read_text(encoding='utf-8')))
    report['queue_blockers'] = blockers
    report['queue_gate'] = a.s15_queue_empty and not blockers
    plan = pl.read_csv(PLAN).to_dicts()
    texts = DP.reference_texts(list(DP.CODEX_WORKTREES.glob('*/r2ai-stage3')))
    canon = DP.MAIN_REPO / 'data/canonical-20261018'
    m = json.loads((canon / 'manifest.json').read_text(encoding='utf-8'))
    protected = {str(Path(s['index'])).lower() for s in m['vector_segments']}
    protected |= {str(Path(s['docs']).parent).lower() for s in m['live_segments']}
    protected |= {str(Path(p).parent).lower() for p in
                  pl.read_parquet(canon / 'provenance.parquet', columns=['source_path'])['source_path'].unique().to_list()}
    items = []
    for row in plan:
        p = Path(row['path'])
        if p.drive.upper() != 'D:':
            items.append({**row, 'action': 'skip', 'why': 'other drive: cross-drive move = copy + delete, needs Vu'})
            continue
        problems = recheck(row, texts, protected)
        items.append({**row, 'action': 'move' if not problems else 'blocked', 'why': '; '.join(problems),
                      'destination': str(destination(p))})
    report['items'] = {k: sum(i['action'] == k for i in items) for k in ('move', 'blocked', 'skip')}
    report['move_bytes'] = sum(i['bytes'] for i in items if i['action'] == 'move')
    pl.DataFrame(items).write_csv(guarded(out / 'items.csv'))
    gates_ok = report['plan_approved'] and report['queue_gate'] and report['items']['blocked'] == 0
    report['gates_ok'] = gates_ok
    if a.run and gates_ok:
        ledger = guarded(out / 'ledger.jsonl').open('x', encoding='utf-8')
        for i in items:
            if i['action'] != 'move':
                continue
            src, dst = Path(i['path']), Path(i['destination'])
            if dst.exists():
                raise FileExistsError(dst)
            ledger.write(json.dumps({'t': time.time(), 'intent': 'rename', 'src': str(src), 'dst': str(dst)}) + '\n')
            ledger.flush()
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)                            # same volume: atomic, no data copied or deleted
            ledger.write(json.dumps({'t': time.time(), 'done': 'rename', 'src': str(src), 'dst': str(dst)}) + '\n')
            ledger.flush()
        ledger.close()
        report['p2_rebuild'] = PR.rebuild(canon, PR.OUT, f'P2-rebuilt-after-move-{time.strftime("%Y%m%dT%H%M%S")}.json')
    elif a.run:
        report['refused'] = 'gates not satisfied; nothing moved'
    guarded(out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'p2_rebuild'} | (
        {'p2_rebuild_pass': report['p2_rebuild']['pass_with_q1_exception']} if 'p2_rebuild' in report else {}),
        indent=2, ensure_ascii=False))
    if a.run and not gates_ok:
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
