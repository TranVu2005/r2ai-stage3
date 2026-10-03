"""`crawl.py status` table and the hourly out/crawl_vi_report.md (reads SQLite/parquet only; crawler need not run)."""
from __future__ import annotations

from r2ai import paths  # load env before libraries

import os
import sqlite3
import statistics
import time
from pathlib import Path

from .classify import STATES
from .state import StateDB

PENDING_LIKE = ('pending', 'in_progress')


def _open(db_path: Path) -> StateDB | None:
    p = Path(db_path).resolve()
    if not p.is_file():
        return None
    wal = p.with_name(p.name + '-wal')
    # Legacy snapshots require stopped writers/WAL0; NEW may be read while crawling.
    from r2ai.paths import OLD_ROOT
    legacy = OLD_ROOT is not None and p.is_relative_to(OLD_ROOT)
    if legacy and wal.exists() and wal.stat().st_size:
        raise ValueError('Legacy state has a nonempty WAL; use a consistent stopped-writer snapshot')
    uri = p.as_uri() + '?mode=ro' + ('&immutable=1' if legacy else '')
    db = StateDB.__new__(StateDB)  # retain counts/domain API without writable constructor
    db.conn = sqlite3.connect(uri, uri=True, timeout=60, check_same_thread=False, isolation_level=None)
    db.conn.execute('PRAGMA query_only=ON')
    return db


def domain_rows(db: StateDB) -> list[dict]:
    counts = db.counts()
    doms = db.all_domains()
    now = time.time()
    retry = dict(db.conn.execute('SELECT domain, COUNT(*) FROM urls WHERE next_try_at IS NOT NULL AND status NOT IN (?,?) GROUP BY domain', PENDING_LIKE).fetchall())
    rows = []
    for d, c in counts.items():
        total = sum(c.values())
        pending = sum(c.get(s, 0) for s in PENDING_LIKE)
        done = total - pending
        info = doms.get(d) or {}
        rate = info.get('rate') or 0.0
        remaining = pending + retry.get(d, 0)
        eta = remaining / rate / 3600 if rate and remaining else 0.0
        rows.append({'domain': d, 'total': total, 'done': done, 'ok': c.get('ok', 0), 'ok_pct': 100.0 * c.get('ok', 0) / max(1, done),
                     'rate': rate, 'conns': info.get('conns') or 0, 'state': info.get('state') or '-', 'eta_h': eta, 'remaining': remaining,
                     'halt': info.get('halt_reason') or '', 'counts': c, 'retry_wait': retry.get(d, 0)})
    rows.sort(key=lambda r: -r['total'])
    return rows


def status_text(db_path: Path) -> str:
    db = _open(db_path)
    if db is None:
        return f'no state database at {db_path} (run `python crawl.py init` then `run`)'
    rows = domain_rows(db)
    head = f'{"domain":<36}{"done/total":>17}{"ok%":>7}{"rate/s":>8}{"conns":>6}  {"state":<14}{"ETA":>8}'
    lines = [head, '-' * len(head)]
    for r in rows:
        eta = f'{r["eta_h"]:.1f}h' if r['eta_h'] else '-'
        lines.append(f'{r["domain"]:<36}{str(r["done"]) + "/" + str(r["total"]):>17}{r["ok_pct"]:>6.1f}%{r["rate"]:>8.2f}{r["conns"]:>6}  {r["state"]:<14}{eta:>8}')
        if r['halt']:
            lines.append(f'    ^ {r["halt"]}')
    tot, done, ok = sum(r['total'] for r in rows), sum(r['done'] for r in rows), sum(r['ok'] for r in rows)
    lines.append('-' * len(head))
    lines.append(f'{"TOTAL":<36}{str(done) + "/" + str(tot):>17}{100.0 * ok / max(1, done):>6.1f}%   ok docs={ok}')
    waiting = sum(r['retry_wait'] for r in rows)
    if waiting:
        lines.append(f'{waiting} URL(s) scheduled for retry (>=1h after their last attempt)')
    return '\n'.join(lines)


def _doc_stats(docs_dir: Path) -> dict:
    """domain -> {n, median_tokens, thin_pct, dup_pct}; empty if nothing extracted yet."""
    import pyarrow.dataset as ds
    docs_dir = Path(docs_dir)
    if not docs_dir.exists() or not list(docs_dir.rglob('*.parquet')):
        return {}
    t = ds.dataset(str(docs_dir), format='parquet').to_table(columns=['domain', 'n_tokens_bge_m3', 'text_sha1', 'status'])
    by: dict[str, list] = {}
    for d, tok, sha, st in zip(t['domain'].to_pylist(), t['n_tokens_bge_m3'].to_pylist(), t['text_sha1'].to_pylist(), t['status'].to_pylist()):
        by.setdefault(d, []).append((tok, sha, st))
    out = {}
    for d, items in by.items():
        toks = [i[0] for i in items if i[0] is not None]
        shas = [i[1] for i in items if i[1]]
        dup = len(shas) - len(set(shas))
        out[d] = {'n': len(items), 'median_tokens': statistics.median(toks) if toks else 0,
                  'thin_pct': 100.0 * sum(i[2] == 'thin' for i in items) / len(items), 'dup_pct': 100.0 * dup / max(1, len(shas))}
    return out


def build_report(db_path: Path, docs_dir: Path, out_path: Path) -> str:
    db = _open(db_path)
    if db is None:
        return ''
    rows = domain_rows(db)
    now = time.time()
    last_h = db.conn.execute('SELECT COUNT(*) FROM urls WHERE fetched_at > ?', (now - 3600,)).fetchone()[0]
    ok_total = sum(r['ok'] for r in rows)
    done, total = sum(r['done'] for r in rows), sum(r['total'] for r in rows)
    remaining = sum(r['remaining'] for r in rows)
    thr = last_h / 3600.0
    eta = f'{remaining / thr / 3600:.1f}h' if thr > 0.01 else 'n/a'
    stats = _doc_stats(docs_dir)
    lines = ['# Crawl report (vi)', '', f'Cập nhật: {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
             '## Tổng quan', '',
             f'- Tiến độ: **{done:,} / {total:,}** URL ({100.0 * done / max(1, total):.1f}%)',
             f'- Doc `ok` (trạng thái lúc crawl): **{ok_total:,}**',
             f'- Throughput 1 giờ qua: **{thr:.2f} URL/s** ({last_h:,} URL)',
             f'- ETA toàn bộ (theo throughput 1h): **{eta}**  (còn {remaining:,} URL)', '']
    if stats:
        n = sum(s['n'] for s in stats.values())
        toks = [s['median_tokens'] for s in stats.values()]
        lines += [f'- Đã extract: {n:,} doc; median token/doc (median theo domain): {statistics.median(toks):.0f}', '']
    else:
        lines += ['- Chưa extract doc nào (`python extract.py`) nên chưa có median token / % thin / % trùng.', '']
    cols = list(STATES)
    lines += ['## Trạng thái theo domain', '', '| domain | tổng | pending | ' + ' | '.join(cols) + ' | ok% | rate | state | median tok | % thin | % trùng ND |',
              '|' + '---|' * (len(cols) + 9)]
    for r in rows:
        c = r['counts']
        s = stats.get(r['domain'])
        pend = sum(c.get(x, 0) for x in PENDING_LIKE)
        lines.append(f'| {r["domain"]} | {r["total"]} | {pend} | ' + ' | '.join(str(c.get(x, 0)) for x in cols)
                     + f' | {r["ok_pct"]:.1f}% | {r["rate"]:.2f} | {r["state"]} | ' + (f'{s["median_tokens"]:.0f} | {s["thin_pct"]:.1f}% | {s["dup_pct"]:.1f}%' if s else '- | - | -') + ' |')
    text = '\n'.join(lines) + '\n'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.md.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, out_path)
    return text
