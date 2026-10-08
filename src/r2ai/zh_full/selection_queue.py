"""Selective queue used only by zh-full; vi/sample semantics remain unchanged."""
from __future__ import annotations
from r2ai import paths
from vicrawl.urlnorm import normalize_url
import datetime
import json
import asyncio
import time

DEFAULT_DEADLINE = '2026-10-20T23:59:00+07:00'
ELIGIBLE = """(status='pending' OR (status NOT IN ('pending','in_progress','deferred_select')
 AND next_try_at IS NOT NULL AND EXISTS(SELECT 1 FROM select_meta m WHERE m.url_norm=urls.url_norm)))"""
LEGACY = "(status='pending' OR (status NOT IN ('pending','in_progress','deferred_select') AND next_try_at IS NOT NULL))"


def selected_domain(conn, domain):
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='select_meta'").fetchone()
    return bool(exists and conn.execute('SELECT 1 FROM select_meta WHERE domain=? LIMIT 1',(domain,)).fetchone())


def is_deferred(conn,url):
    return bool(conn.execute("SELECT 1 FROM urls WHERE url_norm=? AND status='deferred_select'",(normalize_url(url),)).fetchone())


def next_batch(db, domain, n, now, dry_run=False):
    selected = selected_domain(db.conn, domain)
    sql = select_sql(selected)
    if dry_run:
        rows = db.conn.execute(sql,(domain,now,n)).fetchall()
    else:
        with db._tx():
            rows = db.conn.execute(sql,(domain,now,n)).fetchall()
            db.conn.executemany("UPDATE urls SET status='in_progress' WHERE url_norm=?",[(r[0],) for r in rows])
    return [{'url_norm':r[0],'url':r[1],'doc_ids':json.loads(r[2]),'attempts':r[3],'rank':r[4]} for r in rows]


def select_sql(selected):
    predicate = ELIGIBLE if selected else LEGACY
    order = 'rank' if selected else '(next_try_at IS NOT NULL), rank'
    index = 'INDEXED BY urls_select_rank' if selected else ''
    return f'''SELECT url_norm,url,doc_ids,attempts,rank FROM urls {index} WHERE domain=?
        AND {predicate} AND (next_try_at IS NULL OR next_try_at<=?) ORDER BY {order},url_norm LIMIT ?'''


def next_retry_at(db, domain):
    predicate = ELIGIBLE if selected_domain(db.conn,domain) else LEGACY
    return db.conn.execute(f'SELECT MIN(COALESCE(next_try_at,0)) FROM urls WHERE domain=? AND {predicate}',(domain,)).fetchone()[0]


def remaining_count(conn, domain):
    predicate = ELIGIBLE if selected_domain(conn,domain) else LEGACY
    return conn.execute(f"SELECT count(*) FROM urls WHERE domain=? AND (status='in_progress' OR {predicate})",(domain,)).fetchone()[0]


def deadline_timestamp(value=DEFAULT_DEADLINE):
    when = datetime.datetime.fromisoformat(value)
    if when.tzinfo is None:
        raise ValueError('Deadline requires timezone')
    return when.timestamp()


def deadline_hook(deadline, stop, clock=time.time):
    async def check(request):
        if clock() >= deadline:
            stop.set()
            raise asyncio.CancelledError
    return check
