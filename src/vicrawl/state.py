"""SQLite checkpoint (WAL, synchronous=NORMAL). All writes go through one DBWriter thread."""
from __future__ import annotations

from r2ai.paths import ROOT

import asyncio
import json
import logging
import queue
import sqlite3
import threading
from pathlib import Path

from .classify import RETRYABLE_4XX, TRANSIENT

log = logging.getLogger('vicrawl.state')
MAX_ATTEMPTS = 3
RETRY_GAP_S = 3600.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls(
  url_norm TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  doc_ids TEXT NOT NULL,
  domain TEXT NOT NULL,
  rank INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  http_status INTEGER,
  reason TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  fetched_at REAL,
  next_try_at REAL,
  shard TEXT,
  final_url TEXT
);
CREATE INDEX IF NOT EXISTS urls_dom_status_rank ON urls(domain, status, rank);
CREATE TABLE IF NOT EXISTS domains(
  domain TEXT PRIMARY KEY,
  rate REAL, conns INTEGER, baseline_p95 REAL, crawl_delay REAL,
  state TEXT, halt_reason TEXT, approved INTEGER DEFAULT 0, qa_done INTEGER DEFAULT 0,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
DOMAIN_COLS = ('rate', 'conns', 'baseline_p95', 'crawl_delay', 'state', 'halt_reason', 'approved', 'qa_done', 'updated_at')


class StateDB:
    def __init__(self, path: Path, readonly: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=60, check_same_thread=False, isolation_level=None)
        self.conn.execute('PRAGMA busy_timeout=60000')
        if not readonly:
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA synchronous=NORMAL')
            self.conn.executescript(SCHEMA)

    # -- urls ------------------------------------------------------------------------------
    def add_urls(self, groups: list[dict]) -> int:
        rows = [(g['url_norm'], g['url'], json.dumps(g['doc_ids']), g['domain'], g['rank']) for g in groups]
        before = self.conn.total_changes
        with self._tx():
            self.conn.executemany('INSERT OR IGNORE INTO urls(url_norm,url,doc_ids,domain,rank) VALUES (?,?,?,?,?)', rows)
        return self.conn.total_changes - before

    def recover(self) -> int:
        with self._tx():
            return self.conn.execute("UPDATE urls SET status='pending' WHERE status='in_progress'").rowcount

    def next_batch(self, domain: str, n: int, now: float, limit: int | None = None) -> list[dict]:
        scope = '' if limit is None else ' AND rank < :limit'
        sql = f"""SELECT url_norm,url,doc_ids,attempts,rank FROM urls
                  WHERE domain=:d{scope} AND (
                       (status='pending' AND (next_try_at IS NULL OR next_try_at<=:now))
                    OR (status NOT IN ('pending','in_progress') AND next_try_at IS NOT NULL AND next_try_at<=:now))
                  ORDER BY (next_try_at IS NOT NULL), rank LIMIT :n"""
        with self._tx():
            rows = self.conn.execute(sql, {'d': domain, 'now': now, 'limit': limit, 'n': n}).fetchall()
            self.conn.executemany("UPDATE urls SET status='in_progress' WHERE url_norm=?", [(r[0],) for r in rows])
        return [{'url_norm': r[0], 'url': r[1], 'doc_ids': json.loads(r[2]), 'attempts': r[3], 'rank': r[4]} for r in rows]

    def commit_results(self, results: list[dict], retry_policy: dict | None = None):
        """Record fetch outcomes. Caller guarantees the shard holding them was already renamed.

        retry_policy: {status: (gap_s, max_attempts)} per-domain overrides (DomainCfg.retry_policy)."""
        retry_policy = retry_policy or {}
        with self._tx():
            for r in results:
                transient = r['status'] in TRANSIENT or (r['status'] == 'blocked_4xx' and r.get('http_status') in RETRYABLE_4XX)
                gap, max_attempts = retry_policy.get(r['status'], (RETRY_GAP_S, MAX_ATTEMPTS))
                attempts = (self.conn.execute('SELECT attempts FROM urls WHERE url_norm=?', (r['url_norm'],)).fetchone() or (0,))[0] + 1
                next_try = r['fetched_at'] + gap if transient and attempts < max_attempts else None
                self.conn.execute(
                    'UPDATE urls SET status=?, http_status=?, reason=?, attempts=?, fetched_at=?, next_try_at=?, shard=?, final_url=? WHERE url_norm=?',
                    (r['status'], r.get('http_status'), (r.get('reason') or '')[:300], attempts, r['fetched_at'], next_try, r.get('shard'), r.get('final_url'), r['url_norm']))

    def requeue(self, url_norms: list[str]):
        with self._tx():
            self.conn.executemany("UPDATE urls SET status='pending' WHERE url_norm=?", [(u,) for u in url_norms])

    def counts(self, limit: int | None = None) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        where, params = ('WHERE rank < ?', (limit,)) if limit else ('', ())
        for d, s, n in self.conn.execute(f'SELECT domain,status,COUNT(*) FROM urls {where} GROUP BY domain,status', params):
            out.setdefault(d, {})[s] = n
        return out

    def reset_errors(self, domain: str) -> int:
        with self._tx():
            n = self.conn.execute(
                "UPDATE urls SET status='pending', attempts=0, next_try_at=NULL WHERE domain=? AND status IN ('network_error','dead_origin','bot_challenge','blocked_4xx','cookie_challenge','in_progress')",
                (domain,)).rowcount
            self.conn.execute("UPDATE domains SET state='active', halt_reason=NULL WHERE domain=? AND state='halted'", (domain,))
        return n

    # -- domains / meta ----------------------------------------------------------------------
    def upsert_domain(self, domain: str, **fields):
        bad = set(fields) - set(DOMAIN_COLS)
        if bad:
            raise ValueError(f'unknown domain columns {bad}')
        with self._tx():
            self.conn.execute('INSERT OR IGNORE INTO domains(domain) VALUES (?)', (domain,))
            if fields:
                sets = ', '.join(f'{k}=?' for k in fields)
                self.conn.execute(f'UPDATE domains SET {sets} WHERE domain=?', (*fields.values(), domain))

    def get_domain(self, domain: str) -> dict | None:
        cur = self.conn.execute('SELECT * FROM domains WHERE domain=?', (domain,))
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

    def all_domains(self) -> dict[str, dict]:
        cur = self.conn.execute('SELECT * FROM domains')
        cols = [c[0] for c in cur.description]
        return {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

    def get_meta(self, key: str, default=None):
        row = self.conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str):
        with self._tx():
            self.conn.execute('INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))

    def ok_count(self, domain: str | None = None) -> int:
        if domain:
            return self.conn.execute("SELECT COUNT(*) FROM urls WHERE domain=? AND status='ok'", (domain,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM urls WHERE status='ok'").fetchone()[0]

    class _Tx:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            self.conn.execute('BEGIN IMMEDIATE')

        def __exit__(self, et, ev, tb):
            self.conn.execute('COMMIT' if et is None else 'ROLLBACK')

    def _tx(self):
        return self._Tx(self.conn)

    def close(self):
        self.conn.close()


class DBWriter:
    """Single thread owns the connection; async code submits callables taking a StateDB."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.q: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.db: StateDB | None = None
        self._ready = threading.Event()

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.Thread(target=self._run, name='db-writer', daemon=True)
        self.thread.start()
        await asyncio.to_thread(self._ready.wait)

    def _run(self):
        self.db = StateDB(self.path)
        self._ready.set()
        while True:
            item = self.q.get()
            if item is None:
                break
            fn, fut = item
            try:
                res = fn(self.db)
                if fut is not None:
                    self.loop.call_soon_threadsafe(self._resolve, fut, res, None)
            except BaseException as e:  # noqa: BLE001
                log.exception('db op failed')
                if fut is not None:
                    self.loop.call_soon_threadsafe(self._resolve, fut, None, e)
        self.db.close()

    @staticmethod
    def _resolve(fut, res, err):
        if fut.done():
            return
        fut.set_exception(err) if err else fut.set_result(res)

    def post(self, fn):
        self.q.put((fn, None))

    async def call(self, fn):
        fut = self.loop.create_future()
        self.q.put((fn, fut))
        return await fut

    async def stop(self):
        self.q.put(None)
        await asyncio.to_thread(self.thread.join)
