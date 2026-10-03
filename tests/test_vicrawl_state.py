from r2ai.paths import CONFIG_DIR, ROOT

import asyncio
import threading
from pathlib import Path

from vicrawl.state import DBWriter, StateDB


def groups(domain='x.vn', n=10):
    return [{'url_norm': f'{domain}/{i}', 'url': f'https://{domain}/{i}', 'domain': domain, 'doc_ids': [i, i + 1000], 'rank': i} for i in range(n)]


def test_wal_and_normal_sync(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    assert db.conn.execute('pragma journal_mode').fetchone()[0] == 'wal'
    assert db.conn.execute('pragma synchronous').fetchone()[0] == 1  # NORMAL


def test_add_urls_is_idempotent_and_keeps_existing_rows(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    assert db.add_urls(groups()) == 10
    db.commit_results([{'url_norm': 'x.vn/0', 'status': 'ok', 'http_status': 200, 'reason': 'r', 'fetched_at': 100.0, 'shard': 's1', 'final_url': 'u'}])
    assert db.add_urls(groups()) == 0
    row = db.conn.execute("select status, doc_ids from urls where url_norm='x.vn/0'").fetchone()
    assert row[0] == 'ok' and row[1] == '[0, 1000]'


def test_next_batch_marks_in_progress_and_respects_rank_and_limit(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=10))
    b = db.next_batch('x.vn', 3, now=1000.0)
    assert [r['url_norm'] for r in b] == ['x.vn/0', 'x.vn/1', 'x.vn/2']
    assert b[0]['doc_ids'] == [0, 1000]
    assert db.next_batch('x.vn', 100, now=1000.0, limit=5)[-1]['url_norm'] == 'x.vn/4'
    assert db.next_batch('x.vn', 100, now=1000.0, limit=5) == []


def test_recover_returns_in_progress_to_pending(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=4))
    db.next_batch('x.vn', 4, now=1.0)
    assert db.recover() == 4
    assert len(db.next_batch('x.vn', 10, now=1.0)) == 4


def test_transient_error_retries_after_one_hour_max_three_attempts(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=1))
    t = 1000.0
    for attempt in range(1, 4):
        batch = db.next_batch('x.vn', 5, now=t)
        assert len(batch) == 1, attempt
        db.commit_results([{'url_norm': 'x.vn/0', 'status': 'network_error', 'http_status': None, 'reason': 'ConnectError', 'fetched_at': t, 'shard': 's', 'final_url': ''}])
        assert db.next_batch('x.vn', 5, now=t + 3599) == []
        t += 3600
    assert db.next_batch('x.vn', 5, now=t + 99999) == []
    row = db.conn.execute("select attempts, status, next_try_at from urls").fetchone()
    assert row == (3, 'network_error', None)


def test_terminal_status_never_retried(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=1))
    db.next_batch('x.vn', 1, now=1.0)
    db.commit_results([{'url_norm': 'x.vn/0', 'status': 'soft404_or_home', 'http_status': 404, 'reason': 'x', 'fetched_at': 1.0, 'shard': 's', 'final_url': ''}])
    assert db.next_batch('x.vn', 5, now=10 ** 9) == []


def test_requeue_does_not_touch_attempts(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=2))
    db.next_batch('x.vn', 2, now=1.0)
    db.requeue(['x.vn/0'])
    assert db.conn.execute("select attempts, status from urls where url_norm='x.vn/0'").fetchone() == (0, 'pending')


def test_domain_table_roundtrip(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.upsert_domain('x.vn', rate=2.5, conns=2)
    db.upsert_domain('x.vn', baseline_p95=0.4)
    d = db.get_domain('x.vn')
    assert (d['rate'], d['conns'], d['baseline_p95']) == (2.5, 2, 0.4)


def test_counts(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=3))
    db.next_batch('x.vn', 1, now=1.0)
    db.commit_results([{'url_norm': 'x.vn/0', 'status': 'ok', 'http_status': 200, 'reason': '', 'fetched_at': 1.0, 'shard': 's', 'final_url': ''}])
    assert db.counts()['x.vn'] == {'ok': 1, 'pending': 2}


def test_writer_runs_everything_on_one_thread_in_order(tmp_path):
    async def main():
        w = DBWriter(tmp_path / 'c.db')
        await w.start()
        threads = set()
        order = []

        def op(i):
            def f(db):
                threads.add(threading.get_ident())
                order.append(i)
                return i
            return f
        w.post(op(1))
        w.post(op(2))
        assert await w.call(op(3)) == 3
        await w.stop()
        return threads, order
    threads, order = asyncio.run(main())
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert order == [1, 2, 3]


def test_428_blocked_is_retried_after_one_hour(tmp_path):
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=1))
    db.next_batch('x.vn', 1, now=1000.0)
    db.commit_results([{'url_norm': 'x.vn/0', 'status': 'blocked_4xx', 'http_status': 428, 'reason': 'http_428', 'fetched_at': 1000.0, 'shard': 's', 'final_url': ''}])
    assert db.next_batch('x.vn', 5, now=1000.0 + 3599) == []
    assert len(db.next_batch('x.vn', 5, now=1000.0 + 3600)) == 1


def test_per_domain_challenge_policy_retries_after_24h_at_most_twice(tmp_path):
    from vicrawl.settings import Config
    cfg = Config.load(CONFIG_DIR / 'domains.yaml')
    q = cfg.for_domain('qdnd.vn')
    assert (q.rate_cap, q.max_conns) == (0.3, 1)
    d = cfg.for_domain('dantri.com.vn')
    assert (d.rate_cap, d.start_rate, d.max_conns) == (0.3, 0.3, 1) and d.retry_policy() == {}
    policy = q.retry_policy()
    assert policy['bot_challenge'] == (24 * 3600.0, 3)
    db = StateDB(tmp_path / 'c.db')
    db.add_urls(groups(n=1))
    t = 1000.0
    for attempt in range(1, 4):                          # 1st try + 2 retries
        assert len(db.next_batch('x.vn', 5, now=t)) == 1, attempt
        db.commit_results([{'url_norm': 'x.vn/0', 'status': 'bot_challenge', 'http_status': 200, 'reason': 'c', 'fetched_at': t, 'shard': 's', 'final_url': ''}], policy)
        assert db.next_batch('x.vn', 5, now=t + 24 * 3600 - 1) == []
        t += 24 * 3600
    assert db.next_batch('x.vn', 5, now=t + 10 ** 7) == []
    # the policy only covers challenges: a network error still uses the 1h default
    db.add_urls([{'url_norm': 'x.vn/9', 'url': 'https://x.vn/9', 'doc_ids': [9], 'domain': 'x.vn', 'rank': 9}])
    db.next_batch('x.vn', 5, now=t)
    db.commit_results([{'url_norm': 'x.vn/9', 'status': 'network_error', 'http_status': None, 'reason': 'e', 'fetched_at': t, 'shard': 's', 'final_url': ''}], policy)
    assert len(db.next_batch('x.vn', 5, now=t + 3600)) == 1
