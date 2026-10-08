import json
import sqlite3
import time
from pathlib import Path

import pytest
from vicrawl.state import StateDB
from r2ai.selective import manage as m
from r2ai.zh_full import selection_queue as q
from r2ai.selective.runio import guard


@pytest.fixture
def db(tmp_path):
    db = StateDB(tmp_path / 'crawl.db')
    db.add_urls([{'url_norm':f'x/{i}', 'url':f'https://x/{i}', 'doc_ids':[i], 'domain':'x','rank':i} for i in range(8)])
    db.conn.execute("UPDATE urls SET status='ok',http_status=200,attempts=1,fetched_at=5 WHERE rank=0")
    db.conn.execute("UPDATE urls SET status='network_error',next_try_at=10,attempts=1 WHERE rank=2")
    yield db
    db.close()


def plan():
    return [{'url_norm':f'x/{i}', 'old_status':'network_error' if i==2 else 'pending', 'old_rank':i,
             'new_rank':r, 'new_status':('network_error' if i==2 else 'pending') if k<2 else 'deferred_select',
             'score':float(8-k), 'method':'A1', 'domain':'x'} for k,(i,r) in enumerate(zip([7,2,6,1,3,4,5],range(1,8)))]


def test_apply_only_rank_status_rollback_hash_and_status_conservation(db):
    before = m.urls_digest(db.conn)
    snapshot = db.conn.execute('SELECT * FROM urls WHERE rank=0').fetchone()
    assert m.apply_rows(db.conn, plan(), 'v1') == 7
    assert db.conn.execute('SELECT * FROM urls WHERE url_norm=?',('x/0',)).fetchone() == snapshot
    assert dict(db.conn.execute('SELECT status,count(*) FROM urls GROUP BY status')) == {'ok':1,'pending':1,'network_error':1,'deferred_select':5}
    assert m.rollback(db.conn, 'v1') == 7
    assert m.urls_digest(db.conn) == before


def test_deferred_with_due_retry_never_selected_and_rank_beats_retry_flag(db):
    m.apply_rows(db.conn, plan(), 'v1')
    db.conn.execute("UPDATE urls SET next_try_at=1 WHERE status='deferred_select'")
    rows = q.next_batch(db, 'x', 1000, now=100, dry_run=True)
    assert [r['url_norm'] for r in rows] == ['x/7','x/2']
    assert db.conn.execute("SELECT count(*) FROM urls WHERE status='in_progress'").fetchone()[0] == 0
    assert [r['url_norm'] for r in q.next_batch(db,'x',2,100)] == ['x/7','x/2']
    assert q.next_retry_at(db, 'x') is None  # deferred due retries never prolong lane


def test_reopen_in_score_order_keeps_old_retry_schedule_and_status(db):
    m.apply_rows(db.conn, plan(), 'v1')
    assert m.reopen(db.conn,'x',2) == 2
    assert [r['url_norm'] for r in q.next_batch(db,'x',20,100,dry_run=True)] == ['x/7','x/2','x/6','x/1']
    assert m.reopen(db.conn,'absent',2) == 0
    with pytest.raises(ValueError): m.reopen(db.conn,'x',-1)


def test_reopen_urls_restores_only_listed_deferred_rows_of_that_domain(db):
    m.apply_rows(db.conn, plan(), 'v1')
    # x/7 is already selected, y/1 does not exist, x/6 stays deferred because it is not listed.
    assert m.reopen_urls(db.conn, 'x', ['x/3', 'x/5', 'x/7', 'y/1']) == 2
    assert [r['url_norm'] for r in q.next_batch(db,'x',20,100,dry_run=True)] == ['x/7','x/2','x/3','x/5']
    assert m.reopen_urls(db.conn, 'other', ['x/6']) == 0
    assert db.conn.execute("SELECT status FROM urls WHERE url_norm='x/6'").fetchone()[0] == 'deferred_select'


def test_apply_stale_or_duplicate_plan_is_atomic_and_no_reapply(db):
    rows = plan()
    rows[-1]['old_rank']=200
    before = m.urls_digest(db.conn)
    with pytest.raises(ValueError): m.apply_rows(db.conn,rows,'v1')
    assert m.urls_digest(db.conn) == before
    with pytest.raises(ValueError): m.apply_rows(db.conn,plan()+plan()[:1],'v1')
    assert m.urls_digest(db.conn) == before
    m.apply_rows(db.conn,plan(),'v1')
    with pytest.raises(ValueError): m.apply_rows(db.conn,plan(),'v1')


def test_rollback_refuses_to_overwrite_fetch_results(db):
    m.apply_rows(db.conn,plan(),'v1')
    db.conn.execute("UPDATE urls SET status='ok',attempts=2,fetched_at=100 WHERE url_norm='x/7'")
    with pytest.raises(ValueError,match='changed'): m.rollback(db.conn,'v1')
    assert db.conn.execute("SELECT status FROM urls WHERE url_norm='x/7'").fetchone()[0] == 'ok'


def test_original_queue_semantics_when_no_selection(db):
    db.conn.execute("UPDATE urls SET rank=0 WHERE url_norm='x/2'")
    got=q.next_batch(db,'x',20,100,dry_run=True)
    assert got[-1]['url_norm']=='x/2'  # original pending-first ordering


def test_guard_blocks_old_vi_and_outside_allowlist():
    for bad in ['D:/GitHub/r2ai-stage3-old/state/zh_full/crawl.db','state/crawl.db','data/raw_vi/a','out/runs/zh-full/x']:
        with pytest.raises(ValueError): guard(bad)


def test_deadline_requires_tz_and_default_is_20_oct_2359():
    assert q.deadline_timestamp('2026-10-20T23:59:00+07:00') == 1792515540
    with pytest.raises(ValueError): q.deadline_timestamp('2026-10-20T23:59:00')


def test_deadline_exits_before_any_network_and_default_cli_wires_queue():
    import asyncio
    from r2ai.zh_full import crawl
    assert asyncio.run(crawl.run(deadline=1)) == 0


def test_remaining_excludes_deferred_even_with_retry_timestamp(db):
    m.apply_rows(db.conn,plan(),'v1')
    db.conn.execute("UPDATE urls SET next_try_at=1 WHERE status='deferred_select'")
    assert q.remaining_count(db.conn,'x') == 2


def test_gate_matches_plan_order_and_conserves_pending_and_retry_separately(db):
    original = sqlite3.connect(':memory:')
    db.conn.backup(original)
    rows=plan()
    m.apply_rows(db.conn,rows,'v1')
    gate=m.verify(db.conn,original,{'domains':{'x':{}}},rows)
    assert gate['conservation']['x']['pending'] == {'before':6,'after':1,'deferred':5}
    # A changed rank still has allowed columns but must fail the plan-order gate.
    db.conn.execute("UPDATE urls SET rank=100 WHERE url_norm='x/7'")
    with pytest.raises(ValueError,match='plan order'):
        m.verify(db.conn,original,{'domains':{'x':{}}},rows)
    original.close()


def test_retry_generated_after_selection_kept_but_preexisting_terminal_stays_done(db):
    m.apply_rows(db.conn,plan(),'v1')
    db.conn.execute("UPDATE urls SET status='dead_origin',next_try_at=1 WHERE url_norm IN ('x/0','x/7')")
    got=q.next_batch(db,'x',20,100,dry_run=True)
    assert [r['url_norm'] for r in got] == ['x/7','x/2']
    assert q.next_retry_at(db,'x') == 1


def test_select_queue_can_use_rank_index_without_sorting_all_pending(db):
    m.apply_rows(db.conn,plan(),'v1')
    assert 'urls_select_rank' in {r[1] for r in db.conn.execute('PRAGMA index_list(urls)')}
    explanation=' '.join(str(r) for r in db.conn.execute('EXPLAIN QUERY PLAN '+q.select_sql(True),('x',100,1)))
    assert 'urls_select_rank' in explanation and 'TEMP B-TREE' not in explanation


def test_deadline_hook_blocks_request_before_transport():
    import asyncio
    import httpx
    async def check():
        stop=asyncio.Event(); sent=[]
        def transport(request): sent.append(request); return httpx.Response(200)
        hook=q.deadline_hook(100,stop,clock=lambda:100)
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport),event_hooks={'request':[hook]}) as client:
            with pytest.raises(asyncio.CancelledError): await client.get('https://example.invalid/')
        assert stop.is_set() and sent==[]
    asyncio.run(check())


def test_plan_rejects_incomplete_manifest_and_changed_parquet(tmp_path,monkeypatch):
    import pandas as pd
    import hashlib
    monkeypatch.setattr(m,'RUN',tmp_path)
    path=tmp_path/'plan.json'
    path.write_text(json.dumps({'complete':False}),encoding='utf-8')
    with pytest.raises(ValueError,match='complete'): m.load_plan()
    order=tmp_path/'order-cnkang.com.parquet'
    pd.DataFrame([{'url_norm':'cnkang.com/1','domain':'cnkang.com','rank':1,'status':'pending',
                   'new_status':'pending','new_rank':1,'cut_version':'v1'}]).to_parquet(order)
    manifest={'complete':True,'cut_version':'v1','domains':{'cnkang.com':{}},
              'budgets':{'cnkang.com':{'cut':1}},'order_files':{'cnkang.com':{'sha256':hashlib.sha256(order.read_bytes()).hexdigest(),'rows':1}}}
    path.write_text(json.dumps(manifest),encoding='utf-8')
    assert m.load_plan()[1][0]['old_rank']==1
    order.write_bytes(order.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA'): m.load_plan()


def test_redirect_target_deferred_is_also_off_limits(db):
    m.apply_rows(db.conn,plan(),'v1')
    assert q.is_deferred(db.conn,'https://www.x/6')
    assert not q.is_deferred(db.conn,'https://x/7')
