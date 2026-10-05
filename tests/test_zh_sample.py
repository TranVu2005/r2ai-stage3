import pytest


def test_sample_groups_seed_and_floor():
    from r2ai.zh_sample.sample import sample_rows
    rows = [(i, f'https://www.example.cn/q/{i}') for i in range(1000)]
    rows += [(1000, 'http://example.cn/q/0')]
    selected, inventory = sample_rows(rows, {'example.cn'})
    again, _ = sample_rows(list(reversed(rows)), {'example.cn'})
    assert selected == again
    assert len(selected) == 200
    assert inventory[0]['n_unique'] == 1000
    assert [g['rank'] for g in selected] == list(range(200))
    small, _ = sample_rows([(0, 'https://a.cn/q'), (1, 'http://www.a.cn/q')], {'a.cn'})
    assert small[0]['doc_ids'] == [0, 1]


def test_excluded_domains_never_sampled():
    from r2ai.zh_sample.sample import sample_rows
    groups, inventory = sample_rows([(0, 'https://zysjonline.com/a'), (1, 'https://bingli.iiyi.com/a')], {'zysjonline.com', 'bingli.iiyi.com'})
    assert groups == []
    assert all(r['excluded'] for r in inventory)


def test_output_guard_rejects_vi_and_old():
    from r2ai.zh_sample.common import guard_owned
    from r2ai.paths import ROOT, OLD_ROOT
    for p in (ROOT / 'data/raw_vi', ROOT / 'data/index/t256', ROOT / 'state/crawl.db', OLD_ROOT):
        if p is not None:
            with pytest.raises(ValueError):
                guard_owned(p)


@pytest.mark.parametrize('encoding', ['gbk', 'gb2312', 'gb18030'])
def test_zh_charset(encoding):
    from r2ai.zh_sample.crawl import decode_zh
    text = '<html><p>患者头痛，建议及时检查。</p></html>'
    assert decode_zh(text.encode(encoding), '')[0] == text


def test_zh_soft404():
    from r2ai.zh_sample.crawl import classify_zh
    assert classify_zh('<title>您访问的页面不存在</title><p>返回首页</p>', 'https://a.cn/q', 'https://a.cn/q', 200)[0] == 'soft404_or_home'


def test_qa_extractor_keeps_answer_in_body():
    from r2ai.zh_sample.extract import extract_zh
    question = '患者最近反复头痛，请问需要做什么检查？'
    answer = '建议患者及时去医院就诊，完善相关检查，根据检查结果选择治疗。' * 6
    raw = f'<title>头痛</title><div class="q-desc">{question}</div><div class="ans-cont"><p>{answer}</p></div><nav>导航链接</nav>'
    doc = extract_zh(raw, '120ask.com')
    assert doc.question == question
    assert doc.answer == answer
    assert doc.answer in doc.body
    assert '导航链接' not in doc.body


def test_120ask_smoke_selectors_exclude_doctor_metadata():
    from r2ai.zh_sample.extract import extract_zh
    answer = '患者需要完善相关检查并遵医嘱治疗。' * 20
    raw = f'<title>头痛</title><div class="b_askcont"><p class="crazy_new">反复头痛怎么办？</p></div><div class="b_anscontc">2026-01-01 我要投诉<div class="b_anscont_cont"><div class="crazy_new">{answer}</div></div></div>'
    doc = extract_zh(raw, '120ask.com')
    assert doc.question == '反复头痛怎么办？'
    assert doc.answer == answer
    assert '我要投诉' not in doc.body


def test_merge_uses_direct_scores_and_stable_vi_ties():
    from r2ai.zh_sample.evaluate import merge_scores, yield_proxy
    merged = merge_scores([(1, 2.0), (2, 1.0)], [(3, 1.5), (4, 1.0)], k=3)
    assert merged == [(1, 2.0), (3, 1.5), (2, 1.0)]
    result = yield_proxy({10: merged, 11: merged}, {3: 'a.cn', 4: 'b.cn'}, {'a.cn': 200, 'b.cn': 100})
    assert result['a.cn']['hits'] == 2
    assert result['a.cn']['hits_per_1k_extracted_docs'] == 10
    assert result['b.cn']['hits'] == 0


def test_robots_checked_at_redirect_destination(monkeypatch):
    import asyncio
    from r2ai.zh_sample.crawl import PoliteClient
    # Exercise actual redirect/robots logic; only HTTP transport is replaced.
    client = object.__new__(PoliteClient)
    client.domain = 'a.cn'
    calls = []
    async def allowed(url):
        calls.append(('robots', url))
        return (url == 'https://a.cn/q', 'robots_disallowed')
    async def hop(url, kind):
        calls.append((kind, url))
        return 302, {'location': 'https://b.cn/private'}, b'', '', .1
    client.allowed, client.hop = allowed, hop
    row = {'url': 'https://a.cn/q', 'url_norm': 'a.cn/q', 'attempts': 0, 'doc_ids': [1]}
    result = asyncio.run(client.fetch(row))
    assert result['status'] == 'blocked_4xx'
    assert calls == [('robots', 'https://a.cn/q'), ('page', 'https://a.cn/q'), ('robots', 'https://b.cn/private')]


def test_unavailable_robots_retryable():
    import asyncio
    from r2ai.zh_sample.crawl import PoliteClient
    client = object.__new__(PoliteClient)
    client.domain = 'a.cn'
    async def allowed(url):
        return False, 'robots_unavailable'
    client.allowed = allowed
    result = asyncio.run(client.fetch({'url': 'https://a.cn/q', 'attempts': 0}))
    assert result['status'] == 'network_error'
    assert result['attempts'] == 1
    assert result['html'] is None


def test_merge_preserves_d50_vi_order_even_across_tiers():
    from r2ai.zh_sample.evaluate import merge_scores
    vi = [(1, 1.0), (2, 3.0), (3, .5)]
    assert merge_scores(vi, [], 150) == vi
    merged = merge_scores(vi, [(4, 2.0)], 3)
    assert {d for d, _ in merged} == {1, 2, 4}
    assert [d for d, _ in merged if d in (1,2)] == [1,2]


def test_gpu_guard_wddm_distinguishes_compute_from_desktop():
    from r2ai.zh_sample.index import compute_processes
    snapshot = '| 0 N/A N/A 100 C+G C:/Windows/explorer.exe N/A |\n| 0 N/A N/A 200 C C:/python.exe N/A |\n| 0 N/A N/A 300 C C:/python.exe N/A |'
    assert compute_processes(snapshot, own_pid=300) == [{'pid': 200, 'type': 'C', 'name': 'C:/python.exe'}]


def test_interrupt_flushes_private_checkpoint_and_resumes(tmp_path, monkeypatch):
    import asyncio
    import importlib
    import json
    import signal
    import sqlite3
    from vicrawl.state import StateDB
    m = importlib.import_module('r2ai.zh_sample.crawl')
    run, raw, db_path = tmp_path / 'run', tmp_path / 'raw', tmp_path / 'crawl.db'
    run.mkdir()
    groups = [{'url': f'https://a.cn/q/{i}', 'url_norm': f'a.cn/q/{i}', 'domain': 'a.cn', 'doc_ids': [i], 'rank': i} for i in range(10)]
    (run / 'sample.json').write_text(json.dumps(groups), encoding='utf-8')
    db = StateDB(db_path)
    db.add_urls(groups)
    db.conn.close()
    monkeypatch.setattr(m, 'RUN', run)
    monkeypatch.setattr(m, 'RAW', raw)
    monkeypatch.setattr(m, 'DB', db_path)
    handlers = {}
    monkeypatch.setattr(m.signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    async def network_check(self):
        pass
    monkeypatch.setattr(m.DirectNetwork, 'check', network_check)
    calls = []
    async def fetch(self, row):
        calls.append(row['rank'])
        if len(calls) == 1:
            handlers[signal.SIGINT](signal.SIGINT, None)
        await asyncio.sleep(0)
        return {**row, 'domain': 'a.cn', 'status': 'ok', 'reason': 'test', 'html': '检查建议' * 100,
                'http_status': 200, 'fetched_at': 1000, 'final_url': row['url'], 'attempts': row['attempts'] + 1}
    monkeypatch.setattr(m.PoliteClient, 'fetch', fetch)
    assert asyncio.run(m.crawl(domains={'a.cn'})) == 130
    db = StateDB(db_path)
    statuses = dict(db.conn.execute('SELECT status,count(*) FROM urls GROUP BY status'))
    assert 1 <= statuses['ok'] <= 2
    assert statuses.get('in_progress', 0) == 0
    done = set(calls)
    restored = db.next_batch('a.cn', 10, 2000)
    assert {r['rank'] for r in restored} == set(range(10)) - done
    db.conn.close()


def test_robots_unavailable_defers_only_current_url():
    from r2ai.zh_sample.crawl import defer_robots_row
    from vicrawl.state import StateDB, SCHEMA
    import sqlite3
    db = object.__new__(StateDB)
    db.conn = sqlite3.connect(':memory:', isolation_level=None)
    db.conn.executescript(SCHEMA)
    groups = [{'url': f'https://a.cn/{i}', 'url_norm': f'a.cn/{i}', 'domain': 'a.cn', 'doc_ids': [i], 'rank': i} for i in range(3)]
    db.add_urls(groups)
    row = db.next_batch('a.cn', 1, 10)[0]
    defer_robots_row(db, row, 1800)
    result = db.conn.execute('SELECT status,attempts,next_try_at FROM urls ORDER BY rank').fetchall()
    assert result == [('pending',0,1800.0),('pending',0,None),('pending',0,None)]
    db.conn.close()


def test_candidate_stats_can_be_json_serialized():
    import json
    import numpy as np
    from r2ai.zh_sample.evaluate import candidate_stats_json
    st = {'seconds': .5, 'n_candidates': np.array([200,300]), 'n_docs': np.array([150,160])}
    assert json.loads(json.dumps(candidate_stats_json(st))) == {'seconds': .5, 'n_candidates':[200,300], 'n_docs':[150,160]}


def test_geo_refresh_does_not_burst_http():
    import asyncio
    import io
    import time
    from types import SimpleNamespace
    from r2ai.zh_sample.crawl import PoliteClient
    from vicrawl.tuner import Tuner
    starts = []
    class Net:
        def __init__(self):
            self.lock, self.ready = asyncio.Lock(), False
        async def check(self):
            async with self.lock:
                if not self.ready:
                    await asyncio.sleep(.3)
                    self.ready = True
    class Response:
        status_code, headers = 200, {}
        async def __aenter__(self):
            starts.append(time.monotonic())
            return self
        async def __aexit__(self, *args): pass
        async def aiter_bytes(self):
            yield b'body'
    class Transport:
        def stream(self, *args): return Response()
    async def run():
        c = object.__new__(PoliteClient)
        c.domain, c.net, c.stop, c.audit = 'a.cn', Net(), asyncio.Event(), io.StringIO()
        c.tuner = Tuner(4, 2, rate=4)
        c.next_at, c.pace_lock, c.client = 0, asyncio.Lock(), Transport()
        await asyncio.gather(c.hop('https://a.cn/1','page'), c.hop('https://a.cn/2','page'))
    asyncio.run(run())
    assert starts[1] - starts[0] >= .24


def test_pending_retry_is_not_finished():
    import sqlite3
    from vicrawl.state import StateDB, SCHEMA
    from r2ai.zh_sample.crawl import next_retry_at
    db = object.__new__(StateDB)
    db.conn = sqlite3.connect(':memory:', isolation_level=None)
    db.conn.executescript(SCHEMA)
    db.add_urls([{'url':'https://a.cn/a','url_norm':'a.cn/a','domain':'a.cn','doc_ids':[1],'rank':60}])
    db.conn.execute("UPDATE urls SET status='network_error',next_try_at=1000,attempts=1")
    assert next_retry_at(db,'a.cn',None) == 1000
    assert next_retry_at(db,'a.cn',50) is None
    db.conn.execute('UPDATE urls SET next_try_at=NULL,attempts=3')
    assert next_retry_at(db,'a.cn',None) is None
    db.conn.close()


def test_zh_slow_server_rate_cap():
    from r2ai.zh_sample.crawl import ZhTuner
    tuner = ZhTuner(4, 2, rate=3)
    for _ in range(500):
        tuner.record(4,200,False)
    assert tuner.p50 == 4
    assert tuner.rate <= 1
    assert tuner.cap <= 1
    assert tuner.interval() >= 1
