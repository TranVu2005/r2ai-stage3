import json
from pathlib import Path

import polars as pl
import pytest

from r2ai.zh_v2 import direct


def _inventory(tmp_path):
    corpus = tmp_path / 'corpus.parquet'
    pl.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'url': [
            'http://www.120ask.com/question/1.htm',
            'https://120ask.com/question/1.htm',
            'https://article.iiyi.com/detail/3.html',
            'https://bingli.iiyi.com/case/4.html',
            'https://other.example/5',
        ],
        'lang': ['zh', 'zh', 'zh', 'zh', 'zh'],
        'host': ['120ask.com', '120ask.com', 'article.iiyi.com', 'bingli.iiyi.com', 'other.example'],
    }).write_parquet(corpus)
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    seg = chunks / 'seg00100-existing'
    seg.mkdir()
    pl.DataFrame({'doc_id': [10], 'doc_ids_v2': [[2]]}).write_parquet(seg / 'docs.parquet')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'segments': [{'name': 'seg00100-existing', 'docs_v2': 1}]}), encoding='utf-8')
    return corpus, manifest, chunks


def test_build_targets_only_missing_live_ids_from_the_v2_host_allowlist(tmp_path):
    corpus, manifest, chunks = _inventory(tmp_path)

    targets = direct.build_targets(corpus, manifest, chunks)

    assert {d for target in targets for d in target['doc_ids']} == {1, 3, 4}
    assert {target['domain'] for target in targets} == {'120ask.com', 'article.iiyi.com', 'bingli.iiyi.com'}
    # The chosen request URL itself must belong to an ID that is still missing.
    by_id = pl.read_parquet(corpus).filter(pl.col('id').is_in([1, 3, 4])).select('url').to_series().to_list()
    assert all(target['url'] in by_id for target in targets)
    assert all(set(target['doc_ids']) <= {1, 3, 4} for target in targets)


def test_client_options_pin_user_agent_disable_env_proxies_and_redirects():
    options = direct.client_options()

    assert options['headers']['User-Agent'] == 'r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)'
    assert options['trust_env'] is False
    assert options['follow_redirects'] is False


@pytest.mark.parametrize(('sample', 'expected'), [(0, 1.0), (1, 1.1), (.5, 1.05)])
def test_request_interval_is_at_least_one_second_with_positive_jitter(sample, expected):
    assert direct.interval_seconds(sample, jitter_s=.1) == pytest.approx(expected)


def test_backoff_doubles_and_honors_retry_after_beyond_the_cap():
    assert [direct.backoff_seconds(n, cap_s=300) for n in (1, 2, 3, 10)] == [1, 2, 4, 300]
    assert direct.backoff_seconds(1, retry_after_s=420, cap_s=300) == 420


def test_error_window_waits_for_500_then_stops_only_above_twenty_percent():
    at_limit = direct.RollingErrors()
    for error in [False] * 400 + [True] * 100:
        stopped = at_limit.add(error)
    assert stopped is False
    assert at_limit.total == 500

    above_limit = direct.RollingErrors()
    for error in [False] * 399 + [True] * 101:
        stopped = above_limit.add(error)
    assert stopped is True
    assert above_limit.total == 500


def test_challenge_detection_never_treats_captcha_html_as_extractable_text():
    assert direct.is_bot_challenge('<html><title>安全验证</title><body>请完成人机验证</body></html>')
    assert direct.classify_response(200, {'content-type': 'text/html'},
                                    '<html><title>安全验证</title><body>请完成人机验证</body></html>')[0] == 'bot_challenge'
    assert not direct.is_bot_challenge('<html><title>医学问题</title><body>普通问答内容</body></html>')


def test_retry_after_accepts_seconds_and_http_date():
    from datetime import datetime, timezone

    now = datetime(2026, 10, 17, tzinfo=timezone.utc).timestamp()
    assert direct.parse_retry_after('12', now=now) == 12
    assert direct.parse_retry_after('Sat, 17 Oct 2026 00:00:20 GMT', now=now) == 20
    assert direct.parse_retry_after('invalid', now=now) is None


@pytest.mark.parametrize(('code', 'state'), [(403, 'blocked_4xx'), (429, 'blocked_4xx'), (503, 'dead_origin'), (302, 'redirect_disallowed')])
def test_block_statuses_are_classified_before_content_type(code, state):
    assert direct.classify_response(code, {'content-type': 'text/plain'}, '')[0] == state


class _Response:
    def __init__(self, status_code, content, headers):
        self.status_code, self.content, self.headers = status_code, content, headers


class _Client:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _redirect_target():
    url = 'https://120ask.com/question/1.htm'
    return {'url_norm': direct.normalize_url(url), 'url': url, 'domain': '120ask.com', 'doc_ids': [1], 'rank': 0}


def test_fetch_once_does_not_follow_redirects_or_save_redirect_body():
    client = _Client(_Response(302, b'<html>go</html>', {'Location': 'https://evil.example/'}))

    result = direct.fetch_once(client, _redirect_target(), clock=lambda: 1.0)

    assert client.calls == [(_redirect_target()['url'], {'follow_redirects': False})]
    assert result['record']['status'] == 'redirect_disallowed'
    assert result['record']['html'] is None


def test_run_closes_raw_shard_before_recording_resume_pointer(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    raw = tmp_path / 'raw'
    state = tmp_path / 'state'
    for name, value in (('RUN', run), ('RAW', raw), ('STATE', state), ('DB', state / 'crawl.db'),
                        ('LOCK_FILE', run / 'direct.lock'), ('STOP_FILE', run / 'stop_reason.json'),
                        ('TARGETS_FILE', run / 'targets.jsonl'), ('INVENTORY_FILE', run / 'inventory.json'),
                        ('EVENTS_FILE', run / 'events.jsonl'), ('HEARTBEAT_FILE', run / 'heartbeat.json')):
        monkeypatch.setattr(direct, name, value)
    corpus, manifest, chunks = _inventory(tmp_path)
    direct.prepare(corpus_path=corpus, manifest_path=manifest, chunks_root=chunks,
                   db_path=direct.DB, run_root=direct.RUN, scan_outputs=False)
    url = 'https://120ask.com/question/1.htm'
    html = ('<html><head><title>问题详情</title></head><body><h1>问题详情</h1>'
            '<p>' + ('这是医学问答内容。' * 80) + '</p></body></html>').encode()
    client = _Client(_Response(200, html, {'content-type': 'text/html; charset=utf-8'}))

    result = direct.run(db_path=direct.DB, max_requests=1, sleep=lambda _: None, client=client,
                        clock=lambda: 10.0)

    import sqlite3
    c = sqlite3.connect(direct.DB)
    row = c.execute('SELECT status,shard,source,provenance_timestamp FROM urls WHERE url_norm=?',
                    (direct.normalize_url(url),)).fetchone()
    c.close()
    assert result['requests_this_run'] == 1
    assert row[0] == 'ok'
    assert row[1] and (raw / '120ask.com' / Path(row[1]).name).is_file()
    assert row[2] == 'direct'
    assert row[3] == 10.0


def test_run_applies_host_backoff_to_blocked_url(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    raw = tmp_path / 'raw'
    state = tmp_path / 'state'
    for name, value in (('RUN', run), ('RAW', raw), ('STATE', state), ('DB', state / 'crawl.db'),
                        ('LOCK_FILE', run / 'direct.lock'), ('STOP_FILE', run / 'stop_reason.json'),
                        ('TARGETS_FILE', run / 'targets.jsonl'), ('INVENTORY_FILE', run / 'inventory.json'),
                        ('EVENTS_FILE', run / 'events.jsonl'), ('HEARTBEAT_FILE', run / 'heartbeat.json')):
        monkeypatch.setattr(direct, name, value)
    corpus, manifest, chunks = _inventory(tmp_path)
    direct.prepare(corpus_path=corpus, manifest_path=manifest, chunks_root=chunks,
                   db_path=direct.DB, run_root=direct.RUN, scan_outputs=False)
    client = _Client(_Response(429, b'', {'retry-after': '600', 'content-type': 'text/plain'}))

    result = direct.run(db_path=direct.DB, max_requests=1, sleep=lambda _: None, client=client,
                        clock=lambda: 10.0)

    import sqlite3
    c = sqlite3.connect(direct.DB)
    row = c.execute('SELECT status,attempts,next_try_at FROM urls WHERE url_norm=?',
                    (direct.normalize_url('https://120ask.com/question/1.htm'),)).fetchone()
    host = c.execute('SELECT block_streak,next_request_at FROM hosts WHERE domain=?', ('120ask.com',)).fetchone()
    c.close()
    assert result['requests_this_run'] == 1
    assert row[0] == 'pending' and row[1] == 1
    assert row[2] == 610.0
    assert host == (1, 610.0)


def test_resume_requeues_legacy_network_error_and_records_new_success(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    raw = tmp_path / 'raw'
    state = tmp_path / 'state'
    for name, value in (('RUN', run), ('RAW', raw), ('STATE', state), ('DB', state / 'crawl.db'),
                        ('LOCK_FILE', run / 'direct.lock'), ('STOP_FILE', run / 'stop_reason.json'),
                        ('TARGETS_FILE', run / 'targets.jsonl'), ('INVENTORY_FILE', run / 'inventory.json'),
                        ('EVENTS_FILE', run / 'events.jsonl'), ('HEARTBEAT_FILE', run / 'heartbeat.json')):
        monkeypatch.setattr(direct, name, value)
    monkeypatch.setattr(direct, '_manage_ingest', lambda *args, **kwargs: {'assigned': [], 'driver': {'state': 'idle'}})
    corpus, manifest, chunks = _inventory(tmp_path)
    direct.prepare(corpus_path=corpus, manifest_path=manifest, chunks_root=chunks,
                   db_path=direct.DB, run_root=direct.RUN, scan_outputs=False)
    url = 'https://120ask.com/question/1.htm'
    connection = __import__('sqlite3').connect(direct.DB)
    connection.execute("UPDATE urls SET status='network_error',attempts=1,reason='old offline probe' WHERE url_norm=?",
                       (direct.normalize_url(url),))
    connection.commit()
    connection.close()
    html = ('<html><head><title>问题详情</title></head><body><h1>问题详情</h1>'
            '<p>' + ('这是医学问答内容。' * 80) + '</p></body></html>').encode()
    client = _Client(_Response(200, html, {'content-type': 'text/html; charset=utf-8'}))

    result = direct.run(db_path=direct.DB, domains=('120ask.com',), max_requests=1,
                        sleep=lambda _: None, client=client, clock=lambda: 12.0)

    connection = __import__('sqlite3').connect(direct.DB)
    row = connection.execute('SELECT status,attempts,shard FROM urls WHERE url_norm=?',
                             (direct.normalize_url(url),)).fetchone()
    connection.close()
    assert result['requests_this_run'] == 1
    assert client.calls and row[0] == 'ok' and row[1] == 2 and row[2]


def test_success_checkpoint_resumes_to_raw_without_refetching(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    raw = tmp_path / 'raw'
    state = tmp_path / 'state'
    for name, value in (('RUN', run), ('RAW', raw), ('STATE', state), ('DB', state / 'crawl.db'),
                        ('LOCK_FILE', run / 'direct.lock'), ('STOP_FILE', run / 'stop_reason.json'),
                        ('TARGETS_FILE', run / 'targets.jsonl'), ('INVENTORY_FILE', run / 'inventory.json'),
                        ('EVENTS_FILE', run / 'events.jsonl'), ('HEARTBEAT_FILE', run / 'heartbeat.json')):
        monkeypatch.setattr(direct, name, value)
    monkeypatch.setattr(direct, '_manage_ingest', lambda *args, **kwargs: {'assigned': [], 'driver': {'state': 'idle'}})
    corpus, manifest, chunks = _inventory(tmp_path)
    direct.prepare(corpus_path=corpus, manifest_path=manifest, chunks_root=chunks,
                   db_path=direct.DB, run_root=direct.RUN, scan_outputs=False)
    target = _redirect_target()
    target['doc_ids'] = [1, 2]
    target['rank'] = 0
    html = ('<html><head><title>问题详情</title></head><body><h1>问题详情</h1>'
            '<p>' + ('这是医学问答内容。' * 80) + '</p></body></html>')
    record = {'url_norm': target['url_norm'], 'url': target['url'], 'final_url': target['url'],
              'domain': target['domain'], 'doc_ids': target['doc_ids'], 'status': 'ok', 'reason': None,
              'http_status': 200, 'fetched_at': 11.0, 'content_type': 'text/html', 'encoding': 'utf-8',
              'latency': .2, 'html': html, 'source': 'direct', 'provenance_timestamp': 11.0}
    connection, _ = direct._ensure_prepared(direct.DB)
    direct._commit_outcome(connection, target, {'record': record, 'is_error': False, 'retry_after_s': None, 'body_bytes': 20},
                           when=11.0)
    assert connection.execute("SELECT status,staged_record IS NOT NULL FROM urls WHERE url_norm=?",
                              (target['url_norm'],)).fetchone() == ('staged', 1)
    connection.close()

    class NoRequestClient:
        def get(self, *args, **kwargs):
            raise AssertionError('a staged success must be restored without another request')

    result = direct.run(db_path=direct.DB, domains=('120ask.com',), client=NoRequestClient(),
                        sleep=lambda _: None, clock=lambda: 12.0)

    connection = __import__('sqlite3').connect(direct.DB)
    row = connection.execute('SELECT status,shard,staged_record FROM urls WHERE url_norm=?',
                             (target['url_norm'],)).fetchone()
    connection.close()
    assert result['requests_this_run'] == 0
    assert row[0] == 'ok' and row[1] and row[2] is None
    assert (raw / '120ask.com' / Path(row[1]).name).is_file()


def test_ingest_batches_freeze_groups_at_the_10k_v2_id_threshold(tmp_path, monkeypatch):
    state = tmp_path / 'state'
    monkeypatch.setattr(direct, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(direct, 'RAW', tmp_path / 'raw')
    monkeypatch.setattr(direct, 'STATE', state)
    monkeypatch.setattr(direct, 'DB', state / 'crawl.db')
    monkeypatch.setattr(direct, 'EVENTS_FILE', tmp_path / 'run' / 'events.jsonl')
    monkeypatch.setattr(direct, 'INGEST_BATCH_DOCS', 4)
    c = direct._connect(direct.DB)
    rows = [
        ('https://120ask.com/q/1', '[1,2]', '120ask.com', 'ok', 'shard-a.jsonl.zst', 0),
        ('https://120ask.com/q/2', '[3]', '120ask.com', 'thin', 'shard-a.jsonl.zst', 1),
        ('https://120ask.com/q/3', '[4,5]', '120ask.com', 'ok', 'shard-b.jsonl.zst', 2),
        ('https://120ask.com/q/4', '[6]', '120ask.com', 'ok', 'shard-b.jsonl.zst', 3),
    ]
    with c:
        c.executemany('INSERT INTO urls(url_norm,url,doc_ids,domain,rank,status,shard) VALUES (?,?,?,?,?,?,?)',
                      [(url, url, ids, domain, rank, status, shard) for url, ids, domain, status, shard, rank in rows])

    assigned = direct._assign_ingest_batches(c, ('120ask.com',))

    assert assigned == [{'batch': 'zhv2d-00001', 'n_urls': 3, 'n_v2_ids': 5,
                         'crawl_db': str(state / 'ingest-source-zhv2d-00001.db'), 'state': 'queued'}]
    source = __import__('sqlite3').connect(assigned[0]['crawl_db'])
    assert source.execute('SELECT count(*) FROM urls').fetchone()[0] == 3
    assert source.execute('SELECT url_norm,domain,status,shard FROM urls ORDER BY url_norm').fetchall() == [
        ('https://120ask.com/q/1', '120ask.com', 'ok', 'shard-a.jsonl.zst'),
        ('https://120ask.com/q/2', '120ask.com', 'thin', 'shard-a.jsonl.zst'),
        ('https://120ask.com/q/3', '120ask.com', 'ok', 'shard-b.jsonl.zst')]
    source.close()
    assert direct._assign_ingest_batches(c, ('120ask.com',)) == []
    assert direct._assign_ingest_batches(c, ('120ask.com',), finalize=True)[0]['n_v2_ids'] == 1
    from r2ai.zh_v2 import ingest
    monkeypatch.setattr(ingest, 'launch', lambda batch, crawl_db=None: {'pid': 4321, 'command': 'fake'})
    advanced = direct._advance_ingest(c)
    assert advanced['state'] == 'running' and advanced['batch'] == 'zhv2d-00001'
    assert c.execute('SELECT state,pid,launch_attempts FROM ingest_batches WHERE batch=?',
                     ('zhv2d-00001',)).fetchone() == ('running', 4321, 1)
    c.close()


def test_due_selection_does_not_starve_new_urls_after_a_retry(tmp_path, monkeypatch):
    state = tmp_path / 'state'
    monkeypatch.setattr(direct, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(direct, 'RAW', tmp_path / 'raw')
    monkeypatch.setattr(direct, 'STATE', state)
    monkeypatch.setattr(direct, 'DB', state / 'crawl.db')
    c = direct._connect(direct.DB)
    with c:
        c.executemany('INSERT INTO hosts(domain) VALUES (?)', [('article.iiyi.com',), ('bingli.iiyi.com',)])
        c.executemany('INSERT INTO urls(url_norm,url,doc_ids,domain,rank,status,attempts) VALUES (?,?,?,?,?,?,?)', [
            ('a0', 'https://article.iiyi.com/0', '[1]', 'article.iiyi.com', 0, 'pending', 1),
            ('a1', 'https://article.iiyi.com/1', '[2]', 'article.iiyi.com', 1, 'pending', 0),
            ('b0', 'https://bingli.iiyi.com/0', '[3]', 'bingli.iiyi.com', 0, 'pending', 0),
        ])

    row = direct._next_target(c, 100.0, 'article.iiyi.com', ('article.iiyi.com', 'bingli.iiyi.com'))

    assert row[0] == 'a1'
    c.close()


def test_status_reads_a_frozen_queue_before_ingest_schema_migration(tmp_path, monkeypatch):
    db = tmp_path / 'legacy-crawl.db'
    c = __import__('sqlite3').connect(db)
    c.executescript('''
        CREATE TABLE urls(url_norm TEXT, domain TEXT, status TEXT);
        CREATE TABLE requests(request_id INTEGER PRIMARY KEY, is_error INTEGER, status TEXT);
        CREATE TABLE hosts(domain TEXT, block_streak INTEGER, next_request_at REAL, last_block_at REAL, last_status TEXT);
    ''')
    c.execute("INSERT INTO urls VALUES ('u1','120ask.com','pending')")
    c.execute("INSERT INTO hosts VALUES ('120ask.com',0,NULL,NULL,NULL)")
    c.commit()
    c.close()
    monkeypatch.setattr(direct, 'STOP_FILE', tmp_path / 'stop.json')
    monkeypatch.setattr(direct, 'HEARTBEAT_FILE', tmp_path / 'heartbeat.json')

    result = direct.status(db_path=db)

    assert result['statuses'] == {'pending': 1}
    assert result['ingest_batches'] == []


def test_ingest_provenance_sidecars_verify_v2_membership_and_good_text(tmp_path, monkeypatch):
    from r2ai.zh_v2 import ingest, index as zindex

    crawl_db = tmp_path / 'source.db'
    connection = __import__('sqlite3').connect(crawl_db)
    connection.execute('CREATE TABLE urls(url_norm TEXT, fetched_at REAL, doc_ids TEXT, source TEXT, status TEXT)')
    connection.execute('INSERT INTO urls VALUES (?,?,?,?,?)',
                       ('https://120ask.com/q/1', 10.0, '[11,12]', 'direct', 'ok'))
    connection.commit()
    connection.close()
    docs = tmp_path / 'docs' / 'batch-1' / '120ask.com' / 'docs.parquet'
    docs.parent.mkdir(parents=True)
    pl.DataFrame({'doc_ids': [[11, 12]], 'url_norm': ['https://120ask.com/q/1'], 'fetched_at': [10.0],
                  'body': ['医学问答内容。' * 60], 'status': ['ok'], 'n_tokens_bge_m3': [60]}).write_parquet(docs)
    chunk_docs = tmp_path / 'chunks' / 'seg00100' / 'docs.parquet'
    chunk_docs.parent.mkdir(parents=True)
    pl.DataFrame({'doc_id': [11], 'doc_ids_group': [[11, 12]]}).write_parquet(chunk_docs)
    roots = ingest.V2Roots(raw=tmp_path / 'raw', crawl_db=crawl_db, docs=tmp_path / 'docs',
                           chunks=tmp_path / 'chunks', index=tmp_path / 'index', run=tmp_path / 'run',
                           state=tmp_path / 'ingest-state', forbidden=())
    monkeypatch.setattr(ingest.zcrawl, 'DB', crawl_db)
    monkeypatch.setattr(zindex.scope, 'v2_zh', lambda: pl.DataFrame({'id': [11, 12]}))
    monkeypatch.setattr(ingest.base_ingest, 'segment_rows', lambda _: [
        {'name': 'seg00100', 'batch': 'batch-1', 'status': 'active'}])

    result = ingest.write_provenance(roots, 'batch-1')

    assert result['extracted_doc_groups'] == 1
    assert result['empty_docs'] == 0
    assert result['good_text_docs'] == 1
    assert result['v2_ids'] == 2
    assert result['provenance_schema'] == ['doc_id:int64', 'doc_ids_v2:list[int64]', 'url_norm:string',
                                           'source:string', 'fetched_at:float64']
    assert (tmp_path / 'docs' / 'batch-1' / 'provenance.parquet').is_file()
    sidecar = pl.read_parquet(tmp_path / 'chunks' / 'seg00100' / 'provenance.parquet')
    assert sidecar['doc_ids_v2'].to_list() == [[11, 12]]
    assert sidecar['source'].to_list() == ['direct']
