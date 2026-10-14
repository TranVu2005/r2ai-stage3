"""cc_longchau fetch/WARC parsing: no network, a mock transport serves synthetic gzip WARC records by Range."""
import gzip
import json
import re

import httpx
import pandas as pd
import pytest

from r2ai.cc_coverage.net import Blocked, Ledger
from r2ai.cc_longchau import fetch as F
from r2ai.cc_longchau import warc as W
from r2ai.cc_longchau.common import Roots
from vicrawl.shards import list_shards, read_shard

HTML = '<html><head><meta charset="utf-8"><title>Bài viết</title></head><body><h1>Tiêu đề</h1><p>' + 'Nội dung y khoa. ' * 30 + '</p></body></html>'


def record(url, status=200, body=HTML, rtype='response', headers='Content-Type: text/html; charset=utf-8\r\n', chunked=False):
    payload = body.encode('utf-8')
    if chunked:
        payload = f'{len(payload):x}\r\n'.encode() + payload + b'\r\n0\r\n\r\n'
        headers += 'Transfer-Encoding: chunked\r\n'
    http = f'HTTP/1.1 {status} OK\r\n{headers}\r\n'.encode() + payload
    warc = (f'WARC/1.0\r\nWARC-Type: {rtype}\r\nWARC-Target-URI: {url}\r\nWARC-Date: 2026-03-14T13:37:56Z\r\n'
            f'WARC-Record-ID: <urn:uuid:1>\r\nContent-Length: {len(http)}\r\n\r\n').encode()
    return gzip.compress(warc + http + b'\r\n\r\n', mtime=0)


def test_parse_response_plain_and_chunked():
    r = W.parse_response(W.gunzip_member(record('https://x/a')))
    assert r.status == 200 and r.body.decode() == HTML and r.target_uri == 'https://x/a'
    assert W.parse_response(W.gunzip_member(record('https://x/a', chunked=True))).body.decode() == HTML


def test_parse_rejects_non_response_and_truncated_gzip():
    with pytest.raises(W.WarcError) as e:
        W.parse_response(W.gunzip_member(record('https://x/a', rtype='revisit')))
    assert e.value.reason == 'not_response'
    with pytest.raises(W.WarcError) as e:
        W.gunzip_member(record('https://x/a')[:-10])
    assert e.value.reason in ('gzip_truncated', 'gzip_error')
    with pytest.raises(W.WarcError) as e:
        W.gunzip_member(record('https://x/a') + record('https://x/b'))
    assert e.value.reason == 'gzip_extra_member'


class Cc:
    """Fake data.commoncrawl.org: one WARC file made of the given records; counts requests, can inject failures."""

    def __init__(self, records, fail=()):
        self.blob, self.spans, self.fail, self.calls = b'', [], list(fail), 0
        for rec in records:
            self.spans.append((len(self.blob), len(rec)))
            self.blob += rec

    def __call__(self, request):
        self.calls += 1
        if self.fail:
            code = self.fail.pop(0)
            if code:
                return httpx.Response(code, headers={'retry-after': '0'} if code in (429, 503) else {})
        a, b = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.headers['range']).groups())
        return httpx.Response(206, content=self.blob[a:b + 1])


def roots_for(tmp_path, urls, spans, crawl='CC-MAIN-2026-12'):
    corpus = pd.DataFrame({'id': [10 + i for i in range(len(urls))], 'url': urls})
    cov = pd.DataFrame({'url_id': corpus.id, 'prefix': '/bai-viet', 'n_snapshots': 1, 'crawl': crawl,
                        'fetch_time': pd.Timestamp('2026-03-14 13:37:56', tz='UTC'), 'warc_filename': 'crawl-data/x.warc.gz',
                        'warc_record_offset': [float(s[0]) for s in spans], 'warc_record_length': [float(s[1]) for s in spans],
                        'after_corpus': False})
    corpus.to_parquet(tmp_path / 'corpus.parquet')
    cov.to_parquet(tmp_path / 'cov.parquet')
    return Roots(raw=tmp_path / 'raw', docs=tmp_path / 'docs', chunks=tmp_path / 'chunks', index=tmp_path / 'index',
                 run=tmp_path / 'run', state=tmp_path / 'state', corpus=tmp_path / 'corpus.parquet', coverage=tmp_path / 'cov.parquet')


def client_for(roots, cc):
    ledger = Ledger(roots.run / 'ledger.json', max_bytes=10 ** 9, max_requests=1000)
    return F.RangeClient(ledger=ledger, log_path=roots.run / 'req.jsonl', rps=1000, transport=httpx.MockTransport(cc), sleep=lambda s: None)


U = ['https://nhathuoclongchau.com.vn/bai-viet/a', 'https://nhathuoclongchau.com.vn/bai-viet/b', 'https://nhathuoclongchau.com.vn/benh/c',
     'https://nhathuoclongchau.com.vn/bai-viet/d']


def test_fetch_records_provenance_and_resume(tmp_path):
    recs = [record(U[0]), record(U[1], status=404), record('https://nhathuoclongchau.com.vn/other', body=HTML), record(U[3], chunked=True)]
    cc = Cc(recs)
    roots = roots_for(tmp_path, U, cc.spans)
    s = F.run(roots, workers=2, shard_size=2, client=client_for(roots, cc))
    assert s['by_status'] == {'ok': 2, 'bad_http_status': 1, 'url_mismatch': 1}
    got = [r for p in list_shards(roots.raw) for r in read_shard(p)]
    assert sorted(r['doc_ids'][0] for r in got) == [10, 13]
    r = next(r for r in got if r['doc_ids'] == [10])
    assert r['html'] == HTML and r['url'] == U[0] and r['domain'] == 'nhathuoclongchau.com.vn'
    cmeta = r['cc']
    assert (cmeta['crawl'], cmeta['warc_filename'], cmeta['warc_offset'], cmeta['warc_length']) == ('CC-MAIN-2026-12', 'crawl-data/x.warc.gz', 0, len(recs[0]))
    assert cmeta['warc_date'] == '2026-03-14T13:37:56Z' and len(cmeta['record_sha256']) == 64
    calls = cc.calls
    s2 = F.run(roots, workers=2, shard_size=2, client=client_for(roots, cc))
    assert cc.calls == calls and s2['run_urls'] == 0                         # resume: nothing final is requested again


def test_backoff_on_429_503_then_ok_and_403_stops(tmp_path):
    cc = Cc([record(U[0]), record(U[1])], fail=[429, 503, None])
    roots = roots_for(tmp_path, U[:2], cc.spans)
    s = F.run(roots, workers=1, client=client_for(roots, cc))
    assert s['by_status'] == {'ok': 2} and cc.calls == 4                     # 429, 503, ok (first URL), ok (second URL)
    cc = Cc([record(U[0])], fail=[403])
    (tmp_path / 'b').mkdir()
    roots = roots_for(tmp_path / 'b', U[:1], cc.spans)
    s = F.run(roots, workers=1, client=client_for(roots, cc))
    assert s['stopped'].startswith('Blocked') and s['by_status'] == {}
    with pytest.raises(Blocked):
        client_for(roots, Cc([], fail=[403])).get_range('https://x/y', 0, 10)


def test_wrong_length_and_guard(tmp_path):
    recs = [record(U[0])]
    cc = Cc(recs)
    roots = roots_for(tmp_path, U[:1], [(0, len(recs[0]) + 5)])             # coverage claims more bytes than the file holds
    s = F.run(roots, client=client_for(roots, cc))
    assert s['by_status'] == {'range_mismatch': 1}
    with pytest.raises(ValueError):
        roots.guard(tmp_path.parent / 'elsewhere')


def test_targets_reject_url_outside_corpus(tmp_path):
    cc = Cc([record(U[0])])
    roots = roots_for(tmp_path, U[:1], cc.spans)
    cov = pd.read_parquet(roots.coverage)
    cov.loc[0, 'url_id'] = 99999
    cov.to_parquet(roots.coverage)
    with pytest.raises(ValueError):
        F.load_targets(roots)


# ---------------------------------------------------------------- VS-cc-A planning / rendering
import io  # noqa: E402
import json  # noqa: E402

from r2ai.cc_longchau import append_cc as A  # noqa: E402
from r2ai.zh_sample.append import verify_gate  # noqa: E402


def test_plan_replace_vs_union():
    cc = {101, 102, 103, 104}
    slug = [101, 7, 102, 8]                       # 101, 102 have CC text, 7, 8 do not
    ranked = [103, 101, 104, 55]                  # reranker order; 55 is not a CC doc but is never ranked in practice
    tail, picks, removed = A.plan_query('VS-cc-A', [1, 2], slug, cc, ranked)
    assert removed == [101, 102] and picks == [103, 101] and tail == [7, 8, 103, 101]
    tail, picks, removed = A.plan_query('VS-cc-A-union', [1, 2], slug, cc, ranked)
    assert picks == [103, 101] and tail == slug + [103] and removed == []      # same picks as VS-cc-A; 101 is already a slug id
    tail, picks, _ = A.plan_query('VS-cc-A', [103], slug, cc, ranked)       # a doc already in RD150 is never picked
    assert 103 not in tail


def test_render_keeps_rd150_bytes():
    rows = [{'id': 1, 'relevant_docs': [1, 2], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'x'}]},
            {'id': 2, 'relevant_docs': [3], 'relevant_chunks': []}]
    base = json.dumps(rows, ensure_ascii=False)
    slug = {1: [101, 7], 2: [102]}
    ranking = {1: ([103, 101], {103: 2.0, 101: 1.0}), 2: ([101], {101: 0.5})}
    groups = {101: [101], 102: [102], 103: [103]}
    buf = io.StringIO()
    sel = A.render(base, 'VS-cc-A', slug, set(groups), ranking, groups, buf.write)
    out = json.loads(buf.getvalue())
    assert out[0]['relevant_docs'] == [1, 2, 7, 103] and out[1]['relevant_docs'] == [3, 101]
    assert out[0]['relevant_chunks'] == rows[0]['relevant_chunks']
    assert verify_gate(base, buf.getvalue(), sel)['queries'] == 2


def test_priority_ids_fetched_first(tmp_path):
    recs = [record(u) for u in U]
    cc = Cc(recs)
    roots = roots_for(tmp_path, U, cc.spans)
    F.run(roots, workers=1, shard_size=2, limit=2, client=client_for(roots, cc), priority_ids={12, 13})
    db = F.connect(roots)
    assert sorted(r[0] for r in db.execute("select url_id from fetch")) == [12, 13]


# ---------------------------------------------------------------- VS-cc-B merge helpers
import numpy as np  # noqa: E402

from r2ai.cc_longchau import merge_b as MB  # noqa: E402


def test_merge_query_is_exact_top_k_with_tie_rule():
    vi = (np.array([1, 2, 3, 4]), np.array([.9, .8, .8, .1], np.float32), np.array([0, .5, .6, .7], np.float32))
    cc = (np.array([MB.N_VI + 0, MB.N_VI + 1]), np.array([.8, .95], np.float32), np.array([.9, 0], np.float32))
    ids, de, sp = MB.merge_query(vi, cc, k=2)
    # dense top-2 = cc1 (.95), vi1 (.9); sparse top-2 = cc0 (.9), vi4 (.7); union keeps both scores of every member
    assert ids.tolist() == [1, 4, MB.N_VI, MB.N_VI + 1]
    assert de.tolist() == pytest.approx([.9, .1, .8, .95]) and sp.tolist() == pytest.approx([0, .7, .9, 0])
    ids, _, _ = MB.merge_query(vi, cc, k=3)          # dense tie .8 (vi2, vi3, cc0): larger global id first -> cc0, vi3 ahead of vi2
    assert set(ids.tolist()) >= {1, 3, MB.N_VI + 1, MB.N_VI}


def test_hybrid_matches_reference_and_renorm_range_changes_order():
    de, sp = np.array([1., .5, 0.], np.float32), np.array([0., .5, 1.], np.float32)
    assert MB.hybrid(de, sp).tolist() == pytest.approx((.7 * (de - 0) / 1 + .3 * sp).tolist())
    wide = MB.hybrid(de, sp, (0., 1., 0., 10.))           # a wider sparse range (as when CC chunks join) shrinks the sparse weight
    assert wide.argmax() == 0 and MB.hybrid(de, sp).argmax() == 0


def test_cached_reranker_refuses_unknown_pairs_in_gate_mode():
    rr = MB.CachedReranker({(1, 5): 0.25}, fac=None, allow_new=False)
    assert rr.score(1, 'q', [5]) == {5: 0.25} and rr.hit == 1
    with pytest.raises(KeyError):
        rr.score(1, 'q', [5, 6])


def test_member_ids_filter_drops_urls_outside_v2(tmp_path):
    cc = Cc([record(u) for u in U])
    roots = roots_for(tmp_path, U, cc.spans)
    pd.DataFrame({'id': [10, 12]}).to_parquet(tmp_path / 'v2.parquet')
    assert sorted(F.load_targets(roots, tmp_path / 'v2.parquet').url_id) == [10, 12]
    assert len(F.load_targets(roots)) == 4


# ---- production fetch: auto rate, Retry-After, jitter, stop, priority groups, heartbeat / status -------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_autorate_halves_on_503_and_raises_when_clean(tmp_path):
    from r2ai.cc_longchau.throttle import AutoRate
    clk = FakeClock()
    r = AutoRate(0.6, path=tmp_path / 'rate.json', clock=clk)
    for _ in range(4):
        r.record(503); clk.t += 10
    assert r.rps == 0.6                                    # fewer than MIN_EVENTS attempts: no verdict yet
    r.record(503); clk.t += 10
    assert r.rps == 0.3                                    # >30% of 5 attempts throttled in 10 min
    for _ in range(5):
        r.record(503); clk.t += 10
    assert r.rps == 0.15
    for _ in range(8):
        for _ in range(5):
            r.record(503); clk.t += 10
    assert r.rps == 0.1                                    # floor
    assert json.loads((tmp_path / 'rate.json').read_text())['rps'] == 0.1
    assert AutoRate(0.6, path=tmp_path / 'rate.json', clock=clk).rps == 0.1      # survives a restart
    assert AutoRate(0.05, path=tmp_path / 'rate.json', clock=clk).rps == 0.1     # never below the floor
    r2 = AutoRate(0.6, path=tmp_path / 'none.json', clock=clk)
    r2.rps = 0.2
    r2.changed_at = clk.t
    for _ in range(40):                                    # 40 clean attempts spread over 40 min
        r2.record(206); clk.t += 60
    assert r2.rps > 0.2 and r2.rps <= 0.6
    top = AutoRate(0.6, path=tmp_path / 'none2.json', clock=clk)
    for _ in range(400):
        top.record(206); clk.t += 60
    assert top.rps == 0.6                                  # ceiling


def test_retry_after_forms():
    assert F.retry_after_seconds('30') == 30
    assert F.retry_after_seconds(None) is None and F.retry_after_seconds('soon') is None
    assert F.retry_after_seconds('Wed, 21 Oct 2026 07:28:30 GMT', now=0) > 0


def test_backoff_jitter_and_retry_after_floor(tmp_path):
    import random
    class Cc503:
        def __init__(self): self.n = 0
        def __call__(self, request):
            self.n += 1
            if self.n < 3:
                return httpx.Response(503, headers={'retry-after': '100'} if self.n == 1 else {})
            return httpx.Response(206, content=b'x' * 10)
    roots = roots_for(tmp_path, U[:1], [(0, 10)])
    ledger = Ledger(roots.run / 'ledger.json', max_bytes=10 ** 9, max_requests=1000)
    c = F.RangeClient(ledger=ledger, log_path=roots.run / 'req.jsonl', rps=1000, transport=httpx.MockTransport(Cc503()), sleep=lambda s: None,
                      jitter=True, rng=random.Random(1))
    assert c.get_range('https://x/y', 0, 10) == b'x' * 10
    delays = [json.loads(l)['delay_s'] for l in (roots.run / 'req.jsonl').read_text().splitlines() if '"backoff"' in l]
    assert delays[0] >= 100                               # Retry-After is a floor
    assert 15 * 0.8 <= delays[1] <= 15 * 1.2 and delays[1] != 15      # exponential step with jitter
    assert c.requests == 3 and c.throttled == 2


def test_stop_event_leaves_url_pending(tmp_path):
    import threading
    cc = Cc([record(U[0]), record(U[1])])
    roots = roots_for(tmp_path, U[:2], cc.spans)
    stop = threading.Event()
    stop.set()
    s = F.run(roots, workers=1, client=client_for(roots, cc), stop=stop)
    assert s['run_urls'] == 0 and s['stopped'] == 'stop requested' and s['not_attempted'] == 2 and cc.calls == 0


def test_priority_order_groups_and_done_marker(tmp_path):
    cc = Cc([record(U[0]), record(U[1]), record(U[2], body=HTML), record(U[3])])
    roots = roots_for(tmp_path, U, cc.spans)
    order = pd.DataFrame({'url_id': [10, 11, 12, 13], 'best_rank': pd.array([50, 5, None, 30], 'Int64'), 'group': [2, 0, 3, 1]})
    prog = {}
    s = F.run(roots, workers=1, shard_size=2, client=client_for(roots, cc), order=order, progress=prog, limit=3)
    db = F.connect(roots)
    got = [r[0] for r in db.execute('SELECT url_id FROM fetch ORDER BY updated_at, url_id')]
    assert sorted(got) == [10, 11, 13]                    # group 0 (11), group 1 (13), group 2 (10) first; group 3 (12) is cut by --limit
    marks = json.loads((roots.state / 'groups_done.json').read_text())
    assert set(marks) >= {'top20', 'top40', 'top100'}
    assert prog['ok_run'] == 3 and prog['todo_left'] == 0


def test_heartbeat_exit_record_and_status(tmp_path):
    import threading
    from r2ai.cc_longchau import fetchctl as K
    roots = roots_for(tmp_path, U[:1], [(0, 10)])
    stop = threading.Event()
    assert K.status(roots)['state'] == 'NOT_STARTED'
    with K.Beat(roots, lambda: {'ok_total': 7}, stop, deadline=None, interval=3600) as beat:
        st = K.status(roots)
        assert st['state'] == 'RUNNING' and st['ok_total'] == 7
        hb = K.read_json(K.paths(roots)['heartbeat'])
        assert K.status(roots, now=hb['at'] + 301)['state'] == 'DEAD'           # old heartbeat and no exit record
    st = K.status(roots)
    assert st['state'] == 'EXITED' and st['code'] == 0
    ex = K.read_json(K.paths(roots)['exit'])
    assert ex['reason'] == 'finished'
    with pytest.raises(RuntimeError):
        with K.Beat(roots, lambda: {}, stop, interval=3600):
            raise RuntimeError('boom')
    ex = K.read_json(K.paths(roots)['exit'])
    assert ex['code'] == 1 and 'boom' in ex['reason'] and 'boom' in ex['error']
    assert len((K.paths(roots)['exits']).read_text().splitlines()) == 2


def test_beat_stops_on_deadline_and_stop_file(tmp_path):
    import threading, time as _t
    from r2ai.cc_longchau import fetchctl as K
    roots = roots_for(tmp_path, U[:1], [(0, 10)])
    stop = threading.Event()
    b = K.Beat(roots, lambda: {}, stop, deadline=_t.time() - 1, interval=3600)
    with b:
        pass
    assert stop.is_set() and b.reason == 'deadline'
    stop2 = threading.Event()
    K.paths(roots)['stop'].write_text('x')
    b2 = K.Beat(roots, lambda: {}, stop2, interval=3600)
    with b2:
        assert not stop2.is_set()                          # a stale STOP file is cleared at start
        K.paths(roots)['stop'].write_text('x')
        b2._check_stop()
        assert stop2.is_set() and b2.reason == 'STOP file'


def test_priority_group_of():
    from r2ai.cc_longchau.priority import group_of
    assert [group_of(r) for r in (1, 20, 21, 40, 41, 100, 101, None)] == [0, 0, 1, 1, 2, 2, 3, 3]
