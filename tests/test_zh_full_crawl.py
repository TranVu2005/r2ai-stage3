import json

import pytest

from r2ai.zh_full.crawl import LANES, new_groups, network_runtime, domain_config, snapshot_rates
from r2ai.zh_full.common import guard
from vicrawl.settings import Config


def test_skip_all_sample_urls_and_preserve_alias_grouping():
    rows = [(1, 'http://www.120ask.com/a/'), (2, 'https://120ask.com/a'),
            (3, 'https://120ask.com/b'), (4, 'https://120ask.com/c')]
    result = new_groups(rows, '120ask.com', {'120ask.com/b'})
    assert [r['url_norm'] for r in result] == ['120ask.com/a', '120ask.com/c']
    assert result[0]['doc_ids'] == [1, 2]
    assert result[0]['url'] == 'https://120ask.com/a'
    assert [r['rank'] for r in result] == [0, 1]


def test_domain_config_caps_connections_without_mutating_config():
    cfg = Config(domains={'cnkang.com': {'max_conns': 4, 'rate_cap': 1}})
    got = domain_config(cfg, 'cnkang.com')
    assert got.max_conns == 2 and got.rate_cap == 1
    assert cfg.for_domain('cnkang.com').max_conns == 4


def test_lanes_are_disjoint_and_long_sites_start_first():
    assert len(LANES) == 4 and LANES[0][0] == '120ask.com' and LANES[1][0] == 'cnkang.com'
    domains = [d for lane in LANES for d in lane]
    assert len(domains) == len(set(domains)) == 15
    assert not set(domains) & {'zysjonline.com', 'bingli.iiyi.com', 'ask.39.net'}


def test_reusing_network_code_does_not_mutate_default_sample_module():
    from r2ai.zh_sample import crawl as original
    old = original.RUN
    cloned = network_runtime()
    assert cloned.RUN != old and original.RUN == old
    assert cloned is not original


def test_guard_forbids_sample_vi_old_and_other_agent():
    for bad in ('data/raw_zh_sample/a', 'state/crawl.db', 'data/raw_vi/a',
                'out/runs/zh-sample/a', 'out/runs/rrf/a', 'D:/GitHub/r2ai-stage3-old/out/a'):
        with pytest.raises(ValueError):
            guard(bad)


def test_rate_uses_window_elapsed_and_url_attempts_not_redirect_requests():
    events = [{'domain': '120ask.com', 'kind': 'page', 't0': 100, 't1': 101, 'http_status': 200, 'error': ''},
              {'domain': '120ask.com', 'kind': 'page', 't0': 101, 't1': 102, 'http_status': 503, 'error': ''},
              {'domain': '120ask.com', 'kind': 'robots', 't0': 102, 't1': 103, 'http_status': 200, 'error': ''}]
    got = snapshot_rates(events, {'120ask.com': {'start': 100, 'completed': 1, 'remaining': 99}}, 110)
    assert got['120ask.com']['page_req_per_second'] == .2
    assert got['120ask.com']['url_per_second'] == .1
    assert got['120ask.com']['http_error_pct'] == 50
    assert got['120ask.com']['eta_hours_estimated'] == pytest.approx(99 / .1 / 3600)


def test_worker_failure_stops_other_workers_before_join():
    import asyncio
    from r2ai.zh_full.crawl import supervise_workers

    async def check():
        stop = asyncio.Event()

        async def failure():
            raise OSError('disk failure')

        async def sibling():
            await stop.wait()

        with pytest.raises(OSError, match='disk failure'):
            await asyncio.wait_for(supervise_workers([failure(), sibling()], stop), timeout=1)
        assert stop.is_set()

    asyncio.run(check())


def test_monitor_failure_stops_lanes_and_propagates():
    import asyncio
    from r2ai.zh_full.crawl import supervise_lanes

    async def check():
        stop = asyncio.Event()
        async def lane():
            await stop.wait()
        async def monitor():
            raise OSError('heartbeat disk failure')
        with pytest.raises(OSError, match='heartbeat disk failure'):
            await asyncio.wait_for(supervise_lanes([lane()], monitor(), stop), timeout=1)
        assert stop.is_set()
    asyncio.run(check())


def test_transient_outage_waits_but_403_safety_halt_remains():
    from r2ai.zh_full.crawl import FullTuner
    tuner = FullTuner(4, 2, clock=lambda: 100)
    for _ in range(12):
        tuner.record(.1, None, True)
    assert not tuner.halted and tuner.paused_until >= 160
    assert tuner.record(.1, 200, False) is not None
    blocked = FullTuner(4, 2, clock=lambda: 100)
    for _ in range(12):
        blocked.record(.1, 403, False)
    assert blocked.halted


def test_direct_socket_limits_and_cross_domain_redirect_policy():
    from r2ai.zh_full.crawl import direct_limits, same_domain_url
    assert direct_limits(True).max_connections == 1
    assert direct_limits(True).max_keepalive_connections == 0
    assert direct_limits(False).max_connections == 2
    assert same_domain_url('https://www.120ask.com/a', '120ask.com')
    assert not same_domain_url('https://cnkang.com/a', '120ask.com')
    assert not same_domain_url('https://120ask.com@evil.com/a', '120ask.com')


@pytest.mark.parametrize('failures,successes', [(60, 1), (12, 38)])
def test_transient_history_does_not_halt_or_extend_pause_on_recovery(failures, successes):
    from r2ai.zh_full.crawl import FullTuner
    now = [100.0]
    tuner = FullTuner(4, 2, clock=lambda: now[0])
    for _ in range(failures):
        tuner.record(.1, None, True)
    assert not tuner.halted
    rate, paused_until = tuner.rate, tuner.paused_until
    now[0] = 200.0
    for _ in range(successes):
        events = tuner.record(.1, 200, False)
        assert not tuner.halted
        assert tuner.rate == rate and tuner.paused_until == paused_until
        assert not any(e['event'] == 'full_zh_outage_wait' for e in events)


def test_session_start_rate_keeps_learned_rate_unless_it_was_cut():
    from r2ai.zh_full.crawl import session_start_rate
    assert session_start_rate(.5625, 4, had_rate_cut=False) == .5625
    assert session_start_rate(.5625, 4, had_rate_cut=True) == pytest.approx(.421875)
    assert session_start_rate(6, 4, had_rate_cut=False) == 4
    assert session_start_rate(.1, 4, had_rate_cut=False) == .25
    assert session_start_rate(None, 4, had_rate_cut=False) == 1.0


def test_begin_session_remeasures_baseline_and_resumes_rate():
    from r2ai.zh_full.crawl import FullTuner, begin_session
    tuner = FullTuner(4, 2, rate=.421875, baseline_p95=.59, clock=lambda: 100)
    begin_session(tuner, .5625, had_rate_cut=False)
    assert tuner.rate == .5625 and tuner.baseline_p95 is None
    for _ in range(100):
        tuner.record(2.0, 200, False)
    assert tuner.baseline_p95 == 2.0


def test_full_tuner_reports_every_rate_cut():
    from r2ai.zh_full.crawl import FullTuner
    cuts = []
    limited = FullTuner(4, 2, clock=lambda: 100)
    limited.on_rate_cut = lambda: cuts.append('429')
    limited.record(.1, 429, False)
    outage = FullTuner(4, 2, clock=lambda: 100)
    outage.on_rate_cut = lambda: cuts.append('outage')
    for _ in range(12):
        outage.record(.1, None, True)
    slow = FullTuner(4, 2, clock=lambda: 100)
    slow.on_rate_cut = lambda: cuts.append('slow')
    slow.record(5, 200, False)
    assert {'429', 'outage', 'slow'} <= set(cuts)
    quiet = FullTuner(4, 2, clock=lambda: 100)
    quiet.on_rate_cut = lambda: cuts.append('quiet')
    for _ in range(600):
        quiet.record(.1, 200, False)
    assert 'quiet' not in cuts


def test_rate_cut_ledger_is_read_once_per_session(tmp_path, monkeypatch):
    from r2ai.zh_full import crawl
    monkeypatch.setattr(crawl, 'atomic_json', lambda p, v: p.write_text(json.dumps(v), encoding='utf-8'))
    path = tmp_path / 'rate_cuts.json'
    assert not crawl.take_rate_cut(path, 'cnkang.com')
    crawl.mark_rate_cut(path, 'cnkang.com')
    crawl.mark_rate_cut(path, 'cnkang.com')
    assert crawl.take_rate_cut(path, 'cnkang.com')
    assert not crawl.take_rate_cut(path, 'cnkang.com')
    assert not crawl.take_rate_cut(path, '120ask.com')


def test_h2_protocol_error_is_a_recorded_request_failure_not_a_crash():
    # Server closes a pooled HTTP/2 connection while a new stream opens; h2 raises outside httpx.HTTPError.
    import asyncio
    import io
    import h2.exceptions
    from r2ai.zh_full.crawl import FullTuner, network_runtime
    runtime = network_runtime()

    class Net:
        async def check(self): pass

    class Transport:
        def stream(self, *args):
            raise h2.exceptions.ProtocolError('Invalid input ConnectionInputs.SEND_SETTINGS in state ConnectionState.CLOSED')

    async def run():
        c = object.__new__(runtime.PoliteClient)
        c.domain, c.net, c.stop, c.audit = 'cnkang.com', Net(), asyncio.Event(), io.StringIO()
        c.tuner = FullTuner(4, 2, rate=4)
        c.next_at, c.pace_lock, c.client = 0, asyncio.Lock(), Transport()
        return c, await c.hop('https://www.cnkang.com/a', 'page')

    client, (code, _, body, error, _) = asyncio.run(run())
    assert code is None and body == b'' and error.startswith('ProtocolError: ')
    assert json.loads(client.audit.getvalue())['error'] == error
    assert list(client.tuner._errs) == [True]


def test_stop_request_targets_only_its_process_and_ignores_stale_pid():
    from r2ai.zh_full import crawl
    assert crawl.stop_requested({'pid': 13704}, 13704)
    assert not crawl.stop_requested({'pid': 13704}, 13705)
    for invalid in ({}, None, {'pid': True}, {'pid': '13704'}, {'pid': -1}):
        assert not crawl.stop_requested(invalid, 13704)


@pytest.mark.parametrize('stopping', [True, False])
def test_cancelled_fetch_requeues_row_and_only_swallows_requested_stop(stopping):
    import asyncio
    import sqlite3
    from r2ai.zh_full import crawl
    from vicrawl.state import SCHEMA, StateDB

    class CancelledClient:
        async def fetch(self, row):
            raise asyncio.CancelledError

    async def check():
        db = StateDB.__new__(StateDB)
        db.conn = sqlite3.connect(':memory:', isolation_level=None)
        db.conn.executescript(SCHEMA)
        try:
            db.add_urls([{'url_norm': '120ask.com/a', 'url': 'https://120ask.com/a',
                          'doc_ids': [1], 'domain': '120ask.com', 'rank': 0}])
            row = db.next_batch('120ask.com', 1, 0)[0]
            stop = asyncio.Event()
            if stopping:
                stop.set()
                assert await crawl.fetch_record(CancelledClient(), db, row, stop) is None
            else:
                with pytest.raises(asyncio.CancelledError):
                    await crawl.fetch_record(CancelledClient(), db, row, stop)
            assert db.conn.execute('SELECT status FROM urls').fetchone()[0] == 'pending'
        finally:
            db.close()

    asyncio.run(check())
