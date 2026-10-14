from r2ai.zh_v2 import watchdog as w

NOW = 10_000.0


def d(**kw):
    base = dict(now=NOW, procs=[], heartbeat_at=None, last_exit=None, remaining=100, paused=False, restarts_last_hour=0, online=True)
    return w.decide(**{**base, **kw})


def test_alive_when_process_and_fresh_heartbeat():
    assert d(procs=[1], heartbeat_at=NOW - 30)['state'] == 'ALIVE'


def test_hung_process_is_dead_and_killed_before_restart():
    r = d(procs=[1], heartbeat_at=NOW - 2000)
    assert (r['state'], r['action']) == ('DEAD', 'kill_restart')
    assert d(procs=[1], heartbeat_at=None)['action'] == 'kill_restart'


def test_no_process_is_dead_and_restarts_after_reboot():
    r = d(last_exit={'code': 130})
    assert (r['state'], r['action']) == ('DEAD', 'restart')


def test_done_pause_and_holds():
    assert d(remaining=0, last_exit={'code': 0})['state'] == 'DONE'
    assert d(paused=True)['state'] == 'PAUSED' and d(paused=True)['action'] == 'none'
    assert d(restarts_last_hour=6)['state'] == 'HOLD'
    assert d(online=False)['state'] == 'HOLD'


def test_crawler_command_detection():
    assert w.is_crawler_command(['python', '-B', '-u', '-m', 'r2ai.zh_v2.crawl', 'run'])
    assert not w.is_crawler_command(['python', '-m', 'r2ai.zh_full.crawl', 'run'])
    assert not w.is_crawler_command(['python', '-m', 'r2ai.zh_v2.crawl', 'status'])
