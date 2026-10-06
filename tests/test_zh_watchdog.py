import pytest

from r2ai.zh_full.watchdog import CRAWL_ARGS, Proc, crawler_procs, decide, restarts_last_hour

NOW = 1_800_000_000.0
CMD = ['D:/venv/python.exe', *CRAWL_ARGS]


def proc(pid=10, created=NOW - 3600, cmd=CMD):
    return Proc(pid, created, list(cmd))


def go(procs=(), runtime=None, log_mtime=None, last_exit=None, history=(), suspect=None, paused=False):
    return decide(NOW, list(procs), runtime, log_mtime, last_exit, list(history), suspect, paused)


def test_crawler_detection_needs_exact_module_and_run_step():
    procs = [proc(1), proc(2, cmd=['python', '-m', 'r2ai.zh_full.crawl', 'stop']),
             proc(3, cmd=['python', '-m', 'pytest']), proc(4, cmd=[])]
    assert [p.pid for p in crawler_procs(procs)] == [1]


def test_fresh_heartbeat_is_ok_and_clears_suspect():
    got = go([proc()], runtime={'pid': 10, 'updated_at': NOW - 60}, suspect={'pid': 10, 'since': NOW - 900})
    assert got['action'] == 'ok' and got['suspect'] is None


def test_dead_crawler_restarts_with_exit_code_of_last_run():
    got = go([], runtime={'pid': 10, 'updated_at': NOW - 30}, last_exit={'pid': 10, 'code': 130})
    assert got['action'] == 'restart' and got['exit_code'] == 130 and got['reason'] == 'dead'


def test_hard_killed_crawler_has_unknown_exit_code():
    got = go([], runtime={'pid': 10, 'updated_at': NOW - 30}, last_exit={'pid': 7, 'code': 130})
    assert got['action'] == 'restart' and got['exit_code'] is None


def test_finished_crawl_is_not_restarted():
    got = go([], runtime={'pid': 10, 'updated_at': NOW - 30}, last_exit={'pid': 10, 'code': 0})
    assert got['action'] == 'finished'


def test_pause_file_blocks_restart():
    assert go([], paused=True)['action'] == 'paused'


def test_restart_storm_gives_up_after_six_per_hour():
    history = [NOW - 60 * i for i in range(1, 7)]
    assert restarts_last_hour(history, NOW) == 6
    assert go([], history=history)['action'] == 'gave_up'
    assert go([], history=history + [NOW - 7200])['action'] == 'gave_up'
    assert go([], history=history[:5] + [NOW - 3700])['action'] == 'restart'


def test_reused_pid_does_not_count_as_heartbeat():
    # Heartbeat written before this process existed belongs to another incarnation.
    p = proc(created=NOW - 100)
    got = go([p], runtime={'pid': 10, 'updated_at': NOW - 700}, log_mtime=NOW - 900)
    assert got['action'] == 'ok'          # still inside its own startup grace (created 100 s ago)
    p = proc(created=NOW - 2000)
    got = go([p], runtime={'pid': 10, 'updated_at': NOW - 2500}, log_mtime=NOW - 2500)
    assert got['action'] == 'suspect'


def test_stale_heartbeat_needs_two_consecutive_checks_before_kill():
    stale = dict(runtime={'pid': 10, 'updated_at': NOW - 1200}, log_mtime=NOW - 1200)
    first = go([proc()], **stale)
    assert first['action'] == 'suspect' and first['suspect'] == {'pid': 10, 'since': NOW}
    # A wake from sleep: heartbeat old, but the previous suspect is too recent to act.
    assert go([proc()], **stale, suspect={'pid': 10, 'since': NOW - 60})['action'] == 'suspect'
    second = go([proc()], **stale, suspect={'pid': 10, 'since': NOW - 300})
    assert second['action'] == 'kill_restart' and second['reason'] == 'stale_heartbeat'


def test_recent_log_output_counts_as_alive_during_egress_wait():
    got = go([proc()], runtime={'pid': 99, 'updated_at': NOW - 5000}, log_mtime=NOW - 50)
    assert got['action'] == 'ok'


def test_trampoline_parent_and_child_are_one_crawler():
    parent, child = proc(5, created=NOW - 3600), proc(10, created=NOW - 3599)
    got = go([parent, child], runtime={'pid': 10, 'updated_at': NOW - 30})
    assert got['action'] == 'ok' and got['pid'] == 10


@pytest.mark.parametrize('history', [[], [NOW - 10]])
def test_stale_kill_also_respects_restart_budget(history):
    stale = dict(runtime={'pid': 10, 'updated_at': NOW - 1200}, log_mtime=NOW - 1200,
                 suspect={'pid': 10, 'since': NOW - 600})
    assert go([proc()], **stale, history=history)['action'] == 'kill_restart'
    assert go([proc()], **stale, history=[NOW - i for i in range(6)])['action'] == 'gave_up'


def test_keep_awake_sets_and_clears_system_required():
    from r2ai.zh_full.common import keep_awake, ES_CONTINUOUS, ES_SYSTEM_REQUIRED

    class K:
        calls = []

        def SetThreadExecutionState(self, flags):
            self.calls.append(flags)

    k = K()
    with pytest.raises(KeyboardInterrupt):
        with keep_awake(k):
            assert k.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
            raise KeyboardInterrupt
    assert k.calls[-1] == ES_CONTINUOUS


def test_crawl_main_records_exit_code_for_watchdog(monkeypatch, tmp_path):
    import r2ai.zh_full.crawl as crawl
    from contextlib import nullcontext
    written = {}

    async def fake_run():
        return 130
    monkeypatch.setattr(crawl, 'run', fake_run)
    monkeypatch.setattr(crawl, 'exclusive', lambda name: nullcontext())
    monkeypatch.setattr(crawl, 'keep_awake', lambda: nullcontext())
    monkeypatch.setattr(crawl, 'atomic_json', lambda path, value: written.update({path.name: value}))
    assert crawl.main(['run']) == 130
    assert written['last_exit.json']['code'] == 130


def test_crawl_main_records_unknown_code_on_crash(monkeypatch):
    import r2ai.zh_full.crawl as crawl
    from contextlib import nullcontext
    written = {}

    async def boom():
        raise RuntimeError('x')
    monkeypatch.setattr(crawl, 'run', boom)
    monkeypatch.setattr(crawl, 'exclusive', lambda name: nullcontext())
    monkeypatch.setattr(crawl, 'keep_awake', lambda: nullcontext())
    monkeypatch.setattr(crawl, 'atomic_json', lambda path, value: written.update({path.name: value}))
    with pytest.raises(RuntimeError):
        crawl.main(['run'])
    assert written['last_exit.json']['code'] is None
