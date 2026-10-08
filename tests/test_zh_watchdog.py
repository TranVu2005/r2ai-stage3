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


def test_crawler_detection_supports_new_deadline_flag_without_duplicate_spawn():
    procs=[proc(1,cmd=[*CMD,'--deadline','2026-10-20T23:59:00+07:00']),
           proc(2,cmd=[*CMD,'--deadline=2026-10-19T23:59:00+07:00']),
           proc(3,cmd=['python','-m','r2ai.zh_full.crawl','stop','--deadline','x'])]
    assert [p.pid for p in crawler_procs(procs)]==[1,2]
    assert go(procs[:1],runtime={'pid':1,'updated_at':NOW-30})['action']=='ok'


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


def test_deadline_completes_without_restarting_even_when_urls_remain():
    got = decide(NOW, [], {'pid': 10}, None, {'pid': 10, 'code': 130}, [], None, False,
                 deadline=NOW - 1, remaining=123)
    assert got['action'] == 'completed'
    assert got['reason'] == 'deadline reached'


def test_empty_selected_queue_is_completed_after_crawler_exits():
    got = decide(NOW, [], {'pid': 10}, None, {'pid': 10, 'code': 130}, [], None, False,
                 deadline=NOW + 60, remaining=0)
    assert got['action'] == 'completed'
    assert got['reason'] == 'no selected URLs remain'


def test_watchdog_builds_crawl_command_with_same_deadline():
    import r2ai.zh_full.watchdog as watchdog
    deadline = '2026-10-20T23:59:00+07:00'
    build = getattr(watchdog, 'build_crawl_command', None)
    assert callable(build)
    assert build('C:/repo/.venv/Scripts/python.exe', deadline) == [
        'C:/repo/.venv/Scripts/python.exe', '-B', '-u', '-m',
        'r2ai.zh_full.crawl', 'run', '--deadline', deadline]


def test_network_retry_uses_bounded_exponential_backoff():
    import r2ai.zh_full.watchdog as watchdog
    delay = getattr(watchdog, 'network_retry_delay', None)
    assert callable(delay)
    assert [delay(i) for i in (1, 2, 3, 4, 10)] == [60, 120, 240, 480, 600]


def test_network_gate_holds_restart_and_probe_respects_retry_time():
    import r2ai.zh_full.watchdog as watchdog
    gate = getattr(watchdog, 'gate_network_restart', None)
    probe_allowed = getattr(watchdog, 'network_probe_allowed', None)
    assert callable(gate) and callable(probe_allowed)
    state = {}
    decision = {'action': 'restart', 'reason': 'dead'}
    gate(decision, NOW, state, online=False)
    assert decision['action'] == 'waiting_network'
    assert state['next_network_probe_at'] == NOW + 60
    assert not probe_allowed(NOW + 59, state)
    assert probe_allowed(NOW + 60, state)
    retry = {'action': 'restart', 'reason': 'dead'}
    gate(retry, NOW + 60, state, online=True)
    assert retry['action'] == 'restart'
    assert state['network_failures'] == 0


def test_status_alert_timers_clear_when_domain_leaves_active_state():
    import r2ai.zh_full.watchdog as watchdog
    update = getattr(watchdog, '_status_alerts', None)
    assert callable(update)
    state = {'threshold_since': {'cnkang.com:errors': NOW - 3600,
                                 'cnkang.com:zero_ok': NOW - 3600}}
    assert update(NOW, {'cnkang.com': {'state': 'finished', 'ok_per_second_15m': 0,
                                      'error_pct_15m': None}}, True, state) == []
    assert state['threshold_since'] == {}


def test_monitor_bootstrap_keeps_epoch_and_last_robots_without_reading_partial_line(monkeypatch, tmp_path):
    import r2ai.zh_full.watchdog as watchdog
    audit = tmp_path / 'requests.jsonl'
    audit.write_bytes(b'{"domain":"120ask.com","kind":"robots","t0":1,"http_status":521,"error":""}\n'
                      b'{"domain":"120ask.com","kind":"robots","t0":2,"http_status":200,"error":""}\n'
                      b'{"partial":')
    monkeypatch.setattr(watchdog, 'AUDIT_PATH', audit)
    bootstrap = getattr(watchdog, 'bootstrap_monitor_config', None)
    assert callable(bootstrap)
    value = bootstrap(NOW, {'started_at': 12, 'domains': {}})
    assert value['monitor_since'] == 12
    assert value['robots_last'] == {'120ask.com': 200}
    assert value['audit_offset'] == audit.stat().st_size - len(b'{"partial":')


def test_status_row_contains_keepalive_fields_and_public_actions():
    import r2ai.zh_full.watchdog as watchdog
    build = getattr(watchdog, 'status_record', None)
    assert callable(build)
    record = build(NOW, {'action': 'restart', 'reason': 'dead', 'pid': None,
                         'spawned_pid': 20}, {},
                   {'cnkang.com': {'ok_cumulative': 99, 'ok_per_second_15m': 2,
                                   'error_pct_15m': 0, 'remaining': 100,
                                   'state': 'active'}},
                   {'120ask.com': '521'}, 0, 30_000_000_000)
    assert record['action'] == 'started'
    assert record['pid'] == 20
    assert record['domains']['cnkang.com']['remaining'] == 100
    assert record['robots']['120ask.com'] == '521'
    assert record['deferred_fetched'] == 0
    assert record['free_disk_bytes'] == 30_000_000_000
