import json
import os
import time

import pytest

from tests.test_zh_full_ingest import make_roots
from r2ai.zh_full_ingest import driver


def test_heartbeat_writes_pid_step_and_keeps_beating(tmp_path):
    roots = make_roots(tmp_path)
    path = roots.run / 'hb.json'
    with driver.Heartbeat(roots, path, interval=0.05) as hb:
        hb.beat('embed seg', block=7)
        first = json.loads(path.read_text('utf-8'))
        time.sleep(0.3)
        later = json.loads(path.read_text('utf-8'))
    assert first['pid'] == os.getpid() and first['step'] == 'embed seg' and first['detail'] == {'block': 7}
    assert later['at'] > first['at']


def test_exit_record_covers_success_and_exception(tmp_path):
    roots = make_roots(tmp_path)
    ok, bad = roots.run / 'ok.json', roots.run / 'bad.json'
    with driver.Heartbeat(roots, roots.run / 'hb.json') as hb:
        with driver.ExitRecord(roots, ok, hb):
            hb.beat('embed x', block=3)
        with pytest.raises(ValueError):
            with driver.ExitRecord(roots, bad, hb):
                hb.beat('chunk y')
                raise ValueError('boom')
    done, failed = json.loads(ok.read_text('utf-8')), json.loads(bad.read_text('utf-8'))
    assert (done['code'], done['reason'], done['last_step']) == (0, 'ALL_DONE', 'embed x')
    assert failed['code'] == 1 and 'boom' in failed['reason'] and 'ValueError' in failed['error'] and failed['last_step'] == 'chunk y'


def test_status_distinguishes_alive_exited_and_died_silently(tmp_path):
    roots = make_roots(tmp_path)
    p = driver.paths(roots, 'b9')
    assert driver.driver_status(roots, 'b9')['state'] == 'not_started'
    p['heartbeat'].parent.mkdir(parents=True, exist_ok=True)
    p['heartbeat'].write_text(json.dumps({'pid': 1, 'at': 1000.0, 'step': 'embed seg', 'detail': {'block': 361}}), 'utf-8')
    assert driver.driver_status(roots, 'b9', now=1100.0)['state'] == 'alive'
    dead = driver.driver_status(roots, 'b9', now=5000.0)
    assert dead['state'] == 'died_silently' and dead['last_step'] == 'embed seg' and dead['detail'] == {'block': 361}
    p['exit'].write_text(json.dumps({'code': 0, 'reason': 'ALL_DONE', 'last_step': 'done'}), 'utf-8')
    assert driver.driver_status(roots, 'b9', now=5000.0) == {'state': 'exited', 'code': 0, 'reason': 'ALL_DONE', 'last_step': 'done'}


def test_domains_run_priority_first_then_small_first():
    sizes = {'cnkang.com': 500, '120ask.com': 900, 'wujue.com': 10, 'zydcd.com': 20}
    assert driver.order_domains(sizes, ['120ask.com', 'cnkang.com']) == ['120ask.com', 'cnkang.com', 'wujue.com', 'zydcd.com']
    assert driver.order_domains(sizes, ['missing.com']) == ['wujue.com', 'zydcd.com', 'cnkang.com', '120ask.com']


def test_launch_command_is_detached_and_sets_its_own_environment():
    cmd = driver.launch_command(r'D:\GitHub\r2ai-stage3', r'D:\GitHub\r2ai-stage3-main', 'b002', '120ask.com,cnkang.com')
    assert cmd.startswith('cmd.exe /c "cd /d D:\\GitHub\\r2ai-stage3 && set PYTHONPATH=D:\\GitHub\\r2ai-stage3-main\\src')
    assert '-m r2ai.zh_full_ingest.driver run --batch b002 --first 120ask.com,cnkang.com' in cmd
    assert r'driver-b002.stdout.log' in cmd and r'driver-b002.stderr.log' in cmd and '%errorlevel%' not in cmd


class FakeIngest:
    def __init__(self, fail_first_embed=False):
        self.calls, self.fail = [], fail_first_embed

    def extract_batch(self, roots, batch, memory=None, before_shard=None):
        self.calls.append(('extract', batch))
        return {'docs': 3}

    def chunk_segment(self, roots, batch, domain, batch_docs=0):
        self.calls.append(('chunk', domain))
        return {'name': f'seg-{domain}', 'status': 'chunked', 'rows': 5}

    def embed_segment(self, roots, name, memory=None, before_block=None):
        self.calls.append(('embed', name))
        if self.fail:
            self.fail = False
            raise RuntimeError('MEMORY_PAUSES: 4 pauses in 1800s')
        before_block(0)
        return {'status': 'active'}


class FakeView:
    def pending_shards(self):
        return 2

    def domain_sizes(self):
        return {'cnkang.com': 9, '120ask.com': 5, 'wujue.com': 1}

    def chunked_segments(self):
        return {d: (f'seg-{d}', 5) for d in self.domain_sizes()}


def test_run_batch_orders_steps_retries_memory_stops_and_logs(tmp_path):
    from r2ai.zh_full_ingest.common import MemoryGuard
    roots = make_roots(tmp_path)
    roots.run.mkdir(parents=True, exist_ok=True)
    ingest, events, beats, sleeps = FakeIngest(fail_first_embed=True), [], [], []
    driver.run_batch(roots, 'b2', view=FakeView(), ingest=ingest, first=['120ask.com', 'cnkang.com'], log=lambda **kw: events.append(kw),
                     beat=lambda step, **d: beats.append(step), stop=lambda i: None, extract_guard=MemoryGuard(roots),
                     chunk_guard=MemoryGuard(roots, available=lambda: 99), embed_guard=MemoryGuard(roots), sleep=sleeps.append, wait_s=7)
    order = [c for c in ingest.calls if c != ('embed', 'seg-120ask.com')]
    assert [c[0] for c in ingest.calls[:4]] == ['extract', 'chunk', 'chunk', 'chunk']
    assert [c[1] for c in ingest.calls[1:4]] == ['120ask.com', 'cnkang.com', 'wujue.com']
    assert [c[1] for c in ingest.calls if c[0] == 'embed'] == ['seg-120ask.com', 'seg-120ask.com', 'seg-cnkang.com', 'seg-wujue.com']
    assert sleeps == [7] and any(e.get('retry') == 1 for e in events) and events[-1] == {'step': 'ALL_DONE'}
    assert any(b.startswith('waiting: embed') for b in beats) and order


def test_run_batch_stops_on_monitor_alert_without_running_the_step(tmp_path):
    from r2ai.zh_full_ingest.common import MemoryGuard
    roots = make_roots(tmp_path)
    roots.run.mkdir(parents=True, exist_ok=True)
    (roots.run / 'monitor_alert.json').write_text('{}', 'utf-8')
    ingest = FakeIngest()
    with pytest.raises(SystemExit):
        driver.run_batch(roots, 'b2', view=FakeView(), ingest=ingest, log=lambda **kw: None, beat=lambda *a, **k: None,
                         stop=lambda i: None, extract_guard=MemoryGuard(roots), chunk_guard=MemoryGuard(roots),
                         embed_guard=MemoryGuard(roots))
    assert ingest.calls == []
