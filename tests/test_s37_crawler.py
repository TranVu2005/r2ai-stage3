import importlib.util
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('s37_crawler', Path(__file__).parents[1] / 'deploy' / 's37' / 'crawler.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

GOOD = '<html><h1>问题标题</h1><div class="b_askcont">我头疼三天了怎么办</div><div class="b_answercont">多休息，必要时就医</div></html>'


class Handler(BaseHTTPRequestHandler):
    flaky_hits: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path
        n = Handler.flaky_hits[path] = Handler.flaky_hits.get(path, 0) + 1
        if path == '/question/ok.htm':
            self.reply(200, GOOD)
        elif path == '/question/plain.htm':
            self.reply(200, '<html><h1>t</h1><p>nothing</p></html>')
        elif path == '/question/gone.htm':
            self.reply(404, 'x')
        elif path == '/question/flaky.htm':
            self.reply(521 if n <= 2 else 200, 'down' if n <= 2 else GOOD)
        elif path == '/question/down.htm':
            self.reply(521, 'down')
        elif path == '/question/block.htm':
            self.reply(403, 'no')
        elif path == '/question/slow429.htm':
            self.reply(429, 'no')
        elif path == '/question/chl.htm':
            self.reply(200, '<html><title>Just a moment...</title><script src="/cdn-cgi/challenge-platform/x"></script></html>')
        elif path == '/question/off.htm':
            self.send_response(302)
            self.send_header('Location', 'http://example.invalid/x')
            self.end_headers()
        elif path == '/question/hop.htm':
            self.send_response(302)
            self.send_header('Location', '/question/ok.htm')
            self.end_headers()
        else:
            self.reply(404, 'x')

    def reply(self, code, body):
        data = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    Handler.flaky_hits = {}
    srv = HTTPServer(('127.0.0.1', 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f'http://127.0.0.1:{srv.server_port}'
    srv.shutdown()


def crawler(out, sleeps=None):
    sleeps = [] if sleeps is None else sleeps
    return C.Crawler(out, suffix='127.0.0.1', min_gap=0, jitter=0, upgrade_https=False, sleep=sleeps.append), sleeps


def queue(base, *names):
    return [(str(i + 1), f'{base}/question/{n}.htm') for i, n in enumerate(names)]


def test_ok_suspect_gone_offsite_and_same_host_redirect(server, tmp_path):
    c, _ = crawler(tmp_path)
    res = c.run(queue(server, 'ok', 'plain', 'gone', 'off', 'hop'))
    assert res['fetched'] == 5
    log = {r['doc_id']: r for r in C.read_log(tmp_path / 'log.jsonl')}
    assert [log[str(i)]['status'] for i in range(1, 6)] == ['ok', 'ok', 'gone', 'offsite', 'ok']
    assert log['1']['suspect'] is False and log['2']['suspect'] is True and log['5']['suspect'] is False
    recs = C.read_shard(tmp_path / 'shards' / 'r1-00000.jsonl.gz')
    assert [r['doc_id'] for r in recs] == ['1', '2', '5'] and '我头疼三天了怎么办' in recs[0]['html']


def test_retry_backoff_then_success_and_failed_after_three_retries(server, tmp_path):
    c, sleeps = crawler(tmp_path)
    c.run(queue(server, 'flaky', 'down'))
    log = {r['doc_id']: r for r in C.read_log(tmp_path / 'log.jsonl')}
    assert log['1']['status'] == 'ok' and log['1']['attempts'] == 3
    assert log['2']['status'] == 'failed' and log['2']['attempts'] == 4 and log['2']['err'] == 'http_521'
    assert sleeps == [30, 120, 30, 120, 600]


@pytest.mark.parametrize('name,reason', [('block', 'http_403'), ('slow429', 'http_429'), ('chl', 'challenge_page')])
def test_block_and_challenge_stop_everything(server, tmp_path, name, reason):
    c, _ = crawler(tmp_path)
    res = c.run(queue(server, 'ok', name, 'ok'))
    assert res['stopped']['reason'] == reason
    assert [r['doc_id'] for r in C.read_log(tmp_path / 'log.jsonl')] == ['1']        # the blocked URL stays pending
    assert (tmp_path / 'STOP.json').exists()
    with pytest.raises(SystemExit):
        crawler(tmp_path)[0].run(queue(server, 'ok'))


def test_resume_skips_done_and_cuts_torn_shard(server, tmp_path):
    c, _ = crawler(tmp_path)
    q = queue(server, 'ok', 'ok', 'ok')
    c.run(q, limit=2)
    shard = tmp_path / 'shards' / 'r1-00000.jsonl.gz'
    with open(shard, 'ab') as f:
        f.write(b'\x1f\x8b\x08torn')                                    # crash in the middle of the third member
    Handler.flaky_hits = {}
    c2, _ = crawler(tmp_path)
    assert c2.run(q)['fetched'] == 1
    assert Handler.flaky_hits == {'/question/ok.htm': 1}
    assert [r['doc_id'] for r in C.read_shard(shard)] == ['1', '2', '3']


def test_pack_batches_only_complete_shards(server, tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'SHARD_URLS', 2)
    urls = tmp_path / 'urls.tsv'
    q = queue(server, 'ok', 'plain', 'ok', 'gone', 'ok')
    urls.write_text(''.join(f'{d}\t{u}\n' for d, u in q), encoding='utf-8')
    c, _ = crawler(tmp_path)
    c.run(q)
    ns = type('A', (), {'out': str(tmp_path), 'urls': str(urls), 'batch_shards': 2, 'final': False})
    C.cmd_pack(ns)
    batches = sorted((tmp_path / 'batches').glob('*.tar'))
    assert len(batches) == 1                                            # shards 0,1 pair up; the odd shard 2 waits
    with tarfile.open(batches[0]) as t:
        manifest = json.load(t.extractfile('MANIFEST.json'))
        assert manifest['urls'] == 4 and manifest['ok'] == 3 and manifest['suspect'] == 1
        assert {m.name for m in t.getmembers()} >= {'MANIFEST.json', 'log.jsonl', 'shards/r1-00000.jsonl.gz', 'shards/r1-00001.jsonl.gz'}
    ns.final = True
    C.cmd_pack(ns)
    assert len(list((tmp_path / 'batches').glob('*.tar'))) == 2         # leftover shard goes out with --final
    C.cmd_pack(ns)
    assert len(list((tmp_path / 'batches').glob('*.tar'))) == 2         # idempotent


def test_pack_partial_ships_unfinished_shard_once(server, tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'SHARD_URLS', 2)
    urls = tmp_path / 'urls.tsv'
    q = queue(server, 'ok', 'ok', 'ok', 'ok', 'ok')
    urls.write_text(''.join(f'{d}\t{u}\n' for d, u in q), encoding='utf-8')
    c, _ = crawler(tmp_path)
    c.run(q[:3])                                                         # crawl stops inside shard 1 (1 of 2 URLs)
    ns = type('A', (), {'out': str(tmp_path), 'urls': str(urls), 'batch_shards': 2, 'final': False, 'partial': True})
    C.cmd_pack(ns)
    batches = sorted((tmp_path / 'batches').glob('*.tar'))
    assert len(batches) == 1                                             # the complete shard 0 still waits for a partner
    with tarfile.open(batches[0]) as t:
        manifest = json.load(t.extractfile('MANIFEST.json'))
        assert manifest['urls'] == 1 and manifest['ok'] == 1
        assert [m['name'] for m in manifest['shards']] == ['r1-00001.jsonl.gz']
    C.cmd_pack(ns)
    assert len(list((tmp_path / 'batches').glob('*.tar'))) == 1          # same row count: not packed twice
    packed = json.loads((tmp_path / 'packed.json').read_text('utf-8'))
    assert packed['partial'] == {'r1-00001.jsonl.gz': 1} and packed['shards'] == []


def test_retry_round_queue_holds_only_failed(server, tmp_path):
    c, _ = crawler(tmp_path)
    c.run(queue(server, 'ok', 'down', 'gone'))
    assert C.retry_queue(tmp_path, 2) == [('2', f'{server}/question/down.htm')]


def test_min_gap_below_one_second_rejected():
    with pytest.raises(SystemExit):
        C.main(['run', '--min-gap', '0.5'])


def test_pacing_enforces_gap_and_jitter():
    now = [0.0]
    waits = []
    c = C.Crawler.__new__(C.Crawler)
    c.min_gap, c.jitter, c.last_start = 1.0, 0.3, None
    c.clock = lambda: now[0]
    c.sleep = lambda s: (waits.append(s), now.__setitem__(0, now[0] + s))
    for _ in range(5):
        c._pace()
    assert len(waits) == 4 and all(1.0 <= w <= 1.3 for w in waits)
