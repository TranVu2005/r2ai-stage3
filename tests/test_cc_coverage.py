"""Common Crawl index measurement tools (r2ai.cc_coverage): no network, a mock transport serves synthetic parquet files."""
import gzip
import io
import json
import re

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.cc_coverage import choose as CH
from r2ai.cc_coverage import crawls as CR
from r2ai.cc_coverage import index_scan as IX
from r2ai.cc_coverage import net as N
from r2ai.cc_coverage.rangefile import RangeFile
from r2ai.cc_coverage.surt import surt_key

UA = 'test-agent (+contact@example.invalid)'


# ---------------------------------------------------------------- surt / crawls / choose

def test_surt_key_matches_cc_urlkey():
    assert surt_key('https://nhathuoclongchau.com.vn/bai-viet/khong-an-uong-thuoc-co-sao-khong.html') == \
        'vn,com,nhathuoclongchau)/bai-viet/khong-an-uong-thuoc-co-sao-khong.html'
    assert surt_key('http://WWW.NhaThuocLongChau.com.vn/benh/co-giat-765.html') == 'vn,com,nhathuoclongchau)/benh/co-giat-765.html'
    assert surt_key('https://a.vn') == 'vn,a)/' and surt_key('https://a.vn/x?b=2&a=1') == 'vn,a)/x?a=1&b=2'


def _info(n):
    return [{'id': f'CC-MAIN-{y}-{w:02d}', 'from': f'{d}T00:00:00', 'to': f'{d}T23:00:00'} for (y, w, d) in n]


def test_select_crawls_window_and_after_corpus():
    ci = _info([(2025, 21, '2025-05-10'), (2025, 33, '2025-08-10'), (2025, 38, '2025-09-14'), (2025, 42, '2025-10-10'),
                (2026, 17, '2026-04-10'), (2026, 39, '2026-09-04'), (2026, 43, '2026-10-05'), (2026, 47, '2026-11-05')])
    got = CR.select_crawls(ci)
    assert [c['id'] for c in got] == ['CC-MAIN-2025-38', 'CC-MAIN-2025-42', 'CC-MAIN-2026-17', 'CC-MAIN-2026-39',
                                      'CC-MAIN-2026-43', 'CC-MAIN-2026-47']
    assert [c['after_corpus'] for c in got] == [False, False, False, False, True, True]
    with pytest.raises(ValueError):
        CR.select_crawls(ci[:2])


def _rec(crawl, ts, length=100, status='200', mime='text/html'):
    return {'crawl': crawl, 'fetch_time': ts, 'warc_record_length': length, 'fetch_status': status, 'content_mime_type': mime}


def test_choose_newest_not_after_corpus_ties_by_length_and_flags_after():
    a, b = _rec('CC-MAIN-2026-34', '2026-08-15T10:00:00'), _rec('CC-MAIN-2026-39', '2026-09-08T09:00:00')
    late = _rec('CC-MAIN-2026-43', '2026-10-05T00:00:00')
    assert CH.choose([a, b, late], '2026-09-29')['crawl'] == 'CC-MAIN-2026-39' and not CH.choose([a, b], '2026-09-29')['after_corpus']
    t1, t2 = _rec('x', '2026-09-08T09:00:00', 50), _rec('y', '2026-09-08T09:00:00', 70)
    assert CH.choose([t1, t2], '2026-09-29')['crawl'] == 'y'
    c = CH.choose([late, _rec('CC-MAIN-2026-47', '2026-11-05T00:00:00')], '2026-09-29')
    assert c['crawl'] == 'CC-MAIN-2026-43' and c['after_corpus'] is True
    assert CH.choose([_rec('x', '2026-09-01T00:00:00', mime='warc/revisit'), _rec('y', '2026-09-01T00:00:00', status='301')], '2026-09-29') is None


def test_choose_accepts_parquet_timestamps():
    import datetime as dt
    a = _rec('CC-MAIN-2026-34', dt.datetime(2026, 8, 15, 10, 0, 0))
    b = _rec('CC-MAIN-2026-39', dt.datetime(2026, 9, 8, 9, 0, 0))
    assert CH.choose([a, b], '2026-09-29')['crawl'] == 'CC-MAIN-2026-39'


def test_summarize_counts_by_prefix_crawl_and_bytes():
    ch = {1: CH.choose([_rec('CC-MAIN-2026-39', '2026-09-08T00:00:00', 55)], '2026-09-29'), 2: None,
          3: CH.choose([_rec('CC-MAIN-2026-34', '2026-08-08T00:00:00', 65)], '2026-09-29')}
    s = CH.summarize(ch, {1: '/bai-viet', 2: '/bai-viet', 3: '/benh'})
    assert s['urls'] == 3 and s['with_200_html'] == 2 and s['warc_bytes_chosen'] == 120 and s['after_corpus_chosen'] == 0
    assert s['by_prefix'] == {'/bai-viet': {'urls': 2, 'with_200_html': 1}, '/benh': {'urls': 1, 'with_200_html': 1}}
    assert s['chosen_by_crawl'] == {'CC-MAIN-2026-34': 1, 'CC-MAIN-2026-39': 1}


# ---------------------------------------------------------------- net.Client

class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _client(tmp_path, handler, *, rps=1.0, ledger=None, clock=None):
    clock = clock or Clock()
    led = ledger or N.Ledger(tmp_path / 'ledger.json')
    c = N.Client(ua=UA, ledger=led, log_path=tmp_path / 'req.jsonl', rps=rps, transport=httpx.MockTransport(handler),
                 sleep=clock.sleep, clock=clock.now)
    return c, led, clock


def test_client_requires_a_ua(tmp_path):
    with pytest.raises(ValueError):
        N.Client(ua=' ', ledger=N.Ledger(tmp_path / 'l.json'), log_path=tmp_path / 'r.jsonl')


def test_503_is_unavailable_with_exponential_backoff_then_ok(tmp_path):
    seq = iter([503, 503, 200])
    c, led, clock = _client(tmp_path, lambda req: httpx.Response(next(seq), content=b'x' * 10), rps=1000)
    r = c.get('https://h/x')
    assert r.status_code == 200 and clock.slept == [5, 15] and led.requests == 3 and led.bytes == 30
    assert req_ua(c) == UA


def req_ua(c):
    return c.http.headers['user-agent']


def test_breaker_cools_down_on_third_consecutive_failure_and_exhaustion_raises(tmp_path):
    c, led, clock = _client(tmp_path, lambda req: httpx.Response(503, headers={'retry-after': '2'}), rps=1000)
    with pytest.raises(N.Unavailable):
        c.get('https://h/x')
    assert clock.slept == [5, 15, 600, 120, 600] and led.requests == 6                 # fails 3 and 5 trip the cool-down (>= 600)


def test_retry_after_is_honoured(tmp_path):
    seq = iter([(503, '77'), (200, None)])

    def h(req):
        s, ra = next(seq)
        return httpx.Response(s, headers={'retry-after': ra} if ra else {})
    c, _, clock = _client(tmp_path, h, rps=1000)
    c.get('https://h/x')
    assert clock.slept == [77]


@pytest.mark.parametrize('status', [403, 429])
def test_only_403_and_429_are_blocked_and_stop_at_once(tmp_path, status):
    c, led, clock = _client(tmp_path, lambda req: httpx.Response(status))
    with pytest.raises(N.Blocked):
        c.get('https://h/x')
    assert led.requests == 1 and clock.slept == []


def test_404_returns_other_4xx_invalid_and_network_errors_are_unavailable(tmp_path):
    c, _, _ = _client(tmp_path, lambda req: httpx.Response(404))
    assert c.get('https://h/x').status_code == 404
    c, _, _ = _client(tmp_path, lambda req: httpx.Response(400))
    with pytest.raises(N.InvalidResponse):
        c.get('https://h/x')
    n = []

    def h(req):
        n.append(1)
        if len(n) < 3:
            raise httpx.ReadTimeout('slow', request=req)
        return httpx.Response(200, content=b'ok')
    c, _, clock = _client(tmp_path, h, rps=1000)
    assert c.get('https://h/x').content == b'ok' and clock.slept == [5, 15]


def test_pacing_limits_requests_per_second(tmp_path):
    c, _, clock = _client(tmp_path, lambda req: httpx.Response(200), rps=2.0)
    for _ in range(4):
        c.get('https://h/x')
    assert clock.t == pytest.approx(1.5)                                                # 3 gaps of 0.5 s


def test_ledger_caps_stop_and_persist(tmp_path):
    led = N.Ledger(tmp_path / 'l.json', max_bytes=25, max_requests=3)
    c, _, _ = _client(tmp_path, lambda req: httpx.Response(200, content=b'x' * 10), rps=1000, ledger=led)
    c.get('https://h/a')
    c.get('https://h/b')
    with pytest.raises(N.BudgetExceeded):                                               # 20 + planned 10 > 25
        c.get('https://h/c', planned_bytes=10)
    c.get('https://h/c')
    with pytest.raises(N.BudgetExceeded):                                               # 3 requests used
        c.get('https://h/d')
    again = N.Ledger(tmp_path / 'l.json', max_bytes=25, max_requests=3)                 # cumulative across runs
    assert (again.bytes, again.requests) == (30, 3)
    lines = [json.loads(x) for x in (tmp_path / 'req.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [x['status'] for x in lines] == [200, 200, 200] and lines[-1]['ledger_requests'] == 3


# ---------------------------------------------------------------- RangeFile / index_scan on synthetic parquet

COLS = IX.COLUMNS


def _table(keys):
    return pa.table({'url_surtkey': keys, 'fetch_time': ['2026-09-01T00:00:00'] * len(keys), 'fetch_status': pa.array([200] * len(keys), pa.int32()),
                     'content_mime_type': ['text/html'] * len(keys), 'warc_filename': ['w.warc.gz'] * len(keys),
                     'warc_record_offset': pa.array(range(len(keys)), pa.int64()), 'warc_record_length': pa.array([5] * len(keys), pa.int64())})


def _parquet(keys, rg=5000):
    buf = io.BytesIO()
    pq.write_table(_table(keys), buf, row_group_size=rg, write_statistics=True)
    return buf.getvalue()


def _server(files, hits):
    def h(req):
        path = req.url.path.lstrip('/')
        if path.endswith('cc-index-table.paths.gz'):
            return httpx.Response(200, content=gzip.compress('\n'.join(files).encode()))
        if path not in files:
            return httpx.Response(404)
        data = files[path]
        m = re.fullmatch(r'bytes=(-?\d+)(?:-(\d*))?', req.headers.get('range', ''))
        if not m:
            return httpx.Response(200, content=data)
        a, b = m.group(1), m.group(2)
        if a.startswith('-'):
            start, end = max(0, len(data) + int(a)), len(data) - 1
        else:
            start, end = int(a), min(len(data) - 1, int(b) if b else len(data) - 1)
        hits.append((path, start, end))
        return httpx.Response(206, content=data[start:end + 1], headers={'content-range': f'bytes {start}-{end}/{len(data)}'})
    return h


KEYS = [f'vn,com,{c}{i:05d})/p' for c in 'abcdefgh' for i in range(30000)]


def _crawl_files(order=None):
    parts = [KEYS[i * 60000:(i + 1) * 60000] for i in range(4)]
    if order:
        parts = [parts[i] for i in order]
    return {f'cc-index/table/cc-main/warc/crawl=CC-MAIN-X/subset=warc/part-0000{i}.c000.gz.parquet': _parquet(p) for i, p in enumerate(parts)}


def test_range_file_reads_parquet_footer_and_row_groups_with_range_requests(tmp_path):
    files = _crawl_files()
    name = next(iter(files))
    hits = []
    c, led, _ = _client(tmp_path, _server(files, hits), rps=1000)
    pf, rf = IX.open_parquet(c, name)
    assert rf.size == len(files[name]) and pf.metadata.num_rows == 60000 and pf.metadata.num_row_groups == 12
    assert all(h[0] == name for h in hits) and led.bytes < len(files[name])
    assert pf.read_row_group(3, columns=list(COLS)).num_rows == 5000
    assert pf.metadata.serialized_size > 0 and IX.footer_info(pf)['missing_columns'] == []


def test_prefix_scan_binary_searches_sorted_files_and_reads_only_overlapping_row_groups(tmp_path):
    files = _crawl_files()
    hits = []
    c, led, _ = _client(tmp_path, _server(files, hits), rps=1000)
    paths = IX.list_parquet_paths(c, 'CC-MAIN-X')
    assert paths == sorted(files)
    opened = {}

    def probe(i):
        pf, _ = IX.open_parquet(c, paths[i])
        opened[i] = pf
        return IX.file_range(IX.key_ranges(pf))
    prefix = 'vn,com,e00123)/'                                                           # lives in file 2 (e and f keys)
    cands, seen, ok = IX.bisect_files(len(paths), probe, prefix, IX.upper_bound(prefix))
    assert ok and len(cands) == 1 and prefix[:-1] >= '' and len(seen) < len(paths) + 1
    table, info = IX.read_prefix(opened[cands[0]], prefix, max_bytes=10_000_000)
    assert table['url_surtkey'].to_pylist() == ['vn,com,e00123)/p'] and info['rows_matched'] == 1 and len(info['row_groups_read']) == 1
    total = sum(len(v) for v in files.values())
    assert led.bytes < total / 2                                                         # far less than the whole crawl


def test_unsorted_files_are_reported_not_trusted(tmp_path):
    files = _crawl_files(order=[2, 0, 3, 1])
    c, _, _ = _client(tmp_path, _server(files, []), rps=1000)
    paths = sorted(files)

    def probe(i):
        pf, _ = IX.open_parquet(c, paths[i])
        return IX.file_range(IX.key_ranges(pf))
    cands, seen, ok = IX.bisect_files(len(paths), probe, 'vn,com,c00001)/', IX.upper_bound('vn,com,c00001)/'))
    assert ok is False


def test_read_prefix_refuses_row_groups_over_the_byte_cap(tmp_path):
    files = _crawl_files()
    c, _, _ = _client(tmp_path, _server(files, []), rps=1000)
    pf, _ = IX.open_parquet(c, sorted(files)[0])
    with pytest.raises(IX.TooBig):
        IX.read_prefix(pf, 'vn,com,a0001)/', max_bytes=10)


# ---------------------------------------------------------------- step 0 end to end on a mock server

from functools import partial  # noqa: E402

from r2ai.cc_coverage import step0 as S0  # noqa: E402

HOST = 'vn,com,nhathuoclongchau)/'
CORPUS = [(1, 'https://nhathuoclongchau.com.vn/bai-viet/s1.html'), (2, 'https://nhathuoclongchau.com.vn/bai-viet/s2.html'),
          (3, 'https://nhathuoclongchau.com.vn/benh/b1.html'), (4, 'https://nhathuoclongchau.com.vn/bai-viet/s3.html')]
OLD = [(1, CORPUS[0][1], 'found', 'CC-MAIN-2026-39'), (2, CORPUS[1][1], 'not_observed', None), (3, CORPUS[2][1], 'found', 'CC-MAIN-2026-34'),
       (4, CORPUS[3][1], 'not_observed', None)]


def _t(rows):
    keys = [r[0] for r in rows]
    n = len(rows)
    return pa.table({'url_surtkey': keys, 'fetch_time': [r[1] for r in rows], 'fetch_status': pa.array([r[2] for r in rows], pa.int32()),
                     'content_mime_type': [r[3] for r in rows], 'warc_filename': ['w.warc.gz'] * n,
                     'warc_record_offset': pa.array(range(n), pa.int64()), 'warc_record_length': pa.array([r[4] for r in rows], pa.int64())})


def _pq(rows):
    buf = io.BytesIO()
    pq.write_table(_t(rows), buf, row_group_size=2, write_statistics=True)
    return buf.getvalue()


def _mock(sorted_layout=True, fail=None):
    info = [{'id': f'CC-MAIN-{y}-{w:02d}', 'from': f'{d}T00:00:00', 'to': f'{d}T23:00:00'} for (y, w, d) in
            [(2025, 38, '2025-09-14'), (2026, 17, '2026-04-10'), (2026, 21, '2026-05-08'), (2026, 25, '2026-06-05'),
             (2026, 30, '2026-07-10'), (2026, 34, '2026-08-07'), (2026, 39, '2026-09-04'), (2026, 43, '2026-10-05')]]
    data = {
        'CC-MAIN-2026-39': [(HOST + 'bai-viet/s1.html', '2026-09-08T09:00:00', 200, 'text/html', 55)],
        'CC-MAIN-2026-34': [(HOST + 'bai-viet/s2.html', '2026-08-08T09:00:00', 200, 'warc/revisit', 10),
                            (HOST + 'benh/b1.html', '2026-08-08T10:00:00', 200, 'text/html', 66)],
        'CC-MAIN-2026-30': [(HOST + 'benh/b1.html', '2026-07-12T10:00:00', 200, 'text/html', 61)]}
    files = {}
    for c in S0.CR.OLD_SIX:
        mid = sorted([r for r in data.get(c, [])] + [(HOST + 'zzz-other.html', '2026-01-01T00:00:00', 301, 'text/html', 1)])
        parts = [[('vn,a)/x', '2026-01-01T00:00:00', 200, 'text/html', 1), ('vn,b)/x', '2026-01-01T00:00:00', 200, 'text/html', 1)], mid,
                 [('vn,z)/x', '2026-01-01T00:00:00', 200, 'text/html', 1), ('vn,zz)/x', '2026-01-01T00:00:00', 200, 'text/html', 1)]]
        if not sorted_layout:
            parts = [parts[1], parts[0], parts[2]]
        for i, p in enumerate(parts):
            files[f'cc-index/table/cc-main/warc/crawl={c}/subset=warc/part-0000{i}.parquet'] = _pq(p)
    hits = []
    inner = _server(files, hits)

    def h(req):
        url = str(req.url)
        if url.endswith('collinfo.json'):
            return httpx.Response(200, content=json.dumps(info).encode())
        m = re.search(r'crawl-data/(CC-MAIN-[\d-]+)/cc-index-table.paths.gz', url)
        if m:
            names = sorted(f for f in files if f'crawl={m.group(1)}/' in f)
            return httpx.Response(200, content=gzip.compress('\n'.join(names).encode()))
        if fail and fail(req):
            return httpx.Response(403)
        return inner(req)
    return h


def _run(tmp_path, monkeypatch, handler, only=''):
    monkeypatch.setattr(S0, 'corpus_urls', lambda: CORPUS)
    monkeypatch.setattr(S0, 'old_sample', lambda: OLD)
    monkeypatch.setattr(S0, 'Client', partial(N.Client, transport=httpx.MockTransport(handler), sleep=lambda s: None))
    monkeypatch.setattr(N, 'BACKOFF_S', (0, 0, 0, 0, 0))
    rc = S0.main(['--ua', UA, '--out-dir', str(tmp_path / 'cc')] + (['--only', only] if only else []))
    return rc, json.loads((tmp_path / 'cc' / 'step0_report.json').read_text(encoding='utf-8'))


def test_step0_end_to_end_reports_layout_sample_and_by_product(tmp_path, monkeypatch):
    rc, rep = _run(tmp_path, monkeypatch, _mock())
    assert rc == 0 and rep['status'] == 'ok' and list(rep['crawls']) == list(reversed(S0.CR.OLD_SIX))
    c39 = rep['crawls']['CC-MAIN-2026-39']
    assert c39['file_index_in_key_order'] is True and c39['parquet_files'] == 3 and [h['file'] for h in c39['row_group_hits']] == [1]
    assert c39['footer_map']['adjacent_range_overlaps_when_sorted_by_min'] == 0 and c39['footer_map_from_cache'] is False
    assert rep['crawls']['CC-MAIN-2026-34']['revisit_rows'] == 1 and rep['crawls']['CC-MAIN-2026-34']['non_200_rows'] == 1
    assert [c['id'] for c in rep['selected_crawls']][-1] == 'CC-MAIN-2026-43' and rep['selected_crawls'][-1]['after_corpus'] is True
    cmpn = rep['sample_vs_old']
    assert (cmpn['n'], cmpn['old_found_any_crawl'], cmpn['old_records_in_processed_crawls'], cmpn['of_those_found_again']) == (4, 2, 2, 2)
    assert cmpn['new_found_in_processed'] == 2 and cmpn['new_found_but_old_not_found'] == [] and cmpn['old_in_scope_not_found_again'] == []
    bp = rep['by_product_processed_crawls']
    assert bp['with_200_html'] == 2 and bp['chosen_by_crawl'] == {'CC-MAIN-2026-34': 1, 'CC-MAIN-2026-39': 1}
    assert bp['warc_bytes_chosen'] == 55 + 66 and bp['by_prefix']['/benh'] == {'urls': 1, 'with_200_html': 1}
    assert rep['ledger']['requests'] == rep['step_requests'] > 0
    assert (tmp_path / 'cc' / 'rows_CC-MAIN-2026-39.parquet').exists() and (tmp_path / 'cc' / 'footer_map_CC-MAIN-2026-39.json').exists()


def test_step0_only_one_crawl_and_the_footer_map_cache_saves_requests(tmp_path, monkeypatch):
    rc, rep = _run(tmp_path, monkeypatch, _mock(), only='CC-MAIN-2026-39')
    assert rc == 0 and list(rep['crawls']) == ['CC-MAIN-2026-39'] and rep['sample_vs_old']['old_records_in_processed_crawls'] == 1
    first = rep['crawls']['CC-MAIN-2026-39']['requests']
    rc, rep2 = _run(tmp_path, monkeypatch, _mock(), only='CC-MAIN-2026-39')
    assert rc == 0 and rep2['crawls']['CC-MAIN-2026-39']['footer_map_from_cache'] is True
    assert rep2['crawls']['CC-MAIN-2026-39']['requests'] < first and rep2['by_product_processed_crawls'] == rep['by_product_processed_crawls']


def test_step0_unsorted_file_order_is_handled_by_the_footer_map(tmp_path, monkeypatch):
    rc, rep = _run(tmp_path, monkeypatch, _mock(sorted_layout=False), only='CC-MAIN-2026-39')
    c = rep['crawls']['CC-MAIN-2026-39']
    assert rc == 0 and c['file_index_in_key_order'] is False and [h['file'] for h in c['row_group_hits']] == [0]
    assert rep['by_product_processed_crawls']['with_200_html'] == 1                      # only s1 lives in crawl 39


def test_step0_403_stops_everything(tmp_path, monkeypatch):
    rc, rep = _run(tmp_path, monkeypatch, _mock(fail=lambda req: 'part-00001' in str(req.url)))
    assert rc == 2 and rep['status'] == 'stopped_blocked' and '403' in rep['reason']
