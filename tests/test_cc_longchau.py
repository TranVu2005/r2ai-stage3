"""cc_longchau fetch/WARC parsing: no network, a mock transport serves synthetic gzip WARC records by Range."""
import gzip
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
    assert picks == [103, 104] and tail == slug + [103, 104] and removed == []
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
