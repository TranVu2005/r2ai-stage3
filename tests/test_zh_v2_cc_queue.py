import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.zh_v2 import cc_queue as cq


def ids(tmp_path):
    p = tmp_path / 'ids.parquet'
    pl.DataFrame({'id': [1, 2, 3], 'url': ['https://zysjonline.com/books/a/1/', 'https://zysjonline.com/articles/b/2/', 'https://120ask.com/q/1.htm'],
                  'lang': ['zh'] * 3, 'host': ['zysjonline.com', 'zysjonline.com', '120ask.com']}).write_parquet(p)
    return p


def test_prefix_is_the_common_surt_prefix_of_the_host_urls(tmp_path):
    df = cq.host_urls(['zysjonline.com'], ids(tmp_path))
    assert df['surt'].to_list() == ['com,zysjonline)/books/a/1/', 'com,zysjonline)/articles/b/2/']
    assert cq.prefixes_of(df) == {'zysjonline.com': 'com,zysjonline)/'}


def test_row_groups_are_selected_by_key_statistics_or_missing_stats():
    files = [{'i': 0, 'groups': [{'g': 0, 'range': ['com,aaa)/', 'com,bbb)/'], 'rows': 1, 'cost': 10},
                                 {'g': 1, 'range': ['com,zysjonline)/a', 'com,zz)/'], 'rows': 1, 'cost': 20}]},
             {'i': 1, 'groups': [{'g': 0, 'range': None, 'rows': 1, 'cost': 5}]}]
    got = cq.hits(files, ['com,zysjonline)/'])
    assert [(h['file'], h['g']) for h in got] == [(0, 1), (1, 0)]


def test_queue_picks_newest_200_html_snapshot_and_skips_missing(tmp_path, monkeypatch):
    out = tmp_path / 'cc'
    out.mkdir()
    monkeypatch.setattr(cq, 'OUT', out)
    cols = ('url_surtkey', 'fetch_time', 'fetch_status', 'content_mime_type', 'warc_filename', 'warc_record_offset', 'warc_record_length')

    def rows(crawl, recs):
        pq.write_table(pa.table({c: [r[i] for r in recs] for i, c in enumerate(cols)}), out / f'rows_{crawl}_zysjonline.com.parquet')

    rows('CC-MAIN-2026-30', [('com,zysjonline)/books/a/1/', '2026-07-20T00:00:00', 200, 'text/html', 'w1', 10, 100)])
    rows('CC-MAIN-2026-34', [('com,zysjonline)/books/a/1/', '2026-08-20T00:00:00', 200, 'text/html', 'w2', 20, 200),
                             ('com,zysjonline)/articles/b/2/', '2026-08-20T00:00:00', 404, 'text/html', 'w3', 30, 300)])
    rep = cq.build_queue(['zysjonline.com'], out, ids(tmp_path))
    q = pl.read_parquet(out / 'cc_queue.parquet')
    assert q.height == 1 and q['id'][0] == 1 and q['crawl'][0] == 'CC-MAIN-2026-34' and q['warc_record_offset'][0] == 20
    assert rep['hosts']['zysjonline.com'] == {'v2_urls': 2, 'with_200_html': 1, 'warc_bytes': 200}
    assert rep['note'].startswith('metadata only')


def test_scan_refuses_more_than_2_rps():
    with pytest.raises(SystemExit):
        cq.main(['scan', '--hosts', 'zysjonline.com', '--rps', '3'])
