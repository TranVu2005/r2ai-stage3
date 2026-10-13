import json
import sqlite3

import polars as pl
import pytest

from r2ai.zh_v2 import crawl as zc


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(zc, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(zc, 'RAW', tmp_path / 'raw')
    monkeypatch.setattr(zc, 'STATE', tmp_path / 'state')
    monkeypatch.setattr(zc, 'DB', tmp_path / 'state' / 'crawl.db')
    ids = pl.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'url': ['https://ask.39.net/question/1.html', 'http://ask.39.net/question/1.html', 'https://ask.39.net/question/2.html',
                'https://www.familydoctor.com.cn/a/1.html', 'https://zysjonline.com/books/x/1/', 'https://120ask.com/question/1.htm'],
        'lang': ['zh'] * 6,
        'host': ['ask.39.net', 'ask.39.net', 'ask.39.net', 'familydoctor.com.cn', 'zysjonline.com', '120ask.com']})
    p = tmp_path / 'ids.parquet'
    ids.write_parquet(p)
    return p, tmp_path


def test_prepare_loads_only_lane_hosts_groups_duplicates_and_is_idempotent(env):
    p, tmp = env
    out = zc.prepare(p)
    assert out['v2_ids'] == 4 and out['url_groups'] == 3                 # http/https twins of question/1 form one group
    assert out['domains'] == {'ask.39.net': 2, 'familydoctor.com.cn': 1}
    c = sqlite3.connect(zc.DB)
    rows = c.execute('SELECT url_norm,doc_ids,domain,rank,status FROM urls ORDER BY domain,rank').fetchall()
    assert [(r[2], r[3], r[4]) for r in rows] == [('ask.39.net', 0, 'pending'), ('ask.39.net', 1, 'pending'), ('familydoctor.com.cn', 0, 'pending')]
    assert json.loads(rows[0][1]) == [1, 2]
    assert dict(c.execute('SELECT domain,state FROM domains')) == {'ask.39.net': 'ready', 'familydoctor.com.cn': 'ready'}
    c.close()
    again = zc.prepare(p)
    assert again['rows_added_now'] == 0 and again['url_groups'] == 3


def test_robots_blocked_hosts_are_not_lane_hosts():
    assert 'zysjonline.com' not in zc.DOMAINS and 'baidu.com' not in zc.DOMAINS
    assert zc.DOMAINS.isdisjoint({'120ask.com', 'cnkang.com'})


def test_39net_and_familydoctor_families_share_one_lane():
    lane39 = [l for l in zc.LANES if 'baby.39.net' in l][0]
    assert 'heart.39.net' in lane39 and 'ask.39.net' not in lane39
    fam = [l for l in zc.LANES if 'familydoctor.com.cn' in l][0]
    assert 'baidianfeng.familydoctor.com.cn' in fam


def test_guard_blocks_paths_outside_the_zh_v2_crawl_roots(env):
    with pytest.raises(ValueError, match='Not a zh-v2 crawl output'):
        zc.guard(env[1] / 'elsewhere' / 'x.json')


def test_engine_runs_on_zh_v2_roots_not_zh_full(env):
    m = zc.engine()
    assert m.RUN == zc.RUN and m.RAW == zc.RAW and m.DB == zc.DB and m.LANES == zc.LANES
    from r2ai.zh_full import crawl as base
    assert base.RUN != zc.RUN and base.LANES != zc.LANES
