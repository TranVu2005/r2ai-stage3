"""Corpus v2 membership (r2ai.corpus_v2): synthetic parquet, no network, no real data."""
import polars as pl
import pytest

from r2ai import corpus_v2 as C

ROWS = [
    (583, 'https://www.suckhoedoisong.vn/a.htm'),
    (584, 'https://vinmec.com/vie/bai-viet/x-vi'),
    (585, 'http://ask.39.net/question/1.html'),
    (586, 'https://120ask.com/question/2.htm'),
    (587, 'https://pubmed.ncbi.nlm.nih.gov/123/'),
    (7466161, 'https://Baby.39.net:443/z'),
]


def _raw(tmp_path, rows=ROWS):
    p = tmp_path / 'raw.parquet'
    pl.DataFrame(rows, schema={'id': pl.Int64, 'url': pl.String}, orient='row').write_parquet(p)
    return p


def test_host_of_strips_www_port_case():
    assert C.host_of('https://WWW.Example.COM:8080/x?y=1') == 'example.com'
    assert C.host_of('http://phunusuckhoe.giadinhonline.vn/a') == 'phunusuckhoe.giadinhonline.vn'


@pytest.mark.parametrize('host,lang', [
    ('pubmed.ncbi.nlm.nih.gov', 'en'),
    ('ask.39.net', 'zh'), ('39.net', 'zh'), ('baidianfeng.familydoctor.com.cn', 'zh'), ('test.pmphai.com', 'zh'),
    ('120ask.com', 'zh'), ('cnkang.com', 'zh'),
    ('vinmec.com', 'vi'), ('hellobacsi.com', 'vi'), ('thanhnien.vn', 'vi'), ('baocantho.com.vn', 'vi'),
    ('pmc-ecm-healthblog.beta.pharmacity.io', 'vi'),
])
def test_lang_of_host(host, lang):
    assert C.lang_of_host(host) == lang


def test_lang_of_host_unknown_raises_not_guess():
    with pytest.raises(ValueError):
        C.lang_of_host('unknown-medical-site.org')
    # suffix match must be a label boundary: not39.net is not 39.net
    with pytest.raises(ValueError):
        C.lang_of_host('not39.net')


def test_build_schema_lang_and_sorted(tmp_path):
    out = tmp_path / 'ids.parquet'
    df = C.build(_raw(tmp_path), out)
    assert df.columns == ['id', 'url', 'lang', 'host']
    assert df['id'].to_list() == sorted(df['id'].to_list())
    assert dict(zip(df['id'], df['lang'])) == {583: 'vi', 584: 'vi', 585: 'zh', 586: 'zh', 587: 'en', 7466161: 'zh'}
    assert df.filter(pl.col('id') == 7466161)['host'].item() == 'baby.39.net'
    assert pl.read_parquet(out).equals(df)


@pytest.mark.parametrize('rows,msg', [
    (ROWS + [(583, 'https://other.vn/dup-id')], 'id'),
    (ROWS + [(9, 'https://www.suckhoedoisong.vn/a.htm')], 'url'),
    (ROWS + [(9, 'https://nowhere.example.org/x')], 'lang'),
])
def test_build_rejects_bad_input(tmp_path, rows, msg):
    with pytest.raises(ValueError, match=msg):
        C.build(_raw(tmp_path, rows), tmp_path / 'ids.parquet')
    assert not (tmp_path / 'ids.parquet').exists()


def test_is_member_and_mask(tmp_path):
    out = tmp_path / 'ids.parquet'
    C.build(_raw(tmp_path), out)
    assert C.is_member(583, out) and C.is_member(7466161, out)
    assert not C.is_member(582, out) and not C.is_member(588, out) and not C.is_member(10 ** 12, out)
    # bool / non-int are never members (validator rejects them too)
    assert not C.is_member(True, out) and not C.is_member('583', out) and not C.is_member(583.0, out)
    assert C.member_mask([583, 1, 587, 99999999], out).tolist() == [True, False, True, False]
    assert C.member_mask([], out).tolist() == []


def test_load_ids_reloads_when_file_changes(tmp_path):
    out = tmp_path / 'ids.parquet'
    C.build(_raw(tmp_path), out)
    assert not C.is_member(9, out)
    C.build(_raw(tmp_path, ROWS + [(9, 'https://thanhnien.vn/n')]), out)
    assert C.is_member(9, out)
