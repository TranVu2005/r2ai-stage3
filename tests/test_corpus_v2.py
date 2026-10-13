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


def test_host_ids_selects_by_host(tmp_path):
    out = tmp_path / 'ids.parquet'
    C.build(_raw(tmp_path), out)
    assert C.host_ids(['vinmec.com'], out).tolist() == [584]
    assert C.host_ids(['vinmec.com', 'ask.39.net'], out).tolist() == [584, 585]
    assert C.host_ids([], out).tolist() == []
    assert C.host_ids(['www.vinmec.com'], out).tolist() == [584]   # www. is ignored like in host_of


def _api(sha=C.HF_REVISION, corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE, query_sha=C.QUERY_SHA256):
    return {'sha': sha, 'siblings': [
        {'rfilename': C.RAW_NAME, 'size': corpus_size, 'lfs': {'sha256': corpus_sha, 'size': corpus_size}},
        {'rfilename': 'query.parquet', 'size': 1, 'lfs': {'sha256': query_sha, 'size': 1}},
    ]}


def _local(tmp_path, monkeypatch, content=b'raw-bytes'):
    import hashlib
    raw = tmp_path / 'raw.parquet'
    pl.DataFrame(ROWS, schema={'id': pl.Int64, 'url': pl.String}, orient='row').write_parquet(raw)
    ids = tmp_path / 'ids.parquet'
    C.build(raw, ids)
    monkeypatch.setattr(C, 'RAW_SHA256', hashlib.sha256(raw.read_bytes()).hexdigest())
    monkeypatch.setattr(C, 'RAW_SIZE', raw.stat().st_size)
    return raw, ids


def test_check_revision_ok(tmp_path, monkeypatch):
    raw, ids = _local(tmp_path, monkeypatch)
    ok, lines = C.check_revision(_api(corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE), raw, ids, None)
    assert ok, lines
    assert all(l.startswith('PASS') for l in lines)


@pytest.mark.parametrize('api_kw,needle', [
    ({'sha': 'f' * 40}, 'revision'),
    ({'corpus_sha': '0' * 64}, 'HF file sha256'),
])
def test_check_revision_flags_hf_change(tmp_path, monkeypatch, api_kw, needle):
    raw, ids = _local(tmp_path, monkeypatch)
    kw = {'corpus_sha': C.RAW_SHA256, 'corpus_size': C.RAW_SIZE, **api_kw}
    ok, lines = C.check_revision(_api(**kw), raw, ids, None)
    assert not ok
    assert any(l.startswith('FAIL') and needle in l for l in lines)


def test_check_revision_flags_local_drift(tmp_path, monkeypatch):
    raw, ids = _local(tmp_path, monkeypatch)
    api = _api(corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE)
    raw.write_bytes(raw.read_bytes() + b'x')                       # local raw no longer the pinned file
    ok, lines = C.check_revision(api, raw, ids, None)
    assert not ok and any(l.startswith('FAIL') and 'local raw sha256' in l for l in lines)
    raw2, ids2 = _local(tmp_path, monkeypatch)
    pl.read_parquet(ids2).head(3).write_parquet(ids2)             # ids parquet out of sync with raw
    ok, lines = C.check_revision(_api(corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE), raw2, ids2, None)
    assert not ok and any(l.startswith('FAIL') and 'ids parquet' in l for l in lines)


def test_check_revision_missing_files_fail_closed(tmp_path, monkeypatch):
    raw, ids = _local(tmp_path, monkeypatch)
    ok, lines = C.check_revision(_api(corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE), tmp_path / 'nope.parquet', ids, None)
    assert not ok and any(l.startswith('FAIL') and 'local raw' in l for l in lines)


def test_check_main_exit_codes(tmp_path, monkeypatch, capsys):
    raw, ids = _local(tmp_path, monkeypatch)
    monkeypatch.setattr(C, 'RAW_PATH', raw)
    monkeypatch.setattr(C, 'IDS_PATH', ids)
    monkeypatch.setattr(C, 'QUERY_PATH', tmp_path / 'absent_query.parquet')
    monkeypatch.setattr(C, 'fetch_hf_info', lambda: _api(corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE))
    assert C.main(['check']) == 0
    monkeypatch.setattr(C, 'fetch_hf_info', lambda: _api(sha='e' * 40, corpus_sha=C.RAW_SHA256, corpus_size=C.RAW_SIZE))
    assert C.main(['check']) == 1

    def boom():
        raise OSError('no network')
    monkeypatch.setattr(C, 'fetch_hf_info', boom)
    assert C.main(['check']) == 2          # cannot verify -> do not submit
    assert 'FAIL' in capsys.readouterr().out
