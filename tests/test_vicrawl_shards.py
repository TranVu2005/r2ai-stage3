from r2ai.paths import ROOT

import os

import zstandard

from vicrawl.shards import cleanup_tmp, read_shard, write_shard


def recs(n, start=0):
    return [{'url_norm': f'x.vn/{i}', 'html': f'<p>Việt Nam {i}</p>', 'fetched_at': 1.0 + i} for i in range(start, start + n)]


def test_roundtrip_and_final_name(tmp_path):
    p = write_shard(tmp_path, 'x.vn', recs(5))
    assert p.name.endswith('.jsonl.zst') and p.parent == tmp_path / 'x.vn'
    assert not list(tmp_path.rglob('*.tmp'))
    assert [r['url_norm'] for r in read_shard(p)] == [f'x.vn/{i}' for i in range(5)]
    assert read_shard(p)[0]['html'] == '<p>Việt Nam 0</p>'


def test_single_complete_frame(tmp_path):
    p = write_shard(tmp_path, 'x.vn', recs(3))
    out = zstandard.ZstdDecompressor().decompress(p.read_bytes())
    assert len(out.splitlines()) == 3


def test_unique_names(tmp_path):
    names = {write_shard(tmp_path, 'x.vn', recs(1)).name for _ in range(20)}
    assert len(names) == 20


def test_cleanup_tmp_removes_partial(tmp_path):
    d = tmp_path / 'x.vn'
    d.mkdir()
    (d / 'a.jsonl.zst.tmp').write_bytes(b'garbage')
    keep = write_shard(tmp_path, 'x.vn', recs(1))
    assert cleanup_tmp(tmp_path) == 1
    assert keep.exists() and not (d / 'a.jsonl.zst.tmp').exists()


def test_failure_mid_write_leaves_no_final_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError('disk')
    monkeypatch.setattr(os, 'replace', boom)
    try:
        write_shard(tmp_path, 'x.vn', recs(2))
    except OSError:
        pass
    assert not list(tmp_path.rglob('*.jsonl.zst'))


def test_roundtrip_unicode_line_separators(tmp_path):
    html = 'a b c\x85d\x0be\x0cf\x1cg'
    p = write_shard(tmp_path, 'x.vn', [{'url_norm': 'x.vn/1', 'html': html}, {'url_norm': 'x.vn/2', 'html': 'z'}])
    got = read_shard(p)
    assert [r['url_norm'] for r in got] == ['x.vn/1', 'x.vn/2']
    assert got[0]['html'] == html
