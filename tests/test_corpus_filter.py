"""Filter a submission to corpus v2 (r2ai.submit.corpus_filter): synthetic rows, no real data."""
import json
import zipfile

import polars as pl
import pytest

from r2ai import corpus_v2 as C
from r2ai.submit import corpus_filter as F
from r2ai.submit.writer import format_submission

IN = [10, 11, 12]       # in v2
OUT = [90, 91]          # not in v2


@pytest.fixture
def ids(tmp_path):
    raw = tmp_path / 'raw.parquet'
    rows = [(10, 'https://thanhnien.vn/a'), (11, 'https://vinmec.com/vie/b'), (12, 'https://ask.39.net/c')]
    pl.DataFrame(rows, schema={'id': pl.Int64, 'url': pl.String}, orient='row').write_parquet(raw)
    p = tmp_path / 'ids.parquet'
    C.build(raw, p)
    return p


def _chunk(d, n=0):
    return {'doc_id': d, 'chunk_text': f'văn bản {d}/{n} 中文', 'chunk_order': n}


ROWS = [
    {'id': 1, 'relevant_docs': [10, 90, 11], 'relevant_chunks': [_chunk(10), _chunk(90), _chunk(11, 1)]},
    {'id': 2, 'relevant_docs': [12, 10], 'relevant_chunks': [_chunk(12)]},          # untouched row
    {'id': 3, 'relevant_docs': [90, 91], 'relevant_chunks': [_chunk(91), _chunk(90, 1)]},  # everything outside
]


def _write(tmp_path, as_zip):
    text = format_submission(ROWS)
    if as_zip:
        p = tmp_path / 'sub.zip'
        with zipfile.ZipFile(p, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('sub.json', text)
    else:
        p = tmp_path / 'sub.json'
        p.write_text(text, encoding='utf-8', newline='\n')
    return p, text


@pytest.mark.parametrize('as_zip', [True, False])
def test_filter_drops_outside_docs_and_chunks_keeps_rest(tmp_path, ids, as_zip):
    src, text = _write(tmp_path, as_zip)
    dst = tmp_path / 'out' / 'sub_v2.zip'
    st = F.filter_submission(src, dst, ids_path=ids)
    with zipfile.ZipFile(dst) as z:
        assert [(i.filename, i.compress_type) for i in z.infolist()] == [('sub_v2.json', zipfile.ZIP_LZMA)]
        out = z.read('sub_v2.json').decode('utf-8')
    got = json.loads(out)
    assert [r['id'] for r in got] == [1, 2, 3]
    assert got[0]['relevant_docs'] == [10, 11]
    assert got[0]['relevant_chunks'] == [_chunk(10), _chunk(11, 1)]
    assert got[1] == ROWS[1] and got[2] == {'id': 3, 'relevant_docs': [], 'relevant_chunks': []}
    # layout identical to the writer's, and an untouched row is byte-identical
    assert out == format_submission(got)
    assert out.split('\n')[2] + '\n' in text
    assert st == {
        'rows': 3, 'rows_changed': 2, 'rows_empty_docs': 1, 'docs': 7, 'docs_outside': 3, 'docs_distinct_outside': 2, 'docs_dropped_host': 0, 'chunks_dropped_host': 0,
        'chunks': 6, 'chunks_outside': 3, 'chunks_distinct_docs_outside': 2, 'docs_kept': 4, 'chunks_kept': 3,
        'zip_bytes': dst.stat().st_size, 'json_bytes': len(out.encode('utf-8')),
        'json_sha256': st['json_sha256'], 'src_json_bytes': len(text.encode('utf-8')),
    }
    assert len(st['json_sha256']) == 64


def test_filter_refuses_bad_zip_and_oversize(tmp_path, ids):
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as z:
        z.writestr('a.json', '[\n]\n')
        z.writestr('b.json', '[\n]\n')
    with pytest.raises(ValueError, match='exactly one'):
        F.filter_submission(bad, tmp_path / 'o_v2.zip', ids_path=ids)
    src, _ = _write(tmp_path, False)
    with pytest.raises(ValueError, match='limit'):
        F.filter_submission(src, tmp_path / 'o2_v2.zip', ids_path=ids, max_zip_bytes=10)
    assert not (tmp_path / 'o2_v2.zip').exists()


def test_filter_rejects_unexpected_layout(tmp_path, ids):
    p = tmp_path / 'x.json'
    p.write_text('[{"id": 1, "relevant_docs": [], "relevant_chunks": []}]\n', encoding='utf-8')
    with pytest.raises(ValueError, match='layout'):
        F.filter_submission(p, tmp_path / 'x_v2.zip', ids_path=ids)


def test_filter_big_lzma_dict_roundtrips(tmp_path, ids):
    src, _ = _write(tmp_path, False)
    dst = tmp_path / 'big_v2.zip'
    F.filter_submission(src, dst, ids_path=ids, lzma_dict_mib=16)
    with zipfile.ZipFile(dst) as z:
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert json.loads(z.read('big_v2.json'))[0]['relevant_docs'] == [10, 11]
    assert zipfile.LZMACompressor.__module__ == 'zipfile' and 'dict_size' not in zipfile.LZMACompressor._init.__code__.co_names


def test_filter_drop_hosts_removes_docs_and_chunks_of_that_host(tmp_path, ids):
    src, _ = _write(tmp_path, False)
    dst = tmp_path / 'novin_v2.zip'
    st = F.filter_submission(src, dst, ids_path=ids, drop_hosts=['vinmec.com'])
    with zipfile.ZipFile(dst) as z:
        got = json.loads(z.read('novin_v2.json'))
    assert got[0]['relevant_docs'] == [10]                       # 11 is vinmec, 90 outside
    assert got[0]['relevant_chunks'] == [_chunk(10)]
    assert got[1]['relevant_docs'] == [12, 10]                    # id 12 is 39.net, untouched
    assert (st['docs_dropped_host'], st['chunks_dropped_host']) == (1, 1)
    assert (st['docs_outside'], st['chunks_outside']) == (3, 3)
