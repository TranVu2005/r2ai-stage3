"""Per-host statistics of a submission (r2ai.submit.host_stats): synthetic rows."""
import polars as pl
import pytest

from r2ai import corpus_v2 as C
from r2ai.submit import host_stats as H
from r2ai.submit.writer import format_submission


@pytest.fixture
def ids(tmp_path):
    raw = tmp_path / 'raw.parquet'
    rows = [(10, 'https://thanhnien.vn/a'), (11, 'https://www.vinmec.com/vie/b'), (12, 'https://vinmec.com/vie/c')]
    pl.DataFrame(rows, schema={'id': pl.Int64, 'url': pl.String}, orient='row').write_parquet(raw)
    p = tmp_path / 'ids.parquet'
    C.build(raw, p)
    return p


def _c(d, text):
    return {'doc_id': d, 'chunk_text': text}


ROWS = [
    {'id': 1, 'relevant_docs': [10, 11, 12], 'relevant_chunks': [_c(10, 'abc'), _c(11, 'xy'), _c(11, 'đ')]},   # 'đ' = 2 bytes
    {'id': 2, 'relevant_docs': [10], 'relevant_chunks': [_c(10, 'abc')]},
    {'id': 3, 'relevant_docs': [11, 99], 'relevant_chunks': []},     # 99 is outside v2
]


def test_host_stats_counts_bytes_and_queries(tmp_path, ids):
    src = tmp_path / 'sub.json'
    src.write_text(format_submission(ROWS), encoding='utf-8', newline='\n')
    st = H.host_stats(str(src), ['vinmec.com'], ids_path=ids)
    assert st['queries'] == 3 and st['host'] == ['vinmec.com']
    h, t = st['host_stats'], st['total']
    assert (h['docs'], h['docs_distinct'], h['chunks'], h['chunk_text_bytes']) == (3, 2, 2, 2 + 2)
    assert (h['queries_with_doc'], h['queries_with_chunk']) == (2, 1)
    assert h['docs_per_query'] == pytest.approx(1.0) and h['chunks_per_query'] == pytest.approx(2 / 3)
    assert h['chunk_text_bytes_per_query'] == pytest.approx(4 / 3)
    assert (t['docs'], t['chunks'], t['chunk_text_bytes']) == (6, 4, 3 + 2 + 2 + 3)
    assert h['share_docs'] == pytest.approx(3 / 6) and h['share_chunks'] == pytest.approx(2 / 4)
    assert st['docs_outside_v2'] == 1
