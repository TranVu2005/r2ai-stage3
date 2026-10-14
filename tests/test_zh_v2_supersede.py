"""Explicit supersede: a doc listed in the manifest stops being read from its old segment, every other doc is untouched."""
import json

import numpy as np
import polars as pl
import pyarrow as pa
from scipy.sparse import csr_matrix

from r2ai.index import segments as seg
from r2ai.zh_v2 import index as zi
from r2ai.zh_v2 import retrieve as zr

DOCS = np.array([10, 10, 20, 20, 20, 30, 40, 40])        # doc 20 is superseded; rows 2..4
START = 1000


def whole_segment():
    rng = np.random.RandomState(0)
    dense = rng.randn(len(DOCS), 8).astype(np.float16)
    sparse = csr_matrix(rng.rand(len(DOCS), 6).astype(np.float32))
    return seg.virtual_segments(dense, sparse, DOCS, count=1, block_rows=4, global_start=START)[0], dense


def test_superseded_doc_rows_are_left_out_and_other_rows_keep_their_global_ids():
    s, dense = whole_segment()
    pieces = zr.without_superseded('old', s, {20})
    assert [(p.global_start, len(p)) for p in pieces] == [(START, 2), (START + 5, 3)]
    assert [list(p.doc_id) for p in pieces] == [[10, 10], [30, 40, 40]]
    for p in pieces:
        assert np.array_equal(np.asarray(p.dense[0:len(p)]), dense[p.row_start:p.row_start + len(p)])
    assert zr.without_superseded('old', s, set()) == [s] and len(zr.without_superseded('old', s, {99})) == 1


def test_superseded_doc_never_appears_in_a_scan_and_the_others_score_as_before():
    s, dense = whole_segment()
    q = dense[3:4].astype(np.float16)                         # the query is a row of the superseded doc: it would rank first
    qsp = [(np.array([1, 2]), np.array([1., 1.], np.float32))]
    full = seg.scan_topk([s], q, qsp, 8, block_rows=4, query_batch=1)
    cut = seg.scan_topk(zr.without_superseded('old', s, {20}), q, qsp, 8, block_rows=4, query_batch=1)
    for branch in ('dense', 'sparse'):
        assert {1002, 1003, 1004} & set(full[branch][0][0].tolist())
        assert not {1002, 1003, 1004} & set(cut[branch][0][0].tolist())
        want = {g: sc for g, sc in zip(full[branch][0][0].tolist(), full[branch][1][0].tolist()) if g not in (1002, 1003, 1004)}
        got = dict(zip(cut[branch][0][0].tolist(), cut[branch][1][0].tolist()))
        assert got == want                                    # same ids, same scores for every other row of the segment


def test_texts_of_a_view_come_from_the_right_rows_of_the_segment_text_file(tmp_path):
    s, _ = whole_segment()
    table = pa.table({'text': [f't{i}' for i in range(len(DOCS))]})
    with pa.OSFile(str(tmp_path / 'text.arrow'), 'wb') as f, pa.ipc.new_file(f, table.schema) as w:
        w.write_table(table)
    pieces = zr.without_superseded('old', s, {20})
    for p in pieces:
        p.text_path = tmp_path / 'text.arrow'
    assert zr.take_texts_v2(pieces, [START + 7, START, START + 5]) == ['t7', 't0', 't5']


def test_source_gids_of_a_view_follow_the_row_map_of_the_whole_segment(tmp_path, monkeypatch):
    s, _ = whole_segment()
    monkeypatch.setattr(zi, 'CHUNKS_V2', tmp_path)
    (tmp_path / 'old').mkdir()
    pl.DataFrame({'new_local': list(range(8)), 'old_local': [0, 2, 3, 5, 6, 8, 9, 11]}).write_parquet(tmp_path / 'old' / 'row_map.parquet')
    pieces = zr.without_superseded('old', s, {20})
    gids = np.array([START + 1, START + 5, START + 7])        # rows 1, 5, 7 of the segment
    assert zr.source_gids(['old'] * len(pieces), pieces, gids).tolist() == [START + 2, START + 8, START + 11]


def test_doc_groups_and_live_total_exclude_superseded_docs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(zi, 'CHUNKS_V2', tmp_path)
    for name, ids in (('old', [10, 20, 30]), ('new', [20])):
        (tmp_path / name).mkdir()
        pl.DataFrame({'doc_id': ids, 'doc_ids_group': [[i] for i in ids], 'domain': 'd', 'url': 'u', 'title': 't', 'doc_text': 'x'}
                     ).write_parquet(tmp_path / name / 'docs.parquet')
    manifest = {'segments': [{'name': 'old', 'docs_v2': 3}, {'name': 'new', 'docs_v2': 1}],
                'superseded': [{'doc_id': 20, 'segment': 'old', 'by': 'new', 'batch': 'b'}]}
    groups = zr.doc_groups(['old', 'old', 'new'], manifest)      # a segment split into views repeats its name
    assert sorted(groups) == [10, 20, 30] and groups[20]['segment'] == 'new'
    assert zi.live_docs_total(manifest) == 3
    assert zi.segment_docs(manifest, 'old')['doc_id'].to_list() == [10, 30]
    assert zi.segment_docs(manifest, 'new')['doc_id'].to_list() == [20]


def test_build_keeps_the_superseded_list(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    monkeypatch.setattr(zi, 'MANIFEST', path)
    monkeypatch.setattr(zi, 'guard', lambda p: p)
    monkeypatch.setattr(zi.scope, 'IDS_PATH', tmp_path / 'ids.bin')
    (tmp_path / 'ids.bin').write_bytes(b'x')
    zi.write_manifest([], np.array([1]), [{'doc_id': 1, 'segment': 'a', 'by': 'b', 'batch': 'c'}])
    assert json.loads(path.read_text('utf-8'))['superseded'] == [{'doc_id': 1, 'segment': 'a', 'by': 'b', 'batch': 'c'}]
