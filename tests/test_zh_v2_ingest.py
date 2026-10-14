import json

import pytest

from r2ai.zh_full_ingest import ingest as base
from r2ai.zh_full_ingest.common import global_start
from r2ai.zh_v2 import ingest as zi
from r2ai.zh_v2 import index as zindex


@pytest.fixture
def roots(tmp_path):
    return zi.V2Roots(raw=tmp_path / 'raw', crawl_db=tmp_path / 'crawl.db', docs=tmp_path / 'docs', chunks=tmp_path / 'chunks',
                      index=tmp_path / 'index', run=tmp_path / 'run', state=tmp_path / 'state', forbidden=())


def test_segment_ids_start_at_100_and_never_touch_zh_full_ids(roots):
    assert zi.init_manifest(roots) == 99
    seg = base.allocate_segment(roots, 'v2n-001', 'ask.39.net')
    assert seg['segment_id'] == 100 and seg['name'] == 'seg00100-v2n-001-ask.39.net'
    assert global_start(100) > global_start(31)
    assert zi.init_manifest(roots) == 100                                 # not reset once segments exist


def test_default_roots_are_private_and_block_zh_full_outputs():
    r = zi.V2Roots()
    assert r.index == zindex.ZH_V2_INDEX_ROOT and r.raw.name == 'raw_zh_v2'
    with pytest.raises(ValueError, match='protected input'):
        r.guard(r.raw.parent / 'raw_zh_full' / 'x')
    with pytest.raises(ValueError, match='protected input'):
        r.guard(r.index.parent / 'zh_full_segments' / 'seg00002' / 'x')
    r.guard(r.chunks / 'seg00100-x' / 'docs.parquet')


def test_extractor_is_routed_to_zh_v2(monkeypatch):
    from r2ai.zh_v2 import extract
    monkeypatch.setattr(base, 'zf', base.zf)             # restore after the test
    zi.use_v2_extractor()
    assert base.zf is extract


def _zf_roots(tmp_path):
    from r2ai.zh_full_ingest.common import Roots
    return Roots(raw=tmp_path / 'zf_raw', crawl_db=tmp_path / 'zf_crawl.db', docs=tmp_path / 'zf_docs', chunks=tmp_path / 'zf_chunks',
                 index=tmp_path / 'zf_index', run=tmp_path / 'zf_run', state=tmp_path / 'zf_state', forbidden=())


def test_cut_zh_full_takes_only_rows_with_a_v2_id_and_never_a_taken_url(tmp_path, roots):
    import sqlite3
    zf = _zf_roots(tmp_path)
    d = zf.raw / 'cnkang.com'
    d.mkdir(parents=True)
    for name in ('s1.jsonl.zst', 's2.jsonl.zst'):
        (d / name).write_bytes(b'x' + name.encode())
    c = sqlite3.connect(zf.crawl_db)
    c.execute('CREATE TABLE urls (url_norm TEXT, domain TEXT, status TEXT, shard TEXT, doc_ids TEXT)')
    c.executemany('INSERT INTO urls VALUES (?,?,?,?,?)', [
        ('u-v2', 'cnkang.com', 'ok', 's1.jsonl.zst', '[1, 99]'),          # one v2 id among others: kept
        ('u-out', 'cnkang.com', 'ok', 's1.jsonl.zst', '[98, 99]'),        # no v2 id: left alone
        ('u-taken', 'cnkang.com', 'ok', 's2.jsonl.zst', '[2]'),           # v2 but already ingested by zh_full: left alone
        ('u-thin', 'cnkang.com', 'thin', 's2.jsonl.zst', '[3]'),
        ('u-pending', 'cnkang.com', 'pending', None, '[4]')])
    c.commit()
    c.close()
    m = base.connect(zf)
    m.execute("INSERT INTO batches VALUES ('b002','full',0,0,'[]','{}','{}')")
    m.execute("INSERT INTO batch_urls VALUES ('u-taken','b002','cnkang.com','p')")
    m.commit()
    m.close()
    zi.init_manifest(roots)
    out = zi.cut_zh_full(roots, 'v2n-t', ['cnkang.com'], zf=zf, v2_ids=[1, 2, 3, 4])
    assert out['rows_selected'] == 2 and out['urls'] == 2          # u-v2 and u-thin
    c = base.connect(roots)
    got = {r[0] for r in c.execute('SELECT url_norm FROM batch_urls')}
    c.close()
    assert got == {'u-v2', 'u-thin'}


class _L(list):
    pass


def _adopt_setup(monkeypatch, tmp_path, roots, indexed, new):
    import json
    import polars as pl
    monkeypatch.setattr(zindex, 'MANIFEST', tmp_path / 'manifest.json')
    monkeypatch.setattr(zindex, 'CHUNKS_V2', tmp_path / 'v2chunks')
    (tmp_path / 'v2chunks' / 'old').mkdir(parents=True)
    pl.DataFrame({'doc_id': indexed}).write_parquet(tmp_path / 'v2chunks' / 'old' / 'docs.parquet')
    (tmp_path / 'manifest.json').write_text(json.dumps({'segments': [{'name': 'old', 'segment_id': 6}]}), encoding='utf-8')
    segs = []
    for i, ids in enumerate(new):
        name = f'seg{100 + i:05d}-t'
        (roots.chunks / name).mkdir(parents=True)
        pl.DataFrame({'doc_id': ids}).write_parquet(roots.chunks / name / 'docs.parquet')
        segs.append({'segment_id': 100 + i, 'name': name, 'batch': 'b', 'domain': 'd', 'kind': 'x', 'status': 'active', 'rows': 10})
    monkeypatch.setattr(base, 'segment_rows', lambda r: segs)
    written, written_sup = _L(), []
    monkeypatch.setattr(zindex, 'write_manifest', lambda rows, ids, sup=None: written.append(rows) or written_sup.append(sup))
    monkeypatch.setattr(zindex.scope, 'v2_zh', lambda: __import__('polars').DataFrame({'id': [1]}))
    written.sup = written_sup
    return written


def test_adopt_skips_a_segment_whose_docs_are_all_already_indexed(monkeypatch, tmp_path, roots):
    written = _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2], new=[[1, 2], [7, 8]])
    out = zi.adopt(roots)
    assert out['skipped_duplicate'] == ['seg00100-t'] and out['added'] == ['seg00101-t'] and out['docs'] == 2
    assert [r['name'] for r in written[0]] == ['old', 'seg00101-t']


def test_adopt_refuses_a_segment_that_is_only_partly_duplicate(monkeypatch, tmp_path, roots):
    written = _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2], new=[[2, 3]])
    with pytest.raises(ValueError, match='already indexed'):
        zi.adopt(roots)
    assert written == []


def _declare(roots, batch, ids):
    roots.state.mkdir(parents=True, exist_ok=True)
    (roots.state / f'supersede_{batch}.json').write_text(json.dumps({'batch': batch, 'doc_ids': ids}), encoding='utf-8')


def test_adopt_records_declared_supersede_in_the_manifest(monkeypatch, tmp_path, roots):
    written = _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2, 3], new=[[2, 3, 9]])
    _declare(roots, 'b', [2, 3])
    out = zi.adopt(roots)
    assert out['superseded'] == 2 and out['added'] == ['seg00100-t'] and out['skipped_duplicate'] == []
    assert written.sup[0] == [{'doc_id': 2, 'segment': 'old', 'by': 'seg00100-t', 'batch': 'b'},
                              {'doc_id': 3, 'segment': 'old', 'by': 'seg00100-t', 'batch': 'b'}]


def test_adopt_refuses_overlap_that_the_declaration_does_not_cover(monkeypatch, tmp_path, roots):
    written = _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2, 3], new=[[2, 3]])
    _declare(roots, 'b', [2])                       # doc 3 is also in the old segment but was not declared
    with pytest.raises(ValueError, match='not in the supersede declaration'):
        zi.adopt(roots)
    assert written == []


def test_declaration_does_not_relax_the_guard_for_other_batches(monkeypatch, tmp_path, roots):
    _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2], new=[[2, 3]])
    _declare(roots, 'other-batch', [2, 3])          # declared for a different batch: the partial overlap is still an error
    with pytest.raises(ValueError, match='already indexed'):
        zi.adopt(roots)


def test_adopt_does_not_see_docs_that_an_earlier_supersede_removed(monkeypatch, tmp_path, roots):
    import polars as pl
    written = _adopt_setup(monkeypatch, tmp_path, roots, indexed=[1, 2], new=[[2]])
    m = json.loads((tmp_path / 'manifest.json').read_text('utf-8'))
    m['superseded'] = [{'doc_id': 2, 'segment': 'old', 'by': 'x', 'batch': 'b0'}]
    (tmp_path / 'manifest.json').write_text(json.dumps(m), encoding='utf-8')
    out = zi.adopt(roots)                            # doc 2 no longer lives in 'old', so the new segment is not a duplicate
    assert out['added'] == ['seg00100-t'] and out['superseded'] == 0


def test_cut_redo_takes_only_the_listed_docs_from_raw_and_writes_the_declaration(tmp_path):
    import sqlite3
    redo = zi.V2Roots(raw=tmp_path / 'raw', docs=tmp_path / 'docs', chunks=tmp_path / 'chunks', index=tmp_path / 'index',
                      run=tmp_path / 'run', state=tmp_path / 'state_redo', forbidden=())
    d = redo.raw / 'ask.39.net'
    d.mkdir(parents=True)
    (d / 's1.jsonl.zst').write_bytes(b'shard-one')
    db = tmp_path / 'crawl.db'
    c = sqlite3.connect(db)
    c.execute('CREATE TABLE urls (url_norm TEXT, domain TEXT, status TEXT, shard TEXT, doc_ids TEXT)')
    c.executemany('INSERT INTO urls VALUES (?,?,?,?,?)', [
        ('u1', 'ask.39.net', 'ok', 's1.jsonl.zst', '[11, 12]'), ('u2', 'ask.39.net', 'ok', 's1.jsonl.zst', '[21]'),
        ('u3', 'ask.39.net', 'ok', 's1.jsonl.zst', '[31]')])
    c.commit()
    c.close()
    out = zi.cut_redo(redo, 'redo-1', [11, 31, 99], ['ask.39.net'], 'test', crawl_db=db)
    assert out['urls'] == 2 and out['docs_declared'] == 2 and out['docs_not_found'] == 1      # 99 is not a crawled doc
    decl = json.loads((redo.state / 'supersede_redo-1.json').read_text('utf-8'))
    assert decl['doc_ids'] == [11, 31] and decl['batch'] == 'redo-1'
    c = base.connect(redo)
    assert {r[0] for r in c.execute('SELECT url_norm FROM batch_urls')} == {'u1', 'u3'}
    c.close()
    assert zi.init_manifest(redo) == 199                                                        # own segment-id range, from 200
