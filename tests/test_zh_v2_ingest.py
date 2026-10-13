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
