import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.s27_pharmacity.handoff import _prepare_chunk_input, _resumable_lot
from r2ai.s27_pharmacity.pipeline import Roots, _next_handoff_ids


def test_handoff_ids_are_sorted_unique_and_skip_previous_lots():
    assert _next_handoff_ids([9, 3, 7, 3, 2], [3, 8], limit=10) == [2, 7, 9]


def test_handoff_ids_stop_at_the_requested_lot_size():
    assert _next_handoff_ids(range(9, 0, -1), [], limit=4) == [1, 2, 3, 4]


@pytest.mark.parametrize('limit', [0, -1, 1001])
def test_handoff_lot_limit_must_be_between_one_and_one_thousand(limit):
    with pytest.raises(ValueError, match='1..1000'):
        _next_handoff_ids([1, 2], [], limit=limit)


def test_chunker_input_contains_only_s15_docs_not_provenance(tmp_path):
    roots = Roots(root=tmp_path / 'batch')
    lot_root = roots.root / 'handoff' / 'lot-0001'
    docs_dir = lot_root / 'docs'
    docs_dir.mkdir(parents=True)
    docs_file = docs_dir / 'docs.parquet'
    pq.write_table(pa.table({'doc_ids': [[11]], 'fetched_at': [1.0]}), docs_file)
    pq.write_table(pa.table({'doc_id': [11], 'fetched_at': ['2026-10-19T00:00:00Z']}), docs_dir / 'provenance.parquet')

    chunk_input = _prepare_chunk_input(roots, lot_root, docs_file)

    assert [p.name for p in chunk_input.glob('*.parquet')] == ['docs.parquet']
    assert pq.read_table(chunk_input / 'docs.parquet').schema.field('fetched_at').type == pa.float64()


def test_failed_lot_can_resume_only_when_ids_match(tmp_path):
    roots = Roots(root=tmp_path / 'batch')
    lots_root = roots.root / 'handoff' / 'lots'
    partial = lots_root / 'lot-0001' / 'docs'
    partial.mkdir(parents=True)
    pq.write_table(pa.table({'doc_ids': [[1, 2]]}), partial / 'docs.parquet')

    assert _resumable_lot(roots, lots_root, [1, 2]).name == 'lot-0001'
    assert _resumable_lot(roots, lots_root, [1, 3]) is None
