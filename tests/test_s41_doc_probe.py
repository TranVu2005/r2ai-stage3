import pytest

from r2ai.s41.doc_probe import per_query, subset_rows

ROWS = [
    {"id": 1, "relevant_docs": [10, 11, 12], "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "a"}, {"doc_id": 10, "chunk_text": "b"}, {"doc_id": 12, "chunk_text": "c"}]},
    {"id": 2, "relevant_docs": [20, 21], "relevant_chunks": [{"doc_id": 21, "chunk_text": "d"}]},
]


def test_subset_keeps_base_order_and_chunks_of_kept_docs():
    rows, info = subset_rows(ROWS, {1: {12, 10}, 2: {20}})
    assert rows[0]["relevant_docs"] == [10, 12]
    assert [c["chunk_text"] for c in rows[0]["relevant_chunks"]] == ["a", "b", "c"]
    assert rows[1] == {"id": 2, "relevant_docs": [20], "relevant_chunks": []}
    assert info == {"dropped_docs": 2, "dropped_chunks": 1}
    assert ROWS[0]["relevant_docs"] == [10, 11, 12]          # base rows untouched


def test_subset_refuses_probe_doc_outside_base():
    with pytest.raises(ValueError, match="outside the base docs"):
        subset_rows(ROWS, {1: {10, 99}, 2: {20}})


def test_subset_refuses_different_query_ids():
    with pytest.raises(ValueError, match="different query ids"):
        subset_rows(ROWS, {1: {10}})


def test_per_query_counts():
    rows, _ = subset_rows(ROWS, {1: {10, 11, 12}, 2: {20}})
    s = per_query(rows, "relevant_docs")
    assert (s["total"], s["mean"], s["min"], s["max"]) == (4, 2.0, 1, 3)
