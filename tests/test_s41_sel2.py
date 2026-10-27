import pandas as pd

from r2ai.s41.sel2 import _split_module, choose, replace_doc_chunks, two_chunk_pairs

ROWS = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": [
    {"doc_id": 10, "chunk_text": "body10"}, {"doc_id": 10, "chunk_text": "win10"}, {"doc_id": 11, "chunk_text": "body11"}]}]


def test_two_chunk_pairs():
    assert two_chunk_pairs(ROWS) == {(1, 10): ("body10", "win10")}


def test_replace_doc_chunks_keeps_position_and_others():
    out = replace_doc_chunks(ROWS, {(1, 10): ["a", "b"]})
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["a", "b", "body11"]
    assert out[0]["relevant_docs"] == [10, 11]
    assert ROWS[0]["relevant_chunks"][0]["chunk_text"] == "body10"


def test_choose_theta_is_nth_p4b():
    el = pd.DataFrame({"query_id": [1, 1, 2, 2], "doc_id": [1, 2, 3, 4], "p4b": [0.9, 0.3, 0.7, 0.5]})
    theta, chosen = choose(el, 2)
    assert theta == 0.7 and set(chosen.doc_id) == {1, 3}


def test_split_module_in_half_mode():
    sc = _split_module()
    assert sc.MODE == "half" and sc.MAX_PIECES == 4
