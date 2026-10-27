import pandas as pd

from r2ai.s43 import enhedge


def lane():
    return pd.DataFrame({"query_id": [1, 1, 1, 1, 2, 2, 3], "doc_id": [11, 12, 13, 14, 21, 22, 31], "en_rank": [1, 2, 3, 4, 1, 2, 1],
                         "cosine_score": [0.70, 0.69, 0.60, 0.68, 0.50, 0.66, 0.40]})


def test_choose_threshold_then_top_n_in_lane_order():
    got = enhedge.choose(lane(), 2, 0.65)
    assert list(zip(got.query_id, got.doc_id)) == [(1, 11), (1, 12), (2, 22)]
    assert enhedge.choose(lane(), 0, 0.0).empty


def test_append_after_base_docs_and_chunks():
    rows = [{"id": 1, "relevant_docs": [5, 6], "relevant_chunks": [{"doc_id": 5, "chunk_text": "a"}]},
            {"id": 2, "relevant_docs": [22], "relevant_chunks": []},
            {"id": 3, "relevant_docs": [7], "relevant_chunks": []}]
    hedge = enhedge.choose(lane(), 3, 0.65)
    text = {11: "T11\n\nbody 11", 12: "T12\n\nbody 12", 14: "T14", 22: "T22"}
    res = enhedge.append(rows, hedge, text)
    assert rows[0]["relevant_docs"] == [5, 6, 11, 12, 14]
    assert rows[0]["relevant_chunks"][1:] == [{"doc_id": 11, "chunk_text": "T11\n\nbody 11"}, {"doc_id": 12, "chunk_text": "T12\n\nbody 12"},
                                              {"doc_id": 14, "chunk_text": "T14"}]
    assert rows[1]["relevant_docs"] == [22] and rows[1]["relevant_chunks"] == []      # already in the row: skipped
    assert rows[2]["relevant_docs"] == [7]
    assert res["added"] == 3 and res["already_present"] == 1 and res["queries_ge1"] == 1 and res["per_query_dist"] == {0: 2, 3: 1}
