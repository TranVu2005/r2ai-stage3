"""S36 fixed-depth pools: merged canonical + 120ask cosine top-D and the pairs it needs scored."""
import numpy as np
import pandas as pd

from r2ai.s36 import d300


def pool():
    return {"q_ids": np.array([1, 2]), "scores": np.array([[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], np.float32),
            "doc_ids": np.array([[10, 11, 12, 13], [20, 21, 22, 23]])}


def asks():
    # one segment: doc 900 scores 0.85 for query index 0 and 0.45 for query index 1; doc 901 ties canonical 11 (0.8) for query index 0
    return [(np.array([900, 900, 901]), np.array([0, 1, 0]), np.array([0.85, 0.45, 0.8]))]


def test_merged_top_interleaves_120ask_docs_by_cosine_and_cuts_at_depth():
    top = d300.merged_top(3, pool(), np.zeros((2, 4)), asks())
    first = top[top.query_id == 1]
    assert first.doc_id.tolist() == [10, 900, 11] and first.cosine_rank.tolist() == [1, 2, 3]       # 901 ties 11: canonical first, then cut
    assert first.is_ask.tolist() == [False, True, False]
    second = top[top.query_id == 2]
    assert second.doc_id.tolist() == [20, 900, 21] and second.is_ask.tolist() == [False, True, False]


def test_merged_top_refuses_a_depth_beyond_the_kept_cosines():
    try:
        d300.merged_top(5, pool(), np.zeros((2, 4)), [])
    except ValueError as err:
        assert "pool.npz" in str(err)
    else:
        raise AssertionError("no error")


def test_need_and_summary_count_the_pairs_without_a_score():
    top = d300.merged_top(3, pool(), np.zeros((2, 4)), asks())
    scored = {"4b": {(1, 10), (1, 11), (2, 20)}, "cmax": {(1, 10)}, "jina": {(1, 10), (1, 900)}}
    flagged = d300.need(top, scored)
    summary = d300.summarize(flagged, 3)
    assert summary["pairs"] == 6 and summary["pairs_ask"] == 2
    assert summary["pairs_4b_needed"] == 3 and summary["pairs_4b_needed_ask"] == 2 and summary["pairs_4b_needed_canonical"] == 1   # (1,900), (2,900), (2,21)
    assert summary["queries_needing_4b"] == 2
    assert summary["pairs_cmax_needed"] == 5 and summary["pairs_cmax_needed_with_4b"] == 2                                        # (1,11), (2,20) have 4B only
    assert summary["pairs_jina_missing"] == 4 and summary["pairs_jina_missing_ask"] == 1
    assert isinstance(flagged, pd.DataFrame) and {"has_4b", "has_cmax", "has_jina"} <= set(flagged.columns)


def test_ask_cosines_keeps_only_docs_above_the_query_cut(tmp_path):
    seg = tmp_path / "seg1"
    seg.mkdir()
    np.savez(seg / "vectors.npz", doc_id=np.array([7, 8]), dense=np.array([[1.0, 0.0], [0.0, 1.0]], np.float16))
    ids, qi, cos = d300.ask_cosines(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, 0.5]), tmp_path)[0]
    assert sorted(zip(ids.tolist(), qi.tolist())) == [(7, 0), (8, 1)] and np.allclose(cos, 1.0)
