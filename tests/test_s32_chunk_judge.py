import numpy as np
import pandas as pd
import pytest

from r2ai.s32 import build as s32_build
from r2ai.s32.chunks import gather_index, segment_top, select_chunks
from r2ai.s32.scores import reduce_chunk_scores, spearman_table


def test_gather_index_concatenates_ranges():
    idx = gather_index(np.array([10, 0, 5]), np.array([2, 0, 3]))
    assert idx.tolist() == [10, 11, 5, 6, 7]
    assert gather_index(np.array([3]), np.array([0])).tolist() == []


def test_segment_top_ties_keep_file_order():
    scores = np.array([0.5, 0.9, 0.9, 0.1, 0.3, 0.3], dtype=np.float32)
    top = segment_top(scores, np.array([4, 2]), 3)
    assert top[0].tolist() == [1, 2, 0]  # exact tie 0.9 keeps the earlier chunk first
    assert top[1].tolist() == [0, 1]  # fewer chunks than top_n -> all of them


def test_select_chunks_top_n_per_pair_and_missing_docs():
    unit = np.eye(4, dtype=np.float16)
    # doc A: slots 0..2 (vectors e0, e1, e2); doc B: slot 3 (e3)
    vectors = unit[[0, 1, 2, 3]]
    pairs = pd.DataFrame({"query_id": [7, 7, 7], "slot_doc": [0, 1, -1]})
    q = np.array([0.1, 0.8, 0.6, 0.0], dtype=np.float16)
    out = select_chunks(pairs, np.array([0, 3]), np.array([3, 1]), vectors, {7: q}, top_n=2)
    assert out.pair_pos.tolist() == [0, 0, 1]
    assert out.slot.tolist() == [1, 2, 3]
    assert out.chunk_rank.tolist() == [1, 2, 1]
    assert out.n_chunks.tolist() == [3, 3, 1]
    assert out.chunk_cosine.iloc[0] == pytest.approx(0.8, abs=1e-3)


def _frames():
    chunk_pairs = pd.DataFrame({
        "pair_id": [0, 1, 2, 3, 4],
        "src_pair_id": [100, 100, 100, 101, 102],
        "query_id": [1, 1, 1, 1, 2],
        "doc_id": [10, 10, 10, 11, 12],
        "source": ["pool13"] * 5,
        "chunk_rank": [1, 2, 3, 1, 1],
        "field": ["title", "body", "body", "title", "body"],
    })
    scores = pd.DataFrame({"pair_id": [0, 1, 2, 3, 4], "score": [0.2, 0.9, 0.6, 0.4, 0.1]})
    fallback = pd.DataFrame({"src_pair_id": [103], "query_id": [2], "doc_id": [13], "source": ["pool13"]})
    q06 = pd.DataFrame({"src_pair_id": [100, 101, 102, 103], "q06": [0.5, 0.7, 0.05, 0.33]})
    return chunk_pairs, scores, fallback, q06


def test_reduce_chunk_scores_cmax_ctop2_cmix_and_fallback():
    out = reduce_chunk_scores(*_frames()).set_index("src_pair_id")
    assert out.loc[100, "cmax"] == pytest.approx(0.9)
    assert out.loc[100, "ctop2"] == pytest.approx((0.9 + 0.6) / 2)
    assert out.loc[100, "cmix"] == pytest.approx(0.9)
    assert out.loc[101, "ctop2"] == pytest.approx(0.4)  # single chunk: mean of the one score
    assert out.loc[101, "cmix"] == pytest.approx(0.7)  # max(0.4, q06 0.7)
    assert out.loc[103, "cmax"] == pytest.approx(0.33) and bool(out.loc[103, "fallback"])
    assert out.loc[100, "n_scored"] == 3 and out.loc[103, "n_scored"] == 0


def test_reduce_top2_mode_scores_only_ranks_one_and_two():
    chunk_pairs, scores, fallback, q06 = _frames()
    out = reduce_chunk_scores(chunk_pairs, scores[scores.pair_id != 2], fallback, q06).set_index("src_pair_id")
    assert out.loc[100, "n_scored"] == 2 and out.loc[100, "n_selected"] == 3
    assert out.loc[100, "cmax"] == pytest.approx(0.9)


def test_reduce_rejects_missing_q06_and_nonfinite():
    chunk_pairs, scores, fallback, q06 = _frames()
    with pytest.raises(ValueError):
        reduce_chunk_scores(chunk_pairs, scores, fallback, q06.iloc[:-1])
    bad = scores.copy()
    bad.loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        reduce_chunk_scores(chunk_pairs, bad, fallback, q06)


def test_spearman_table_global_and_per_query():
    frame = pd.DataFrame({"query_id": [1, 1, 1, 2, 2, 2], "a": [1, 2, 3, 1, 2, 3], "b": [3, 2, 1, 1, 2, 3]})
    table = spearman_table(frame, [("a", "b")])["a~b"]
    assert table["spearman_per_query_mean"] == pytest.approx(0.0)
    assert table["n"] == 6


def test_variant_table_rank_keys_start_with_cosine():
    for index, source, keys, ref in s32_build.VARIANTS.values():
        assert keys[0] == "cosine_rank" and index in ("pool13", "ph979") and source in ("q06", "chunk") and ref in ("q06", "j4")
    assert s32_build.VARIANT_KIND["C06-rrf-nt"] == "cmaxnt"
    assert s32_build.VARIANTS["C06-rrf-nt"][2] == s32_build.VARIANTS["C06-rrf"][2]
    assert s32_build.VARIANTS["C06-rrf4"][2] == ("cosine_rank", "q06_rank", "judge_rank")


def test_cmax_nt_skips_title_unless_only_title():
    out = reduce_chunk_scores(*_frames()).set_index("src_pair_id")
    # pair 100: title 0.2, bodies 0.9 and 0.6 -> best is a body chunk
    assert out.loc[100, "cmaxnt"] == pytest.approx(0.9) and not out.loc[100, "cmax_is_title"]
    # pair 101 has only a title chunk: keeps the title score
    assert out.loc[101, "cmaxnt"] == pytest.approx(0.4) and out.loc[101, "cmax_is_title"]
    # fallback: q06
    assert out.loc[103, "cmaxnt"] == pytest.approx(0.33)


def test_cmax_nt_differs_when_title_is_best():
    chunk_pairs, scores, fallback, q06 = _frames()
    scores = scores.copy()
    scores.loc[0, "score"] = 0.95  # title of pair 100 now the best chunk
    out = reduce_chunk_scores(chunk_pairs, scores, fallback, q06).set_index("src_pair_id")
    assert out.loc[100, "cmax"] == pytest.approx(0.95) and bool(out.loc[100, "cmax_is_title"])
    assert out.loc[100, "cmaxnt"] == pytest.approx(0.9)
