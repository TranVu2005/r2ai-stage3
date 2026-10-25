import numpy as np
import pandas as pd
import pytest

from r2ai.s31.select import select_rrf
from r2ai.s35 import build as s35_build
from r2ai.s35 import jina, part


def _pools():
    rows = [{"pair_id": i, "doc_id": 100 + i, "cosine_score": c, "judge_score": j, "cosine_rank": r, "judge_rank": jr, "origin": "pool13"}
            for i, (c, j, r, jr) in enumerate([(0.9, 0.1, 1, 3), (0.8, 0.9, 2, 1), (0.8, 0.5, 3, 2), (0.5, 0.0, 4, 4)])]
    return {1: rows}


def test_add_part_cosine_max_ties_and_missing():
    pools = _pools()
    best = pd.Series({0: 0.70, 1: 0.95, 2: 0.40}, name="chunk_cosine")  # pair 0 and 2 stay; pair 1 jumps over; pair 3 has no chunk
    part.add_part_cosine(pools, best)
    got = {r["pair_id"]: (r["cos_doc2"], r["cos2_rank"]) for r in pools[1]}
    assert np.isnan(pools[1][3]["cos_part"]) and got[3][0] == 0.5
    assert got[1] == (0.95, 1) and got[0] == (0.9, 2)
    assert got[2] == (0.8, 3) and got[3][1] == 4


def test_cos_doc2_equal_to_doc_cosine_gives_back_cosine_rank():
    pools = _pools()
    part.add_part_cosine(pools, pd.Series({0: 0.1, 1: 0.2, 2: 0.3}, name="chunk_cosine"))  # chunks never beat the doc
    assert [r["cos2_rank"] for r in pools[1]] == [r["cosine_rank"] for r in pools[1]]


def test_select_rrf_accepts_cos2_first_key_and_matches_cosine_when_identical():
    rows = _pools()[1]
    for r in rows:
        r["cos2_rank"] = r["cosine_rank"]
    a = select_rrf(rows, 3, ("cosine_rank", "judge_rank"))
    b = select_rrf(rows, 3, ("cos2_rank", "judge_rank"))
    assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]
    assert [r["rrf_score"] for r in a] == [r["rrf_score"] for r in b]
    with pytest.raises(ValueError):
        select_rrf(rows, 3, ("judge_rank",))


def test_variants_follow_k4_thr_c_rule_with_swapped_rank_source():
    assert s35_build.VARIANTS["PART-thrC"][1:] == s35_build.VARIANTS["Check-PART-id"][1:] == ("judge_rank",)
    assert s35_build.VARIANTS["J3-thrC"] == ("cosine_rank", "judge_rank", "jina_rank")


@pytest.mark.parametrize("n,expected", [(1, 1), (50, 1), (51, 2), (211, 5), (339, 7)])
def test_layout_blocks_are_balanced_and_cover_every_doc(n, expected):
    for layout in ("b50cos", "b50s0", "b50s1"):
        blocks = jina.layout_blocks(n, 7, layout)
        assert len(blocks) == expected and max(len(b) for b in blocks) <= jina.MAX_BLOCK
        assert max(len(b) for b in blocks) - min(len(b) for b in blocks) <= 1
        assert sorted(np.concatenate(blocks).tolist()) == list(range(n))
    assert not np.array_equal(np.concatenate(jina.layout_blocks(n, 7, "b50s0")), np.concatenate(jina.layout_blocks(n, 8, "b50s0"))) or n == 1
    assert np.array_equal(np.concatenate(jina.layout_blocks(n, 7, "b50s0")), np.concatenate(jina.layout_blocks(n, 7, "b50s0")))
    with pytest.raises(ValueError):
        jina.layout_blocks(n, 7, "b51s0")


class _FakeModel:
    """Doc text '<k>' has embedding [k, 1, 0...]; the query embedding drifts with the block to exercise the merge."""

    def __init__(self):
        self.calls = []


def _fake_embed(model, query, docs):
    model.calls.append(len(docs))
    d = np.array([[float(t), 1.0] + [0.0] * 510 for t in docs], dtype=np.float32)
    return d, np.array([1.0, float(len(model.calls))] + [0.0] * 510, dtype=np.float32)


def test_score_query_merges_blocks_with_one_reference_vector():
    docs = [str(k) for k in range(120)]
    model = _FakeModel()
    res = jina.score_query(model, "q", docs, 3, "b40s0", embed=_fake_embed)
    assert model.calls == [40, 40, 40] and res["n_blocks"].tolist() == [3] * 120
    assert sorted(res["block"].tolist()) == [0] * 40 + [1] * 40 + [2] * 40
    # replay the merge: weights max((1 + cos) / 2) per block, one averaged query vector for every doc
    emb = np.array([[float(k), 1.0] for k in range(120)])
    q_blocks, weights = [], []
    for b, idx in enumerate(jina.layout_blocks(120, 3, "b40s0")):
        q = np.array([1.0, float(b + 1)])
        cos = emb[idx] @ q / (np.linalg.norm(emb[idx], axis=1) * np.linalg.norm(q))
        q_blocks.append(q)
        weights.append(((1.0 + cos) / 2.0).max())
    q = np.average(np.array(q_blocks), axis=0, weights=weights)
    expected = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q))
    np.testing.assert_allclose(res["score"], expected, atol=1e-6)
    assert np.isfinite(res["z_block"]).all()


def test_jina_hook_adds_rank_and_rejects_missing(tmp_path):
    ranked = {1: [{"query_id": 1, "doc_id": 10, "cosine_rank": 1, "judge_rank": 1}, {"query_id": 1, "doc_id": 11, "cosine_rank": 2, "judge_rank": 2},
                  {"query_id": 1, "doc_id": 12, "cosine_rank": 3, "judge_rank": 3}]}
    path = tmp_path / "scores.parquet"
    pd.DataFrame({"query_id": [1, 1, 1], "doc_id": [10, 11, 12], "score": [0.1, 0.9, 0.9]}).to_parquet(path)
    info = s35_build.jina_hook(path)(ranked)
    assert [r["jina_rank"] for r in ranked[1]] == [3, 1, 2]  # tie 11/12 by cosine rank
    assert info["pairs"] == 3
    pd.DataFrame({"query_id": [1], "doc_id": [10], "score": [0.1]}).to_parquet(path)
    with pytest.raises(ValueError, match="no Jina score"):
        s35_build.jina_hook(path)(ranked)
