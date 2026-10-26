import json

import numpy as np
import pandas as pd
import pytest

from r2ai import newq


def test_read_queries_checks_and_sorts(tmp_path):
    p = tmp_path / "q.parquet"
    pd.DataFrame({"id": [3, 1, 2], "query": ["c", "a", "b"], "extra": [0, 0, 0]}).to_parquet(p)
    q = newq.read_queries(p)
    assert q.id.tolist() == [1, 2, 3] and list(q.columns) == ["id", "query"]
    pd.DataFrame({"id": [1, 1], "query": ["a", "b"]}).to_parquet(p)
    with pytest.raises(SystemExit):
        newq.read_queries(p)
    pd.DataFrame({"id": [1, 2], "query": ["a", " "]}).to_parquet(p)
    with pytest.raises(SystemExit):
        newq.read_queries(p)


def test_split_queries_balanced_contiguous():
    per = pd.Series([300] * 10, index=range(101, 111))
    blocks = newq.split_queries(per, 3)
    assert sum(blocks, []) == list(range(101, 111))
    assert max(len(b) for b in blocks) - min(len(b) for b in blocks) <= 1
    assert newq.split_queries(per, 1) == [list(range(101, 111))]


def test_notebook_only_tag_expect_dataset_change(tmp_path):
    ds = tmp_path / "kaggle-dataset-x-4b-1"
    nb = newq.notebook_for({"x-4b-reg": 7, "x-4b-1": 123}, ds)
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert 'TAGS = ["x-4b-reg", "x-4b-1"]' in src and 'EXPECT = {"x-4b-reg": 7, "x-4b-1": 123}' in src
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "final-4b" not in code and "x-4b-1/qwen3-reranker-4b/parts" in src
    assert all(c.get("outputs", []) == [] for c in nb["cells"] if c["cell_type"] == "code")


def test_pool_frame_rrf_and_missing_rule():
    rows = [{"doc_id": d, "cosine_rank": r, "cosine_score": 1 - r / 10, "judge_score": p, "judge_rank": jr}
            for d, r, p, jr in ((10, 1, 0.2, 2), (11, 2, 0.9, 1), (12, 3, 0.1, 3))]
    jina = {(1, 10): 0.5, (1, 11): 0.7, (1, 12): 0.1}
    f = newq.PoolFrame(("cosine_rank", "judge_rank", "jina_rank"), {1: rows}, jina, {}, allow_missing=False)
    assert [r["doc_id"] for r in f.order[1]] == [11, 10, 12]
    with pytest.raises(ValueError):
        newq.PoolFrame(("cosine_rank", "judge_rank", "cmax_rank"), {1: [dict(r) for r in rows]}, jina, {}, allow_missing=False)
    g = newq.PoolFrame(("cosine_rank", "judge_rank"), {1: [dict(r) for r in rows]}, {}, {}, allow_missing=False)   # sources left out: no check
    assert g.missing["jina"] == 3 and g.qids == [1]


def test_pools_from_imputes_missing_4b_last_in_4b_source():
    top = pd.DataFrame({"pair_id": [1, 2, 3, 4], "query_id": [7, 7, 7, 7], "doc_id": [10, 11, 12, 13],
                        "cosine_rank": [1, 2, 3, 4], "cosine_score": [0.9, 0.8, 0.7, 0.6]})
    scores = {1: 0.3, 3: 0.8, 4: 0.1}
    with pytest.raises(SystemExit):
        newq.pools_from(top, scores)
    a, b = 0.5, -1.4
    imp = {"c": 10.0, "d": -8.0, "a": a, "b": b}
    rows = {r["doc_id"]: r for r in newq.pools_from(top, scores, imp)[7]}
    assert rows[11]["p4b_imputed"] and not rows[10]["p4b_imputed"]
    assert rows[11]["judge_rank"] == 4 and [rows[d]["judge_rank"] for d in (12, 10, 13)] == [1, 2, 3]
    from r2ai.s36 import cal
    p_cal = cal.sigmoid(a * cal.logit(np.array([rows[11]["judge_score"]])) + b)[0]
    assert abs(p_cal - cal.sigmoid(10.0 * 0.8 - 8.0)) < 1e-9
