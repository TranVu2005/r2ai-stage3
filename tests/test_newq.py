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


def test_choose_carriers_rank_order_is_ch_rrf():
    sel = pd.DataFrame({"query_id": [1, 1, 1, 1, 2, 2], "doc_id": [10, 11, 12, 13, 20, 21], "rank": [1, 2, 3, 4, 1, 2],
                        "p4b": [0.1, 0.9, 0.8, 0.7, 0.2, 0.6]})
    sel["chunk"] = sel.p4b >= 0.5
    car, info, target = newq.choose_carriers(sel, {"rule": "fair", "order": "rank", "target_per_query": 1.5}, 2)
    assert target == 3 and info["order"] == "rank"
    assert sorted(zip(car.query_id, car.doc_id)) == [(1, 10), (1, 11), (2, 20)]        # top ceil(rho |D_q|) in RRF rank, not P4B
    car_p, info_p, _ = newq.choose_carriers(sel, {"rule": "fair", "target_per_query": 1.5}, 2)
    assert info_p["order"] == "p4b" and sorted(zip(car_p.query_id, car_p.doc_id)) == [(1, 11), (1, 12), (2, 21)]


def test_default_config_is_frozen_private():
    cfg = newq.load_config(None)
    assert cfg["carriers"] == {"rule": "fair", "order": "rank", "target_per_query": 146.29}
    assert cfg["depth"] == 300 and abs(cfg["a"] - 0.5298075151496865) < 1e-15


def test_pin_key_repo_relative(tmp_path):
    assert newq.pin_key(newq.ROOT / "src" / "r2ai" / "newq.py") == "repo:src/r2ai/newq.py"
    assert newq.pin_path("repo:src/r2ai/newq.py") == newq.ROOT / "src" / "r2ai" / "newq.py"
    assert not newq.pin_key(tmp_path / "x").startswith("repo:")


def test_pool_frame_cosine_weight_reorders_and_default_is_unweighted():
    rows = [{"doc_id": d, "cosine_rank": c, "judge_rank": j} for d, c, j in ((10, 1, 3), (11, 3, 1), (12, 2, 2))]
    base = newq.PoolFrame(("cosine_rank", "judge_rank"), {1: [dict(r) for r in rows]}, {}, {}, allow_missing=True)
    same = newq.PoolFrame(("cosine_rank", "judge_rank"), {1: [dict(r) for r in rows]}, {}, {}, allow_missing=True, w_cos=1.0)
    assert [r["doc_id"] for r in base.order[1]] == [r["doc_id"] for r in same.order[1]] == [10, 11, 12]   # tie 10/11 by cosine rank
    heavy = newq.PoolFrame(("cosine_rank", "judge_rank"), {1: [dict(r) for r in rows]}, {}, {}, allow_missing=True, w_cos=100.0)
    assert [r["doc_id"] for r in heavy.order[1]] == [10, 12, 11]


def test_choose_carriers_cosine_weight():
    sel = pd.DataFrame({"query_id": [1, 1, 1, 1], "doc_id": [10, 11, 12, 13], "rank": [1, 2, 3, 4], "p4b": [0.9, 0.9, 0.9, 0.9],
                        "cosine_rank": [50, 40, 2, 1], "judge_rank": [1, 2, 30, 40]})
    rule = {"rule": "fair", "order": "rank", "target_per_query": 2}
    car, info, _ = newq.choose_carriers(sel, rule, 1, w_cos=None, rrf_keys=("cosine_rank", "judge_rank"))
    assert info["order"] == "rank" and sorted(car.doc_id) == [10, 11]
    car, info, _ = newq.choose_carriers(sel, rule, 1, w_cos=3.0, rrf_keys=("cosine_rank", "judge_rank"))
    assert info["order"] == "wrrf" and info["w_cos"] == 3.0 and sorted(car.doc_id) == [12, 13]
    with pytest.raises(ValueError):
        newq.choose_carriers(sel, {"rule": "fair", "target_per_query": 2}, 1, w_cos=3.0, rrf_keys=("cosine_rank", "judge_rank"))
