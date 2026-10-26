import numpy as np
import pandas as pd
import pytest

from r2ai.s41.carriers import ef2_counts, ordered, select, solve_s

DOCSEL = pd.DataFrame({
    "query_id": [1, 1, 1, 1, 2, 2],
    "doc_id": [10, 11, 12, 13, 20, 21],
    "rank": [1, 2, 3, 4, 1, 2],
    "p4b": [0.9, 0.2, 0.8, 0.1, 0.5, 0.6],
    "cmax": [0.1, 0.9, np.nan, 0.5, 0.3, 0.2],
    "p_cal": [0.8, 0.5, 0.3, 0.1, 0.6, 0.2],
})


def test_orders():
    assert ordered(DOCSEL, "rank").query("query_id == 1").doc_id.tolist() == [10, 11, 12, 13]
    assert ordered(DOCSEL, "p4b").query("query_id == 1").doc_id.tolist() == [10, 12, 11, 13]
    assert ordered(DOCSEL, "cmax").query("query_id == 1").doc_id.tolist() == [11, 13, 10, 12]   # missing C-max last


def test_fair_by_rank_hits_target():
    car, info = select(DOCSEL, "fair", "rank", 3)
    assert len(car) == 3 and info["order"] == "rank"
    assert set(zip(car.query_id, car.doc_id)) == {(1, 10), (1, 11), (2, 20)}


def test_ef2_counts_follow_the_marginal_rule():
    p = {1: np.array([0.9, 0.5, 0.05])}
    # sum p / (4 s g + n): n=1 0.9/5, n=2 1.4/6, n=3 1.45/7 for 4 s g = 4
    assert ef2_counts(p, {1: 1.0}, 1.0) == {1: 2}
    assert ef2_counts(p, {1: 1.0}, 1e-6) == {1: 1}       # tiny gold count: one carrier
    assert ef2_counts(p, {1: 1.0}, 1e4) == {1: 3}        # huge gold count: every doc


def test_solve_s_total_and_bounds():
    p = {1: np.array([0.9, 0.5, 0.3, 0.1]), 2: np.array([0.6, 0.2])}
    g = {1: 2.0, 2: 1.0}
    s, n = solve_s(p, g, 4)
    assert sum(n.values()) == 4
    with pytest.raises(ValueError):
        solve_s(p, g, 7)


def test_ef2_select_uses_order():
    car, info = select(DOCSEL, "ef2", "rank", 6, {1: 1.0, 2: 1.0})
    assert len(car) == 6 and info["s"] > 0
