import numpy as np

from r2ai.s41.kfit import query_features, solve_c


def test_solve_c_hits_total_with_floor_and_cap():
    c, keep = solve_c({1: 10.0, 2: 50.0, 3: 0.1}, {1: 300, 2: 60, 3: 300}, 80)
    assert sum(keep.values()) == 80 and keep[3] == 1 and keep[2] <= 60


def test_query_features_counts_and_zh_share():
    rows = [{"doc_id": d, "judge_score": s, "cosine_score": 0.9 - d / 100, "cmax_score": c}
            for d, s, c in ((1, 0.9, 0.8), (2, 0.6, float("nan")), (3, 0.05, 0.1), (4, 0.2, 0.6))]
    p = np.array([0.8, 0.5, 0.05, 0.15])
    f = query_features(rows, p, 2, {1: "zh", 2: "vi", 3: "zh", 4: "vi"}, "ba bốn năm")
    assert f["ef2_keep"] == 2 and f["query_words"] == 3 and f["zh_pool"] == 0.5
    assert f["n_pcal_ge0.5"] == 2 and f["n_p4b_ge0.5"] == 2 and f["n_cmax_ge0.5"] == 2
    assert f["pcal_at10"] == 0.05 and abs(f["pcal_sum"] - 1.5) < 1e-12
