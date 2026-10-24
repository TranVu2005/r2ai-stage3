"""S36 chunk-carrier rules and the S37 data application (union merge, Jina assembly, missing pairs)."""
import json

import numpy as np
import pandas as pd
import pytest

from r2ai.s36 import carriers, fuse, s37


def docsel(sizes, t_c=0.5):
    rows = []
    for q, n in enumerate(sizes, 1):
        for k in range(n):
            rows.append((q, q * 1000 + k, k + 1, round(0.95 - 0.9 * k / max(n - 1, 1), 4)))
    frame = pd.DataFrame(rows, columns=["query_id", "doc_id", "rank", "p4b"])
    frame["chunk"] = frame.p4b >= t_c
    return frame


def test_fair_carriers_use_one_global_rho_and_match_the_target():
    frame = docsel([10, 40, 7, 100])
    target = 90
    chosen, info = carriers.carriers_fair(frame, target)
    assert len(chosen) == target and 0 < info["rho"] <= 1
    sizes = frame.groupby("query_id").size()
    per_query = chosen.groupby("query_id").size()
    assert (per_query == np.ceil(info["rho"] * sizes - 1e-12)).all()              # ceil(rho * |D_q|) in every query
    # the carriers of a query are its highest-p docs
    for q, g in chosen.groupby("query_id"):
        best = frame[frame.query_id == q].sort_values("p4b", ascending=False).head(len(g)).doc_id
        assert set(g.doc_id) == set(best)


def test_fair_rho_is_the_smallest_on_a_tie_and_rejects_impossible_targets():
    sizes = np.array([4, 4])
    assert carriers.fair_total(sizes, 0.5) == 4 and carriers.solve_rho(sizes, 4) <= 0.5
    with pytest.raises(ValueError):
        carriers.solve_rho(sizes, 0)
    with pytest.raises(ValueError):
        carriers.solve_rho(sizes, 9)


def test_swap_drops_the_lowest_p_carriers_and_raises_the_threshold():
    frame = docsel([10, 10, 10])
    kept, info = carriers.carriers_swap(frame, 3)
    before = frame[frame.chunk]
    assert len(kept) == len(before) - 3
    dropped = before.merge(kept, on=["query_id", "doc_id"], how="left", indicator=True).query("_merge == 'left_only'")
    assert dropped.p4b.max() <= kept.merge(before, on=["query_id", "doc_id"]).p4b.min()
    assert info["t_c_after"] >= info["t_c_before"] and info["drop"] == 3
    with pytest.raises(ValueError):
        carriers.carriers_swap(frame, len(before))


def test_select_carriers_threshold_is_the_docsel_flags_and_run_writes_the_pins(tmp_path):
    frame = docsel([6, 6])
    path = tmp_path / "docsel.parquet"
    frame.to_parquet(path)
    report = carriers.run(path, "fair", "t", tmp_path / "out", target=int(frame.chunk.sum()))
    assert report["carriers"] == int(frame.chunk.sum()) and (tmp_path / "out" / "carriers-t.parquet").exists()
    assert json.loads((tmp_path / "out" / "carriers-t.json").read_text(encoding="utf-8"))["carriers_sha256"] == report["carriers_sha256"]
    same, _ = carriers.select_carriers(frame, "threshold")
    assert len(same) == int(frame.chunk.sum())


def union_of():
    return {1: [{"pair_id": 1, "doc_id": 10, "cosine_score": 0.9, "judge_score": 0.5, "origin": "pool13", "order": (0, 1), "cosine_rank": 1},
                {"pair_id": 2, "doc_id": 11, "cosine_score": 0.5, "judge_score": 0.1, "origin": "pool13", "order": (0, 2), "cosine_rank": 2}],
            2: [{"pair_id": 3, "doc_id": 20, "cosine_score": 0.7, "judge_score": 0.2, "origin": "pool13", "order": (0, 3), "cosine_rank": 1}]}


def test_merge_union_inserts_new_pairs_in_cosine_order_and_drops_known_docs():
    pairs = pd.DataFrame({"pair_id": [1_000_001, 1_000_002, 1_000_003], "query_id": [1, 1, 2], "doc_id": [900, 11, 901],
                          "cosine_score": [0.95, 0.4, 0.3]})
    out = s37.merge_union(union_of(), pairs, {1_000_001: 0.8, 1_000_002: 0.7, 1_000_003: 0.05})
    assert [r["doc_id"] for r in out[1]] == [900, 10, 11] and [r["cosine_rank"] for r in out[1]] == [1, 2, 3]   # doc 11 already known
    assert out[1][2]["judge_score"] == 0.1 and out[1][0]["judge_score"] == 0.8
    assert [r["doc_id"] for r in out[2]] == [20, 901]
    assert union_of()[1][0]["cosine_rank"] == 1                                                              # input untouched


def test_merge_union_refuses_unscored_pairs_and_unknown_queries():
    pairs = pd.DataFrame({"pair_id": [1_000_001], "query_id": [1], "doc_id": [900], "cosine_score": [0.9]})
    with pytest.raises(ValueError, match="no 4B score"):
        s37.merge_union(union_of(), pairs, {})
    pairs = pd.DataFrame({"pair_id": [1_000_001], "query_id": [9], "doc_id": [900], "cosine_score": [0.9]})
    with pytest.raises(ValueError, match="outside the union"):
        s37.merge_union(union_of(), pairs, {1_000_001: 0.5})


def test_assemble_jina_keeps_unrescored_queries_and_replaces_rescored_ones():
    base = {(1, 10): 0.1, (1, 11): 0.2, (2, 20): 0.3, (2, 21): 0.4}
    new = {(1, 10): 0.9, (1, 900): 0.8}
    out = s37.assemble_jina(base, new)
    assert out == {(1, 10): 0.9, (1, 900): 0.8, (2, 20): 0.3, (2, 21): 0.4}      # (1, 11) is gone: it must be rescored or listed


def test_missing_pairs_lists_pool_pairs_without_a_score():
    pools = {1: [{"doc_id": 10, "cosine_rank": 1, "judge_score": 0.5}, {"doc_id": 11, "cosine_rank": 2, "judge_score": 0.1}]}
    gap = s37.missing_pairs(pools, {(1, 10): 0.3})
    assert gap.to_dict("records") == [{"query_id": 1, "doc_id": 11, "cosine_rank": 2, "p4b": 0.1}]


def test_jina_missing_lists_the_s36_pairs_of_rescored_queries_only(tmp_path, monkeypatch):
    pd.DataFrame({"query_id": [1, 1, 2], "doc_id": [10, 11, 20]}).to_parquet(tmp_path / "pool.parquet")
    pd.DataFrame({"query_id": [1, 1, 3], "doc_id": [11, 12, 30], "score": [0.2, 0.3, 0.4]}).to_parquet(tmp_path / "supp.parquet")
    monkeypatch.setattr(fuse, "run_root", lambda: tmp_path / "run")
    report = s37.jina_missing(tmp_path / "pool.parquet", tmp_path / "supp.parquet")
    missing = pd.read_parquet(tmp_path / "run" / "cal" / "jina-missing-pairs.parquet")
    assert missing[["query_id", "doc_id"]].values.tolist() == [[1, 12]]              # (1, 11) is in the S37 pool; query 3 is not rescored
    assert report["pairs"] == 1 and report["s36_supplement_pairs_in_unrescored_queries"] == 1 and report["rescored_queries"] == 2


def test_apply_stops_while_an_artifact_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not there yet"):
        s37.apply({"pairs_new": str(tmp_path / "none.parquet"), "scores_4b": [], "cmax": [], "jina": str(tmp_path / "none2.parquet"), "out": str(tmp_path)})


def test_merge_union_drops_unscored_pairs_above_the_coverage_floor():
    pairs = pd.DataFrame({"pair_id": [1, 2, 3, 4], "query_id": [1, 1, 1, 2], "doc_id": [900, 901, 902, 903], "cosine_score": [0.95, 0.9, 0.8, 0.3]})
    scores = {1: 0.8, 2: 0.7, 3: 0.6}                                # 75% covered
    with pytest.raises(ValueError, match="0.7500 covered"):
        s37.merge_union(union_of(), pairs, scores, 0.98)
    out = s37.merge_union(union_of(), pairs, scores, 0.7)
    assert [r["doc_id"] for r in out[2]] == [20]                     # pair 4 (query 2) was not scored: dropped
    assert {900, 901, 902} <= {r["doc_id"] for r in out[1]}


def test_fair_with_a_given_rho_is_the_default_rule_and_needs_no_target():
    frame = docsel([10, 40, 7, 100])
    by_rho, info = carriers.carriers_fair(frame, rho=0.5)
    sizes = frame.groupby("query_id").size()
    assert info["rho"] == 0.5 and (by_rho.groupby("query_id").size() == np.ceil(0.5 * sizes - 1e-12)).all()
    default, info = carriers.select_carriers(frame, "fair")
    assert info["rho"] == carriers.DEFAULT_RHO
    solved, info = carriers.select_carriers(frame, "fair", target=60)
    assert len(solved) == 60 and info["rho"] != carriers.DEFAULT_RHO


def test_read_pairs_counts_a_pair_once_and_prefers_the_copy_with_a_4b_score(tmp_path):
    first = pd.DataFrame({"pair_id": [1, 2], "query_id": [1, 1], "doc_id": [10, 11], "cosine_score": [0.9, 0.8]})
    second = pd.DataFrame({"pair_id": [2, 7, 8], "query_id": [1, 1, 2], "doc_id": [11, 10, 20], "cosine_score": [0.8, 0.9, 0.7]})   # (1, 10) again as pair 7
    for name, frame in (("a", first), ("b", second)):
        frame.to_parquet(tmp_path / f"{name}.parquet")
    files = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    plain = s37.read_pairs(files)
    assert plain.pair_id.tolist() == [1, 2, 8]                       # first registry wins without scores
    scored = s37.read_pairs(files, {7: 0.5, 2: 0.1, 8: 0.2})
    assert scored.pair_id.tolist() == [2, 7, 8]                      # pair 7 has the score, pair 1 (same query, doc) does not
    clash = pd.DataFrame({"pair_id": [1], "query_id": [3], "doc_id": [30], "cosine_score": [0.1]})
    clash.to_parquet(tmp_path / "c.parquet")
    with pytest.raises(ValueError, match="two different pairs"):
        s37.read_pairs([tmp_path / "a.parquet", tmp_path / "c.parquet"])


def test_fixed_pools_cut_every_query_at_the_depth_and_refuse_a_short_union():
    rows = lambda n: [{"doc_id": 100 + k, "cosine_rank": k + 1, "cosine_score": 1 - k / 100, "judge_score": 0.9 - 0.01 * k, "origin": "x", "order": (0, k)}  # noqa: E731
                      for k in range(n)]
    pools, report = s37.fixed_pools({1: rows(12), 2: rows(20)}, 10)
    assert [len(v) for v in pools.values()] == [10, 10] and report["depth"] == 10 and pools[1][-1]["doc_id"] == 109
    assert pools[1][0]["judge_rank"] == 1 and "judge_rank" in pools[2][-1]
    with pytest.raises(ValueError, match="only 12 scored pairs"):
        s37.fixed_pools({1: rows(12)}, 20)


def test_pool13_plus_keeps_pool13_docs_and_the_new_docs_with_renumbered_cosine_ranks():
    row = lambda d, rank, origin: {"doc_id": d, "cosine_rank": rank, "cosine_score": 1 - rank / 100, "judge_score": 0.5, "origin": origin, "order": (0, rank)}  # noqa: E731
    union = {1: [row(10, 1, "s31"), row(11, 2, "s37"), row(12, 3, "s31"), row(13, 4, "s31"), row(14, 5, "s37")]}
    pools, report = s37.pool13_plus({1: [{"doc_id": 10}, {"doc_id": 12}]}, union)
    assert [r["doc_id"] for r in pools[1]] == [10, 11, 12, 14] and [r["cosine_rank"] for r in pools[1]] == [1, 2, 3, 5]
    assert report["docs_added"] == 2 and "judge_rank" in pools[1][0]
    with pytest.raises(ValueError, match="not in the scored union"):
        s37.pool13_plus({1: [{"doc_id": 99}]}, union)


def test_cosine13_pools_keep_the_old_pool_size_and_push_out_the_lowest_cosine_docs():
    row = lambda d, rank, origin: {"doc_id": d, "cosine_rank": rank, "cosine_score": 1 - rank / 100, "judge_score": 0.5, "origin": origin, "order": (0, rank)}  # noqa: E731
    union = {1: [row(10, 1, "s31"), row(11, 2, "s37"), row(12, 3, "s31"), row(13, 4, "s31")]}
    pools, report = s37.cosine13_pools({1: [{"doc_id": 10}, {"doc_id": 12}, {"doc_id": 13}]}, union)
    assert [r["doc_id"] for r in pools[1]] == [10, 11, 12] and report["old_pool13_docs_pushed_out"] == 1 and "judge_rank" in pools[1][0]
