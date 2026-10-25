import numpy as np
import pandas as pd
import pytest

from r2ai.s31.select import select_rrf
from r2ai.s36 import build as s36_build
from r2ai.s36 import fuse


def _rows():
    return [{"doc_id": 10 + i, "cosine_rank": i + 1, "judge_rank": r} for i, r in enumerate([2, 1, 3, 4])]


def test_rank_missing_last_ties_and_missing():
    rows = _rows()
    lookup = {(1, 10): 0.5, (1, 11): 0.9, (1, 12): 0.9}  # doc 13 has no score
    missing = fuse.rank_missing_last(rows, 1, lookup, "jina_score", "jina_rank")
    assert missing == 1
    assert [r["jina_rank"] for r in rows] == [3, 1, 2, 4]  # tie 11/12 by cosine rank; the unscored doc is last (= len(rows))
    assert rows[3]["jina_score"] is None


def test_rank_missing_last_ties_all_missing_at_pool_size():
    rows = _rows()
    fuse.rank_missing_last(rows, 1, {(1, 10): 0.1}, "s", "k")
    assert [r["k"] for r in rows] == [1, 4, 4, 4]


def test_source_hook_counts_and_strict():
    ranked = {1: _rows(), 2: _rows()}
    jina = {(1, d): 0.1 * d for d in (10, 11, 12, 13)}
    cmax = {(1, 10): 0.3, (2, 10): 0.2, (2, 11): 0.9}
    info = fuse.source_hook(jina, cmax)(ranked)
    assert info["pairs"] == 8 and info["missing_jina"] == 4 and info["missing_cmax"] == 5
    assert info["queries_missing_jina"] == 1 and info["queries_missing_cmax"] == 2
    with pytest.raises(ValueError):
        fuse.source_hook(jina, cmax, strict=True)({1: _rows()})


def test_weighted_rrf_default_is_unweighted_and_weight_two_changes_order():
    rows = [{"doc_id": 1, "cosine_rank": 1, "judge_rank": 3}, {"doc_id": 2, "cosine_rank": 3, "judge_rank": 1},
            {"doc_id": 3, "cosine_rank": 2, "judge_rank": 2}]
    keys = ("cosine_rank", "judge_rank")
    a = select_rrf(rows, 3, keys)
    b = select_rrf(rows, 3, keys, 60, None)
    c = select_rrf(rows, 3, keys, 60, (1.0, 1.0))
    assert [r["rrf_score"] for r in a] == [r["rrf_score"] for r in b] == [r["rrf_score"] for r in c]
    d = select_rrf(rows, 3, keys, 60, (1.0, 2.0))
    assert [r["doc_id"] for r in d][0] == 2  # judge rank 1 wins when its weight doubles
    assert d[0]["rrf_score"] == pytest.approx(1 / 63 + 2 / 61)
    with pytest.raises(ValueError):
        select_rrf(rows, 3, keys, 60, (1.0,))


def test_variant_table():
    assert s36_build.VARIANTS["FUSE4w-thrC"]["weights"] == (1.0, 2.0, 1.0, 1.0)
    assert s36_build.VARIANTS["FUSE4-thrC"]["keys"] == fuse.FOUR_KEYS == ("cosine_rank", "judge_rank", "jina_rank", "cmax_rank")
    with pytest.raises(ValueError):
        s36_build.run_settings(["FUSE4-thrC", "Check-FUSE4-thrC-pre"])
