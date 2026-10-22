import argparse

import numpy as np
import pandas as pd
import pytest

from r2ai.s31 import judge_score as js
from r2ai.s31.pool import _shard_groups, merge_with_s29, pool_size, truncate_heads


class CharTokenizer:
    """One token per character; offsets are character spans."""

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        return {
            "input_ids": [[ord(c) for c in t] for t in texts],
            "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts],
        }

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids)


def test_pool_size_uses_python_round():
    assert pool_size(90) == 180
    assert pool_size(103, 1.3) == 134


def test_merge_keeps_s29_head_and_appends_new_tail():
    s29 = pd.DataFrame({"doc_id": [5, 6], "cosine_rank": [1, 2], "cosine_score": [0.9, 0.8]})
    merged, swapped = merge_with_s29(1, s29, [5, 6, 7, 8], [0.9, 0.8, 0.7, 0.6], 3)
    assert merged == [(5, 0.9), (6, 0.8), (7, 0.7)]
    assert swapped == 0


def test_merge_accepts_only_boundary_ties():
    s29 = pd.DataFrame({"doc_id": [5, 6], "cosine_rank": [1, 2], "cosine_score": [0.9, 0.5]})
    # doc 9 ties doc 6 exactly at the cut: S29 kept 6, the new top-k happened to pick 9.
    merged, swapped = merge_with_s29(1, s29, [5, 9, 6, 7], [0.9, 0.5, 0.5, 0.4], 4)
    assert merged == [(5, 0.9), (6, 0.5), (9, 0.5), (7, 0.4)] and swapped == 1
    with pytest.raises(ValueError):  # a non-tied different doc
        merge_with_s29(1, s29, [5, 9, 6, 7], [0.9, 0.6, 0.5, 0.4], 4)
    with pytest.raises(ValueError):  # same docs, different score
        merge_with_s29(1, s29, [5, 6, 7], [0.9, 0.51, 0.4], 3)


def test_truncate_heads_accepts_whole_or_safe_window_only():
    tok = CharTokenizer()
    texts = ["short", "x" * 50, "y" * 12]
    out = truncate_heads(tok, texts, limit=9, windows=((10, 2),))
    assert out[0] == ("short", 5)
    assert out[1] is None  # 9th token ends at 9 > 10 - 2
    assert out[2] is None
    out = truncate_heads(tok, texts, limit=9, windows=((10, 2), (40, 4)))
    assert out[1] == ("x" * 9, 9) and out[2] == ("y" * 9, 9)


def test_shard_groups_keep_whole_queries():
    groups = _shard_groups({1: 5, 2: 5, 3: 5, 4: 1}, target=10)
    assert groups == [[1, 2], [3, 4]]


def test_prompt_matches_s29_layout():
    tok = CharTokenizer()
    ids = js.build_prompt_ids(tok, "q", "d", "vi", [1], [2])
    text = "".join(chr(i) for i in ids[1:-1])
    assert text == f"<Instruct>: {js.INSTRUCTIONS['vi']}\n<Query>: q\n<Document>: d"
    assert ids[0] == 1 and ids[-1] == 2


def test_token_budget_batches_respect_budget_and_cover_all():
    lengths = [10, 300, 20, 290, 5, 5, 5]
    batches = js.token_budget_batches(lengths, max_batch_tokens=600, max_batch_size=4)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 4 and max(lengths[i] for i in batch) * len(batch) <= 600


def test_yes_probability_equals_two_way_softmax():
    no, yes = np.array([1.0, -3.0]), np.array([2.0, 4.0])
    ref = np.exp(yes) / (np.exp(yes) + np.exp(no))
    assert np.allclose(js.yes_probability(no, yes), ref)


class FakeScorer:
    dtype_name = "float32"
    resolved_revision = "fake"
    prefix_ids, suffix_ids = [], []

    def __init__(self):
        self.tokenizer = CharTokenizer()
        self.calls = 0

    def logits(self, sequences):
        self.calls += 1
        lens = np.array([len(s) for s in sequences], dtype=np.float32)
        return -lens / 100.0, lens / 100.0

    def is_oom(self, exc):
        return False

    def empty_cache(self):
        pass


def _write_pool(tmp_path, n_pairs=7):
    shards = tmp_path / "pool" / "shards"
    shards.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=np.int64),
            "query_id": np.repeat([1, 2], [4, n_pairs - 4]),
            "doc_id": np.arange(100, 100 + n_pairs),
            "cosine_rank": [1, 2, 3, 4] + list(range(1, n_pairs - 3)),
            "cosine_score": np.linspace(0.9, 0.1, n_pairs),
            "in_pool13": [i % 3 != 2 for i in range(n_pairs)],
            "lang": ["vi", "zh"] * (n_pairs // 2) + ["vi"] * (n_pairs % 2),
            "query_text": ["q"] * n_pairs,
            "doc_text": ["d" * (i + 1) for i in range(n_pairs)],
            "doc_tokens": [i + 1 for i in range(n_pairs)],
        }
    )
    frame.iloc[:4].to_parquet(shards / "pool-00000.parquet", index=False)
    frame.iloc[4:].to_parquet(shards / "pool-00001.parquet", index=False)
    return tmp_path / "pool"


def test_score_resume_and_collect(tmp_path, monkeypatch):
    pool = _write_pool(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(js, "make_scorer", lambda args, log: FakeScorer())
    base = ["--pool", str(pool), "--out", str(out), "--model", "qwen3-reranker-4b", "--part-pairs", "2"]
    parser = js.build_parser()
    report = js.run_score(parser.parse_args(["score", *base, "--worker", "0", "--workers", "2"]))
    assert report["status"] == "worker_complete"
    with pytest.raises(FileNotFoundError):
        js.run_collect(parser.parse_args(["collect", *base]))
    js.run_score(parser.parse_args(["score", *base, "--worker", "1", "--workers", "2"]))
    again = js.run_score(parser.parse_args(["score", *base, "--worker", "1", "--workers", "2"]))
    assert again["status"] == "nothing_to_do"
    collected = js.run_collect(parser.parse_args(["collect", *base]))
    scores = pd.read_parquet(out / "qwen3-reranker-4b" / "scores-all.parquet")
    assert collected["pairs"] == 7 and scores.pair_id.tolist() == list(range(7))
    frame = pd.concat([pd.read_parquet(p) for p in sorted((pool / "shards").glob("*.parquet"))])
    lengths = np.array([len(js.build_prompt_ids(CharTokenizer(), q, d, l, [], [])) for q, d, l in zip(frame.query_text, frame.doc_text, frame.lang)])
    assert np.allclose(scores.score, js.yes_probability(-lengths / 100.0, lengths / 100.0), atol=1e-6)


def test_resume_from_copies_earlier_parts(tmp_path, monkeypatch):
    pool = _write_pool(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(js, "make_scorer", lambda args, log: FakeScorer())
    parser = js.build_parser()
    common = ["--pool", str(pool), "--model", "qwen3-reranker-4b", "--part-pairs", "2"]
    js.run_score(parser.parse_args(["score", *common, "--out", str(first)]))
    report = js.run_score(parser.parse_args(["score", *common, "--out", str(second), "--resume-from", str(first)]))
    assert report["status"] == "nothing_to_do"


def test_select_rrf_matches_s29_qrrf_exactly():
    import random

    from r2ai.s29.qwen_judge import select_qrrf
    from r2ai.s31.select import add_judge_rank, select_rrf

    rng = random.Random(7)
    rows = [{"doc_id": 1000 + i, "cosine_rank": i + 1, "qwen_score": rng.choice([0.1, 0.5, rng.random()])} for i in range(300)]
    add_judge_rank(rows, "qwen_score", "qwen_rank")
    ours = select_rrf(rows, 170, ("cosine_rank", "qwen_rank"))
    ref = select_qrrf(rows, 170)
    assert [r["doc_id"] for r in ours] == [r["doc_id"] for r in ref]
    assert [r["rrf_score"] for r in ours] == [r["rrf_score"] for r in ref]


def test_select_rrf_three_sources_and_diff_stats():
    from r2ai.s31.select import doc_set_diff, judge_correlations, select_rrf

    rows = [
        {"doc_id": 1, "cosine_rank": 1, "r06": 3, "r4": 3},
        {"doc_id": 2, "cosine_rank": 2, "r06": 1, "r4": 1},
        {"doc_id": 3, "cosine_rank": 3, "r06": 2, "r4": 2},
    ]
    assert [r["doc_id"] for r in select_rrf(rows, 2, ("cosine_rank", "r06", "r4"))] == [2, 1]  # 1/61+2/63 > 1/63+2/62
    with pytest.raises(ValueError):
        select_rrf(rows, 2, ("r06", "cosine_rank"))
    diff = doc_set_diff({1: [1, 2], 2: [3]}, {1: [2, 4], 2: [3]})
    assert diff["mean_docs_only_in_first"] == 0.5 and diff["mean_symmetric_difference"] == 1.0
    assert diff["queries_identical"] == 1
    frame = pd.DataFrame({"query_id": [1, 1, 1, 2, 2, 2], "a": [1, 2, 3, 1, 2, 3], "b": [1, 2, 3, 3, 2, 1]})
    corr = judge_correlations(frame, ["a", "b"])["a~b"]
    assert corr["spearman_per_query_mean"] == 0.0


def test_subset_pool13_then_tail_collects_each_and_all(tmp_path, monkeypatch):
    pool = _write_pool(tmp_path, n_pairs=9)
    out = tmp_path / "out"
    monkeypatch.setattr(js, "make_scorer", lambda args, log: FakeScorer())
    parser = js.build_parser()
    base = ["--pool", str(pool), "--out", str(out), "--model", "qwen3-reranker-4b", "--part-pairs", "2"]
    js.run_score(parser.parse_args(["score", *base, "--subset", "pool13"]))
    status = js.run_status(parser.parse_args(["status", *base, "--subset", "pool13"]))
    assert status["complete"] and status["pairs_total"] == 6
    assert not js.run_status(parser.parse_args(["status", *base, "--subset", "tail"]))["complete"]
    pool13 = js.run_collect(parser.parse_args(["collect", *base, "--subset", "pool13"]))
    assert pool13["pairs"] == 6
    with pytest.raises(FileNotFoundError):
        js.run_collect(parser.parse_args(["collect", *base, "--subset", "all"]))
    js.run_score(parser.parse_args(["score", *base, "--subset", "tail"]))
    full = js.run_collect(parser.parse_args(["collect", *base, "--subset", "all"]))
    assert full["pairs"] == 9 and full["duplicate_pairs_dropped"] == 0
    scores = pd.read_parquet(out / "qwen3-reranker-4b" / "scores-all.parquet")
    assert scores.pair_id.tolist() == list(range(9))
    names = sorted(p.name for p in (out / "qwen3-reranker-4b" / "parts").glob("*.parquet"))
    assert any(".pool13.part-" in n for n in names) and any(".tail.part-" in n for n in names)


def test_build_variant_rank_keys():
    from r2ai.s31.build import variant_rank_keys

    assert variant_rank_keys("J4-rrf", "J4") == ("cosine_rank", "judge_rank")
    assert variant_rank_keys("J4-rrf3", "J4") == ("cosine_rank", "q06_rank", "judge_rank")
    assert variant_rank_keys("Jcheck-q06-rrf", None) == ("cosine_rank", "q06_rank")
    with pytest.raises(ValueError):
        variant_rank_keys("J4-rrf-pool20", "J4")


def test_build_load_candidates_joins_pool13_and_checks_alignment(tmp_path):
    from r2ai.s31.build import load_candidates, rank_candidates

    pool = tmp_path / "pool"
    pool.mkdir()
    index = pd.DataFrame({"pair_id": [0, 1, 2, 3], "query_id": [1, 1, 1, 2], "doc_id": [10, 11, 12, 20],
                          "cosine_rank": [1, 2, 3, 1], "cosine_score": [0.9, 0.8, 0.7, 0.5],
                          "in_pool13": [True, True, False, True]})
    index.to_parquet(pool / "pool_index.parquet", index=False)
    pd.DataFrame({"pair_id": [0, 1, 3], "query_id": [1, 1, 2], "doc_id": [10, 11, 20],
                  "qwen_score": [0.1, 0.9, 0.5]}).to_parquet(pool / "s29_qwen06_scores.parquet", index=False)
    judge = tmp_path / "scores-pool13.parquet"
    pd.DataFrame({"pair_id": [0, 1, 3], "query_id": [1, 1, 2], "doc_id": [10, 11, 20],
                  "score": [0.7, 0.2, 0.4]}).to_parquet(judge, index=False)
    frame = load_candidates(pool, judge)
    assert frame.pair_id.tolist() == [0, 1, 3]
    ranked = rank_candidates(frame)
    assert [(r["doc_id"], r["q06_rank"], r["judge_rank"]) for r in ranked[1]] == [(10, 2, 1), (11, 1, 2)]
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"pair_id": [0, 1, 3], "query_id": [1, 1, 2], "doc_id": [10, 99, 20],
                  "score": [0.7, 0.2, 0.4]}).to_parquet(bad, index=False)
    with pytest.raises(ValueError):
        load_candidates(pool, bad)
    short = tmp_path / "short.parquet"
    pd.DataFrame({"pair_id": [0], "query_id": [1], "doc_id": [10], "score": [0.7]}).to_parquet(short, index=False)
    with pytest.raises(ValueError):
        load_candidates(pool, short)


def test_pharmacity_merge_pool_keeps_size_and_evicts_lowest():
    from r2ai.s31.pharmacity_ext import merge_pool

    existing = pd.DataFrame({"pair_id": [1, 2, 3], "doc_id": [10, 11, 12], "cosine_score": [0.9, 0.6, 0.5],
                             "cosine_rank": [1, 2, 3]})
    rows, evicted = merge_pool(existing, np.array([20, 21]), np.array([0.7, 0.5]))
    assert rows.doc_id.tolist() == [10, 20, 11] and rows.cosine_rank.tolist() == [1, 2, 3]
    assert rows.is_new.tolist() == [False, True, False] and evicted == 1
    with pytest.raises(ValueError):
        merge_pool(existing, np.array([10]), np.array([0.8]))


def test_variable_k_variants_use_cosine_and_judge():
    from r2ai.s31.build import variant_k_mode, variant_rank_keys

    for mode in ("scale", "thr", "mix"):
        assert variant_k_mode(f"K4-{mode}", "J4") == mode
        assert variant_rank_keys(f"K4-{mode}", "J4") == ("cosine_rank", "judge_rank")
    assert variant_k_mode("J4-rrf", "J4") is None
    assert variant_k_mode("K4-scale", None) is None


def test_allocate_total_hits_target_within_bounds():
    from r2ai.s31.select import allocate_total

    raw = np.array([1.0, 2.0, 3.0, 50.0])
    lo, hi = np.array([2, 2, 2, 2]), np.array([6, 6, 6, 6])
    counts, c = allocate_total(raw, lo, hi, 16)
    assert counts.sum() == 16 and (counts >= lo).all() and (counts <= hi).all()
    assert counts[3] == 6 and counts[0] <= counts[1] <= counts[2]
    with pytest.raises(ValueError):
        allocate_total(raw, lo, hi, 30)


def test_threshold_counts_exact_total_with_ties():
    from r2ai.s31.select import threshold_counts

    qidx = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
    tie = np.array([1, 2, 3, 1, 2, 3])
    res = threshold_counts(qidx, scores, tie, np.array([1, 1]), np.array([3, 3]), 4)
    assert res["k"].sum() == 4 and res["t"] == 0.5
    assert res["pairs_at_t"] == 4 and res["pairs_at_t_above_cut"] == 2
    assert res["k"].tolist() == [3, 1]


def test_thin_chunks_add_carriers_only_when_enabled():
    from r2ai.s29.build import _carriers, _thin_carrier

    row = {"relevant_chunks": [{"doc_id": 1, "chunk_text": "a"}, {"doc_id": 2, "chunk_text": "b"}]}
    picked = [(1, 1), (7, 7), (3, 3), (4, 4)]
    own, groups = {1: "a"}, {3: [3], 4: [4]}
    assert _carriers(row, picked, own, {}, groups, {}, 1.5) == [(1, 1), (3, 3), (4, 4)]
    thin = {7: "thin doc text, verbatim"}
    assert _carriers(row, picked, own, {}, groups, {}, 1.5, thin) == [(1, 1), (7, 7), (3, 3)]
    assert _thin_carrier(7, 7, own, {}, groups, {}, thin)
    assert not _thin_carrier(3, 3, own, {}, groups, {}, thin)
    assert not _thin_carrier(7, 7, own, {}, groups, {}, None)


def test_tfree_and_thr_c_variant_names():
    from r2ai.s31.build import variant_chunk_mode, variant_k_mode, variant_rank_keys

    assert variant_k_mode("K4-tfree-t0.06", "J4") == "tfree:0.06"
    assert variant_k_mode("K4-tfree-t0.1", "J4") == "tfree:0.1"
    assert variant_k_mode("K4-thr-C", "J4") == "thr"
    assert variant_chunk_mode("K4-thr-C", "J4") == "threshold"
    assert variant_chunk_mode("K4-thr", "J4") is None
    assert variant_rank_keys("K4-tfree-t0.08", "J4") == ("cosine_rank", "judge_rank")


def test_threshold_chunk_selector_exact_total_in_rrf_order():
    from r2ai.s31.build import threshold_chunk_selector

    score_of = {1: {10: 0.9, 11: 0.2, 12: 0.8}, 2: {20: 0.5, 21: 0.95}}
    offered = {1: [(10, 10, True), (11, 11, True), (12, 12, True)], 2: [(20, 20, False), (21, 21, True)]}
    info = {}
    result = threshold_chunk_selector(score_of, 3, info)(offered)
    assert result == {1: [(10, 10), (12, 12)], 2: [(21, 21)]}
    assert info["t_c"] == 0.8 and info["chunks"] == 3 and info["carriers_without_text"] == 1


def test_subset_extra_uses_pairs_file(tmp_path):
    shard = tmp_path / "shards" / "pool-00000.parquet"
    shard.parent.mkdir()
    pd.DataFrame({"pair_id": np.arange(6), "in_pool13": [True, True, False, False, False, True]}).to_parquet(shard)
    pd.DataFrame({"pair_id": [3, 4]}).to_parquet(tmp_path / js.PAIRS_FILE)
    args = argparse.Namespace(subset="extra", pairs_file=None, pool=str(tmp_path))
    ids = js.subset_pair_ids(args)
    parts = js.plan_parts([shard], 1, "extra", ids)
    assert [p["positions"].tolist() for p in parts] == [[3], [4]]
    assert js.part_name(parts[0]) == "pool-00000.extra.part-0000.parquet"
    assert js.subset_pair_ids(argparse.Namespace(subset="tail", pairs_file=None, pool=str(tmp_path))) is None
    with pytest.raises(ValueError):
        js.subset_positions(shard, "extra")


def test_free_pool_and_keep_use_fixed_constants(monkeypatch):
    from r2ai.s31 import build as b

    monkeypatch.setattr(b, "P_FREE", 3)
    monkeypatch.setattr(b, "F_FREE", 1)
    monkeypatch.setattr(b, "T_FREE", 0.5)
    rows = [{"doc_id": d, "cosine_rank": r, "judge_score": s} for d, r, s in
            [(10, 1, 0.1), (11, 2, 0.9), (12, 3, 0.7), (13, 4, 0.99)]]
    pool = b.free_pool({1: rows, 2: rows[:2]})
    assert [r["doc_id"] for r in pool[1]] == [10, 11, 12]
    assert [r["judge_rank"] for r in pool[1]] == [3, 1, 2]
    keep, report = b.free_keep(pool)
    assert keep == {1: 2, 2: 1} and report["queries_pool_below_p"] == 1
    assert b.variant_k_mode("K4-free-p13", "J4") == "free-p13"
    assert b.variant_chunk_mode("K4-free-p13", "J4") == "threshold-free"


def test_pool_extra_needed_pairs_dedup_and_flags(monkeypatch):
    from r2ai.s31 import pool_extra as pe

    monkeypatch.setattr(pe, "P_FREE", 2)
    fresh = {1: [(10, 0.9), (11, 0.8), (12, 0.7), (13, 0.6)]}
    rows = pe.needed_pairs(fresh, {1: 4}, {(1, 10)}, {1: {12, 14}}, {1})
    got = {r[1]: (r[2], r[4], r[5]) for r in rows}
    assert got == {11: (2, True, True), 12: (3, False, True), 13: (4, False, True), 14: (0, False, True)}
    rows = pe.needed_pairs(fresh, {1: 2}, {(1, 10)}, {}, set())
    assert [(r[1], r[4], r[5]) for r in rows] == [(11, True, False)]


def test_free_adaptive_deepens_only_when_211th_doc_passes(monkeypatch):
    from r2ai.s31 import build as b

    monkeypatch.setattr(b, "P_FREE", 2)
    monkeypatch.setattr(b, "F_FREE", 1)
    monkeypatch.setattr(b, "T_FREE", 0.5)

    def rows(scores):
        return [{"doc_id": i, "cosine_rank": i + 1, "judge_score": s} for i, s in enumerate(scores)]

    union = {1: rows([0.9, 0.2, 0.8]), 2: rows([0.9, 0.6, 0.7, 0.1]), 3: rows([0.9, 0.6])}
    pools, keep, report = b.free_adaptive(union)
    assert [len(pools[q]) for q in (1, 2, 3)] == [2, 4, 2]
    assert keep == {1: 1, 2: 3, 3: 2}
    assert report["queries_deepened"] == 1
    assert report["queries_last_scored_doc_still_at_or_above_t_ids"] == [3]
    assert b.variant_k_mode("K4-thr-C-ext", "J4") == "thr-ext" and b.variant_k_mode("K4-free", "J4") == "free"
    assert b.variant_chunk_mode("K4-free", "J4") == "fixed"


def test_mix_rows_keeps_chunk_order_and_drops_unselected_docs():
    from r2ai.s31.mix import mix_rows

    docs = [{"id": 1, "relevant_docs": [3, 1, 2], "relevant_chunks": []}, {"id": 2, "relevant_docs": [5], "relevant_chunks": []}]
    chunks = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id": 2, "chunk_text": "b"}, {"doc_id": 1, "chunk_text": "a"}]},
              {"id": 2, "relevant_docs": [6], "relevant_chunks": [{"doc_id": 6, "chunk_text": "x"}]}]
    rows, stats = mix_rows(docs, chunks)
    assert rows[0] == {"id": 1, "relevant_docs": [3, 1, 2], "relevant_chunks": chunks[0]["relevant_chunks"]}
    assert rows[1]["relevant_chunks"] == []
    assert stats["chunks_dropped_doc_not_selected"] == 1 and stats["queries_chunks_differ_from_chunk_source"] == 1
