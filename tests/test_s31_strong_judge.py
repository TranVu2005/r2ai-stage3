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
