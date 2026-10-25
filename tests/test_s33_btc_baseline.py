import numpy as np
import pandas as pd

from r2ai.s33 import bge_score
from r2ai.s33.build import reduce_chunks


class FakeScorer:
    """Logit = number of passage tokens; tokens = whitespace words, capped like max_length."""

    def __init__(self):
        self.batches = []

    def encode(self, queries, passages):
        return [[0] + list(range(len(q.split()))) + [1] + list(range(len(p.split())))[: bge_score.MAX_LENGTH] for q, p in zip(queries, passages)]

    def logits(self, sequences):
        self.batches.append(len(sequences))
        return np.array([float(len(s)) for s in sequences], dtype=np.float32)

    def is_oom(self, exc):
        return False

    def empty_cache(self):
        pass


def _frame(texts):
    n = len(texts)
    return pd.DataFrame({"pair_id": np.arange(n) + 100, "query_id": np.ones(n, dtype=int), "doc_id": np.arange(n) + 7,
                         "query_text": ["q a"] * n, "doc_text": texts})


def test_score_frame_keeps_row_order_across_length_batches():
    texts = ["x " * k for k in (5, 1, 9, 3, 7)]
    scorer = FakeScorer()
    result, stats = bge_score.score_frame(scorer, _frame(texts), max_batch_tokens=24, max_batch_size=2, log=lambda *a, **k: None)
    expected = np.array([len(t.split()) + 4 for t in texts], dtype=np.float32)  # [0] + 2 query + [1] + passage
    assert result.pair_id.tolist() == [100, 101, 102, 103, 104]
    np.testing.assert_array_equal(result.score.to_numpy(), expected)
    np.testing.assert_allclose(result.prob.to_numpy(), 1 / (1 + np.exp(-expected.astype(np.float64))))
    assert result.input_tokens.tolist() == expected.astype(int).tolist()
    assert stats["pairs"] == 5 and max(scorer.batches) <= 2


def test_reduce_chunks_takes_max_and_lowest_rank_on_ties():
    pairs = pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "src_pair_id": [10, 10, 10, 11, 11],
                          "chunk_rank": [1, 2, 3, 1, 2], "field": ["title", "body", "body", "body", "answer"]})
    scores = pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "score": [0.5, 2.0, -1.0, 3.0, 3.0]})
    best = reduce_chunks(scores, pairs).set_index("pair_id")
    assert best.loc[10, "r_chk"] == 2.0 and best.loc[10, "best_chunk_rank"] == 2 and best.loc[10, "best_field"] == "body"
    assert best.loc[11, "r_chk"] == 3.0 and best.loc[11, "best_chunk_rank"] == 1


def test_reduce_chunks_rejects_missing_scores():
    pairs = pd.DataFrame({"pair_id": [1, 2], "src_pair_id": [10, 10], "chunk_rank": [1, 2], "field": ["body", "body"]})
    try:
        reduce_chunks(pd.DataFrame({"pair_id": [1], "score": [0.1]}), pairs)
    except ValueError as exc:
        assert "no bge score" in str(exc)
    else:
        raise AssertionError("missing chunk score must fail")
