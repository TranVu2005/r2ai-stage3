import pytest
import pandas as pd

from r2ai.s29.qwen_judge import (
    INSTRUCTIONS,
    _validate_pilot_report,
    candidate_pool_size,
    select_qfilter,
    select_qrrf,
    select_weighted_qrrf,
    truncate_document,
    _validate_pool_extension_alignment,
    _select_pilot_pairs,
)
from r2ai.s29.build import (
    _chunk_target,
    _resolve_variant_options,
    _validate_candidate_pool_sha,
    _validate_score_alignment,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


def test_pool_rounding_is_independent_of_k90_target():
    assert candidate_pool_size(103) == 134
    assert candidate_pool_size(10) == 13
    assert candidate_pool_size(103, multiplier=1.6) == 165


def test_weighted_qrrf_uses_normalized_cosine_and_qwen_weights():
    candidates = [
        {"doc_id": 1, "cosine_rank": 1, "qwen_rank": 3},
        {"doc_id": 2, "cosine_rank": 2, "qwen_rank": 1},
    ]

    selected = select_weighted_qrrf(candidates, keep=1, rrf_k=1, qwen_weight=0.75)

    assert selected[0]["doc_id"] == 2
    assert selected[0]["rrf_score"] == pytest.approx(0.25 / 3 + 0.75 / 2)


def test_chunk_target_scales_p1_n_c_with_python_rounding_and_caps_at_selected_docs():
    assert _chunk_target(n_c=5, selected_docs=10, multiplier=1.4) == 7
    assert _chunk_target(n_c=5, selected_docs=6, multiplier=1.4) == 6


def test_qrrf_variant_defaults_to_nc140_and_whole_body():
    assert _resolve_variant_options("QRRF-nc140-ph", None, None) == (1.4, True)
    assert _resolve_variant_options("QRRF-pool16", None, None) == (1.4, True)


def test_legacy_variant_defaults_preserve_original_build_options():
    assert _resolve_variant_options("Q-filter", None, None) == (1.0, False)
    assert _resolve_variant_options("Q-rrf", None, None) == (1.0, False)


def test_explicit_build_options_override_variant_defaults():
    assert _resolve_variant_options("QRRF-nc140-ph", 1.0, False) == (1.0, False)


def test_candidate_pool_sha_is_required_and_must_match():
    _validate_candidate_pool_sha({"candidate_pool_sha256": "abc"}, "abc")
    with pytest.raises(ValueError, match="missing candidate_pool_sha256"):
        _validate_candidate_pool_sha({}, "abc")
    with pytest.raises(ValueError, match="different candidate pool SHA256"):
        _validate_candidate_pool_sha({"candidate_pool_sha256": "old"}, "abc")


def test_expanded_pool_preserves_shared_scores_and_allows_cutoff_ties():
    source = pd.DataFrame(
        [
            {"query_id": 1, "doc_id": 10, "cosine_rank": 1, "cosine_score": 0.8},
            {"query_id": 1, "doc_id": 11, "cosine_rank": 2, "cosine_score": 0.7},
        ]
    )
    target = pd.DataFrame(
        [
            {"query_id": 1, "doc_id": 10, "cosine_rank": 1, "cosine_score": 0.8},
            {"query_id": 1, "doc_id": 12, "cosine_rank": 2, "cosine_score": 0.7},
            {"query_id": 1, "doc_id": 13, "cosine_rank": 3, "cosine_score": 0.7},
        ]
    )

    aligned = _validate_pool_extension_alignment(source, target)
    assert int(aligned["_merge"].eq("both").sum()) == 1
    assert int(aligned["_merge"].eq("left_only").sum()) == 1
    with pytest.raises(ValueError, match="changed a source cosine score"):
        _validate_pool_extension_alignment(
            source,
            target.assign(cosine_score=[0.81, 0.7, 0.7]),
        )


def test_expanded_pool_rejects_missing_source_docs_below_its_cosine_cutoff():
    source = pd.DataFrame(
        [{"query_id": 1, "doc_id": 11, "cosine_rank": 2, "cosine_score": 0.6}]
    )
    target = pd.DataFrame(
        [
            {"query_id": 1, "doc_id": 10, "cosine_rank": 1, "cosine_score": 0.8},
            {"query_id": 1, "doc_id": 12, "cosine_rank": 2, "cosine_score": 0.7},
        ]
    )

    with pytest.raises(ValueError, match="missing source pairs are not tied at the expanded cutoff"):
        _validate_pool_extension_alignment(source, target)


def test_pilot_samples_only_pairs_missing_from_a_seeded_checkpoint():
    pairs = [
        {"query_id": 1, "doc_id": 10},
        {"query_id": 1, "doc_id": 11},
        {"query_id": 2, "doc_id": 20},
    ]

    selected = _select_pilot_pairs(pairs, done={(1, 10)}, limit=2)

    assert {(row["query_id"], row["doc_id"]) for row in selected} == {
        (1, 11),
        (2, 20),
    }


def test_qfilter_keeps_highest_qwen_scores_and_breaks_ties_by_cosine_rank():
    candidates = [
        {"doc_id": 30, "cosine_rank": 3, "qwen_score": 0.9},
        {"doc_id": 20, "cosine_rank": 2, "qwen_score": 0.9},
        {"doc_id": 10, "cosine_rank": 1, "qwen_score": 0.1},
    ]

    selected = select_qfilter(candidates, keep=2)

    assert [row["doc_id"] for row in selected] == [20, 30]


def test_qrrf_combines_one_based_cosine_and_qwen_ranks_with_k60():
    candidates = [
        {"doc_id": 1, "cosine_rank": 1, "qwen_rank": 3},
        {"doc_id": 2, "cosine_rank": 2, "qwen_rank": 1},
        {"doc_id": 3, "cosine_rank": 3, "qwen_rank": 2},
    ]

    selected = select_qrrf(candidates, keep=2, rrf_k=60)

    assert [row["doc_id"] for row in selected] == [2, 1]


def test_passage_is_truncated_by_token_count_and_instruction_tracks_language():
    passage, token_count = truncate_document(TinyTokenizer(), "x" * 600)

    assert passage == "x" * 512
    assert token_count == 512
    assert INSTRUCTIONS["vi"] == (
        "Given a Vietnamese medical question, judge whether the Vietnamese passage answers it"
    )
    assert INSTRUCTIONS["zh"] == (
        "Given a Vietnamese medical question, judge whether the Chinese passage answers it"
    )


def test_full_run_requires_a_completed_200_pair_pilot_for_same_pool_size():
    pilot = {
        "status": "pilot_complete",
        "pairs_scored_this_run": 200,
        "pairs_checkpointed_total": 200,
        "pairs_total": 1000,
        "candidate_pool_sha256": "pool-a",
        "model_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    }

    _validate_pilot_report(pilot, expected_pairs=1000, expected_candidate_sha256="pool-a")
    with pytest.raises(ValueError, match="Pilot"):
        _validate_pilot_report(
            {**pilot, "pairs_scored_this_run": 199},
            expected_pairs=1000,
            expected_candidate_sha256="pool-a",
        )
    with pytest.raises(ValueError, match="candidate pool"):
        _validate_pilot_report(pilot, expected_pairs=1001, expected_candidate_sha256="pool-a")
    with pytest.raises(ValueError, match="SHA256"):
        _validate_pilot_report(pilot, expected_pairs=1000, expected_candidate_sha256="pool-b")


def test_full_score_pairs_must_match_current_candidate_pool():
    candidates = [
        {"query_id": 1, "doc_id": 10, "cosine_rank": 1, "cosine_score": 0.8},
        {"query_id": 1, "doc_id": 11, "cosine_rank": 2, "cosine_score": 0.7},
    ]
    scores = [
        (1, 10, 1, 0.8),
        (1, 11, 2, 0.7),
    ]

    _validate_score_alignment(candidates, scores)
    with pytest.raises(ValueError, match="query/document pairs"):
        _validate_score_alignment(candidates, [(1, 10, 1, 0.8), (2, 11, 2, 0.7)])
