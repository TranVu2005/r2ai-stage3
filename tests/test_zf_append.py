import json

import pytest

from r2ai.zh_full_ingest.append_zf import render, select_zf
from r2ai.zh_sample.append import verify_gate

GROUPS = {10: [10, 11], 20: [20], 30: [30, 31], 40: [40]}


def test_select_orders_by_max_logit_and_expands_the_group():
    pairs = [(20, 1.0), (10, 0.5), (10, 3.0), (30, 2.0), (40, -1.0)]
    ids, docs, available = select_zf(pairs, GROUPS, [], 3)
    assert docs == [10, 30, 20] and ids == [10, 11, 30, 31, 20] and available == 4


def test_select_skips_docs_already_present_and_ids_already_present():
    ids, docs, available = select_zf([(10, 3.0), (30, 2.0), (20, 1.0)], GROUPS, [10, 31], 2)
    assert docs == [30, 20] and ids == [30, 20] and available == 2   # 10 present -> dropped; 31 not repeated


def test_select_rejects_unknown_or_nonfinite():
    with pytest.raises(ValueError):
        select_zf([(99, 1.0)], GROUPS, [], 1)
    with pytest.raises(ValueError):
        select_zf([(10, float('nan'))], GROUPS, [], 1)


def test_render_appends_after_vi_docs_and_gate_proves_nothing_else_changed():
    base = json.dumps([{'id': 1, 'relevant_docs': [1, 2], 'relevant_chunks': [{'x': 'aé'}]},
                       {'id': 2, 'relevant_docs': [], 'relevant_chunks': []}])
    cache = {1: [(10, 2.0), (20, 1.0)], 2: [(40, 0.0), (30, 5.0)]}
    out = []
    selections = render(base, cache, GROUPS, 1, out.append)
    result = ''.join(out)
    assert json.loads(result)[0]['relevant_docs'] == [1, 2, 10, 11]
    assert json.loads(result)[1]['relevant_docs'] == [30, 31]
    assert verify_gate(base, result, selections) == {'queries': 2, 'vi_docs_diff': 0, 'chunks_byte_diff': 0}
    assert render(base, cache, GROUPS, 0, [].append)[1]['appended'] == []


def test_select_tau_keeps_only_docs_at_or_above_the_threshold():
    pairs = [(10, 3.0), (20, 1.0), (30, 2.0), (40, 0.5)]
    ids, docs, available = select_zf(pairs, GROUPS, [], 10, tau=1.0)
    assert docs == [10, 30, 20] and available == 3
    assert select_zf(pairs, GROUPS, [], 2, tau=1.0)[1] == [10, 30]            # cap still applies
    assert select_zf(pairs, GROUPS, [], 10, tau=9.0) == ([], [], 0)


def test_select_allow_restricts_to_one_domain_in_reranker_order():
    pairs = [(10, 3.0), (20, 2.5), (30, 2.0), (40, 1.0)]
    ids, docs, _ = select_zf(pairs, GROUPS, [], 2, allow={30, 40, 20})
    assert docs == [20, 30] and ids == [20, 30, 31]


def test_calibrate_tau_hits_the_target_mean():
    from r2ai.zh_full_ingest.append_zf import calibrate_tau
    base = json.dumps([{'id': 1, 'relevant_docs': [], 'relevant_chunks': []}, {'id': 2, 'relevant_docs': [20], 'relevant_chunks': []}])
    cache = {1: [(10, 3.0), (20, 2.0), (30, 1.0), (40, 0.0)], 2: [(10, 3.0), (20, 9.0), (30, 1.0), (40, 0.0)]}
    tau, mean = calibrate_tau(base, cache, GROUPS, cap=10, target=2.5)
    assert (tau, mean) == (0.01, 2.5)   # q1 keeps 10,20,30 ; q2 (20 already in base) keeps 10,30


def test_delta_doc_r_estimate_carries_both_bounds():
    from r2ai.zh_full_ingest.append_zf import delta_doc_r_estimate
    e = delta_doc_r_estimate(100)
    assert (e['lower'], e['upper']) == (0.0017, 0.037) and 'estimate' in e['basis']
    assert delta_doc_r_estimate(10)['upper'] == 0.0037


def test_primary_only_appends_the_primary_id_and_default_keeps_the_group():
    pairs = [(10, 3.0), (30, 2.0), (20, 1.0)]
    assert select_zf(pairs, GROUPS, [], 2)[0] == [10, 11, 30, 31]                    # default unchanged
    ids, docs, available = select_zf(pairs, GROUPS, [], 2, primary_only=True)
    assert (ids, docs, available) == ([10, 30], [10, 30], 3)
    assert select_zf(pairs, GROUPS, [10], 2, primary_only=True)[0] == [30, 20]       # doc already present is still skipped


def test_primary_variants_parse_and_other_variants_are_unchanged():
    from r2ai.zh_full_ingest.append_zf import parse_variant
    assert parse_variant('ZF-5p', {}, None) == ({'k': 5, 'primary_only': True}, {'k': 5, 'primary_only': True})
    assert parse_variant('ZF-10', {}, None) == ({'k': 10}, {'k': 10})
    assert parse_variant('ZG', {}, 0.96) == ({'k': 10, 'tau': 0.96}, {'tau': 0.96, 'cap': 10})
    assert parse_variant('ZF-120ask-5', {1: ('b', '120ask.com')}, None)[0] == {'k': 5, 'allow': {1}}


def test_render_primary_only_writes_one_id_per_doc_and_passes_the_byte_gate():
    base = json.dumps([{'id': 1, 'relevant_docs': [1], 'relevant_chunks': []}])
    out = []
    selections = render(base, {1: [(10, 2.0), (30, 1.0)]}, GROUPS, 2, out.append, primary_only=True)
    result = ''.join(out)
    assert json.loads(result)[0]['relevant_docs'] == [1, 10, 30]
    assert verify_gate(base, result, selections)['queries'] == 1
