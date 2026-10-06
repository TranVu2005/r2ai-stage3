import importlib
import io
import json

import pytest

from r2ai.eval.scorer import doc_prf, load_submission, score, whitespace_tokenizer


BASE = '[\r\n {"id":7,"relevant_docs":[1, 2 \r\n ],"relevant_chunks":[{"doc_id":1,"chunk_text":"\\u4e2d ]"}]}\r\n]'
SOURCE = {7: {'available': 6, 'appended': [34, 31, 30, 32]}}
DOMAINS = {34: 'other.cn', 31: '120ask.com', 30: 'cnkang.com', 32: '120ask.com'}


def module():
    return importlib.import_module('r2ai.zh_sample.domain_append')


def test_filter_preserves_source_order_without_topup_or_alias_expansion():
    m = module()
    assert m.filter_selections(SOURCE, DOMAINS, '120ask.com') == {
        7: {'available': 2, 'appended': [31, 32]}}
    assert m.filter_selections(SOURCE, DOMAINS, 'cnkang.com') == {
        7: {'available': 1, 'appended': [30]}}
    assert m.filter_selections(SOURCE, DOMAINS, 'missing.cn') == {
        7: {'available': 0, 'appended': []}}


def test_render_preserves_literal_vi_prefix_and_chunks():
    m = module()
    out = io.StringIO()
    selected = m.filter_selections(SOURCE, DOMAINS, '120ask.com')
    m.render_selected(BASE, selected, out.write)
    assert out.getvalue() == BASE.replace('[1, 2 \r\n ]', '[1, 2 \r\n , 31, 32]')
    from r2ai.zh_sample.append import verify_gate
    assert verify_gate(BASE, out.getvalue(), selected) == {
        'queries': 1, 'vi_docs_diff': 0, 'chunks_byte_diff': 0}


@pytest.mark.parametrize('ids', [[31, 31], [True], [999], list(range(101))])
def test_invalid_source_ids_or_cap_are_rejected(ids):
    with pytest.raises(ValueError):
        module().filter_selections({7: {'appended': ids}}, DOMAINS, '120ask.com')


@pytest.mark.parametrize('selections', [{}, {7: {'appended': [1]}},
                                      {7: {'appended': []}, 8: {'appended': []}}])
def test_render_rejects_query_mismatch_or_vi_overlap(selections):
    with pytest.raises(ValueError):
        module().render_selected(BASE, selections, io.StringIO().write)


def test_source_gate_rejects_selection_reorder_and_changed_chunks():
    m = module()
    out = io.StringIO()
    m.render_selected(BASE, SOURCE, out.write)
    assert m.verify_source(BASE, out.getvalue(), SOURCE)['queries'] == 1
    for changed in (out.getvalue().replace('34, 31', '31, 34'),
                    out.getvalue().replace('\\u4e2d', '中')):
        with pytest.raises(ValueError):
            m.verify_source(BASE, changed, SOURCE)


def test_output_guard_rejects_old_vi_and_other_diagnostics():
    m = module()
    assert m.guard_output(m.DIAGNOSTICS / 'ZA-120ask/stats.json').is_relative_to(m.DIAGNOSTICS)
    for path in ('D:/GitHub/r2ai-stage3-old/out/domain', 'data/docs_vi/domain',
                 'out/runs/zh-sample/domain', 'out/runs/zh-full/index/stats.json'):
        with pytest.raises(ValueError):
            m.guard_output(path)


def test_output_guard_rejects_old_even_when_configured_output_root_is_old(monkeypatch, tmp_path):
    m = module()
    old = tmp_path / 'old'
    monkeypatch.setattr(m, 'OLD_ROOT', old, raising=False)
    monkeypatch.setattr(m, 'DIAGNOSTICS', old / 'out/runs/zh-full/diagnostics')
    with pytest.raises(ValueError):
        m.guard_output(m.DIAGNOSTICS / 'ZA-cnkang/stats.json')


def test_free_memory_gate_blocks_validation_below_four_gib():
    with pytest.raises(RuntimeError, match='memory'):
        module().require_validation_memory(4 * 1024**3 - 1)
    assert module().require_validation_memory(4 * 1024**3) == 4 * 1024**3


@pytest.mark.parametrize('macro,denominator', [('all', 3), ('has_gold', 2)])
def test_raw_id_doc_recall_deltas_add_for_disjoint_domains_with_alias_ids(tmp_path, macro, denominator):
    # IDs 10/11 represent aliases in the corpus, but metric never collapses them.
    gold = [{'id': 1, 'relevant_docs': [1, 10, 11, 20], 'relevant_chunks': []},
            {'id': 2, 'relevant_docs': [30], 'relevant_chunks': []},
            {'id': 3, 'relevant_docs': [], 'relevant_chunks': []}]
    gold_path = tmp_path / 'gold.json'
    gold_path.write_text(json.dumps(gold), encoding='utf-8')
    gold = load_submission(gold_path)
    assert gold[0]['relevant_docs'] == [1, 10, 11, 20]
    assert doc_prf([10], [10, 11])[1] == .5
    base = {1: [1], 2: [], 3: [999]}
    a = {1: [10, 11, 888], 2: [], 3: [10]}
    b = {1: [20], 2: [30, 777], 3: [20]}

    def recall(*additions):
        pred = [{'id': q, 'relevant_docs': docs + [x for part in additions for x in part[q]],
                 'relevant_chunks': []} for q, docs in base.items()]
        result = score(pred, gold, whitespace_tokenizer(), macro)
        return sum(r.get('doc_r', 0) for r in result.per_query) / result.n_doc_queries

    r0, ra, rb, rab = recall(), recall(a), recall(b), recall(a, b)
    assert r0 == pytest.approx(.25 / denominator)
    assert ra - r0 == pytest.approx(.5 / denominator)
    assert rb - r0 == pytest.approx(1.25 / denominator)
    assert rab - r0 == pytest.approx((ra - r0) + (rb - r0))
