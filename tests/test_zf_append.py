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
