import copy

import pytest
import yaml

from r2ai.compose import batch3 as C
from r2ai.compose import build as B
from r2ai.compose import manifest as M

SHA = 'ff20ae1a40e4251425e5fe02c2827704a414e34a3d65132518db8d24baf7225b'


def config():
    return {'name': 'C3', 'variant': 'C3-batch3', 'policy': {**M.POLICY, 'candidate_batch': 3}, 'expected': {'coverage_docs': 152036},
            'inputs': {'candidate': {'path': 'x', 'sha256': SHA}, 'source_snapshot_report': {'path': 'qa.json', 'sha256': 'a' * 64}}}


def snapshot():
    return {'s8_variant': 'C3-batch3', 'segments': [{'name': 'old', 'docs_v2': 2}, {'name': 'new', 'docs_v2': 1}],
            'superseded': [{'doc_id': 1, 'segment': 'old', 'by': 'new', 'batch': 'redo'}],
            's8_live_docs': 2, 's8_physical_docs': 3, 's8_superseded_docs': 1}


def test_c1_default_stays_batch2_and_c3_is_explicit():
    M.check_policy({'policy': dict(M.POLICY)})
    assert M.POLICY['candidate_batch'] == 2
    with pytest.raises(M.PinError):
        M.check_policy({'policy': {**M.POLICY, 'candidate_batch': 3}})
    M.check_policy(config())
    assert M.POLICY['candidate_batch'] == 2


@pytest.mark.parametrize('field,value', [('tau_doc', 0), ('doc_cap', 31), ('candidate_batch', 2), ('chunk_tau', .44),
                                         ('chunk_doc_cap', 1), ('dict_mib', 64)])
def test_c3_cannot_change_selection(field, value):
    m = config()
    m['policy'][field] = value
    with pytest.raises(ValueError):
        M.check_policy(m)


@pytest.mark.parametrize('change', [lambda m: m['inputs'].pop('source_snapshot_report'),
                                    lambda m: m['inputs']['candidate'].update(sha256='b' * 64),
                                    lambda m: m['expected'].update(coverage_docs=153604),
                                    lambda m: m.update(name='C1')])
def test_c3_gate_missing_identity_or_pin(change):
    m = config()
    change(m)
    with pytest.raises(ValueError):
        M.check_policy(m)


def test_c3_name_requires_optin_even_with_batch2():
    m = config()
    m.pop('variant')
    m['policy']['candidate_batch'] = 2
    with pytest.raises(ValueError):
        M.check_policy(m)


def test_live_count_and_invalid_superseded():
    m = snapshot()
    assert C.live_count(m) == 2
    for mutate in [lambda x: x['superseded'].append(dict(x['superseded'][0])),
                   lambda x: x['superseded'][0].update(by='unknown'),
                   lambda x: x['superseded'][0].update(by='old'),
                   lambda x: x.update(s8_live_docs=3),
                   lambda x: x.update(s8_physical_docs=4)]:
        bad = copy.deepcopy(m)
        mutate(bad)
        with pytest.raises(ValueError):
            C.live_count(bad)


def test_superseded_cycle_rejected():
    m = snapshot()
    m['superseded'] = [{'doc_id': 1, 'segment': 'old', 'by': 'new'}, {'doc_id': 1, 'segment': 'new', 'by': 'old'}]
    m['s8_live_docs'] = 1
    m['s8_superseded_docs'] = 2
    with pytest.raises(ValueError, match='cycle'):
        C.live_count(m)


def test_superseded_candidate_rejected():
    m = snapshot()
    with pytest.raises(ValueError):
        C.check_candidate_live([{'segment': 'old', 'doc_id': 1}], m)
    C.check_candidate_live([{'segment': 'new', 'doc_id': 1}], m)


def test_doc_mean_stop_gate():
    C.check_mean({q: {'docs': [0] * 20} for q in range(1200)})
    with pytest.raises(ValueError):
        C.check_mean({q: {'docs': [0] * 21} for q in range(1200)})
    with pytest.raises(ValueError):
        C.check_mean({q: {'docs': []} for q in range(1199)})


def test_source_report_must_match_candidate_snapshot_and_pins(tmp_path):
    report = {'ok': True, 'candidate_sha256': SHA, 'snapshot_manifest_sha256': 'c' * 64, 'physical_docs': 153604,
              'superseded_docs': 1568, 'live_docs': 152036, 'source_pins': {'a': 'd' * 64}}
    p = tmp_path / 'qa.json'
    import json
    p.write_text(json.dumps(report), encoding='utf-8')
    m = {'inputs': {'zh_manifest': {'path': 'm', 'sha256': 'c' * 64}}}
    C.check_source_report(m, {'source_snapshot_report': p}, {'a': 'd' * 64})
    for bad_pins in ({'a': 'e' * 64}, {}):
        with pytest.raises(ValueError):
            C.check_source_report(m, {'source_snapshot_report': p}, bad_pins)
    p.write_text(json.dumps({**report, 'ok': False}), encoding='utf-8')
    with pytest.raises(ValueError):
        C.check_source_report(m, {'source_snapshot_report': p}, {'a': 'd' * 64})


def test_c1_source_manifest_count_unchanged(tmp_path):
    s = {'segments': [{'name': 'old', 'docs_v2': 1, 'verified': True, 's8_docs_file_sha256': 'a' * 64}]}
    assert len(B.source_manifest_pins(s, tmp_path, 1)) == 1
    with pytest.raises(ValueError):
        B.source_manifest_pins(s, tmp_path, 2)


def test_c3_source_manifest_uses_live_coverage(tmp_path):
    s = {**snapshot(), 'segments': [{'name': n, 'docs_v2': c, 's8_doc_source_verified': True, 's8_docs_file_sha256': 'a' * 64}
                                     for n, c in (('old', 2), ('new', 1))]}
    assert len(B.source_manifest_pins(s, tmp_path, 2)) == 2          # 3 physical - 1 superseded
    with pytest.raises(ValueError):
        B.source_manifest_pins(s, tmp_path, 3)


def test_legacy_generator_sources_stay_supported(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.compose.base_io import sha
    from r2ai.compose.zh_select import load_source_texts
    folder = tmp_path / 'seg'
    folder.mkdir()
    p = folder / 'docs.parquet'
    pq.write_table(pa.Table.from_pylist([{'doc_id': 1, 'doc_ids_group': [1], 'doc_ids_v2': [1], 'doc_text': 'whole text', 'url': 'url'}]), p)
    cp = tmp_path / 'corpus.parquet'
    pq.write_table(pa.table({'id': [1], 'lang': ['zh']}), cp)
    m = {'segments': [{'name': 'seg', 's8_doc_source_verified': True, 's8_docs_file_sha256': sha(p)}]}
    src, _ = load_source_texts(iter([{'segment': 'seg', 'doc_id': 1, 'doc_ids_group': [1], 'doc_ids_v2': [1]}]), m, tmp_path, cp)
    assert src[('seg', 1)]['doc_text'] == 'whole text'


def test_load_source_texts_rejects_superseded_candidate(tmp_path):
    from r2ai.compose.zh_select import load_source_texts
    m = {**snapshot(), 'segments': [{'name': 'old', 'docs_v2': 2, 's8_doc_source_verified': True, 's8_docs_file_sha256': 'a' * 64},
                                    {'name': 'new', 'docs_v2': 1, 's8_doc_source_verified': True, 's8_docs_file_sha256': 'a' * 64}]}
    with pytest.raises(ValueError, match='superseded'):
        load_source_texts([{'segment': 'old', 'doc_id': 1}], m, tmp_path, tmp_path / 'none.parquet')


def test_c3_manifest_is_complete_and_pins_every_input():
    from r2ai.paths import CONFIG_DIR
    m = M.load(CONFIG_DIR / 'c3.yaml')
    M.check_policy(m)
    M.check_declared(m, (*M.COMPOSE_PINS, *M.NATIVE_PINS, *M.S6F_PINS, 'zh_sources', 'reference_zip', 'source_snapshot_report'))
    assert m['expected']['json_sha256'] == '6cf9284d90b9fad816425ef33865a58065b35513e43b61eb043487b7978d3436'
    assert m['expected']['zip_sha256'] == '1c2472d1ace8e241a018f2f861d997ed055c064c4f76715b3a761fe70cbb1af9'
    assert M._entries(m, 'candidate')[0]['sha256'] == SHA
    assert M._entries(m, 'base_json')[0]['sha256'] == 'd0da38b1ee8943b6cb3a5dbc6a7606b4ef7abd79b6cab39fe29567e3e763dca1'
    assert len(M._entries(m, 'zh_sources')) == 62
    assert yaml.safe_load((CONFIG_DIR / 'best-c1.yaml').read_text(encoding='utf-8'))['policy']['candidate_batch'] == 2
