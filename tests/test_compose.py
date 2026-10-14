import hashlib
import json
import zipfile

import pytest
import yaml

from r2ai.compose import archive as A
from r2ai.compose import build as B
from r2ai.compose import manifest as M
from r2ai.compose.base_io import base_sha, baseline, lzma_dictionary, sha
from r2ai.compose.zh_select import project_candidate_v2, select_chunks, select_docs, selections_for

MEMBER = {1: 'vi', 2: 'vi', 3: 'vi', 10: 'zh', 11: 'zh', 12: 'zh', 13: 'zh', 20: 'en'}


def cand(rank, doc_id, logit, group, v2, segment='seg-a'):
    return {'query_id': 1, 'rank': rank, 'doc_id': doc_id, 'logit': logit, 'segment': segment, 'domain': 'd',
            'chunk_id_v2': rank, 'doc_ids_group': group, 'doc_ids_v2': v2}


ROWS = [cand(1, 10, 2.0, [10, 11, 99], [10, 11]), cand(2, 12, 1.0, [12], [12]), cand(3, 13, -0.5, [13], [13]),
        cand(4, 14, -3.0, [14], [14])]


def test_select_docs_tau_cap_and_present_skip():
    got = select_docs(ROWS, [12], -1.0, 30)
    assert [r['doc_id'] for r in got['docs']] == [10, 13]            # 12 already present (primary skip); 14 below tau
    assert got['ids'] == [10, 11, 13]                                # only v2 aliases of the group, in group order
    assert [r['doc_id'] for r in select_docs(ROWS, [], -1.0, 2)['docs']] == [10, 12]


def test_select_docs_absolute_tau_independent_of_other_rows():
    only_first = select_docs(ROWS[:1], [], -0.87255859375, 30)
    assert [r['doc_id'] for r in only_first['docs']] == [10]


def test_chunks_only_for_selected_docs_and_logit_threshold():
    rows = [cand(1, 10, 2.0, [10], [10]), cand(2, 12, 1.5, [12], [12]), cand(3, 13, 0.5, [13], [13])]
    sels, body = selections_for({1: rows}, {1: []}, tau=1.6, cap=30)
    assert [r['doc_id'] for r in sels[1]['docs']] == [10]
    assert [r['doc_id'] for r in body[1]] == [10]                   # 12 passes .96 but is not a selected doc -> no chunk
    assert select_chunks(rows, 3, tau=.96)[1]['doc_id'] == 12


def test_project_candidate_v2_drops_outside_aliases_and_rejects_empty():
    rows, dropped = project_candidate_v2([cand(1, 10, 1.0, [10, 20], [10, 20])], MEMBER)
    assert rows[0]['doc_ids_v2'] == [10] and dropped == 1
    with pytest.raises(ValueError):
        project_candidate_v2([cand(1, 20, 1.0, [20], [20])], MEMBER)


def test_extend_many_appends_docs_then_whole_text_chunks_and_no_dupes():
    base = {'id': 1, 'relevant_docs': [1, 2], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'a'}]}
    sel = {'ids': [10, 11], 'docs': []}
    rows = [cand(1, 10, 2.0, [10, 11], [10, 11])]
    got, extra = A.extend_many(base, sel, rows, {('seg-a', 10): 'whole text'})
    assert got['relevant_docs'] == [1, 2, 10, 11] and extra['protected_ids'] == []
    assert [c['doc_id'] for c in got['relevant_chunks']] == [1, 10, 11]   # one chunk per v2 id
    assert base['relevant_docs'] == [1, 2]                                # input untouched


def test_shape_rejects_chunk_outside_docs_and_non_v2():
    ok = {'id': 1, 'relevant_docs': [1], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'x'}]}
    A.shape(ok, MEMBER)
    with pytest.raises(ValueError):
        A.shape({**ok, 'relevant_chunks': [{'doc_id': 2, 'chunk_text': 'x'}]}, MEMBER)
    with pytest.raises(ValueError):
        A.shape({**ok, 'relevant_docs': [999]}, MEMBER)


def write_base(path, rows):
    path.write_text('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n]\n', encoding='utf-8')


def test_render_rows_keeps_base_bytes_and_json_sha_matches_streamed_output(tmp_path):
    base = tmp_path / 'base.json'
    rows = [{'id': 1, 'relevant_docs': [1, 2], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'a'}]},
            {'id': 2, 'relevant_docs': [3], 'relevant_chunks': []}]
    write_base(base, rows)
    sels = {1: {'ids': [10], 'docs': [ROWS[0]]}, 2: {'ids': [], 'docs': []}}
    body = {1: [cand(1, 10, 2.0, [10], [10])], 2: []}
    kw = dict(base=base, selections=sels, chunk_rows=body, texts={('seg-a', 10): 'zh body'}, membership=MEMBER, vi_k=10 ** 9, expected_queries=2)
    stream = b''.join(p for p, _ in A.render_rows(**kw))
    digest, audit = A.json_sha256(**kw)
    assert digest == hashlib.sha256(stream).hexdigest()
    assert [a['new_ids'] for a in audit] == [1, 0] and [a['new_chunks'] for a in audit] == [1, 0]
    lines = stream.decode().splitlines()
    assert lines[2].rstrip(',') == json.dumps(rows[1], ensure_ascii=False)          # untouched row: bytes of the base row
    out = tmp_path / 'out.json'
    out.write_bytes(stream)
    new = [r for r, _ in baseline(out)]
    assert new[0]['relevant_docs'] == [1, 2, 10] and new[0]['relevant_chunks'][1] == {'doc_id': 10, 'chunk_text': 'zh body'}


def test_render_rows_stops_on_base_zh_chunk_order_or_missing_queries(tmp_path):
    base = tmp_path / 'base.json'
    write_base(base, [{'id': 1, 'relevant_docs': [1], 'relevant_chunks': []}])
    with pytest.raises(ValueError):                                   # selection for query 1 missing -> incomplete / mismatch
        list(A.render_rows(base=base, selections={}, chunk_rows={}, texts={}, membership=MEMBER, vi_k=10 ** 9, expected_queries=1))


def test_compression_profile_reads_dictionary_from_zip(tmp_path):
    z = tmp_path / 'x.zip'
    with lzma_dictionary(8), zipfile.ZipFile(z, 'w', compression=zipfile.ZIP_LZMA) as f:
        f.writestr('x.json', b'[\n]\n')
    prof = A.compression_profile(z)
    assert prof['dictionary_mib'] == 8 and (prof['lc'], prof['lp'], prof['pb']) == (3, 0, 2)
    assert base_sha(z) == hashlib.sha256(b'[\n]\n').hexdigest()
    assert sha(z) != base_sha(z)


def test_row_diff_reports_docs_and_chunks_by_field():
    ref = {'id': 1, 'relevant_docs': [1, 2, 3], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'a'}, {'doc_id': 2, 'chunk_text': 'b'}]}
    new = {'id': 1, 'relevant_docs': [1, 3, 2, 4], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'a2'}, {'doc_id': 5, 'chunk_text': 'c'}]}
    d = B.row_diff(ref, new)
    assert d['relevant_docs']['only_new'] == [4] and d['relevant_docs']['only_reference'] == []
    assert d['relevant_chunks']['text_differs'] == [1] and d['relevant_chunks']['doc_ids_only_new'] == [5]
    assert B.row_diff(ref, ref) == {}


def test_check_base_stats_and_scores():
    stats = {'variant': 'S6-f-K150', 'queries': 1200, 'k': 150, 'validator': {'ok': True, 'errors': 0, 'rows': 1200}}
    assert B.check_base_stats(stats) is stats
    with pytest.raises(ValueError):
        B.check_base_stats({**stats, 'variant': 'VS-append'})
    with pytest.raises(ValueError):
        B.check_base_stats({**stats, 'validator': {'ok': True, 'errors': 1, 'rows': 1200}})
    good = {'final': .2599, 'doc_f2': .2593, 'doc_p': .1603, 'doc_r': .363, 'chunk_f2': .2605, 'chunk_p': .1858, 'chunk_r': .3366}
    assert B.check_base_scores(good)['final'] == .2599
    with pytest.raises(ValueError):
        B.check_base_scores({**good, 'final': .3})
    with pytest.raises(ValueError):
        B.check_base_scores({k: v for k, v in good.items() if k != 'chunk_r'})


def test_source_manifest_pins_needs_verified_sha_and_exact_coverage(tmp_path):
    seg = {'name': 's1', 'docs_v2': 3, 's8_doc_source_verified': True, 's8_docs_file_sha256': 'a' * 64}
    assert B.source_manifest_pins({'segments': [seg, {'name': 'empty', 'docs_v2': 0}]}, tmp_path, 3) == {str(tmp_path / 's1' / 'docs.parquet'): 'a' * 64}
    with pytest.raises(ValueError):
        B.source_manifest_pins({'segments': [seg]}, tmp_path, 4)
    with pytest.raises(ValueError):
        B.source_manifest_pins({'segments': [{**seg, 's8_doc_source_verified': False}]}, tmp_path, 3)
    with pytest.raises(ValueError):
        B.source_manifest_pins({'segments': [{**seg, 's8_docs_file_sha256': 'zz'}]}, tmp_path, 3)


# ----------------------------------------------------------------------------- manifest pins

def write_manifest(tmp_path, **inputs):
    f = tmp_path / 'f.bin'
    f.write_bytes(b'abc')
    base = {'path': str(f), 'sha256': hashlib.sha256(b'abc').hexdigest()}
    m = {'name': 'T', 'inputs': {k: base for k in M.COMPOSE_PINS}}
    m['inputs'].update(inputs)
    p = tmp_path / 'm.yaml'
    p.write_text(yaml.safe_dump(m), encoding='utf-8')
    return p, m


def test_manifest_missing_pin_stops(tmp_path):
    p, m = write_manifest(tmp_path)
    del m['inputs']['query']
    p.write_text(yaml.safe_dump(m), encoding='utf-8')
    with pytest.raises(M.PinError, match='missing pin: query'):
        M.check_declared(M.load(p), M.COMPOSE_PINS)


@pytest.mark.parametrize('bad', [None, '', 'abc', 'G' * 64, 'a' * 63])
def test_manifest_malformed_sha_stops(tmp_path, bad):
    p, m = write_manifest(tmp_path)
    m['inputs']['candidate']['sha256'] = bad
    p.write_text(yaml.safe_dump(m), encoding='utf-8')
    with pytest.raises(M.PinError):
        M.check_declared(M.load(p), M.COMPOSE_PINS)


def test_manifest_verify_detects_changed_file_and_missing_file(tmp_path):
    p, m = write_manifest(tmp_path)
    got = M.verify(M.load(p), M.COMPOSE_PINS)
    assert set(got) == set(M.COMPOSE_PINS)
    (tmp_path / 'f.bin').write_bytes(b'abd')
    with pytest.raises(M.PinError, match='SHA256 differs'):
        M.verify(M.load(p), M.COMPOSE_PINS)
    (tmp_path / 'f.bin').unlink()
    with pytest.raises(M.PinError, match='missing file'):
        M.verify(M.load(p), M.COMPOSE_PINS)


def test_manifest_tree_pin_and_path_tokens(tmp_path):
    d = tmp_path / 'docs'
    d.mkdir()
    (d / 'a.parquet').write_bytes(b'1')
    (d / 'b.parquet').write_bytes(b'22')
    digest, n = M.tree_sha256(d)
    p, m = write_manifest(tmp_path, vi_docs_dir={'path': str(d), 'tree_sha256': digest, 'files': n})
    assert M.verify(M.load(p), M.NATIVE_PINS)['vi_docs_dir'] == [d]
    (d / 'c.parquet').write_bytes(b'3')
    with pytest.raises(M.PinError, match='tree differs'):
        M.verify(M.load(p), M.NATIVE_PINS)
    assert M.expand('{OUT_DIR}/x').name == 'x'
    with pytest.raises(M.PinError):
        M.expand('{NOPE}/x')


def test_policy_is_fixed_in_code(tmp_path):
    ok = {'policy': dict(M.POLICY)}
    M.check_policy(ok)
    for key, value in (('tau_doc', -1.0), ('doc_cap', 31), ('dict_mib', 64), ('ceiling_bytes', 104857600)):
        with pytest.raises(M.PinError):
            M.check_policy({'policy': {**M.POLICY, key: value}})


def test_load_job_stops_on_missing_zh_sources_before_any_output(tmp_path):
    p, m = write_manifest(tmp_path)
    m.update(policy=dict(M.POLICY), output_dir=str(tmp_path / 'out'), zh_chunk_root=str(tmp_path), expected={}, base_measured_scores={})
    p.write_text(yaml.safe_dump(m), encoding='utf-8')
    with pytest.raises(M.PinError, match='zh_sources'):
        B.load_job(p)
    assert not (tmp_path / 'out').exists()


def test_best_c1_manifest_is_complete_and_pins_every_input():
    from r2ai.paths import CONFIG_DIR
    m = M.load(CONFIG_DIR / 'best-c1.yaml')
    M.check_policy(m)
    M.check_declared(m, (*M.COMPOSE_PINS, *M.NATIVE_PINS, *M.S6F_PINS, 'zh_sources', 'reference_c1_zip'))
    assert m['expected']['json_sha256'] == '83375428b79957b1580299f97b05299bb292957826bd4cb8bdc602a83b8dd5f3'
    assert M._entries(m, 'base_json')[0]['sha256'] == 'd0da38b1ee8943b6cb3a5dbc6a7606b4ef7abd79b6cab39fe29567e3e763dca1'
    assert M._entries(m, 'candidate')[0]['sha256'] == '86d326173ce2e0ec5d6268a28416c7e6f9b2cb9f6fe11fd1371521d8d8ebd25c'
    assert len(M._entries(m, 'zh_sources')) == 57


# ----------------------------------------------------------------------------- release check

def make_ready_zip(tmp_path):
    z = tmp_path / 'R.zip'
    with lzma_dictionary(8), zipfile.ZipFile(z, 'w', compression=zipfile.ZIP_LZMA) as f:
        f.writestr('R.json', b'[\n]\n')
    validation = {'ok': True, 'errors': 0, 'rows': 1200, 'v2': True, 'cap_bytes': 102_600_000, 'zip_sha256': sha(z), 'json_sha256': base_sha(z)}
    meta = {'rows': 1200, 'ceiling_bytes': 102_600_000, 'native_validator_pending': False, 'native_validator': validation,
            'zip_sha256': sha(z), 'json_sha256': base_sha(z)}
    z.with_suffix('.validator.json').write_text(json.dumps(validation), encoding='utf-8')
    z.with_suffix('.approval.json').write_text(json.dumps(meta), encoding='utf-8')
    return z, meta, validation


def test_release_check_zip_record(tmp_path):
    from r2ai.compose import release_check as R
    z, meta, validation = make_ready_zip(tmp_path)
    assert R.check_zip(z, base_sha(z)) == []
    assert R.check_zip(z, '0' * 64)[0].startswith('JSON SHA256')
    z.with_suffix('.validator.json').write_text(json.dumps({**validation, 'errors': 1}), encoding='utf-8')
    assert any('native validator' in p for p in R.check_zip(z))
    z.with_suffix('.validator.json').unlink()
    assert R.check_zip(z)[0].startswith('missing ZIP/approval/validator record')


def test_release_check_fails_closed_when_hf_unreachable(tmp_path, monkeypatch, capsys):
    from r2ai import corpus_v2
    from r2ai.compose import release_check as R
    z, _, _ = make_ready_zip(tmp_path)
    m = tmp_path / 'm.yaml'
    m.write_text(yaml.safe_dump({'inputs': {}, 'expected': {'json_sha256': base_sha(z)}}), encoding='utf-8')

    def offline():
        raise OSError('offline')
    monkeypatch.setattr(corpus_v2, 'fetch_hf_info', offline)
    assert R.main(['--manifest', str(m), '--zip', str(z)]) == 2
    assert 'NOT READY' in capsys.readouterr().out
    monkeypatch.setattr(corpus_v2, 'main', lambda argv=None: 0)
    assert R.main(['--manifest', str(m), '--zip', str(z)]) == 0
