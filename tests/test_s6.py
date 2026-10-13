import hashlib
import json
import zipfile

import pytest

from r2ai.cc_longchau import s6
from r2ai.submit import zipio
from r2ai.submit.doc_cut import cut_a1
from r2ai.submit.writer import format_submission


def _write(tmp_path, payload=b'[\n{"id": 1, "relevant_docs": [1], "relevant_chunks": []}\n]\n'):
    p = tmp_path / 'sub.json'
    p.write_bytes(payload)
    return p, hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('method,code', [('deflate', zipfile.ZIP_DEFLATED), ('lzma', zipfile.ZIP_LZMA)])
def test_zip_methods_roundtrip(tmp_path, method, code):
    js, sha = _write(tmp_path, ('[\n' + ',\n'.join(json.dumps({'id': i, 'relevant_docs': [i], 'relevant_chunks': [{'doc_id': i, 'chunk_text': 'đau ' * 50}]},
                                                                  ensure_ascii=False, separators=(', ', ': ')) for i in range(30)) + '\n]\n').encode())
    zp = zipio.write_submission_zip(js, tmp_path / 'sub.zip', method=method)
    with zipfile.ZipFile(zp) as z:                       # plain Python zipfile extracts it
        assert [i.compress_type for i in z.infolist()] == [code] and z.namelist() == ['sub.json']
        assert z.testzip() is None
        assert hashlib.sha256(z.read('sub.json')).hexdigest() == sha
    r = zipio.verify_submission_zip(zp, sha, expected_member='sub.json')
    assert r['json_sha256_match'] and r['method'] == method


def test_default_method_is_deflate_and_cli_default(tmp_path):
    js, sha = _write(tmp_path)
    zp = zipio.write_submission_zip(js, tmp_path / 'a.zip')
    assert zipio.verify_submission_zip(zp, sha)['method'] == 'deflate'
    assert s6.METHODS[0] == 'deflate'
    from unittest import mock
    seen = {}
    with mock.patch.object(s6, 'cmd_build', lambda ns: seen.update(vars(ns)) or 0):
        s6.main(['build', '--name', 'x', '--k', '5'])
    assert seen['zip_method'] == 'deflate' and seen['protect'] is True and seen['pool'] == 'B'


def test_zip_rejects_bad_input(tmp_path):
    js, sha = _write(tmp_path)
    with pytest.raises(ValueError):
        zipio.write_submission_zip(js, tmp_path / 'b.zip', method='gzip')
    with pytest.raises(ValueError):
        zipio.write_submission_zip(js, tmp_path / 'b.zip', arcname='dir/sub.json')
    zp = zipio.write_submission_zip(js, tmp_path / 'c.zip', method='lzma', date_time=(2026, 10, 13, 3, 48, 0))
    with zipfile.ZipFile(zp) as z:
        assert z.infolist()[0].date_time == (2026, 10, 13, 3, 48, 0)
    assert not zipio.verify_submission_zip(zp, '0' * 64)['json_sha256_match']


DOCS = {1: [10, 11, 12], 2: [20, 21]}
CHUNKS = {1: [(10, 'a a'), (13, 'b b'), (11, 'c c'), (14, 'd d')], 2: [(20, 'e e'), (22, 'f f')]}


def test_render_protection_on_appends_missing_chunk_docs(tmp_path):
    p = tmp_path / 'on.json'
    c = s6.render(p, [2, 1], DOCS, CHUNKS, protect=True, cc_ids={13, 22})
    rows = json.loads(p.read_text('utf-8'))
    assert [r['id'] for r in rows] == [1, 2]
    assert rows[0]['relevant_docs'] == [10, 11, 12, 13, 14] and rows[1]['relevant_docs'] == [20, 21, 22]
    assert all(ch['doc_id'] in r['relevant_docs'] for r in rows for ch in r['relevant_chunks'])
    assert c['docs_added_by_protection_per_query'] == 1.5 and c['docs_added_by_protection_pct'] == round(100 * 3 / 8, 4)
    assert c['cc_chunks_total'] == 2 and c['chunk_docs_outside_relevant_docs_per_query'] == 0
    assert p.read_text('utf-8') == format_submission(rows)            # same bytes as the standard writer


def test_render_protection_off_keeps_docs_and_counts_outside(tmp_path):
    p = tmp_path / 'off.json'
    c = s6.render(p, [1, 2], DOCS, CHUNKS, protect=False)
    rows = json.loads(p.read_text('utf-8'))
    assert rows[0]['relevant_docs'] == [10, 11, 12] and rows[1]['relevant_docs'] == [20, 21]
    assert c['docs_added_by_protection_per_query'] == 0 and c['chunk_docs_outside_relevant_docs_per_query'] == 1.5


def test_protection_adds_nothing_when_chunk_docs_inside(tmp_path):
    p = tmp_path / 'same.json'
    inside = {1: [(10, 'a'), (11, 'b')], 2: [(21, 'c')]}
    c = s6.render(p, [1, 2], DOCS, inside, protect=True)
    assert c['docs_added_by_protection_pct'] == 0 and json.loads(p.read_text('utf-8'))[0]['relevant_docs'] == [10, 11, 12]


def test_read_docs_ignores_chunk_text_that_looks_like_the_marker(tmp_path):
    rows = [{'id': 1, 'relevant_docs': [3, 4], 'relevant_chunks': [{'doc_id': 3, 'chunk_text': 'x, "relevant_chunks": [ y'}]},
            {'id': 2, 'relevant_docs': [], 'relevant_chunks': []}]
    p = tmp_path / 'r.json'
    p.write_text(format_submission(rows), encoding='utf-8')
    assert s6.read_docs(p) == {1: [3, 4], 2: []}
    assert s6.read_chunk_doc_ids(p) == {1: {3}, 2: set()}


def test_chunk_lists_slice_and_skip_empty():
    new = {1: [5, 6, 7, 8]}
    texts = {5: 'a', 6: '  ', 7: 'c', 8: 'd'}
    assert s6.chunk_lists([1], new, texts, 3) == {1: [(5, 'a'), (7, 'c')]}
    assert s6.chunk_lists([1], new, texts, 4) == {1: [(5, 'a'), (7, 'c'), (8, 'd')]}


def test_calibrate_tau_a1_targets_mean_after_protected_docs():
    docs = {1: list(range(10)), 2: list(range(10))}
    scores = {q: {d: 10.0 - d for d in ids} for q, ids in docs.items()}
    protect = {1: {9}, 2: {8, 9}}
    tau = s6.calibrate_tau_a1(docs, scores, target_mean=5.5, floor=2, ceil=10, protect=protect)
    kept = [cut_a1(docs[q], scores[q], tau, floor=2, ceil=10, protect=protect[q])[0] for q in docs]
    assert sum(map(len, kept)) / len(kept) == 5.5


def test_guard_stays_inside_s6_run(tmp_path):
    assert s6.guard(s6.S6_RUN / 'x' / 'y.json').is_relative_to(s6.resolve_path(s6.S6_RUN))
    with pytest.raises(ValueError):
        s6.guard(tmp_path / 'y.json')
