import io
import json

import pytest

from r2ai.zh_sample import append
from r2ai.zh_sample.append import (
    RD150_SHA256, guarded_output, pinned_base, render_append, select_docs, verify_gate,
)


BASE = '[\r\n {"id": 7, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id":1,"chunk_text":"\\u4e2d relevant_docs ]"}]}\r\n]'


def render(base, cache, known):
    out = io.StringIO()
    rows = render_append(base, cache, known, out.write)
    return out.getvalue(), rows


def test_raw_vi_bytes_and_escaped_chunk_are_preserved():
    result, rows = render(BASE, {7: [[31, -2], [30, 4]]}, {30, 31})
    assert result == BASE.replace('[1, 2]', '[1, 2, 30, 31]')
    assert rows[7]['available'] == 2
    assert verify_gate(BASE, result, rows) == {'queries': 1, 'vi_docs_diff': 0, 'chunks_byte_diff': 0}


def test_cap_short_query_and_max_chunk_score_stable_ties():
    scores = [[d, d / 10] for d in range(1000, 1120)]
    selected, available = select_docs(scores, set(range(1000, 1120)), [], 100)
    assert available == 120 and selected == list(range(1119, 1019, -1))
    assert select_docs([[30, 1], [31, 2], [30, 3], [32, 3]], {30, 31, 32}, [], 100) == ([30, 32, 31], 3)
    assert select_docs([[30, -4], [31, -6]], {30, 31}, [30], 100) == ([31], 1)


@pytest.mark.parametrize('pairs', [[[True, 1]], [[30, float('nan')]], [[30, float('inf')]], [[99, 1]], [[30, True]]])
def test_invalid_cache_is_rejected(pairs):
    with pytest.raises(ValueError):
        select_docs(pairs, {30}, [], 100)


@pytest.mark.parametrize('cache', [{}, {7: [], 8: []}])
def test_query_mismatch_is_rejected(cache):
    with pytest.raises(ValueError, match='query'):
        render(BASE, cache, set())


def test_duplicate_queries_and_duplicate_object_keys_are_rejected():
    row = json.loads(BASE)[0]
    with pytest.raises(ValueError):
        render(json.dumps([row, row]), {7: []}, set())
    with pytest.raises(ValueError):
        render(BASE.replace('"id": 7', '"id": 7, "id": 7'), {7: []}, set())


def test_gate_detects_reordering_and_chunk_byte_changes():
    result, rows = render(BASE, {7: [[30, 1]]}, {30})
    for changed in (result.replace('[1, 2,', '[2, 1,'), result.replace('\\u4e2d', '中')):
        with pytest.raises(ValueError):
            verify_gate(BASE, changed, rows)


def test_pin_and_output_guard_fail_before_writing(tmp_path):
    p = tmp_path / 'base.json'
    p.write_text(BASE, encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        pinned_base(p)
    assert len(RD150_SHA256) == 64
    for bad in ('D:/GitHub/r2ai-stage3-old/out/Z', 'D:/GitHub/r2ai-stage3/data/docs_vi/Z', 'D:/GitHub/r2ai-stage3/out/runs/H1/Z'):
        with pytest.raises(ValueError):
            guarded_output(bad)


def test_empty_vi_docs_and_trailing_data():
    base = '[{"id":7,"relevant_docs": [],"relevant_chunks": []}]'
    result, rows = render(base, {7: [[30, 1]]}, {30})
    assert json.loads(result)[0]['relevant_docs'] == [30]
    assert verify_gate(base, result, rows)['vi_docs_diff'] == 0
    with pytest.raises(ValueError):
        render(base + ' garbage', {7: []}, set())


def test_multiple_rows_with_other_field_order_and_unicode_whitespace():
    base = '[{"relevant_chunks":[],"relevant_docs":[1 \r\n ],"id":7},\r\n{"id":8,"relevant_chunks":[{"doc_id":2,"chunk_text":"中"}],"relevant_docs":[2]}]'
    result, rows = render(base, {7: [[30, 5]], 8: [[31, 4]]}, {30, 31})
    assert result == base.replace('[1 \r\n ]', '[1 \r\n , 30]').replace('[2]', '[2, 31]')
    assert verify_gate(base, result, rows)['queries'] == 2


def _private_test_root(monkeypatch, tmp_path):
    # Isolated synthetic filesystem, independent of the real owned output guard.
    monkeypatch.setattr(append, 'RUN', tmp_path)
    monkeypatch.setattr(append, 'guard_owned', lambda p: p.resolve())


@pytest.mark.parametrize('exception', [ValueError, KeyboardInterrupt])
def test_validator_failure_or_interrupt_does_not_publish(monkeypatch, tmp_path, exception):
    _private_test_root(monkeypatch, tmp_path)
    final_json, final_zip, stats = [tmp_path / n for n in ('Z_append.json', 'Z_append.zip', 'stats.json')]
    for p in (final_json, final_zip, stats):
        p.write_bytes(b'previous success')
    stage = tmp_path / 'Z_append.pending.zip'
    stage.write_bytes(b'staging')

    def fail(args):
        assert args[0].endswith('.zip')
        if exception is KeyboardInterrupt:
            raise KeyboardInterrupt
        print('{"errors":1,"ok":false}')
        return 1

    with pytest.raises(exception):
        append.validate_stage(stage, fail, [], tmp_path / 'validator.log')
    assert all(p.read_bytes() == b'previous success' for p in (final_json, final_zip, stats))


def test_interrupted_publication_invalidates_old_success(monkeypatch, tmp_path):
    _private_test_root(monkeypatch, tmp_path)
    monkeypatch.setattr(append, 'atomic_json', lambda p, v: p.write_text(json.dumps(v)))
    paths = [tmp_path / name for name in ('a.tmp', 'a.pending.zip', 'a.json', 'a.zip')]
    for p in paths:
        p.write_bytes(b'data')

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(append.os, 'replace', interrupted)
    with pytest.raises(KeyboardInterrupt):
        append.publish_artifacts(*paths, {'validator': {'errors': 0}})
    assert json.loads((tmp_path / 'stats.json').read_text())['complete'] is False


def test_successful_publication_orders_metadata_inside_incomplete_window(monkeypatch, tmp_path):
    _private_test_root(monkeypatch, tmp_path)
    writes = []

    def atomic(p, value):
        writes.append((p.name, value))
        p.write_text(json.dumps(value))

    monkeypatch.setattr(append, 'atomic_json', atomic)
    paths = [tmp_path / name for name in ('a.tmp', 'a.pending.zip', 'a.json', 'a.zip')]
    paths[0].write_bytes(b'new JSON')
    paths[1].write_bytes(b'new ZIP')
    append.publish_artifacts(*paths, {'validator': {'errors': 0}}, {7: {'appended': [30]}})
    assert writes[0][0] == 'stats.json' and writes[0][1]['complete'] is False
    assert writes[1][0] == 'query_append.json'
    assert writes[2][0] == 'stats.json' and writes[2][1]['complete'] is True
    assert paths[2].read_bytes() == b'new JSON' and paths[3].read_bytes() == b'new ZIP'


@pytest.mark.parametrize('problem', ['checksum', 'missing', 'extra', 'wrong_id'])
def test_cache_bundle_and_query_integrity(monkeypatch, tmp_path, problem):
    monkeypatch.setattr(append, 'RUN', tmp_path)
    monkeypatch.setattr(append, 'INDEX', tmp_path / 'index')
    monkeypatch.setattr(append, 'QUERY_FILE', tmp_path / 'query.parquet')
    monkeypatch.setattr(append, 'digest', lambda p: 'bad' if problem == 'checksum' and p == append.QUERY_FILE else 'ok')
    (tmp_path / 'index').mkdir()
    (tmp_path / 'retrieval').mkdir()
    (tmp_path / 'index/input_manifest.json').write_text(json.dumps({'extract_manifest_sha256': 'ok', 'chunks_sha256': 'ok'}))
    config = {'query_sha256': 'ok', 'chunk_sha256': 'ok', 'index_manifest_sha256': 'ok', 'seed': 42,
              'vi': {'reranker': 'BAAI/bge-reranker-v2-m3'}}
    (tmp_path / 'retrieval/config.json').write_text(json.dumps(config))
    if problem != 'missing':
        (tmp_path / 'retrieval/q7.json').write_text(json.dumps({'query_id': 8 if problem == 'wrong_id' else 7, 'dense': []}))
    if problem == 'extra':
        (tmp_path / 'retrieval/q8.json').write_text('{}')
    with pytest.raises(ValueError):
        append._load_cache({7})
