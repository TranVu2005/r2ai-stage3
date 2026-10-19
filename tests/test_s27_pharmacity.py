import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.s27_pharmacity.pipeline import (
    MAPPING_SCHEMA,
    Roots,
    allowed_redirect_url,
    _challenge,
    _embedding_cache_hashes,
    _embedding_cache_matches,
    _load_mapping,
    crawl_coverage,
    mapped_url,
    parse_beta_slug,
    selector_document,
    _s26_completion_marker_ok,
    title_match,
    verification_is_fresh,
    verify_document_ids,
)


def test_beta_url_mapping_keeps_slug_and_adds_htm():
    source = 'https://pmc-ecm-healthblog.beta.pharmacity.io/dau-hieu-thieu-canxi/'
    assert parse_beta_slug(source) == 'dau-hieu-thieu-canxi'
    assert mapped_url(source) == 'https://www.pharmacity.vn/dau-hieu-thieu-canxi.htm'


def test_beta_url_mapping_rejects_wrong_shape():
    assert parse_beta_slug('https://pmc-ecm-healthblog.beta.pharmacity.io/a/b/') is None
    assert parse_beta_slug('https://example.com/a/') is None


def test_crawl_coverage_requires_terminal_status_for_each_mapped_url():
    mapping = [{'doc_id': 1, 'mapped_url': 'https://www.pharmacity.vn/a.htm'},
               {'doc_id': 2, 'mapped_url': None}]
    assert crawl_coverage(mapping, {1: 'ok'})['complete'] is True
    incomplete = crawl_coverage(mapping, {1: 'retryable_error'})
    assert incomplete['complete'] is False
    assert incomplete['incomplete_urls'] == 1


def test_redirects_stay_on_secure_pharmacity_hosts():
    assert allowed_redirect_url('https://www.pharmacity.vn/dau-hieu.htm')
    assert allowed_redirect_url('https://pharmacity.vn/dau-hieu.htm')
    assert not allowed_redirect_url('https://other.example/dau-hieu.htm')
    assert not allowed_redirect_url('http://www.pharmacity.vn/dau-hieu.htm')
    assert not allowed_redirect_url('https://www.pharmacity.vn:8443/dau-hieu.htm')
    assert not allowed_redirect_url('https://user@www.pharmacity.vn/dau-hieu.htm')


def test_challenge_classifier_returns_reason(monkeypatch):
    import vicrawl.classify as classify

    monkeypatch.setattr(classify, 'classify_response', lambda **kwargs: ('bot_challenge', 'challenge marker'))
    assert _challenge(b'<html></html>', 404) == (True, 'challenge marker')


def test_verification_matches_full_provenance_and_only_eligible_handoff_docs():
    extracted = [
        {'doc_ids': [11], 'status': 'ok', 'n_tokens_bge_m3': 50, 'body': 'x' * 200},
        {'doc_ids': [12], 'status': 'thin', 'n_tokens_bge_m3': 10, 'body': 'short'},
    ]
    result = verify_document_ids(extracted, [11], [11, 12], [11])
    assert result['extracted_doc_ids_unique']
    assert result['handoff_doc_ids_unique']
    assert result['provenance_ids_equal']
    assert result['handoff_doc_ids_match_eligible']
    assert result['chunk_doc_ids_present']


def test_embedding_gates_require_fresh_verification_and_s26_done_marker():
    hashes = {'chunks': 'abc'}
    assert verification_is_fresh({'complete': True, 'artifacts_sha256': hashes}, hashes)
    assert not verification_is_fresh({'complete': True, 'artifacts_sha256': {'chunks': 'old'}}, hashes)
    assert not _s26_completion_marker_ok('block 153 done soon')
    assert _s26_completion_marker_ok('block 153 done\ndone 887151')


def test_embedding_block_cache_is_bound_to_ordered_text_and_encoder_config(tmp_path):
    ids = np.asarray([10, 11], dtype=np.int64)
    texts = ['article a', 'article b']
    input_hash, config_hash = _embedding_cache_hashes(ids, texts)
    path = tmp_path / 'block.npz'
    np.savez(path, doc_id=ids, dense=np.zeros((2, 1024), dtype=np.float16),
             input_sha256=np.asarray(input_hash), config_sha256=np.asarray(config_hash))
    with np.load(path, allow_pickle=False) as cache:
        assert _embedding_cache_matches(cache, ids, input_hash, config_hash)
        changed_hash, _ = _embedding_cache_hashes(ids, ['article a edited', 'article b'])
        assert not _embedding_cache_matches(cache, ids, changed_hash, config_hash)
        assert not _embedding_cache_matches(cache, ids, input_hash, 'changed-config')
        reordered_hash, _ = _embedding_cache_hashes(ids[::-1], texts[::-1])
        assert not _embedding_cache_matches(cache, ids[::-1], reordered_hash, config_hash)


def test_load_mapping_revalidates_parquet_against_recorded_corpus(monkeypatch, tmp_path):
    import r2ai.s27_pharmacity.pipeline as pipeline

    roots = Roots(root=tmp_path / 'batch')
    roots.mapping.parent.mkdir(parents=True)
    expected = [{'doc_id': 7, 'original_url': 'https://source/a', 'mapped_url': 'https://target/a.htm',
                 'slug': 'a', 'source_type': 'beta', 'slug_parseable': True}]
    pq.write_table(pa.Table.from_pylist(expected, schema=MAPPING_SCHEMA), roots.mapping)
    report = {'source_corpus': str(tmp_path / 'corpus.parquet'), 'source_corpus_sha256': 'source-hash'}
    roots.reports.mkdir(parents=True)
    (roots.reports / 'mapping.json').write_text(json.dumps(report), encoding='utf-8')
    monkeypatch.setattr(pipeline, 'build_mapping', lambda source: (expected, {'source_corpus_sha256': 'source-hash'}))
    assert _load_mapping(roots) == expected

    different = [dict(expected[0], mapped_url='https://target/changed.htm')]
    pq.write_table(pa.Table.from_pylist(different, schema=MAPPING_SCHEMA), roots.mapping)
    import pytest
    with pytest.raises(ValueError, match='differs from source corpus'):
        _load_mapping(roots)


def test_title_match_normalizes_diacritics_stopwords_and_numbers():
    score = title_match('cach-dau-hieu-thieu-canxi-2024', 'Dấu hiệu thiếu canxi là gì?')
    assert score['matched'] == 4
    assert score['terms'] == 4
    assert score['ratio'] == 1.0


def test_title_match_normalizes_vietnamese_d_stroke():
    score = title_match('duong-dan', 'Đường dẫn thông tin')
    assert score['matched'] == 2
    assert score['ratio'] == 1.0


def test_title_match_uses_half_threshold():
    assert title_match('thieu-hut-collagen-omega-3', 'Thiếu hụt collagen') ['accepted'] is True
    assert title_match('thieu-hut-collagen-omega-3-vitamin-d', 'Tin tức Pharmacity') ['accepted'] is False


def test_selector_document_reads_article_only():
    html = '''<html><head><title>Article title</title></head><body>
      <h1>Tiêu đề bài</h1><nav>Điều hướng</nav>
      <main id="pmc-content-blog-detail"><p>Nội dung bài báo đủ dài.</p><h2>Tiêu đề mục</h2><p>Đoạn tiếp theo.</p></main>
      <aside>Quảng cáo</aside></body></html>'''
    doc = selector_document(html)
    assert doc['title'] == 'Tiêu đề bài'
    assert 'Nội dung bài báo đủ dài.' in doc['body']
    assert 'Điều hướng' not in doc['body']
    assert 'Quảng cáo' not in doc['body']
