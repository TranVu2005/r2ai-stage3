from r2ai.s27_pharmacity.cc_backfill import (classify_capture_group, index_surt_keys,
                                             index_lookup_key, match_warc_target, pending_cc_ids)


def test_warc_target_match_distinguishes_exact_normalized_and_mismatch():
    mapped = 'https://www.pharmacity.vn/dau-hieu.htm'
    assert match_warc_target(mapped, mapped) == 'exact'
    assert match_warc_target(mapped, 'http://pharmacity.vn/dau-hieu.htm/') == 'normalized'
    assert match_warc_target(mapped, 'https://www.pharmacity.vn/bai-viet-khac.htm') == 'mismatch'


def test_index_lookup_checks_www_and_bare_host_surt_forms():
    assert index_surt_keys('https://www.pharmacity.vn/duong-dan.htm') == {
        'vn,pharmacity)/duong-dan.htm', 'vn,pharmacity,www)/duong-dan.htm'}


def test_index_lookup_preserves_query_variant_for_later_warc_match_check():
    assert index_lookup_key('vn,pharmacity)/a.htm?from=home') == 'vn,pharmacity)/a.htm'


def test_cc_scope_excludes_every_doc_already_in_live_state_including_unknown():
    assert pending_cc_ids([1, 2, 3, 4], [1, 3]) == [2, 4]


def test_capture_group_reports_no_capture_dead_usable_and_unusable():
    assert classify_capture_group([], []) == 'no_capture'
    dead = [{'fetch_status': 404}, {'fetch_status': 410}]
    assert classify_capture_group(dead, []) == 'dead'
    mixed = dead + [{'fetch_status': 200}]
    assert classify_capture_group(mixed, [{'status': 'body_too_short'}]) == 'unusable'
    assert classify_capture_group(mixed, [{'status': 'ok'}]) == 'usable'
