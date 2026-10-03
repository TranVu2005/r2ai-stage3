from r2ai.paths import ROOT

from vicrawl.urlnorm import normalize_url, domain_of, group_urls


def test_normalize_drops_scheme_www_fragment_trailing_slash_lowercases_host():
    assert normalize_url('HTTPS://WWW.Example.VN/Bai-Viet/?a=1#frag') == 'example.vn/Bai-Viet?a=1'
    assert normalize_url('http://example.vn/') == 'example.vn'
    assert normalize_url('http://example.vn') == 'example.vn'


def test_normalize_keeps_query_and_path_case():
    assert normalize_url('https://x.vn/A?q=1') != normalize_url('https://x.vn/A?q=2')
    assert normalize_url('https://x.vn/A') != normalize_url('https://x.vn/a')


def test_normalize_default_ports_dropped_other_kept():
    assert normalize_url('http://x.vn:80/a') == 'x.vn/a'
    assert normalize_url('https://x.vn:443/a') == 'x.vn/a'
    assert normalize_url('http://127.0.0.1:8080/a') == '127.0.0.1:8080/a'


def test_http_https_www_variants_collapse():
    urls = ['http://x.vn/a', 'https://x.vn/a', 'https://www.x.vn/a/', 'http://www.x.vn/a']
    assert len({normalize_url(u) for u in urls}) == 1


def test_domain_of_strips_www_and_lowercases():
    assert domain_of('https://WWW.Vinmec.com/vie/x') == 'vinmec.com'
    assert domain_of('http://phunusuckhoe.giadinhonline.vn/a') == 'phunusuckhoe.giadinhonline.vn'


def test_group_urls_prefers_https_and_keeps_all_doc_ids():
    rows = [(5, 'http://www.x.vn/a'), (3, 'https://x.vn/a'), (9, 'https://www.x.vn/a/'), (7, 'https://x.vn/b')]
    groups = {g['url_norm']: g for g in group_urls(rows)}
    assert set(groups) == {'x.vn/a', 'x.vn/b'}
    g = groups['x.vn/a']
    assert g['url'].startswith('https://')
    assert g['doc_ids'] == [3, 5, 9]
    assert g['domain'] == 'x.vn'
    assert groups['x.vn/b']['doc_ids'] == [7]
