from r2ai.paths import ROOT

from vicrawl.robots import interpret_robots

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'


def test_disallow_rule_applies():
    r = interpret_robots(200, 'User-agent: *\nDisallow: /admin\n', UA)
    assert r.state == 'robots_ok'
    assert r.can_fetch('https://x.vn/bai-viet') and not r.can_fetch('https://x.vn/admin/x')


def test_missing_robots_allows_all():
    for code in (404, 410):
        r = interpret_robots(code, '', UA)
        assert r.state == 'no_robots' and r.can_fetch('https://x.vn/a')


def test_401_403_disallow_everything():
    r = interpret_robots(403, '', UA)
    assert r.state == 'DISALLOWED' and not r.can_fetch('https://x.vn/a')


def test_5xx_is_unavailable_not_permission_and_retries_in_30_min():
    r = interpret_robots(503, '', UA, now=1000.0)
    assert r.state == 'UNAVAILABLE' and not r.can_fetch('https://x.vn/a')
    assert r.retry_at == 1000.0 + 1800


def test_timeout_is_unavailable():
    r = interpret_robots(None, '', UA, now=5.0, error='ReadTimeout')
    assert r.state == 'UNAVAILABLE' and r.retry_at == 5.0 + 1800


def test_decimal_crawl_delay_from_matching_group_only():
    txt = 'User-agent: Yahoo\nCrawl-delay: 1000\n\nUser-agent: *\nCrawl-delay: 1.5\nDisallow:\n'
    assert interpret_robots(200, txt, UA).crawl_delay == 1.5


def test_html_instead_of_robots_is_treated_as_missing():
    r = interpret_robots(200, '<!DOCTYPE html><html><body>SPA</body></html>', UA)
    assert r.state == 'no_robots_html_fallback' and r.can_fetch('https://x.vn/a')
