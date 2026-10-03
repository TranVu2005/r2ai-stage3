from r2ai.paths import LEGACY_OUT_DIR, ROOT, resolve_legacy_artifact

import csv
import pathlib

from vicrawl.classify import STATES, analyze_html, classify_response, find_cookie_challenge


def run(html: str, **kw):
    stats = analyze_html(html)
    base = dict(url='https://x.vn/a', final_url='https://x.vn/a', http_status=200, error='', robots='robots_ok')
    base.update(kw)
    return classify_response(stats=stats, raw=html, **base)[0]


ARTICLE = '<html><title>Bai viet</title><body><article>' + ''.join('<p>Noi dung dieu tri benh nhan rat dai.</p>' for _ in range(20)) + '</article></body></html>'


def test_ok_article():
    assert run(ARTICLE) == 'ok'


def test_thin_short_page():
    assert run('<html><title>t</title><p>Short.</p></html>') == 'thin'


def test_cookie_challenge_detected_and_parsed():
    html = '<script>document.cookie="D1N=abc123"+"; path=/";window.location.reload(true);</script>'
    assert find_cookie_challenge(html) == ('D1N', 'abc123')
    assert run(html) == 'cookie_challenge'


def test_bot_challenge_cloudflare():
    html = '<title>Just a moment...</title><script src="/cdn-cgi/challenge-platform/a"></script>'
    assert run(html, http_status=403) == 'bot_challenge'


def test_article_mentioning_captcha_is_ok():
    html = ARTICLE.replace('Bai viet', 'Captcha study') + '<div class="g-recaptcha"></div>'
    assert run(html) == 'ok'


def test_home_redirect_is_soft404():
    assert run(ARTICLE, final_url='https://x.vn/') == 'soft404_or_home'


def test_http_codes():
    assert run('', http_status=404) == 'soft404_or_home'
    assert run('', http_status=403) == 'blocked_4xx'
    assert run('', http_status=429) == 'blocked_4xx'
    assert run('', http_status=503) == 'dead_origin'
    assert run('', http_status=502) == 'dead_origin'


def test_network_error_and_robots():
    assert run('', http_status=None, error='ConnectError: x') == 'network_error'
    assert run('', http_status=None, error='robots_disallowed', robots='DISALLOWED') == 'blocked_4xx'


def test_spa_shell_maps_to_thin():
    assert run('<div id="app"></div><script src="b.js"></script>') == 'thin'


def test_states_closed_set():
    assert set(STATES) == {'ok', 'thin', 'soft404_or_home', 'cookie_challenge', 'bot_challenge', 'blocked_4xx', 'dead_origin', 'network_error'}


def test_parity_with_fetcher_on_samples():
    """New classifier must agree with the probe-era fetcher.classify on saved HTML."""
    from r2ai.probe import fetcher
    rows = list(csv.DictReader(open(LEGACY_OUT_DIR / 'crawl_sample.csv', encoding='utf-8-sig')))
    n = agree = 0
    for r in rows:
        p = resolve_legacy_artifact(r['raw_path']) if r.get('raw_path') else None
        if not p or not p.exists() or not r.get('status'):
            continue
        body = p.read_bytes()
        rec = {'url': r['url'], 'final_url': r.get('final_url') or r['url'], 'http_status': r['status'], 'error': r.get('error', ''), 'robots': r.get('robots', '')}
        old = fetcher.classify(rec, body)['state']
        old = {'needs_js_real': 'thin', 'other': 'thin'}.get(old, old)
        raw = body.decode('utf-8', errors='replace')
        new = classify_response(stats=analyze_html(raw), raw=raw, url=rec['url'], final_url=rec['final_url'], http_status=int(rec['http_status']), error=rec['error'], robots=rec['robots'])[0]
        n += 1
        agree += old == new
    assert n > 100
    assert agree / n >= 0.9, (agree, n)


def test_428_rate_limit_is_retryable_blocked_4xx():
    from vicrawl.classify import is_transient
    page = '<html><head><title>Server maintaining...</title></head><body><h1> 428 Too Many Requests via Varnish</h1></body></html>'
    assert run(page, http_status=428) == 'blocked_4xx'
    assert is_transient('blocked_4xx', 428) and is_transient('blocked_4xx', 429)
    assert not is_transient('blocked_4xx', 403)


def test_aspnet_page_wrapped_in_form_is_ok_not_thin():
    """qdnd.vn (DNN/ASP.NET) wraps the whole page in <form id="Form">; the article inside must still count."""
    raw = (pathlib.Path(__file__).resolve().parent / 'fixtures' / 'qdnd.vn__8eef6385.html').read_text(encoding='utf-8')
    stats = analyze_html(raw)
    assert stats.text_len >= 1500
    assert run(raw) == 'ok'


def test_small_search_form_still_dropped_as_chrome():
    page = ('<html><title>t</title><body><form><p>Tìm kiếm</p><input></form>'
            + '<article>' + ''.join('<p>Noi dung dieu tri benh nhan rat dai.</p>' for _ in range(3)) + '</article></body></html>')
    assert analyze_html(page).text_len == 3 * len('Noi dung dieu tri benh nhan rat dai.')
