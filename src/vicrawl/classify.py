"""Exclusive response classification. Same rules as fetcher.classify / step1_reclassify, but takes
pre-computed page stats so the crawl loop never calls trafilatura (extract.py refines ok/thin later)."""
from __future__ import annotations

from r2ai.paths import ROOT

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from lxml import html as lhtml

STATES = ('ok', 'thin', 'soft404_or_home', 'cookie_challenge', 'bot_challenge', 'blocked_4xx', 'dead_origin', 'network_error')
# 4xx that are worth another try later (rate-limit style); everything else 4xx is final.
# 428: dantri.com.vn's Varnish answers "428 Too Many Requests" when it rate-limits.
RETRYABLE_4XX = (408, 425, 428, 429)
TRANSIENT = ('network_error', 'dead_origin', 'bot_challenge', 'cookie_challenge')

COOKIE = re.compile(r'document\s*\.\s*cookie\s*=\s*["\'](\w+)=(\w+)["\']', re.I)
BOT = re.compile(r'just a moment|attention required|access denied|verify you are human|are you a robot|验证码|滑动验证|安全验证|请完成验证|访问频繁|captcha', re.I)
SOFT404 = re.compile(r'^(?:404(?:\s|$)|page not found|not found|trang không tồn tại|không tìm thấy trang|页面不存在|页面未找到)', re.I)
_CHALLENGE_RAW = re.compile(r'cf-chl|cf_chl|/cdn-cgi/challenge-platform|g-recaptcha|hcaptcha', re.I)
_SPA = re.compile(r'__NEXT_DATA__|<[^>]+id\s*=\s*["\'](?:app|root|__next)["\']|<app-root\b', re.I)
_JUNK = '//script|//style|//noscript|//template'
# A <form> is chrome (search / login box) unless it wraps the page: ASP.NET/DNN sites (qdnd.vn) put everything in <form id="Form">.
_CHROME = '//nav|//header|//footer|//aside|//form[count(.//p) < 5]|//select|//button'
_BLOCKS = '//p|//h1|//h2|//h3|//h4|//td|//th|//pre|//blockquote|//dd|//dt|//figcaption|//li'


@dataclass
class PageStats:
    title: str = ''
    visible_len: int = 0
    visible_head: str = ''   # first chars of visible text, for regexes only
    text_len: int = 0        # cheap proxy for main-content length (trafilatura replaces it in extract.py)


def find_cookie_challenge(raw: str):
    m = COOKIE.search(raw)
    return (m[1], m[2]) if m else None


def analyze_html(raw: str) -> PageStats:
    if not raw or not raw.strip():
        return PageStats()
    try:
        tree = lhtml.fromstring(re.sub(r'^\s*<\?xml[^>]*>', '', raw))
    except Exception:
        return PageStats()
    try:
        title = ' '.join(tree.xpath('//title/text()')).strip()
        for n in tree.xpath(_JUNK):
            n.drop_tree()
        visible = re.sub(r'\s+', ' ', tree.text_content()).strip()
        for n in tree.xpath(_CHROME):
            n.drop_tree()
        total = 0
        for el in tree.xpath(_BLOCKS):
            if el.tag == 'li' and el.xpath('.//p|.//li'):
                continue  # count nested blocks once
            t = re.sub(r'\s+', ' ', el.text_content()).strip()
            if el.tag == 'li' and len(t) < 40:
                continue  # menu items
            if el.xpath('.//p') and el.tag not in ('p',):
                continue
            total += len(t)
    except Exception:
        return PageStats()
    return PageStats(title=title, visible_len=len(visible), visible_head=visible[:6000], text_len=total)


def classify_response(*, stats: PageStats, raw: str, url: str, final_url: str, http_status, error: str = '', robots: str = '') -> tuple[str, str]:
    """Return (state, reason); exactly one of STATES."""
    code = int(http_status) if http_status not in (None, '') else None
    final = final_url or url
    title, text_len, visible_len = stats.title, stats.text_len, stats.visible_len
    original_path, final_path = urlsplit(url).path, urlsplit(final).path
    cookie = find_cookie_challenge(raw)
    small = text_len < 2000 and visible_len < 4000
    challenge = small and (
        re.search(r'^(?:just a moment|attention required|access denied)', title, re.I)
        or (text_len < 200 and (BOT.search(title) or BOT.search(stats.visible_head) or _CHALLENGE_RAW.search(raw))))
    redirected_home = original_path not in ('', '/') and final_path in ('', '/')
    home_url = bool(final) and final_path in ('', '/')
    generic_title = bool(re.fullmatch(r'(?:home|homepage|trang chủ|首页|网站首页|39健康网)(?:\s*[-|–].*)?', title, re.I))
    redirected_listing = final != url and original_path != final_path and bool(re.fullmatch(r'/(?:index\.(?:html?|php)|(?:news|category|tag|search|question|diseases?)/?)', final_path, re.I))
    empty_spa = text_len < 200 and visible_len < 300 and bool(_SPA.search(raw))
    if robots in ('DISALLOWED', 'UNAVAILABLE'):
        return 'blocked_4xx', 'robots_disallowed' if robots == 'DISALLOWED' else 'robots_unavailable'
    if error and code is None:
        return 'network_error', str(error)
    if cookie and small and code is not None and 200 <= code < 400:
        return 'cookie_challenge', 'literal_document_cookie_and_short_body'
    if challenge:
        return 'bot_challenge', 'challenge_signature_in_short_response'
    if code is not None and code >= 500:
        return 'dead_origin', f'http_{code}'
    if code in (404, 410) or home_url or redirected_home or redirected_listing or generic_title or (small and SOFT404.search(title or stats.visible_head)):
        return 'soft404_or_home', 'http_not_found' if code in (404, 410) else 'home_listing_or_soft404'
    if code is not None and 400 <= code < 500:
        return 'blocked_4xx', f'http_{code}'
    if error:
        return 'network_error', str(error)
    if code is not None and 200 <= code < 300 and empty_spa:
        return 'thin', 'empty_spa_shell'
    if code is not None and 200 <= code < 300:
        return ('ok', 'text_ge_200') if text_len >= 200 else ('thin', 'short_or_empty_content')
    return 'thin', f'unexpected_http_status_{code}'


def is_transient(state: str, http_status) -> bool:
    if state == 'blocked_4xx':
        return http_status in RETRYABLE_4XX
    return state in TRANSIENT
