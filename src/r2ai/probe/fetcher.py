"""Reusable polite HTTP fetcher and exclusive offline response classification."""
from __future__ import annotations

from r2ai.paths import ROOT

import hashlib
import json
import re
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from trafilatura.utils import decode_file
from lxml import html as lhtml

CHROME_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
OLD_UA = 'R2AI-Stage3-SampleProbe/0.1 (academic competition research; ~300 URL sample; honors robots.txt)'
STATES = ('ok', 'thin', 'cookie_challenge', 'bot_challenge', 'soft404_or_home', 'blocked_4xx', 'dead_origin', 'needs_js_real', 'network_error', 'other')
COOKIE = re.compile(r'document\s*\.\s*cookie\s*=\s*["\'](\w+)=(\w+)["\']', re.I)
BOT = re.compile(r'just a moment|attention required|access denied|verify you are human|are you a robot|验证码|滑动验证|安全验证|请完成验证|访问频繁|captcha', re.I)
SOFT404 = re.compile(r'^(?:404(?:\s|$)|page not found|not found|trang không tồn tại|không tìm thấy trang|页面不存在|页面未找到)', re.I)


def extract(body: bytes) -> tuple[str, str, str]:
    """Return extracted content, title and visible text (never execute scripts)."""
    if not body:
        return '', '', ''
    if body.startswith(b'%PDF'):
        try:
            import fitz
            with fitz.open(stream=body, filetype='pdf') as doc:
                return '\n'.join(p.get_text() for p in doc).strip(), '', ''
        except Exception:
            return '', '', ''
    try:
        decoded = decode_file(body)
        text = (trafilatura.extract(decoded, favor_recall=True, include_comments=False, include_tables=True) or '').strip()
    except Exception:
        text = ''
    try:
        tree = lhtml.fromstring(re.sub(r'^\s*<\?xml[^>]*>', '', decode_file(body)))
        title = ' '.join(tree.xpath('//title/text()')).strip()
        for node in tree.xpath('//script|//style|//noscript|//template'):
            node.drop_tree()
        visible = re.sub(r'\s+', ' ', tree.text_content()).strip()
    except Exception:
        title, visible = '', ''
    return text, title, visible


def classify(rec: dict, body: bytes = b'') -> dict:
    """One and only one state; `reason` makes exceptional outcomes auditable."""
    rec = dict(rec)
    raw = body.decode(rec.get('encoding') or 'utf-8', errors='replace')
    text, title, visible = extract(body)
    code = rec.get('http_status', rec.get('status'))
    code = int(code) if code not in (None, '') else None
    final = rec.get('final_url') or rec.get('url', '')
    original_path = urlsplit(rec.get('url', '')).path
    final_path = urlsplit(final).path
    match = COOKIE.search(raw)
    if match:
        rec['cookie_name'] = match[1]
        rec['cookie_value_hash'] = hashlib.sha256(match[2].encode()).hexdigest()[:16]
    small = len(text) < 2000 and len(visible) < 4000
    challenge = small and (re.search(r'^(?:just a moment|attention required|access denied)', title, re.I) or (len(text) < 200 and (BOT.search(title) or BOT.search(visible) or re.search(r'cf-chl|cf_chl|/cdn-cgi/challenge-platform|g-recaptcha|hcaptcha', raw, re.I))))
    redirected_home = original_path not in ('', '/') and final_path in ('', '/')
    home_url = bool(final) and final_path in ('', '/')
    generic_title = bool(re.fullmatch(r'(?:home|homepage|trang chủ|首页|网站首页|39健康网)(?:\s*[-|–].*)?', title, re.I))
    redirected_listing = final != rec.get('url') and original_path != final_path and bool(re.fullmatch(r'/(?:index\.(?:html?|php)|(?:news|category|tag|search|question|diseases?)/?)', final_path, re.I))
    empty_spa = len(text) < 200 and len(visible) < 300 and bool(re.search(r'__NEXT_DATA__|<[^>]+id\s*=\s*["\'](?:app|root|__next)["\']|<app-root\b', raw, re.I))
    if rec.get('robots') in ('DISALLOWED', 'UNAVAILABLE'):
        state, reason = 'blocked_4xx', 'robots_disallowed' if rec['robots'] == 'DISALLOWED' else 'robots_unavailable'
    elif rec.get('error') and code is None:
        state, reason = 'network_error', str(rec['error'])
    elif match and small and code is not None and 200 <= code < 400:
        state, reason = 'cookie_challenge', 'literal_document_cookie_and_short_body'
    elif challenge:
        state, reason = 'bot_challenge', 'challenge_signature_in_short_response'
    elif code is not None and code >= 500:
        state, reason = 'dead_origin', f'http_{code}'
    elif code in (404, 410) or home_url or redirected_home or redirected_listing or generic_title or (small and SOFT404.search(title or visible)):
        state, reason = 'soft404_or_home', 'http_not_found' if code in (404, 410) else 'home_listing_or_soft404'
    elif code is not None and 400 <= code < 500:
        state, reason = 'blocked_4xx', f'http_{code}'
    elif rec.get('error'):
        state, reason = 'network_error', str(rec['error'])
    elif code is not None and 200 <= code < 300 and empty_spa:
        state, reason = 'needs_js_real', 'empty_spa_shell'
    elif code is not None and 200 <= code < 300:
        state = 'ok' if len(text) >= 200 else 'thin'
        reason = 'extracted_text_ge_200' if state == 'ok' else 'short_or_empty_content_without_spa_evidence'
    else:
        state, reason = 'other', f'unexpected_http_status_{code}'
    rec.update(state=state, reason=reason, title=title, body_prefix=raw[:300], bytes=len(body) if body else rec.get('bytes', 0), text_len=len(text), visible_chars=len(visible), http_status=code)
    assert rec['state'] in STATES and bool(rec['reason'])
    return rec


class Pacer:
    """Shared host limiter covers robots, redirects, cookie retries and UA A/B."""
    def __init__(self):
        self.lock = threading.Lock()
        self.last: dict[str, float] = {}
        self.delays: dict[str, float] = {}

    def wait(self, url: str):
        host = (urlsplit(url).hostname or '').lower().removeprefix('www.')
        with self.lock:
            delay = max(1.0, self.delays.get(host, 1.0))
            sleep = self.last.get(host, 0) + delay - time.monotonic()
            # Release global lock while sleeping: unrelated domains stay parallel.
        while sleep > 0:
            time.sleep(sleep)
            with self.lock:
                sleep = self.last.get(host, 0) + delay - time.monotonic()
                if sleep <= 0:
                    self.last[host] = time.monotonic()
                    return
        with self.lock:
            # Re-check in case another thread acquired the slot meanwhile.
            sleep = self.last.get(host, 0) + delay - time.monotonic()
            if sleep <= 0:
                self.last[host] = time.monotonic()
                return
        self.wait(url)


class Fetcher:
    def __init__(self, ua: str = CHROME_UA, *, pacer: Pacer | None = None, timeout: float = 15, respect_robots: bool = True):
        self.ua, self.pacer, self.timeout = ua, pacer or Pacer(), timeout
        self.respect_robots = respect_robots
        self.sessions: dict[str, requests.Session] = {}
        self.robots: dict[str, tuple[RobotFileParser | None, str, float | None]] = {}
        self.robots_details: dict[str, dict] = {}
        self.cookie_values: dict[str, list[str]] = {}
        self.request_events: list[dict] = []
        self.lock = threading.RLock()

    def session(self, url: str) -> requests.Session:
        host = (urlsplit(url).hostname or '').lower().removeprefix('www.')
        with self.lock:
            if host not in self.sessions:
                s = requests.Session()
                s.headers.update({'User-Agent': self.ua, 'Accept-Language': 'vi,zh-CN;q=0.9,en;q=0.8', 'Accept': 'text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5'})
                self.sessions[host] = s
            return self.sessions[host]

    def request(self, url: str, *, headers: dict | None = None) -> tuple[requests.Response, bytes]:
        self.pacer.wait(url)
        t0 = time.monotonic()
        event = {'url': url, 'ua': self.ua, 'started_monotonic': t0}
        try:
            r = self.session(url).get(url, timeout=self.timeout, allow_redirects=False, stream=True, headers=headers)
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size >= 5_000_000:
                    break
                if time.monotonic() - t0 > self.timeout:
                    raise requests.Timeout('total body timeout')
            body = b''.join(chunks)[:5_000_000]
            r.close()
            event['http_status'] = r.status_code
            return r, body
        except requests.RequestException as e:
            event['error'] = f'{type(e).__name__}: {str(e)[:300]}'
            raise
        finally:
            event['elapsed_s'] = round(time.monotonic() - t0, 4)
            self.request_events.append(event)

    def allowed(self, url: str) -> tuple[bool, str, float | None]:
        if not self.respect_robots:
            return True, 'disabled_for_test', None
        u = urlsplit(url)
        origin = f'{u.scheme}://{u.netloc}'
        if origin not in self.robots:
            try:
                robots_url = origin + '/robots.txt'
                seen = set()
                robot_cookie_attempted = False
                for _ in range(6):
                    if robots_url in seen:
                        raise requests.TooManyRedirects('robots redirect loop')
                    seen.add(robots_url)
                    r, body = self.request(robots_url)
                    match = COOKIE.search(body.decode('utf-8', errors='replace')) if r.status_code == 200 and len(body) < 3000 else None
                    if match and not robot_cookie_attempted:
                        name, value = match.groups()
                        self.session(robots_url).cookies.set(name, value, domain=urlsplit(robots_url).hostname, path='/')
                        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
                        self.cookie_values.setdefault(urlsplit(robots_url).hostname or '', []).append(digest)
                        robot_cookie_attempted = True
                        seen.remove(robots_url)
                        continue
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get('Location'):
                        robots_url = urljoin(robots_url, r.headers['Location'])
                        if urlsplit(robots_url).scheme not in ('http', 'https'):
                            raise requests.InvalidURL('invalid robots redirect')
                        continue
                    break
                else:
                    raise requests.TooManyRedirects('robots redirect limit')
                rp, delay = None, None
                robot_state = classify({'url': robots_url, 'http_status': r.status_code}, body)['state'] if r.status_code == 200 else ''
                if r.status_code == 200 and robot_state not in ('cookie_challenge', 'bot_challenge'):
                    rp = RobotFileParser()
                    rp.parse(body.decode('utf-8-sig', errors='replace').splitlines())
                    delay = rp.crawl_delay(self.ua) or rp.crawl_delay('*')
                    # robotparser ignores decimals; inspect ONLY its matching group.
                    # Other bots' delays (e.g. Yahoo=1000s) must not apply to Chrome.
                    agent = self.ua.split('/')[0].lower()
                    agents, rules, entries = [], [], []
                    for line in body.decode('utf-8-sig', errors='replace').splitlines() + ['User-agent: __end__']:
                        line = line.split('#', 1)[0].strip()
                        if ':' not in line:
                            continue
                        name, value = [v.strip() for v in line.split(':', 1)]
                        if name.lower() == 'user-agent':
                            if rules:
                                entries.append((agents, rules))
                                agents, rules = [], []
                            agents.append(value.lower())
                        elif agents:
                            rules.append((name.lower(), value))
                    matched = next((rules for agents, rules in entries if any(a != '*' and a in agent for a in agents)), None)
                    if matched is None:
                        matched = next((rules for agents, rules in entries if '*' in agents), [])
                    decimals = [float(v) for k, v in matched if k == 'crawl-delay' and re.fullmatch(r'\d+(?:\.\d+)?', v)]
                    if decimals:
                        delay = max(decimals)
                    rate = rp.request_rate(self.ua) or rp.request_rate('*')
                    if rate:
                        delay = max(delay or 0, rate.seconds / rate.requests)
                    state = 'no_robots_html_fallback' if re.search(rb'<(?:html|!doctype)', body[:500], re.I) else 'robots_ok'
                elif r.status_code in (404, 410):
                    state = 'no_robots'
                elif r.status_code in (401, 403):
                    state = 'DISALLOWED'
                else:
                    state = 'UNAVAILABLE'
                self.robots[origin] = rp, state, delay
                self.robots_details[origin] = {'robots_http_status': r.status_code, 'robots_server': r.headers.get('Server', ''), 'robots_body_prefix': body.decode('utf-8', errors='replace')[:300], 'robots_cookie_attempted': robot_cookie_attempted, 'robots_cookie_solved': robot_cookie_attempted and state in ('robots_ok', 'no_robots', 'no_robots_html_fallback'), 'robots_cookie_name': name if robot_cookie_attempted else None, 'robots_cookie_hash': digest if robot_cookie_attempted else None}
                if delay:
                    host = (u.hostname or '').removeprefix('www.')
                    self.pacer.delays[host] = max(self.pacer.delays.get(host, 1), delay)
            except requests.RequestException:
                self.robots[origin] = None, 'UNAVAILABLE', None
        rp, state, delay = self.robots[origin]
        return state not in ('DISALLOWED', 'UNAVAILABLE') and (rp is None or rp.can_fetch(self.ua, url)), state, delay

    def fetch(self, url: str, *, solve_cookie: bool = True) -> tuple[dict, bytes]:
        t0 = time.monotonic()
        rec = {'url': url, 'final_url': url, 'redirect_chain': [], 'server': '', 'set_cookie': '', 'http_status': None, 'error': '', 'cookie_solved': False, 'cookie_attempted': False, 'cookie_hashes': [], 'retry_count': 0, 'request_count': 0, 'crawl_delay': None}
        current, body, cookies, retries, redirects = url, b'', 0, 0, 0
        initial_state = None
        while True:
            allowed, robots, delay = self.allowed(current)
            origin = f'{urlsplit(current).scheme}://{urlsplit(current).netloc}'
            rec.update(self.robots_details.get(origin, {}))
            rec['robots'], rec['crawl_delay'] = robots if allowed else ('UNAVAILABLE' if robots == 'UNAVAILABLE' else 'DISALLOWED'), delay
            if not allowed:
                rec['error'] = 'robots_unavailable' if robots == 'UNAVAILABLE' else 'robots_disallowed'
                body = b''
                break
            try:
                r, body = self.request(current)
                rec['request_count'] += 1
                rec.update(http_status=r.status_code, final_url=r.url, server=r.headers.get('Server', ''), set_cookie=r.headers.get('Set-Cookie', ''), content_type=r.headers.get('Content-Type', ''), encoding=r.encoding if r.encoding and r.encoding.lower() != 'iso-8859-1' else 'utf-8')
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get('Location'):
                    target = urljoin(current, r.headers['Location'])
                    rec['redirect_chain'].append({'url': current, 'status': r.status_code, 'location': target})
                    redirects += 1
                    if redirects > 10 or urlsplit(target).scheme not in ('http', 'https'):
                        rec['error'] = 'redirect_limit_or_invalid_scheme'
                        break
                    current = target
                    continue
                analyzed = classify(rec, body)
                initial_state = initial_state or analyzed['state']
                match = COOKIE.search(body.decode('utf-8', errors='replace')) if analyzed['state'] == 'cookie_challenge' else None
                if match:
                    name, value = match.groups()
                    rec['cookie_name'] = name
                    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
                    rec['cookie_hashes'].append(digest)
                    self.cookie_values.setdefault(urlsplit(current).hostname or '', []).append(digest)
                    if solve_cookie and cookies < 1:
                        self.session(current).cookies.set(name, value, domain=urlsplit(current).hostname, path='/')
                        cookies += 1
                        rec['cookie_attempted'] = True
                        continue
                if (r.status_code == 429 or r.status_code >= 500) and retries < 2:
                    delay_s = 2 ** retries
                    retry_after = r.headers.get('Retry-After', '')
                    try:
                        delay_s = max(delay_s, float(retry_after))
                    except ValueError:
                        try:
                            delay_s = max(delay_s, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (ValueError, TypeError):
                            pass
                    time.sleep(delay_s)
                    retries += 1
                    rec['retry_count'] = retries
                    continue
                break
            except requests.RequestException as e:
                rec['request_count'] += 1
                rec.update(error=f'{type(e).__name__}: {str(e)[:300]}', http_status=None, final_url=current)
                body = b''
                break
        result = classify(rec, body)
        result['initial_state'] = 'cookie_challenge' if rec.get('robots_cookie_attempted') else initial_state or result['state']
        result['cookie_solved'] = bool((rec['cookie_attempted'] or rec.get('robots_cookie_solved')) and result['state'] == 'ok')
        result['elapsed_s'] = round(time.monotonic() - t0, 4)
        host = urlsplit(current).hostname or ''
        values = self.cookie_values.get(host, [])
        result['cookie_stability'] = 'changes_observed' if len(set(values)) > 1 else ('same_value_observed' if len(values) >= 2 else 'insufficient_observations')
        return result, body
