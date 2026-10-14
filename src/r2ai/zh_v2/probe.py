"""robots.txt + 5-URL probe per new v2 zh host (S11 D13 item 3a). One request per second overall, sequential.

Per host: GET <scheme>://<host>/robots.txt, interpret it with the crawler's policy (vicrawl.robots.interpret_robots, same UA),
then fetch up to 5 random v2 URLs (seed 42) of that host that robots allows. Never retries a challenge/403 and never
bypasses anything; a host whose robots disallows its sampled URLs gets no page request. Pages are kept as raw HTML under
out/runs/zh-v2/probe/html/<host>/ for extractor work. Writes out/runs/zh-v2/probe/{results.json,table.md}.

    python -m r2ai.zh_v2.probe [--hosts a b ...] [--n 5]
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, WORK_DATA_DIR, assert_writable
from vicrawl.fetch import CHROME_UA
from vicrawl.robots import interpret_robots

import argparse
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import polars as pl

OUT = RUNS_DIR / 'zh-v2' / 'probe'
IDS = WORK_DATA_DIR / 'corpus_v2_ids.parquet'
FIRST = ('ask.39.net', 'familydoctor.com.cn', 'zysjonline.com', 'baby.39.net')
CRAWLED = {'120ask.com', 'cnkang.com', 'zhongyibaodian.net', 'a-hospital.com', 'youlai.cn', 'test.pmphai.com', 'jbk.39.net',
           'care.39.net', 'msdmanuals.cn', 'wujue.com', 'jb39.com', 'zydcd.com', 'health.people.com.cn', 'qihuangzhishu.com'}
MAX_BODY = 2_000_000
INTERVAL = 1.0
CHALLENGE = re.compile(r'just a moment|cf-chl|captcha|验证码|安全验证|访问验证|are you a robot|access denied|waf', re.I)
QA_CLASS = re.compile(r'(?:ask|answer|question|reply|doctor|wenda|qa)', re.I)
CJK = re.compile(r'[一-鿿]')


def guard(path) -> Path:
    p = assert_writable(path)
    if not p.is_relative_to(assert_writable(OUT)):
        raise ValueError(f'Not a probe output: {p}')
    return p


def new_hosts(ids_path=IDS) -> list[str]:
    ids = pl.read_parquet(ids_path, columns=['host', 'lang']).filter(pl.col('lang') == 'zh')
    counts = ids.group_by('host').len().sort('len', descending=True)
    hosts = [h for h in counts['host'].to_list() if h not in CRAWLED]
    return [h for h in FIRST if h in hosts] + [h for h in hosts if h not in FIRST]


class Pacer:
    def __init__(self, interval=INTERVAL):
        self.interval, self.last = interval, 0.0

    def wait(self):
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def get(client, pacer, url, max_hops=3):
    """(status, headers, body bytes, final url, error). Redirects followed by hand, every hop paced."""
    for _ in range(max_hops + 1):
        pacer.wait()
        try:
            with client.stream('GET', url) as r:
                body = b''
                for chunk in r.iter_bytes():
                    body += chunk
                    if len(body) >= MAX_BODY:
                        break
                headers = {k.lower(): v for k, v in r.headers.items()}
                if r.status_code in (301, 302, 303, 307, 308) and headers.get('location'):
                    url = str(httpx.URL(url).join(headers['location']))
                    continue
                return r.status_code, headers, body, url, ''
        except (httpx.HTTPError, OSError) as e:
            return None, {}, b'', url, f'{type(e).__name__}: {str(e)[:160]}'
    return None, {}, b'', url, 'too many redirects'


def analyse(html: bytes, headers: dict) -> dict:
    from vicrawl.extractors.base import parse_html
    text = html.decode(headers_charset(headers, html), errors='replace')
    tree = parse_html(text)
    scripts = len(tree.xpath('//script'))
    for bad in tree.xpath('//script|//style|//noscript'):
        bad.drop_tree()
    visible = ' '.join(tree.text_content().split())
    cjk = len(CJK.findall(visible))
    qa_nodes = [n for n in tree.xpath('//*[@class or @id]') if QA_CLASS.search((n.get('class') or '') + ' ' + (n.get('id') or ''))]
    title = ' '.join((tree.xpath('string(//title)') or '').split())
    return {'title': title[:80], 'visible_chars': len(visible), 'cjk_chars': cjk, 'scripts': scripts,
            'qa_class_nodes': len(qa_nodes), 'has_ask_answer_words': bool(re.search(r'提问|问题|回答|解答|答', visible)),
            'challenge': bool(CHALLENGE.search(text[:4000]))}


def headers_charset(headers, body: bytes) -> str:
    m = re.search(r'charset=([\w-]+)', headers.get('content-type', ''), re.I) or re.search(rb'charset=["\']?([\w-]+)', body[:2000], re.I)
    name = (m.group(1).decode() if isinstance(m.group(1), bytes) else m.group(1)) if m else 'utf-8'
    try:
        ''.encode(name)
        return name
    except LookupError:
        return 'utf-8'


def classify(pages: list[dict]) -> str:
    ok = [p for p in pages if p.get('status') == 200]
    if not ok:
        return 'no page fetched'
    if any(p['analysis']['challenge'] for p in ok):
        return 'challenge/anti-bot page'
    static = [p for p in ok if p['analysis']['cjk_chars'] >= 200]
    if len(static) >= max(1, len(ok) // 2 + len(ok) % 2):
        return 'static HTML'
    return 'JS render likely (little CJK text in HTML)'


def probe_host(client, pacer, host, urls: list[str], n: int, seed=42) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(urls, min(n, len(urls)))
    first = sample[0]
    robots_url = f'{urlsplit(first).scheme}://{urlsplit(first).netloc}/robots.txt'
    status, headers, body, final, err = get(client, pacer, robots_url)
    text = body.decode(headers_charset(headers, body), errors='replace') if body else ''
    robots = interpret_robots(status, text, CHROME_UA, error=err)
    res = {'host': host, 'v2_urls': len(urls), 'robots_url': robots_url, 'robots_status': status, 'robots_error': err,
           'robots_state': robots.state, 'crawl_delay': robots.crawl_delay,
           'robots_sample': text[:600] if status == 200 else '', 'sampled': sample, 'pages': []}
    out = guard(OUT / 'html' / host)
    out.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(sample):
        allowed = robots.can_fetch(url)
        page = {'url': url, 'robots_allows': allowed}
        if allowed:
            st, h, b, final, e = get(client, pacer, url)
            page.update({'status': st, 'final_url': final, 'error': e, 'bytes': len(b), 'content_type': h.get('content-type', '')})
            if b and st == 200:
                (out / f'{i}.html').write_bytes(b)
                page['analysis'] = analyse(b, h)
        res['pages'].append(page)
    allowed = [p['robots_allows'] for p in res['pages']]
    res['robots_verdict'] = ('blocked' if not any(allowed) else 'allowed' if all(allowed) else 'partly')
    res['fetch_mode'] = classify(res['pages']) if any(allowed) else 'not probed (robots)'
    return res


def table(results: list[dict]) -> str:
    lines = ['| host | v2 URL | robots | crawl-delay | cách lấy | CJK ký tự (trung vị) | hỏi-đáp | ghi chú |', '|---|---:|---|---|---|---:|---|---|']
    for r in results:
        ok = [p for p in r['pages'] if p.get('analysis')]
        cjk = sorted(p['analysis']['cjk_chars'] for p in ok)
        qa = sum(p['analysis']['qa_class_nodes'] > 0 and p['analysis']['has_ask_answer_words'] for p in ok)
        robots = f"{r['robots_verdict']} ({r['robots_status'] if r['robots_status'] else r['robots_error'][:30]}, {r['robots_state']})"
        note = ', '.join(sorted({str(p.get('status')) for p in r['pages'] if p.get('robots_allows')})) or ''
        lines.append(f"| {r['host']} | {r['v2_urls']} | {robots} | {r['crawl_delay'] or ''} | {r['fetch_mode']} | "
                     f"{cjk[len(cjk) // 2] if cjk else ''} | {qa}/{len(ok)} | HTTP {note} |")
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hosts', nargs='*')
    ap.add_argument('--n', type=int, default=5)
    a = ap.parse_args(argv)
    ids = pl.read_parquet(IDS, columns=['host', 'url', 'lang']).filter(pl.col('lang') == 'zh')
    hosts = a.hosts or new_hosts()
    by_host = {h: sorted(ids.filter(pl.col('host') == h)['url'].to_list()) for h in hosts}
    pacer = Pacer()
    results = []
    headers = {'User-Agent': CHROME_UA, 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5',
               'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5'}
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=10.0), headers=headers, follow_redirects=False) as client:
        for host in hosts:
            res = probe_host(client, pacer, host, by_host[host], a.n)
            results.append(res)
            print(json.dumps({k: res[k] for k in ('host', 'robots_status', 'robots_state', 'robots_verdict', 'fetch_mode')}, ensure_ascii=False), flush=True)
            guard(OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    guard(OUT / 'table.md').write_text(table(results), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
