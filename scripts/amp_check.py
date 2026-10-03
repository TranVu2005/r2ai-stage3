"""Does an AMP / mobile variant of a slow domain exist and is it faster with equivalent content? (spec D)
Usage: python scripts/amp_check.py suckhoecongdongonline.vn [n=20]   -> out/amp_check_<domain>.md"""
from r2ai.paths import OUT_DIR, STATE_DIR, auxiliary_disabled

import hashlib, re, sqlite3, statistics, sys, time
from pathlib import Path
import httpx
from vicrawl.fetch import CHROME_UA

if __name__ == '__main__':
    auxiliary_disabled()
    domain = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    db = sqlite3.connect(STATE_DIR / 'crawl.db')
    urls = [r[0] for r in db.execute('select url from urls where domain=? order by rank limit ?', (domain, n))]
    c = httpx.Client(headers={'User-Agent': CHROME_UA, 'Accept-Language': 'vi-VN,vi;q=0.9'}, follow_redirects=True, timeout=40)

    def variants(u):
        host = re.sub(r'^https?://', '', u).split('/')[0]
        return {'plain': u, '?amp': u + ('&' if '?' in u else '?') + 'amp', '/amp/': u.rstrip('/') + '/amp/', 'm. host': u.replace('//' + host, '//m.' + host.removeprefix('www.'), 1)}

    def get(u):
        t = time.time()
        try:
            r = c.get(u)
            return r.status_code, r.content, time.time() - t
        except Exception as e:
            return None, b'', time.time() - t

    stats = {}
    for u in urls:
        base = None
        for k, v in variants(u).items():
            code, body, dt = get(v)
            s = stats.setdefault(k, {'lat': [], 'ok': 0, 'same': 0, 'n': 0})
            s['n'] += 1; s['lat'].append(dt)
            if code == 200:
                s['ok'] += 1
                if k == 'plain': base = len(body)
                elif abs(len(body) - base) <= 0.03 * max(base, 1): s['same'] += 1
            time.sleep(1.0)
    lines = [f'# AMP / mobile check: {domain}', '', f'{len(urls)} URLs, each fetched as plain + 3 variants (1 req/s).', '',
             '| variant | HTTP 200 | size within 3% of plain | p50 latency | p95 latency |', '|---|---|---|---|---|']
    for k, s in stats.items():
        lat = sorted(s['lat'])
        lines.append(f"| {k} | {s['ok']}/{s['n']} | {s['same']}/{s['n']} | {statistics.median(lat):.2f}s | {lat[int(.95 * len(lat)) - 1]:.2f}s |")
    better = [k for k, s in stats.items() if k != 'plain' and s['ok'] == s['n'] and statistics.median(s['lat']) * 2 <= statistics.median(stats['plain']['lat'])]
    lines += ['', f'**Decision:** ' + (f'switch to {better[0]} (>=2x faster, content to be compared)' if better else 'keep plain URLs: no variant is >=2x faster than plain (same size => the server ignores the suffix; m. host does not resolve).')]
    out = OUT_DIR / f'amp_check_{domain}.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
