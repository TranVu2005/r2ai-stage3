"""Download a few real pages per vi domain into tests/fixtures/ (extractor test inputs). 1 req/s/domain, robots not needed for 2 pages."""
from r2ai.paths import CORPUS_FILE, FIXTURES_DIR, auxiliary_disabled

import hashlib, random, re, sys, time, pathlib
import httpx, polars as pl
if __name__ == '__main__':
    auxiliary_disabled()
    from vicrawl.fetch import CHROME_UA
    from vicrawl.classify import find_cookie_challenge
    from vicrawl.domains import _domain_expr

    OUT = FIXTURES_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    DOMAINS = sys.argv[1].split(',')
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    extra = sys.argv[3:]  # explicit urls
    df = pl.read_parquet(CORPUS_FILE).with_columns(_domain_expr())
    c = httpx.Client(http2=True, headers={'User-Agent': CHROME_UA, 'Accept-Language': 'vi-VN,vi;q=0.9'}, follow_redirects=True, timeout=30)

    def get(url):
        for _ in range(2):
            r = c.get(url)
            ch = find_cookie_challenge(r.text) if r.status_code == 200 and len(r.text) < 8000 else None
            if ch:
                c.cookies.set(ch[0], ch[1], domain=httpx.URL(url).host, path='/')
                continue
            return r
        return r

    todo = []
    for d in DOMAINS:
        urls = df.filter(pl.col('domain') == d)['url'].to_list()
        random.Random(1).shuffle(urls)
        todo += [(d, u) for u in urls[:N]]
    for u in extra:
        todo.append((re.sub(r'^https?://(www\.)?', '', u).split('/')[0], u))
    for d, u in todo:
        try:
            r = get(u)
        except Exception as e:
            print('ERR', u, e); continue
        name = f"{d}__{hashlib.md5(u.encode()).hexdigest()[:8]}.html"
        (OUT / name).write_bytes(r.content)
        print(r.status_code, len(r.content), name, u)
        time.sleep(1.0)
