"""S26 vi pool control: same-site article links found in raw HTML of corpus pages, labelled in/out of corpus v2.

vi v2 has no dropped set (vi v2 = vi v1), so the control is the site's own articles that BTC did not list. For each host,
read raw pages (data/raw_vi/<host>/*.jsonl.zst) until --pages-per-host, extract <a href> with title-like anchor text
(>= 6 words, >= 25 chars), keep same-host links, normalise (scheme/www/query/fragment/trailing slash dropped) and label
in_corpus against the normalised URL set of links_corpus_vi_en_zh.parquet. One row per URL (longest anchor).
Output: <out>/vi_links.parquet (host, url_norm, anchor, in_corpus, section, n_seen).
"""
import glob, html, io, json, os, re, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urljoin, urlparse
import pandas as pd, zstandard as zs
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

A = re.compile(r'<a\s[^>]*?href\s*=\s*["\']([^"\'#]+)["\'][^>]*>(.*?)</a>', re.S | re.I)
TAG = re.compile(r'<[^>]+>')
PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 1500


def norm(u):
    p = urlparse(u.strip())
    host = p.netloc.lower().split(':')[0].removeprefix('www.')
    return host + p.path.rstrip('/')


def section(path):
    seg = [s for s in path.split('/') if s]
    return seg[0] if len(seg) > 1 else '(root)'


def work(host):
    files = sorted(glob.glob(f'data/raw_vi/{host}/*.jsonl.zst'))
    seen = defaultdict(lambda: ['', 0]); pages = 0
    for f in files:
        with open(f, 'rb') as fh:
            for line in io.TextIOWrapper(zs.ZstdDecompressor().stream_reader(fh), encoding='utf-8', errors='replace'):
                r = json.loads(line)
                if r.get('status') != 'ok' or not r.get('html'):
                    continue
                base = r.get('final_url') or r['url']
                for href, text in A.findall(r['html']):
                    t = re.sub(r'\s+', ' ', html.unescape(TAG.sub(' ', text))).strip()
                    if len(t) < 25 or len(t.split()) < 6:
                        continue
                    u = urljoin(base, html.unescape(href))
                    if not u.startswith('http'):
                        continue
                    n = norm(u)
                    if not n.startswith(host.removeprefix('www.') + '/') or n.count('/') < 1:
                        continue
                    s = seen[n]
                    if len(t) > len(s[0]):
                        s[0] = t
                    s[1] += 1
                pages += 1
                if pages >= PAGES:
                    break
        if pages >= PAGES:
            break
    return host, pages, [(host, n, t, c) for n, (t, c) in seen.items()]


if __name__ == '__main__':
    out = Path(sys.argv[1])
    v2 = pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['url'])
    corpus = set(v2.url.map(norm))
    hosts = sorted(os.listdir('data/raw_vi'))
    rows = []
    with Pool(8) as p:
        for host, pages, r in p.imap_unordered(work, hosts):
            rows += r; print(host, pages, len(r), flush=True)
    df = pd.DataFrame(rows, columns=['host', 'url_norm', 'anchor', 'n_seen'])
    df['in_corpus'] = df.url_norm.isin(corpus)
    df['section'] = df.url_norm.map(lambda n: section(n.split('/', 1)[1] if '/' in n else ''))
    df.to_parquet(out / 'vi_links.parquet', index=False)
    print(df.groupby('host').in_corpus.agg(['size', 'sum', 'mean']).sort_values('size', ascending=False).head(30).to_string())
