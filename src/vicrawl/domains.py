"""Select Vietnamese domains from the links corpus and build the de-duplicated URL table."""
from __future__ import annotations

from r2ai.paths import ROOT

import csv
import hashlib
import re
import time
from pathlib import Path

import polars as pl

from .urlnorm import group_urls, normalize_url

DEFERRED = {'nhathuoclongchau.com.vn': 'direct crawl 0% in probe (blocked); handle via Common Crawl in a later stage'}
EXTRA_VI = {'vinmec.com', 'hellobacsi.com'}
CJK = re.compile(r'[一-鿿]')


def _domain_expr():
    return pl.col('url').str.extract(r'^https?://([^/?#]+)', 1).str.to_lowercase().str.replace(r'^www\.', '').alias('domain')


def classify_domains(corpus_path: Path, *, probed_lang: dict, extra_vi=EXTRA_VI, probe_fn=None, probe_n: int = 3) -> list[dict]:
    """probed_lang: domain -> main_lang from earlier probes. probe_fn(domain, urls)->lang|None for unprobed long tail."""
    df = pl.read_parquet(corpus_path, columns=['id', 'url']).with_columns(_domain_expr())
    out = []
    for domain, sub in sorted(df.group_by('domain'), key=lambda kv: -kv[1].height):
        domain = domain[0]
        urls = sub['url'].to_list()
        n_unique = len({normalize_url(u) for u in urls})
        lang, is_vi, reason = None, False, ''
        if domain.endswith('.vn'):
            is_vi, reason = True, 'tld .vn'
            if probed_lang.get(domain) not in (None, '', 'vi'):
                reason += f' (probe saw {probed_lang[domain]}: language label unreliable, TLD rule wins)'
        elif domain in extra_vi:
            is_vi, reason = True, 'known vi domain without .vn'
        elif probed_lang.get(domain):
            is_vi = probed_lang[domain] == 'vi'
            reason = f'probe main_lang={probed_lang[domain]}'
        elif probe_fn is not None:
            sample = sorted(urls)[:probe_n] if len(urls) <= probe_n else [urls[i * len(urls) // probe_n] for i in range(probe_n)]
            lang = probe_fn(domain, sample)
            if lang:
                is_vi, reason = lang == 'vi', f'fetched {len(sample)} urls, detected lang={lang}'
            else:
                reason = 'probe_failed: could not fetch/detect language, excluded'
        else:
            reason = 'probe_failed: no probe available, excluded'
        out.append({'domain': domain, 'n_rows': len(urls), 'n_unique': n_unique, 'is_vi': is_vi, 'reason': reason, 'deferred': is_vi and domain in DEFERRED})
    return out


def write_domain_files(results: list[dict], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vi = [r for r in results if r['is_vi'] and not r['deferred']]
    with open(out_dir / 'vi_domains.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'n_rows', 'n_unique', 'reason'], extrasaction='ignore')
        w.writeheader()
        w.writerows(vi)
    with open(out_dir / 'deferred_domains.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'n_rows', 'n_unique', 'reason'])
        for r in results:
            if r['deferred']:
                w.writerow([r['domain'], r['n_rows'], r['n_unique'], DEFERRED[r['domain']]])
    with open(out_dir / 'domain_classification_all.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'n_rows', 'n_unique', 'is_vi', 'deferred', 'reason'])
        w.writeheader()
        w.writerows(results)


def build_url_table(corpus_path: Path, domains: set[str]) -> list[dict]:
    """Group duplicate urls (url_norm), one representative each, deterministic pseudo-random rank per domain
    so that a partial crawl is a uniform sample of the domain instead of its first-listed pages."""
    df = pl.read_parquet(corpus_path, columns=['id', 'url']).with_columns(_domain_expr()).filter(pl.col('domain').is_in(sorted(domains)))
    groups = group_urls(zip(df['id'].to_list(), df['url'].to_list()))
    by_domain: dict[str, list[dict]] = {}
    for g in groups:
        by_domain.setdefault(g['domain'], []).append(g)
    out = []
    for d, items in by_domain.items():
        items.sort(key=lambda g: hashlib.md5(g['url_norm'].encode()).digest())
        for i, g in enumerate(items):
            g['rank'] = i
            out.append(g)
    return out


def load_probed_lang(out_dir: Path) -> dict:
    p = Path(out_dir) / 'crawl_domain_summary.csv'
    if not p.exists():
        return {}
    with open(p, encoding='utf-8-sig', newline='') as f:
        return {r['domain']: r['main_lang'] for r in csv.DictReader(f) if r.get('main_lang')}


def read_vi_domains(path: Path) -> set[str]:
    with open(path, encoding='utf-8-sig', newline='') as f:
        return {r['domain'] for r in csv.DictReader(f)}


def live_probe(domain: str, urls: list[str]):
    """Fetch up to 3 urls politely (1 req/s), return 'vi' / 'zh' / 'en' / other langdetect code, or None."""
    import httpx
    from langdetect import DetectorFactory, detect
    from lxml import html as lhtml
    DetectorFactory.seed = 0
    votes = []
    with httpx.Client(timeout=20, follow_redirects=True, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36', 'Accept-Language': 'vi,en;q=0.8'}) as c:
        for u in urls:
            try:
                r = c.get(u)
                if r.status_code != 200 or not r.text:
                    continue
                tree = lhtml.fromstring(r.text)
                for n in tree.xpath('//script|//style'):
                    n.drop_tree()
                text = re.sub(r'\s+', ' ', tree.text_content())[:4000]
                if len(text) < 80:
                    continue
                votes.append('zh' if len(CJK.findall(text)) > 0.2 * len(text) else detect(text))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
    if not votes:
        return None
    return max(set(votes), key=votes.count)
