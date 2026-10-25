"""S26 extractor fingerprint, step 1: rebuild doc_text of the zh 2% sample (29,385 docs) from raw HTML with several
off-the-shelf extractors. Run with an interpreter that has trafilatura, bs4, readability-lxml, newspaper4k, justext,
html2text (scratch venv with --system-site-packages; the repo environment is not changed).

Raw: data/raw_zh_sample/<domain>/*.jsonl.zst, latest fetch per url_norm (same rule as r2ai.zh_sample.extract), records
with status ok/thin and HTML. A sample doc (data/chunks_zh_sample/docs.parquet) takes the record whose doc_ids meet its
doc_ids_group. Output <out>/ext_texts.parquet: doc_id + one text column per variant (None when the extractor fails).
Usage: ext_variants.py <out> [workers=6]
"""
import io, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3')
VARIANTS = ['traf_default', 'traf_recall', 'traf_nocomments', 'traf_recall_nocomments', 'bs_page', 'bs_body',
            'readability', 'newspaper', 'justext', 'html2text']


def clean(t):
    """collapse whitespace inside lines, drop empty lines; paragraphs joined by a blank line"""
    if t is None:
        return None
    lines = [re.sub(r'[ \t\u3000\xa0]+', ' ', x).strip() for x in t.splitlines()]
    return '\n\n'.join(x for x in lines if x)


def extract_all(html, url):
    import trafilatura, html2text, justext, lxml.html
    from bs4 import BeautifulSoup
    from readability import Document
    out = {}

    def run(name, fn):
        try:
            out[name] = clean(fn())
        except Exception:
            out[name] = None
    run('traf_default', lambda: trafilatura.extract(html, url=url))
    run('traf_recall', lambda: trafilatura.extract(html, url=url, favor_recall=True))
    run('traf_nocomments', lambda: trafilatura.extract(html, url=url, include_comments=False))
    run('traf_recall_nocomments', lambda: trafilatura.extract(html, url=url, favor_recall=True, include_comments=False))

    def bs(body_only):
        soup = BeautifulSoup(html, 'lxml')
        for t in soup(['script', 'style', 'noscript', 'template']):
            t.decompose()
        node = soup.body if body_only and soup.body is not None else soup
        return node.get_text('\n', strip=True)
    run('bs_page', lambda: bs(False))
    run('bs_body', lambda: bs(True))

    def rd():
        d = Document(html)
        return '\n'.join(x.strip() for x in lxml.html.fromstring(d.summary(html_partial=True)).itertext())
    run('readability', rd)

    def np_():
        from newspaper import Article
        a = Article(url, language='zh', fetch_images=False)
        a.download(input_html=html); a.parse()
        return a.text
    run('newspaper', np_)
    # jusText has no Chinese stoplist: empty stoplist, stopword thresholds 0, length thresholds scaled to CJK (10/50 chars)
    run('justext', lambda: '\n'.join(p.text for p in justext.justext(html.encode('utf-8'), frozenset(), length_low=10,
                                      length_high=50, stopwords_low=0, stopwords_high=0, max_link_density=0.2)
                                      if not p.is_boilerplate))

    def h2t():
        h = html2text.HTML2Text(); h.ignore_links = True; h.ignore_images = True; h.body_width = 0
        return h.handle(html)
    run('html2text', h2t)
    return out


def shard_rows(path):
    import zstandard
    rows = []
    with open(path, 'rb') as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding='utf-8'):
            r = json.loads(line)
            if r.get('status') not in ('ok', 'thin') or not r.get('html'):
                continue
            rows.append({'url_norm': r['url_norm'], 'fetched_at': r['fetched_at'], 'doc_ids': r['doc_ids'],
                         **extract_all(r['html'], r.get('final_url') or r['url'])})
    return rows


if __name__ == '__main__':
    import pandas as pd
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    shards = sorted(Path(ROOT, 'data/raw_zh_sample').glob('*/*.jsonl.zst'))
    part = out / 'ext_parts'; part.mkdir(exist_ok=True)
    todo = [s for s in shards if not (part / (s.parent.name + '__' + s.stem + '.parquet')).exists()]
    print(f'shards {len(shards)} todo {len(todo)}', flush=True); t0 = time.time()
    with ProcessPoolExecutor(W) as ex:
        for k, (s, rows) in enumerate(zip(todo, ex.map(shard_rows, todo))):
            pd.DataFrame(rows, columns=['url_norm', 'fetched_at', 'doc_ids', *VARIANTS]).to_parquet(
                part / (s.parent.name + '__' + s.stem + '.parquet'), index=False)
            if k % 20 == 0:
                print(f'{k + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    recs = pd.concat([pd.read_parquet(f) for f in sorted(part.glob('*.parquet'))], ignore_index=True)
    recs = recs.sort_values('fetched_at').drop_duplicates('url_norm', keep='last')
    docs = pd.read_parquet(Path(ROOT, 'data/chunks_zh_sample/docs.parquet'), columns=['doc_id', 'doc_ids_group'])
    cid = {int(c): i for i, ids in zip(recs.index, recs.doc_ids) for c in ids}
    hit = [next((cid[int(c)] for c in g if int(c) in cid), None) for g in docs.doc_ids_group]
    res = docs[['doc_id']].copy()
    for v in VARIANTS:
        res[v] = [recs.at[i, v] if i is not None else None for i in hit]
    res.to_parquet(out / 'ext_texts.parquet', index=False)
    print(json.dumps({'docs': len(docs), 'matched': sum(h is not None for h in hit), 'records': len(recs),
                      'nonempty': {v: int(res[v].fillna('').str.len().gt(0).sum()) for v in VARIANTS},
                      'median_chars': {v: float(res[v].fillna('').str.len().median()) for v in VARIANTS},
                      'seconds': round(time.time() - t0)}, indent=1), flush=True)
