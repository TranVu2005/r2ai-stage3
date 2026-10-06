"""Step 1.3: slug meaningfulness per domain + 10 sample URLs (seed 42). Read-only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai.paths import CORPUS_FILE
from r2ai.selective.common import CJK, OUT, alpha_tokens, host_of, slug_text

VI_FOCUS = ['vov.vn', 'vov2.vov.vn', 'baolangson.vn', 'youmed.vn', 'thaythuocvietnam.vn', 'nhathuoclongchau.com.vn']


def kind(url):
    s = slug_text(url)
    if CJK.search(s):
        return 'cjk'
    return 'latin3' if len(alpha_tokens(url)) >= 3 else 'id_only'


def distinct_counts(urls):
    """Per-URL count of alpha tokens whose domain doc-frequency <=0.5% (drops dir/category words shared by many URLs)."""
    from collections import Counter
    toks = [set(alpha_tokens(u)) for u in urls]
    df = Counter(t for s in toks for t in s)
    cut = max(2, 0.005 * len(urls))
    return [sum(1 for t in s if df[t] <= cut) for s in toks]


def main():
    c = pq.read_table(CORPUS_FILE).to_pandas()
    g = pd.read_parquet(OUT / 'corpus_groups.parquet')
    c = c.merge(g[['id', 'host', 'grp']], on='id')
    zh = c[c.grp.isin(['zh_full', 'zh_sample', 'zh_not_in_full'])]
    sel = pd.concat([zh, c[c.host.isin(VI_FOCUS)]])
    rows, samples = [], {}
    for host, d in sel.groupby('host'):
        k = d.url.map(kind)
        vc = k.value_counts(normalize=True) * 100
        dc = np.array(distinct_counts(d.url.tolist()))
        rows.append({'host': host, 'n': len(d), 'cjk%': round(vc.get('cjk', 0), 1), 'latin3%': round(vc.get('latin3', 0), 1),
                     'id_only%': round(vc.get('id_only', 0), 1), 'distinct>=1%': round(100 * (dc >= 1).mean(), 1),
                     'distinct>=3%': round(100 * (dc >= 3).mean(), 1)})
        samples[host] = d.url.sample(min(10, len(d)), random_state=42).tolist()
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    df.to_csv(OUT / 'slug_meaning.csv', index=False)
    json.dump(samples, open(OUT / 'slug_samples.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 200, 'display.max_rows', 100)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
