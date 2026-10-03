"""Seeded stratified human search tasks. Generates links; never opens them."""
from __future__ import annotations

from r2ai.paths import auxiliary_disabled

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter
from urllib.parse import urlencode

from r2ai.gold_check.gold_check_common import OUT, data_path, normalize_text, stratum

GENERIC = ('bác sĩ cho em hỏi', 'có sao không ạ', 'cho em hỏi', 'xin hỏi bác sĩ', 'bác sĩ tư vấn', 'em cảm ơn', 'xin cảm ơn', 'thưa bác sĩ', 'chào bác sĩ')
# Transparent heuristic for Latin/medical terms, not an exhaustive dictionary.
LATIN = re.compile(r'^(?:[a-z]+(?:itis|osis|oma|emia|coccus|bacter|virus)|helicobacter|pylori|escherichia|coli|streptococcus|staphylococcus|candida|salmonella|dengue|insulin|metformin|amoxicillin|paracetamol|covid(?:-?19)?|hpv|hiv|hbsag|hbv|hcv)$', re.I)


def search_phrases(query, idf):
    words = query.split()
    # Preserve source punctuation internally for literal searches.
    full = query.strip()
    while full and (unicodedata.category(full[-1]).startswith('P') or full[-1].isspace()):
        full = full[:-1]
    if len(words) < 3:
        return [full]
    short = len(words) <= 12
    candidates = []
    widths = range(min(6, len(words)-1), min(12, len(words)-1) + 1) if short else range(8, 13)
    for width in widths:
        for start in range(len(words) - width + 1):
            tokens = words[start:start + width]
            clean = [normalize_text(w) for w in tokens]
            rarity = sum(idf.get(t, 0) for t in clean) / width
            numeric = sum(bool(re.search(r'\d', t)) for t in tokens)
            acronyms = sum(bool(re.fullmatch(r'[A-Z][A-Z0-9-]{1,}', t.strip('.,:;()?!'))) for t in tokens)
            latin = sum(bool(LATIN.fullmatch(t)) for t in clean)
            generic = sum(p in normalize_text(' '.join(tokens)) for p in GENERIC)
            score = rarity + 1.5 * numeric + 2 * acronyms + 2 * latin - 6 * generic
            candidates.append((score, start, width, ' '.join(tokens)))
    # Whole short query remains the primary search; extra exact snippets provide alternatives.
    chosen, spans = ([full] if short else []), []
    for score, start, width, phrase in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if phrase in chosen:
            continue
        overlap = max((max(0, min(start + width, b) - max(start, a)) / min(width, b - a) for a, b in spans), default=0)
        if overlap > .6:
            continue
        chosen.append(phrase)
        spans.append((start, start + width))
        if len(chosen) == 3:
            break
    # For 13-15 words all 8-word windows overlap: allow a second distinct phrase.
    if len(chosen) < 2:
        for _, _, _, phrase in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
            if phrase not in chosen:
                chosen.append(phrase)
                break
    return chosen


def links(phrase):
    exact = '"' + phrase.replace('"', ' ') + '"'
    sites = '(site:vinmec.com OR site:suckhoedoisong.vn OR site:nhathuoclongchau.com.vn OR site:hellobacsi.com)'
    return {'phrase': exact, 'links': {
        'google': 'https://www.google.com/search?' + urlencode({'q': exact}),
        'google_sites': 'https://www.google.com/search?' + urlencode({'q': exact + ' ' + sites}),
        'bing': 'https://www.bing.com/search?' + urlencode({'q': exact}),
        'coccoc': 'https://coccoc.com/search?' + urlencode({'query': exact}),
    }}


def make_tasks(rows, per_stratum=10, seed=42):
    df = Counter()
    for r in rows:
        df.update(set(normalize_text(r['query']).split()))
    idf = {w: math.log((len(rows) + 1) / (n + 1)) + 1 for w, n in df.items()}
    rng, tasks = random.Random(seed), []
    for s in ('short', 'medium', 'long'):
        eligible = sorted((r for r in rows if stratum(len(r['query'].split())) == s), key=lambda r: str(r['id']))
        if len(eligible) < per_stratum:
            raise ValueError(f'{s}: only {len(eligible)} queries; need {per_stratum}')
        for r in rng.sample(eligible, per_stratum):
            tasks.append(dict(id=str(r['id']), query=r['query'], word_count=len(r['query'].split()), stratum=s, searches=[links(p) for p in search_phrases(r['query'], idf)]))
    return tasks


def main():
    auxiliary_disabled()
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=str, default=str(data_path('query.parquet')))
    p.add_argument('--out', type=str, default=str(OUT / 'gold_check_tasks.json'))
    p.add_argument('--per-stratum', type=int, default=10)
    args = p.parse_args()
    from pathlib import Path
    rows = pq.read_table(args.queries, columns=['id', 'query']).to_pylist()
    tasks = make_tasks(rows, args.per_stratum)
    populations = Counter(stratum(len(r['query'].split())) for r in rows)
    output = dict(seed=42, population=len(rows), population_strata=dict(populations), tasks=tasks)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(tasks)} tasks -> {dest}')


if __name__ == '__main__':
    main()
