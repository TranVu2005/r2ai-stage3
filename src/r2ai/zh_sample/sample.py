from __future__ import annotations

from r2ai.paths import CORPUS_FILE, LEGACY_OUT_DIR, require_inputs
from .common import RUN, DB, atomic_json, guard_owned, preflight, exclusive
import csv
import hashlib
import math
import random
import json
from collections import defaultdict
from vicrawl.urlnorm import group_urls

EXCLUDED = {'zysjonline.com', 'bingli.iiyi.com'}


def sample_rows(rows, domains: set[str]):
    by_domain = defaultdict(list)
    for g in group_urls(rows):
        if g['domain'] in domains:
            by_domain[g['domain']].append(g)
    selected, inventory = [], []
    for domain, groups in sorted(by_domain.items()):
        groups.sort(key=lambda g: g['url_norm'])
        n = min(len(groups), max(200, math.ceil(len(groups) * .02)))
        excluded = domain in EXCLUDED
        inventory.append({'domain': domain, 'n_rows': sum(len(g['doc_ids']) for g in groups),
                          'n_unique': len(groups), 'n_sample': 0 if excluded else n, 'excluded': excluded})
        if excluded:
            continue
        chosen = random.Random(42).sample(groups, n)
        for rank, g in enumerate(chosen):
            selected.append({**g, 'rank': rank})
    return selected, inventory


def prepare():
    import polars as pl
    from vicrawl.domains import _domain_expr
    from vicrawl.state import StateDB
    preflight()
    classification = LEGACY_OUT_DIR / 'domain_classification_all.csv'
    require_inputs(CORPUS_FILE, classification)
    with exclusive('prepare'):
        if (RUN / 'sample.json').exists():
            require_inputs(RUN / 'sample_manifest.json', RUN / 'inventory.json')
            manifest = json.loads((RUN / 'sample_manifest.json').read_text('utf-8'))
            if manifest['sample_sha256'] != hashlib.sha256((RUN / 'sample.json').read_bytes()).hexdigest():
                raise ValueError('Saved sample hash differs; refusing to regenerate a live checkpoint')
            selected = json.loads((RUN / 'sample.json').read_text('utf-8'))
            db = StateDB(DB)
            try:
                added = db.add_urls(selected)
                print(f'Sample resumed: {len(selected)} groups, {added} new DB rows', flush=True)
            finally:
                db.conn.close()
            return
        with classification.open(encoding='utf-8-sig', newline='') as f:
            catalog = list(csv.DictReader(f))
        domains = {r['domain'] for r in catalog if 'lang=zh' in r['reason']} | EXCLUDED
        df = pl.read_parquet(CORPUS_FILE, columns=['id', 'url']).with_columns(_domain_expr()).filter(pl.col('domain').is_in(sorted(domains)))
        selected, inventory = sample_rows(zip(df['id'].to_list(), df['url'].to_list()), domains)
        atomic_json(RUN / 'sample.json', selected)
        atomic_json(RUN / 'inventory.json', inventory)
        atomic_json(RUN / 'sample_manifest.json', {'seed': 42, 'fraction': .02, 'floor': 200, 'rounding': 'ceil',
                    'corpus': str(CORPUS_FILE), 'classification': str(classification),
                    'selection': 'legacy detected lang=zh; unknown language excluded',
                    'sample_sha256': hashlib.sha256((RUN / 'sample.json').read_bytes()).hexdigest(),
                    'unclassified': [r for r in catalog if r['is_vi'] == 'False' and r['domain'] not in domains]})
        guard_owned(DB)
        db = StateDB(DB)
        try:
            db.add_urls(selected)
            db.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        finally:
            db.conn.close()
        print(f'{len(selected)} URL groups, {sum(not r["excluded"] for r in inventory)} zh domains', flush=True)


if __name__ == '__main__':
    prepare()
