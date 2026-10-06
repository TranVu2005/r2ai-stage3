"""Step 1: schema + domain x group table + unclassified URLs. Read-only."""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from r2ai.paths import CORPUS_FILE, OLD_ROOT
from r2ai.paths import STATE_DIR
from r2ai.selective.common import OUT, VI_DB, ZH_DB, ro, host_of

ZS_DB = STATE_DIR / 'crawl_zh_sample.db'

OLD_OUT = OLD_ROOT / 'out'


def explode_ids(db):
    c = ro(db)
    rows = c.execute('select domain,status,doc_ids from urls').fetchall()
    ids, dom, st = [], [], []
    for d, s, ids_json in rows:
        for i in json.loads(ids_json):
            ids.append(i); dom.append(d); st.append(s)
    return pd.DataFrame({'id': ids, 'dom': dom, 'status': st})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(CORPUS_FILE)
    schema = [{'col': c, 'type': str(t[c].type), 'nonnull_pct': round(100 * (1 - t[c].null_count / len(t[c])), 3)} for c in t.column_names]
    corp = t.to_pandas()
    corp['host'] = corp.url.map(host_of)
    vi, zh = explode_ids(VI_DB), explode_ids(ZH_DB)
    corp = corp.merge(vi.rename(columns={'dom': 'vi_dom', 'status': 'vi_status'}).drop_duplicates('id'), on='id', how='left')
    corp = corp.merge(zh.rename(columns={'dom': 'zh_dom', 'status': 'zh_status'}).drop_duplicates('id'), on='id', how='left')
    zs = explode_ids(ZS_DB)
    corp = corp.merge(zs.rename(columns={'dom': 'zs_dom', 'status': 'zs_status'}).drop_duplicates('id'), on='id', how='left')
    inv = {r['domain']: r for r in json.load(open('out/runs/zh-sample/inventory.json', encoding='utf-8'))}
    clf = pd.read_csv(OLD_OUT / 'domain_classification_all.csv')
    zh_hosts = set(clf[(clf.is_vi == False) & (clf.reason.str.contains('lang=zh|main_lang=zh', regex=True))].domain) | set(inv)
    vi_hosts = set(clf[clf.is_vi == True].domain)
    sample_docs = set()
    sm = json.load(open('out/runs/zh-sample/sample.json', encoding='utf-8'))
    def grp(r):
        if pd.notna(r.vi_dom): return 'vi_crawled'
        if pd.notna(r.zh_dom): return 'zh_full'
        if pd.notna(r.zs_dom): return 'zh_sample'
        if r.host in zh_hosts: return 'zh_not_in_full'
        if r.host in vi_hosts: return 'vi_not_in_db'
        return 'unclassified'
    corp['grp'] = [grp(r) for r in corp.itertuples()]
    print(corp.grp.value_counts())
    corp[['id', 'host', 'grp', 'vi_status', 'zh_status', 'zs_status']].to_parquet(OUT / 'corpus_groups.parquet')
    tab = corp.groupby(['host', 'grp']).size().unstack(fill_value=0)
    tab.to_csv(OUT / 'domain_group_table.csv')
    un = corp[corp.grp.isin(['unclassified', 'vi_not_in_db'])].groupby(['host', 'grp']).size().sort_values(ascending=False)
    un.head(40).to_csv(OUT / 'unclassified_top.csv')
    json.dump({'schema': schema, 'n': len(corp), 'groups': corp.grp.value_counts().to_dict(),
               'vi_status': corp[corp.grp == 'vi_crawled'].vi_status.value_counts().to_dict(),
               'zh_full_status': corp[corp.grp == 'zh_full'].zh_status.value_counts().to_dict()},
              open(OUT / 'step1.json', 'w'), indent=1)
    print(schema); print(un.head(25))


if __name__ == '__main__':
    main()
