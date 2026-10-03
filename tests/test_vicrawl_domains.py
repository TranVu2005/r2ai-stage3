from r2ai.paths import ROOT

import csv

import polars as pl

from vicrawl.domains import DEFERRED, build_url_table, classify_domains, write_domain_files


def corpus(tmp_path):
    rows = []
    i = 0
    for dom, n in [('a.vn', 3), ('nhathuoclongchau.com.vn', 2), ('vinmec.com', 2), ('cnkang.com', 2), ('unknown-vi.org', 3), ('dead.example', 2)]:
        for k in range(n):
            i += 1
            rows.append((i, f'https://{dom}/p{k}'))
    rows.append((100, 'http://www.a.vn/p0'))       # duplicate of a.vn/p0 (id 1)
    rows.append((101, 'https://a.vn/p1/'))         # trailing slash duplicate
    df = pl.DataFrame({'id': [r[0] for r in rows], 'url': [r[1] for r in rows]})
    p = tmp_path / 'c.parquet'
    df.write_parquet(p)
    return p


def test_classification_rules(tmp_path):
    p = corpus(tmp_path)
    probed = {'cnkang.com': 'zh'}
    calls = []

    def probe(domain, urls):
        calls.append((domain, len(urls)))
        return {'unknown-vi.org': 'vi'}.get(domain)

    res = {r['domain']: r for r in classify_domains(p, probed_lang=probed, extra_vi={'vinmec.com'}, probe_fn=probe)}
    assert res['a.vn']['is_vi'] and '.vn' in res['a.vn']['reason']
    assert res['vinmec.com']['is_vi']
    assert not res['cnkang.com']['is_vi'] and 'probe' in res['cnkang.com']['reason']
    assert res['unknown-vi.org']['is_vi']
    assert not res['dead.example']['is_vi'] and 'probe_failed' in res['dead.example']['reason']
    assert ('unknown-vi.org', 3) in calls and all(d not in ('a.vn', 'vinmec.com', 'cnkang.com') for d, _ in calls)
    assert res['a.vn']['n_rows'] == 5 and res['a.vn']['n_unique'] == 3
    assert res['nhathuoclongchau.com.vn']['is_vi'] and res['nhathuoclongchau.com.vn']['deferred']


def test_files_written_and_deferred_excluded(tmp_path):
    p = corpus(tmp_path)
    res = classify_domains(p, probed_lang={}, extra_vi=set(), probe_fn=lambda d, u: None)
    write_domain_files(res, tmp_path / 'out')
    vi = list(csv.DictReader(open(tmp_path / 'out/vi_domains.csv', encoding='utf-8')))
    assert {r['domain'] for r in vi} == {'a.vn'}
    assert list(vi[0].keys()) == ['domain', 'n_rows', 'n_unique', 'reason']
    deferred = list(csv.DictReader(open(tmp_path / 'out/deferred_domains.csv', encoding='utf-8')))
    assert [r['domain'] for r in deferred] == ['nhathuoclongchau.com.vn']
    assert 'nhathuoclongchau.com.vn' in DEFERRED


def test_build_url_table_groups_dups_and_ranks(tmp_path):
    p = corpus(tmp_path)
    groups = build_url_table(p, {'a.vn'})
    assert len(groups) == 3
    by = {g['url_norm']: g for g in groups}
    assert by['a.vn/p0']['doc_ids'] == [1, 100] and by['a.vn/p0']['url'].startswith('https://')
    assert by['a.vn/p1']['doc_ids'] == [2, 101]
    assert sorted(g['rank'] for g in groups) == [0, 1, 2]
    again = build_url_table(p, {'a.vn'})
    assert [g['url_norm'] for g in sorted(groups, key=lambda g: g['rank'])] == [g['url_norm'] for g in sorted(again, key=lambda g: g['rank'])]
