import polars as pl

from r2ai.zh_v2 import scope


def zh():
    return pl.DataFrame({'id': [1, 2, 3, 4, 5, 6, 7, 8], 'host': ['a.com', 'a.com', 'a.com', 'b.com', 'b.com', 'wujue.com', 'c.com', 'a.com']})


def rows():
    # one row = one normalised URL with all its corpus ids; ids 100/101 are not v2
    return pl.DataFrame({
        'url_norm': ['a/1', 'a/2', 'a/3', 'a/out', 'a/mixed', 'b/1', 'w/1', 'a/err', 'a/err-out'],
        'domain': ['a.com'] * 5 + ['b.com', 'wujue.com', 'a.com', 'a.com'],
        'status': ['ok', 'pending', 'thin', 'pending', 'pending', 'soft404_or_home', 'deferred_select', 'network_error',
                   'network_error'],
        'doc_ids': ['[1]', '[2]', '[3]', '[100]', '[101,8]', '[4]', '[6]', '[5]', '[102]'],
        'retry': [False, False, False, False, False, False, False, True, True]})


def test_explode_ids_and_v2_flags_count_v2_members_per_row():
    flags = scope.v2_flags(rows(), zh()).sort('url_norm')
    got = dict(zip(flags['url_norm'], zip(flags['n_ids'], flags['n_v2'])))
    assert got['a/1'] == (1, 1) and got['a/out'] == (1, 0) and got['a/mixed'] == (2, 1)


def test_defer_plan_keeps_open_rows_with_any_v2_id_and_defers_the_rest():
    plan = scope.defer_plan(rows(), zh())
    assert plan['url_norm'].to_list() == ['a/err-out', 'a/out']          # mixed row a/mixed stays; terminal rows never listed
    assert plan.columns == ['url_norm', 'domain', 'status']


def test_host_table_counts_each_v2_id_once_by_status_and_flags_absent_and_held_hosts():
    t = scope.host_table(zh(), rows()).to_dicts()
    by = {r['host']: r for r in t}
    a = by['a.com']
    assert (a['total'], a['ok'], a['thin'], a['pending'], a['other']) == (4, 1, 1, 2, 0)
    assert by['b.com']['other'] == 2 and by['b.com']['absent'] == 0     # soft404 and a network_error row
    assert by['c.com']['absent'] == 1 and by['c.com']['crawl_domain'] is None
    w = by['wujue.com']
    assert (w['deferred'], w['note']) == (1, 'hoãn')
    assert sum(r['total'] for r in t) == 8
