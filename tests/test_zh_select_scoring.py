import numpy as np
import pytest

from r2ai.selective import scoring as s


def test_hash_split_groups_aliases_deterministically():
    urls = ['x/a', 'x/a', 'x/b', 'x/c']
    a = s.holdout_half(urls, seed=42)
    assert a[0] == a[1]
    assert np.array_equal(a, s.holdout_half(urls, seed=42))
    assert set(s.holdout_half([f'x/{i}' for i in range(1000)])) == {False, True}


def test_prefix_depth_numeric_id_and_decoded_semantic_slug():
    p1, p2, num, slug = s.url_features('https://x.com/book/2007/12345.html')
    assert (p1, p2, num, slug) == ('book', 'book/2007', 12345, '')
    assert s.url_features('https://x.com/w/%E8%82%BA%E7%82%8E')[-1] == '肺炎'
    assert s.url_features('https://x.com/a/72284parp.html')[-1] == ''  # opaque code
    assert s.url_features('https://x.com/wiki/heart-failure')[-1] == 'heart failure'


def test_beta_prior_binary_doc_yield_and_unseen_domain_mean():
    prior, mean = s.fit_beta(['a', 'a', 'b'], [2, 0, 0], [2, 2, 2], strength=6)
    assert mean == pytest.approx(1 / 3)
    assert prior['a'] == pytest.approx((2 + 6 / 3) / 10)
    assert prior['b'] == pytest.approx((0 + 6 / 3) / 8)
    assert s.score_prior(['a', 'unseen'], prior, mean).tolist() == [prior['a'], mean]
    with pytest.raises(ValueError):
        s.fit_beta(['a'], [3], [2])


def test_id_bins_learn_boundaries_only_from_train():
    bounds = s.fit_id_bounds([10, 20, 30, 40, None], buckets=2)
    assert bounds.tolist() == [25.0]
    assert s.id_keys([1, 20, 25, 40, 10**9, None], bounds) == [0, 0, 1, 1, 1, -1]


def test_max_cosine_is_global_url_score_over_all_queries():
    q = np.array([[1, 0], [0, 2]], dtype=np.float32)
    e = np.array([[3, 0], [1, 1], [0, 0]], dtype=np.float32)
    assert s.max_cosine(q, e, batch=1).tolist() == pytest.approx([1, 2**-.5, 0])


def test_order_ties_by_alias_count_then_url():
    order = s.ranked_indices([.2, .8, .8, .8], [3, 1, 2, 2], ['z', 'b', 'c', 'a'])
    assert order.tolist() == [3, 2, 1, 0]
    c = s.recall_at(order, [0, 1, 1, 1], fractions=(.25, .5, 1))
    assert c == pytest.approx({.25: 1/3, .5: 2/3, 1: 1})


def test_saturation_and_lane_budget_are_sequential_not_summed():
    assert s.saturation_fraction({.1: .2, .5: .8, .9: .97, 1.: 1.}) == .9
    assert s.saturation_fraction({.1: .2, .5: .5, 1.: 1.}) == 1
    out = s.allocate_lane([('a', 100, 100., .5), ('b', 1000, 100., 1.)], hours=10, uptime=.5)
    assert out[0]['cut'] == 50 and out[0]['budget_urls'] == 500
    assert out[1]['budget_urls'] == 450 and out[1]['cut'] == 450
    assert out[1]['hours_active'] == 4.5
    assert s.allocate_lane([('a', 100, None, 1.)], 10, .5)[0]['cut'] == 0


def test_cut_zero_budget_and_negative_inputs():
    assert s.allocate_lane([('a', 100, 0, .5)], 10, .8)[0]['cut'] == 0
    with pytest.raises(ValueError):
        s.allocate_lane([('a', 100, 1, .5)], -1, .8)
