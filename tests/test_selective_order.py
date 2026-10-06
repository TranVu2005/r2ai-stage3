import numpy as np
import pandas as pd

from r2ai.selective.cnkang_order import fit_prior, order_pending, recall_curve


def test_fit_prior_smooths_and_unseen_gets_mean():
    prior, mean = fit_prior(['a', 'a', 'b', 'b'], np.array([4.0, 4.0, 0.0, 0.0]), m=0.0)
    assert prior == {'a': 4.0, 'b': 0.0} and mean == 2.0
    prior, _ = fit_prior(['a'], np.array([4.0]), m=1.0)
    assert prior['a'] == 4.0  # single bucket: smoothed rate == global mean


def test_order_by_prior_ties_by_url_norm_unseen_mean():
    pend = pd.DataFrame({'url_norm': ['cnkang.com/x/3.html', 'cnkang.com/y/2.html', 'cnkang.com/x/1.html', 'cnkang.com/z/9.html'],
                         'url': ['http://www.cnkang.com/x/3.html', 'http://www.cnkang.com/y/2.html',
                                 'http://www.cnkang.com/x/1.html', 'http://www.cnkang.com/z/9.html'],
                         'rank': [10, 11, 12, 13]})
    out = order_pending(pend, {'x': 5.0, 'y': 1.0}, mean=2.0)
    assert out.url_norm.tolist() == ['cnkang.com/x/1.html', 'cnkang.com/x/3.html', 'cnkang.com/z/9.html', 'cnkang.com/y/2.html']
    assert out['rank'].tolist() == [0, 1, 2, 3]
    assert out.db_rank.tolist() == [10, 11, 12, 13]       # same rank values, permuted to the new order
    assert out.prefix.tolist() == ['x', 'x', 'z', 'y']
    assert sorted(out.url_norm) == sorted(pend.url_norm)   # no URL dropped


def test_recall_curve_monotone_and_ends_at_one():
    rng = np.random.default_rng(0)
    score = rng.random(100)
    hits = (score * 10).round()
    c = recall_curve(score, hits, (0.1, 0.5, 1.0))
    assert c[0.1] <= c[0.5] <= c[1.0] and abs(c[1.0] - 1.0) < 1e-9 and c[0.1] > 0.1
