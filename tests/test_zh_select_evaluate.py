import pandas as pd
import numpy as np
from r2ai.selective import evaluate as e


def test_proxy_is_binary_top100_not_occurrence_count_or_top30():
    assert e.proxy_ids({1: {'dense': [[i, 0] for i in range(101)]},
                        2: {'dense': [[5, 1], [200, 0]]}}) == set(range(100)) | {200}


def test_group_sample_retains_unchunked_negatives_and_url_alias_split():
    df = pd.DataFrame({'url_norm': ['x/a', 'x/a', 'x/b'], 'url': ['https://x/a']*2+['https://x/b'],
                       'domain': ['x']*3, 'doc_ids': [[1, 2], [3], [4]]})
    out = e.group_sample(df, {1, 2})
    assert out.documents.tolist() == [2, 1]
    assert out.relevant.tolist() == [1, 0]  # two aliases do not become two docs
    assert out.aliases.tolist() == [3, 1]


def test_holdout_uses_fit_half_only_and_evaluates_all_methods():
    d = pd.DataFrame({'url_norm': [f'x/{i}' for i in range(1000)],
                       'url': [f'https://x/{"hit" if i%2 else "miss"}/{i}.html' for i in range(1000)],
                       'domain': ['x']*1000, 'documents': [1]*1000,
                       'relevant': [i%2 for i in range(1000)], 'aliases': [1]*1000})
    got = e.evaluate_domain(d)
    assert got['fit_urls'] + got['holdout_urls'] == 1000
    assert got['chosen_method'] in got['curves']
    assert got['curves']['A1']['0.5'] == 1.0
    assert {'alphabetical', 'random42', 'A1', 'A2', 'C20', 'A1+C20'} <= got['curves'].keys()


def test_less_than_200_relevant_docs_cannot_choose_prior():
    d = pd.DataFrame({'url_norm': [f'x/{i}' for i in range(20)], 'url': [f'https://x/{i}.html' for i in range(20)],
                       'domain': ['x']*20, 'documents': [1]*20, 'relevant': [1]*20, 'aliases': [1]*20})
    got = e.evaluate_domain(d)
    assert got['chosen_method'] == 'insufficient_signal'
    assert got['saturation_fraction'] is None


def test_no_sample_still_scores_unseen_urls_without_crash():
    empty=pd.DataFrame(columns=['url_norm','url','domain','documents','relevant','aliases'])
    pending=pd.DataFrame({'url':['https://x/1.html','https://x/2.html']})
    scores=e.structural_scores(empty,pending)
    assert all(v.tolist()==[0,0] for v in scores.values())
    result=e.evaluate_domain(empty)
    assert result['chosen_method']=='insufficient_signal' and result['sample_docs']==0
