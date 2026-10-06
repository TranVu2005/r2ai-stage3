from r2ai.selective.bm25 import Slug, toks, slug_toks


def test_toks_strip_accents_and_digits():
    assert toks('Đau đầu, viêm gan B 2024!') == ['dau', 'dau', 'viem', 'gan']


def test_slug_toks_ignore_host_and_scheme():
    assert slug_toks('https://vov.vn/suc-khoe/viem-gan-b-post123.vov?x=1') == ['suc', 'khoe', 'viem', 'gan', 'vov']


def test_ranking_prefers_matching_slug():
    urls = ['https://a.vn/tin/viem-gan-b-lay-qua-dau', 'https://a.vn/tin/thoai-hoa-khop-goi', 'https://a.vn/id/12345']
    s = Slug(urls)
    scores = (s.queries(['Viêm gan B lây qua đường nào?']) @ s.WT).toarray()[0]
    assert scores.argmax() == 0 and scores[2] == 0 and scores[1] == 0


def test_unknown_query_terms_score_zero():
    s = Slug(['https://a.vn/benh-tim'])
    assert (s.queries(['zzzz']) @ s.WT).toarray().sum() == 0
