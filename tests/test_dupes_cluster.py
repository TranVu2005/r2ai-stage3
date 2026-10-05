"""H1 duplicate clustering: exact (metric-normalised body), near (MinHash LSH + true Jaccard >= 0.8), boilerplate flags."""
import random

import numpy as np
import pytest

from r2ai.dupes import cluster as C
from r2ai.eval.scorer import normalize_metric

WORDS = [f'tu{i}' for i in range(400)]


def article(seed: int, n: int = 200) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(WORDS) + str(rng.randrange(10**6)) for _ in range(n)]      # near-unique words: no accidental overlap


def doc(doc_id, body_words, domain='a.vn', title=None, n_tokens=None, **kw):
    body = ' '.join(body_words) if not isinstance(body_words, str) else body_words
    return C.Doc(doc_id=doc_id, domain=domain, title=f'title {doc_id}' if title is None else title, body=body,
                 n_tokens=len(body.split()) if n_tokens is None else n_tokens, **kw)


@pytest.fixture(scope='module')
def fixture_docs():
    a, d = article(1), article(2)
    near = list(a)
    for i in range(0, 200, 50):                       # 4 of 200 words changed -> 5-word shingles: Jaccard ~0.85
        near[i] = 'changed' + str(i)
    boiler = 'bài viết đang được cập nhật vui lòng quay lại sau ' * 2
    template = article(3, 120)
    return [
        doc(1, a, domain='a.vn', title='Bệnh tiểu đường'),
        doc(2, a, domain='b.vn', title='Khác tiêu đề'),                                  # exact (same words)
        doc(3, ' '.join(a).upper().replace(' ', '  '), domain='c.vn'),  # exact after normalisation
        doc(4, near, domain='d.vn'),                                                      # near
        doc(5, d, domain='a.vn'),                                                         # unique
        doc(6, boiler, domain='x.vn', title='t6'),                                        # short boilerplate x2 (exact, < 50 tokens)
        doc(7, boiler, domain='x.vn', title='t7'),
        doc(8, template, domain='t.vn', title='alpha'),                                   # same-template trio: long, same domain, distinct titles
        doc(9, template, domain='t.vn', title='beta'),
        doc(10, template, domain='t.vn', title='gamma'),
    ]


def test_exact_key_uses_metric_normalisation():
    assert C.exact_key('Hello &amp;  World\n') == C.exact_key(' hello & world')
    assert C.exact_key('ＡＢＣ') == C.exact_key('abc')                                    # NFKC
    assert C.exact_key('abc') != C.exact_key('abd')
    assert C.exact_key('') is None and C.exact_key(None) is None and C.exact_key(' \n ') is None
    assert normalize_metric('x  Y') == 'x y'


def test_shingles_are_5_word_sets():
    sh = C.shingle_hashes(C.words('a b c d e f g'))
    assert len(sh) == 3 and (np.diff(sh.astype(np.float64)) > 0).all()                   # unique, sorted
    assert len(C.shingle_hashes(C.words('a b c'))) == 1                                  # shorter than the window
    assert C.shingle_hashes([]).size == 0
    assert np.array_equal(C.shingle_hashes(C.words('A b C d E')), C.shingle_hashes(C.words('a b c d e')))


def test_minhash_estimates_jaccard_and_is_seed_stable():
    a, b = article(5, 400), article(5, 400)
    b[::10] = [f'z{i}' for i in range(40)]
    sa, sb = C.shingle_hashes(a), C.shingle_hashes(b)
    true = C.jaccard(sa, sb)
    mh = C.MinHasher(128, seed=42)
    est = float((mh.signature(sa) == mh.signature(sb)).mean())
    assert abs(est - true) < 0.15
    assert np.array_equal(mh.signature(sa), C.MinHasher(128, seed=42).signature(sa))
    assert not np.array_equal(mh.signature(sa), C.MinHasher(128, seed=7).signature(sa))
    assert C.jaccard(sa, sa) == 1.0 and C.jaccard(sa, np.array([], np.uint64)) == 0.0


def test_lsh_pairs_find_similar_not_unrelated():
    mh = C.MinHasher(128, seed=42)
    a = article(1)
    near = list(a)
    near[::50] = ['q' + str(i) for i in range(4)]
    sigs = np.stack([mh.signature(C.shingle_hashes(x)) for x in (a, near, article(2), article(3))])
    pairs = C.lsh_pairs(sigs)
    assert (0, 1) in pairs and not any(2 in p or 3 in p for p in pairs)


def test_cluster_corpus_exact_near_boilerplate(fixture_docs):
    res = C.cluster_corpus(fixture_docs)
    by = {frozenset(c.doc_ids): c for c in res.clusters}
    assert set(by) == {frozenset({1, 2, 3, 4}), frozenset({6, 7}), frozenset({8, 9, 10})}      # 5 stays a singleton
    main = by[frozenset({1, 2, 3, 4})]
    assert main.kind == 'near' and not main.boilerplate and main.exact_groups == 2 and main.size == 4
    assert main.domains == 4 and main.min_jaccard >= 0.8
    short = by[frozenset({6, 7})]
    assert short.kind == 'exact' and short.short and short.boilerplate                       # < 50 tokens: exact only, flagged
    tpl = by[frozenset({8, 9, 10})]
    assert tpl.kind == 'exact' and tpl.template and tpl.boilerplate and not tpl.short
    assert res.cluster_of[1] == res.cluster_of[4] != res.cluster_of[6]
    assert 5 not in res.cluster_of
    assert res.stats['docs'] == 10 and res.stats['docs_in_clusters'] == 9 and res.stats['clusters'] == 3


def test_short_docs_never_join_near_clusters():
    a = article(1, 30)
    near = list(a)
    near[10] = 'changed'
    res = C.cluster_corpus([doc(1, a), doc(2, near)])                                       # 30 tokens < 50
    assert res.clusters == []


def test_below_jaccard_threshold_is_not_clustered():
    a = article(1, 200)
    far = list(a)
    far[::5] = [f'w{i}' for i in range(40)]                                                 # 20% of words changed -> Jaccard << 0.8
    res = C.cluster_corpus([doc(1, a), doc(2, far)])
    assert res.clusters == []


def test_cluster_result_is_deterministic_and_order_independent(fixture_docs):
    r1 = C.cluster_corpus(fixture_docs)
    r2 = C.cluster_corpus(list(reversed(fixture_docs)))
    assert sorted(sorted(c.doc_ids) for c in r1.clusters) == sorted(sorted(c.doc_ids) for c in r2.clusters)


def test_chunk_key_follows_builder_rule():
    body = 'Câu hỏi?\n\nTrả lời đầy đủ ở đây.'
    assert C.chunk_key(body, 'Trả lời đầy đủ ở đây.', 't', 'd') == C.exact_key('Trả lời đầy đủ ở đây.')   # answer inside body
    assert C.chunk_key(body, 'không có trong body', 't', 'd') == C.exact_key(body)
    assert C.chunk_key('  ', '', 'T', 'D') == C.exact_key('T\n\nD')                                       # title + description
