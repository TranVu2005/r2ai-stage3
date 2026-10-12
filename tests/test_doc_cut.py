from r2ai.submit.doc_cut import cut_a1, cut_a2, make_cut

DOCS = list(range(100, 130))                       # RRF order
SCORE = {d: 5.0 - i * 0.2 for i, d in enumerate(DOCS)}   # logit falls 0.2 per doc


def test_a1_keeps_logit_above_tau_in_rrf_order():
    kept, fl, ce = cut_a1(DOCS, SCORE, tau=0.0, floor=5, ceil=30)
    assert kept == DOCS[:26] and not fl and not ce


def test_a1_floor_and_ceil():
    kept, fl, _ = cut_a1(DOCS, SCORE, tau=99, floor=7, ceil=30)
    assert kept == DOCS[:7] and fl
    kept, _, ce = cut_a1(DOCS, SCORE, tau=-99, floor=5, ceil=10)
    assert kept == DOCS[:10] and ce


def test_a1_protect_keeps_chunk_docs():
    kept, _, _ = cut_a1(DOCS, SCORE, tau=99, floor=3, ceil=30, protect={125, 128})
    assert kept == [100, 101, 102, 125, 128]


def test_a2_cuts_at_largest_drop_with_floor():
    sc = {d: 10.0 for d in DOCS[:12]} | {d: 1.0 for d in DOCS[12:]}
    assert cut_a2(DOCS, sc, floor=5)[0] == DOCS[:12]
    assert cut_a2(DOCS, sc, floor=20)[0] == DOCS[:20] or len(cut_a2(DOCS, sc, floor=20)[0]) >= 20


def test_make_cut_default_off_and_errors():
    assert make_cut('a1', tau=-99)(DOCS, SCORE)[0] == DOCS
    try:
        make_cut('a1')
    except ValueError:
        pass
    else:
        raise AssertionError
