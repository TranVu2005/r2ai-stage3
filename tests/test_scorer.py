"""Hand-written cases for eval/scorer.py and eval/validate.py.

Most cases use a whitespace tokenizer so token counts (and therefore the 40 % / 80 % thresholds) are exact by
construction; a few use the real BGE-M3 tokenizer.
"""
from r2ai.paths import ROOT

import json
import random
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.eval import scorer as S
from r2ai.eval import validate as V

WS = S.whitespace_tokenizer()


def words(n, prefix='w'):
    return ' '.join(f'{prefix}{i}' for i in range(n))


def q(qid, docs, chunks):
    return {'id': qid, 'relevant_docs': docs, 'relevant_chunks': [{'doc_id': d, 'chunk_text': t} for d, t in chunks]}


# ---- LCS ------------------------------------------------------------------------------------------------------
def test_lcs_bitparallel_equals_dp_random():
    rnd = random.Random(42)
    for _ in range(300):
        a = [rnd.randrange(6) for _ in range(rnd.randrange(0, 90))]
        b = [rnd.randrange(6) for _ in range(rnd.randrange(0, 90))]
        assert S.lcs(a, b) == S.lcs_dp(a, b)


def test_lcs_long_pattern_over_64_bits():
    a = list(range(200))
    b = list(range(0, 200, 2)) + [999] * 50
    assert S.lcs(a, b) == 100 == S.lcs_dp(a, b)


# ---- doc level ------------------------------------------------------------------------------------------------
def test_doc_f2_formula():
    p, r, f = S.doc_prf([1, 2, 3, 4], [1, 9])          # P = 1/4, R = 1/2
    assert (p, r) == (0.25, 0.5)
    assert f == pytest.approx(5 * 0.25 * 0.5 / (4 * 0.25 + 0.5))


def test_doc_empty_prediction_is_zero():
    assert S.doc_prf([], [1, 2]) == (0.0, 0.0, 0.0)


def test_doc_duplicates_count_once():
    assert S.doc_prf([5, 5, 5], [5]) == (1.0, 1.0, 1.0)


# ---- chunk level ----------------------------------------------------------------------------------------------
def test_chunk_identical_is_perfect():
    g = words(30)
    s = S.score([q(1, [7], [(7, g)])], [q(1, [7], [(7, g)])], WS)
    assert (s.doc_f2, s.chunk_f2, s.final) == (1.0, 1.0, 1.0)


def test_chunk_exactly_40_percent_matches():
    g = words(10)                       # len(g) = 10 -> needs LCS >= 4
    pred = 'w0 w1 w2 w3 x y z'          # LCS 4 = exactly 40 %
    p, r, _ = S.chunk_prf([(1, WS.encode(pred))], [(1, WS.encode(g))])
    assert (p, r) == (1.0, 1.0)


def test_chunk_just_below_40_percent_misses():
    g = words(10)
    pred = 'w0 w1 w2 x y z'             # LCS 3 < 4
    assert S.chunk_prf([(1, WS.encode(pred))], [(1, WS.encode(g))]) == (0.0, 0.0, 0.0)


def test_chunk_threshold_is_relative_to_gold_length_not_pred():
    g = words(5)                        # needs LCS >= 2
    pred = 'w0 w1 ' + words(200, 'x')   # huge pred, LCS 2 -> match
    p, r, _ = S.chunk_prf([(1, WS.encode(pred))], [(1, WS.encode(g))])
    assert (p, r) == (1.0, 1.0)


def test_chunk_other_doc_id_never_matches():
    g = words(20)
    p, r, f = S.chunk_prf([(2, WS.encode(g))], [(1, WS.encode(g))])
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_duplicate_preds_both_count_in_precision():
    a = words(10)
    c = words(10, 'c')                  # unrelated, wrong
    p, r, _ = S.chunk_prf([(1, WS.encode(x)) for x in (a, a, c)], [(1, WS.encode(a))])
    assert p == pytest.approx(2 / 3)    # no merging: the two identical preds are both correct, c is wrong
    assert r == 1.0


def test_near_duplicate_preds_not_merged():
    a = words(10)
    b = 'w0 w1 w2 w3 w4 w5 w6 w7 w8 zz'         # LCS 9 / 10: old rule would have merged a and b
    c = words(10, 'c')
    p, _, _ = S.chunk_prf([(1, WS.encode(x)) for x in (a, b, c)], [(1, WS.encode(a))])
    assert p == pytest.approx(2 / 3)


def test_same_text_in_other_doc_is_wrong_pred():
    a = words(10)
    p, _, _ = S.chunk_prf([(1, WS.encode(a)), (2, WS.encode(a))], [(1, WS.encode(a))])
    assert p == 0.5


def test_recall_counts_gold_chunks():
    g1, g2 = words(10, 'a'), words(10, 'b')
    p, r, _ = S.chunk_prf([(1, WS.encode(g1))], [(1, WS.encode(g1)), (1, WS.encode(g2))])
    assert (p, r) == (1.0, 0.5)


def test_query_without_gold_is_excluded_from_macro():
    gold = [q(1, [7], [(7, words(10))]),
            {'id': 2, 'relevant_docs': [], 'relevant_chunks': []},          # no gold: skipped at both levels
            {'id': 3, 'relevant_docs': [8], 'relevant_chunks': []}]          # doc gold only: skipped at chunk level
    pred = [q(1, [7], [(7, words(10))]), q(2, [99], [(99, 'anything')]), q(3, [8], [])]
    s = S.score(pred, gold, WS)
    assert (s.n_doc_queries, s.n_chunk_queries) == (2, 1)
    assert (s.doc_f2, s.chunk_f2) == (1.0, 1.0)


def test_missing_prediction_scores_zero():
    gold = [q(1, [7], [(7, words(10))]), q(2, [8], [(8, words(10))])]
    s = S.score([q(1, [7], [(7, words(10))])], gold, WS)
    assert s.doc_f2 == 0.5 and s.chunk_f2 == 0.5 and s.final == 0.5


def test_macro_all_vs_has_gold():
    gold = [q(1, [7], [(7, words(10))]),
            {'id': 2, 'relevant_docs': [8], 'relevant_chunks': []}]          # doc gold only
    pred = [q(1, [7], [(7, words(10))]), q(2, [8], [])]
    hg = S.score(pred, gold, WS)                                              # default has_gold
    al = S.score(pred, gold, WS, macro='all')
    assert (hg.macro, hg.chunk_f2, hg.doc_f2) == ('has_gold', 1.0, 1.0)
    assert (al.macro, al.chunk_f2, al.doc_f2) == ('all', 0.5, 1.0)          # query 2 counts 0 at chunk level
    assert hg.by_macro == al.by_macro and set(hg.by_macro) == {'all', 'has_gold'}
    with pytest.raises(ValueError):
        S.score(pred, gold, WS, macro='x')


def test_empty_everything_is_zero():
    gold = [q(1, [7], [(7, words(10))])]
    s = S.score([q(1, [], [])], gold, WS)
    assert (s.doc_f2, s.chunk_f2, s.final) == (0.0, 0.0, 0.0)


# ---- normalisation --------------------------------------------------------------------------------------------
def test_normalize_keeps_punctuation_and_tags():
    assert S.normalize_metric('  Bệnh  Tiểu-đường,\n LÀ gì?! ') == 'bệnh tiểu-đường, là gì?!'
    assert S.normalize_metric('<p>A</p>') == '<p>a</p>'          # tags are not stripped
    assert S.normalize_metric('A&amp;B &lt;x&gt; &nbsp;') == 'a&b <x>'


def test_punctuation_counts_in_tokens():
    g = 'a b c d e'                                   # whitespace tokenizer: punctuation stays glued to the word
    pred = 'a, b. c; x y'                            # without punctuation stripping only none matches
    assert S.chunk_prf([(1, WS.encode(pred))], [(1, WS.encode(g))]) == (0.0, 0.0, 0.0)
    assert S.chunk_prf([(1, WS.encode('a b c x y'))], [(1, WS.encode(g))])[:2] == (1.0, 1.0)


def test_normalize_nfkc_fullwidth():
    assert S.normalize_metric('ＡＢＣ １２３') == 'abc 123'
    assert S.normalize_metric('ｗ０ ｗ１') == 'w0 w1'
    assert WS.encode('ｗ０ ｗ１ ｗ２') == WS.encode('w0 w1 w2')


def test_normalize_nfkc_composes_vietnamese():
    decomposed = 'Việt'                # "Việt" written with combining marks
    assert S.normalize_metric(decomposed) == S.normalize_metric('Việt') == 'việt'


def test_normalisation_makes_chunks_equal_real_tokenizer():
    tok = S.Tokenizer()                               # real BAAI/bge-m3 tokenizer, local cache only
    a = tok.encode('Trẻ  bị SỐT cao,\nnên làm gì?')
    b = tok.encode('trẻ bị sốt cao, nên làm gì?')
    assert a == b and len(a) > 0
    assert tok.encode('') == ()


# ---- validator ------------------------------------------------------------------------------------------------
@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / 'docs'
    d.mkdir()
    rows = [{'doc_ids': [10, 11], 'title': 'Tiêu đề', 'description': None, 'question': None, 'answer': None,
             'body': 'Đoạn một  có\nxuống dòng.\n\nĐoạn hai.'}]
    schema = pa.schema([('doc_ids', pa.list_(pa.int64()))] + [(k, pa.string()) for k in V.TEXT_FIELDS])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), d / 'x.parquet')
    return d


def test_validator_accepts_good_and_rejects_bad(docs_dir):
    corpus = np.array([10, 11, 12])
    good = [q(1, [10, 11], [(10, 'Đoạn một  có\nxuống dòng.'), (11, 'Đoạn hai.')]), q(2, [12], [])]
    errs, _, _ = V.validate(good, [1, 2], corpus, docs_dir)
    assert errs == []
    bad = [q(1, [10, 999], [(10, 'không có trong trang')]), {'id': '2', 'relevant_docs': [], 'relevant_chunks': []},
           q(1, [10], [])]
    errs, _, _ = V.validate(bad, [1, 2], corpus, docs_dir)
    text = '\n'.join(errs)
    assert 'not in links_corpus' in text and 'not a substring' in text and "id '2' is not an int" in text
    assert 'duplicated' in text and 'missing' in text


def test_validator_verbatim_substring_and_chunk_order(docs_dir):
    corpus = np.array([10, 11, 12])
    ok = [q(1, [10], [(10, 'Đoạn một  có\nxuống dòng.')])]
    ok[0]['relevant_chunks'][0]['chunk_order'] = 0
    assert V.validate(ok, [1], corpus, docs_dir)[0] == []
    ws_only = [q(1, [10], [(10, 'Đoạn một có xuống dòng.')])]          # whitespace changed -> no longer verbatim
    assert 'not a substring' in chr(10).join(V.validate(ws_only, [1], corpus, docs_dir)[0])
    for bad in (-1, 1.5, '0', True):
        row = [q(1, [10], [(10, 'Đoạn hai.')])]
        row[0]['relevant_chunks'][0]['chunk_order'] = bad
        assert 'chunk_order' in chr(10).join(V.validate(row, [1], corpus, docs_dir)[0])


def test_validator_zip_must_hold_one_file(tmp_path):
    z = tmp_path / 's.zip'
    with zipfile.ZipFile(z, 'w') as f:
        f.writestr('a.json', json.dumps([]))
        f.writestr('b.json', json.dumps([]))
    _, errs = V.load_raw(z)
    assert errs and 'exactly 1 file' in errs[0]
