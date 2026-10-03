from r2ai.paths import ROOT

from r2ai.eval.scorer import whitespace_tokenizer
from r2ai.submit.make_submission import dedupe


def test_exact_after_normalisation():
    keep, ne, nn = dedupe(['Hello  World', 'hello world', 'other text here'], whitespace_tokenizer())
    assert keep == [True, False, True] and (ne, nn) == (1, 0)


def test_near_dup_drops_later_vs_shorter():
    a = ' '.join(f'w{i}' for i in range(20))
    b = a + ' extra1 extra2 extra3'                      # contains a: LCS/len(a) = 1.0
    c = ' '.join(f'z{i}' for i in range(20))
    keep, ne, nn = dedupe([a, b, c], whitespace_tokenizer())
    assert keep == [True, False, True] and (ne, nn) == (0, 1)


def test_low_overlap_kept():
    a = ' '.join(f'w{i}' for i in range(10))
    b = ' '.join(f'w{i}' for i in range(5)) + ' ' + ' '.join(f'y{i}' for i in range(5))
    assert dedupe([a, b], whitespace_tokenizer())[0] == [True, True]


def test_split_cap_verbatim_and_bounded():
    from r2ai.submit.make_submission import split_cap
    tok = whitespace_tokenizer()
    text = '\n'.join(' '.join(f'p{i}w{j}' for j in range(12)) + '.' for i in range(10))
    ps = split_cap(text, 30, tok)
    assert len(ps) > 1 and all(p in text for p in ps)
    assert all(len(tok.encode(p)) <= 30 for p in ps)
    assert split_cap('a b c', 30, tok) == ['a b c']
