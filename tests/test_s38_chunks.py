"""S38: paragraph rebuild from canonical chunks, structure flag, chunk choice and row rewrite."""
import numpy as np
import pandas as pd

from r2ai.index.chunk import chunk_paragraphs, doc_layout
from r2ai.s38.build import replace_chunks
from r2ai.s38.score import choose, select_variant
from r2ai.s38.sources import doc_pieces, drop_blobs, has_structure, rebuild_paragraphs


def ntok(s):
    return len(s.split())


def words(n, p):
    return ' '.join(f'{p}{i}' for i in range(n))


def canonical(row, target):
    doc_text, spans, paras = doc_layout(row)
    ptoks = [ntok(doc_text[s:e]) for _, s, e in paras]
    pieces = spans + chunk_paragraphs(doc_text, paras, ptoks, target, ntok)
    pieces = drop_blobs(doc_text, pieces, [ntok(doc_text[s:e]) for _, s, e in pieces], target)[0]
    return doc_text, paras, pieces


def qa_row():
    q = 'Bác sĩ cho hỏi ' + words(12, 'q') + '?'
    a = 'Chào bạn. ' + words(40, 'a') + '.'
    paras = [q, a, 'Trả lời tiếp: ' + words(30, 'b') + '.', words(70, 'c') + '.', words(5, 'd')]
    return {'title': 'Tiêu đề ' + words(4, 't'), 'question': q, 'answer': '\n\n'.join(paras[1:3]),
            'body': '\n\n'.join(paras), 'paragraphs': paras}


def plain_row():
    paras = [words(n, f'p{k}_') + '.' for k, n in enumerate((30, 50, 20, 90, 10, 400, 25))]
    return {'title': 'T ' + words(3, 't'), 'question': None, 'answer': None, 'body': '\n\n'.join(paras), 'paragraphs': paras}


def test_rebuild_paragraphs_gives_the_extractor_paragraphs_and_fields():
    for row in (qa_row(), plain_row()):
        doc_text, paras, pieces = canonical(row, 64)
        assert rebuild_paragraphs(doc_text, pieces) == paras


def test_rebuilt_paragraphs_reproduce_chunks_at_other_targets():
    for row in (qa_row(), plain_row()):
        for target in (32, 64, 128, 256):
            doc_text, paras, pieces = canonical(row, 64)          # canonical chunks made at another target
            rebuilt = rebuild_paragraphs(doc_text, pieces)
            ptoks = [ntok(doc_text[s:e]) for _, s, e in rebuilt]
            assert chunk_paragraphs(doc_text, rebuilt, ptoks, target, ntok) == chunk_paragraphs(doc_text, paras, ptoks, target, ntok)


def test_doc_pieces_gate_t128_and_sec():
    row = qa_row()
    doc_text, paras, pieces = canonical(row, 256)
    chunks = [(k, f, s, e) for k, (f, s, e) in enumerate(pieces)]
    res = doc_pieces(doc_text, chunks, lambda ts: [ntok(t) for t in ts], ntok, ())
    assert res['gate']
    assert res['structured']                                   # question field in the body
    assert all(f in ('answer', 'body') for _, f, _, _ in res['t256'] + res['t128'] + res['sec'])
    assert all(doc_text[s:e].strip() for _, _, s, e in res['t128'] + res['sec'])
    t256_spans = {(s, e): cid for cid, _, s, e in res['t256']}
    assert all(cid == t256_spans.get((s, e), -1) for cid, _, s, e in res['t128'] + res['sec'])
    assert len(res['t128']) >= len(res['t256'])


def test_doc_without_structure_takes_t256_as_sec():
    row = plain_row()
    doc_text, paras, pieces = canonical(row, 256)
    chunks = [(k, f, s, e) for k, (f, s, e) in enumerate(pieces)]
    res = doc_pieces(doc_text, chunks, lambda ts: [ntok(t) for t in ts], ntok, ())
    assert res['gate'] and not res['structured']
    assert res['sec'] == res['t256']


def test_has_structure_rules():
    text = 'Nguyên liệu:\n\nThịt bò 200 g.\n\n- muối\n\n- tiêu'
    paras = []
    pos = 0
    for p in text.split('\n\n'):
        paras.append(('body', pos, pos + len(p)))
        pos += len(p) + 2
    assert has_structure(text, paras)
    assert has_structure(text, paras[1:2] + paras[2:3])        # list item
    assert not has_structure(text, paras[1:2])                  # a plain sentence
    assert has_structure(text, [('question', *paras[1][1:])])
    assert has_structure(text, paras[1:2], headings=['Thịt bò 200 g.'])


def test_choose_best_and_second_ratio():
    assert choose([0.2, 0.9, 0.5], [0, 1, 2], 1) == [1]
    assert choose([0.2, 0.9, 0.45], [0, 1, 2], 2) == [1, 2]         # exactly 0.5 x 0.9
    assert choose([0.2, 0.9, 0.44], [0, 1, 2], 2) == [1]
    assert choose([0.7, 0.7], [3, 1], 2) == [1, 0]                  # tie: lower piece index first
    assert choose([], [], 2) == []


def test_select_variant_matches_choose():
    rng = np.random.default_rng(38)
    rows = []
    for q in range(30):
        for d in range(3):
            n = int(rng.integers(1, 6))
            for i in range(n):
                rows.append({'query_id': q, 'doc_id': d, 'idx': i, 'score': float(rng.choice([0.1, 0.3, 0.6, 0.9]))})
    cands = pd.DataFrame(rows)
    for n in (1, 2):
        got = select_variant(cands, n)
        for (q, d), g in cands.groupby(['query_id', 'doc_id']):
            want = g.idx.to_numpy()[choose(g.score.to_numpy(), g.idx.to_numpy(), n)].tolist()
            sub = got[(got.query_id == q) & (got.doc_id == d)].sort_values('rank')
            assert sub.idx.tolist() == want and sub['rank'].tolist() == list(range(1, len(want) + 1))


def test_select_variant_skips_normalised_duplicates():
    cands = pd.DataFrame({'query_id': [1, 1, 1], 'doc_id': [7, 7, 7], 'idx': [0, 1, 2], 'score': [0.9, 0.8, 0.6],
                          'norm_sha': ['a', 'a', 'b']})
    got = select_variant(cands, 2)
    assert got.idx.tolist() == [0, 2] and got['rank'].tolist() == [1, 2]


def test_replace_chunks_keeps_docs_and_order():
    row = {'id': 5, 'relevant_docs': [9, 3, 7], 'relevant_chunks': [{'doc_id': 3, 'chunk_text': 'whole 3'},
                                                                    {'doc_id': 9, 'chunk_text': 'whole 9'}]}
    out = replace_chunks(row, {(5, 3): ['a', 'b'], (5, 9): ['c']})
    assert out == {'id': 5, 'relevant_docs': [9, 3, 7], 'relevant_chunks': [
        {'doc_id': 3, 'chunk_text': 'a'}, {'doc_id': 3, 'chunk_text': 'b'}, {'doc_id': 9, 'chunk_text': 'c'}]}
    assert replace_chunks(row, {(5, 9): ['c']})['relevant_chunks'] == [{'doc_id': 3, 'chunk_text': 'whole 3'},
                                                                   {'doc_id': 9, 'chunk_text': 'c'}]


# ---- CH-win ---------------------------------------------------------------------------------------------------
import re as _re

from r2ai.s38.window import boundaries, pieces_for, prefix, window


def ws_offsets(text):
    return [(m.start(), m.end()) for m in _re.finditer(r'\S+', text)]


def sentences(n, words_per=10):
    return ' '.join(' '.join(f's{k}w{i}' for i in range(words_per - 1)) + f' s{k}end.' for k in range(n))


def test_window_whole_body_when_short():
    text = sentences(3)
    res = window(text, ws_offsets(text), (0, 5), 30)
    assert res['mode'] == 'whole' and (res['start'], res['end']) == (0, len(text))


def test_window_centred_on_chunk_cut_on_sentences_and_within_width():
    text = sentences(40)                                  # 400 tokens, sentences of 10
    offs = ws_offsets(text)
    cs = text.index('s20w0')
    ce = text.index('s21end.') + len('s21end.')           # chunk = sentences 20..21 (20 tokens)
    res = window(text, offs, (cs, ce), 64)
    piece = text[res['start']:res['end']]
    assert res['mode'] == 'window' and len(piece.split()) <= 64
    assert res['start'] <= cs and res['end'] >= ce       # chunk kept whole
    assert piece.startswith('s') and 'w0' in piece.split()[0] and piece.endswith('end.')
    assert res['snap_start'] and res['snap_end']


def test_window_clamped_at_body_edges():
    text = sentences(40)
    offs = ws_offsets(text)
    res = window(text, offs, (0, text.index('s0end.') + 6), 64)
    assert res['start'] == 0 and len(text[res['start']:res['end']].split()) <= 64
    res = window(text, offs, (text.index('s39w0'), len(text)), 64)
    assert res['end'] == len(text)


def test_window_without_boundaries_keeps_token_cut():
    text = ' '.join(f'w{i}' for i in range(500))
    offs = ws_offsets(text)
    cs = text.index('w250')
    res = window(text, offs, (cs, cs + 4), 100)
    piece = text[res['start']:res['end']]
    assert len(piece.split()) == 100 and 'w250' in piece.split()
    assert not res['snap_start'] and not res['snap_end']


def test_boundaries_zh_and_breaks():
    s, e = boundaries('甲乙。丙丁！\n戊')
    assert 3 in s and 3 in e and 7 in s and 6 in e


def test_prefix_cuts_head_on_sentence_end():
    text = sentences(40)
    offs = ws_offsets(text)
    res = prefix(text, offs, 64)
    piece = text[res['start']:res['end']]
    assert res['start'] == 0 and piece.endswith('end.') and 48 <= len(piece.split()) <= 64 and res['snap_end']
    assert prefix(text, offs, 400)['mode'] == 'whole'


def test_pieces_for_rules():
    text = sentences(40)
    offs = ws_offsets(text)
    chunk = (text.index('s20w0'), text.index('s21end.') + 7)
    assert pieces_for('window', 64, 400, text, offs, chunk) == [None]          # body <= threshold
    one = pieces_for('window', 64, 64, text, offs, chunk)
    assert len(one) == 1 and one[0][0] == 'window'
    two = pieces_for('full+window', 64, 150, text, offs, chunk)
    assert two[0] is None and two[1][0] == 'window' and (two[1][1], two[1][2]) != (0, len(text))
    assert pieces_for('full+window', 64, 400, text, offs, chunk) == [None]


def test_replace_chunks_none_keeps_j3_chunk():
    row = {'id': 5, 'relevant_docs': [9, 3], 'relevant_chunks': [{'doc_id': 3, 'chunk_text': 'whole 3'}]}
    out = replace_chunks(row, {(5, 3): [None, 'w']})
    assert out['relevant_chunks'] == [{'doc_id': 3, 'chunk_text': 'whole 3'}, {'doc_id': 3, 'chunk_text': 'w'}]


# ---- CH-clean -------------------------------------------------------------------------------------------------
from r2ai.s38 import clean as CL


def _body(k, extra_tail=True):
    head = '\n\n'.join(f'Đoạn nội dung {"abcdefgh"[i]} của bài {"klmnopqrstuv"[k]} khá dài để làm thân bài.'
                       for i in range(8))                          # lines differ by letters (digits are masked)
    tail = '\n\nHotline: 024 3872 3872 – cơ sở 12\n\nCập nhật lần cuối: 05:01 10/07/2026' if extra_tail else ''
    return head + tail


def test_line_key_masks_digits_and_case():
    assert CL.line_key('  Cập nhật lần cuối: 05:01 10/07/2026 ') == CL.line_key('cập nhật  lần cuối: 11:22 01/01/2025')


def test_discover_needs_doc_share_and_tail_position():
    bodies = {k: _body(k, extra_tail=k < 4) for k in range(10)}       # tail lines in 40% of docs
    keys = {m['key'] for m in CL.discover(bodies)}
    assert CL.line_key('Hotline: 024 3872 3872 – cơ sở 12') in keys
    bodies = {k: _body(k, extra_tail=k < 2) for k in range(10)}       # 20% < 30%
    assert not CL.discover(bodies)
    early = {k: 'Hotline: 024 3872 3872 – cơ sở 12\n\n' + _body(k, extra_tail=False) for k in range(10)}
    assert not CL.discover(early)                                      # not in the last 30%


def test_cut_point_is_first_marker_in_tail():
    body = _body(1)
    markers = {CL.line_key('Hotline: 024 3872 3872 – cơ sở 12'), CL.line_key('Cập nhật lần cuối: 05:01 10/07/2026')}
    at = CL.cut_point(body, markers)
    assert body[:at].rstrip().endswith('làm thân bài.') and 'Hotline' not in body[:at]
    assert CL.cut_point(_body(1, extra_tail=False), markers) is None
    assert CL.cut_point('Hotline: 024 3872 3872 – cơ sở 12\n' + _body(2, extra_tail=False), markers) is None


def test_clean_cut_base64_anywhere_and_markers():
    blob = 'QUJD' * 80
    body = '{title}\n\n{publish}\n\n' + blob + '\n\nÝ kiến của bạn sẽ được biên tập trước khi đăng.'
    assert CL.base64_start(body) == body.index(blob)
    assert CL.clean_cut(body, set(), True) == '{title}\n\n{publish}'
    assert CL.clean_cut(body, set(), False) is None
    assert CL.base64_start('ngắn QUJD\n' + 'QUJD' * 10) is None          # < 200 chars, or not a whole line


# ---- compose --------------------------------------------------------------------------------------------------
from r2ai.s38 import compose as CO


def test_check_docsel_counts_differences():
    rows = [{'id': 1, 'relevant_docs': [5, 6, 7], 'relevant_chunks': [{'doc_id': 5, 'chunk_text': 'a'}]},
            {'id': 2, 'relevant_docs': [8], 'relevant_chunks': []}]
    ds = pd.DataFrame({'query_id': [1, 1, 1, 2], 'doc_id': [5, 6, 7, 8], 'rank': [1, 2, 3, 1],
                       'chunk': [True, False, False, False]})
    rep = CO.check_docsel(rows, ds)
    assert rep['queries_docs_differ'] == 0 and rep['carriers_only_base'] == 0 and rep['carriers_only_docsel'] == 0
    ds2 = ds.assign(rank=[2, 1, 3, 1], chunk=[True, True, False, False])
    rep = CO.check_docsel(rows, ds2)
    assert rep['queries_docs_differ'] == 1 and rep['carriers_only_docsel'] == 1


def test_estimate_thresholds_fits_budget():
    frame = pd.DataFrame({'rank': [1] * 5 + [2] * 5, 'body_tokens': [0] * 5 + [1600, 1700, 1800, 1900, 2000]})
    est = CO.estimate_thresholds(frame, CO.CEILING - int(2.5 * CO.RATE_FULL))
    assert est['k_estimate'] == 2 and est['t_estimate'] == 1800              # bodies 1900, 2000 keep the window
    assert CO.estimate_thresholds(frame, CO.CEILING - 10 ** 6)['t_estimate'] == CO.FULL_MIN_T
