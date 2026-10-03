"""index/chunk.py on synthetic docs (whitespace 'tokens' so sizes are exact)."""
from r2ai.paths import ROOT

from r2ai.index.chunk import chunk_paragraphs, doc_layout


def ntok(s):
    return len(s.split())


def para(i, n):
    return ' '.join(f'p{i}w{j}.' if j % 4 == 3 else f'p{i}w{j}' for j in range(n))


def build(sizes, question=None, answer_idx=()):
    paragraphs = [para(i, n) for i, n in enumerate(sizes)]
    r = {'title': 'Tiêu đề?', 'question': question, 'answer': '\n\n'.join(paragraphs[i] for i in answer_idx) or None,
         'body': '\n\n'.join(paragraphs), 'paragraphs': paragraphs}
    return doc_layout(r)


def run(sizes, target, **kw):
    doc_text, spans, paras = build(sizes, **kw)
    pt = [ntok(doc_text[s:e]) for _, s, e in paras]
    return doc_text, spans, chunk_paragraphs(doc_text, paras, pt, target, ntok)


def test_verbatim_offsets_and_title_span():
    doc_text, spans, chunks = run([5, 5, 20, 3], 10)
    assert spans == [('title', 0, len('Tiêu đề?'))]
    for _, s, e in chunks:
        assert doc_text[s:e].strip() and doc_text[s:e] == doc_text[s:e].strip()


def test_one_paragraph_overlap_and_target():
    doc_text, _, chunks = run([5, 5, 5, 5], 10)
    texts = [doc_text[s:e] for _, s, e in chunks]
    assert len(texts) == 3                       # [p0 p1] [p1 p2] [p2 p3]
    assert texts[0].split('\n\n')[1] == texts[1].split('\n\n')[0]
    assert all(ntok(t) <= 10 for t in texts)


def test_long_paragraph_split_at_sentence_end_only():
    doc_text, _, chunks = run([40], 10)          # 40 > 1.5 * 10 -> split; sentences of 4 words
    texts = [doc_text[s:e] for _, s, e in chunks]
    assert len(texts) > 1 and all(t.endswith('.') for t in texts) and all(ntok(t) <= 10 for t in texts)


def test_paragraph_between_target_and_1_5x_kept_whole():
    doc_text, _, chunks = run([14], 10)
    assert len(chunks) == 1 and ntok(doc_text[chunks[0][1]:chunks[0][2]]) == 14


def test_answer_field_and_no_cross_field_chunk():
    _, _, chunks = run([3, 3, 3, 3], 10, answer_idx=(2, 3))
    fields = [f for f, _, _ in chunks]
    assert fields == ['body', 'answer']


def test_question_not_in_body_gets_own_span():
    doc_text, spans, _ = run([5], 10, question='Câu hỏi riêng?')
    f, s, e = spans[1]
    assert f == 'question' and doc_text[s:e] == 'Câu hỏi riêng?'
