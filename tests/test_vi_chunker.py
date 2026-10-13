from r2ai.vi_quality.chunker import classify, structured_chunks


def make(paras_text, fields=None):
    doc = '\n\n'.join(paras_text)
    spans, pos = [], 0
    for i, t in enumerate(paras_text):
        spans.append(((fields or {}).get(i, 'body'), pos, pos + len(t)))
        pos += len(t) + 2
    return doc, spans


def run(paras_text, target=20, fields=None, headings=(), min_tokens=None):
    doc, spans = make(paras_text, fields)
    ptoks = [len(t.split()) for t in paras_text]
    ntok = lambda s: len(s.split())          # noqa: E731
    out = structured_chunks(doc, spans, ptoks, target, ntok, headings=headings, min_tokens=min_tokens)
    for _, s, e in out:
        assert doc[s:e] == doc[s:e].strip()
    return [doc[s:e] for _, s, e in out], out


def words(n, w='x'):
    return ' '.join(f'{w}{i}' for i in range(n)) + '.'


def test_classify():
    hs = {'Triệu chứng'}
    assert classify('Triệu chứng', hs) == 'heading'
    assert classify('Nguyên liệu:', set()) == 'heading'
    assert classify('2.1. Chẩn đoán', set()) == 'heading'
    assert classify('- Uống 2 viên/ngày', set()) == 'item'
    assert classify('a | b | c', set()) == 'row'
    assert classify('Đây là một câu hoàn chỉnh.', set()) == 'para'
    assert classify('10. Chân đau khó đi bộ.', set()) == 'item'


def test_heading_stays_with_text_below_and_never_ends_a_chunk():
    chunks, _ = run([words(15), 'Triệu chứng', words(15), words(15)], target=20, headings=['Triệu chứng'])
    assert any(c.startswith('Triệu chứng') and 'x0' in c for c in chunks)
    assert not any(c.endswith('Triệu chứng') for c in chunks)


def test_small_sections_merge_only_while_chunk_is_small():
    chunks, _ = run(['A:', 'ab cd', 'B:', 'ef gh', 'C:', words(12)], target=20, min_tokens=5)
    assert len(chunks) == 2 and chunks[0].startswith('A:') and 'B:' in chunks[0]
    assert chunks[1].startswith('C:')


def test_qa_pair_in_one_chunk_when_it_fits():
    chunks, out = run(['Hỏi gì đó?', 'Trả lời ngắn không.', 'Hỏi tiếp?', 'Có thể.'], target=12,
                      fields={0: 'question', 1: 'answer', 2: 'question', 3: 'answer'}, min_tokens=1)
    assert chunks == ['Hỏi gì đó?\n\nTrả lời ngắn không.', 'Hỏi tiếp?\n\nCó thể.']
    assert [f for f, _, _ in out] == ['answer', 'answer']


def test_list_run_kept_whole_when_it_fits_and_split_at_items_when_not():
    items = [f'- mục {i} có hai chữ' for i in range(4)]
    chunks, _ = run(['Cách làm:'] + items, target=30)
    assert chunks == ['Cách làm:\n\n' + '\n\n'.join(items)]
    chunks, _ = run(['Cách làm:'] + items, target=10)
    assert all(c.count('- mục') >= 1 for c in chunks[1:]) and '\n\n'.join(chunks).count('- mục') == 4


def test_trailing_heading_attaches_to_previous_chunk():
    chunks, _ = run([words(15), 'Xem thêm:'], target=20)
    assert chunks == [words(15) + '\n\nXem thêm:']


def test_verbatim_numbers_and_negation_and_no_overlap():
    texts = ['Không dùng quá 500 mg/ngày.', 'Không được tiêm khi sốt > 38,5 °C.', words(30)]
    chunks, out = run(texts, target=12)
    joined = '\n\n'.join(chunks)
    assert 'Không dùng quá 500 mg/ngày.' in joined and '> 38,5 °C' in joined
    assert all(out[i][2] <= out[i + 1][1] for i in range(len(out) - 1))
