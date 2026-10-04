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


class WordTokenizer:
    """Stand-in for the BGE-M3 tokenizer: one token per whitespace-separated word."""

    def __call__(self, x, add_special_tokens=False):
        if isinstance(x, str):
            return {'input_ids': list(range(len(x.split())))}
        return {'input_ids': [list(range(len(s.split()))) for s in x]}


def write_docs(path, docs):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(docs), path)


def doc(doc_id, sizes, status='ok', n_tokens=100, question=None):
    paragraphs = [para(i, n) for i, n in enumerate(sizes)]
    return {'doc_ids': [doc_id, doc_id + 1000], 'url': f'https://x.vn/{doc_id}', 'domain': 'x.vn', 'title': f'T{doc_id}',
            'question': question, 'answer': None, 'body': '\n\n'.join(paragraphs), 'paragraphs': paragraphs,
            'status': status, 'n_tokens_bge_m3': n_tokens}


def run_main(monkeypatch, docs_dir, out_dir, batch_docs):
    import r2ai.gold_check.gold_check_common as gcc
    from r2ai.index.chunk import main
    monkeypatch.setattr(gcc, 'load_tokenizer', lambda *a, **k: WordTokenizer())
    assert main(['--docs-dir', str(docs_dir), '--out-dir', str(out_dir), '--targets', '10,20',
                 '--batch-docs', str(batch_docs)]) == 0


def test_batched_output_matches_single_batch(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq
    docs_dir = tmp_path / 'docs'
    docs_dir.mkdir()
    # doc_ids out of order across files; one file has an all-null question column (null type in parquet)
    write_docs(docs_dir / 'a.parquet', [doc(9, [5, 30]), doc(2, [12, 3, 3], question='Hỏi?'), doc(7, [4], status='thin')])
    write_docs(docs_dir / 'b.parquet', [doc(5, [6, 6, 6]), doc(1, [3], n_tokens=10), doc(3, [25, 2])])
    write_docs(docs_dir / 'c.parquet', [doc(8, [2, 2, 2, 2, 2]), doc(4, [40])])
    for bd in (1, 2, 100):
        run_main(monkeypatch, docs_dir, tmp_path / f'out{bd}', bd)
    for name in ('docs.parquet', 'chunks_t10.parquet', 'chunks_t20.parquet', 'chunk_report.json'):
        ref = tmp_path / 'out100' / name
        for bd in (1, 2):
            other = tmp_path / f'out{bd}' / name
            if name.endswith('.json'):
                assert json.loads(other.read_text()) == json.loads(ref.read_text())
            else:
                assert pq.read_table(other).equals(pq.read_table(ref)), (bd, name)
    out = tmp_path / 'out1'
    assert pq.read_table(out / 'docs.parquet')['doc_id'].to_pylist() == [2, 3, 4, 5, 8, 9]   # filtered, doc_id order
    chunks = pq.read_table(out / 'chunks_t10.parquet')
    assert chunks['chunk_id'].to_pylist() == list(range(chunks.num_rows))
    assert sorted(p.name for p in out.iterdir()) == ['chunk_report.json', 'chunks_t10.parquet', 'chunks_t20.parquet',
                                                     'docs.parquet']                             # no tmp / partial left


def test_failed_run_leaves_no_partial_outputs(tmp_path, monkeypatch):
    import pytest
    import r2ai.index.chunk as chunk
    docs_dir = tmp_path / 'docs'
    docs_dir.mkdir()
    write_docs(docs_dir / 'a.parquet', [doc(1, [5, 5]), doc(2, [5])])
    monkeypatch.setattr(chunk, 'chunk_paragraphs', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError, match='boom'):
        run_main(monkeypatch, docs_dir, tmp_path / 'out', 1)
    assert list((tmp_path / 'out').iterdir()) == []
