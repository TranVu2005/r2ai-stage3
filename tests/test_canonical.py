import pytest

from r2ai.canonical import guard, versions
from r2ai.index.chunk import doc_layout


ROWS = [
    {'title': ' T ', 'question': 'Q?', 'answer': 'A.', 'body': 'para one\npara two', 'paragraphs': ['para one', 'para two']},
    {'title': '', 'question': 'Q?', 'answer': '', 'body': 'x Q? y', 'paragraphs': ['x Q? y']},
    {'title': None, 'question': None, 'answer': None, 'body': 'only', 'paragraphs': ['only']},
]


@pytest.mark.parametrize('row', ROWS)
def test_doc_text_matches_chunker_layout(row):
    text, body = versions.doc_text_of(row)
    assert text == doc_layout({**row, 'body': row['body'] or ''})[0]
    assert body == row['body']


def test_body_of_doc_text_strips_title_block():
    assert versions.body_of_doc_text('T\n\nbody', 'T') == 'body'
    assert versions.body_of_doc_text('body', 'T') == 'body'
    assert versions.body_of_doc_text('T\n\nbody', None) == 'T\n\nbody'


def test_quality_counts_code_points_after_ws_collapse_and_han():
    assert versions.quality('  a \n\n b  ') == (3, 0)
    assert versions.quality('中文 abc') == (6, 2)


def test_guard_refuses_outside_and_existing(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    monkeypatch.setattr(guard, 'ALLOWED_ROOTS', (root,))
    with pytest.raises(ValueError):
        guard.guarded(tmp_path / 'elsewhere' / 'x', roots=(root,))
    p = guard.atomic_write_bytes(root / 'a.txt', b'1', roots=(root,))
    assert p.read_bytes() == b'1'
    with pytest.raises(FileExistsError):
        guard.atomic_write_bytes(root / 'a.txt', b'2', roots=(root,))
    assert p.read_bytes() == b'1'


def test_test_paths_are_never_versions():
    from r2ai.canonical.select import is_test_path
    assert is_test_path(r'D:\GitHub\r2ai-stage3\out\pytest-reranker-s15\x\docs_vi\p0.parquet')
    assert is_test_path(r'D:\w\out\runs\zh\test_bad_candidates_rejected_n0\chunks\docs.parquet')
    assert is_test_path(r'D:\w\out\diag\synthetic-test-green\a.parquet')
    assert not is_test_path(r'D:\GitHub\r2ai-stage3\data\docs_vi\thanhnien.vn__2026.parquet')
    assert not is_test_path(r'D:\GitHub\r2ai-stage3\data\chunks_zh_v2\seg00109-v2n-003-ask.39.net\docs.parquet')
