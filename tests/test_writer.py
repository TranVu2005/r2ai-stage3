from r2ai.paths import ROOT

import json

from r2ai.eval.validate import check_format
from r2ai.submit.writer import write_submission

ROWS = [
    {'id': 3, 'relevant_docs': [1], 'relevant_chunks': [{'doc_id': 1, 'chunk_text': '中文 文本\n第二行'}]},
    {'id': 1, 'relevant_docs': [2, 3], 'relevant_chunks': [{'doc_id': 2, 'chunk_text': 'Điều 5:\n"trích dẫn" khoản 1\n\nHết'}]},
    {'id': 2, 'relevant_docs': [], 'relevant_chunks': []},
]


def test_write_roundtrip_and_lines(tmp_path):
    p = tmp_path / 's.json'
    write_submission(ROWS, p)
    raw = p.read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf') and b'\r' not in raw
    text = raw.decode('utf-8')
    lines = text.split('\n')
    assert lines.pop() == ''
    assert len(lines) == len(ROWS) + 2 and lines[0] == '[' and lines[-1] == ']'
    assert all(l.endswith(',') for l in lines[1:-2]) and not lines[-2].endswith(',')
    assert json.loads(text) == sorted(ROWS, key=lambda r: r['id'])
    assert [json.loads(l.rstrip(','))['id'] for l in lines[1:-1]] == [1, 2, 3]
    assert '中文' in text and '\\n' in lines[3]
    assert check_format(text, 3) == []
    assert check_format(text.replace(',\n', '\n', 1), 3)
