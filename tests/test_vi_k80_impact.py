import json

import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.vi_quality import k80_impact

WIDGET = '\n\nBình luận:\n\nBạn nghĩ gì về nội dung này?\n\nBạn cần đăng nhập để gửi bình luận.\n\nGửi bình luận'


def test_impact_counts_removed_and_stripped(tmp_path):
    docs = pa.table({'doc_id': [1, 2, 3], 'doc_ids_group': [[1], [2, 20], [3]],
                     'domain': ['laodong.vn', 'suckhoecongdongonline.vn', 'vinmec.com']})
    pq.write_table(docs, tmp_path / 'docs.parquet')
    (tmp_path / 'drop.tsv').write_text('doc_id\tdomain\treason\turl\n2\tsuckhoecongdongonline.vn\tlottery_result\tu\n', encoding='utf-8')
    row = {'id': 1, 'relevant_docs': [1, 20, 3, 99], 'relevant_chunks': [
        {'doc_id': 1, 'chunk_text': 'Bài viết.' + WIDGET}, {'doc_id': 2, 'chunk_text': 'XS 1/1'},
        {'doc_id': 3, 'chunk_text': 'Đau ngực'}, {'doc_id': 99, 'chunk_text': 'cc'}]}
    (tmp_path / 's.json').write_text('[\n' + json.dumps(row, ensure_ascii=False) + '\n]\n', encoding='utf-8')
    k80_impact.main(['--json', str(tmp_path / 's.json'), '--out-json', str(tmp_path / 'c.json'), '--report', str(tmp_path / 'r.json'),
                     '--docs', str(tmp_path / 'docs.parquet'), '--drop-tsv', str(tmp_path / 'drop.tsv')])
    rep = json.loads((tmp_path / 'r.json').read_text(encoding='utf-8'))
    assert rep['docs'] == {'total': 4, 'removed': 1, 'outside_index': 1}
    assert rep['chunks'] == {'total': 4, 'outside_index': 1, 'removed': 1, 'stripped': 1}
    assert rep['bytes']['stripped'] == len(WIDGET.encode('utf-8'))
    clean = json.loads((tmp_path / 'c.json').read_text(encoding='utf-8'))[0]
    assert clean['relevant_docs'] == [1, 3, 99]
    assert [c['chunk_text'] for c in clean['relevant_chunks']] == ['Bài viết.', 'Đau ngực', 'cc']
