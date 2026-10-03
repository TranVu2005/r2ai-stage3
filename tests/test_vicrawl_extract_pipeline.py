from r2ai.paths import FIXTURES_DIR

import hashlib
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from vicrawl.extract_pipeline import ExtractPipeline, build_qa_report
from vicrawl.settings import Config
from vicrawl.shards import write_shard

FIX = FIXTURES_DIR
HTML = {
    'article': (FIX / 'vinmec.com__5f28d781.html').read_text(encoding='utf-8'),
    'qa': (FIX / 'vinmec.com__44ae4a45.html').read_text(encoding='utf-8'),
    'other': (FIX / 'medlatec.vn__f722bfdd.html').read_text(encoding='utf-8'),
}


def rec(url_norm, kind, fetched_at, status='ok', domain='vinmec.com', doc_ids=(1,)):
    return {'url_norm': url_norm, 'url': 'https://www.' + url_norm, 'final_url': 'https://www.' + url_norm, 'doc_ids': list(doc_ids), 'domain': domain,
            'status': status, 'reason': '', 'http_status': 200, 'attempts': 1, 'fetched_at': fetched_at, 'html': HTML[kind] if kind else None}


def cfg(tmp_path):
    p = tmp_path / 'd.yaml'
    p.write_text('domains:\n  vinmec.com: {extractor: vinmec}\n  medlatec.vn: {extractor: medlatec}\n', encoding='utf-8')
    return Config.load(p)


def pipeline(tmp_path):
    return ExtractPipeline(raw_dir=tmp_path / 'raw', docs_dir=tmp_path / 'docs', state_dir=tmp_path / 'state', cfg=cfg(tmp_path), tokenizer=lambda text: len(text.split()))


def table(tmp_path):
    return ds.dataset(str(tmp_path / 'docs'), format='parquet').to_table().to_pylist()


def test_extracts_columns_and_keeps_paragraph_boundaries(tmp_path):
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/vie/bai-viet/a-vi', 'article', 10.0, doc_ids=(5, 9)), rec('vinmec.com/vie/bai-viet/q-vi', 'qa', 11.0)])
    n = pipeline(tmp_path).run_once()
    assert n == 1
    rows = {r['url_norm']: r for r in table(tmp_path)}
    a = rows['vinmec.com/vie/bai-viet/a-vi']
    assert a['doc_ids'] == [5, 9] and a['domain'] == 'vinmec.com' and a['status'] == 'ok'
    assert a['body'] == '\n\n'.join(a['paragraphs']) and len(a['paragraphs']) > 10
    assert a['title'] == 'Các triệu chứng mọc răng khôn điển hình' and a['lang'] == 'vi'
    assert a['n_tokens_bge_m3'] == len(a['body'].split())
    assert a['text_sha1'] == hashlib.sha1(a['body'].encode('utf-8')).hexdigest()
    assert a['fetched_at'] == 10.0 and a['final_url'].startswith('https://')
    q = rows['vinmec.com/vie/bai-viet/q-vi']
    assert q['question'].startswith('Chào bác sĩ') and q['answer'].startswith('Chào bạn')
    assert set(a) >= {'doc_ids', 'url', 'final_url', 'domain', 'title', 'description', 'question', 'answer', 'body', 'paragraphs', 'lang', 'n_tokens_bge_m3', 'text_sha1', 'fetched_at'}


def test_only_new_shards_are_processed_incrementally(tmp_path):
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'article', 1.0)])
    p = pipeline(tmp_path)
    assert p.run_once() == 1
    assert p.run_once() == 0
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/b', 'article', 2.0)])
    assert pipeline(tmp_path).run_once() == 1           # fresh object, state comes from disk
    assert {r['url_norm'] for r in table(tmp_path)} == {'vinmec.com/a', 'vinmec.com/b'}


def test_duplicate_url_across_shards_keeps_latest(tmp_path):
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'article', 1.0), rec('vinmec.com/z', 'qa', 1.0)])
    pipeline(tmp_path).run_once()
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'qa', 5.0)])         # re-fetched after a crash, newer
    pipeline(tmp_path).run_once()
    rows = [r for r in table(tmp_path) if r['url_norm'] == 'vinmec.com/a']
    assert len(rows) == 1 and rows[0]['fetched_at'] == 5.0 and rows[0]['question']
    assert len(table(tmp_path)) == 2
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'article', 3.0)])    # older duplicate arriving later is ignored
    pipeline(tmp_path).run_once()
    rows = [r for r in table(tmp_path) if r['url_norm'] == 'vinmec.com/a']
    assert len(rows) == 1 and rows[0]['fetched_at'] == 5.0


def test_same_content_docs_are_kept_and_share_sha1(tmp_path):
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'article', 1.0), rec('vinmec.com/b', 'article', 2.0, doc_ids=(7,))])
    pipeline(tmp_path).run_once()
    rows = table(tmp_path)
    assert len(rows) == 2 and len({r['text_sha1'] for r in rows}) == 1


def test_non_ok_records_are_skipped_and_thin_short_pages_flagged(tmp_path):
    short = rec('vinmec.com/s', 'article', 1.0)
    short['html'] = '<html><body><div id="main-article"><p>Quá ngắn.</p></div></body></html>'
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/e', None, 1.0, status='network_error'), rec('vinmec.com/n', 'article', 1.0, status='soft404_or_home'), short])
    pipeline(tmp_path).run_once()
    rows = table(tmp_path)
    assert [r['url_norm'] for r in rows] == ['vinmec.com/s'] and rows[0]['status'] == 'thin'


def test_parquet_written_atomically_no_tmp_left(tmp_path):
    write_shard(tmp_path / 'raw', 'vinmec.com', [rec('vinmec.com/a', 'article', 1.0)])
    pipeline(tmp_path).run_once()
    assert not list((tmp_path / 'docs').rglob('*.tmp'))
    assert all(pq.ParquetFile(p).metadata.num_rows == 1 for p in (tmp_path / 'docs').rglob('*.parquet'))


def test_qa_report_lists_five_docs_with_head_tail_and_repeated_paragraphs(tmp_path):
    recs = [rec(f'vinmec.com/a{i}', 'article' if i % 2 else 'qa', float(i)) for i in range(12)]
    write_shard(tmp_path / 'raw', 'vinmec.com', recs)
    out = tmp_path / 'out' / 'qa_extract' / 'vinmec.com.md'
    build_qa_report('vinmec.com', raw_dir=tmp_path / 'raw', out_path=out, cfg=cfg(tmp_path), n_docs=300, n_show=5)
    text = out.read_text(encoding='utf-8')
    assert text.count('\n## Doc ') == 5
    assert 'URL:' in text and 'Title:' in text and 'Đầu' in text and 'Cuối' in text and 'Question' in text
    assert 'Đoạn lặp' in text
