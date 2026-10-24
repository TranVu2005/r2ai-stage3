import json

import polars as pl
import pytest

from r2ai.s37 import export as E


class FakeTok:
    """One token per character: enough to exercise truncate_document without the Qwen3 files."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ''.join(chr(i) for i in ids)


@pytest.fixture
def small(tmp_path, monkeypatch):
    run, store = tmp_path / 'run', tmp_path / 'store'
    seg = store / 'seg37001-test'
    seg.mkdir(parents=True)
    run.mkdir()
    long_doc = 'x' * 600
    pl.DataFrame({'doc_id': [10, 11, 12], 'doc_text': ['title\n\nquestion\n\nanswer one', long_doc, 'only question']}).write_parquet(seg / 'docs.parquet')
    pl.DataFrame({'chunk_id': [1, 2, 3], 'field': ['answer', 'answer', 'question'],
                  'text': ['answer one', 'y' * 20, 'only question']}).write_parquet(seg / 'chunks_t256.parquet')
    # pair 1000000 -> doc 10 (chunks 1, 2), pair 1000001 -> doc 11 (chunk 2), pair 1000002 -> doc 12 (no body/answer chunk: fallback)
    pl.DataFrame({'pair_id': [1000000, 1000000, 1000001], 'kind': ['t256'] * 3, 'rank': [1, 2, 1], 'chunk_id': [1, 2, 2],
                  'chunk_cosine': [0.9, 0.5, 0.4]}).write_parquet(seg / 'cmax_top3.parquet')
    pl.DataFrame({'query_id': [7, 7, 8], 'doc_id': [10, 11, 12], 'cosine_rank': [1, 2, 1], 'cosine_score': [0.9, 0.8, 0.7],
                  'in_pool13': [True, True, False], 'lang': ['zh'] * 3, 'pair_id': [1000000, 1000001, 1000002], 'shard': [None] * 3,
                  'in_top211': [True, False, True], 'entry_rank': [1, 2, 1], 'segment': ['seg37001-test'] * 3,
                  'batch': ['b0001', 'b0001', 'b0002']}).write_parquet(run / 'pairs_new.parquet')
    pl.DataFrame({'id': [7, 8], 'query': ['câu hỏi 7', 'câu hỏi 8']}).write_parquet(tmp_path / 'query.parquet')
    judge = tmp_path / 'judge_score.py'
    judge.write_text('# stand-in scorer\n', encoding='utf-8')
    monkeypatch.setattr(E, 'RUN', run)
    monkeypatch.setattr(E, 'STORE', store)
    monkeypatch.setattr(E, 'QUERY', tmp_path / 'query.parquet')
    monkeypatch.setattr(E, 'JUDGE_SCORE', judge)
    monkeypatch.setattr(E, 'JUDGE_SHA', E.sha256_file(judge))
    monkeypatch.setattr(E, 'tokenizer', lambda: FakeTok())
    monkeypatch.setattr(E, 'assert_writable', lambda p: None)
    return run


def test_truncate_document_cuts_to_limit():
    text, n = E.truncate_document(FakeTok(), 'x' * 600)
    assert n == 512 and len(text) == 512
    assert E.truncate_document(FakeTok(), 'short') == ('short', 5)


def test_kaggle4b_keeps_registry_ids_and_batch_filter(small):
    man = E.export_kaggle4b('t', ['b0001'])
    out = small / 'kaggle-4b' / 't'
    shard = pl.read_parquet(out / 'shards' / E.SHARD_NAME)
    extra = pl.read_parquet(out / 'extra_pairs.parquet')
    assert man['pairs'] == 2 and man['batches'] == ['b0001'] and man['pair_id_range'] == [1000000, 1000001]
    assert shard['pair_id'].to_list() == extra['pair_id'].to_list() == [1000000, 1000001]
    assert shard['lang'].to_list() == ['zh', 'zh'] and shard['query_text'].to_list() == ['câu hỏi 7', 'câu hỏi 7']
    assert shard['doc_tokens'].to_list() == [len('title\n\nquestion\n\nanswer one'), 512]            # long doc cut to 512
    assert extra['free'].to_list() == [True, False] and extra['source'].to_list() == ['s37-b0001'] * 2
    assert man['shard_sha256'] == E.sha256_file(out / 'shards' / E.SHARD_NAME)


def test_cmax_round_trip_chunk_rows_fallback_and_reduce(small):
    man = E.export_cmax('rt', None)
    pool = small / 'cmax-rt' / 'pool'
    rows = pl.read_parquet(pool / 'shards' / E.SHARD_NAME)
    assert man['pairs'] == 3 and man['pairs_with_chunks'] == 2 and man['pairs_fallback_doc_text'] == 1 and man['rows'] == 4
    assert rows['source'].to_list().count('s37-chunk') == 3 and rows.filter(pl.col('source') == 's37-fallback-doc')['src_pair_id'].to_list() == [1000002]
    assert rows['pair_id'].min() >= E.CHUNK_ROW_BASE and rows['pair_id'].n_unique() == rows.height
    # fake scorer output: one P(yes) per row id
    by_row = {int(r): s for r, s in zip(rows['pair_id'], [0.2, 0.9, 0.4, 0.6])}
    scores = pl.DataFrame({'pair_id': list(by_row), 'score': list(by_row.values())})
    (small / 'cmax-rt' / 'gpu' / 'qwen3-reranker-0.6b').mkdir(parents=True)
    scores.write_parquet(small / 'cmax-rt' / 'gpu' / 'qwen3-reranker-0.6b' / 'scores-all.parquet')
    res = E.reduce_cmax(tag='rt')
    cm = pl.read_parquet(small / 'cmax_scores-rt.parquet').sort('pair_id')
    assert res['pairs'] == 3 and res['fallback'] == 1
    assert cm['pair_id'].to_list() == [1000000, 1000001, 1000002] and cm['n_chunks'].to_list() == [2, 1, 1]
    assert cm['cmax'].to_list() == pytest.approx([0.9, 0.4, 0.6]) and cm['fallback'].to_list() == [False, False, True]
    assert json.loads((pool / 'manifest.json').read_text('utf-8'))['rows'] == 4
