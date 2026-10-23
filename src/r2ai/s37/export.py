"""S37 scoring inputs for the new 120ask pairs (no scoring is started here).

    python -m r2ai.s37.export kaggle4b --tag s37-b001-003      # Qwen3-Reranker-4B subset for Kaggle (S31 pool-extra layout)
    python -m r2ai.s37.export cmax                              # Qwen3-Reranker-0.6B chunk rows for C-max (S32 layout)
    python -m r2ai.s37.export reduce                            # C-max per pair from the collected chunk scores

Both exports read out/runs/s37/pairs_new.parquet (pair_id registry, >= 1,000,000) and the segments' docs/chunks.
Query text, prompt and truncation follow S29/S31/S32: doc text cut to 512 Qwen3 tokens with truncate_document
(tokenizer of Qwen3-Reranker-0.6B e61197ed, shared with the 4B), lang 'zh', instruction from judge_score.py.
"""
from __future__ import annotations

from r2ai.paths import DATA_DIR, RUNS_DIR, assert_writable

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from r2ai.s37.ingest import RUN, STORE, atomic_json, atomic_parquet, sha256_file

SNAP06 = DATA_DIR / 'hf_model_cache' / 'models--Qwen--Qwen3-Reranker-0.6B' / 'snapshots' / 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
REV4B = '22e683669bc0f0bd69640a1354a6d0aebcfeede5'
JUDGE_SCORE = RUNS_DIR / 's31-strong-judge' / 'pool-extra' / 'kaggle' / 'judge_score.py'       # supports --subset extra
JUDGE_SHA = '688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186'
QUERY = DATA_DIR / 'raw' / 'query.parquet'
DOC_TOKEN_LIMIT = 512
SHARD_NAME = 'pool-93001.parquet'            # S31 uses pool-00000..00021 and pool-91000
CHUNK_ROW_BASE = 3_000_000                   # ids of chunk rows; the source pair keeps its registry id in src_pair_id
KAGGLE_PAIRS_PER_S = 5.8                     # 4B fp16 on 2x T4 (S31, measured on pool13)
LOCAL_06B_PAIRS_PER_S = 24.0                 # S32 full run: 34,003.6 s for 825,318 pairs


def truncate_document(tokenizer, text: str, limit: int = DOC_TOKEN_LIMIT) -> tuple[str, int]:
    """Same function as r2ai.s29.qwen_judge.truncate_document: decoded leading passage of at most `limit` tokens."""
    ids = tokenizer.encode(text or '', add_special_tokens=False)[:limit]
    passage = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    while len(tokenizer.encode(passage, add_special_tokens=False)) > limit:
        ids = ids[:-1]
        passage = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return passage, len(tokenizer.encode(passage, add_special_tokens=False))


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(SNAP06), padding_side='left')


def load_pairs(tag_batches: list[str] | None = None) -> pl.DataFrame:
    p = pl.read_parquet(RUN / 'pairs_new.parquet')
    if tag_batches:
        p = p.filter(pl.col('batch').is_in(tag_batches))
    return p.sort('pair_id')


def doc_texts(segments: list[str]) -> dict[int, str]:
    out = {}
    for s in segments:
        d = pl.read_parquet(STORE / s / 'docs.parquet', columns=['doc_id', 'doc_text'])
        out.update(dict(d.iter_rows()))
    return out


def truncated(tok, texts: dict[int, str]) -> dict[int, tuple[str, int]]:
    cache, res = {}, {}
    for did, t in texts.items():
        if t not in cache:
            cache[t] = truncate_document(tok, t)
        res[did] = cache[t]
    return res


def base_rows(pairs: pl.DataFrame, queries: dict[int, str], docs: dict[int, tuple[str, int]]) -> pl.DataFrame:
    return pl.DataFrame({
        'pair_id': pairs['pair_id'], 'query_id': pairs['query_id'], 'doc_id': pairs['doc_id'],
        'cosine_rank': pairs['cosine_rank'].cast(pl.Int32), 'cosine_score': pairs['cosine_score'],
        'in_pool13': pairs['in_pool13'], 'lang': pl.Series(['zh'] * pairs.height),
        'query_text': [queries[q] for q in pairs['query_id'].to_list()],
        'doc_text': [docs[d][0] for d in pairs['doc_id'].to_list()],
        'doc_tokens': pl.Series([docs[d][1] for d in pairs['doc_id'].to_list()], dtype=pl.Int32)})


def check_judge() -> None:
    got = sha256_file(JUDGE_SCORE)
    if got != JUDGE_SHA:
        raise SystemExit(f'judge_score.py SHA {got} != {JUDGE_SHA}: refusing to package an unknown scorer')


def export_kaggle4b(tag: str, batches: list[str] | None) -> dict:
    check_judge()
    out = RUN / 'kaggle-4b' / tag
    assert_writable(out)
    pairs = load_pairs(batches)
    queries = dict(pl.read_parquet(QUERY).iter_rows())
    tok = tokenizer()
    t0 = time.time()
    docs = truncated(tok, doc_texts(sorted(pairs['segment'].unique().to_list())))
    shard = base_rows(pairs, queries, docs)
    (out / 'shards').mkdir(parents=True, exist_ok=True)
    atomic_parquet(shard, out / 'shards' / SHARD_NAME)
    extra = pl.DataFrame({
        'pair_id': pairs['pair_id'], 'query_id': pairs['query_id'], 'doc_id': pairs['doc_id'],
        'cosine_rank': pairs['cosine_rank'], 'cosine_score': pairs['cosine_score'],
        'source': pl.Series([f's37-{b}' for b in pairs['batch'].to_list()]),
        'free': pairs['in_top211'], 'capped': pl.Series([False] * pairs.height)})
    atomic_parquet(extra, out / 'extra_pairs.parquet')
    shutil.copyfile(JUDGE_SCORE, out / 'judge_score.py')
    man = {
        'status': 'prepared', 'tag': tag, 'batches': sorted(pairs['batch'].unique().to_list()),
        'pairs': pairs.height, 'pairs_in_top211': int(pairs['in_top211'].sum()), 'pairs_in_pool13': int(pairs['in_pool13'].sum()),
        'queries': int(pairs['query_id'].n_unique()), 'docs': int(pairs['doc_id'].n_unique()),
        'pair_id_range': [int(pairs['pair_id'].min()), int(pairs['pair_id'].max())], 'lang': 'zh',
        'model': 'qwen3-reranker-4b', 'revision': REV4B, 'truncate_document': f'Qwen3 tokenizer e61197ed, {DOC_TOKEN_LIMIT} tokens',
        'doc_tokens_mean': float(shard['doc_tokens'].mean()), 'docs_at_token_limit': int((shard['doc_tokens'] >= DOC_TOKEN_LIMIT).sum()),
        'eta_hours_kaggle_2xT4': round(pairs.height / KAGGLE_PAIRS_PER_S / 3600, 2),
        'eta_rule': f'{KAGGLE_PAIRS_PER_S} pairs/s on 2x T4 (4B fp16, measured on S31 pool13); not re-measured on S37 pairs',
        'extra_pairs_sha256': sha256_file(out / 'extra_pairs.parquet'), 'shard_sha256': sha256_file(out / 'shards' / SHARD_NAME),
        'judge_score_py_sha256': JUDGE_SHA, 'seconds': round(time.time() - t0, 1)}
    atomic_json(out / 'manifest.json', man)
    return {**man, 'dir': str(out)}


def export_cmax() -> dict:
    """Rows for Qwen3-Reranker-0.6B: the top-3 t256 body/answer chunks of every pair (cmax_top3.parquet), plus the
    doc_text of pairs whose doc has no body/answer chunk (S32 fallback: doc score instead of chunk scores)."""
    check_judge()
    out = RUN / 'cmax'
    assert_writable(out)
    pairs = load_pairs()
    queries = dict(pl.read_parquet(QUERY).iter_rows())
    tok = tokenizer()
    t0 = time.time()
    frames, picked = [], set()
    segs = sorted(pairs['segment'].unique().to_list())
    docs_full = doc_texts(segs)
    for s in segs:
        cm = pl.read_parquet(STORE / s / 'cmax_top3.parquet')
        ch = pl.read_parquet(STORE / s / 'chunks_t256.parquet', columns=['chunk_id', 'field', 'text'])
        j = cm.join(ch, on='chunk_id', how='left').join(pairs.select('pair_id', 'query_id', 'doc_id', 'cosine_rank', 'cosine_score', 'in_pool13'), on='pair_id')
        if j['text'].null_count():
            raise SystemExit(f'{s}: chunk ids without text')
        picked.update(j['pair_id'].to_list())
        frames.append(j)
    chunk = pl.concat(frames).sort('pair_id', 'rank')
    texts, toks = [], []
    cache = {}
    for t in chunk['text'].to_list():
        if t not in cache:
            cache[t] = truncate_document(tok, t)
        texts.append(cache[t][0])
        toks.append(cache[t][1])
    n = chunk.height
    rows = pl.DataFrame({
        'pair_id': pl.Series(range(CHUNK_ROW_BASE, CHUNK_ROW_BASE + n), dtype=pl.Int64), 'query_id': chunk['query_id'], 'doc_id': chunk['doc_id'],
        'cosine_rank': chunk['cosine_rank'].cast(pl.Int32), 'cosine_score': chunk['cosine_score'], 'in_pool13': chunk['in_pool13'],
        'lang': pl.Series(['zh'] * n), 'query_text': [queries[q] for q in chunk['query_id'].to_list()], 'doc_text': texts,
        'doc_tokens': pl.Series(toks, dtype=pl.Int32), 'chunk_id': chunk['chunk_id'], 'chunk_rank': chunk['rank'],
        'chunk_cosine': chunk['chunk_cosine'], 'field': chunk['field'], 'src_pair_id': chunk['pair_id'],
        'prompt_tokens': pl.Series([0] * n, dtype=pl.Int32), 'source': pl.Series(['s37-chunk'] * n)})
    fb = pairs.filter(~pl.col('pair_id').is_in(sorted(picked)))
    docs = truncated(tok, {d: docs_full[d] for d in fb['doc_id'].unique().to_list()})
    f = base_rows(fb, queries, docs)
    m = f.height
    f = f.with_columns(
        pair_id=pl.Series(range(CHUNK_ROW_BASE + n, CHUNK_ROW_BASE + n + m), dtype=pl.Int64), chunk_id=pl.Series([-1] * m, dtype=pl.Int64),
        chunk_rank=pl.Series([0] * m, dtype=pl.Int32), chunk_cosine=pl.Series([None] * m, dtype=pl.Float64),
        field=pl.Series(['doc_text'] * m), src_pair_id=pl.Series(fb['pair_id'].to_list(), dtype=pl.Int64),
        prompt_tokens=pl.Series([0] * m, dtype=pl.Int32), source=pl.Series(['s37-fallback-doc'] * m))
    allrows = pl.concat([rows, f.select(rows.columns)])
    pool = out / 'pool'
    (pool / 'shards').mkdir(parents=True, exist_ok=True)
    atomic_parquet(allrows, pool / 'shards' / SHARD_NAME)
    shutil.copyfile(JUDGE_SCORE, pool / 'judge_score.py')
    man = {
        'status': 'prepared', 'model': 'qwen3-reranker-0.6b', 'revision': 'e61197ed45024b0ed8a2d74b80b4d909f1255473',
        'pairs': pairs.height, 'pairs_with_chunks': len(picked), 'pairs_fallback_doc_text': m, 'chunk_rows': n, 'rows': allrows.height,
        'chunks_per_pair': {str(k): int(v) for k, v in chunk.group_by('pair_id').len()['len'].value_counts().sort('len').iter_rows()},
        'field_share': {k: round(v / n, 4) for k, v in chunk['field'].value_counts().iter_rows()},
        'eta_hours_local': round(allrows.height / LOCAL_06B_PAIRS_PER_S / 3600, 2),
        'eta_rule': f'{LOCAL_06B_PAIRS_PER_S} pairs/s (S32 local RTX 3050 6 GB, hf bf16); not re-measured on S37 rows',
        'shard_sha256': sha256_file(pool / 'shards' / SHARD_NAME), 'judge_score_py_sha256': JUDGE_SHA, 'seconds': round(time.time() - t0, 1)}
    atomic_json(pool / 'manifest.json', man)
    return {**man, 'dir': str(pool)}


def reduce_cmax(scores_path: Path | None = None) -> dict:
    """C-max per source pair = max P(yes) over its chunk rows; fallback rows carry the doc_text score."""
    scores_path = scores_path or RUN / 'cmax' / 'gpu' / 'qwen3-reranker-0.6b' / 'scores-all.parquet'
    sc = pl.read_parquet(scores_path)
    col = next(c for c in ('p_yes', 'score', 'yes_prob') if c in sc.columns)
    pool = pl.read_parquet(RUN / 'cmax' / 'pool' / 'shards' / SHARD_NAME, columns=['pair_id', 'src_pair_id', 'chunk_rank', 'source'])
    j = pool.join(sc.select('pair_id', pl.col(col).alias('p')), on='pair_id', how='left')
    if j['p'].null_count():
        raise SystemExit(f"{j['p'].null_count()} rows without score")
    g = (j.sort('src_pair_id', 'p', descending=[False, True]).group_by('src_pair_id', maintain_order=True)
         .agg(pl.col('p').first().alias('cmax'), pl.col('p').head(2).mean().alias('ctop2'), pl.len().alias('n_chunks'),
              (pl.col('source').first() == 's37-fallback-doc').alias('fallback')))
    out = g.rename({'src_pair_id': 'pair_id'}).sort('pair_id')
    atomic_parquet(out, RUN / 'cmax_scores.parquet')
    return {'pairs': out.height, 'fallback': int(out['fallback'].sum()), 'cmax_p10_p50_p90': [float(out['cmax'].quantile(x)) for x in (0.1, 0.5, 0.9)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('kaggle4b')
    k.add_argument('--tag', required=True)
    k.add_argument('--batches', nargs='*', help='batch ids (b0001 ...); default every pair in pairs_new')
    sub.add_parser('cmax')
    r = sub.add_parser('reduce')
    r.add_argument('--scores', type=Path)
    a = ap.parse_args(argv)
    out = export_kaggle4b(a.tag, a.batches) if a.cmd == 'kaggle4b' else export_cmax() if a.cmd == 'cmax' else reduce_cmax(a.scores)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
