"""S37: C-max (Qwen3-Reranker-0.6B, top-3 t256 body/answer chunks by cosine, S32 rule) for the S36 d300 pairs that have no C-max.

    python d300_cmax.py prepare [--limit N]    out/runs/s37/cmax-d300/pool/{shards, judge_score.py, manifest.json}, gate on 1,000 S32 pairs
    python d300_cmax.py reduce                 pool scores -> out/runs/s37/cmax_scores-d300.parquet (query_id, doc_id, cmax, fallback)

Standalone: it needs the S33 / S36 packages of the worktree `s35-part-jina` (PYTHONPATH=<worktree>/src, see run_d300_cmax.ps1).
Chunks are chosen exactly as S32 / S36 `extra cmax-prepare` (r2ai.s33.pool.select_like_s32, same gate); the 0.6B scoring itself is
scripts/s37/run_cmax.ps1 on the pool written here. Chunk vectors missing from the canonical bundle are embedded on the CPU (fp32, stored fp16)
so this step never takes the GPU from a running job; pairs whose doc has no body/answer chunk are scored on doc_text (S32 fallback).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import DOC_TOKEN_LIMIT, _atomic_json, _model_path, _sha256, truncate_document
from r2ai.s33 import pool as s33_pool

RUN = paths.RUNS_DIR / 's37'
SRC = paths.RUNS_DIR / 's36-fuse4' / 'd300' / 'cmax_pairs.parquet'
OUT = RUN / 'cmax-d300'
POOL = OUT / 'pool'
CANONICAL = paths.DATA_DIR / 'canonical-20261018'
QUERY = paths.DATA_DIR / 'raw' / 'query.parquet'
JUDGE = paths.RUNS_DIR / 's31-strong-judge' / 'pool-extra' / 'kaggle' / 'judge_score.py'
JUDGE_SHA = '688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186'
SRC_ID_BASE = 50_000_000       # src_pair_id of these pairs (S36 extra used 40,000,000)
ROW_ID_BASE = 60_000_000       # pair_id of the scored rows
GATE_PAIRS, GATE_SEED = 1_000, 36
RCHK = s33_pool.RCHK_COLUMNS


def cpu_encoder() -> None:
    """select_like_s32 builds a CUDA BGE-M3 for the few chunks without a stored vector: run it on the CPU instead."""
    import r2ai.index.bge_m3 as m3
    orig = m3.M3Encoder.__init__
    m3.M3Encoder.__init__ = lambda self, device='cpu', fp16=False, max_len=512: orig(self, 'cpu', False, max_len)


def prepare(limit: int | None = None) -> dict:
    t0 = time.monotonic()
    cpu_encoder()
    s33_pool._prepare_out(OUT)
    log = s33_pool._logger(OUT / 'prepare.log')
    tab = pd.read_parquet(SRC)
    if limit:
        tab = tab.head(limit)
    if JUDGE_SHA != _sha256(JUDGE):
        raise SystemExit('judge_score.py is not the S31 pool-extra scorer')
    canon = pd.read_parquet(CANONICAL / 'docs.parquet', columns=['doc_id', 'lang'])
    lang = dict(zip(canon.doc_id.astype(int), canon.lang))
    miss = sorted(set(tab.doc_id.astype(int)) - set(lang))
    if miss:                                    # docs outside the canonical bundle: the S36 pair tables know their language, Pharmacity docs are vi
        for name in ('pairs_final', 'pairs_d300'):
            t = pd.read_parquet(SRC.with_name(name + '.parquet'), columns=['doc_id', 'lang'])
            for d, lg in zip(t.doc_id.astype(int), t.lang):
                lang.setdefault(d, lg)
        miss = sorted(set(tab.doc_id.astype(int)) - set(lang))
    if miss:
        from r2ai.s29.qwen_judge import _inputs
        from r2ai.s32 import chunks as s32
        ph_docs = set(s32.load_pharmacity_chunks(_inputs()).doc_id.astype(int))
        for d in miss:
            if d in ph_docs:
                lang[d] = 'vi'
        miss = sorted(set(tab.doc_id.astype(int)) - set(lang))
    if miss:
        raise SystemExit(f'{len(miss)} docs of the pair list are not in canonical-20261018 (e.g. {miss[:5]})')
    queries = dict(pd.read_parquet(QUERY).itertuples(index=False))
    log(f'd300 pairs without C-max: {len(tab)} in {tab.query_id.nunique()} queries, {tab.doc_id.nunique()} docs')
    pairs = pd.DataFrame({'src_pair_id': SRC_ID_BASE + np.arange(len(tab), dtype=np.int64), 'query_id': tab.query_id.astype(np.int64).to_numpy(),
                          'doc_id': tab.doc_id.astype(np.int64).to_numpy(), 'cosine_rank': tab.cosine_rank.astype(np.int64).to_numpy(),
                          'cosine_score': tab.cosine_score.astype(float).to_numpy(), 'lang': [lang[int(d)] for d in tab.doc_id], 'source': 'd300'})
    s32_dir = s33_pool.s32_pool()
    s32_index = pd.read_parquet(s32_dir / 'chunk_pairs.parquet')
    covered = s32_index.drop_duplicates('src_pair_id')[['src_pair_id', 'query_id', 'doc_id', 'cosine_rank', 'cosine_score', 'lang']]
    gate_pairs = covered.sample(n=GATE_PAIRS, random_state=GATE_SEED).copy()
    gate_pairs['source'] = 'gate'
    sel, fallback, info = s33_pool.select_like_s32(pd.concat([pairs, gate_pairs], ignore_index=True), log, s32_dir / 'supplement_vectors.npz', OUT)
    gsel = sel[sel.source == 'gate']
    ref = s32_index[s32_index.src_pair_id.isin(gate_pairs.src_pair_id)]
    got = gsel.sort_values(['src_pair_id', 'chunk_rank']).groupby('src_pair_id').chunk_id.apply(list)
    exp = ref.sort_values(['src_pair_id', 'chunk_rank']).groupby('src_pair_id').chunk_id.apply(list)
    mismatch = [int(k) for k in exp.index if got.get(k) != exp[k]]
    gate = {'pairs': GATE_PAIRS, 'seed': GATE_SEED, 'mismatches': len(mismatch), 'examples': mismatch[:5], 'gate_fallback': int((fallback.source == 'gate').sum())}
    _atomic_json(OUT / 'gate.json', gate)
    log(f'gate {json.dumps(gate)}')
    if mismatch:
        raise SystemExit('chunk selection differs from S32 on the gate sample; see gate.json')
    sel = s33_pool.attach_texts(sel[sel.source == 'd300'], log)
    sel['query_text'] = sel.query_id.map(queries)
    sel['chunk_source'] = 'd300'
    sel['in_pool13'] = True
    sel['pair_id'] = -1

    fb_keys = fallback[fallback.source == 'd300']
    fb = pairs[pairs.src_pair_id.isin(fb_keys.src_pair_id)].copy()
    tok = None
    if len(fb):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
        dt = pq.read_table(CANONICAL / 'docs.parquet', columns=['doc_id', 'doc_text'], filters=[('doc_id', 'in', sorted(fb.doc_id.unique().tolist()))]).to_pandas()
        text = dict(zip(dt.doc_id.astype(int), dt.doc_text))
        cut = {}
        for d in fb.doc_id.unique():
            cut[int(d)] = truncate_document(tok, text[int(d)])
        fb['doc_text'] = [cut[int(d)][0] for d in fb.doc_id]
        fb['doc_tokens'] = np.array([cut[int(d)][1] for d in fb.doc_id], dtype=np.int32)
    fb['query_text'] = fb.query_id.map(queries)
    fb['chunk_id'], fb['chunk_rank'], fb['chunk_cosine'] = -1, 0, np.nan
    fb['field'], fb['origin'], fb['chunk_source'], fb['in_pool13'], fb['pair_id'] = 'doc_text', 'd', 'fallback', True, -1

    cols = list(RCHK)
    rows = pd.concat([sel.assign(src_pair_id=sel.src_pair_id.astype(np.int64))[cols], fb[cols]], ignore_index=True)
    rows = rows.sort_values(['query_id', 'cosine_rank', 'doc_id', 'chunk_rank'], kind='stable').reset_index(drop=True)
    rows['pair_id'] = ROW_ID_BASE + np.arange(len(rows), dtype=np.int64)
    if rows.duplicated(['query_id', 'doc_id', 'chunk_rank', 'chunk_id']).any():
        raise SystemExit('duplicate rows')
    keys = set(zip(pairs.query_id, pairs.doc_id))
    got_keys = set(zip(rows.query_id.astype(int), rows.doc_id.astype(int)))
    if got_keys != keys:
        raise SystemExit(f'{len(keys - got_keys)} pairs without any row, {len(got_keys - keys)} extra')
    (POOL / 'shards').mkdir(parents=True, exist_ok=True)       # write_shards writes <dir>/shards/pool-NNNNN.parquet; the scorer reads <pool>/shards
    shards = s33_pool.write_shards(POOL, rows, RCHK, log)
    s33_pool._atomic_frame(OUT / 'chunk_pairs.parquet', rows[[c for c in cols if c not in ('query_text', 'doc_text')]])
    shutil.copyfile(JUDGE, POOL / 'judge_score.py')
    per_pair = rows.groupby(['query_id', 'doc_id']).size()
    man = {'status': 'prepared', 'model': 'qwen3-reranker-0.6b', 'revision': 'e61197ed45024b0ed8a2d74b80b4d909f1255473', 'pairs': len(pairs),
           'rows': len(rows), 'rows_chunks': int((rows.chunk_source == 'd300').sum()), 'rows_doc_text_fallback': int((rows.chunk_source == 'fallback').sum()),
           'pairs_with_chunks': int(len(pairs) - len(fb)), 'pairs_doc_text': int(len(fb)), 'rows_per_pair': {str(k): int(v) for k, v in per_pair.value_counts().sort_index().items()},
           'doc_tokens_mean': float(rows.doc_tokens.mean()), 'gate': gate, **info, 'shards': shards,
           'eta_hours_local': round(len(rows) / 24.0 / 3600, 2), 'eta_rule': '24 rows/s (S32 local RTX 3050 6 GB bf16)',
           'src_pair_file_sha256': _sha256(SRC), 'judge_score_py_sha256': JUDGE_SHA, 'seconds': round(time.monotonic() - t0, 1)}
    _atomic_json(POOL / 'manifest.json', man)
    return man


def reduce() -> dict:
    index = pd.read_parquet(OUT / 'chunk_pairs.parquet', columns=['pair_id', 'query_id', 'doc_id', 'chunk_source'])
    scores = pd.read_parquet(OUT / 'gpu' / 'qwen3-reranker-0.6b' / 'scores-all.parquet', columns=['pair_id', 'score'])
    if scores.pair_id.duplicated().any() or not np.isfinite(scores.score).all():
        raise SystemExit('duplicate or non-finite scores')
    m = index.merge(scores, on='pair_id', how='left')
    g = m.groupby(['query_id', 'doc_id'])
    done = g.score.apply(lambda s: s.notna().all())
    if not done.all():
        raise SystemExit(f'{int((~done).sum())} pairs have unscored rows')
    res = g.score.max().rename('cmax').reset_index()
    fb = m[m.chunk_source == 'fallback'].set_index(['query_id', 'doc_id']).index
    res['fallback'] = res.set_index(['query_id', 'doc_id']).index.isin(fb)
    dest = RUN / 'cmax_scores-d300.parquet'
    res.to_parquet(dest, index=False)
    return {'pairs': len(res), 'fallback': int(res.fallback.sum()), 'cmax_p10_p50_p90': [float(res.cmax.quantile(x)) for x in (.1, .5, .9)],
            'sha256': _sha256(dest)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('prepare', 'reduce'))
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(prepare(a.limit) if a.command == 'prepare' else reduce(), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
