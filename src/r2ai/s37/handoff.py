"""Hand-off manifest for S35 (and the S38 chunk stage): every S37 artefact, its SHA256 and its score coverage, in one file.

    python -m r2ai.s37.handoff manifest        # out/runs/s37/handoff/{manifest.json, pairs_score_flags.parquet, jina_queries_scored.parquet}
    python -m r2ai.s37.handoff merge4b         # out/runs/s37/gpu/qwen3-reranker-4b/scores-120ask-4b.parquet (needs every 4B subset imported)

A source that is not finished is written as status "pending" with what exists so far, so S35 can build 120A-p4b (cosine + 4B + C-max)
without Jina and then rerun with the final manifest. Nothing here recomputes a score; it only lists, hashes and checks.
"""
from __future__ import annotations

from r2ai.paths import DATA_DIR, RUNS_DIR, assert_writable

import argparse
import json
import time

import pandas as pd

from r2ai.s37.ingest import RUN, STORE, atomic_json, sha256_file

OUT = RUN / 'handoff'
KAGGLE = RUN / 'kaggle-4b'
FOURB_TAGS = ('s37-b001-003', 's37-b004-006', 's37-b007')
FOURB_DIR = RUN / 'gpu' / 'qwen3-reranker-4b'
MERGED_4B = FOURB_DIR / 'scores-120ask-4b.parquet'
JINA_FINAL = RUN / 'jina-final'
JINA_FINAL2 = RUN / 'jina-final2'
PAIRS_V1 = RUN / 'pairs_new_final.parquet'
PAIRS_V2 = RUN / 'pairs_new_final_v2.parquet'
PAIRS_FINAL = PAIRS_V2 if PAIRS_V2.exists() else PAIRS_V1     # v2 = after the last crawl batches (b0008, b0009)
JINA_FIRST = RUN / 'jina'
S35_JINA = RUNS_DIR / 's35-part-jina' / 'jina'
SEGMENT_FILES = ('docs.parquet', 'chunks_t256.parquet', 'chunks_struct.parquet', 'provenance.parquet', 'vectors.npz', 'chunk_vectors.npz',
                 'cmax_top3.parquet', 'segment.json', 'manifest.json')


def rel(p) -> str:
    try:
        return str(p.relative_to(RUN.parent.parent)).replace('\\', '/')
    except ValueError:
        return str(p)


def file_entry(p) -> dict:
    return {'path': rel(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}


def segments() -> list[dict]:
    out = []
    for seg in sorted(STORE.glob('seg37*')):
        man = json.loads((seg / 'segment.json').read_text('utf-8')) if (seg / 'segment.json').exists() else {}
        out.append({'segment': seg.name, 'lang': 'zh', 'state': man.get('state'), 'docs': int(len(pd.read_parquet(seg / 'docs.parquet', columns=['doc_id']))),
                    'files': {f: file_entry(seg / f) for f in SEGMENT_FILES if (seg / f).exists()}})
    en = DATA_DIR / 's37_en' / 'seg37900-en-pubmed'          # PubMed en segment (canonical-20261025): doc-level vectors only, no chunk vectors, no scores
    if (en / 'manifest.json').exists():
        man = json.loads((en / 'manifest.json').read_text('utf-8'))
        out.append({'segment': en.name, 'lang': 'en', 'state': man.get('state'), 'docs': man['docs'], 'docs_thin': man['docs_thin'],
                    'doc_text_rule': man['doc_text_rule'], 'source': man['source'],
                    'files': {f: file_entry(en / f) for f in ('docs.parquet', 'vectors.npz', 'manifest.json') if (en / f).exists()},
                    'note': 'no 4B / Jina / C-max scores for en (Vu, 25/10)'})
    return out


def fourb(pairs: pd.DataFrame) -> dict:
    subsets, have = {}, []
    for tag in FOURB_TAGS:
        man_p = KAGGLE / tag / 'manifest.json'
        if not man_p.exists():
            continue
        man = json.loads(man_p.read_text('utf-8'))
        sc = KAGGLE / tag / 'gpu' / 'qwen3-reranker-4b' / 'scores-extra.parquet'
        entry = {'status': 'scored' if sc.exists() else 'pending', 'pairs': man['pairs'], 'batches': man['batches'],
                 'pair_id_range': man['pair_id_range'], 'revision': man['revision']}
        if sc.exists():
            entry['scores'] = file_entry(sc)
            have.append(pd.read_parquet(sc, columns=['pair_id']))
        subsets[tag] = entry
    ids = set(pd.concat(have).pair_id) if have else set()
    res = {'model': 'qwen3-reranker-4b', 'revision': '22e683669bc0f0bd69640a1354a6d0aebcfeede5', 'subsets': subsets,
           'pairs_scored': len(ids & set(pairs.pair_id)), 'pairs_total': len(pairs), 'status': 'complete' if len(ids & set(pairs.pair_id)) == len(pairs) else 'partial'}
    if MERGED_4B.exists():
        res['merged'] = file_entry(MERGED_4B)
    return res


def cmax(pairs: pd.DataFrame) -> dict:
    files = sorted(RUN.glob('cmax_scores*.parquet'))
    ids = set(pd.concat([pd.read_parquet(f, columns=['pair_id']) for f in files]).pair_id) if files else set()
    return {'model': 'qwen3-reranker-0.6b', 'revision': 'e61197ed45024b0ed8a2d74b80b4d909f1255473', 'files': {f.name: file_entry(f) for f in files},
            'pairs_scored': len(ids & set(pairs.pair_id)), 'pairs_total': len(pairs),
            'status': 'complete' if len(ids & set(pairs.pair_id)) == len(pairs) else 'partial'}


def jina_state(pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame, set]:
    """Jina: S35 scores stay valid for queries outside the 976; for those 976 the valid scores are the first-run parts that did not change
    (jina-final/reused_scores.parquet) plus whatever jina-final has scored so far (parts, or scores-all.parquet when complete)."""
    run = JINA_FINAL2 if (JINA_FINAL2 / 'pool_manifest.json').exists() else JINA_FINAL   # the top-up run is complete on its own (it carries the reused scores)
    pool_man = json.loads((run / 'pool_manifest.json').read_text('utf-8')) if (run / 'pool_manifest.json').exists() else None
    full = pd.read_parquet(run / 'pool_full.parquet', columns=['query_id', 'doc_id']) if pool_man else pd.DataFrame(columns=['query_id', 'doc_id'])
    parts = sorted((run / 'parts-b10s0').glob('part-*.parquet'))
    reused = pd.read_parquet(run / 'reused_scores.parquet') if (run / 'reused_scores.parquet').exists() else pd.DataFrame(columns=['query_id', 'doc_id', 'score'])
    new = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True) if parts else pd.DataFrame(columns=['query_id', 'doc_id', 'score'])
    rows = pd.concat([reused[['query_id', 'doc_id', 'score']], new[['query_id', 'doc_id', 'score']]], ignore_index=True).drop_duplicates(['query_id', 'doc_id'])
    per_q = rows.groupby('query_id').size()
    need = full.groupby('query_id').size() if len(full) else pd.Series(dtype=int)
    complete_q = sorted(q for q, n in need.items() if per_q.get(q, 0) == n)
    done = (run / 'scores-all.parquet').exists()
    res = {'status': 'complete' if done and len(complete_q) == len(need) and len(need) else 'pending', 'layout': 'b10s0',
           'revision': 'e8a93f33f0b22108f8c2364f8484ce3422552fbc', 'queries_in_final_pool': int(len(need)),
           'queries_with_final_scores': len(complete_q), 'queries_pending': int(len(need) - len(complete_q)),
           'rule': 'queries outside the 976 keep the S35 Jina scores (s35-part-jina/jina/scores-all.parquet); the 976 use the scores listed here',
           's35_scores_sha256': 'e9a4818eeb720b83711616b497c94f87e3c5d1cceffa6fd26b655f6c6489cf39'}
    if pool_man:
        res['final_pool'] = {k: pool_man[k] for k in ('pairs_final_pool', 'pairs_by_source', 'queries_to_score', 'pool_full_sha256', 'pool_sha256', 's36_missing_sha256')}
        res['final_pool']['run'] = run.name
    if (run / 'reused_scores.parquet').exists():
        res['reused_scores'] = file_entry(run / 'reused_scores.parquet')
    if done:
        res['scores_final_run'] = file_entry(run / 'scores-all.parquet')
    return res, rows, set(complete_q)


def manifest() -> dict:
    pairs = pd.read_parquet(PAIRS_FINAL)
    jina, jrows, jdone_q = jina_state(pairs)
    jk = set(zip(jrows.query_id, jrows.doc_id))
    ids4 = set()
    for tag in FOURB_TAGS:
        sc = KAGGLE / tag / 'gpu' / 'qwen3-reranker-4b' / 'scores-extra.parquet'
        if sc.exists():
            ids4 |= set(pd.read_parquet(sc, columns=['pair_id']).pair_id)
    idsc = set()
    for f in RUN.glob('cmax_scores*.parquet'):
        idsc |= set(pd.read_parquet(f, columns=['pair_id']).pair_id)
    flags = pairs[['pair_id', 'query_id', 'doc_id', 'batch', 'in_top211', 'in_pool13']].copy()
    flags['has_4b'] = flags.pair_id.isin(ids4)
    flags['has_cmax'] = flags.pair_id.isin(idsc)
    flags['has_jina'] = [(q, d) in jk for q, d in zip(flags.query_id, flags.doc_id)]
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(OUT / 'pairs_score_flags.parquet', index=False)
    pd.DataFrame({'query_id': sorted(jdone_q)}).to_parquet(OUT / 'jina_queries_scored.parquet', index=False)
    man = {
        'schema': 'r2ai.s37.handoff.v1', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': {'crawl': '120ask closed: batches 1-6 (55,994 URLs ok) + batch-0007 (shard r1-00028, 388 ok); no more data after this',
                  'canonical_base': 'canonical-20261018 (unchanged); 120ask lives in the segments below'},
        'segments': segments(),
        'pairs': {'file': file_entry(PAIRS_FINAL), 'pairs': len(pairs), 'queries': int(pairs.query_id.nunique()),
                  'docs': int(pairs.doc_id.nunique()), 'pair_id_range': [int(pairs.pair_id.min()), int(pairs.pair_id.max())],
                  'in_top211': int(pairs.in_top211.sum()), 'in_pool13': int(pairs.in_pool13.sum())},
        'scores': {'4b': fourb(pairs), 'cmax': cmax(pairs), 'jina': jina},
        'coverage_pct_of_pairs': {'4b': round(100 * flags.has_4b.mean(), 2), 'cmax': round(100 * flags.has_cmax.mean(), 2), 'jina': round(100 * flags.has_jina.mean(), 2)},
        'flags': {'pairs_score_flags': file_entry(OUT / 'pairs_score_flags.parquet'), 'jina_queries_scored': file_entry(OUT / 'jina_queries_scored.parquet')},
        'final': bool(jina['status'] == 'complete' and flags.has_cmax.all()),
        'pairs_without_4b': int((~flags.has_4b).sum()),   # S35 merges these with the d300 pack into one Kaggle subset; 4B is not exported here
        'note': 'final=false: Jina (or C-max) still pending. build 120A-p4b from cosine + 4B + C-max when the pairs without 4B are covered; Jina is not needed for it',
    }
    atomic_json(OUT / 'manifest.json', man)
    return {'manifest': rel(OUT / 'manifest.json'), 'sha256': sha256_file(OUT / 'manifest.json'), 'final': man['final'],
            'coverage_pct_of_pairs': man['coverage_pct_of_pairs'], 'jina_status': jina['status'], 'jina_queries_scored': len(jdone_q),
            'jina_queries_in_final_pool': jina['queries_in_final_pool']}


def merge4b() -> dict:
    pairs = pd.read_parquet(RUN / 'pairs_new_final.parquet', columns=['pair_id', 'query_id', 'doc_id'])
    parts = []
    for tag in FOURB_TAGS:
        sc = KAGGLE / tag / 'gpu' / 'qwen3-reranker-4b' / 'scores-extra.parquet'
        if not sc.exists():
            raise SystemExit(f'{tag}: no imported 4B scores yet (run export import4b)')
        parts.append(pd.read_parquet(sc).assign(subset=tag))
    s = pd.concat(parts, ignore_index=True)
    if not s.pair_id.is_unique:
        raise SystemExit('duplicate pair_id across 4B subsets')
    j = pairs.merge(s, on=['pair_id', 'query_id', 'doc_id'], how='outer', indicator=True)
    if (j._merge != 'both').any() or s.score.isna().any():
        raise SystemExit(f'4B scores do not match pairs_new_final: {int((j._merge != "both").sum())} rows differ, NaN={int(s.score.isna().sum())}')
    assert_writable(FOURB_DIR)
    FOURB_DIR.mkdir(parents=True, exist_ok=True)
    s.sort_values('pair_id').reset_index(drop=True).to_parquet(MERGED_4B, index=False)
    res = {'rows': len(s), 'subsets': {t: int((s.subset == t).sum()) for t in FOURB_TAGS}, 'sha256': sha256_file(MERGED_4B), 'path': rel(MERGED_4B)}
    atomic_json(FOURB_DIR / 'scores-120ask-4b.report.json', res)
    return res


def pairs_v2() -> dict:
    """pairs_new_final_v2.parquet = the registry after the last crawl batches. Checks that the ids of v1 did not move and the new ids continue after them."""
    v1 = pd.read_parquet(PAIRS_V1)
    cur = pd.read_parquet(RUN / 'pairs_new.parquet')
    if not cur.pair_id.is_unique:
        raise SystemExit('duplicate pair_id in pairs_new.parquet')
    head = cur[cur.pair_id.isin(v1.pair_id)].sort_values('pair_id').reset_index(drop=True)
    old = v1.sort_values('pair_id').reset_index(drop=True)
    if len(head) != len(old) or not (head[['pair_id', 'query_id', 'doc_id']].values == old[['pair_id', 'query_id', 'doc_id']].values).all():
        raise SystemExit('pair ids of v1 changed in pairs_new.parquet')
    new = cur[~cur.pair_id.isin(v1.pair_id)]
    if len(new) and (new.pair_id.min() <= v1.pair_id.max() or new.pair_id.max() - new.pair_id.min() + 1 != len(new)):
        raise SystemExit('new pair ids do not continue after v1 without gaps')
    cur.sort_values('pair_id').reset_index(drop=True).to_parquet(PAIRS_V2, index=False)
    return {'pairs_v1': len(v1), 'pairs_v2': len(cur), 'pairs_added': len(new), 'queries_v2': int(cur.query_id.nunique()), 'docs_v2': int(cur.doc_id.nunique()),
            'queries_touched_by_new': int(new.query_id.nunique()), 'docs_new': int(new.doc_id.nunique()),
            'by_batch': {b: int(n) for b, n in new.batch.value_counts().sort_index().items()}, 'sha256': sha256_file(PAIRS_V2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('manifest', 'merge4b', 'pairsv2'))
    a = ap.parse_args(argv)
    print(json.dumps({'manifest': manifest, 'merge4b': merge4b, 'pairsv2': pairs_v2}[a.command](), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
