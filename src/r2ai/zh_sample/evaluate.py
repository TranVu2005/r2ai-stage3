"""Dense/hybrid measurements and direct-logit yield proxy; no leaderboard labels."""
from __future__ import annotations

from r2ai.paths import QUERY_FILE, ROOT, require_inputs
from .common import RUN, CHUNKS, INDEX, preflight, atomic_json, exclusive
from .index import digest, gpu_idle
import argparse
import json
import os
import pickle
import time
from collections import defaultdict


def merge_scores(vi, zh, k=150):
    # Membership uses direct logits. Preserve D50's vi order across its tier boundary.
    # Insert zh before the first retained item with a lower score; ties retain vi first.
    unique = {}
    for d, score in list(vi) + list(zh):
        if d not in unique:
            unique[d] = float(score)
    selected = dict(sorted(unique.items(), key=lambda r: -r[1])[:k])
    vi_ids = {d for d, _ in vi}
    result = [(d, float(s)) for d, s in vi if d in selected]
    for d, score in sorted(((d, s) for d, s in selected.items() if d not in vi_ids), key=lambda r: -r[1]):
        pos = next((i for i, (_, s) in enumerate(result) if s < score), len(result))
        result.insert(pos, (d, score))
    return result


def yield_proxy(merged, domain_for, sampled_docs):
    hits, unique, queries = defaultdict(int), defaultdict(set), defaultdict(set)
    for qid, docs in merged.items():
        for doc, _ in docs:
            if doc in domain_for:
                domain = domain_for[doc]
                hits[domain] += 1
                unique[domain].add(doc)
                queries[domain].add(qid)
    return {domain: {'hits': hits[domain], 'unique_docs': len(unique[domain]), 'queries': len(queries[domain]),
                     'extracted_docs': n, 'hits_per_1k_extracted_docs': hits[domain] * 1000 / n if n else None,
                     'note': 'proxy, no relevance labels; hits are query-document occurrences'}
            for domain, n in sorted(sampled_docs.items())}


def vi_rankings():
    import pyarrow.parquet as pq
    from r2ai.submit.best import load_config
    cfg = load_config(ROOT / 'configs/submission-best.yaml')
    cache = ROOT / cfg['submission']['runs_dir'] / 'vi_k100.parquet'
    candidates = ROOT / cfg['submission']['doc_ranking']
    require_inputs(cache, candidates)
    cached, scores, ranked = defaultdict(list), defaultdict(dict), defaultdict(list)
    for r in pq.read_table(cache).to_pylist():
        scores[r['query_id']][r['doc_id']] = float(r['score'])
        if r['rank'] <= 100:
            cached[r['query_id']].append((r['rank'], r['doc_id']))
    for q, rows in cached.items():
        ranked[q] = [d for _, d in sorted(rows)]
    seen = {q: set(ds) for q, ds in ranked.items()}
    cr = pq.read_table(candidates, columns=['query_id', 'rank', 'doc_id']).to_pylist()
    for r in sorted(cr, key=lambda r: (r['query_id'], r['rank'])):
        q, d = r['query_id'], r['doc_id']
        if q in ranked and len(ranked[q]) < 150 and d not in seen[q]:
            ranked[q].append(d)
            seen[q].add(d)
    missing = [(q, d) for q, ds in ranked.items() for d in ds if d not in scores[q]]
    if missing:
        raise ValueError(f'D50 docs lack comparable reranker scores: {len(missing)} pairs, e.g. {missing[:5]}')
    return {q: [(d, scores[q][d]) for d in ds] for q, ds in ranked.items()}, {
        'vi_cache': str(cache), 'vi_cache_sha256': digest(cache),
        'vi_candidate_ranking': str(candidates), 'vi_candidate_sha256': digest(candidates),
        'reranker': 'BAAI/bge-reranker-v2-m3', 'fp16_padding_note': 'scores across runs may differ by fp16 batch padding ULPs'}


def save_pickle(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def candidate_stats_json(stats):
    return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in stats.items()}


def retrieve():
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.sparse import load_npz
    from r2ai.index.bge_m3 import M3Encoder, Reranker
    from r2ai.retrieve.exact import exact_candidates, dense_topk
    from r2ai.retrieve.run import minmax, ws
    import torch
    preflight()
    cf = CHUNKS / 'chunks_t256.parquet'
    require_inputs(cf, INDEX / 'meta.json', QUERY_FILE)
    index_manifest = json.loads((INDEX / 'input_manifest.json').read_text('utf-8'))
    if index_manifest['chunks_sha256'] != digest(cf) or index_manifest['extract_manifest_sha256'] != digest(RUN / 'extract_manifest.json'):
        raise ValueError('Index/chunks/extract bundle hashes differ; refusing retrieval')
    qs = pq.read_table(QUERY_FILE).to_pylist()
    if len(qs) != 1200:
        raise ValueError(f'Need 1200 queries, found {len(qs)}')
    vi, vi_manifest = vi_rankings()
    cfg = {'query_sha256': digest(QUERY_FILE), 'chunk_sha256': digest(cf),
           'index_manifest_sha256': digest(INDEX / 'input_manifest.json'), 'vi': vi_manifest,
           'seed': 42, 'candidate_k_per_branch': 200, 'hybrid_weight': .7,
           'zh_rerank_docs_per_branch': 150, 'max_len': 512, 'reranker_batch': 32}
    out = RUN / 'retrieval'
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text('utf-8')) != cfg:
        raise ValueError('Retrieval resume config differs; refusing to mix bundles')
    atomic_json(config_path, cfg)
    with exclusive('gpu'):
        gpu_idle()
        np.random.seed(42)
        torch.manual_seed(42)
        query_cache = out / 'queries.pkl'
        if query_cache.exists():
            qd, qsp = pickle.loads(query_cache.read_bytes())
        else:
            enc = M3Encoder(max_len=512)
            dense, qsp = [], []
            for lo in range(0, len(qs), 32):
                d, s = enc.encode_batch([ws(r['query']) for r in qs[lo:lo+32]])
                dense.append(d)
                qsp += s
            qd = np.concatenate(dense)
            save_pickle(query_cache, (qd, qsp))
            del enc
            torch.cuda.empty_cache()
        chunks = pq.read_table(cf, columns=['doc_id', 'text']).to_pydict()
        doc_ids, texts = np.array(chunks['doc_id']), chunks['text']
        dense = np.load(INDEX / 'dense.npy', mmap_mode='r')
        sp = load_npz(INDEX / 'sparse.npz').tocsr()
        cand_cache = out / 'candidates.pkl'
        if cand_cache.exists():
            hybrid_cand, dense_cand = pickle.loads(cand_cache.read_bytes())
        else:
            hybrid_cand, st = exact_candidates(dense, (sp.data, sp.indices, sp.indptr), doc_ids, qd, qsp,
                                              k=200, block_rows=10000, device='cpu')
            dense_cand, _ = dense_topk(dense, qd, 200, 10000, doc_ids, 'cpu')
            save_pickle(cand_cache, (hybrid_cand, dense_cand))
            atomic_json(out / 'candidate_stats.json', candidate_stats_json(st))
        rr = Reranker(max_len=512)
        t_start = time.perf_counter()
        for i, query in enumerate(qs):
            qid = int(query['id'])
            destination = out / f'q{qid}.json'
            if destination.exists():
                continue
            if i % 25 == 0:
                gpu_idle()
            t0 = time.perf_counter()
            cand, dsc, ssc = hybrid_cand[i]
            scores = rr.score(ws(query['query']), [texts[int(c)] for c in cand])
            pair_score = {int(c): float(score) for c, score in zip(cand, scores)}
            branches = {}
            branch_inputs = {'hybrid': (cand, .7 * minmax(dsc) + .3 * minmax(ssc)),
                             'dense': (dense_cand[i], np.asarray(dense[dense_cand[i]], dtype=np.float32) @ qd[i].astype(np.float32))}
            for branch, (ids, first_scores) in branch_inputs.items():
                first_docs = {}
                for c in np.asarray(ids)[np.argsort(-first_scores, kind='stable')]:
                    first_docs.setdefault(int(doc_ids[c]), None)
                selected = set(list(first_docs)[:150])
                by_doc = {}
                for c in ids:
                    d = int(doc_ids[c])
                    if d in selected:
                        by_doc[d] = max(by_doc.get(d, -float('inf')), pair_score[int(c)])
                branches[branch] = sorted(by_doc.items(), key=lambda row: -row[1])
            atomic_json(destination, {'query_id': qid, **branches, 'rr_pairs': len(cand),
                                      'sparse_nonzero': int(np.count_nonzero(ssc)),
                                      'seconds': time.perf_counter() - t0})
            if (i+1) % 25 == 0:
                print(f'zh retrieve {i+1}/1200 ({time.perf_counter()-t_start:.1f}s this run)', flush=True)
        del rr
        torch.cuda.empty_cache()
    complete = [json.loads((out / f'q{int(q["id"])}.json').read_text('utf-8')) for q in qs]
    docs = pq.read_table(CHUNKS / 'docs.parquet', columns=['doc_id', 'domain']).to_pylist()
    domain_for = {r['doc_id']: r['domain'] for r in docs}
    stats = json.loads((RUN / 'extract_stats.json').read_text('utf-8'))
    # Denominator is all extracted documents, including thin; also report index coverage separately.
    sampled_docs = {d: s['n_docs'] for d, s in stats.items()}
    summary = {}
    for branch in ('hybrid', 'dense'):
        merged = {r['query_id']: merge_scores(vi[r['query_id']], r[branch]) for r in complete}
        summary[branch] = {'yield': yield_proxy(merged, domain_for, sampled_docs),
                           'zh_hits': sum(d in domain_for for rows in merged.values() for d, _ in rows),
                           'queries_with_zh': sum(any(d in domain_for for d, _ in rows) for rows in merged.values())}
        atomic_json(out / (branch + '_merged.json'), merged)
    summary['measurement'] = {'queries': 1200, 'rr_pairs': sum(r['rr_pairs'] for r in complete),
          'rr_query_seconds_sum': sum(r['seconds'] for r in complete),
          'queries_sparse_nonzero': sum(r['sparse_nonzero'] > 0 for r in complete),
          'same_zh_top150_doc_set_queries': sum({d for d, _ in r['hybrid']} == {d for d, _ in r['dense']} for r in complete),
          'LB': 'chưa đo', 'proxy_is_relevance': False}
    atomic_json(RUN / 'yield.json', summary)
    print(json.dumps(summary['measurement']), flush=True)


if __name__ == '__main__':
    retrieve()
