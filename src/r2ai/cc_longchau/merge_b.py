"""VS-cc-B: the CC longchau segments take part in the vi retrieval itself (flag-only; defaults and VS-append replay untouched).

  python -m r2ai.cc_longchau.merge_b prepare    # merged candidates (vi + CC), renormalisation report, merged docs/chunks dirs
  python -m r2ai.cc_longchau.merge_b gate       # vi-only replay of both rerank configs from the pair caches: must equal the old parquet
  python -m r2ai.cc_longchau.merge_b rerank     # the two rerank configs over the merged candidates; only pairs missing from the caches hit the GPU
  python -m r2ai.cc_longchau.merge_b build [--target-mean 115]   # builder -> expand -> rrf chunks -> rrf doc set (-> cut) -> slug (CC-text docs excluded)

Facade chunk ids: vi chunks keep their ids (0..N_VI-1); CC chunks follow in segment order (N_VI + running row). The merged first stage
is the exact merge of the two scans: dense top-200 and sparse top-200 over vi + CC (ties by global chunk id, descending, as the
scans), the union re-scored on both branches, min-max per query over that union, hybrid 0.7/0.3. The vi side is reconstructed from
vi_cand.chunks.parquet (checked: reproduces the recorded union for 1,200/1,200 queries). Reranker logits of the old runs are reused
(pair caches vi_k100_pairs.parquet); fp16 batch padding can only differ for pairs that were not scored before.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.paths import QUERY_FILE, WORK_DATA_DIR, chunks_file, index_dir, resolve_path
from r2ai.index.segments import take_segment_texts
from r2ai.retrieve.exact import take_texts, text_mmap
from r2ai.retrieve.run import minmax
from r2ai.retrieve import run_retrieval_k100 as RK
from r2ai.submit.best import load_config, sha256

from .build import active_segments, zroots
from .common import Roots, atomic_json, log_event
from .retrieve import open_segments

N_VI = 3_909_586
W = 0.7
CONFIG = 'configs/submission-best.yaml'
DEEP = dict(tier1_docs=100, tier2_docs=50, levels=())
R200 = dict(tier1_docs=200, tier2_docs=0, levels=(100,))


def _p(x):
    return str(resolve_path(x))


class Facade:
    """Index stand-in for run_retrieval_k100.tier1_pairs / rerank_order: doc_id by facade chunk id, texts of vi + CC chunks."""

    def __init__(self, roots: Roots, named):
        self.named = named
        self.starts = np.array([s.global_start for _, s in named], dtype=np.int64)
        self.offset = np.concatenate([[0], np.cumsum([len(s) for _, s in named])]).astype(np.int64)[:-1] + N_VI
        vi_doc = pq.read_table(chunks_file(256), columns=['doc_id'])['doc_id'].to_numpy()
        if len(vi_doc) != N_VI:
            raise ValueError('vi chunk count changed')
        self.doc_id = np.concatenate([vi_doc] + [np.asarray(s.doc_id, dtype=np.int64) for _, s in named])
        self.text, _ = text_mmap(chunks_file(256), index_dir(256) / 'text.arrow')
        self.segments = [s for _, s in named]

    def to_facade(self, gids):
        gids = np.asarray(gids, dtype=np.int64)
        seg = np.searchsorted(self.starts, gids, side='right') - 1
        return self.offset[seg] + (gids - self.starts[seg])

    def to_global(self, fids):
        fids = np.asarray(fids, dtype=np.int64)
        seg = np.searchsorted(self.offset, fids, side='right') - 1
        return self.starts[seg] + (fids - self.offset[seg])

    def texts(self, ids) -> list[str]:
        ids = np.asarray(ids, dtype=np.int64)
        out = [None] * len(ids)
        vi = ids < N_VI
        if vi.any():
            for i, t in zip(np.flatnonzero(vi), take_texts(self.text, ids[vi])):
                out[i] = t
        if (~vi).any():
            for i, t in zip(np.flatnonzero(~vi), take_segment_texts(self.segments, self.to_global(ids[~vi]))):
                out[i] = t
        return out


def top_k(ids, scores, k=200):
    """Indices of the top-k by (score desc, chunk id desc) -- the tie rule of the exact scans."""
    return np.lexsort((-ids, -scores))[:k]


def merge_query(vi, cc, k=200):
    """vi / cc: (ids, dense, sparse) with both scores for every member of their own union -> merged (ids, dense, sparse)."""
    ids = np.concatenate([vi[0], cc[0]])
    de = np.concatenate([vi[1], cc[1]])
    sp = np.concatenate([vi[2], cc[2]])
    d_sel = ids[top_k(ids, de, k)]
    s_idx = top_k(ids, sp, k)
    s_sel = ids[s_idx][sp[s_idx] > 0]
    union = np.union1d(d_sel, s_sel)
    pos = np.searchsorted(ids[np.argsort(ids, kind='stable')], union)
    order = np.argsort(ids, kind='stable')
    return union, de[order][pos], sp[order][pos]


def hybrid(de, sp, stats=None):
    """Hybrid 0.7/0.3 with per-query min-max over the union; stats=(dmin, dmax, smin, smax) overrides the normalisation range."""
    if stats is None:
        return W * minmax(de) + (1 - W) * minmax(sp)
    dmin, dmax, smin, smax = stats
    nd = (de - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(de)
    ns = (sp - smin) / (smax - smin) if smax > smin else np.zeros_like(sp)
    return W * nd + (1 - W) * ns


def doc_order(ids, doc_of, h):
    """Distinct docs by max hybrid; stable on the ascending chunk order (as write_candidates)."""
    seen = {}
    for j in np.argsort(-h, kind='stable'):
        seen.setdefault(int(doc_of[ids[j]]), float(h[j]))
    return list(seen), seen


def load_vi_candidates(cand_dir):
    t = pq.read_table(Path(cand_dir) / 'vi_cand.chunks.parquet').to_pandas()
    out = {}
    for q, g in t.groupby('query_id', sort=False):
        o = np.argsort(g['chunk_id'].values, kind='stable')
        out[int(q)] = (g['chunk_id'].values[o].astype(np.int64), g['dense'].values[o].astype(np.float32), g['sparse'].values[o].astype(np.float32))
    return out


def merged_candidates(roots: Roots, cand_dir, run):
    named = open_segments(roots)
    fac = Facade(roots, named)
    first = pickle.loads((roots.run / run / 'candidates.pkl').read_bytes())
    if first['segments'] != [n for n, _ in named]:
        raise ValueError('retrieval run segments differ from the active segments')
    qids = [int(x) for x in pq.read_table(QUERY_FILE, columns=['id'])['id'].to_pylist()]
    vi = load_vi_candidates(cand_dir)
    merged, vi_only, report = {}, {}, {'queries': len(qids)}
    changed_norm = changed_cc = set_changed_norm = set_changed_cc = 0
    docs_before, docs_cc = [], []
    for i, q in enumerate(qids):
        cid, cde, csp = first['hybrid'][i]
        cc = (fac.to_facade(cid), np.asarray(cde, np.float32), np.asarray(csp, np.float32))
        o = np.argsort(cc[0], kind='stable')
        cc = tuple(a[o] for a in cc)
        m = merge_query(vi[q], cc)
        merged[q] = m
        # stage O (old) = vi alone; stage R = old vi candidates, merged normalisation range; stage M = merged union (vi + CC)
        ids0, de0, sp0 = vi[q]
        h0 = hybrid(de0, sp0)
        old_docs, _ = doc_order(ids0, fac.doc_id, h0)
        rng = (float(m[1].min()), float(m[1].max()), float(m[2].min()), float(m[2].max()))
        r_docs, _ = doc_order(ids0, fac.doc_id, hybrid(de0, sp0, rng))
        m_docs, _ = doc_order(m[0], fac.doc_id, hybrid(m[1], m[2]))
        vi_docs = set(old_docs)
        m_vi = [d for d in m_docs if d in vi_docs]
        n_cc_docs = sum(d not in vi_docs for d in m_docs[:150])
        docs_cc.append(n_cc_docs)
        changed_norm += old_docs[:150] != r_docs[:150]
        set_changed_norm += set(old_docs[:150]) != set(r_docs[:150])
        changed_cc += r_docs[:150] != m_vi[:150]
        set_changed_cc += set(r_docs[:150]) != set(m_vi[:150])
        vi_only[q] = (old_docs[:150], r_docs[:150], m_vi[:150])
    n = len(qids)
    report |= {'vi_top150_order_changed_by_renormalisation_only': int(changed_norm), 'vi_top150_set_changed_by_renormalisation_only': int(set_changed_norm),
               'vi_top150_order_changed_by_cc_docs': int(changed_cc), 'vi_top150_set_changed_by_cc_docs': int(set_changed_cc),
               'cc_docs_in_merged_hybrid_top150': {'mean': float(np.mean(docs_cc)), 'queries_with_any': int(sum(x > 0 for x in docs_cc)), 'total': int(sum(docs_cc))},
               'note': 'hybrid doc order (max hybrid chunk), first 150 vi docs; O = old, R = old vi chunk set with the merged min-max range, '
                       'M = merged union restricted to vi docs; order change = list differs, set change = set differs'}
    return fac, merged, report, qids


def write_cand_dir(out: Path, fac, merged, qids):
    out.mkdir(parents=True, exist_ok=True)
    docs = {k: [] for k in ('query_id', 'rank', 'doc_id', 'score')}
    chunks = {k: [] for k in ('query_id', 'chunk_id', 'doc_id', 'dense', 'sparse', 'hybrid')}
    cnt = {k: [] for k in ('query_id', 'n_candidates', 'n_docs')}
    for q in qids:
        ids, de, sp = merged[q]
        h = hybrid(de, sp)
        order, score = doc_order(ids, fac.doc_id, h)
        for k, d in enumerate(order, 1):
            for key, v in zip(docs, (q, k, d, score[d])):
                docs[key].append(v)
        for key, v in zip(chunks, ([q] * len(ids), ids.tolist(), fac.doc_id[ids].tolist(), de.tolist(), sp.tolist(), h.tolist())):
            chunks[key] += v
        for key, v in zip(cnt, (q, len(ids), len(order))):
            cnt[key].append(v)
    pq.write_table(pa.table({'query_id': pa.array(docs['query_id'], pa.int64()), 'rank': pa.array(docs['rank'], pa.int32()),
                             'doc_id': pa.array(docs['doc_id'], pa.int64()), 'score': pa.array(docs['score'], pa.float32())}), out / 'vi_cand.docs.parquet')
    pq.write_table(pa.table({'query_id': pa.array(chunks['query_id'], pa.int64()), 'chunk_id': pa.array(chunks['chunk_id'], pa.int64()),
                             'doc_id': pa.array(chunks['doc_id'], pa.int64()), 'dense': pa.array(chunks['dense'], pa.float32()),
                             'sparse': pa.array(chunks['sparse'], pa.float32()), 'hybrid': pa.array(chunks['hybrid'], pa.float32())}),
                   out / 'vi_cand.chunks.parquet')
    pq.write_table(pa.table({'query_id': pa.array(cnt['query_id'], pa.int64()), 'n_candidates': pa.array(cnt['n_candidates'], pa.int32()),
                             'n_docs': pa.array(cnt['n_docs'], pa.int32())}), out / 'vi_cand.candidates.parquet')


def load_pair_cache(*paths):
    """{(query, chunk): score}; earlier paths win (the config's own run first)."""
    out = {}
    for p in paths:
        t = pq.read_table(_p(p), columns=['query_id', 'chunk_id', 'score']).to_pydict()
        for q, c, s in zip(t['query_id'], t['chunk_id'], t['score']):
            out.setdefault((int(q), int(c)), float(s))
    return out


class CachedReranker:
    """Scores from the pair cache; pairs outside it go to the real reranker (created on first use) in one call per group."""

    def __init__(self, cache, fac, allow_new=True):
        self.cache, self.fac, self.allow_new, self.rr = cache, fac, allow_new, None
        self.hit = self.new = 0

    def score(self, qid, query, ids):
        ids = [int(c) for c in ids]
        miss = [c for c in ids if (qid, c) not in self.cache]
        self.hit += len(ids) - len(miss)
        if miss:
            if not self.allow_new:
                raise KeyError(f'q{qid}: {len(miss)} pairs are not cached')
            if self.rr is None:
                from r2ai.index.bge_m3 import Reranker
                from r2ai.zh_full_ingest.common import require_gpu_idle
                require_gpu_idle(zroots(Roots()))
                self.rr = Reranker(max_len=512)
            sc = self.rr.score(query, self.fac.texts(miss)).tolist()
            self.cache.update({(qid, c): float(s) for c, s in zip(miss, sc)})
            self.new += len(miss)
        return {c: self.cache[(qid, c)] for c in ids}


def rerank_config(cfg, fac, cands, qrows, cache, out: Path, allow_new=True, label=''):
    """run_retrieval_k100's one_query (chunk_score_docs = 0, pair scores kept) over given candidates, scores via the pair cache."""
    from r2ai.retrieve.run import rerank_order, ws
    rr = CachedReranker(cache, fac, allow_new)
    rows = {k: [] for k in ('query_id', 'rank', 'doc_id', 'score', 'tier')}
    prow = {k: [] for k in ('query_id', 'doc_id', 'chunk_id', 'score', 'tier')}
    cstat = []
    k_max = cfg['tier1_docs'] + cfg['tier2_docs']
    t0 = time.time()
    for n, r in enumerate(qrows, 1):
        qid = int(r['id'])
        q = ws(r['query'])
        cand, dsc, ssc = cands[qid]
        nd = len(np.unique(fac.doc_id[cand]))
        cstat.append((qid, len(cand), nd))
        ranked, pairs, groups, _ = RK.nested_tier1(fac, cand, dsc, ssc, cfg['tier1_docs'], cfg['levels'])
        first = set(ranked[:RK.TIER1_DOCS])
        p_a = [c for c in groups[0] if int(fac.doc_id[c]) in first]
        p_b = [c for c in groups[0] if int(fac.doc_id[c]) not in first]
        s = dict(rr.score(qid, q, p_a)) if p_a else {}
        for p_x in (p_b, *groups[1:]):
            if p_x:
                s.update(rr.score(qid, q, p_x))
        tier1 = rerank_order(fac, pairs, ranked, s)
        dscore = {}
        for c in pairs:
            d = int(fac.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s[c])
        h = W * minmax(dsc) + (1 - W) * minmax(ssc)
        seen, tier2 = set(tier1), []
        for j in np.argsort(-h, kind='stable'):
            if len(tier2) >= cfg['tier2_docs']:
                break
            d = int(fac.doc_id[cand[j]])
            if d not in seen:
                seen.add(d)
                tier2.append(d)
        t2 = set(tier2)
        p2 = [int(c) for c in cand if int(fac.doc_id[c]) in t2]
        s2 = rr.score(qid, q, p2) if p2 else {}
        for c in p2:
            d = int(fac.doc_id[c])
            dscore[d] = max(dscore.get(d, -1e9), s2[c])
        tier2.sort(key=lambda d: -dscore[d])
        order = ([(d, 1) for d in tier1] + [(d, 2) for d in tier2])[:k_max]
        for k, (d, t) in enumerate(order, 1):
            for key, v in zip(rows, (qid, k, d, dscore[d], t)):
                rows[key].append(v)
        for tier, plist, sc in ((1, pairs, s), (2, p2, s2)):
            for c in plist:
                for key, v in zip(prow, (qid, int(fac.doc_id[c]), int(c), sc[int(c)], tier)):
                    prow[key].append(v)
        if n % 100 == 0:
            print(f'{label} {n}/{len(qrows)} cache_hit={rr.hit} new={rr.new} {time.time() - t0:.0f}s', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'query_id': pa.array(rows['query_id'], pa.int64()), 'rank': pa.array(rows['rank'], pa.int32()),
                             'doc_id': pa.array(rows['doc_id'], pa.int64()), 'score': pa.array(rows['score'], pa.float32()),
                             'tier': pa.array(rows['tier'], pa.int8())}), out / 'vi_k100.parquet')
    pq.write_table(pa.table({'query_id': pa.array(prow['query_id'], pa.int64()), 'doc_id': pa.array(prow['doc_id'], pa.int64()),
                             'chunk_id': pa.array(prow['chunk_id'], pa.int64()), 'score': pa.array(prow['score'], pa.float32()),
                             'tier': pa.array(prow['tier'], pa.int8())}), out / 'vi_k100_pairs.parquet')
    pq.write_table(pa.table({'query_id': pa.array([], pa.int64()), 'doc_id': pa.array([], pa.int64()), 'chunk_id': pa.array([], pa.int64()),
                             'score': pa.array([], pa.float32())}), out / 'vi_k100_chunk_scores.parquet')
    pq.write_table(pa.table({'query_id': pa.array([c[0] for c in cstat], pa.int64()), 'n_candidates': pa.array([c[1] for c in cstat], pa.int32()),
                             'n_docs': pa.array([c[2] for c in cstat], pa.int32())}), out / 'vi_k100.candidates.parquet')
    return {'pairs_cache_hit': rr.hit, 'pairs_new': rr.new, 'seconds': round(time.time() - t0, 1)}


def same_parquet(a, b, cols=('query_id', 'rank', 'doc_id', 'score', 'tier')):
    x, y = pq.read_table(a, columns=list(cols)), pq.read_table(b, columns=list(cols))
    return x.equals(y)


def paths(cfg):
    s, rds = cfg['submission'], cfg['postprocess']['rrf_doc_set']
    return {'deep_dir': _p(s['runs_dir']), 'r200': _p(rds['reranker_cache']), 'hyb': _p(rds['hybrid_ranks'])}


def cmd_gate(roots: Roots):
    """Both configs from the old pair caches over the OLD vi candidates only: must reproduce the old parquet bit for bit."""
    cfg = load_config(_p(CONFIG))
    pp = paths(cfg)
    cand_dir = Path(pp['hyb']).parent
    named = open_segments(roots)
    fac = Facade(roots, named)
    vi = load_vi_candidates(cand_dir)
    qrows = pq.read_table(QUERY_FILE).to_pylist()
    out = {}
    deep_pairs, r200_pairs = Path(pp['deep_dir']) / 'vi_k100_pairs.parquet', Path(pp['r200']).parent / 'vi_k100_pairs.parquet'
    for name, conf, own, other, ref in (('deep', DEEP, deep_pairs, r200_pairs, Path(pp['deep_dir']) / 'vi_k100.parquet'),
                                        ('r200', R200, r200_pairs, deep_pairs, Path(pp['r200']))):
        d = roots.guard(roots.run / 'B' / 'gate' / name)
        st = rerank_config(conf, fac, vi, qrows, load_pair_cache(own, other), d, allow_new=False, label=f'gate-{name}')
        out[name] = st | {'identical_to_old': same_parquet(d / 'vi_k100.parquet', ref)}
    atomic_json(roots, roots.run / 'B' / 'gate.json', out)
    return out


def cmd_prepare(roots: Roots, run: str):
    cfg = load_config(_p(CONFIG))
    pp = paths(cfg)
    fac, merged, report, qids = merged_candidates(roots, Path(pp['hyb']).parent, run)
    out = roots.guard(roots.run / 'B' / 'cand')
    write_cand_dir(out, fac, merged, qids)
    report['segments'] = [n for n, _ in fac.named]
    report['cc_chunks'] = int(len(fac.doc_id) - N_VI)
    atomic_json(roots, roots.run / 'B' / 'renorm_report.json', report)
    # merged chunks dir: docs.parquet = vi + CC docs (build only needs doc ids / groups, but keep every column)
    cdir = roots.guard(roots.run / 'B' / 'chunks')
    cdir.mkdir(parents=True, exist_ok=True)
    srcs = [chunks_file(256).parent / 'docs.parquet'] + [roots.chunks / n / 'docs.parquet' for n, _ in fac.named]
    schema = pq.ParquetFile(srcs[0]).schema_arrow
    with pq.ParquetWriter(cdir / 'docs.parquet', schema, compression='zstd') as w:
        for sp in srcs:
            for b in pq.ParquetFile(sp).iter_batches(batch_size=50000):
                w.write_table(pa.Table.from_batches([b]).cast(schema))
    # merged docs dir: hard links of data/docs_vi files (main repo copy, never OLD) + the CC docs files
    ddir = roots.guard(roots.run / 'B' / 'docs')
    ddir.mkdir(parents=True, exist_ok=True)
    for f in sorted((WORK_DATA_DIR / 'docs_vi').glob('*.parquet')):
        if not (ddir / f.name).exists():
            os.link(f, ddir / f.name)
    for f in sorted(roots.docs.glob('*.parquet')):
        if not (ddir / f.name).exists():
            os.link(f, ddir / f.name)
    return report


def cmd_rerank(roots: Roots):
    cfg = load_config(_p(CONFIG))
    pp = paths(cfg)
    named = open_segments(roots)
    fac = Facade(roots, named)
    qrows = pq.read_table(QUERY_FILE).to_pylist()
    # merged candidates are read back from B/cand (written by prepare) so rerank is independent of the CC first-stage pickle
    t = pq.read_table(roots.run / 'B' / 'cand' / 'vi_cand.chunks.parquet').to_pandas()
    cands = {}
    for q, g in t.groupby('query_id', sort=False):
        o = np.argsort(g['chunk_id'].values, kind='stable')
        cands[int(q)] = (g['chunk_id'].values[o].astype(np.int64), g['dense'].values[o].astype(np.float32), g['sparse'].values[o].astype(np.float32))
    deep_pairs, r200_pairs = Path(pp['deep_dir']) / 'vi_k100_pairs.parquet', Path(pp['r200']).parent / 'vi_k100_pairs.parquet'
    out = {}
    for name, conf, own, other in (('deep', DEEP, deep_pairs, r200_pairs), ('r200', R200, r200_pairs, deep_pairs)):
        d = roots.guard(roots.run / 'B' / name)
        if (d / 'vi_k100.parquet').exists():
            out[name] = {'skipped': 'exists'}
            continue
        cache = load_pair_cache(own, other)
        cc_cache = roots.run / 'B' / f'pair_cache_{name}.pkl'
        if cc_cache.exists():
            cache.update(pickle.loads(cc_cache.read_bytes()))
        out[name] = rerank_config(conf, fac, cands, qrows, cache, d, allow_new=True, label=name)
        # CC-side logits of this run stay reusable by the other config and by later batches
        new = {k: v for k, v in cache.items() if k[1] >= N_VI}
        cc_cache.write_bytes(pickle.dumps(new))
    atomic_json(roots, roots.run / 'B' / 'rerank_summary.json', out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prepare', 'gate', 'rerank', 'build'])
    ap.add_argument('--run', default=None, help='cc retrieval run (first stage) providing the CC candidates')
    ap.add_argument('--target-mean', type=float, default=None)
    ap.add_argument('--k-chunk', type=int, default=None, help='build: docs that get a chunk (config 50); smaller = smaller ZIP')
    ap.add_argument('--tag', default='')
    a = ap.parse_args(argv)
    roots = Roots()
    roots.preflight()
    from .retrieve import default_run
    if a.step == 'prepare':
        r = cmd_prepare(roots, a.run or default_run(roots))
    elif a.step == 'gate':
        r = cmd_gate(roots)
    elif a.step == 'rerank':
        r = cmd_rerank(roots)
    else:
        from . import build_b
        r = build_b.build(roots, a.target_mean, a.tag, a.k_chunk)
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
