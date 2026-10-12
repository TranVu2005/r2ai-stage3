"""Adaptive cut of the primary relevant_docs of VS-append (S5, opt-in; code defaults and configs/submission-best.yaml unchanged).

Applied inside r2ai.submit.rrf_docs.rrf_doc_submission (cut=...), i.e. BEFORE the H1-expand cluster mates and slug_append.
Input per query: the RRF primary docs (RD150 order, <= 150) and their reranker logit (score of --rerank-run, tier-1 = 200).
Kept docs stay in RRF order; relevant_chunks are copied from the base untouched.

  A1(tau): keep docs with logit >= tau; fewer than `floor` -> the `floor` best by logit; at most `ceil` (best by logit).
  A2:      sort the docs by logit desc; keep the first i, i in [floor, n-1], where logit[i-1] - logit[i] is largest (first max).
  protect: primary docs whose ids (group + cluster mates) cover a doc_id of the base relevant_chunks are always kept, because every
           chunk doc_id must stay inside relevant_docs (validate_submission) and relevant_chunks are not touched (rrf_docs passes
           the set; empty set = no protection).
"""
from __future__ import annotations

import numpy as np


def _by_logit(docs, score):
    pos = {d: i for i, d in enumerate(docs)}
    return sorted(docs, key=lambda d: (-score[d], pos[d]))


def cut_a1(docs, score, tau, floor=20, ceil=150, protect=()):
    """(kept docs in RRF order, hit_floor, hit_ceil) -- floor/ceil flags refer to the logit rule before `protect`."""
    keep = [d for d in docs if score[d] >= tau]
    hit_floor, hit_ceil = len(keep) < floor, len(keep) > ceil
    if hit_floor or hit_ceil:
        keep = _by_logit(docs, score)[:floor if hit_floor else ceil]
    ks = set(keep) | set(protect)
    return [d for d in docs if d in ks], hit_floor, hit_ceil


def cut_a2(docs, score, floor=20, protect=()):
    """(kept docs in RRF order, hit_floor, hit_ceil): cut at the largest logit drop of the logit-sorted list."""
    order = _by_logit(docs, score)
    n = len(order)
    if n <= floor:
        ks = set(order) | set(protect)
        return [d for d in docs if d in ks], True, False
    s = np.array([score[d] for d in order], dtype=float)
    gap = s[floor - 1:n - 1] - s[floor:n]             # gap[j] = drop after keeping floor + j docs
    i = floor + int(np.argmax(gap))
    ks = set(order[:i]) | set(protect)
    return [d for d in docs if d in ks], i == floor, False


def make_cut(mode, *, tau=None, floor=20, ceil=150, protect_chunks=True):
    """cut(docs, score, protect) -> (kept, hit_floor, hit_ceil) for `a1` / `a2`."""
    if mode == 'a1':
        if tau is None:
            raise ValueError('a1 needs tau')
        fn = lambda docs, score, protect=(): cut_a1(docs, score, tau, floor, ceil, protect)
    elif mode == 'a2':
        fn = lambda docs, score, protect=(): cut_a2(docs, score, floor, protect)
    else:
        raise ValueError(f'unknown cut mode {mode}')
    fn.protect_chunks = protect_chunks      # False: rrf_docs passes no protected docs (chunk doc_ids may leave relevant_docs)
    return fn


def pool_scores(rerank_run) -> dict[int, dict[int, float]]:
    """query -> doc -> reranker logit of the pool cache (first row per doc)."""
    import pyarrow.parquet as pq
    from r2ai.paths import resolve_path
    t = pq.read_table(resolve_path(rerank_run), columns=['query_id', 'doc_id', 'score']).to_pydict()
    out: dict[int, dict[int, float]] = {}
    for q, d, s in zip(t['query_id'], t['doc_id'], t['score']):
        out.setdefault(int(q), {}).setdefault(int(d), float(s))
    return out
