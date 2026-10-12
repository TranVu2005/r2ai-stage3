"""Exact dense top-k over zh segments (the dense half of r2ai.index.segments.scan_topk).

Global tie order (score descending, global chunk id descending) via segments.merge_topk.
"""
from __future__ import annotations

import numpy as np

from r2ai.index.segments import _validate_segments, merge_topk
from r2ai.retrieve.exact import dense_topk, doc_aligned_bounds


def dense_candidates(segments, qd, k, *, block_rows=16384, query_batch=128):
    """Return [(global ids int64, scores float32)] per query, exact over every row of every segment."""
    segments = list(segments)
    _validate_segments(segments)
    nq = len(qd)
    empty = lambda: ([np.zeros(0, np.int64) for _ in range(nq)], [np.zeros(0, np.float32) for _ in range(nq)])
    total = empty()
    for s in segments:
        local = empty()
        for lo, hi in zip(*(lambda b: (b[:-1], b[1:]))(doc_aligned_bounds(np.asarray(s.doc_id), block_rows))):
            block = np.asarray(s.dense[int(lo):int(hi)], dtype=np.float32)
            result = empty()
            for a in range(0, nq, query_batch):
                ids, scores = dense_topk(block, qd[a:a + query_batch], k, block_rows=len(block))
                result[0][a:a + query_batch] = [np.asarray(i, np.int64) + s.global_start + int(lo) for i in ids]
                result[1][a:a + query_batch] = scores
            del block
            local = merge_topk([local, result], k)
        total = merge_topk([total, local], k)
    return list(zip(*total))
