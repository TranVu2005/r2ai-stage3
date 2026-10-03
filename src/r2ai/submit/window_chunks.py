"""Window chunks: one contiguous verbatim passage of <= N BGE-M3 tokens per (query, doc), centred on the best chunk.

Used by scripts/make_submission.py --chunk-mode window.

text of a doc  = `answer` if present (and inside the body), else `body` (same as chunk-mode full).
doc <= N tokens -> the whole text. Otherwise:
  * centre chunk = the answer/body chunk (t256) with the highest score: bge-reranker score from
    data/runs/vi_k100_chunk_scores.parquet (top-50 docs of the query); for other docs the dense score
    (query . chunk embedding of the t256 index) -> counted as `fallback`.
  * the text is cut into units: paragraphs (lines), over-long ones into sentences, over-long sentences into word
    groups (whitespace-free blobs by characters). The window starts at the unit holding the centre of the centre chunk
    and grows unit by unit towards the side whose neighbouring unit overlaps the higher-scored chunk, until no
    neighbour fits in N tokens. The window is text[first_unit.start : last_unit.end]: a verbatim substring.
Token counts: BGE-M3 tokenizer on text normalised as the scorer does (eval.scorer.normalize_metric), no special tokens;
the finished window is re-counted and shrunk (last added unit first) if the sum of unit counts under-estimated it.
"""
from __future__ import annotations

from r2ai.paths import chunks_file, index_dir

import re
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from r2ai.eval.scorer import normalize_metric



def count_tokens(tok, texts: list[str], batch: int = 32) -> list[int]:
    """BGE-M3 token counts of normalised texts, nothing cached (Tokenizer would keep every id tuple)."""
    out: list[int] = []
    for i in range(0, len(texts), batch):
        norm = [normalize_metric(t) for t in texts[i:i + batch]]
        ids = iter(tok._encode([n for n in norm if n]))
        out += [len(next(ids)) if n else 0 for n in norm]
    return out


class Windows:
    def __init__(self, n: int, tok, full: dict[int, str], off: dict[int, int], rerank: dict, qd: dict[int, np.ndarray],
                 target: int = 256, *, chunks_dir: Path | None = None, index_base: Path | None = None):
        self.n, self.tok, self.full, self.off, self.rerank, self.qd = n, tok, full, off, rerank, qd
        ch = pq.read_table(chunks_file(target, base=chunks_dir),
                           columns=['chunk_id', 'doc_id', 'field', 'char_start', 'char_end']).to_pydict()
        cid = np.asarray(ch['chunk_id'], dtype=np.int64)
        if not np.array_equal(cid, np.arange(len(cid))):
            raise RuntimeError('chunk_id is not the row index of dense.npy')
        m = np.isin(np.asarray(ch['field']), ['answer', 'body'])
        did = np.asarray(ch['doc_id'], dtype=np.int64)[m]
        order = np.argsort(did, kind='stable')
        self._did = did[order]
        self._cid = cid[m][order]
        self._cs = np.asarray(ch['char_start'], dtype=np.int64)[m][order]
        self._ce = np.asarray(ch['char_end'], dtype=np.int64)[m][order]
        self.dense = np.load(index_dir(target, base=index_base) / 'dense.npy', mmap_mode='r')
        self._dlen: dict[int, int] = {}
        self._units: dict[int, list] = {}
        self._memo: dict[tuple[int, int], tuple] = {}

    # ---- lengths / units
    def ensure_len(self, docs) -> None:
        todo = [d for d in dict.fromkeys(docs) if d not in self._dlen and d in self.full]
        for i in range(0, len(todo), 64):
            part = todo[i:i + 64]
            for d, n in zip(part, count_tokens(self.tok, [self.full[d] for d in part], batch=8)):
                self._dlen[d] = n

    def _force_split(self, base: str, a: int, b: int, n: int) -> list[tuple[int, int, int]]:
        step = max(1, int((b - a) * self.n / max(n, 1) * 0.9))
        spans = [(x, min(x + step, b)) for x in range(a, b, step)]
        out = []
        for (x, y), k in zip(spans, count_tokens(self.tok, [base[x:y] for x, y in spans], batch=8)):
            out += [(x, y, k)] if k <= self.n else self._force_split(base, x, y, k)
        return out

    def units(self, d: int) -> list[tuple[int, int, int]]:
        if d in self._units:
            return self._units[d]
        base, N = self.full[d], self.n
        paras = [(m.start(), m.end()) for m in re.finditer(r'[^\n]+', base)]
        out: list[tuple[int, int, int]] = []
        for (s, e), n in zip(paras, count_tokens(self.tok, [base[s:e] for s, e in paras], batch=16)):
            if n <= N:
                out.append((s, e, n))
                continue
            sents, last = [], s
            for m in re.finditer(r'(?<=[.!?…;:])\s+', base[s:e]):
                sents.append((last, s + m.start()))
                last = s + m.end()
            sents.append((last, e))
            for (a, b), k in zip(sents, count_tokens(self.tok, [base[a:b] for a, b in sents], batch=16)):
                if k <= N:
                    out.append((a, b, k))
                    continue
                words = [(a + m.start(), a + m.end()) for m in re.finditer(r'\S+', base[a:b])]
                g = max(1, int(N * 0.9 * len(words) / k))
                groups = [(words[i][0], words[min(i + g, len(words)) - 1][1]) for i in range(0, len(words), g)]
                for (x, y), kk in zip(groups, count_tokens(self.tok, [base[x:y] for x, y in groups], batch=8)):
                    out += [(x, y, kk)] if kk <= N else self._force_split(base, x, y, kk)
        self._units[d] = out
        return out

    # ---- chunk scores of one (query, doc): relative char ranges in the text + scores
    def _chunks(self, q: int, d: int):
        lo, hi = np.searchsorted(self._did, [d, d + 1])
        cid, cs, ce = self._cid[lo:hi], self._cs[lo:hi] - self.off[d], self._ce[lo:hi] - self.off[d]
        ok = (ce > 0) & (cs < len(self.full[d]))
        cid, cs, ce = cid[ok], cs[ok], ce[ok]
        sc = self.rerank.get((q, d))
        if sc is not None:
            return cs, ce, np.array([sc.get(int(c), -1e9) for c in cid], dtype=np.float32), False
        if len(cid) == 0:
            return cs, ce, np.zeros(0, np.float32), True
        return cs, ce, np.asarray(self.dense[cid], dtype=np.float32) @ self.qd[q].astype(np.float32), True

    def get(self, q: int, d: int) -> tuple[str, int, bool, bool]:
        """-> (text, token count, cut by window, centre from dense fallback)."""
        key = (q, d)
        if key in self._memo:
            return self._memo[key]
        base = self.full[d]
        self.ensure_len([d])
        if self._dlen[d] <= self.n:
            res = (base, self._dlen[d], False, False)
        else:
            res = self._window(q, d, base)
        self._memo[key] = res
        return res

    def _window(self, q: int, d: int, base: str):
        units = self.units(d)
        starts = [u[0] for u in units]
        cs, ce, sc, fallback = self._chunks(q, d)
        if len(sc):
            j = int(np.argmax(sc))
            mid = (max(int(cs[j]), 0) + min(int(ce[j]), len(base))) // 2
        else:
            mid = 0
        c = min(max(bisect_right(starts, mid) - 1, 0), len(units) - 1)

        def uscore(i: int) -> float:
            if not len(sc):
                return 0.0
            m = (cs < units[i][1]) & (ce > units[i][0])
            return float(sc[m].max()) if m.any() else -1e9

        lo = hi = c
        tot, added = units[c][2], []
        while True:
            opts = []
            if lo > 0 and tot + units[lo - 1][2] <= self.n:
                opts.append((uscore(lo - 1), 0, 'L'))
            if hi < len(units) - 1 and tot + units[hi + 1][2] <= self.n:
                opts.append((uscore(hi + 1), 1, 'R'))
            if not opts:
                break
            side = max(opts)[2]                                  # higher score; tie -> right
            if side == 'L':
                lo -= 1
                tot += units[lo][2]
            else:
                hi += 1
                tot += units[hi][2]
            added.append(side)
        text = base[units[lo][0]:units[hi][1]]
        n = count_tokens(self.tok, [text], batch=1)[0]
        while n > self.n and added:                              # sum of unit counts under-estimated
            if added.pop() == 'L':
                lo += 1
            else:
                hi -= 1
            text = base[units[lo][0]:units[hi][1]]
            n = count_tokens(self.tok, [text], batch=1)[0]
        return text, n, True, fallback
