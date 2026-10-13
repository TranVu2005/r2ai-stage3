"""Structure-aware chunker (probe candidate, not wired into the default pipeline).

Same inputs and output shape as r2ai.index.chunk.chunk_paragraphs: paragraphs (field, start, end) of a doc_text, returns
(field, start, end) spans with doc_text[start:end] == chunk text, so numbers, units and negations are never rewritten.

* heading = a paragraph listed in the extractor's `headings`, or a short label ending with ":" (e.g. "Nguyên liệu:"), or a
  short numbered / markdown title without sentence-final punctuation. A heading is glued to the first block below it and
  never ends a chunk.
* a question paragraph (field "question") opens a section with its answer paragraphs: the Q&A pair shares a chunk whenever it
  fits in `target` tokens (field "answer" if the chunk holds answer text).
* a run of list items / table rows is one block; it is split at item boundaries only when it alone exceeds `target`.
* a section (heading + blocks until the next heading / question) that fits in `target` is one chunk; consecutive sections
  are merged only while the open chunk is smaller than `min_tokens`. No overlap between chunks.
* an over-long block is split by r2ai.index.chunk._split_long (sentence ends, then whitespace).
Token counts of paragraphs are summed (separators ignored), so a chunk may exceed `target` by a few tokens.
"""
from __future__ import annotations

import re

from r2ai.index.chunk import _split_long

LIST_ITEM = re.compile(r'^\s*(?:[-•*·–+]|\d{1,2}[.)])\s+\S')
NUMBERED_TITLE = re.compile(r'^(?:#{1,6}\s+\S|\d{1,2}(?:\.\d{1,2})*[.)]\s+\S)')
SENT_FINAL = '.!?;…。！？'
HEADING_MAX_CHARS = 120
LABEL_MAX_CHARS = 80


def _norm(s: str) -> str:
    return ' '.join(s.split())


def classify(text: str, headings: set[str]) -> str:
    """-> 'heading' | 'item' | 'row' | 'para'."""
    t = text.strip()
    if _norm(t) in headings:
        return 'heading'
    if '\n' in t:
        return 'para'
    if t.endswith(':') and len(t) <= LABEL_MAX_CHARS and not LIST_ITEM.match(t):
        return 'heading'
    if len(t) <= HEADING_MAX_CHARS and NUMBERED_TITLE.match(t) and t[-1] not in SENT_FINAL and not t.endswith(':'):
        return 'heading'
    if t.count('|') >= 2 or '\t' in t:
        return 'row'
    if LIST_ITEM.match(t):
        return 'item'
    return 'para'


def structured_chunks(doc_text: str, paras: list[tuple[str, int, int]], ptoks: list[int], target: int, ntok,
                      headings=(), min_tokens: int | None = None) -> list[tuple[str, int, int]]:
    if min_tokens is None:
        min_tokens = target // 4
    hs = {_norm(h) for h in headings if h and h.strip()}
    n = len(paras)
    kind = [classify(doc_text[s:e], hs) if f != 'question' else 'heading' for f, s, e in paras]
    # blocks: [first, last] paragraph index; a run of items / rows is one block
    blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        if kind[i] in ('item', 'row'):
            j = i
            while j + 1 < n and kind[j + 1] == kind[i]:
                j += 1
            blocks.append((i, j, 'run'))
            i = j + 1
        else:
            blocks.append((i, i, kind[i]))
            i += 1
    btok = [sum(ptoks[a:b + 1]) for a, b, _ in blocks]
    # sections: a heading block starts one and is glued to the following block
    sections: list[list[int]] = []
    for k, (_, _, bk) in enumerate(blocks):
        if bk == 'heading' and not (sections and len(sections[-1]) == 1 and blocks[sections[-1][0]][2] == 'heading'):
            sections.append([k])
        elif not sections:
            sections.append([k])
        else:
            sections[-1].append(k)

    out: list[tuple[str, int, int]] = []
    cur: list[int] = []               # block indices of the open chunk
    cur_tok = 0

    def span(first_para: int, last_para: int):
        field = 'answer' if any(paras[p][0] == 'answer' for p in range(first_para, last_para + 1)) else 'body'
        return field, paras[first_para][1], paras[last_para][2]

    def flush():
        nonlocal cur, cur_tok
        if cur:
            out.append(span(blocks[cur[0]][0], blocks[cur[-1]][1]))
        cur, cur_tok = [], 0

    def emit_pieces(first_para: int, last_para: int):
        """One block too large for a chunk: split paragraph-wise, then by sentence."""
        k = first_para
        while k <= last_para:
            tot, m = ptoks[k], k + 1
            if ptoks[k] > 1.5 * target:
                s, e = paras[k][1], paras[k][2]
                out.extend((paras[k][0] if paras[k][0] != 'question' else 'body', a, b) for a, b in _split_long(doc_text[s:e], s, ntok, target))
                k += 1
                continue
            while m <= last_para and ptoks[m] <= 1.5 * target and tot + ptoks[m] <= target:
                tot += ptoks[m]
                m += 1
            out.append(span(k, m - 1))
            k = m

    for sec in sections:
        stok = sum(btok[b] for b in sec)
        if stok <= target:
            if cur and cur_tok < min_tokens and cur_tok + stok <= target:
                cur += sec
                cur_tok += stok
            else:
                flush()
                cur, cur_tok = list(sec), stok
            continue
        flush()
        # oversized section: heading + first block stay together, the rest is packed block by block
        units = [sec[:2]] + [[b] for b in sec[2:]] if blocks[sec[0]][2] == 'heading' and len(sec) > 1 else [[b] for b in sec]
        for unit in units:
            utok = sum(btok[b] for b in unit)
            if utok > target:
                flush()
                lead = blocks[unit[0]][0] if blocks[unit[0]][2] == 'heading' else None
                a, b = blocks[unit[0]][0], blocks[unit[-1]][1]
                if lead is not None and len(unit) > 1:
                    # heading rides on the first piece of the block below it
                    first = len(out)
                    emit_pieces(blocks[unit[1]][0], b)
                    f, _, e = out[first]
                    out[first] = (f, paras[a][1], e)
                else:
                    emit_pieces(a, b)
            elif cur and cur_tok + utok > target:
                flush()
                cur, cur_tok = list(unit), utok
            else:
                cur += unit
                cur_tok += utok
    if cur and out and all(blocks[b][2] == 'heading' for b in cur):
        f, a, _ = out[-1]                 # a lone heading at the end of the doc rides on the previous chunk
        out[-1] = (f, a, paras[blocks[cur[-1]][1]][2])
        cur = []
    flush()
    return out
