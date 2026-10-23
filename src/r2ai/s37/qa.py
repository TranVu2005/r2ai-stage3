"""120ask question/answer structure for S37: per-answer extraction, quality flag and the two chunk families.

Extraction keeps the existing 120ask extractor (r2ai.zh_full_ingest.extract) and only adds the split of its `answer` into one
string per answer node (`.b_anscont_cont`, page order). doc_text = title + question + answers in page order (the canonical
doc_layout rule), so t256 chunks come from the canonical chunker unchanged.

Structural chunks (chunks_struct): one chunk = question + one answer, as a verbatim span of doc_text.
  * answer 1 directly follows the question in doc_text, so its chunk is the contiguous span question..answer 1;
  * a long answer is cut with the canonical t256 paragraph chunker, the question stays in the first piece only;
  * answer k >= 2 is not adjacent to the question, so (to stay verbatim) its chunks hold the answer alone.
"""
from __future__ import annotations

import re

HOST = '120ask.com'
GOOD_CP, GOOD_HAN = 200, 50           # good zh text: >= 200 code points OR >= 50 Han characters (S22 rule); below -> thin
TARGET = 256
WS = re.compile(r'\s+')
HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]')


def quality(body: str) -> tuple[int, int, str]:
    clean = WS.sub(' ', body).strip()
    cp, han = len(clean), len(HAN.findall(clean))
    return cp, han, ('good' if cp >= GOOD_CP or han >= GOOD_HAN else 'thin')


def split_answers(html: str, doc) -> list[str]:
    """One string per answer node. Falls back to the single joined answer when the split cannot reproduce doc.answer."""
    whole = (doc.answer or '').strip()
    if not whole:
        return []
    if doc.extractor != 'zh_' + HOST:
        return [whole]
    from vicrawl.extractors.base import blocks, parse_html
    from r2ai.zh_sample.extract import DROP_TAGS, SELECTORS
    tree = parse_html(html)
    for node in tree.xpath('|'.join('//' + tag for tag in DROP_TAGS)):
        if node.getparent() is not None:
            node.drop_tree()
    for selector in SELECTORS[HOST][1]:
        nodes = tree.cssselect(selector)
        if nodes:
            answers = ['\n\n'.join(p for p in blocks(n) if p) for n in nodes]
            answers = [a for a in answers if a]
            return answers if '\n\n'.join(answers) == doc.answer else [whole]
    return [whole]


def extract(html: str) -> dict:
    """-> the layout record {title, question, answer, answers, body, paragraphs} plus extractor name and quality numbers."""
    from r2ai.zh_full_ingest.extract import extract as base_extract
    doc = base_extract(html, HOST)
    answers = split_answers(html, doc)
    cp, han, q = quality(doc.body)
    return {'title': doc.title, 'question': doc.question, 'answer': doc.answer, 'answers': answers, 'body': doc.body,
            'paragraphs': doc.paragraphs, 'extractor': doc.extractor, 'body_cp': cp, 'han': han, 'quality': q}


def struct_chunks(doc_text: str, r: dict, paras: list, ntok, target: int = TARGET) -> list[tuple[str, int, int, int]]:
    """-> [(field, start, end, answer_idx)] verbatim spans of doc_text. field: qa | answer | question."""
    from r2ai.index.chunk import chunk_paragraphs
    body = r['body']
    q = (r.get('question') or '').strip()
    answers = r.get('answers') or []
    b0 = len(doc_text) - len(body)                      # body is the last part of doc_text
    if q and q in body:
        q_s = b0 + body.find(q)
    elif q and q in doc_text:
        q_s = doc_text.find(q)
    else:
        q_s = None
    q_e = q_s + len(q) if q_s is not None else None
    out = []
    off = (q_e - b0) if (q_e is not None and q_e >= b0) else 0
    for k, a in enumerate(answers):
        i = body.find(a, max(off, 0))
        if i < 0:
            continue
        a_s, a_e = b0 + i, b0 + i + len(a)
        off = i + len(a)
        inside = [p for p in paras if p[1] >= a_s and p[2] <= a_e]
        if not inside:
            continue
        pieces = chunk_paragraphs(doc_text, inside, [ntok(doc_text[s:e]) for _, s, e in inside], target, ntok)
        for j, (f, s, e) in enumerate(pieces):
            if k == 0 and j == 0 and q_s is not None and q_e <= s and not doc_text[q_e:s].strip():
                out.append(('qa', q_s, e, k))                # question + answer 1 (first piece) is contiguous
            else:
                out.append(('answer', s, e, k))
    if not answers and q_s is not None:
        out.append(('question', q_s, q_e, -1))
    return out
