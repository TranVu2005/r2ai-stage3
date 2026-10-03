"""Selector-driven article / Q&A extraction that keeps paragraph boundaries and original wording.

Only whitespace is normalised (runs of space/nbsp/zero-width -> one space). No translation, no rewriting."""
from __future__ import annotations

from r2ai.paths import ROOT

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from lxml import etree
from lxml import html as lhtml

WS = re.compile(r'[\s ​‌‍﻿]+')
INLINE = {'a', 'span', 'strong', 'b', 'em', 'i', 'u', 'sup', 'sub', 'small', 'mark', 'font', 'abbr', 'label', 'code', 's', 'del', 'ins', 'time',
          'cite', 'q', 'bdi', 'big', 'tt', 'kbd', 'var', 'wbr', 'img', 'strike', 'samp', 'dfn', 'data', 'bdo', 'nobr'}
ALWAYS_DROP = ('script', 'style', 'noscript', 'iframe', 'svg', 'button', 'input', 'select', 'textarea', 'nav', 'figure', 'figcaption',
               'aside', 'video', 'audio', 'canvas', 'template', 'adrender', 'object', 'embed', 'picture')
HEADING_TAGS = ('h2', 'h3', 'h4')

# Paragraphs that are never article text on any site (related-links, credits, captions).
GENERIC_JUNK = re.compile(
    r'^(?:xem thêm|đọc thêm|tin liên quan|bài viết liên quan|bài liên quan|bài viết khác|có thể bạn quan tâm|có thể bạn chưa biết|đừng bỏ lỡ|tham khảo thêm|xem tiếp|'
    r'ảnh minh họa|\(?ảnh\s*:|nguồn ảnh|ảnh\s*:|theo dõi (?:.+ )?trên|chia sẻ bài viết|từ khóa|từ khoá|tags?\s*:|nguồn\s*:)', re.I)


def norm(s: str | None) -> str:
    return WS.sub(' ', s or '').strip()


EDGE_LINES = 3                                   # caps-label rule only looks at the first/last N paragraphs
END_PUNCT = tuple('.!?:;…,')


def _is_caps_label(p: str) -> bool:
    """Standalone ALL-CAPS label of <=4 words without closing punctuation ("QUỐC HỘI", "VŨ HOÀNG")."""
    return (any(c.isalpha() for c in p) and p == p.upper() and len(p.split()) <= 4
            and not p.rstrip().endswith(END_PUNCT))


def _is_own_domain(p: str, domain: str) -> bool:
    return p.strip().lower().removeprefix('www.') == domain.lower().removeprefix('www.')


def strip_site_junk(paragraphs: list[str], domain: str | None) -> list[str]:
    """Drop lines that are only the doc's own bare domain, and ALL-CAPS section labels in the first/last 3 lines."""
    if not domain:
        return paragraphs
    n = len(paragraphs)
    out = []
    for i, p in enumerate(paragraphs):
        if _is_own_domain(p, domain):
            continue
        if (i < EDGE_LINES or i >= n - EDGE_LINES) and _is_caps_label(p):
            continue
        out.append(p)
    return out


@dataclass
class Spec:
    name: str
    content: list[str]
    remove: list[str] = field(default_factory=list)
    title: list[str] = field(default_factory=lambda: ['h1'])
    sapo: list[str] = field(default_factory=list)
    junk: list[str] = field(default_factory=list)          # regexes; matching paragraph is dropped
    cut_after: list[str] = field(default_factory=list)     # regex; matching paragraph and everything after is dropped
    qa: Callable | None = None                              # (tree, paragraphs) -> (question, answer) | None
    ld_description: bool = False                            # prefer JSON-LD description over generic <meta>
    body_drop: list[str] = field(default_factory=list)     # regexes dropped from the body only, after the Q&A split ("Hỏi" / "Trả lời" labels)
    scrub: list[str] = field(default_factory=list)         # regexes deleted inside paragraph / description text (CMS escape debris)
    drop_title_para: bool = False                           # drop body paragraphs that only repeat the title
    sapo_fallback: bool = False                             # no usable body (video pages): the sapo becomes the body


@dataclass
class Doc:
    title: str = ''
    description: str = ''
    headings: list[str] = field(default_factory=list)
    question: str = ''
    answer: str = ''
    paragraphs: list[str] = field(default_factory=list)
    extractor: str = ''

    @property
    def body(self) -> str:
        return '\n\n'.join(self.paragraphs)


PREMATURE_CLOSE = re.compile(r'</(?:html|body)\s*>(?=.*?<(?:body|div|p|article)\b)', re.I | re.S)


def fix_premature_close(html: str) -> str:
    """Drop </html> / </body> that are followed by more page content (baotayninh.vn: `<html> </html> <head>…`);
    lxml and trafilatura otherwise stop there and see an empty page."""
    return PREMATURE_CLOSE.sub('', html)


def parse_html(html: str):
    html = fix_premature_close(re.sub(r'^\s*<\?xml[^>]*>', '', html))
    tree = lhtml.fromstring(html)
    etree.strip_elements(tree, etree.Comment, etree.ProcessingInstruction, with_tail=False)
    return tree


def _is_inline(el) -> bool:
    return isinstance(el.tag, str) and (el.tag in INLINE or ':' in el.tag)


def _has_block(el) -> bool:
    return any(isinstance(d.tag, str) and not _is_inline(d) for d in el.iterdescendants())


class _Walker:
    """Flatten a DOM subtree to paragraphs. Block boundaries and <br><br> split paragraphs; single <br> is a space."""

    def __init__(self):
        self.out: list[str] = []
        self.buf: list[str] = []
        self.br = 0

    def flush(self):
        t = norm(''.join(self.buf))
        if t:
            self.out.append(t)
        self.buf, self.br = [], 0

    def text(self, t):
        if t:
            self.buf.append(t)
            if t.strip():
                self.br = 0

    def walk(self, el):
        self.text(el.text)
        for ch in el:
            tag = ch.tag
            if not isinstance(tag, str):
                self.text(ch.tail)
                continue
            if tag == 'br':
                self.br += 1
                if self.br >= 2:
                    self.flush()
                    self.br = 2
                else:
                    self.buf.append(' ')
            elif tag == 'tr':
                self.flush()
                cells = [norm(c.text_content()) for c in ch if isinstance(c.tag, str) and c.tag in ('td', 'th')]
                row = ' | '.join(c for c in cells if c)
                if row:
                    self.out.append(row)
            elif _is_inline(ch) and not _has_block(ch):
                self.walk(ch)
            else:
                self.flush()
                self.walk(ch)
                self.flush()
            self.text(ch.tail)


def blocks(el) -> list[str]:
    w = _Walker()
    w.walk(el)
    w.flush()
    return w.out


def _link_density(el) -> float:
    total = len(norm(el.text_content()))
    if not total:
        return 0.0
    return sum(len(norm(a.text_content())) for a in el.iter('a')) / total


def select(tree, selectors: list[str]):
    for sel in selectors:
        try:
            found = tree.cssselect(sel)
        except Exception:  # noqa: BLE001 - bad selector in config must not kill the crawl
            continue
        if found:
            yield sel, found


def first_text(tree, selectors: list[str]) -> str:
    for _, found in select(tree, selectors):
        for el in found:
            t = norm(el.text_content())
            if t:
                return t
    return ''


def meta(tree, *names) -> str:
    for n in names:
        v = tree.xpath(f'//meta[@property="{n}"]/@content|//meta[@name="{n}"]/@content')
        if v and norm(v[0]):
            return norm(v[0])
    return ''


def ld_json(tree) -> list[dict]:
    out = []
    for s in tree.xpath('//script[@type="application/ld+json"]/text()'):
        try:
            j = json.loads(s)
        except ValueError:
            continue
        items = j if isinstance(j, list) else j.get('@graph', [j]) if isinstance(j, dict) else []
        out += [i for i in items if isinstance(i, dict)]
    return out


def ld_description(tree) -> str:
    for o in ld_json(tree):
        t = o.get('@type')
        t = t if isinstance(t, list) else [t]
        if any(x in ('NewsArticle', 'Article', 'BlogPosting', 'MedicalWebPage') for x in t) and o.get('description'):
            return norm(str(o['description']))
    return ''


def _scrub(text: str, patterns) -> str:
    for c in patterns:
        text = c.sub('', text)
    return norm(text)


def _strip_domain_lines(text: str, domain: str | None) -> str:
    if not domain:
        return text
    return '\n\n'.join(p for p in text.split('\n\n') if not _is_own_domain(p, domain))


def extract_with_spec(html: str, spec: Spec, domain: str | None = None) -> Doc | None:
    """Return None when the spec's container is absent / empty (caller falls back to the generic extractor)."""
    tree = parse_html(html)
    # metadata first: some of it lives in nodes the cleaning pass deletes
    title = first_text(tree, spec.title) or meta(tree, 'og:title') or norm(' '.join(tree.xpath('//title/text()')))
    description = ''
    if spec.ld_description:
        description = ld_description(tree)
    full_tree = tree
    description = description or first_text(full_tree, spec.sapo) or meta(full_tree, 'og:description', 'description')
    scrub = [re.compile(x, re.I) for x in spec.scrub]
    if scrub:
        description = _scrub(description, scrub)
    qa_tree = parse_html(html) if spec.qa else None
    for tag in ALWAYS_DROP:
        etree.strip_elements(tree, tag, with_tail=False)
    for sel in spec.remove:
        try:
            for el in tree.cssselect(sel):
                if el.getparent() is not None:
                    el.drop_tree()
        except Exception:  # noqa: BLE001
            continue
    node = None
    for _, found in select(tree, spec.content):
        node = found[0]
        if len(norm(node.text_content())) >= 150:
            break
        node = None
    if node is None and not spec.sapo_fallback:
        return None
    junk = [re.compile(p, re.I) for p in spec.junk]
    cut = [re.compile(p, re.I) for p in spec.cut_after]
    # drop link-only containers (related-article boxes) before flattening
    for el in list(node.iter('div', 'ul', 'ol', 'section')) if node is not None else []:
        if el is not node and el.getparent() is not None:
            t = norm(el.text_content())
            if t and len(t) < 700 and _link_density(el) > 0.8:
                el.drop_tree()
    for el in list(node.iter('p', 'li')) if node is not None else []:      # "<p><a>Related article title</a></p>" lines
        if el.getparent() is not None:
            t = norm(el.text_content())
            if t and len(t) < 250 and _link_density(el) >= 0.98 and len(el.xpath('.//a')) == 1:
                el.drop_tree()
    paras: list[str] = []
    for p in blocks(node) if node is not None else []:
        if scrub:
            p = _scrub(p, scrub)
            if not p:
                continue
        if any(c.search(p) for c in cut):
            break
        if GENERIC_JUNK.search(p) or any(j.search(p) for j in junk):
            continue
        if paras and paras[-1] == p:
            continue
        paras.append(p)
    qa_paras, paras = paras, strip_site_junk(paras, domain)   # Q&A markers may be caps labels: split on the unfiltered list
    if spec.body_drop:
        drop = [re.compile(x, re.I) for x in spec.body_drop]
        paras = [p for p in paras if not any(d.search(p) for d in drop)]
    if spec.drop_title_para and title:
        paras = [p for p in paras if p.casefold() != title.casefold()]
    if sum(len(p) for p in paras) < 100:
        if not (spec.sapo_fallback and len(description) >= 100):
            return None
        qa_paras = paras = strip_site_junk([description], domain)
    heads = []
    pset = set(paras)
    for h in node.xpath('.//h2|.//h3|.//h4') if node is not None else []:
        t = norm(h.text_content())
        if t and t in pset and t not in heads:
            heads.append(t)
    doc = Doc(title=title, description=description, headings=heads, paragraphs=paras, extractor=spec.name)
    if spec.qa:
        res = spec.qa(qa_tree, qa_paras)
        if res:
            doc.question, doc.answer = (_strip_domain_lines(x, domain) for x in res)
    return doc
