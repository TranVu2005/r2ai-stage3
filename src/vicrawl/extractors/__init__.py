"""Extractor registry: site specs first, trafilatura generic as the fallback."""
from __future__ import annotations

from r2ai.paths import ROOT

import re

from .base import Doc, Spec, extract_with_spec, fix_premature_close, norm, parse_html, strip_site_junk
from .sites import SPECS

__all__ = ['Doc', 'extract_doc', 'SPECS']


def generic_extract(html: str) -> Doc:
    import trafilatura
    html = fix_premature_close(html)
    tree = None
    try:
        tree = parse_html(html)
    except Exception:  # noqa: BLE001
        pass
    text = ''
    try:
        text = trafilatura.extract(html, include_tables=True, favor_recall=True, include_comments=False, output_format='txt') or ''
    except Exception:  # noqa: BLE001
        pass
    paras = [norm(line) for line in text.split('\n') if norm(line)]
    title = description = ''
    heads: list[str] = []
    if tree is not None:
        from .base import meta
        h1 = tree.xpath('//h1')
        title = norm(h1[0].text_content()) if h1 else meta(tree, 'og:title') or norm(' '.join(tree.xpath('//title/text()')))
        description = meta(tree, 'og:description', 'description')
        pset = set(paras)
        heads = [t for t in dict.fromkeys(norm(h.text_content()) for h in tree.xpath('//h2|//h3|//h4')) if t and t in pset]
    return Doc(title=title, description=description, headings=heads, paragraphs=paras, extractor='generic')


def extract_doc(html: str, extractor: str = 'generic', domain: str | None = None) -> Doc:
    """`domain` (the doc's own site) enables the site-level junk rules in `strip_site_junk`."""
    spec = SPECS.get(extractor)
    if spec is not None:
        try:
            doc = extract_with_spec(html, spec, domain)
        except Exception:  # noqa: BLE001
            doc = None
        if doc is not None:
            return doc
    doc = generic_extract(html)
    if domain:
        doc.paragraphs = strip_site_junk(doc.paragraphs, domain)
        doc.headings = [h for h in doc.headings if h in set(doc.paragraphs)]
    return doc
