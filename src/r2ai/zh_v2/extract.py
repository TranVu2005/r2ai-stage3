"""Extractor for the v2 zh hosts that were never in the zh_full plan (S11 D13); every other host is delegated unchanged.

Rules come from the 5-URL probes (out/runs/zh-v2/probe/html/) and are checked on the crawled pages:
  ask.39.net            Q&A. question = `.ask_cont` (title, onset time, supplement) minus the related-search box, answer =
                        every `.sele_txt` (selected doctor answers, doctor name/hospital line dropped). Doc.question and
                        Doc.answer are filled, so the t256 chunker emits `question` and `answer` chunks as for 120ask; body
                        keeps the question-answer pairs in page order.
  *.39.net articles     `div.art_content` (baby, heart, fk, nk, cancer, tnb, woman, food, ek, man, shen, gan, wei, sj,
                        fitness, gk, pf, gc, byby).
  familydoctor.com.cn   `#viewContent` (also baidianfeng./qy. subdomains).
A page without its container falls back to the generic zh_full extractor, never to an empty doc.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from vicrawl.extractors.base import Doc, blocks, meta, norm, parse_html
from r2ai.zh_full_ingest import extract as base
from r2ai.zh_sample.extract import DROP_TAGS

ARTICLE = {
    **{d: (['div.art_content'], []) for d in (
        'baby.39.net', 'heart.39.net', 'fk.39.net', 'nk.39.net', 'cancer.39.net', 'tnb.39.net', 'woman.39.net', 'food.39.net',
        'ek.39.net', 'man.39.net', 'shen.39.net', 'gan.39.net', 'wei.39.net', 'sj.39.net', 'fitness.39.net', 'gk.39.net',
        'pf.39.net', 'gc.39.net', 'byby.39.net')},
    'familydoctor.com.cn': (['#viewContent'], []),
    'baidianfeng.familydoctor.com.cn': (['#viewContent'], []),
    'qy.familydoctor.com.cn': (['#viewContent'], []),
}
QA = {
    'ask.39.net': {'question': ['.ask_cont'], 'question_drop': ['section.sys-box', '.sys-label'], 'answer': ['.sele_txt'],
                   'answer_drop': []},
}
JUNK_39 = [r'39健康网\(www\.39\.net\)专稿，未经书面授权请勿转载。', r'（以上内容仅授权39健康网独家使用，未经版权方授权请勿转载。）',
           r'未经作者允许授权，禁止转载',
           r'[（(](?:实习|责任)?编辑[:：][^）)]*[）)]']
JUNK = {
    'ask.39.net': [r'其他人还在看.*', r'相关问题', r'我要咨询', r'[^\s]{1,14}\*+[^\s]{0,6} \d{4}-\d{2}-\d{2} \d{2}:\d{2}'],   # masked asker + time
    **{d: JUNK_39 for d in ARTICLE if d.endswith('.39.net')},
}
MIN_BODY_CHARS = 50
SUPPLEMENT = re.compile(r'^补充说明[:：]\s*(.*)$', re.S)
EXTRACTOR_TAG = 'zhv2_1'


def extractor_version():
    source = Path(__file__).read_bytes().replace(b'\r\n', b'\n')
    rules = {'base': base.extractor_version(), 'article': ARTICLE, 'qa': QA, 'junk': JUNK, 'min_body_chars': MIN_BODY_CHARS,
             'source_sha256': hashlib.sha256(source).hexdigest()}
    return hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def doc_status(body: str, domain: str) -> str:
    if domain in ARTICLE or domain in QA:
        return 'ok' if len(body) >= MIN_BODY_CHARS else 'thin'
    return base.doc_status(body, domain)


def _title(tree) -> str:
    for h in tree.xpath('//h1'):
        text = norm(h.text_content())
        if text:
            return text
    return meta(tree, 'og:title') or norm(' '.join(tree.xpath('//title/text()')))


def _clean(tree):
    for node in tree.xpath('|'.join('//' + tag for tag in DROP_TAGS)):
        if node.getparent() is not None:
            node.drop_tree()


def _paragraphs(tree, selectors, drops, junk):
    for selector in drops:
        for node in tree.cssselect(selector):
            if node.getparent() is not None:
                node.drop_tree()
    patterns = [re.compile(x) for x in junk]
    for selector in selectors:
        found = tree.cssselect(selector)
        if found:
            return [p for node in found for p in blocks(node) if p and not any(j.fullmatch(p) for j in patterns)]
    return []


def extract_article(raw: str, domain: str) -> Doc | None:
    selectors, drops = ARTICLE[domain]
    tree = parse_html(raw)
    title = _title(tree)
    _clean(tree)
    paras = _paragraphs(tree, selectors, drops, JUNK.get(domain, []))
    if not paras:
        return None
    pset = set(paras)
    heads = [t for t in dict.fromkeys(norm(h.text_content()) for h in tree.xpath('//h2|//h3|//h4')) if t and t in pset]
    return Doc(title=title, description=meta(tree, 'og:description', 'description'), headings=heads, paragraphs=paras,
               extractor='zhv2_' + domain)


def extract_qa(raw: str, domain: str) -> Doc | None:
    rule = QA[domain]
    tree = parse_html(raw)
    title = _title(tree)
    _clean(tree)
    junk = JUNK.get(domain, [])
    q = _paragraphs(tree, rule['question'], rule['question_drop'], junk)
    a = _paragraphs(tree, rule['answer'], rule['answer_drop'], junk)
    # The page repeats the title as the question and again after the supplement label: keep each wording once.
    q = [p for p in q if p != title and not (SUPPLEMENT.match(p) and SUPPLEMENT.match(p).group(1).strip() == title)]
    q = [p for i, p in enumerate(q) if p not in q[:i]]
    if not q and not a:
        return None
    return Doc(title=title, description=meta(tree, 'og:description', 'description'), question='\n\n'.join(q),
               answer='\n\n'.join(a), paragraphs=q + a, extractor='zhv2_qa_' + domain)


def extract(raw: str, domain: str) -> Doc:
    doc = None
    if domain in QA:
        doc = extract_qa(raw, domain)
    elif domain in ARTICLE:
        doc = extract_article(raw, domain)
    return doc if doc is not None else base.extract(raw, domain)
