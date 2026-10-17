"""zh-full extractor: zh_sample extraction plus reviewed per-domain selectors."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from vicrawl.extractors.base import Doc, blocks, meta, norm, parse_html
from r2ai.zh_sample.extract import DROP_TAGS, extract_zh, extractor_version as sample_version

CONTENT = {
    'care.39.net': (['div.art_summary, div.art_content'], ['div.ppshow']),
    'jbk.39.net': (['article.w740', 'div.list_left'], ['div.doctor_item', 'p.disease_title', 'div.disease_type_slick',
                                                       'div.authority_art', 'ul.articles_type', 'ul.drug_ul',
                                                       'ul.experience_ul', 'div.experience_box',
                                                       'div.medical-information-box']),
    'wujue.com': (['div.art-text'], []),
    'youlai.cn': (['div[class*="textContent--"]'], []),
    'zydcd.com': (['div.info_txt'], []),
    'zhongyibaodian.net': (['#bodyContent'], ['#jump-to-nav', '.wz-left', '.wz-right', '#belownav', '#catlinks',
                                              '.printfooter', '#toc', '#siteSub', '#contentSub', '.mw-editsection']),
}
DROP_PARAGRAPHS = {
    'wujue.com': [r'(?:\[\d+\]\s*)+下一页'],
    'zhongyibaodian.net': [r'资助【中医宝典】>>>'],
    'jbk.39.net': [r'目录:', r'\.\.\.\[详细\]\[收起\]', r'在线购药：', r'以上疾病都不是您要找的\? 试试症状自查工具'],
    'care.39.net': [r'39健康网\(www\.39\.net\)专稿，未经书面授权请勿转载。',
                    r'（以上内容仅授权39健康网独家使用，未经版权方授权请勿转载。）', r'未经作者允许授权，禁止转载'],
}
DROP_SECTIONS = {
    'zhongyibaodian.net': [r'.*专题', r'更多医学电子书', r'全部\d+本中医学电子书按拼音首字母分页索引', r'人体穴位图',
                           r'制药企业分区域查询'],
}
TITLE_SUFFIX = {'health.people.com.cn': '--健康·生活--人民网'}
DEFAULT_MIN_BODY_CHARS = 200
MIN_BODY_CHARS = {'120ask.com': 50, **{domain: 50 for domain in CONTENT}}


def extractor_version():
    source = Path(__file__).read_bytes().replace(b'\r\n', b'\n')
    rules = {'sample': sample_version(), 'content': CONTENT, 'drop_paragraphs': DROP_PARAGRAPHS,
             'drop_sections': DROP_SECTIONS, 'title_suffix': TITLE_SUFFIX, 'min_body_chars': MIN_BODY_CHARS,
             'default_min_body_chars': DEFAULT_MIN_BODY_CHARS,
             'source_sha256': hashlib.sha256(source).hexdigest()}
    return hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def doc_status(body: str, domain: str) -> str:
    return 'ok' if len(body) >= MIN_BODY_CHARS.get(domain, DEFAULT_MIN_BODY_CHARS) else 'thin'


def _h1(tree) -> str:
    for h in tree.xpath('//h1'):
        text = norm(h.text_content())
        if text:
            return text
    return ''


def _html_title(tree) -> str:
    return norm(' '.join(tree.xpath('//title/text()')))


def _drop_sections(tree, patterns):
    sections = [re.compile(x) for x in patterns]
    for h2 in tree.xpath('//h2'):
        if h2.getparent() is None or not any(x.fullmatch(norm(h2.text_content())) for x in sections):
            continue
        doomed, node = [h2], h2.getnext()
        while node is not None and node.tag != 'h2':
            doomed.append(node)
            node = node.getnext()
        for node in doomed:
            node.drop_tree()


def extract(raw: str, domain: str) -> Doc:
    if domain in TITLE_SUFFIX:
        doc = extract_zh(raw, domain)
        if not doc.title:
            tree = parse_html(raw)
            title, suffix = _html_title(tree), TITLE_SUFFIX[domain]
            doc.title = _h1(tree) or (title[:-len(suffix)] if title.endswith(suffix) else title)
        return doc
    if domain not in CONTENT:
        return extract_zh(raw, domain)
    selectors, drops = CONTENT[domain]
    tree = parse_html(raw)
    title = _h1(tree) or meta(tree, 'og:title') or _html_title(tree)
    nodes = tree.xpath('|'.join('//' + tag for tag in DROP_TAGS)) + [n for selector in drops for n in tree.cssselect(selector)]
    for node in nodes:
        if node.getparent() is not None:
            node.drop_tree()
    _drop_sections(tree, DROP_SECTIONS.get(domain, []))
    paras = []
    for selector in selectors:
        found = tree.cssselect(selector)
        if found:
            junk = [re.compile(x) for x in DROP_PARAGRAPHS.get(domain, [])]
            paras = [p for node in found for p in blocks(node) if p and not any(j.fullmatch(p) for j in junk)]
            break
    if not paras:
        return extract_zh(raw, domain)
    pset = set(paras)
    heads = [t for t in dict.fromkeys(norm(h.text_content()) for h in tree.xpath('//h2|//h3|//h4')) if t and t in pset]
    return Doc(title=title, description=meta(tree, 'og:description', 'description'), headings=heads,
               paragraphs=paras, extractor='zhf_' + domain)
