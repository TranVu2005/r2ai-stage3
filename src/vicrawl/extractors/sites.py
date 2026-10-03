"""Per-site specs. Selectors verified against saved pages in tests/fixtures (see tests/test_vicrawl_extractors.py)."""
from __future__ import annotations

from r2ai.paths import ROOT

import re

from .base import Spec, blocks, first_text, norm

# --- shared boilerplate -----------------------------------------------------------------------------
BOOKING = r'^(?:để đặt lịch|quý khách (?:vui lòng|có nhu cầu)|đặt lịch (?:khám|xét nghiệm)|liên hệ (?:hotline|ngay)|hotline\b)'


def split_markers(paragraphs, q_marks, a_marks, ignore=()):
    """Q&A pages that mark the sections with short label paragraphs ("Hỏi" ... "Trả lời")."""
    def label(p):
        return p.strip(' :.').lower()
    qi = next((i for i, p in enumerate(paragraphs) if len(p) <= 30 and label(p) in q_marks), None)
    if qi is None:
        return None
    ai = next((i for i, p in enumerate(paragraphs) if i > qi and len(p) <= 30 and label(p) in a_marks), None)
    if ai is None:
        return None
    ign = [re.compile(x, re.I) for x in ignore]
    q = [p for p in paragraphs[qi + 1:ai] if not any(x.search(p) for x in ign)]
    a = [p for p in paragraphs[ai + 1:] if not any(x.search(p) for x in ign)]
    return '\n\n'.join(q), '\n\n'.join(a)


def vinmec_qa(tree, paragraphs):
    return split_markers(paragraphs, {'hỏi', 'câu hỏi'}, {'trả lời', 'bác sĩ trả lời'},
                         ignore=(r'^khách hàng', r'^được giải đáp bởi', r'^người hỏi', r'^trân trọng'))


def medlatec_qa(tree, paragraphs):
    items = tree.cssselect('div.faq-item')
    if not items:
        return None
    item = items[0]
    title = norm(next(iter(item.cssselect(':scope > .faq-item-title')), item).text_content()) if item.cssselect(':scope > .faq-item-title') else ''
    qdesc = item.cssselect(':scope > .faq-item-desc')
    ans = item.cssselect(':scope > .faq-item-answer > .faq-item-desc')
    if not ans:
        return None
    q = ([title] if title else []) + (blocks(qdesc[0]) if qdesc else [])
    a = [p for p in blocks(ans[0]) if not re.search(BOOKING + r'|^mọi chi tiết về dịch vụ', p, re.I)]
    return '\n\n'.join(q), '\n\n'.join(a)


def thanhnien_qa(tree, paragraphs):
    """"Hỏi bác sĩ" columns: the reader's question is the sapo, the body starts with "- Trả lời:"."""
    if not paragraphs or not re.match(r'^[-–]?\s*trả lời\s*:', paragraphs[0], re.I):
        return None
    sapo = first_text(tree, ['h2.detail-sapo', 'div.detail-sapo'])
    return (re.sub(r'^\*\s*', '', sapo), '\n\n'.join(paragraphs)) if sapo else None


def hellobacsi_qa(tree, paragraphs):
    return split_markers(paragraphs, {'bạn đọc hỏi'}, {'bác sĩ trả lời'}, ignore=(r'^trân trọng',))


SPECS: dict[str, Spec] = {}


def reg(spec: Spec):
    SPECS[spec.name] = spec


reg(Spec('vinmec', content=['#main-article'], title=['h1'], ld_description=True,
         remove=['#toc-container', '.table-of-contents', '.modal', '.box-booking'],
         junk=[BOOKING, r'^mục lục$', r'^xem thêm\b', r'^bài viết được (?:tư vấn|biên soạn|viết)', r'^trân trọng!?$', r'^khách hàng ẩn danh$',
               r'^\(?\s*nguồn\s*:\s*https?://\S+\)?$'],      # bare-URL source line "(Nguồn: https://…"
         # sponsored supplement block appended to nutrition articles ("Thực Phẩm bảo vệ sức khỏe LAMINKID I:" ... Elepharma)
         cut_after=[r'^thực phẩm bảo vệ sức khỏe (?-i:[A-ZĐ0-9]{3,})', r'^đăng ký tư vấn dinh dưỡng cho bé'],
         body_drop=[r'^(?:hỏi|câu hỏi|trả lời|bác sĩ trả lời)$'], drop_title_para=True, qa=vinmec_qa))

reg(Spec('medlatec', content=['div.faq-item', 'div.block-posts-single div.description'], title=['div.faq-item-title', 'h1.page-title', 'h1'],
         sapo=['div.shortdescription'],
         remove=['.faq-item-meta', '.faq-item-actions', '.news-verification-box', '.single-new', '.keywords', '.post-review', '.share',
                 '.faq-item-answer .faq-item-meta'],
         junk=[BOOKING, r'^mọi chi tiết về dịch vụ', r'^hãy gọi đến', r'^bác sĩ duyệt bài', r'^từ khoá', r'^bình luận'], qa=medlatec_qa))

# video pages ("Sức khỏe TV", "Bản tin sức khỏe") carry only the sapo + an embedded player: the sapo becomes the body
reg(Spec('suckhoedoisong', content=['div.detail-content'], title=['h1.detail-title', 'h1'], sapo=['div.detail-sapo', 'h2.detail-sapo'],
         remove=['div[type="RelatedOneNews"]', 'div[type="VideoStream"]', '.box-news-relate', '.kbwscwl-relatedbox', '.detail-author', '.detail-tags', '.detail-social'],
         junk=[r'^skđs\s*[-–]', r'^(?:>>|->)\s', r'^theo dõi\b', r'^mời\b(?=.{0,110}$).{0,60}\b(?:xem|theo dõi)\b', r'^ý kiến của bạn$'],
         sapo_fallback=True))

reg(Spec('thanhnien', content=['div.detail-content', 'div.detail-cmain'], title=['h1.detail-title', 'h1'], sapo=['h2.detail-sapo', 'div.detail-sapo'],
         remove=['.tn-ggr2-popup', '.detail-tags', '.box-related', '.detail-author'],
         junk=[r'^nhấp vào nút', r'^tại trang vừa mở'], qa=thanhnien_qa))

reg(Spec('laodong', content=['div.art-body', 'article.article-detail'], title=['h1.title', 'h1'],
         remove=['h1', '.related-news', '.box-tag', '.art-author', '.relate-news', '.news-relate'],
         junk=[r'^(?:>>|->)\s', r'^\S.* \d{2}/\d{2}/\d{4} \d{2}:\d{2} \(GMT']))

reg(Spec('suckhoecongdongonline', content=['div.article__body', 'div.article'], title=['h1.article__title', 'h1'], sapo=['div.article__sapo'],
         remove=['.article__tag', '.article__related', '.comment', '.article__source'],
         junk=[r'^đăng nhập hoặc']))

reg(Spec('giadinhonline_phunu', content=['#article-body'], title=['h1.post-title', 'h1'], sapo=['div.post-description'],
         remove=['#article_Inview', '.box-related', '.tag-list', 'adrender', '.relate-news', '.box-news'],
         junk=[r'^(?:->|>>)\s', r'^xem thêm']))

reg(Spec('giadinhonline', content=['#explus-editor', 'div.content_detail_news'], title=['h1'], sapo=['h2.sapo', 'div.sapo'],
         remove=['.box-related', '.tag-list', '.news-relate'],
         junk=[r'^(?:->|>>)\s', r'^xem thêm', r'^ảnh minh họa$']))

reg(Spec('tiemchunglongchau', content=['#content-article', 'article'], title=['h1'],
         remove=['.pointer-events-none', '[data-toc]', 'ul.toc'],
         junk=[r'^mục lục$', r'^xem thêm', r'^đặt lịch', r'^liên hệ', r'^để được tư vấn', r'^bài viết (?:có tham khảo|được biên soạn)']))

reg(Spec('hellobacsi', content=['div.unique-content-wrapper', 'div.article-content-item', 'article'], title=['h1'],
         # [role=tooltip]: article-type badge ("Hướng dẫn" + "Đây là bài viết hướng dẫn ...")
         remove=['.related-articles', '.author-info', '.references', '[class*=advert]', '[class*=banner]', '[role="tooltip"]'],
         junk=[r'^hello (?:bác sĩ|bacsi|health group) không (?:cung cấp|đưa ra)', r'^.{0,40}\bhello bacsi$', r'^[–-]?\s*bạn có thể xem thêm', r'^>>>', r'^nội dung của hello bacsi', r'^góc nhìn$', r'^đây là bài viết có thể hiện', r'^trân trọng!?$',
               r'^miễn trừ trách nhiệm',
               r'(?-i:\bTẠI ĐÂY\b)'],                     # "… miễn phí ngay TẠI ĐÂY!" screening call-to-action
         body_drop=[r'^(?:bạn đọc hỏi|bác sĩ trả lời)\s*:?$'], qa=hellobacsi_qa))

reg(Spec('vietnamnet', content=['#maincontent', 'div.content-detail'], title=['h1.content-detail-title', 'h1'], sapo=['h2.content-detail-sapo', 'div.content-detail-sapo'],
         remove=['article.ck-cms-wiki-news-full', '.article-relate', '.inner-article', '.box-tag', '.comment'],
         junk=[r'^(?:>>|->)\s']))

reg(Spec('suckhoeviet', content=['div.__MASTERCMS_CONTENT', 'div.article-detail-body-right'], title=['h1'],
         remove=['.article-detail-similar', '.article-detail-other', '.article-detail-tags', 'table.__MB_ARTICLE_A'],
         junk=[r'^(?:>>|->)\s', r'^https?://\S+$']))

reg(Spec('baoangiang', content=['div.content-fck-font-size[data]', 'div.content-fck'], title=['h1'],
         remove=['p.author.ignore', '.news-relate', '.box-relate'],
         junk=[r'^theo\s+vietnamnet']))

# baodanang.vn / baohaiphong.vn / baonghean.vn share one CMS template (the entry repeats title, category and byline)
reg(Spec('cms_entry', content=['div.c-news-detail .entry', 'div.c-news-detail article.entry', 'article.entry'], title=['h1'],
         sapo=['p.sc-longform-header-sapo'],
         remove=['.c-box', '.c-sticky-share', '.c-sticky-mobile', '.c-powered', '.c-news-related', '.c-tag',
                 '.block-sc-cate-name', '.block-sc-title', '.sc-longform-header-meta'],
         junk=[r'^(?:>>|->)\s', r'^cỡ chữ$', r'^mặc định$']))

reg(Spec('baodaklak', content=['div.td-post-content', 'div.content-detail'], title=['div.detail-main-title h1.title', 'h1.title'],
         remove=['.td-post-sharing', '.related-post', '.box-tag'],
         junk=[r'^theo\s+vietnam\+', r'^ảnh có tính minh họa']))

reg(Spec('baocantho', content=['div.content-fck'], title=['h1.fw700', 'h1.margintopbot10'],    # the first <h1> is the breadcrumb
         remove=['.detail-related', '.sharemxh', '.detail-fb-like-share'],
         junk=[r'^(?:>>|->)\s']))

# same CMS template as cms_entry; old pages carry Word debris escaped as "<_o3a_p>" and a keyword line as sapo
reg(Spec('khoahocphothong', content=['div.c-news-detail .entry', 'article.entry'], title=['h1'],
         sapo=['p.sc-longform-header-sapo'],
         remove=['.c-box', '.c-sticky-share', '.c-sticky-mobile', '.c-powered', '.c-news-related', '.c-tag', '.box-category',
                 '.block-sc-cate-name', '.block-sc-title', '.sc-longform-header-meta'],
         junk=[r'^(?:>>|->)\s', r'^cỡ chữ$', r'^mặc định$', r'^khoa học phổ thông, khoa học, tạp chí'],
         scrub=[r'</?_o3a_p>'], drop_title_para=True, sapo_fallback=True))
