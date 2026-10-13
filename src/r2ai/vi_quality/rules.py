"""Rule-based quality rules for vi docs (no LLM). A doc is only ever edited at its tail or dropped whole.

doc_text = "\n\n".join([title, body]) as written by r2ai.index.chunk (data/chunks/docs.parquet).

* strip_tail(domain, text): cut a boilerplate block that runs to the end of the doc (comment widget, related-article
  list, CMS footer). Returns (new_text, rule, cut_start) or (text, None, None). The head is kept as is, rstripped.
* drop_reason(domain, text): reason a doc carries no article (lottery result page, error page as title, menu only,
  title is only the site name) or None.
Thresholds come from the scan in reports/s12_vi_quality.md; every rule is exact-pattern, none is a fuzzy score.
"""
from __future__ import annotations

import re

COMMENT_WIDGET = (r'\n*Bình luận:\n\nBạn nghĩ gì về nội dung này\?\n\nBạn cần đăng nhập để gửi bình luận\.\n\n'
                  r'Gửi bình luận.*\Z')
RELATED_TAIL = r'\n\n(?:Nên đọc|Tin liên quan)\n\n.*\Z'
ONECMS_FOOTER = r'(?:\n\n\| TIN LIÊN QUAN \|\n\n\|---\|)?(?:\n\nPOWERED BY)?\n\nONECMS - A PRODUCT OF\n\n×\n\nĐăng nhập\s*\Z'

# (rule name, hosts, compiled pattern); hosts None = every host
STRIP_RULES = (
    ('laodong_comment', {'laodong.vn'}, re.compile(COMMENT_WIDGET, re.S)),
    ('giadinhonline_related', {'giadinhonline.vn', 'phunusuckhoe.giadinhonline.vn'}, re.compile(RELATED_TAIL, re.S)),
    ('baonghean_onecms_footer', {'baonghean.vn'}, re.compile(ONECMS_FOOTER)),
)

LOTTERY_TITLE = re.compile(r'^(?:XS[A-Z]{0,6}\b|KQXS\b|SXMB\b|SXMT\b|Kết quả (?:xổ số|xố số|XS)|Trực tiếp (?:kết quả )?(?:xổ số|XS)'
                           r'|Xổ số|Dự đoán (?:XS|xổ số|kết quả XS))', re.I)
RESULT_DATE = re.compile(r'\d{1,2}/\d{1,2}')   # result pages carry a draw date; news about the lottery does not
ERROR_TITLE = re.compile(r'^(?:https?://\S+/)?(?:404|error)(?:\.html?)?$|^(?:404 )?(?:page )?not found$|^không tìm thấy trang', re.I)
NAV_LINES = frozenset({
    'Trang chủ', 'Video', 'Thời sự', 'Tôi viết', 'Thế giới', 'Chính trị', 'Văn hóa', 'Thể thao', 'Đời sống', 'Kinh doanh',
    'Giới trẻ', 'Giáo dục', 'Công nghệ', 'Game', 'Sức khỏe', 'Xe', 'XEM NHANH', 'ĐÓNG', 'BÌNH LUẬN', 'XEM THÊM', 'Xem thêm',
    'Báo Thanh Niên', 'Chia sẻ', 'Chia sẻ Facebook', 'Chia sẻ Zalo', 'Bình luận', 'Xem nhanh', 'Chuyên mục', 'Gửi bình luận',
    'Quan tâm nhất', 'Mới nhất', 'Xem thêm bình luận', 'Top', 'Đăng nhập để bình luận', 'Bình luận (', ')', 'giây nữa.',
    'Bạn không thể gửi bình luận liên tục. Xin hãy đợi'})
NAV_MIN_LINES = 10
NAV_MIN_SHARE = 0.8
HOST_SITE_NAME = re.compile(r'^(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})$', re.I)


def strip_tail(domain: str, text: str) -> tuple[str, str | None, int | None]:
    for rule, hosts, pat in STRIP_RULES:
        if hosts is not None and domain not in hosts:
            continue
        m = pat.search(text)
        if m and text[:m.start()].strip():
            return text[:m.start()].rstrip(), rule, m.start()
    return text, None, None


def first_line(text: str) -> str:
    return text.split('\n', 1)[0].strip()


def _digit_line(line: str) -> bool:
    return line.isdigit()


def menu_only(text: str) -> bool:
    body = [l.strip() for l in text.split('\n')[1:] if l.strip()]
    if len(body) < NAV_MIN_LINES:
        return False
    nav = sum(l in NAV_LINES or _digit_line(l) for l in body)
    return nav / len(body) >= NAV_MIN_SHARE


def drop_reason(domain: str, text: str) -> str | None:
    title = first_line(text)
    if LOTTERY_TITLE.search(title) and RESULT_DATE.search(title):
        return 'lottery_result'
    if ERROR_TITLE.search(title):
        return 'error_title'
    m = HOST_SITE_NAME.match(title)
    if m and (m.group(1).lower() == domain.lower() or domain.lower().endswith('.' + m.group(1).lower())):
        return 'title_is_site'
    if menu_only(text):
        return 'menu_only'
    return None
