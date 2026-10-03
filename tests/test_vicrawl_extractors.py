"""Site extractors on real saved pages: tests/fixtures (fresh, complete) and out/sample_raw (probe-era)."""
from r2ai.paths import FIXTURES_DIR, LEGACY_OUT_DIR, ROOT

import re
from pathlib import Path

import pytest

from vicrawl.extractors import extract_doc
from vicrawl.extractors.base import blocks, norm, parse_html

FIX = FIXTURES_DIR
SAMPLE = LEGACY_OUT_DIR / 'sample_raw'


def load(name, folder=FIX):
    return (folder / name).read_text(encoding='utf-8', errors='replace')


# -- walker ------------------------------------------------------------------------------------------
def test_blocks_keep_paragraph_boundaries_and_collapse_whitespace_only():
    tree = parse_html('<div><p>Một   dòng với\n khoảng trắng</p><ul><li>mục 1</li><li>mục <b>2</b></li></ul>text trần<br><br>sau br kép<br>cùng đoạn</div>')
    assert blocks(tree.cssselect('div')[0]) == ['Một dòng với khoảng trắng', 'mục 1', 'mục 2', 'text trần', 'sau br kép cùng đoạn']


def test_blocks_table_rows_become_one_paragraph_each():
    tree = parse_html('<div><table><tr><th>Chỉ số</th><th>Giá trị</th></tr><tr><td>WBC</td><td>5.2</td></tr></table></div>')
    assert blocks(tree.cssselect('div')[0]) == ['Chỉ số | Giá trị', 'WBC | 5.2']


def test_norm_only_touches_whitespace():
    assert norm('  Tiếng Việt​  có dấu  ') == 'Tiếng Việt có dấu'


# -- vinmec: article + Q&A ------------------------------------------------------------------------------
def test_vinmec_article():
    d = extract_doc(load('vinmec.com__5f28d781.html'), 'vinmec')
    assert d.extractor == 'vinmec' and d.title == 'Các triệu chứng mọc răng khôn điển hình'
    assert d.description.startswith('Mọc răng khôn được coi là nỗi sợ')
    assert '1. Quá trình mọc răng khôn diễn ra như thế nào?' in d.headings
    assert d.paragraphs[0].startswith('Mọc răng khôn')
    assert not any(p.startswith(('Để đặt lịch khám', 'Xem thêm')) for p in d.paragraphs)       # booking form + related
    assert not any('Mục lục' in p for p in d.paragraphs)
    assert d.body == '\n\n'.join(d.paragraphs) and not d.question


def test_vinmec_qa_split():
    d = extract_doc(load('vinmec.com__44ae4a45.html'), 'vinmec')
    assert d.question.startswith('Chào bác sĩ,') and 'gãy đầu dưới xương quay' in d.question
    assert 'Khách hàng ẩn danh' not in d.question and 'Trả lời' not in d.question
    assert d.answer.startswith('Chào bạn,') and 'bác sĩ xin giải đáp như sau' in d.answer
    assert 'Được giải đáp bởi' not in d.answer and 'HOTLINE' not in d.answer
    assert d.question.count('\n\n') >= 1                                                   # paragraph boundaries kept


# -- medlatec: article + FAQ ------------------------------------------------------------------------------
def test_medlatec_article_drops_related_review_and_comments():
    d = extract_doc(load('medlatec.vn__f722bfdd.html'), 'medlatec')
    assert d.title.startswith('Dấu hiệu COVID biến chủng mới')
    assert d.description.startswith('Trước diễn biến dịch bệnh')
    assert d.paragraphs[0] == '1. Những điều cần biết về biến thể Omicron XEC'
    body = d.body
    for junk in ('Bác sĩ duyệt bài', 'Từ khoá', 'Bình luận', 'Omicron XEC - Biến thể lây lan nhanh', 'Mọi chi tiết về dịch vụ'):
        assert junk not in body, junk
    assert d.paragraphs[-1].startswith('Dễ thấy rằng')


def test_medlatec_faq_question_and_answer():
    d = extract_doc(load('medlatec.vn__5127c5cc.html'), 'medlatec')
    assert d.title == 'Bị thoát vị địa đệm không phẫu thuật được không'
    assert d.question.startswith('Bị thoát vị địa đệm không phẫu thuật được không\n\nTôi bị thoát vị đĩa đệm')
    assert d.answer.startswith('Chào bạn!') and d.answer.rstrip().endswith('Chúc bạn và gia đình nhiều sức khỏe!')
    assert 'Mọi chi tiết về dịch vụ' not in d.answer


# -- suckhoedoisong / thanhnien -------------------------------------------------------------------------
def test_suckhoedoisong_drops_captions_related_and_video_blocks():
    d = extract_doc(load('suckhoedoisong.vn__022cbe03.html'), 'suckhoedoisong')
    assert d.title.startswith('7 món ăn bài thuốc')
    assert d.description.startswith('SKĐS - Thiên môn đông')
    assert d.paragraphs[0] == '1. Đặc điểm của cây thiên môn đông'
    assert d.paragraphs[-1].startswith('Kiêng kỵ:')
    assert not any('vị thuốc làm đẹp da mặt' in p or 'Biến thể COVID' in p for p in d.paragraphs)
    assert 'Cây thiên môn đông vừa là cây cảnh' not in d.body            # <figcaption>


def test_thanhnien_ask_the_doctor_column_is_qa():
    d = extract_doc(load('thanhnien.vn__0ba286dd.html'), 'thanhnien')
    assert d.question.startswith('Em là nữ, 19 tuổi') and not d.question.startswith('*')
    assert d.answer.startswith('- Trả lời:') and 'Phẫu thuật là phương pháp' in d.answer


def test_thanhnien_plain_article_has_no_qa():
    d = extract_doc(load('thanhnien.vn__1c4a97aa.html'), 'thanhnien')
    assert not d.question and d.title == 'Cơn lốc lạm dụng thuốc kê toa ở Mỹ'
    assert 'Thêm Thanh Niên trên Google' not in d.body


# -- hellobacsi / laodong / tiemchung -----------------------------------------------------------------
def test_hellobacsi_ask_doctor_page_is_qa_and_related_links_dropped():
    d = extract_doc(load('hellobacsi.com__e58f76ce.html'), 'hellobacsi')
    assert d.question.startswith('Chào bác sĩ') and d.answer.startswith('Chào bạn')
    assert 'Nội dung của Hello Bacsi' not in d.body
    assert 'Sữa cho bà bầu: Loại nào mới thực sự tốt cho thai nhi?' not in d.body


def test_laodong_body_without_title_byline_or_captions():
    d = extract_doc(load('laodong.vn__80d31607.html'), 'laodong')
    assert d.title == 'Tập thiền định hoặc hít thở sâu để giảm mỡ nội tạng'
    assert d.paragraphs[0].startswith('Thiền định là một kỹ thuật')
    assert d.title not in d.paragraphs


def test_tiemchung_longchau_toc_removed():
    d = extract_doc(load('tiemchunglongchau.com.vn__a73a4267.html'), 'tiemchunglongchau')
    assert d.paragraphs[0].startswith('Nhiều bạn trẻ hiện nay lo lắng')
    assert 'Lão hóa da sớm là gì?' in d.headings


# -- probe-era samples in out/sample_raw (complete pages only) --------------------------------------------------
@pytest.mark.parametrize('sample,extractor,title_part,first', [
    ('354877.html', 'suckhoecongdongonline', 'Cô gái vĩ cầm ở Na Uy', 'Cô bé vĩ cầm ngày nào'),
    ('294727.html', 'giadinhonline_phunu', 'Da đen như than', 'Như các bạn đã biết, chanh chứa'),
    ('102457.html', 'giadinhonline', '7 món ăn hàng đầu', 'Dưới đây là những thực phẩm hàng đầu'),
])
def test_sample_raw_pages(sample, extractor, title_part, first):
    d = extract_doc(load(sample, SAMPLE), extractor)
    assert d.extractor == extractor and title_part in d.title
    assert d.paragraphs[0].startswith(first)
    assert sum(map(len, d.paragraphs)) > 800


# -- fallback -------------------------------------------------------------------------------------------
def test_unknown_domain_uses_trafilatura_generic():
    html = '<html><head><title>T</title></head><body><article><h1>Tiêu đề</h1>' + ''.join(f'<p>Đoạn văn số {i} nói về điều trị bệnh nhân rất chi tiết.</p>' for i in range(12)) + '</article></body></html>'
    d = extract_doc(html, 'generic')
    assert d.extractor == 'generic' and len(d.paragraphs) >= 10 and d.title == 'Tiêu đề'


def test_site_spec_falls_back_when_container_missing():
    html = '<html><body><article>' + ''.join(f'<p>Đoạn {i} có nội dung đầy đủ và dài đủ để trafilatura nhận ra.</p>' for i in range(15)) + '</article></body></html>'
    assert extract_doc(html, 'vinmec').extractor == 'generic'


# -- sites whose generic (trafilatura) output carried sidebar / footer boilerplate -----------------------------
def test_suckhoeviet_drops_similar_other_and_related_table():
    d = extract_doc(load('suckhoeviet.org.vn__87523904.html'), 'suckhoeviet')
    assert d.extractor == 'suckhoeviet'
    body = d.body
    assert 'Cùng chuyên mục' not in body and 'Các tin khác' not in body and 'Hoạt động hội' not in body
    assert not d.paragraphs[0].startswith('Nước ép cần tây')                    # related-article table above the text
    assert not any(p.startswith('http') for p in d.paragraphs)
    assert d.paragraphs[0].startswith('Nhiều người vẫn hay truyền tai')


def test_baoangiang_content_and_no_sidebar():
    d = extract_doc(load('baoangiang.com.vn__67d824ac.html'), 'baoangiang')
    assert d.extractor == 'baoangiang' and d.title == 'Lý do không nên ngoáy tai thường xuyên'
    assert 'Cách đây' not in d.body and 'Podcast' not in d.body
    assert not any(p.startswith('Theo Vietnamnet') for p in d.paragraphs)
    assert d.paragraphs[0].startswith('Tôi hay có thói quen ngoáy tai')


@pytest.mark.parametrize('name', ['baodanang.vn__170025f4', 'baohaiphong.vn__3a903b0c', 'baonghean.vn__8b53899e'])
def test_cms_entry_template_no_footer_no_duplicate_title(name):
    d = extract_doc(load(name + '.html'), 'cms_entry')
    assert d.extractor == 'cms_entry'
    assert not any(x in d.body for x in ('POWERED BY', 'NEKO', 'Cỡ chữ', 'Mặc định', 'TIN LIÊN QUAN', '|---|', 'Bài liên quan'))
    assert d.title not in d.paragraphs                                           # headline was repeated inside the entry
    assert not any(re.search(r'•\s*\d{2}/\d{2}/\d{4}', p) for p in d.paragraphs)  # byline "AUTHOR • date"


def test_baodaklak_title_is_article_not_site_name():
    d = extract_doc(load('phuyen.baodaklak.vn__d5b69ecb.html'), 'baodaklak')
    assert d.extractor == 'baodaklak' and d.title == 'Thiếu máu'
    d2 = extract_doc(load('phuyen.baodaklak.vn__0f24e558.html'), 'baodaklak')
    assert not any(p.startswith(('Theo Vietnam+', 'Ảnh có tính minh họa')) for p in d2.paragraphs)


# -- site-level junk: bare own-domain lines, ALL-CAPS section labels at the edges of the body --------------
from vicrawl.extractors.base import strip_site_junk  # noqa: E402

BODY = ['Đoạn mở đầu bài viết về sức khỏe tim mạch.', 'Đoạn thứ hai có nội dung.', 'Đoạn thứ ba.', 'Đoạn thứ tư.',
        'Đoạn thứ năm.', 'Đoạn thứ sáu.', 'Đoạn thứ bảy.', 'Đoạn thứ tám.']


@pytest.mark.parametrize('line', ['suckhoeviet.org.vn', 'Suckhoeviet.org.vn', 'www.suckhoeviet.org.vn', 'SUCKHOEVIET.ORG.VN'])
def test_bare_own_domain_line_dropped_anywhere(line):
    paras = BODY[:4] + [line] + BODY[4:]
    assert strip_site_junk(paras, 'suckhoeviet.org.vn') == BODY


def test_bare_domain_rule_uses_the_docs_own_domain_only():
    paras = BODY + ['nhandan.vn', 'Theo suckhoeviet.org.vn', 'suckhoeviet.org.vn/bai-viet']
    assert strip_site_junk(paras, 'suckhoeviet.org.vn') == paras          # other site / sentence / URL path: kept
    assert strip_site_junk(BODY + ['baodaklak.vn'], 'phuyen.baodaklak.vn') == BODY + ['baodaklak.vn']
    assert strip_site_junk(BODY + ['phuyen.baodaklak.vn'], 'phuyen.baodaklak.vn') == BODY
    assert strip_site_junk(BODY + ['suckhoeviet.org.vn'], None) == BODY + ['suckhoeviet.org.vn']


@pytest.mark.parametrize('pos', [0, 1, 2, -1, -2, -3])
def test_caps_label_dropped_in_first_or_last_three_lines(pos):
    paras = list(BODY)
    paras.insert(pos if pos >= 0 else len(paras) + 1 + pos, 'QUỐC HỘI')
    assert strip_site_junk(paras, 'phuyen.baodaklak.vn') == BODY


def test_caps_label_rule_negative_cases():
    mid = BODY[:4] + ['QUỐC HỘI'] + BODY[4:]                             # sub-heading in the middle: kept
    assert strip_site_junk(mid, 'x.vn') == mid
    for line in ['QUỐC HỘI KHÓA XV KỲ HỌP', 'CẢNH BÁO!', 'LƯU Ý:', 'Quốc hội', 'Hà Nội', '2024', 'COVID-19 vẫn đang lây.']:
        paras = [line] + BODY                                              # >4 words / end punctuation / not all caps / no letters
        assert strip_site_junk(paras, 'x.vn') == paras, line


def test_site_junk_applied_through_extract_doc_with_domain():
    html = ('<html><body><h1>Tiêu đề</h1><div class="entry"><p>QUỐC HỘI</p>' + ''.join(f'<p>{p}</p>' for p in BODY)
            + '<p>x.vn</p></div></body></html>')
    from vicrawl.extractors.base import Spec
    from vicrawl.extractors.sites import SPECS
    SPECS['_t'] = Spec('_t', content=['div.entry'])
    try:
        assert extract_doc(html, '_t', domain='x.vn').paragraphs == BODY
        assert extract_doc(html, '_t').paragraphs == ['QUỐC HỘI'] + BODY + ['x.vn']
    finally:
        del SPECS['_t']


def test_premature_html_close_tag_does_not_empty_the_page():
    """baotayninh.vn emits `<html ...> </html> <head>…` (a stray </html> right after the start tag)."""
    d = extract_doc(load('baotayninh.vn__d70cdb05.html'), 'generic', 'baotayninh.vn')
    assert sum(map(len, d.paragraphs)) >= 1000
    assert any(p.startswith('Với biến chủng Omicron') for p in d.paragraphs)


# -- QA-review fixes (2026-10-03) ---------------------------------------------------------------------------
def _doc(name, extractor, domain):
    return extract_doc(load(name), extractor, domain)


def test_vinmec_product_ad_block_is_cut():
    d = _doc('vinmec.com__aac8c983.html', 'vinmec', 'vinmec.com')
    assert not any(re.search(r'laminkid|elepharma|dangkytuvandinhduong', p, re.I) for p in d.paragraphs)
    assert d.paragraphs[-1].startswith('Tóm lại')


def test_vinmec_qa_labels_and_anonymous_marker_out_of_body_but_split_kept():
    d = _doc('vinmec.com__dbc9ee03.html', 'vinmec', 'vinmec.com')
    assert not {'Hỏi', 'Trả lời', 'Khách hàng ẩn danh'} & set(d.paragraphs)
    assert 'Khách hàng ẩn danh' not in d.question
    assert d.question.startswith('Chào bác sĩ') and 'bướu đa nhân hai thùy' in d.question
    assert d.answer.startswith('Được giải đáp bởi') or d.answer.startswith('Chào bạn')
    d2 = _doc('vinmec.com__69445c9e.html', 'vinmec', 'vinmec.com')
    assert d2.question and d2.answer and 'Hỏi' not in d2.paragraphs


def test_vinmec_paragraph_equal_to_title_dropped():
    d = _doc('vinmec.com__5f28d781.html', 'vinmec', 'vinmec.com')
    assert d.title.casefold() not in {p.casefold() for p in d.paragraphs}
    assert '1. Quá trình mọc răng khôn diễn ra như thế nào?' in d.headings      # numbered headings stay


def test_hellobacsi_article_type_badge_removed():
    d = _doc('hellobacsi.com__646308d3.html', 'hellobacsi', 'hellobacsi.com')
    assert not any(p.startswith('Đây là bài viết') for p in d.paragraphs)
    assert 'Hướng dẫn' not in d.paragraphs[:4]
    assert d.paragraphs[0].startswith('Bò kho')


def test_hellobacsi_byline_and_group_disclaimer_removed():
    d = _doc('hellobacsi.com__55cf54ef.html', 'hellobacsi', 'hellobacsi.com')
    assert not any(p.upper().endswith('HELLO BACSI') for p in d.paragraphs)
    d = _doc('hellobacsi.com__4d77c9b1.html', 'hellobacsi', 'hellobacsi.com')
    assert not any('không đưa ra các lời khuyên' in p for p in d.paragraphs)
    assert d.paragraphs[-1].startswith('Nếu bạn có bất kỳ câu hỏi nào')     # ordinary advice sentence kept


def test_hellobacsi_answer_has_no_see_also_tail_and_qa_still_split():
    d = _doc('hellobacsi.com__24f212cc.html', 'hellobacsi', 'hellobacsi.com')
    assert d.question.startswith('Chào bác sĩ') and d.answer.startswith('Chào bạn Bích Hảo')
    d = _doc('hellobacsi.com__e58f76ce.html', 'hellobacsi', 'hellobacsi.com')
    assert d.question and not re.search(r'bạn có thể xem thêm', d.answer, re.I)


def test_suckhoedoisong_video_prompts_and_comment_box_removed():
    d = _doc('suckhoedoisong.vn__1f3fb477.html', 'suckhoedoisong', 'suckhoedoisong.vn')
    assert not any(re.match(r'^mời .*xem', p, re.I) for p in d.paragraphs)
    assert d.paragraphs[-1].startswith('Lưu ý, bài viết chỉ mang tính tham khảo')


def test_suckhoedoisong_video_page_uses_sapo_not_generic():
    d = _doc('suckhoedoisong.vn__c60b6a5e.html', 'suckhoedoisong', 'suckhoedoisong.vn')
    assert d.extractor == 'suckhoedoisong'
    assert d.paragraphs == [d.description] and d.description.startswith('SKĐS - Nam thanh niên đi bơi')
    assert 'Ý kiến của bạn' not in d.paragraphs


def test_baocantho_title_from_article_h1_no_table_debris():
    d = _doc('baocantho.com.vn__9623c267.html', 'baocantho', 'baocantho.com.vn')
    assert d.extractor == 'baocantho' and d.title == 'Chống mù lòa bằng thuốc nhỏ mắt thay vì tiêm thuốc'
    assert not any(re.fullmatch(r'[|\s]+', p) or re.search(r'ảnh\s*:', p, re.I) for p in d.paragraphs)
    assert d.paragraphs[0].startswith('Việc điều trị các bệnh có thể gây mù lòa')
    d = _doc('baocantho.com.vn__2ff7b1db.html', 'baocantho', 'baocantho.com.vn')
    assert d.title == 'Tốp 10 rau củ, trái cây giàu chất chống ôxy hóa' and len(d.paragraphs) == len(set(d.paragraphs))


def test_khoahocphothong_no_title_para_related_list_keywords_or_o3a_debris():
    d = _doc('khoahocphothong.vn__460868e2.html', 'khoahocphothong', 'khoahocphothong.vn')
    assert d.extractor == 'khoahocphothong' and d.title == 'Thành lập Trung tâm cấp cứu 115'
    assert d.paragraphs[0] != d.title
    assert not any(p.startswith('Hơn 20 năm theo đuổi vi phẫu') for p in d.paragraphs)   # related-news sapo list
    d = _doc('khoahocphothong.vn__2a48bb55.html', 'khoahocphothong', 'khoahocphothong.vn')
    assert not any('o3a_p' in p for p in d.paragraphs) and 'o3a_p' not in d.description
    d = _doc('khoahocphothong.vn__10ff4899.html', 'khoahocphothong', 'khoahocphothong.vn')
    assert not any(p.startswith('Khoa Học Phổ Thông, khoa học') for p in d.paragraphs)
    assert d.paragraphs[0].startswith('KHPT-Lòng bàn chân')


def test_khoahocphothong_sapo_only_page_keeps_spec_not_generic_related_list():
    d = _doc('khoahocphothong.vn__0459f939.html', 'khoahocphothong', 'khoahocphothong.vn')
    assert d.extractor == 'khoahocphothong' and d.paragraphs == [d.description]
    assert d.description.startswith('Bệnh nhân Trần Thị Bé Năm')


def test_hellobacsi_qa_markers_out_of_body():
    d = _doc('hellobacsi.com__24f212cc.html', 'hellobacsi', 'hellobacsi.com')
    assert d.question and not {'Bạn đọc hỏi', 'Bác sĩ trả lời', 'Bác sĩ trả lời:'} & set(d.paragraphs)


@pytest.mark.parametrize('line', ['Mời xem tiếp số 58', 'Mời bạn xem thêm video', 'Mời quý vị theo dõi:',
                                  'Mời bạn xem thêm video GS.TS Nguyễn Gia Bình chia sẻ về cấp cứu ngoại viện:'])
def test_suckhoedoisong_video_prompt_variants(line):
    from vicrawl.extractors.sites import SPECS
    assert any(re.search(j, line, re.I) for j in SPECS['suckhoedoisong'].junk)


def test_suckhoedoisong_long_sentence_starting_with_moi_is_kept():
    from vicrawl.extractors.sites import SPECS
    line = ('Mời người dân đến các điểm tiêm chủng để được tư vấn, xem xét tình trạng sức khỏe trước khi tiêm vaccine '
            'theo đúng hướng dẫn của Bộ Y tế và cơ sở y tế địa phương.')
    assert not any(re.search(j, line, re.I) for j in SPECS['suckhoedoisong'].junk)


def test_hellobacsi_screening_call_to_action_dropped_but_tai_day_in_text_kept():
    from vicrawl.extractors.sites import SPECS
    junk = [re.compile(j, re.I) for j in SPECS['hellobacsi'].junk]
    cta = 'Bạn đang có dấu hiệu rối loạn cương dương. Làm tầm soát rối loạn cương dương miễn phí ngay TẠI ĐÂY!'
    text = 'Đầu ngón tay sưng phồng, chuyển đỏ và có thể hơi nóng tại đây'
    assert any(j.search(cta) for j in junk) and not any(j.search(text) for j in junk)


# -- QA-review fixes (2026-10-03, round 2) ------------------------------------------------------------------
def test_hellobacsi_triple_chevron_see_also_paragraphs_dropped():
    d = _doc('hellobacsi.com__5f6c8d5c.html', 'hellobacsi', 'hellobacsi.com')
    assert d.paragraphs and len(d.body) > 1000
    assert not [p for p in d.paragraphs if p.lstrip().startswith('>>>')]


def test_vinmec_bare_url_source_line_dropped():
    d = _doc('vinmec.com__c4e39311.html', 'vinmec', 'vinmec.com')
    assert d.paragraphs and len(d.body) > 1000
    assert not [p for p in d.paragraphs if re.match(r'^\(?\s*nguồn\s*:\s*https?://', p, re.I)]
