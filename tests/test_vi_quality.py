from r2ai.vi_quality.rules import drop_reason, menu_only, strip_tail

WIDGET = 'Bình luận:\n\nBạn nghĩ gì về nội dung này?\n\nBạn cần đăng nhập để gửi bình luận.\n\nGửi bình luận'


def test_laodong_comment_widget_and_adblock_tail():
    head = 'Tiêu đề\n\nTiêu đề\n\nThứ tư, 28/01/2026 13:00 (GMT+7)\n\nNội dung bài.'
    assert strip_tail('laodong.vn', head + '\n\n' + WIDGET) == (head, 'laodong_comment', len(head))
    ad = '\n\n' + WIDGET + ' Có vẻ như bạn đang dùng một trình chặn quảng cáo.\n\n4 Tải lại trang'
    assert strip_tail('laodong.vn', head + ad)[0] == head
    assert strip_tail('laodong.vn', head + '\n\nCùng chuyên mục\n\nBài khác\n\n' + WIDGET)[2] is not None
    assert strip_tail('thanhnien.vn', head + '\n\n' + WIDGET)[1] is None


def test_related_tail_only_on_named_hosts_and_keeps_head():
    text = 'Tiêu đề\n\nThân bài một.\n\nThân bài hai.\n\nNên đọc\n\nBài A\n\nTóm tắt A...\n\nBài B\n\nTóm tắt B...'
    new, rule, cut = strip_tail('giadinhonline.vn', text)
    assert new == 'Tiêu đề\n\nThân bài một.\n\nThân bài hai.' and rule == 'giadinhonline_related'
    assert strip_tail('phunusuckhoe.giadinhonline.vn', text.replace('Nên đọc', 'Tin liên quan'))[1] == 'giadinhonline_related'
    assert strip_tail('vinmec.com', text)[1] is None


def test_strip_never_empties_a_doc():
    assert strip_tail('giadinhonline.vn', '\n\nNên đọc\n\nBài A') == ('\n\nNên đọc\n\nBài A', None, None)


def test_baonghean_footer():
    text = 'T\n\nNội dung.\n\nTheo VNE\n\n| TIN LIÊN QUAN |\n\n|---|\n\nPOWERED BY\n\nONECMS - A PRODUCT OF\n\n×\n\nĐăng nhập'
    assert strip_tail('baonghean.vn', text)[0] == 'T\n\nNội dung.\n\nTheo VNE'


def test_lottery_needs_result_date():
    assert drop_reason('suckhoecongdongonline.vn', 'XSDN 26/1 - Kết quả xổ số Đồng Nai\n\nbody') == 'lottery_result'
    assert drop_reason('suckhoecongdongonline.vn', 'Kết quả xổ số Miền Bắc ngày 7/12/2020 - XSMB\n\nb') == 'lottery_result'
    assert drop_reason('suckhoecongdongonline.vn', 'Xổ số Vietlott: Giải Jackpot vượt mốc 271 tỷ đồng\n\nb') is None
    assert drop_reason('suckhoecongdongonline.vn', 'Chân dung người trúng xổ số 92 tỉ\n\nb') is None


def test_error_title_and_site_name_title():
    assert drop_reason('baonghean.vn', 'https://baonghean.vn/404\n\nTiếng Việt') == 'error_title'
    assert drop_reason('x.vn', 'Đề thi và đáp án môn Tiếng Anh mã đề 404\n\nb') is None
    assert drop_reason('vinmec.com', 'vinmec.com\n\nb') == 'title_is_site'
    assert drop_reason('vinmec.com', 'www.vinmec.com\n\nb') == 'title_is_site'
    assert drop_reason('vinmec.com', 'Đau ngực\n\nb') is None


def test_menu_only_needs_most_lines_to_be_nav():
    nav = '\n\n'.join(['Trang chủ', 'Video', 'Thời sự', 'Tôi viết', 'Thế giới', 'Chính trị', 'Văn hóa', 'Thể thao', 'Đời sống',
                       'Kinh doanh', 'Xem thêm', '0'])
    assert menu_only('Tiêu đề\n\n' + nav)
    assert not menu_only('Tiêu đề\n\n' + nav + '\n\n' + '\n\n'.join(f'Đoạn văn số {i} của bài viết thật.' for i in range(20)))
    assert not menu_only('Tiêu đề\n\nTrang chủ\n\nVideo')
