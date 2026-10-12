import pytest


def page(body, title='T', head=''):
    return f'<html><head><title>{title}</title>{head}</head><body>{body}</body></html>'


ARTICLE = '这是一段很长的正文内容，' * 3


def test_content_selector_keeps_only_the_article_container():
    from r2ai.zh_full_ingest.extract import extract
    html = page('<h1>标题</h1><div class="art_wrap"><div class="art_left"><p>栏目关注：</p>'
                f'<div class="art_content"><p>{ARTICLE}</p><p>第二段{ARTICLE}</p></div>'
                '<p>热门问答更多</p></div></div><div class="side"><p>健康资讯推荐</p></div>')
    doc = extract(html, 'care.39.net')
    assert doc.paragraphs == [ARTICLE, '第二段' + ARTICLE]
    assert doc.title == '标题'
    assert doc.extractor == 'zhf_care.39.net'


def test_care_summary_box_is_kept_before_the_article():
    from r2ai.zh_full_ingest.extract import extract
    html = page(f'<div class="art_left"><p>栏目关注：</p><div class="art_summary">核心提示：白领、体力劳作者来看病。</div>'
                f'<div class="art_content"><p>{ARTICLE}</p></div></div>')
    assert extract(html, 'care.39.net').paragraphs == ['核心提示：白领、体力劳作者来看病。', ARTICLE]


def test_missing_container_falls_back_to_the_sample_extractor():
    from r2ai.zh_full_ingest.extract import extract
    from r2ai.zh_sample.extract import extract_zh
    html = page(f'<h1>标题</h1><article><p>{ARTICLE}</p><p>{ARTICLE}再来</p></article>')
    doc = extract(html, 'care.39.net')
    assert doc == extract_zh(html, 'care.39.net')
    assert doc.extractor == 'generic'


def test_domains_without_rules_are_byte_identical_to_the_sample_extractor():
    from r2ai.zh_full_ingest.extract import extract
    from r2ai.zh_sample.extract import extract_zh
    html = page(f'<h1>标题</h1><div class="art_content"><p>{ARTICLE}</p></div><p>{ARTICLE}尾</p>')
    for domain in ('a-hospital.com', 'msdmanuals.cn'):
        assert extract(html, domain) == extract_zh(html, domain)
    qa = page('<div class="b_askcont"><p>问题内容</p></div><div class="b_anscont_cont"><p>回答内容</p></div>')
    assert extract(qa, '120ask.com') == extract_zh(qa, '120ask.com')


def test_drop_selectors_remove_navigation_inside_the_container():
    from r2ai.zh_full_ingest.extract import extract
    html = page('<div id="content"><h1>近亲系数</h1><div id="bodyContent">'
                '<div id="jump-to-nav">跳转到： 导航, 搜索</div><div class="wz-left">中医宝典 &gt; 近亲系数</div>'
                '<p>近亲系数（F）：是指个体中某个基因座位上两个等位基因来自双亲共同祖先的某个基因的概率。</p>'
                '<div id="catlinks" class="catlinks">3个分类: 生物 | 生物学</div></div></div>'
                '<div id="panel"><p>首页</p><p>资助中医宝典</p></div>')
    doc = extract(html, 'zhongyibaodian.net')
    assert doc.paragraphs == ['近亲系数（F）：是指个体中某个基因座位上两个等位基因来自双亲共同祖先的某个基因的概率。']
    assert doc.title == '近亲系数'


def test_selector_order_picks_the_first_matching_container():
    from r2ai.zh_full_ingest.extract import extract
    html = page(f'<div class="list_left"><p>{ARTICLE}列表</p></div>'
                f'<article class="w740"><p>{ARTICLE}文章</p></article>')
    assert extract(html, 'jbk.39.net').paragraphs == [ARTICLE + '文章']
    only_list = page(f'<div class="list_left"><p>{ARTICLE}列表</p></div>')
    assert extract(only_list, 'jbk.39.net').paragraphs == [ARTICLE + '列表']


def test_ads_and_recommendation_blocks_inside_containers_are_dropped():
    from r2ai.zh_full_ingest.extract import extract
    care = page(f'<div class="art_content"><p>{ARTICLE}</p><div class="ppshow"><div class="pptitle">'
                '<a>最受欢迎保健热文</a></div></div></div>')
    assert extract(care, 'care.39.net').paragraphs == [ARTICLE]
    jbk = page(f'<div class="list_left"><div class="disease_box"><ul class="information_ul"><li>临床检查： 视力</li></ul>'
               '<div class="doctor_item"><ul class="doctor_item_top"><li>成都青羊草堂医院</li></ul></div></div></div>')
    assert extract(jbk, 'jbk.39.net').paragraphs == ['临床检查： 视力']
    zhong = page(f'<div id="bodyContent"><p>{ARTICLE}</p><div class="zyfl-ul"><dl><dd><p class="biem">别名：凤尾金星、七星剑</p>'
                 '</dd></dl><ul class="zyfl-book"><li><a>资助【中医宝典】&gt;&gt;&gt;</a></li></ul></div></div>')
    assert extract(zhong, 'zhongyibaodian.net').paragraphs == [ARTICLE, '别名：凤尾金星、七星剑']


def test_template_sections_are_dropped_up_to_the_next_h2():
    from r2ai.zh_full_ingest.extract import extract
    html = page('<div id="bodyContent"><p>【摘录】 《袖珍方》卷二</p>'
                '<h2><span class="mw-headline" id="a">中药方专题</span></h2><p>参看中药方剂主页面。</p>'
                '<h3><span class="mw-headline" id="b">按功效分类的常用方剂</span></h3><p>解表剂、清热剂</p>'
                '<p>A: 安宫牛黄丸 阿魏化痞膏 安神丸 安肾丸 艾汤</p>'
                '<h2><span class="mw-headline" id="c">注意事项</span></h2><p>孕妇慎用。</p>'
                '<h2><span class="mw-headline" id="d">更多医学电子书</span></h2><p>《中药学》</p></div>')
    assert extract(html, 'zhongyibaodian.net').paragraphs == ['【摘录】 《袖珍方》卷二', '注意事项', '孕妇慎用。']


def test_jbk_disease_page_keeps_only_its_own_facts():
    from r2ai.zh_full_ingest.extract import extract
    html = page('<div class="list_left"><div class="disease_box"><p class="disease_title">基本信息</p>'
                '<div class="information_box"><p class="information_l">IgA肾病是一种肾小球疾病。</p></div>'
                '<ul class="information_ul"><li>别名： IgA肾炎</li></ul></div>'
                '<div class="disease_box"><p class="disease_title">其他类似疾病</p><div class="disease_type_slick"><div>急性肾盂肾炎</div></div></div>'
                '<div class="disease_box"><div class="authority_art"><a>糖尿病肾病能吃黑豆吗</a></div><ul class="articles_type"><li>病因|耳聋</li></ul></div>'
                '<div class="disease_box"><ul class="drug_ul"><li>肾炎康复片</li></ul></div>'
                '<div class="disease_box"><ul class="experience_ul"><li>用药经验</li></ul><div class="experience_box">匿名：：头晕</div></div>'
                '<div class="disease_box medical-information-box"><a class="medical-information-item">尿素</a></div></div>')
    assert extract(html, 'jbk.39.net').paragraphs == ['IgA肾病是一种肾小球疾病。', '别名： IgA肾炎']
    labels = page('<article class="w740"><p>目录:</p><p>咽痛是常见症状。</p><p>...[详细][收起]</p><p>在线购药：</p>'
                  '<p>以上疾病都不是您要找的? 试试症状自查工具</p><p>目录:咽痛</p></article>')
    assert extract(labels, 'jbk.39.net').paragraphs == ['咽痛是常见症状。', '目录:咽痛']


def test_fixed_paragraph_patterns_drop_only_whole_matching_paragraphs():
    from r2ai.zh_full_ingest.extract import extract
    wujue = page(f'<div class="art-text"><p>{ARTICLE}</p><p><b><a>[1]</a> <a>[2]</a> <a>下一页</a></b></p>'
                 f'<p>{ARTICLE}见下一页</p></div>')
    assert extract(wujue, 'wujue.com').paragraphs == [ARTICLE, ARTICLE + '见下一页']
    care = page(f'<div class="art_content"><p>{ARTICLE}</p><p><strong>39健康网(www.39.net)专稿，未经书面授权请勿转载。</strong></p>'
                '<center><strong>（以上内容仅授权39健康网独家使用，未经版权方授权请勿转载。）</strong></center>'
                '<p>未经作者允许授权，禁止转载</p>'
                f'<p>{ARTICLE}，请勿转载他人文章</p></div>')
    assert extract(care, 'care.39.net').paragraphs == [ARTICLE, ARTICLE + '，请勿转载他人文章']


def test_people_title_falls_back_to_html_title_without_site_suffix():
    from r2ai.zh_full_ingest.extract import extract
    body = f'<h1></h1><h1> </h1><div class="artDet"><p>{ARTICLE}</p></div>'
    doc = extract(page(body, title='识别毒蘑菇，两大数据库显身手--健康·生活--人民网'), 'health.people.com.cn')
    assert doc.title == '识别毒蘑菇，两大数据库显身手'
    with_h1 = extract(page(f'<h1>真标题</h1><p>{ARTICLE}</p>', title='别的--健康·生活--人民网'), 'health.people.com.cn')
    assert with_h1.title == '真标题'
    second_h1 = extract(page(f'<h1></h1><h1>全球埃博拉疫苗研发加速冲刺</h1><div class="artDet"><p>{ARTICLE}</p></div>',
                             title='全球埃博拉疫苗研发加速冲刺--健康·生活--人民网'), 'health.people.com.cn')
    assert second_h1.title == '全球埃博拉疫苗研发加速冲刺'
    other_suffix = extract(page(f'<p>{ARTICLE}</p>', title='标题--人民网'), 'health.people.com.cn')
    assert other_suffix.title == '标题--人民网'


@pytest.mark.parametrize('domain,chars,status', [
    ('120ask.com', 49, 'thin'), ('120ask.com', 50, 'ok'),
    ('zhongyibaodian.net', 50, 'ok'), ('wujue.com', 50, 'ok'), ('care.39.net', 50, 'ok'),
    ('jbk.39.net', 50, 'ok'), ('youlai.cn', 50, 'ok'), ('zydcd.com', 50, 'ok'),
    ('cnkang.com', 199, 'thin'), ('cnkang.com', 200, 'ok'), ('a-hospital.com', 199, 'thin'),
])
def test_status_threshold_is_domain_specific(domain, chars, status):
    from r2ai.zh_full_ingest.extract import doc_status
    assert doc_status('字' * chars, domain) == status


def test_version_is_deterministic_and_differs_from_the_sample():
    from r2ai.zh_full_ingest import extract as zf
    from r2ai.zh_sample.extract import extractor_version as sample_version
    assert zf.extractor_version() == zf.extractor_version()
    assert len(zf.extractor_version()) == 64 and zf.extractor_version() != sample_version()
    before = zf.extractor_version()
    zf.MIN_BODY_CHARS['120ask.com'] = 51
    try:
        assert zf.extractor_version() != before
    finally:
        zf.MIN_BODY_CHARS['120ask.com'] = 50


def test_version_changes_with_extractor_source(tmp_path, monkeypatch):
    from r2ai.zh_full_ingest import extract as zf
    before = zf.extractor_version()
    edited = tmp_path / 'extract.py'
    edited.write_bytes(open(zf.__file__, 'rb').read() + b'\n# logic edit\n')
    monkeypatch.setattr(zf, '__file__', str(edited))
    assert zf.extractor_version() != before


def test_version_ignores_line_endings_of_the_checkout(tmp_path, monkeypatch):
    from r2ai.zh_full_ingest import extract as zf
    before = zf.extractor_version()
    crlf = tmp_path / 'extract.py'
    crlf.write_bytes(open(zf.__file__, 'rb').read().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    monkeypatch.setattr(zf, '__file__', str(crlf))
    assert zf.extractor_version() == before
