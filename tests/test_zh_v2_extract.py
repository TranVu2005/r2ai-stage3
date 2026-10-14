from r2ai.zh_v2 import extract as x

ASK = """<html><head><title>你好，乳房发胀_39问医生</title></head><body><h1>你好，乳房发胀</h1>
<div class="ask_cont"><p>你好，乳房发胀</p><p>发病时间：最近一个星期</p><p>补充说明：你好，乳房发胀</p>
<p>a******W 2012-11-30 19:44</p><p>乳房发胀 副乳 彩超</p><section class="sys-box"><div class="sys-label">其他人还在看 治疗人乳头留病毒效果好</div></section>
<p>我要咨询</p></div>
<div class="selected"><div class="sele_all"><p>尹文迪 医师 某某卫生所 擅长：临床全科</p><p class="sele_txt">病情分析：根据描述多见于乳痛症。</p></div>
<div class="sele_all"><p class="sele_txt">建议做乳腺彩超。</p></div></div>
<footer>声明：39问医生</footer></body></html>"""

ART = """<html><body><h1>小儿干咳当心支原体肺炎</h1><div class="art_wrap"><div class="art_content">
<p>在省妇幼儿保科就诊的朵朵今年5岁，近一周的时间一直有点干咳，家人以为是入冬孩子不适应天气变化。</p>
<h2>孩子咳嗽要先去痰</h2><p>秋冬气候变化大，寒冷的空气可引起呼吸道的黏液分泌增加，孩子很容易咳嗽并反复发作。</p>
<p>39健康网(www.39.net)专稿，未经书面授权请勿转载。</p><p>（实习编辑：刘茵莹）</p></div><div class="art_rec">相关推荐 很多链接</div></div></body></html>"""

FD = """<html><body><h1>儿童患风疹接种法</h1><div class="endContent"><div id="viewContent" class="viewContent">
<p>有时全身感到不适，有时颈部的淋巴结还会肿大。发烧2-3天即退。</p></div><div class="hotyy">热门医院 推荐文章</div></div></body></html>"""


def test_ask_39_keeps_question_answer_pairs_without_page_furniture():
    d = x.extract(ASK, 'ask.39.net')
    assert d.title == '你好，乳房发胀'
    assert d.question == '发病时间：最近一个星期\n\n乳房发胀 副乳 彩超'          # title repeats, asker line, button, related box dropped
    assert d.answer == '病情分析：根据描述多见于乳痛症。\n\n建议做乳腺彩超。'     # every selected answer, doctor line dropped
    assert d.paragraphs == [*d.question.split('\n\n'), *d.answer.split('\n\n')]
    assert d.extractor == 'zhv2_qa_ask.39.net' and x.doc_status(d.body, 'ask.39.net') == 'ok'


def test_39_net_article_keeps_only_art_content_and_drops_credits():
    d = x.extract(ART, 'baby.39.net')
    assert d.title == '小儿干咳当心支原体肺炎' and d.question == d.answer == ''
    assert len(d.paragraphs) == 3 and '专稿' not in d.body and '编辑' not in d.body and '相关推荐' not in d.body
    assert d.headings == ['孩子咳嗽要先去痰']


def test_familydoctor_uses_view_content_only():
    d = x.extract(FD, 'familydoctor.com.cn')
    assert d.paragraphs == ['有时全身感到不适，有时颈部的淋巴结还会肿大。发烧2-3天即退。']


def test_missing_container_and_other_hosts_fall_back_to_the_zh_full_extractor():
    from r2ai.zh_full_ingest import extract as base
    html = '<html><body><h1>标题</h1><div class="other"><p>' + '内容' * 200 + '</p></div></body></html>'
    assert x.extract(html, 'baby.39.net').body == base.extract(html, 'baby.39.net').body
    assert x.extract(html, '120ask.com').body == base.extract(html, '120ask.com').body


def test_version_changes_with_rules():
    assert len(x.extractor_version()) == 64


def test_ask39_nested_sele_txt_answer_is_extracted_from_the_real_page():
    from pathlib import Path
    raw = (Path(__file__).parent / 'fixtures' / 'zh_v2_ask39_100084093.html').read_text('utf-8')
    doc = x.extract(raw, 'ask.39.net')
    assert 600 <= len(doc.answer) <= 900 and '黄体酮' in doc.answer and '硝酸咪康唑' in doc.answer
    assert '子宫肥大和子宫糜烂用什么药' in doc.question
    for chrome in ('举报', '找医生', '副主任医师', '擅长'):
        assert chrome not in doc.answer
    assert doc.answer in '\n\n'.join(doc.paragraphs)


def test_unnest_sele_txt_handles_both_markups_and_leaves_other_pages_alone():
    nested = '<div><p class="sele_txt"><p>甲</p><p>乙</p></p><div class="x">后</div></div>'
    flat = '<div><p class="sele_txt">甲乙</p><div class="x">后</div></div>'
    assert x.unnest_sele_txt(nested) == '<div><div class="sele_txt"><p>甲</p><p>乙</p></div><div class="x">后</div></div>'
    assert x.unnest_sele_txt(flat) == '<div><div class="sele_txt">甲乙</div><div class="x">后</div></div>'
    assert x.unnest_sele_txt('<p class="other">x</p>') == '<p class="other">x</p>'
    assert x.unnest_sele_txt('<p class="sele_txt_x">x</p>') == '<p class="sele_txt_x">x</p>'      # not the same class


def test_familydoctor_junk_drops_copyright_and_recommended_reading_lines_but_keeps_content_sentences():
    body = ('<div id="viewContent"><p>肾结石与饮食、饮水量和遗传都有关系，建议多喝水并定期复查，不要自行服用偏方药物治疗。</p>'
            '<p>相关文章指出，该病多见于老年人，需要结合病史综合判断，必要时再做影像学检查。</p>'
            '<p>患者应注意参考相关阅读材料中提到的饮食建议，同时遵医嘱，推荐阅读：饮食指南也有帮助，但不能替代治疗。</p>'
            '<p>推荐阅读：怀孕大补“补”出妊娠糖尿病</p><p>相关文章：</p>'
            '<p>家庭医生在线（www.familydoctor.com.cn）原创内容，未经授权不得转载，违者必究，内容合作请联系：020-37617238</p></div>')
    doc = x.extract('<html><head><title>t</title></head><body><h1>肾结石</h1>' + body + '</body></html>', 'familydoctor.com.cn')
    text = '\n'.join(doc.paragraphs)
    assert '肾结石与饮食' in text and '相关文章指出' in text and '患者应注意参考相关阅读材料' in text
    assert '怀孕大补' not in text and '内容合作请联系' not in text and '相关文章：' not in text
